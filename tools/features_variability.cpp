#include <iostream>
#include <unistd.h>
#include <relevance_map/relevance_map_node.hpp>
#include <rgbd_utils/rgbd_subscriber.hpp>
#include <rgbd_utils/rgbd_to_pointcloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <eigen3/Eigen/Core>
#include <pcl/octree/octree_search.h>
#include <relevance_map/utilities.hpp>

namespace rm = relevance_map;
namespace ip = image_processing;
namespace ru = rgbd_utils;


typedef struct data_stats_t{
    Eigen::VectorXd mean;
    Eigen::MatrixXd covariance;
    int nb_samples;
} data_stats_t;

class features_variability : public rm::relevance_map_node
{

public:
    features_variability() : rm::relevance_map_node(){
        XmlRpc::XmlRpcValue params;
        XmlRpc::XmlRpcValue wks_params;
        _nh.getParam("params",params);
        _nh.getParam("workspace",wks_params);
        rm::init_workspace(wks_params,_workspace);

        _images_sub.reset(new rgbd_utils::RGBD_Subscriber(
                            params["rgb_info"],
                            params["rgb_topic"],
                            params["depth_info"],
                            params["depth_topic"],_nh));

        _modality = static_cast<std::string>(params["modality"]);
        _nbr_iteration = (int) params["number_of_iteration"];

        _octree.reset(new pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(128.0f));

        _var_pub.reset(new ros::Publisher(
                           _nh.advertise<sensor_msgs::PointCloud2>("variance_cloud",5)));
    }


    void update(){

        std::cout << "iteration " << _iteration << std::endl;

        ip::PointCloudT::Ptr input_cloud(new ip::PointCloudT);
        if(!retrieve_input_cloud(input_cloud))
            return;

        _compute_supervoxels(input_cloud);
        _soi.compute_feature(_modality);

        ip::SupervoxelArray supervoxels = _soi.getSupervoxels();

        if(_iteration == 0){
            _base_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
            for(const auto& sv : supervoxels){
                pcl::PointXYZ coord;
                coord.x = sv.second->centroid_.x;
                coord.y = sv.second->centroid_.y;
                coord.z = sv.second->centroid_.z;
                _base_cloud->push_back(coord);
                data_stats_t stat;
                stat.mean = _soi.get_feature(sv.first,_modality);
                stat.covariance = Eigen::MatrixXd::Zero(stat.mean.rows(),stat.mean.rows());
                stat.nb_samples = 1;
                _statistics.push_back(stat);
            }
            _octree->setInputCloud(_base_cloud);
            _octree->addPointsFromInputCloud();
            _iteration++;
            return;
        }


        std::cout << "computing mean and variance" << std::endl;
        double N;
        Eigen::VectorXd X;
        Eigen::VectorXd mean;
        Eigen::MatrixXd covariance;
        for(const auto& sv : supervoxels){
            pcl::PointXYZ coord;
            coord.x = sv.second->centroid_.x;
            coord.y = sv.second->centroid_.y;
            coord.z = sv.second->centroid_.z;
            std::vector<int> index;
            std::vector<float> dist;
            _octree->nearestKSearch(coord,1,index,dist);

            X = _soi.get_feature(sv.first,_modality);
            mean = _statistics[index[0]].mean;
            covariance = _statistics[index[0]].covariance;

            _statistics[index[0]].nb_samples++;
            N = _statistics[index[0]].nb_samples;
            _statistics[index[0]].mean = 1.f/N*X + (N-1.f)/N*mean;

            mean = _statistics[index[0]].mean;


            _statistics[index[0]].covariance =  (N - 2.f)/(N - 1.f)*covariance +
                    N/((N-1.f)*(N-1.f))*(X-mean)*(X-mean).transpose();


        }




        _iteration++;
    }

    bool is_finish(){return _iteration > _nbr_iteration;}


    void build_variance_cloud(){
        std::cout << "building variance cloud" << std::endl;
        _variance_cloud.clear();
        for(size_t i = 0; i < _statistics.size(); i++){
            pcl::PointXYZI pt;
            pt.x = _base_cloud->at(i).x;
            pt.y = _base_cloud->at(i).y;
            pt.z = _base_cloud->at(i).z;
            Eigen::VectorXd variance(_statistics[i].covariance.rows());
            for(size_t k = 0; k < variance.rows(); k++)
                variance(k) = _statistics[i].covariance(k,k);

            pt.intensity = variance.squaredNorm();
            _variance_cloud.push_back(pt);
        }
    }

    void publish_results(){
        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(_variance_cloud,cloud_msg);

        cloud_msg.header = _images_sub->get_depth().header;
        _var_pub->publish(cloud_msg);
    }

private:
    ros::NodeHandle _nh;
    std::vector<data_stats_t> _statistics;
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::Ptr _octree;
    pcl::PointCloud<pcl::PointXYZ>::Ptr _base_cloud;
    pcl::PointCloud<pcl::PointXYZI> _variance_cloud;
    std::shared_ptr<ros::Publisher> _var_pub;
    int _iteration = 0;
    int _nbr_iteration;
};

int main(int argc, char** argv){

    ros::init(argc,argv,"features_variability");

    features_variability fv;

    while(ros::ok() && !fv.is_finish()){
        fv.update();
        fv.build_variance_cloud();
        fv.publish_results();
//        usleep(10000);
        ros::spinOnce();
    }

    fv.build_variance_cloud();

    while(ros::ok()){
        fv.publish_results();
        usleep(10000);
        ros::spinOnce();
    }

    return 0;
}
