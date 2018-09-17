#include <iostream>
#include <relevance_map/relevance_map_node.hpp>
#include <rgbd_utils/rgbd_subscriber.hpp>
#include <rgbd_utils/rgbd_to_pointcloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <eigen3/Eigen/Core>
#include <pcl/octree/octree_search.h>

namespace rm = relevance_map;
namespace ip = image_processing;
namespace ru = rgbd_utils;


typedef struct data_stats_t{
    Eigen::VectorXd mean;
    Eigen::MatrixXd covariance;
    int nb_samples;
} data_stats_t;

class features_variablity : public rm::relevance_map_node
{

public:
    features_variablity(){


        XmlRpc::XmlRpcValue params;
        _nh.getParam("params",params);

        _images_sub.reset(new rgbd_utils::RGBD_Subscriber(
                            static_cast<std::string>(params["rgb_info"]),
                            static_cast<std::string>(params["rgb_topic"]),
                            static_cast<std::string>(params["depth_info"]),
                            static_cast<std::string>(params["depth_topic"]),_nh));

        _modality = static_cast<std::string>(params["modality"]);

        _octree.reset(new pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(128.0f));
    }


    void update(){
        ip::PointCloudT::Ptr input_cloud;
        if(!retrieve_input_cloud(input_cloud))
            return;

        ip::SurfaceOfInterest soi;
        soi.compute_feature(_modality);

        ip::SupervoxelArray supervoxels = soi.getSupervoxels();

        if(_iteration == 0){
            _base_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
            for(const auto& sv : supervoxels){
                pcl::PointXYZ coord;
                coord.x = sv.second->centroid_.x;
                coord.y = sv.second->centroid_.y;
                coord.z = sv.second->centroid_.z;
                _base_cloud->push_back(coord);
                data_stats_t stat;
                stat.mean = soi.get_feature(sv.first,_modality);
                stat.covariance = Eigen::MatrixXd::Zero(stat.mean.rows(),stat.mean.cols());
                stat.nb_samples = 1;
                _statistics.push_back(stat);
            }
            _octree->setInputCloud(_base_cloud);
            return;
        }

        int N;
        Eigen::VectorXd X;
        Eigen::VectorXd mean;
        for(const auto& sv : supervoxels){
            pcl::PointXYZ coord;
            coord.x = sv.second->centroid_.x;
            coord.y = sv.second->centroid_.y;
            coord.z = sv.second->centroid_.z;
            std::vector<int> index;
            std::vector<float> dist;
            _octree->nearestKSearch(coord,1,index,dist);


            std::cout << "actual centroid " << coord
                      << " centroid found in map " << _base_cloud->at(index[0]) << std::endl;

            X = soi.get_feature(sv.first,_modality);
            _statistics[index[0]].nb_samples++;
            N = _statistics[index[0]].nb_samples;
            _statistics[index[0]].mean = 1/N*X +
                    (N-1)/N*_statistics[index[0]].mean;
            mean = _statistics[index[0]].mean;
            _statistics[index[0]].covariance =  (N - 2)/(N - 1)*_statistics[index[0]].covariance +
                    N/((N-1)*(N-1))*(X-mean)*(X-mean).transpose();
        }


    }

private:
    ros::NodeHandle _nh;
    std::vector<data_stats_t> _statistics;
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::Ptr _octree;
    pcl::PointCloud<pcl::PointXYZ>::Ptr _base_cloud;
//    std::map<Eigen::VectorXd,data_stats_t> _statistics;
    int _iteration = 0;
};

int main(int argc, char** argv){
    return 0;
}
