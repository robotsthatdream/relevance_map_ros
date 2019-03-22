#include <iostream>
#include <relevance_map/relevance_map_node.hpp>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <boost/archive/text_iarchive.hpp>

namespace ip = image_processing;
namespace rm = relevance_map;

class pcd_to_relevance_map : public rm::relevance_map_node{
public:
    pcd_to_relevance_map(){}
    pcd_to_relevance_map(const std::string& pcd_file, const std::string& gmm_archive){
        _input_cloud.reset(new ip::PointCloudT);
        pcl::io::loadPCDFile(pcd_file,*_input_cloud);

        std::ifstream ifs(gmm_archive);
        if(!ifs){
            ROS_ERROR_STREAM("unable to open archive file");
            return;
        }
        _method = "gmm";
        _modality = "meanFPFHLabHist";
        boost::archive::text_iarchive iarch(ifs);
        iarch >> _gmm_class["meanFPFHLabHist"];

        _rm_pub = _nh.advertise<sensor_msgs::PointCloud2>("relevance_map",5);
        _cloud_pub = _nh.advertise<sensor_msgs::PointCloud2>("input_cloud",5);
    }

    bool compute_rm(){
        if(!_compute_supervoxels(_input_cloud,false)){
            ROS_ERROR_STREAM("unable to generate supervoxels");
            return false;
        }
        if(!_compute_relevance_map()){
            ROS_ERROR_STREAM("unable to compute relevance map");
            return false;
        }
        pcl::PointCloud<pcl::PointXYZI> rm_cloud = _soi.getColoredWeightedCloud("meanFPFHLabHist",1);
        pcl::toROSMsg(rm_cloud,_rm_msg);
        _rm_msg.header.frame_id = "world";
        pcl::toROSMsg(*_input_cloud,_cloud_msg);
        _cloud_msg.header.frame_id = "world";

        return true;
    }

    void publish(){
        _rm_pub.publish(_rm_msg);
        _cloud_pub.publish(_cloud_msg);
    }

private:
    ip::PointCloudT::Ptr _input_cloud;
    sensor_msgs::PointCloud2 _rm_msg;
    sensor_msgs::PointCloud2 _cloud_msg;
    ros::NodeHandle _nh;
    ros::Publisher _rm_pub;
    ros::Publisher _cloud_pub;
    cmm::CollabMM _gmm;
};

int main(int argc, char** argv){

    ros::init(argc,argv,"pcd_to_relevance_map");

    if(argc != 3){
        ROS_WARN_STREAM("Usage :\n \targ1 : path to a pcd_file\n \targ2 : archive of a classifier");
        return 1;
    }
    pcd_to_relevance_map prm(argv[1],argv[2]);
    while(ros::ok() && !prm.compute_rm()) ros::spinOnce();
    while(ros::ok()){
        prm.publish();
        ros::spinOnce();
    }
    return 0;
}
