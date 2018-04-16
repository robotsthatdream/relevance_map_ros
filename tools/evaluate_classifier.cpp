#include <iostream>
#include <relevance_map/relevance_map_node.hpp>
#include <relevance_map/score_computation.hpp>

#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <boost/archive/text_iarchive.hpp>

namespace ip = image_processing;
namespace rm = relevance_map;

class evaluate_classifier : public rm::relevance_map_node{
public:
    evaluate_classifier(const std::string& input_cloud, const std::string& background, const std::string& gmm_archive){
        _input_cloud.reset(new ip::PointCloudT);
        pcl::io::loadPCDFile(input_cloud,*_input_cloud);

        _background.reset(new ip::PointCloudT);
        pcl::io::loadPCDFile(background,*_background);

        std::ifstream ifs(gmm_archive);
        if(!ifs){
            ROS_ERROR_STREAM("unable to open archive file");
            return;
        }
        _method = "gmm";
        _modality = "meanFPFHLabHist";
        boost::archive::text_iarchive iarch(ifs);
        iarch >> _gmm_class["meanFPFHLabHist"];
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
        rm::score_computation<evaluate_classifier> sc(this,"out_score");
        sc.compute_scores_results(0);

        return true;
    }



private:
    ip::PointCloudT::Ptr _input_cloud;
    ros::NodeHandle _nh;
    iagmm::GMM _gmm;
};

int main(int argc, char** argv){

    ros::init(argc,argv,"evaluate_classifier");

    if(argc != 4){
        ROS_WARN_STREAM("Usage :\n \targ1 : path to a point cloud pcd file");
        ROS_WARN_STREAM("\targ2 : background cloud pcd file");
        ROS_WARN_STREAM("\targ3 : archive of a classifier");
        return 1;
    }
    evaluate_classifier ec(argv[1],argv[2],argv[3]);
    while(ros::ok() && !ec.compute_rm()) ros::spinOnce();
    return 0;
}
