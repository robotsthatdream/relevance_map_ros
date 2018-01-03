#ifndef RELEVANCE_MAP_NODE_HPP
#define RELEVANCE_MAP_NODE_HPP

#include <map>
#include <vector>

#include <ros/ros.h>

#include <iagmm/gmm.hpp>
#include <iagmm/nnmap.hpp>
#include <iagmm/mcs.hpp>

#include <image_processing/SurfaceOfInterest.h>
#include <image_processing/pcl_types.h>

#include <rgbd_utils/rgbd_subscriber.hpp>
#include <rgbd_utils/rgbd_to_pointcloud.h>

namespace ip = image_processing;
namespace rgbd = rgbd_utils;

namespace relevance_map{

class relevance_map_node
{
public:
    relevance_map_node();

    void ros_init(const ros::NodeHandlePtr nh);


    void publish_feedback();

private:
    rgbd::RGBD_Subscriber::Ptr _images_sub; /**<RGB image, Depth image and camera info subscriber*/
    ip::SurfaceOfInterest _soi;/**<Attribute to compute the saliency map*/

    std::string _soi_method; /**< which classifier will be use to compute the saliency map : expert, nnmap, gmm, random or sift */
    std::string _mode; /**< exploration or exploitation mode */
    std::string _load_exp; /**< path to archive of a previous experiment*/
    std::string _modality;
    int _dimension;
    std::map<std::string,int> _modalities;
    std::vector<std::string> _mcs_mod_mapping;

    ip::PointCloudT::Ptr _background;/**< pointcloud of the background. only for export mode */
    bool _background_saved; /**< if the bachground is already saved */

    std::map<std::string,iagmm::NNMap> _nnmap_class; /**< nnmap classifiers */
    std::map<std::string,iagmm::GMM> _gmm_class; /**< gmm classifiers */
    iagmm::MCS _mcs;

    Eigen::VectorXd _choice_dist_map;
    std::unique_ptr<ip::workspace_t> _workspace; /**< workspace of the robot */

    std::map<std::string,std::unique_ptr<Publisher>> _weighted_cloud_pub;
    std::unique_ptr<Publisher> _choice_dist_cloud_pub;

    bool _compute_relevance_map(const ip::PointCloudT::Ptr input_cloud);
    bool _compute_choice_map(const pcl::SuperVoxel<ip::PointT> &sv, uint32_t &lbl);
};

}

#endif //RELEVANCE_MAP_NODE_HPP
