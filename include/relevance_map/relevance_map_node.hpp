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

    typedef std::map<uint32_t,double> map_t;

    relevance_map_node(){}

    void initialize(const ros::NodeHandlePtr nh);
    void init_classifiers(const std::string& folder_name);
    void release();

    bool retrieve_input_cloud(ip::PointCloudT::Ptr cloud);

    void publish_feedback();

    const ip::PointCloudT::Ptr get_background(){return _background;}
    ip::SurfaceOfInterest& get_soi(){return _soi;}
    const std::string& get_modality(){return _modality;}
    const std::string& get_method(){return _method;}
    double get_threshold(){return _threshold;}

    std::map<std::string,iagmm::NNMap> _nnmap_class; /**< nnmap classifiers */
    std::map<std::string,iagmm::GMM> _gmm_class; /**< gmm classifiers */
    iagmm::MCS _mcs;

protected:
    rgbd::RGBD_Subscriber::Ptr _images_sub; /**<RGB image, Depth image and camera info subscriber*/
    ip::SurfaceOfInterest _soi;/**<Attribute to compute the saliency map*/

    std::string _method; /**< which classifier will be use to compute the saliency map : expert, nnmap, gmm, random or sift */
    std::string _mode; /**< exploration or exploitation mode */
    std::string _load_exp; /**< path to archive of a previous experiment*/
    std::string _modality;
    int _dimension;
    double _threshold = 0.5;
    std::map<std::string,int> _modalities;
    std::vector<std::string> _mcs_mod_mapping;

    ip::PointCloudT::Ptr _background;/**< pointcloud of the background. only for export mode */
    bool _background_saved; /**< if the bachground is already saved */



    map_t _choice_map;
    std::unique_ptr<ip::workspace_t> _workspace; /**< workspace of the robot */

    std::map<std::string,std::unique_ptr<ros::Publisher>> _weighted_cloud_pub;
    std::unique_ptr<ros::Publisher> _choice_dist_cloud_pub;

    /**
     * @brief Compute the saliency map for the next iteration and choose the next supervoxel to explore.
     * @param input cloud
     * @return true if all went good false otherwise
     */
    bool _compute_relevance_map(const ip::PointCloudT::Ptr input_cloud);
    bool _compute_choice_map(pcl::Supervoxel<image_processing::PointT> &sv, uint32_t &lbl);
};

}

#endif //RELEVANCE_MAP_NODE_HPP
