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

/**
 * @brief Base class to implent a ros node for computing a relevance map.
 */
class relevance_map_node
{
public:

    typedef std::map<uint32_t,double> map_t;

    /**
     * @brief default constructor
     */
    relevance_map_node(){}
    relevance_map_node(const relevance_map_node& rm) :
        _soi(rm._soi), _images_sub(rm._images_sub),
        _method(rm._method), _mode(rm._mode), _load_exp(rm._load_exp),
        _modality(rm._modality), _dimension(rm._dimension), _threshold(rm._threshold),
        _nbr_class(rm._nbr_class), _modalities(rm._modalities),
        _mcs_mod_mapping(rm._mcs_mod_mapping),
        _background(rm._background), _background_saved(rm._background_saved),
        _true_labels(rm._true_labels), _choice_map(rm._choice_map),
        _workspace(rm._workspace)
    {

    }

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
    int get_nbr_class(){return _nbr_class;}

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
    int _nbr_class = 2;
    std::map<std::string,int> _modalities;
    std::vector<std::string> _mcs_mod_mapping;

    ip::PointCloudT::Ptr _background;/**< pointcloud of the background. only for export mode */
    bool _background_saved; /**< if the bachground is already saved */

    std::map<uint32_t,int> _true_labels;

    map_t _choice_map;
    std::shared_ptr<ip::workspace_t> _workspace; /**< workspace of the robot */

    std::map<std::string,std::vector<std::shared_ptr<ros::Publisher>>> _weighted_cloud_pub;
    std::unique_ptr<ros::Publisher> _choice_dist_cloud_pub;
    std::unique_ptr<ros::Publisher> _input_cloud_pub;

    template <typename param>
    void _clear_supervoxels(){
              _soi.clear<param>();
    }

    bool _compute_supervoxels(const ip::PointCloudT::Ptr input_cloud, bool with_workspace = true);
    /**
     * @brief Compute the saliency map for the next iteration and choose the next supervoxel to explore.
     * @param input cloud
     * @return true if all went good false otherwise
     */
    bool _compute_relevance_map();
    bool _compute_choice_map(pcl::Supervoxel<image_processing::PointT> &sv, uint32_t &lbl);
};

}

#endif //RELEVANCE_MAP_NODE_HPP
