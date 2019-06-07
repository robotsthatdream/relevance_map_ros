#ifndef RELEVANCE_MAP_NODE_HPP
#define RELEVANCE_MAP_NODE_HPP

#include <map>
#include <vector>

#include <ros/ros.h>

#include <cmm/gmm.hpp>
#include <cmm/nnmap.hpp>
#include <cmm/mcs.hpp>

#include <image_processing/SurfaceOfInterest.h>
#include <image_processing/pcl_types.h>

#include <rgbd_utils/rgbd_subscriber.hpp>
#include <rgbd_utils/rgbd_to_pointcloud.h>

#include <relevance_map/cnn_features.h>

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
    relevance_map_node(){
        srand(time(NULL));
        _gen.seed(rand());
    }
    relevance_map_node(const relevance_map_node& rm) :
        _soi(rm._soi), _images_sub(rm._images_sub),
        _method(rm._method), _mode(rm._mode), _load_exp(rm._load_exp), _load_comp(rm._load_comp),
        _modality(rm._modality), _dimension(rm._dimension), _threshold(rm._threshold),
        _nbr_class(rm._nbr_class), _modalities(rm._modalities),
        _mcs_mod_mapping(rm._mcs_mod_mapping),
        _background(rm._background), _background_saved(rm._background_saved),
        _true_labels(rm._true_labels), _choice_map(rm._choice_map),
        _workspace(rm._workspace), _gen(rm._gen), _composition_gmm(rm._composition_gmm)
    {

    }

    /**
     * @brief initialize the node : retrieve all the parameter of the ros parameter server and initialise the subscribers, publishers, clients and services.
     * @param nh : a Nodehandler
     */
    void initialize(const ros::NodeHandlePtr nh);

    /**
     * @brief initialize the classifiers
     * @param name of the folder in which the archive of the classifiers can be found.
     */
    void init_classifiers(const std::string& folder_name);

    /**
     * @brief release the memory of the different pointers
     */
    void release();

    /**
     * @brief retrieve the input cloud from the topic of an color and depth stream.
     * @param The inputcloud
     * @param True if artificial noise must be added (default at false)
     * @param the noise intensity
     * @return true if the pointcloud have been retrieved successfully
     */
    bool retrieve_input_cloud(ip::PointCloudT::Ptr cloud, bool with_noise = false,
                              float noise_intensity = 0.);

    /**
     * @brief publish visual feedback on topics to be displayed in rviz
     */
    void publish_feedback();

    //** GETTERS & SETTERS **\\
    const ip::PointCloudT::Ptr get_background(){return _background;}
    void set_background(const ip::PointCloudT::Ptr& cloud){_background.reset(cloud.get());}
    ip::SurfaceOfInterest& get_soi(){return _soi;}
    const std::string& get_modality(){return _modality;}
    const std::string& get_method(){return _method;}
    double get_threshold(){return _threshold;}
    int get_nbr_class(){return _nbr_class;}
    void set_nbr_max_comp(int n){_nbr_max_comp = n;}
    //*/


    std::map<std::string,cmm::NNMap> _nnmap_class; /**< nnmap classifiers */
    std::map<std::string,cmm::CollabMM> _gmm_class; /**< gmm classifiers */
    cmm::MCS _mcs;/**<the multi classifier system*/

protected:
    rgbd::RGBD_Subscriber::Ptr _images_sub; /**<RGB image, Depth image and camera info subscriber*/
    ip::SurfaceOfInterest _soi;/**<Attribute to compute the saliency map*/

    std::string _method; /**<Which classifier will be use to compute the saliency map : expert, nnmap, gmm, random or sift */
    std::string _mode; /**<Exploration or exploitation mode */
    std::string _load_exp; /**<Path to archive of a previous experiment*/
    std::string _load_comp; /**<Path to an archive of classifier to compose with current classifier into training*/
    std::string _modality; /**<The name of the visual feature extracted*/
    int _dimension; /**<Dimension of the feature space*/
    double _threshold = 0.5; /**<The threshold to binarize the classification*/
    int _nbr_class = 2; /**<The number of classes in the problem*/
    std::map<std::string,int> _modalities; /**<A list of names of visual features, one classifier for each of this modality will be trained*/
    std::vector<std::string> _mcs_mod_mapping; /**<Computation operation used by the multiclassifier system*/
    boost::random::mt19937 _gen; /**<Boost random number generator*/



    int _nbr_max_comp = 0; /**<The number of component in the GMMs of each class. If set at 0 then no limit is defined.*/

    cmm::CollabMM _composition_gmm; /**<The classifier on top of which the new classifier is trained*/

    ip::PointCloudT::Ptr _background;/**<Pointcloud of the background. only for export mode*/
    bool _background_saved; /**<if the bachground is already saved */

    std::map<uint32_t,int> _true_labels; /**<The labels from the groundtruth*/

    map_t _choice_map; /**<the choice distribution*/
    std::shared_ptr<ip::workspace_t> _workspace; /**<workspace of the robot*/

    //**Publisher for the visual feedback*/
    std::map<std::string,std::vector<std::shared_ptr<ros::Publisher>>> _weighted_cloud_pub; /**<relevance map publisher*/
    std::unique_ptr<ros::Publisher> _choice_dist_cloud_pub; /**<choice distribution map publisher*/
    std::unique_ptr<ros::Publisher> _input_cloud_pub;
    //*/

    std::unique_ptr<ros::ServiceClient> _cnn_features_client; /**<Client to retrieve the CNN features*/


    /**
     * @brief clear supervoxels. Take as template static parameters
     * relative to the supervoxels extraction
     */
    template <typename param>
    void _clear_supervoxels(){
        _soi.clear<param>();
    }

    /**
     * @brief _compute_supervoxels
     * @param input_cloud
     * @param with_workspace
     * @return
     */
    bool _compute_supervoxels(const ip::PointCloudT::Ptr input_cloud, bool with_workspace = true);

    /**
     * @brief Compute the relevance map for the next iteration
     * @param input cloud
     * @return true if all went good false otherwise
     */
    bool _compute_relevance_map();

    /**
     * @brief compute the choice distribution map and choose the next supervoxel to explore
     * @param chosen supervoxel
     * @param chosen supervoxel's label
     * @return true if all went good
     */
    bool _compute_choice_map(pcl::Supervoxel<image_processing::PointT> &sv, uint32_t &lbl);

    /**
     * @brief _add_new_sample
     * @param label
     */
    void _add_new_sample(int label);

    /**
     * @brief _update_classifiers
     */
    void _update_classifiers();

};

}

#endif //RELEVANCE_MAP_NODE_HPP
