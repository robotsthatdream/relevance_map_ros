/* Written by Léni K. Le Goff
 *
 * This sample of code is a minimal example of how to implement quickly a node
 * to produce a relevance map from an archive of a trained classifier and from a rgbd camera flow.
 */


#include <iostream>
#include <relevance_map/relevance_map_node.hpp>
#include <relevance_map/parameters.hpp>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <boost/archive/text_iarchive.hpp>

namespace ip = image_processing;
namespace rm = relevance_map;
namespace rgbd = rgbd_utils;

class ExampleNode : public rm::relevance_map_node{
public:
    ExampleNode(){
        _nh.reset(new ros::NodeHandle); //Instantiate the nodehandle.

        /* Initialize the node by instanciating all the publishers, subcribers, clients and services needed.
         * And retrieve the parameters in the paramter server.
         */
        initialize(_nh);

        /* Initialize the classifer from the folder _load_exp containing an archive of the classifier.
         */
        init_classifiers(_load_exp);
    }

    /*The destructor calls release() to destroy all the pointers.
     */
    ~ExampleNode(){
        release();
    }

    /* The execute function contain all the computation to produce a relevance on a current pointcloud.
     * The computation is done in 3 steps.
     */
    void execute(){

        /* Step 1 :
         *      The current pointcloud is retrieved.
         *      This pointcloud comes from an image stream composed of color images and depth images
         *      Their are retrieved by listening to topics specified in the launch file of this node :
         *      example_node.launch.
         */
        ip::PointCloudT::Ptr cloud(new ip::PointCloudT);
        if(!retrieve_input_cloud(cloud)){
            ROS_ERROR_STREAM("Unable to retrieve inputcloud");
            return;
        }

        /* Step 2 :
         *      Extract the supervoxels from the current pointcloud.
         *      First the previous supervoxel segmentation is cleared.
         *      Then the new supervoxel segmentation is computed.
         */
        _clear_supervoxels<rm::sv_param>();
        if(!_compute_supervoxels(cloud, true /* use workspace crop information */)){
            ROS_ERROR_STREAM("Unable to extract supervoxels");
            return;
        }

        /*Step 3 :
         *      Compute the relevance map itself.
         */
        if(!_compute_relevance_map()){
            ROS_ERROR_STREAM("Unable to compute the relevance map");
            return;
        }


        publish_feedback();
   }
   private:
    ros::NodeHandlePtr  _nh;
};

int main(int argc, char** argv){

    ros::init(argc,argv,"example_node");

    ExampleNode en;

    while(ros::ok()){
        en.execute();
        ros::spinOnce();
    }
    return 0;
}
