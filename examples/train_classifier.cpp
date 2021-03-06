/* Written by Léni K. Le Goff
 *
 * In this sample of code, it is shown how to train a new classifier or how to
 * continue the training of a stored classifier.
 * The training is done by processing a rgb-d video stream. An expert is used to
 * label the samples collected for the trainging.
 */

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <relevance_map/parameters.hpp>
#include <relevance_map/relevance_map_node.hpp>
#include <relevance_map/score_computation.hpp>
#include <relevance_map/utilities.hpp>
#include <sensor_msgs/PointCloud2.h>

namespace ip = image_processing;
namespace rm = relevance_map;
namespace rgbd = rgbd_utils;

class TrainClassifier : public rm::relevance_map_node {
  public:
    TrainClassifier() {
        _nh.reset(new ros::NodeHandle("~"));

        /* Initialize the node by instanciating all the publishers, subcribers,
         * clients and services needed.
         * And retrieve the parameters in the paramter server.
         */
        initialize(_nh);

        /* Initialize the classifer from the folder _load_exp containing an
         * archive of the classifier.
         */
        init_classifiers(_load_exp);

        /* A reference pointcloud must be taken to be able to distinguish
         * if a supervoxel is part of the background (class 0) or from objects
         * (class 1).
         * To do so you can use any method you want. In the following to method
         * is shown.
         * * * *
         * First possibility : you take the background for the input rgb-d
         * stream.
         */
        if (1) {
            ip::PointCloudT::Ptr input_cloud(new ip::PointCloudT);
              /* This is the step that causes these lines:
               * [ERROR] [1563441618.205405755]: Waiting for input images : depth 0; rgb 0
               */
            while (!retrieve_input_cloud(input_cloud)) {
                ros::spinOnce();
            }
            std::cout << "BABBLING_NODE : take the background" << std::endl;
            set_background(input_cloud);
            std::cout << "BABBLING_NODE : done" << std::endl;

            std::cout << "BABBLING_NODE : Press enter to start.";
            std::cin.ignore();
        } else {
            /* Second possibility : you take the background from pointcloud file
             * (.pcd).
         */
            ip::PointCloudT::Ptr background(new ip::PointCloudT);

            std::string pcd_file;
            _nh->getParam("/global/background",
                          pcd_file); // retrieve the name of the pcd file from
                                     // the parameters server. The parameter is
                                     // given in the launch file.

            if (pcl::io::loadPCDFile<ip::PointT>(pcd_file, *background) ==
                -1) //* load the file
            {
                ROS_ERROR_STREAM("Couldn't read file " << pcd_file);
                exit(1);
            }
            set_background(background);
        }
    }

    /*The destructor calls release() to destroy all the pointers.
     */
    ~TrainClassifier() { release(); }

    /* The execute function contain all the computation to produce a relevance
     * map on a current pointcloud
     * and add a new sample in the training dataset.
     * The computation is done in 5 steps.
     */
    void execute() {

        /* Step 1 :
         *      The current pointcloud is retrieved.
         *      This pointcloud comes from an image stream composed of color
         * images and depth images
         *      Their are retrieved by listening to topics specified in the
         * launch file of this node :
         *      example_node.launch.
         */
        ip::PointCloudT::Ptr cloud(new ip::PointCloudT);
        if (!retrieve_input_cloud(cloud)) {
            ROS_ERROR_STREAM("Unable to retrieve inputcloud");
            return;
        }

        /* Step 2 :
         *      Extract the supervoxels from the current pointcloud.
         *      First the previous supervoxel segmentation is cleared.
         *      Then the new supervoxel segmentation is computed.
         */
        _clear_supervoxels<rm::sv_param>();
        if (!_compute_supervoxels(cloud,
                                  true /* use workspace crop information */)) {
            ROS_ERROR_STREAM("Unable to extract supervoxels");
            return;
        }

        /*Step 3 :
         *      Compute the relevance map itself.
         * This is the step that displays:
         * [ INFO] [1563441866.597143702]: Computing saliency map !
         * [ INFO] [1563441866.597276566]: Computing saliency map finish, time spent : 0
         * [ INFO] [1563441867.626821770]: Computing saliency map !
         * [ INFO] [1563441871.167163565]: Computing features finish for centralFPFHLabHist, time spent : 3540
         */
        if (!_compute_relevance_map()) {
            ROS_ERROR_STREAM("Unable to compute the relevance map");
            return;
        }

        /* Step 4:
         *      A choice distribution is computed to choose a supervoxel in the
         * current segmentation
         *      to be explored by a robotic system.
         *      This step is useful when the classifier is in training.
         */
        pcl::Supervoxel<ip::PointT>
            sv;          // the variable to retrieve the chosen supervoxel
        uint32_t sv_lbl; // the variable to retrieve the label of the chosen in
                         // supervoxel
        if (!_compute_choice_map(sv, sv_lbl)) {
            ROS_ERROR_STREAM("Unable to compute the choice map");
            return;
        }

        /* Step 5:
         * Add a new sample with a label determined by looking if the centroid
         * of the selected supervoxel is in the background.
         * If true the label is equal to 0 otherwise the label is equal to 1.
         * And update the parameters of the classifier.
         */
        int label = rm::is_in_cloud(sv.centroid_, _background) ? 0 : 1;
        _add_new_sample(sv_lbl, label);
        _update_classifiers();

        /* Step 6: Save the classifier by serialization and save the training
         * dataset.
         * This piece of code is not compilable !!
         */
        std::stringstream folder;
        // folder << ros_home << "/cafer_db/dream_babbling/babbling/iteration_"
        // << _counter_iter;
        folder << "."; // TODO number

        ROS_INFO_STREAM("Saving to " << folder.str() );

        for (const auto &classifier : _gmm_class) {
            if (classifier.second.dataset_size() != 0) {
                // Write dataset in a file in a yml format.=
                classifier.second.get_samples().save_yml(
                    folder.str() + "/dataset_" + classifier.first + ".yml");
                // serializartion of the classifier
                std::stringstream sstream;
                boost::archive::text_oarchive oarch(sstream);
                oarch << classifier.second;
                // write serialized classifier in a file
                rm::write_data(folder.str() + "/gmm_" + classifier.first,
                               sstream.str());
            }
        }

        /* Step 7: Computation of perfomence scores of the classifier.
		TODO encapsulate this part
         */
        std::map<uint32_t,int> true_labels;
        std::vector<uint32_t> lbls;
        for(const auto& sv : _soi.getSupervoxels()){
            true_labels.emplace(sv.first,0);
            lbls.push_back(sv.first);
        }
        tbb::parallel_for(tbb::blocked_range<size_t>(0,lbls.size()),
                                      [&](const tbb::blocked_range<size_t>& r){
                        ip::SupervoxelArray supervoxel = _soi.getSupervoxels();
                        for(int i = r.begin(); i != r.end(); i++)
                            true_labels[lbls[i]] = rm::is_in_cloud(supervoxel[lbls[i]]->centroid_,_background) ? 0 : 1;
        });

	rm::score_computation<TrainClassifier> sc(this,true_labels, "out_score");
        sc.compute_scores_results(_counter_iter);

        publish_feedback();
        _counter_iter++;
    }
    bool is_finished() { return _counter_iter >= _max_iter; }

  private:
    ros::NodeHandlePtr _nh;
    int _counter_iter = 0;
    const int _max_iter = 200;
};

int main(int argc, char **argv) {

    ros::init(argc, argv, "train_classifier");

    TrainClassifier tc;

    while (ros::ok() && !tc.is_finished()) {
        tc.execute();
        ros::spinOnce();
    }
    return 0;
}
