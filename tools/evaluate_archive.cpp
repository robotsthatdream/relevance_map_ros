#include <iostream>

#include <relevance_map/relevance_map_node.hpp>
#include <relevance_map/score_computation.hpp>
#include <relevance_map/utilities.hpp>

#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

#include <boost/archive/text_iarchive.hpp>

namespace ip = image_processing;
namespace rm = relevance_map;

class evaluate_archive : public rm::relevance_map_node{
public:
    evaluate_archive(const std::string& clouds_folder, const std::string& exp_archive, const ros::NodeHandlePtr& nh){

//        initialize(nh);



        load_clouds(clouds_folder);
        XmlRpc::XmlRpcValue param;
        nh->getParam("/eval_arch_param", param);
        std::string output_file = static_cast<std::string>(param["output_file"]);

        _output_file = exp_archive + std::string("/") + output_file;
        _method = static_cast<std::string>(param["method"]);
        _modality = "centralFPFHLabHist";
        _dimension = 48;
        _modalities.emplace(_modality,0);

        rm::load_archive(exp_archive,_iter_list);
        _progress = 0;
        _files_total_nb = _input_clouds.size();
        _files_total_nb = _files_total_nb * _backgrounds.size();
        _files_total_nb = _files_total_nb * _iter_list.size();
    }

    void load_clouds(const std::string& folder){
        ROS_INFO_STREAM("load point clouds");
        boost::filesystem::directory_iterator dir_it(folder);
        boost::filesystem::directory_iterator end_it;
        std::vector<std::string> split_str;

        for(;dir_it != end_it; dir_it++){
            boost::split(split_str,dir_it->path().string(),boost::is_any_of("/"));
            boost::filesystem::directory_iterator subdir_it(dir_it->path().string());

            for(;subdir_it != end_it;subdir_it++){
                ip::PointCloudT::Ptr cloud(new ip::PointCloudT);
                pcl::io::loadPCDFile(subdir_it->path().string(),*cloud);
                if(split_str.back() != "back")
                    _input_clouds.push_back(cloud);
                else _backgrounds.push_back(cloud);
            }
        }
    }

    bool compute_rm(const ip::PointCloudT::Ptr& input_cloud){
        if(!_compute_supervoxels(input_cloud,false)){
            ROS_ERROR_STREAM("unable to generate supervoxels");
            return false;
        }
        if(!_compute_relevance_map()){
            ROS_ERROR_STREAM("unable to compute relevance map");
            return false;
        }

        return true;
    }

    void run(){
        _gmm_class.clear();
        if(!rm::load_experiment(_method,_iter_list.front(),_modalities,_gmm_class,_nnmap_class,_mcs)){
            ROS_INFO_STREAM("Empty archive");
            _iter_list.pop();
            _progress+=_input_clouds.size()*_backgrounds.size();
            return;
        }
        if(_method=="composition")
            rm::load_compo_gmm(_load_comp,_composition_gmm);

        _iter_list.pop();
        std::vector<double> p = std::vector<double>(_nbr_class,0),
                r = std::vector<double>(_nbr_class,0),
                a = std::vector<double>(_nbr_class,0),
                P = std::vector<double>(_nbr_class,0),
                R = std::vector<double>(_nbr_class,0),
                A = std::vector<double>(_nbr_class,0);
        rm::score_computation<evaluate_archive> sc(this,_output_file);
        sc.statistics_name = {"nbr_samples","precision","recall","accuracy"};
        for(const auto& cloud : _input_clouds){
            compute_rm(cloud);
            sc.init();
            for(const auto& back : _backgrounds){
                _progress += 1.;
                _background = back;

                std::map<uint32_t,int> true_labels;
                for(const auto& sv: _soi.getSupervoxels()){
                    if(rm::is_in_cloud(sv.second->centroid_,_background))
                        true_labels.emplace(sv.first,0);
                    else true_labels.emplace(sv.first,1);
                }
                sc.set_true_labels(true_labels);
                std::vector<double> precision, recall, accuracy;
                sc.compute_precision_recall(precision,recall,accuracy);
//                ROS_INFO_STREAM("--------------------------------------------------------");
//                ROS_INFO_STREAM("scores \n"
//                                << " precision : " << precision << "\n"
//                                << " recall : " << recall << "\n"
//                                << " accuracy : " << accuracy << "\n");
//                ROS_INFO_STREAM("--------------------------------------------------------");
                for(int i = 0; i < precision.size(); i++){
                    p[i] += precision[i];
                    r[i] += recall[i];
                    a[i] += accuracy[i];
                }
            }
            for(int i = 0; i < p.size(); i++){
                p[i] = p[i]/(double)_backgrounds.size();
                r[i] = r[i]/(double)_backgrounds.size();
                a[i] = a[i]/(double)_backgrounds.size();
                P[i] += p[i];
                R[i] += r[i];
                A[i] += a[i];
            }
//            ROS_INFO_STREAM("--------------------------------------------------------");
//            ROS_INFO_STREAM("scores \n"
//                            << " precision : " << p << "\n"
//                            << " recall : " << r << "\n"
//                            << " accuracy : " << a << "\n");
//            ROS_INFO_STREAM("--------------------------------------------------------");
            p = std::vector<double>(_nbr_class,0); r = std::vector<double>(_nbr_class,0); a = std::vector<double>(_nbr_class,0);
            ROS_INFO_STREAM(_progress/_files_total_nb*100. << "% progress");
        }
        for(int i = 0; i < P.size(); i++){
            P[i] = P[i]/(double)_input_clouds.size();
            R[i] = R[i]/(double)_input_clouds.size();
            A[i] = A[i]/(double)_input_clouds.size();
        }


        std::stringstream str;
        str << "iteration_" << _gmm_class[_modality].get_samples().size();
        std::vector<std::vector<double>> scores = {P,R,A};
        sc.set_results(str.str(),std::make_tuple((double)_gmm_class[_modality].get_samples().size(),scores));


        ROS_INFO_STREAM("--------------------------------------------------------");
        ROS_INFO_STREAM("scores for iteration " << str.str() << "\n"
                        << sc.results_to_string());
        ROS_INFO_STREAM("--------------------------------------------------------");

        sc.write_result(_output_file);
    }

    bool is_finish(){
        return _iter_list.empty();
    }

private:
    std::vector<ip::PointCloudT::Ptr> _input_clouds;
    std::vector<ip::PointCloudT::Ptr> _backgrounds;
    ros::NodeHandle _nh;
    iagmm::GMM _gmm;
    std::queue<std::string> _iter_list;
    std::string _output_file;
    double _progress;
    double _files_total_nb;

};

int main(int argc, char** argv){

    ros::init(argc,argv,"evaluate_archive");
    ros::NodeHandlePtr nh(new ros::NodeHandle);

    if(argc != 3){
        ROS_WARN_STREAM("Usage :\n\targ1 : path to a folder containing point cloud pcd files\n\targ2 : archive of a classifier");
        return 1;
    }
    evaluate_archive ea(argv[1],argv[2],nh);
    while(ros::ok() && !ea.is_finish()){
        ea.run();
        ros::spinOnce();
    }
    return 0;
}
