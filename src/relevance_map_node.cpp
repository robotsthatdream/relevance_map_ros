#include <relevance_map/relevance_map_node.hpp>

using namespace relevance_map;
namespace ip = image_processing;

void relevance_map_node::initialize(const ros::NodeHandlePtr nh){

    XmlRpc::XmlRpcValue exp_params; // parameters stored in experiment_params.yml
    XmlRpc::XmlRpcValue modalities;
    XmlRpc::XmlRpcValue wks;

    nh->getParam("modalities",modalities);
    nh->getParam("experiment", exp_params);
    nh->getParam("experiment/workspace", wks);

    init_workspace(wks,_workspace);
    _soi_method = static_cast<std::string>(exp_params["soi"]["method"]);
    _mode = static_cast<std::string>(exp_params["soi"]["mode"]);
    _load_exp = static_cast<std::string>(exp_params["soi"]["load_exp"]);
    _modality = static_cast<std::string>(exp_params["soi"]["modality"]);
    _dimension = std::stoi(exp_params["soi"]["dimension"]);

    for(const auto& mod: modalities){
        moda = mod.second;
        _modalities.emplace(static_cast<std::string>(moda["name"]),moda["dimension"]);
    }

    for(const auto& mod : _modalities)
        _weighted_cloud_pub.emplace(mod.first,std::unique_ptr<Publisher>(
                                        new Publisher(nh->advertise<sensor_msgs::PointCloud2>
                                                      ("weighted_color_cloud_"+mod.first, 5))));
    if(_soi_method == "mcs")
        _weighted_cloud_pub.emplace("merge",std::unique_ptr<Publisher>(
                                        new Publisher(nh->advertise<sensor_msgs::PointCloud2>
                                                      ("weighted_color_cloud_mcs", 5))));

    _choice_dist_cloud_pub.reset(new Publisher(ros_nh->advertise<sensor_msgs::PointCloud2>("choice_dist_cloud",5)));


    _soi.init<relevance_map::sv_param>();


    if(_soi_method == "nnmap"){
        for(const auto& mod : _modalities)
            _nnmap_class.emplace(mod.first,iagmm::NNMap(mod.second,2,.3,0.05));
    }
    else if (_soi_method == "gmm"){
        if(!load_experiment(_soi_method,_load_exp,_modalities,_gmm_class,_nnmap_class,_mcs))
            for(const auto& mod : _modalities)
                _gmm_class.emplace(mod.first,iagmm::GMM(mod.second,2));
    }
    else if (_soi_method == "mcs"){
        if(!load_experiment(_soi_method,_load_exp,_modalities,_gmm_class,_nnmap_class,_mcs)){
            std::map<std::string,iagmm::GMM::Ptr> gmms;
            for(const auto& mod: _modalities){
                gmms.emplace(mod.first,iagmm::GMM::Ptr(new iagmm::GMM(mod.second,2)));
                _mcs = iagmm::MCS(gmms,iagmm::combinatorial::fct_map.at("sum"),iagmm::param_estimation::fct_map.at("linear"));
            }
        }
    }

}

bool relevance_map_node::_compute_soi(const ip::PointCloudT::Ptr input_cloud){
    ROS_INFO_STREAM("Computing saliency map !");
    std::chrono::system_clock::time_point timer, timer2;
    timer  = std::chrono::system_clock::now();
    timer2 = std::chrono::system_clock::now();


    //* compute the saliency map according the method selected
    _soi.clear<babbling::sv_param>();


    _soi.setInputCloud(input_cloud);

    if (_soi_method == "nnmap"){ // Use a Nearst Neighbor map
        if(!_soi.computeSupervoxel(*_workspace))
            return false;
        for(auto& classifier: _nnmap_class){
           _soi.init_weights(classifier.first,.5);
           classifier.second.default_estimation = .5;
           _soi.compute_feature(classifier.first);
           ROS_INFO_STREAM("Computing features finish for " << classifier.first << ", time spent : "
                                          << std::chrono::duration_cast<std::chrono::milliseconds>(
                                              std::chrono::system_clock::now() - timer2).count());
                          timer2 = std::chrono::system_clock::now();

           _soi.compute_weights<iagmm::NNMap>(classifier.first,classifier.second);
           ROS_INFO_STREAM("Computing weights finish for " << classifier.first << ", time spent : "
                                          << std::chrono::duration_cast<std::chrono::milliseconds>(
                                              std::chrono::system_clock::now() - timer2).count());
        }
    }
    if(_soi_method == "gmm"){ // Use a Gaussian Mixture Model
        if(!_soi.computeSupervoxel(*_workspace))
            return false;
        for(auto& classifier: _gmm_class){
           _soi.compute_feature(classifier.first);
           ROS_INFO_STREAM("Computing features finish for " << classifier.first << ", time spent : "
                                          << std::chrono::duration_cast<std::chrono::milliseconds>(
                                              std::chrono::system_clock::now() - timer2).count());
                          timer2 = std::chrono::system_clock::now();

           _soi.compute_weights<iagmm::GMM>(classifier.first,classifier.second);
           ROS_INFO_STREAM("Computing weights finish for " << classifier.first << ", time spent : "
                                          << std::chrono::duration_cast<std::chrono::milliseconds>(
                                              std::chrono::system_clock::now() - timer2).count());

        }
    }
    if(_soi_method == "mcs"){
        if(!_soi.computeSupervoxel(*_workspace))
            return false;
        for(const auto& classifier: _mcs.access_classifiers()){
            _soi.compute_feature(classifier.first);
            ROS_INFO_STREAM("Computing features finish for " << classifier.first << ", time spent : "
                                          << std::chrono::duration_cast<std::chrono::milliseconds>(
                                              std::chrono::system_clock::now() - timer2).count());
                          timer2 = std::chrono::system_clock::now();

            _soi.compute_weights<iagmm::GMM>(classifier.first,*dynamic_cast<iagmm::GMM*>(classifier.second.get()));
        }
        _soi.compute_weights<iagmm::MCS>(_mcs);
        ROS_INFO_STREAM("Computing weights finish for mcs, time spent : "
                                   << std::chrono::duration_cast<std::chrono::milliseconds>(
                                       std::chrono::system_clock::now() - timer2).count());

    }
    if (_soi_method == "random") { // Use nothing. the choice of the next supervoxel will random.
        if (!_soi.generate(*_workspace)) {
            return false;
        }
    }
    if (_soi_method == "expert") { // Use a background substraction.
        if (!_soi.generate(_background, *_workspace)) {
            return false;
        }
    }
//        if (_soi_method == "sift") { // Use SIFT Keypoints.
//            ip::DescriptorExtraction de(cv_bridge::toCvShare(_images_sub->get_rgbConstPtr())->image, "SIFT");
//            de.extract();
//            de.align(*input_cloud);
//            pcl::PointCloud<pcl::PointXYZ>::Ptr key_pts(new pcl::PointCloud<pcl::PointXYZ>(de.get_key_points_cloud()));
//            if (!_soi.generate(key_pts, *_workspace)) {
//                return false;
//            }
//        }

    //*/

    return true;
}

bool relevance_map_node::_compute_choice_map(){

    if(_soi.empty()){
        ROS_ERROR_STREAM("BABBLING : No SOI extracted");
        return false;
    }

    //* Choice of the next supervoxel to explore.

    if(_mode == "exploration" || _mode == "experiment"){
        //-_- TODO _-_ DECISION BETWEEN WHICH MODALITY TO CHOOSE FOR THE NEXT POINT TO EXPLORE

        ROS_INFO_STREAM("BABBLING_NODE: Choosing next sv to explore,  time spent so far : "
                        << std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now() - timer).count());

        if(_soi_method == "nnmap"){
            _possible_choice = _soi.choice_of_soi(_modality,sv, lbl);
        }else if(_soi_method == "gmm"){
            std::vector<std::pair<Eigen::VectorXd,double>> samples;

            std::vector<uint32_t> lbl;
            ip::SurfaceOfInterest::saliency_map_t weights = _soi.get_weights()[_modality];
            for(const auto& sv : _soi.getSupervoxels()){
                samples.push_back(std::make_pair(_soi.get_feature(sv.first,_modality),weights[sv.first]));
                lbl.push_back(sv.first);
            }

            _gmm_class[_modality].set_distance_function(ip::HistogramFactory::chi_squared_distance);
            int index = _gmm_class[_modality].next_sample(samples,_choice_dist_map);
            lbl = lbl[index];
            sv = *(_soi.getSupervoxels()[lbl]);


        }
        else if(_soi_method == "mcs"){
            std::vector<uint32_t> lbl;

            std::map<std::string,std::vector<std::pair<Eigen::VectorXd,double>>> samples;
            ip::SurfaceOfInterest::saliency_map_t weights = _soi.get_weights()[_modality];
            std::map<std::string, Eigen::VectorXd> sample;
            for(const auto& sv : _soi.getSupervoxels()){
                sample = _soi.get_features(sv.first);
                for(const auto& feature : sample){
                    samples[feature.first].push_back(
                                std::make_pair(feature.second,weights[sv.first]));
                }
                lbl.push_back(sv.first);
            }
            int ind = _mcs.next_sample(samples,_choice_dist_map);
            lbl = lbl[ind];
            sv = *(_soi.getSupervoxels()[lbl]);

        }
        else ROS_ERROR_STREAM("BABBLING : Unknown soi method" << _soi_method);
        // TODO the other modes !
    }

        // -_- TODO _-_ EXPLOITATION MODE
//            else if(_mode == "exploration")
//                _possible_choice = _soi.choice_of_soi_by_uncertainty(_sv,_lbl);

    //*/
    ROS_INFO_STREAM("Computing saliency map finish, time spent : "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::system_clock::now() - timer).count());
    return true;

}

void relevance_map_node::publish_feedback(){
    sensor_msgs::PointCloud2 weighted_cloud;

    ip::PointCloudT w_cl;

    //visualisation of relevance map
    for(auto& pub: _weighted_cloud_pub){
        w_cl = _soi.getColoredWeightedCloud(pub.first);
        pcl::toROSMsg(w_cl, weighted_cloud);
        weighted_cloud.header = _images_sub->get_depth().header;
        pub.second->publish(weighted_cloud);
    }

    //visualization of choice distribution map
    ip::PointCloudT choice_ptcl;
    for(size_t i = 0; i < lbl.size(); i++){
        pcl::Supervoxel<ip::PointT>::Ptr current_sv = _soi.getSupervoxels()[lbl[i]];
        float c = 255.*_choice_dist_map(i);
        uint8_t color = c;
        ip::PointT pt;
        for(auto v : *(current_sv->voxels_)){
            pt.x = v.x;
            pt.y = v.y;
            pt.z = v.z;
            pt.r = color;
            pt.g = color;
            pt.b = color;
            choice_ptcl.push_back(pt);
        }
    }

    sensor_msgs::PointCloud2 choice_cloud;
    pcl::toROSMsg(choice_ptcl, choice_cloud);
    choice_cloud.header = _images_sub->get_depth().header;
    _choice_dist_cloud_pub->publish(choice_cloud);
}
