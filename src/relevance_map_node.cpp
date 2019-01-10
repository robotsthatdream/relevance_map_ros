#include <relevance_map/relevance_map_node.hpp>
#include <relevance_map/utilities.hpp>
#include <relevance_map/parameters.hpp>

#include <pcl_conversions/pcl_conversions.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <chrono>

using namespace relevance_map;
namespace ip = image_processing;

void relevance_map_node::initialize(const ros::NodeHandlePtr nh){

    XmlRpc::XmlRpcValue glob_params; // parameters stored in global_params.yml
    XmlRpc::XmlRpcValue exp_params; // parameters stored in experiment_params.yml
    XmlRpc::XmlRpcValue modalities, moda;
    XmlRpc::XmlRpcValue wks;

    nh->getParam("/global", glob_params);
    nh->getParam("modalities",modalities);
    nh->getParam("experiment", exp_params);
    nh->getParam("experiment/workspace", wks);

    init_workspace(wks,_workspace);
    _method = static_cast<std::string>(exp_params["soi"]["method"]);
    _mode = static_cast<std::string>(exp_params["soi"]["mode"]);
    _load_exp = static_cast<std::string>(exp_params["soi"]["load_exp"]);
    _load_comp = static_cast<std::string>(exp_params["soi"]["load_comp"]);
    _modality = static_cast<std::string>(exp_params["soi"]["modality"]);
    _dimension = std::stoi(exp_params["soi"]["dimension"]);
    _nbr_class = 2; /*std::stod(exp_params["soi"]["nbr_class"]);*/
    _threshold = std::stod(exp_params["soi"]["threshold"]); // 1./(double)_nbr_class;

    iagmm::Component::_alpha = std::stod(exp_params["soi"]["alpha"]);
    iagmm::Component::_outlier_thres = 0;



    for(const auto& mod: modalities){
        moda = mod.second;
        _modalities.emplace(static_cast<std::string>(moda["name"]),moda["dimension"]);
    }

    // Initialisation of the RGB and Depth images subscriber.
    _images_sub.reset(new rgbd_utils::RGBD_Subscriber(glob_params["rgb_info_topic"],
                                          glob_params["rgb_topic"],
                                          glob_params["depth_info_topic"],
                                          glob_params["depth_topic"],
                                          *nh));

    for(const auto& mod : _modalities){
        std::vector<std::shared_ptr<ros::Publisher>> vect;
        for(int i = 0; i < _nbr_class; i++){
            vect.push_back(std::shared_ptr<ros::Publisher>(
                               new ros::Publisher(
                                   nh->advertise<sensor_msgs::PointCloud2>
                                   ("weighted_color_cloud_"+std::to_string(i)+"_"+mod.first, 5))));
        }
        _weighted_cloud_pub.emplace(mod.first,vect);
    }

    if(_method == "mcs"){
        std::vector<std::shared_ptr<ros::Publisher>> vect;
        for(int i = 0; i < _nbr_class; i++){
            vect.push_back(std::shared_ptr<ros::Publisher>(
                               new ros::Publisher(
                                   nh->advertise<sensor_msgs::PointCloud2>
                                   ("weighted_color_cloud_"+std::to_string(i)+"_mcs", 5))));
        }
        _weighted_cloud_pub.emplace("merge",vect);
    }

    _input_cloud_pub.reset(new ros::Publisher(
                               nh->advertise<sensor_msgs::PointCloud2>("input_cloud",5)));


    _choice_dist_cloud_pub.reset(new ros::Publisher(
                                 nh->advertise<sensor_msgs::PointCloud2>("choice_dist_cloud",5)));

    _cnn_features_client.reset(new ros::ServiceClient(
                             nh->serviceClient<cnn_features>("cnn_features")));

    _soi.init<sv_param>();

    _background_saved = false;
    _background.reset(new ip::PointCloudT);
}

void relevance_map_node::init_classifiers(const std::string &folder_name){
    if(_method == "nnmap"){
        _nnmap_class.clear();
        for(const auto& mod : _modalities)
            _nnmap_class.emplace(mod.first,iagmm::NNMap(mod.second,.3,0.05));
    }
    else if (_method == "gmm"){
        _gmm_class.clear();
        if(!load_experiment(_method,folder_name,_modalities,_gmm_class,_nnmap_class,_mcs)){
            for(const auto& mod : _modalities){
                iagmm::GMM gmm(mod.second,_nbr_class,_nbr_max_comp);
                _gmm_class.emplace(mod.first,gmm);
            }
        }
        for(auto& gmm: _gmm_class){
            gmm.second.set_loglikelihood_driver(false);
            gmm.second.set_max_nb_components(4);
            gmm.second.use_confidence(true);
            gmm.second.use_uncertainty(true);
            gmm.second.use_novelty(false);
        }
    }
    else if (_method == "composition"){
        _gmm_class.clear();
        if(!load_experiment(_method,folder_name,_modalities,_gmm_class,_nnmap_class,_mcs)){
            for(const auto& mod : _modalities){
                iagmm::GMM gmm(mod.second,_nbr_class,_nbr_max_comp);
                _gmm_class.emplace(mod.first,gmm);
            }
        }
        for(auto& gmm: _gmm_class){
            gmm.second.set_loglikelihood_driver(false);
            gmm.second.set_max_nb_components(4);
            gmm.second.use_confidence(true);
            gmm.second.use_uncertainty(true);
            gmm.second.use_novelty(false);
            gmm.second.skip_bootstrap = true;
        }
        load_compo_gmm(_load_comp,_composition_gmm);
        _composition_gmm.skip_bootstrap = true;
    }
    else if (_method == "mcs"){
        if(!load_experiment(_method,folder_name,_modalities,_gmm_class,_nnmap_class,_mcs)){
            std::map<std::string,iagmm::GMM::Ptr> gmms;
            for(const auto& mod: _modalities){
                gmms.emplace(mod.first,iagmm::GMM::Ptr(new iagmm::GMM(mod.second,_nbr_class)));
                _mcs = iagmm::MCS(gmms,iagmm::combinatorial::fct_map.at("sum"),
                                  iagmm::param_estimation::fct_map.at("linear"));
            }
        }
    }else ROS_ERROR_STREAM(_method << " unknown relevance map method");
}

void relevance_map_node::release(){
    _images_sub.reset();

    for(auto& pub : _weighted_cloud_pub)
        for(int i = 0; i < _nbr_class; i++)
            pub.second[i].reset();
    _choice_dist_cloud_pub.reset();
    _background.reset();
    _input_cloud_pub.reset();

}

bool relevance_map_node::retrieve_input_cloud(ip::PointCloudT::Ptr cloud, bool with_noise, float noise_intensity){

    sensor_msgs::ImageConstPtr depth_msg(
                new sensor_msgs::Image(_images_sub->get_depth()));
    sensor_msgs::ImageConstPtr rgb_msg(
                new sensor_msgs::Image(_images_sub->get_rgb()));
    sensor_msgs::CameraInfoConstPtr info_msg(
                new sensor_msgs::CameraInfo(_images_sub->get_rgb_info()));

    if(depth_msg->data.empty() || rgb_msg->data.empty()){
        ROS_ERROR_STREAM("Waiting for input images : depth " <<  !depth_msg->data.empty()
                         << "; rgb " << !rgb_msg->data.empty());
        return false;
    }

    rgbd_utils::RGBD_to_Pointcloud converter;
    converter.set_depth(*depth_msg);
    converter.set_rgb(*rgb_msg);
    converter.set_rgb_info(*info_msg);
    converter.set_with_noise(with_noise);
    converter.set_std_dev_noise(noise_intensity);
    converter.convert();
    sensor_msgs::PointCloud2 cloud_msg = converter.get_pointcloud();

    cloud_msg.header = depth_msg->header;
//    _input_cloud_pub->publish(cloud_msg);

    pcl::fromROSMsg(cloud_msg,*cloud);

    return true;

}

bool relevance_map_node::_compute_supervoxels(const ip::PointCloudT::Ptr input_cloud,
                                              bool with_workspace){
    _soi.clear<sv_param>();
    _soi.setInputCloud(input_cloud);

    if(with_workspace){
        if(!_soi.computeSupervoxel(*_workspace))
            return false;
    }
    else {
        if(!_soi.computeSupervoxel())
            return false;
    }
    _soi.filter_supervoxels(6);
    return true;
}

bool relevance_map_node::_compute_relevance_map(){
    ROS_INFO_STREAM("Computing saliency map !");
    std::chrono::system_clock::time_point timer, timer2;
    timer  = std::chrono::system_clock::now();
    timer2 = std::chrono::system_clock::now();


    //* compute the relevance map according the method selected
    _soi.init_features();

    if (_method == "nnmap"){ // Use a Nearst Neighbor map (only for 2 class problem)
        for(auto& classifier: _nnmap_class){
           _soi.init_weights(classifier.first,2,.5);
           classifier.second.default_estimation = .5;
           if(classifier.first == "cnn"){
                Eigen::Matrix4f projection_matrix;
                camera_info_to_proj_matrix(_images_sub->get_rgb_info(),projection_matrix);
                compute_cnn_features(_cnn_features_client,
                                     *(cv_bridge::toCvCopy(_images_sub->get_rgb(),"rgb8")),
                                     _soi,projection_matrix);
           }else
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
    else if(_method == "gmm"){ // Use Collaborative Mixture Models
        for(const auto& classifier: _gmm_class){
            if(classifier.first == "cnn"){
                 Eigen::Matrix4f projection_matrix;
                 camera_info_to_proj_matrix(_images_sub->get_rgb_info(),projection_matrix);
                 compute_cnn_features(_cnn_features_client,
                                      *(cv_bridge::toCvCopy(_images_sub->get_rgb(),"rgb8")),
                                      _soi,projection_matrix);
            }else _soi.compute_feature(classifier.first);
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
    else if(_method == "composition"){ // Use Collaborative Mixture Models
        for(const auto& classifier: _gmm_class){
            if(classifier.first == "cnn"){
                 Eigen::Matrix4f projection_matrix;
                 camera_info_to_proj_matrix(_images_sub->get_rgb_info(),projection_matrix);
                 compute_cnn_features(_cnn_features_client,
                                      *(cv_bridge::toCvCopy(_images_sub->get_rgb(),"rgb8")),
                                      _soi,projection_matrix);
            }else _soi.compute_feature(classifier.first);
           ROS_INFO_STREAM("Computing features finish for " << classifier.first << ", time spent : "
                                          << std::chrono::duration_cast<std::chrono::milliseconds>(
                                              std::chrono::system_clock::now() - timer2).count());
                          timer2 = std::chrono::system_clock::now();

           _soi.compute_weights<iagmm::GMM>(classifier.first,classifier.second,_composition_gmm);
           ROS_INFO_STREAM("Computing weights finish for " << classifier.first << ", time spent : "
                                          << std::chrono::duration_cast<std::chrono::milliseconds>(
                                              std::chrono::system_clock::now() - timer2).count());

        }
    }

    else if(_method == "mcs"){ //Use Multi Classifier System with GMM
        for(const auto& classifier: _mcs.access_classifiers()){
            if(classifier.first == "cnn"){
                 Eigen::Matrix4f projection_matrix;
                 camera_info_to_proj_matrix(_images_sub->get_rgb_info(),projection_matrix);
                 compute_cnn_features(_cnn_features_client,
                                      *(cv_bridge::toCvCopy(_images_sub->get_rgb(),"rgb8")),
                                      _soi,projection_matrix);
            }else
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
    else if (_method == "random") { // Use nothing. the choice of the next supervoxel will be random.
        _soi.init_weights("random",2);

    }
    else if (_method == "expert") { // Use a background substraction.
        if (!_soi.generate(_background, *_workspace)) {
            return false;
        }
    }else {
        ROS_ERROR_STREAM(_method << " unknown relevance map method");
        return false;
    }
//        if (_method == "sift") { // Use SIFT Keypoints.
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

bool relevance_map_node::_compute_choice_map(pcl::Supervoxel<ip::PointT> &sv, uint32_t &lbl){

    std::chrono::system_clock::time_point timer;
    timer  = std::chrono::system_clock::now();
    Eigen::VectorXd choice_dist_map;

    if(_soi.empty()){
        ROS_ERROR_STREAM("No SOI extracted");
        return false;
    }

    //* Choice of the next supervoxel to explore.

    if(_mode == "exploration" || _mode == "experiment"){
        //-_- TODO _-_ DECISION BETWEEN WHICH MODALITY TO CHOOSE FOR THE NEXT POINT TO EXPLORE

        ROS_INFO_STREAM("Choosing next sv to explore");

        if(_method == "nnmap"){
            _soi.choice_of_soi(_modality,sv, lbl);
        }else if(_method == "gmm"){
            std::vector<std::pair<Eigen::VectorXd,std::vector<double>>> samples;

            std::vector<uint32_t> lbl_vct;
            ip::SurfaceOfInterest::relevance_map_t weights = _soi.get_weights()[_modality];
            for(const auto& sv : _soi.getSupervoxels()){
                samples.push_back(std::make_pair(_soi.get_feature(sv.first,_modality),weights[sv.first]));
                lbl_vct.push_back(sv.first);
            }

            _gmm_class[_modality].set_distance_function(ip::HistogramFactory::chi_squared_distance);
            int index = _gmm_class[_modality].next_sample(samples,choice_dist_map);
            lbl = lbl_vct[index];
            sv = *(_soi.getSupervoxels()[lbl]);

            _choice_map.clear();
            for(int i = 0; i < choice_dist_map.rows(); i++)
                _choice_map.emplace(lbl_vct[i], choice_dist_map(i));


        }else if(_method == "composition"){
            std::vector<std::pair<Eigen::VectorXd,std::vector<double>>> samples_est;
            std::vector<Eigen::VectorXd> samples;
            Eigen::VectorXd comp_filter;
            std::vector<uint32_t> lbl_vct;
            ip::SurfaceOfInterest::relevance_map_t weights = _soi.get_weights()[_modality];
            for(const auto& sv : _soi.getSupervoxels()){
                Eigen::VectorXd feat = _soi.get_feature(sv.first,_modality);
                samples_est.push_back(std::make_pair(feat,weights[sv.first]));
                lbl_vct.push_back(sv.first);
                samples.push_back(feat);
            }

            _gmm_class[_modality].set_distance_function(ip::HistogramFactory::chi_squared_distance);
            _composition_gmm.estimate_features(samples,comp_filter);
            int index = _gmm_class[_modality].next_sample(samples_est,choice_dist_map,comp_filter);

//            boost::random::uniform_int_distribution<> dist_uni(0,samples.size()-1);
//            boost::random::uniform_real_distribution<> distrib(0,1);

//            int index;
//            double cumul = 0, total = 0;
//            bool all_zero = true;
//            for(int i = 0; i < choice_dist_map.rows(); ++i){
//                all_zero = all_zero && choice_dist_map(i) == 0;
//                total += choice_dist_map(i);
//            }
//            if(all_zero)
//                index =  dist_uni(_gen);
//            else{
//                double rand_nb = distrib(_gen);
//                for(int i = 0; i < choice_dist_map.rows(); ++i){
//                    cumul += choice_dist_map(i);
//                    if(rand_nb < cumul/total){
//                        index = i;
//                        break;
//                    }
//                }
//            }

            lbl = lbl_vct[index];
            sv = *(_soi.getSupervoxels()[lbl]);

            _choice_map.clear();
            for(int i = 0; i < choice_dist_map.rows(); i++)
                _choice_map.emplace(lbl_vct[i], choice_dist_map(i));


        }else if(_method == "mcs"){
            std::vector<uint32_t> lbl_vct;

            std::map<std::string,std::vector<std::pair<Eigen::VectorXd,std::vector<double>>>> samples;
            ip::SurfaceOfInterest::relevance_map_t weights = _soi.get_weights()[_modality];
            std::map<std::string, Eigen::VectorXd> sample;
            for(const auto& sv : _soi.getSupervoxels()){
                sample = _soi.get_features(sv.first);
                for(const auto& feature : sample){
                    samples[feature.first].push_back(
                                std::make_pair(feature.second,weights[sv.first]));
                }
                lbl_vct.push_back(sv.first);
            }
            int ind = _mcs.next_sample(samples,choice_dist_map);
            _choice_map.clear();
            for(int i = 0; i < choice_dist_map.rows(); i++)
                _choice_map.emplace(lbl_vct[i], choice_dist_map(i));

            lbl = lbl_vct[ind];
            sv = *(_soi.getSupervoxels()[lbl]);

        }
        else ROS_ERROR_STREAM(_method << " unknown relevance map method");

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

    pcl::PointCloud<pcl::PointXYZI> w_cl;

    //visualisation of relevance map
    for(auto& pub: _weighted_cloud_pub){
        for(int i = 0; i < _nbr_class; i++){
            w_cl = _soi.getColoredWeightedCloud(pub.first,i);
            pcl::toROSMsg(w_cl, weighted_cloud);
            weighted_cloud.header = _images_sub->get_depth().header;
            pub.second[i]->publish(weighted_cloud);
        }
    }

    //visualization of choice distribution map
    pcl::PointCloud<pcl::PointXYZI> choice_ptcl;
    for(const auto& val: _choice_map){
        pcl::Supervoxel<ip::PointT>::Ptr current_sv = _soi.getSupervoxels()[val.first];
        pcl::PointXYZI pt;
        for(auto v : *(current_sv->voxels_)){
            pt.x = v.x;
            pt.y = v.y;
            pt.z = v.z;
            pt.intensity = val.second;
            choice_ptcl.push_back(pt);
        }
    }

    sensor_msgs::PointCloud2 choice_cloud;
    pcl::toROSMsg(choice_ptcl, choice_cloud);
    choice_cloud.header = _images_sub->get_depth().header;
    _choice_dist_cloud_pub->publish(choice_cloud);
}
