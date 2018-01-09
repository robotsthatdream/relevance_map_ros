#ifndef _UTILITIES_HPP
#define _UTILITIES_HPP

#include <memory>
#include <iostream>
#include <fstream>

#include <ros/ros.h>

#include <XmlRpcValue.h>
#include <yaml-cpp/yaml.h>

#include <image_processing/pcl_types.h>
#include <image_processing/SupervoxelSet.h>

#include <iagmm/data.hpp>
#include <iagmm/gmm.hpp>
#include <iagmm/nnmap.hpp>
#include <iagmm/mcs.hpp>


#include <gazebo_msgs/SpawnModel.h>
#include <gazebo_msgs/DeleteModel.h>
#include <gazebo_msgs/GetModelState.h>
#include <gazebo_msgs/SetModelState.h>

#include <boost/random.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <pcl/kdtree/kdtree_flann.h>

namespace ip = image_processing;

namespace relevance_map{

typedef struct setup_param_t{
    double x_max;
    double x_min;
    double y_max;
    double y_min;
    double z_max;
    double z_min;
}setup_param_t;

typedef struct model_t{
    std::string desc;
    std::array<double,3> position;
}model_t;

typedef std::array<double,10> result_array_t;

void define_frames(const std::string& robot, std::string& base_frame, std::string& camera_frame){
    if(robot == "baxter" || robot == "crustcrawler"){
        base_frame = "/base";
        camera_frame = "/camera_depth_optical_frame";
    }else if(robot == "pr2"){
        base_frame = "/odom_combined";
        camera_frame = "/head_mount_kinect2_rgb_optical_frame";
    }
}

typedef std::map<std::string,model_t> model_map_t;

std::array<double,3> load_start_models_pose(const std::string& file_name){
    YAML::Node yml_file = YAML::LoadFile(file_name);
    std::array<double,3> position;

    position[0] = yml_file["start_pose"]["position"][0].as<double>();
    position[1] = yml_file["start_pose"]["position"][1].as<double>();
    position[2] = yml_file["start_pose"]["position"][2].as<double>();
    return position;
}

int load_results(std::string file_name, int &fp, int &fn, int &nbr_iteration){
    std::cout << "load results : " << file_name << std::endl;

    YAML::Node file_node = YAML::LoadFile(file_name);
    if(file_node.IsNull())
        return 0;

    std::stringstream sstream;
    sstream << "iteration_" << file_node.size()  - 2;
    std::cout << "load for " << sstream.str() << std::endl;
    fp = file_node[sstream.str()]["false_positives"].as<int>();
    fn = file_node[sstream.str()]["false_negatives"].as<int>();
    nbr_iteration = file_node.size()-1;
    return 1;
}

bool load_models(XmlRpc::XmlRpcValue &params,
                 std::map<std::string,std::string>& sdf_models,
                 setup_param_t& setup_params){
    std::ifstream ifs;
    std::string file;
    std::string folder = static_cast<std::string>(params["models_folder"]);
    std::cout << "load models : " << folder << std::endl;

    XmlRpc::XmlRpcValue models = params["models"];
    for(int i = 0; i < models.size(); i++){
        file = folder + static_cast<std::string>(models[i]) + ".sdf";
        ROS_ERROR_STREAM(file);

        ifs.open(file);
        if(!ifs){
            ROS_ERROR_STREAM("unable to open : " << file);
            return false;
        }

        std::string buff,sdf;

        while(std::getline(ifs,buff))
            sdf += buff;
        ifs.close();
        sdf_models.emplace(static_cast<std::string>(models[i]),sdf);
    }

    setup_params.x_max = params["environments"]["positions"]["x_max"];
    setup_params.x_min = params["environments"]["positions"]["x_min"];
    setup_params.y_max = params["environments"]["positions"]["y_max"];
    setup_params.y_min = params["environments"]["positions"]["y_min"];
    setup_params.z_max = params["environments"]["positions"]["z_max"];
    setup_params.z_min = params["environments"]["positions"]["z_min"];

    return true;
}

iagmm::TrainingData load_dataset(const std::string& filename){
    std::cout << "load dataset : " << filename << std::endl;

    iagmm::TrainingData dataset;

    YAML::Node fileNode = YAML::LoadFile(filename);
    if (fileNode.IsNull()) {
        ROS_ERROR("File not found.");
        return dataset;
    }

    YAML::Node features = fileNode["frame_0"]["features"];


    for (unsigned int i = 0; i < features.size(); ++i) {
        std::stringstream stream;
        stream << "feature_" << i;
        YAML::Node tmp_node = features[stream.str()];

        Eigen::VectorXd feature(tmp_node["value"].size());
        for(size_t i = 0; i < tmp_node["value"].size(); ++i)
            feature(i) = tmp_node["value"][i].as<double>();


        dataset.add(tmp_node["label"].as<int>(),feature);
    }
    return dataset;
}

/**
* @brief load an archive of a previous experiment
* @return true if something was loaded false otherwise
*/
bool load_experiment(const std::string& soi_method, const std::string &folder,
                const std::map<std::string,int>& modalities,
                std::map<std::string,iagmm::GMM> &gmm_class,
                std::map<std::string,iagmm::NNMap> &nnmap_class,
                iagmm::MCS &mcs){
    std::cout << "load experiment : " << folder << std::endl;
    if(folder.empty())
        return false;
    boost::filesystem::directory_iterator dir_it(folder);
    boost::filesystem::directory_iterator end_it;
    std::vector<std::string> split_str;
    std::string type;
    std::map<std::string,std::string> gmm_arch_file;
    std::map<std::string,std::string> dataset_file;

    for(;dir_it != end_it; ++dir_it){
        boost::split(split_str,dir_it->path().string(),boost::is_any_of("/"));
        boost::split(split_str,split_str.back(),boost::is_any_of("_"));
        type = split_str[0];
        boost::split(split_str,split_str.back(),boost::is_any_of("."));

        for(const auto& mod : modalities){
            if(split_str[0] == mod.first)
            {
                if(type == "gmm" && (soi_method == "gmm" || soi_method == "mcs"))
                    gmm_arch_file.emplace(split_str[0],dir_it->path().string());
                if(type == "dataset")
                    dataset_file.emplace(split_str[0],dir_it->path().string());
            }
        }
    }

    if(soi_method == "gmm"){
        for(const auto& arch: gmm_arch_file){
            iagmm::GMM gmm;
            std::ifstream ifs(arch.second);
            boost::archive::text_iarchive iarch(ifs);
            iarch >> gmm;
            gmm_class.emplace(arch.first,gmm);
        }
    }
    else if(soi_method == "mcs"){
        std::map<std::string,iagmm::GMM::Ptr> gmms;
        for(const auto& arch: gmm_arch_file){
            iagmm::GMM gmm;
            std::ifstream ifs(arch.second);
            boost::archive::text_iarchive iarch(ifs);
            iarch >> gmm;
            gmms.emplace(arch.first,iagmm::GMM::Ptr(new iagmm::GMM(gmm)));
        }
        mcs = iagmm::MCS(gmms,iagmm::combinatorial::fct_map.at("sum"),iagmm::param_estimation::fct_map.at("linear"));
    }

    for(const auto& file : dataset_file){
        iagmm::TrainingData data = load_dataset(file.second);
        if(soi_method == "gmm")
            gmm_class[file.first].set_samples(data);
        else if(soi_method == "nnmap")
            nnmap_class[file.first].set_samples(data);
        else if(soi_method == "mcs")
            mcs.set_samples(file.first,data);
    }
    return true;
}

bool load_archive(std::string folder_name, std::queue<std::string> &folder_list){
    std::cout << "load archive : " << folder_name << std::endl;
    if(folder_name.empty())
        return false;

    if(!boost::filesystem::exists(folder_name))
        return false;

    boost::filesystem::directory_iterator dir_it(folder_name);
    boost::filesystem::directory_iterator end_it;

    for(;dir_it != end_it; ++dir_it){
        if(!boost::filesystem::is_directory(dir_it->path().string()))
            continue;

        folder_list.push(dir_it->path().string());
    }

    if(folder_list.empty())
        return false;

    return true;
}

Eigen::Vector3d get_model_position(const std::string& model_name,std::unique_ptr<ros::ServiceClient> &client){
    gazebo_msgs::GetModelState msg;
    Eigen::Vector3d position;
    position << 0, 0, 0;
    msg.request.model_name = model_name;
//    ros::ServiceClient client = ros_nh->serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state");

    if(client->call(msg)){
        if(msg.response.success){
            position(0) = msg.response.pose.position.x;
            position(1) = msg.response.pose.position.y;
            position(2) = msg.response.pose.position.z;
        }else ROS_ERROR_STREAM("Unable to get state of model of name " << model_name);
    }else ROS_ERROR_STREAM("Unable to call get_model_state service");
    return position;
}


bool change_models_state(const model_map_t &sdf_models,
                         const setup_param_t &setup_params,
                         std::unique_ptr<ros::ServiceClient> &client,
                         boost::random::mt19937& gen, bool with_orientation = false){
    gazebo_msgs::SetModelState model_state_msg;

    boost::random::uniform_real_distribution<> dist_x(setup_params.x_min,setup_params.x_max);
    boost::random::uniform_real_distribution<> dist_y(setup_params.y_min,setup_params.y_max);
    boost::random::uniform_real_distribution<> dist_z(setup_params.z_min,setup_params.z_max);
    boost::random::uniform_real_distribution<> roll(-3.14,3.14);
    boost::random::uniform_real_distribution<> pitch(-3.14,3.14);
    boost::random::uniform_real_distribution<> yaw(-3.14,3.14);

    for(const auto& model: sdf_models){
        float r = roll(gen),p = pitch(gen),y = yaw(gen);
        model_state_msg.request.model_state.model_name = model.first;

        if(setup_params.x_min == setup_params.x_max)
            model_state_msg.request.model_state.pose.position.x = setup_params.x_min;
        else model_state_msg.request.model_state.pose.position.x = dist_x(gen);
        if(setup_params.y_min == setup_params.y_max)
            model_state_msg.request.model_state.pose.position.y = setup_params.y_min;
        else model_state_msg.request.model_state.pose.position.y = dist_y(gen);
        if(setup_params.z_min == setup_params.z_max)
            model_state_msg.request.model_state.pose.position.z = setup_params.z_min;
        else model_state_msg.request.model_state.pose.position.z = dist_z(gen);

        if(with_orientation)
        {
            model_state_msg.request.model_state.pose.orientation.x = cos(y)*cos(r)*cos(p) + sin(y)*sin(r)*sin(p);
            model_state_msg.request.model_state.pose.orientation.y = cos(y)*sin(r)*cos(p) - sin(y)*cos(r)*sin(p);
            model_state_msg.request.model_state.pose.orientation.z = cos(y)*cos(r)*sin(p) + sin(y)*sin(r)*cos(p);
            model_state_msg.request.model_state.pose.orientation.w = sin(y)*cos(r)*cos(p) - cos(y)*sin(r)*sin(p);
        }
        model_state_msg.request.model_state.reference_frame = "world";

        if(client->call(model_state_msg)){
            if(model_state_msg.response.success)
                ROS_INFO_STREAM(model.first << " state modified");
            else ROS_INFO_STREAM("fail to modify state of " << model.first);
        }else{
            ROS_ERROR_STREAM("unbale to call service set_model_state");
            return false;
        }
    }
    return true;
}

bool change_model_state(const std::string& model_name,
                         const setup_param_t &setup_params,
                         std::unique_ptr<ros::ServiceClient> &client,
                         boost::random::mt19937& gen, bool with_orientation = false){
    gazebo_msgs::SetModelState model_state_msg;

    boost::random::uniform_real_distribution<> dist_x(setup_params.x_min,setup_params.x_max);
    boost::random::uniform_real_distribution<> dist_y(setup_params.y_min,setup_params.y_max);
    boost::random::uniform_real_distribution<> dist_z(setup_params.z_min,setup_params.z_max);
    boost::random::uniform_real_distribution<> roll(-3.14,3.14);
    boost::random::uniform_real_distribution<> pitch(-3.14,3.14);
    boost::random::uniform_real_distribution<> yaw(-3.14,3.14);

    float r = roll(gen),p = pitch(gen),y = yaw(gen);
    model_state_msg.request.model_state.model_name = model_name;

    if(setup_params.x_min == setup_params.x_max)
        model_state_msg.request.model_state.pose.position.x = setup_params.x_min;
    else model_state_msg.request.model_state.pose.position.x = dist_x(gen);
    if(setup_params.y_min == setup_params.y_max)
        model_state_msg.request.model_state.pose.position.y = setup_params.y_min;
    else model_state_msg.request.model_state.pose.position.y = dist_y(gen);
    if(setup_params.z_min == setup_params.z_max)
        model_state_msg.request.model_state.pose.position.z = setup_params.z_min;
    else model_state_msg.request.model_state.pose.position.z = dist_z(gen);

    if(with_orientation)
    {
        model_state_msg.request.model_state.pose.orientation.x = cos(y)*cos(r)*cos(p) + sin(y)*sin(r)*sin(p);
        model_state_msg.request.model_state.pose.orientation.y = cos(y)*sin(r)*cos(p) - sin(y)*cos(r)*sin(p);
        model_state_msg.request.model_state.pose.orientation.z = cos(y)*cos(r)*sin(p) + sin(y)*sin(r)*cos(p);
        model_state_msg.request.model_state.pose.orientation.w = sin(y)*cos(r)*cos(p) - cos(y)*sin(r)*sin(p);
    }
    model_state_msg.request.model_state.reference_frame = "world";

    if(client->call(model_state_msg)){
        if(model_state_msg.response.success)
            ROS_INFO_STREAM(model_name << " state modified");
        else ROS_INFO_STREAM("fail to modify state of " << model_name);
    }else{
        ROS_ERROR_STREAM("unbale to call service set_model_state");
        return false;
    }

    return true;
}

bool spawn_models(const std::map<std::string,std::string> &sdf_models,
                  const setup_param_t &setup_params,
                  std::unique_ptr<ros::ServiceClient> &client,
                  boost::random::mt19937& gen){
    gazebo_msgs::SpawnModel msg;

    boost::random::uniform_real_distribution<> dist_x(setup_params.x_min,setup_params.x_max);
    boost::random::uniform_real_distribution<> dist_y(setup_params.y_min,setup_params.y_max);
    boost::random::uniform_real_distribution<> dist_z(setup_params.z_min,setup_params.z_max);

    for(const auto& model: sdf_models){
        msg.request.model_name = model.first;
        msg.request.model_xml = model.second;
        if(setup_params.x_min == setup_params.x_max)
            msg.request.initial_pose.position.x = setup_params.x_min;
        else msg.request.initial_pose.position.x = dist_x(gen);
        if(setup_params.y_min == setup_params.y_max)
            msg.request.initial_pose.position.y = setup_params.y_min;
        else msg.request.initial_pose.position.y = dist_y(gen);
        if(setup_params.z_min == setup_params.z_max)
            msg.request.initial_pose.position.z = setup_params.z_min;
        else msg.request.initial_pose.position.z = dist_z(gen);

        ROS_INFO_STREAM(model.first << " : " << msg.request.initial_pose.position.x << " "
                        << msg.request.initial_pose.position.y << " "
                        << msg.request.initial_pose.position.z);

        if(client->call(msg)){
            if(msg.response.success)
                ROS_INFO_STREAM(model.first << " spawned");
            else
                ROS_ERROR_STREAM("fail to spawn " << model.first);
        }
        else{
            ROS_ERROR_STREAM("unbale to call service spawn_sdf_model");
            return false;
        }
    }
    return true;

}

bool spawn_models(const model_map_t &sdf_models,
                  std::unique_ptr<ros::ServiceClient> &client){
    gazebo_msgs::SpawnModel msg;



    for(const auto& model: sdf_models){
        msg.request.model_name = model.first;
        msg.request.model_xml = model.second.desc;

        msg.request.initial_pose.position.x = model.second.position[0];
        msg.request.initial_pose.position.y = model.second.position[1];
        msg.request.initial_pose.position.z = model.second.position[2];


        if(client->call(msg)){
            if(msg.response.success)
                ROS_INFO_STREAM(model.first << " spawned");
            else
                ROS_ERROR_STREAM("fail to spawn " << model.first);
        }
        else{
            ROS_ERROR_STREAM("unbale to call service spawn_sdf_model");
            return false;
        }
    }
    return true;
}

bool delete_models(const std::map<std::string, std::string> &sdf_models,
                   std::unique_ptr<ros::ServiceClient> &client){
    gazebo_msgs::DeleteModel msg;
    for(const auto& model: sdf_models){
        msg.request.model_name = model.first;

        if(client->call(msg)){
            if(msg.response.success)
                ROS_INFO_STREAM(model.first << " successfuly deleted");
            else
                ROS_ERROR_STREAM("fail to delete " << model.first);
        }
        else{
            ROS_ERROR_STREAM("unable to call delete_model service");
            return false;

        }
    }
    return true;
}

bool is_in_cloud(const image_processing::PointT &pt, const ip::PointCloudT::Ptr cloud){
    pcl::KdTreeFLANN<ip::PointT>::Ptr tree(new pcl::KdTreeFLANN<ip::PointT>);
    tree->setInputCloud(cloud);

    std::vector<int> nn_indices(1);
    std::vector<float> nn_distance(1);

    if(!tree->nearestKSearch(pt,1,nn_indices,nn_distance))
        return true;

    if(nn_distance[0] < 0.0001)
        return true;
    else return false;
}

Eigen::VectorXd noise(const Eigen::VectorXd& v,double std_dev,boost::random::mt19937& gen){
    Eigen::VectorXd v_noise(v.rows());

    for(int i = 0; i < v.rows(); ++i){
        boost::random::normal_distribution<> dist(v(i),std_dev);
        v_noise(i) = dist(gen);
    }
    return v_noise;
}

void init_workspace(XmlRpc::XmlRpcValue &wks, std::unique_ptr<ip::workspace_t>& workspace){

    workspace.reset(
                new ip::workspace_t(true,
                                    static_cast<double>(wks["sphere"]["x"]),
            static_cast<double> (wks["sphere"]["y"]),
            static_cast<double>(wks["sphere"]["z"]),
            static_cast<double> (wks["sphere"]["radius"]),
            static_cast<double> (wks["sphere"]["threshold"]),
    {static_cast<double>(wks["csg_intersect_cuboid"]["x_min"]),
                static_cast<double>(wks["csg_intersect_cuboid"]["x_max"]),
                static_cast<double> (wks["csg_intersect_cuboid"]["y_min"]),
                static_cast<double>(wks["csg_intersect_cuboid"]["y_max"]),
                static_cast<double> (wks["csg_intersect_cuboid"]["z_min"]),
                static_cast<double>(wks["csg_intersect_cuboid"]["z_max"])}));
}

}

#endif
