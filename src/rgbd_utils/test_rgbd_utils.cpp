#include "../../include/rgbd_utils/rgbd_subscriber.hpp"
#include "../../include/rgbd_utils/rgbd_to_pointcloud.h"

int main(int argc, char** argv){

    ros::init(argc,argv,"test_rgbd_utils");
    ros::NodeHandle nh;

    XmlRpc::XmlRpcValue glob_params;
    nh.getParam("/global",glob_params);


    rgbd_utils::RGBD_Subscriber rgbd_sub(glob_params["rgb_info_topic"],
            glob_params["rgb_topic"],
            glob_params["depth_info_topic"],
            glob_params["depth_topic"],
            nh);
    rgbd_utils::RGBD_to_Pointcloud converter;

    ros::Publisher cloud_pub(nh.advertise<sensor_msgs::PointCloud2>("cloud",5));

    while(ros::ok){
        ros::spinOnce();
        usleep(1000);
        if(rgbd_sub.get_depth().data.empty() || rgbd_sub.get_rgb().data.empty())
            continue;
        converter.set_depth(rgbd_sub.get_depth());
        converter.set_rgb(rgbd_sub.get_rgb());
        converter.set_rgb_info(rgbd_sub.get_rgb_info());
        converter.convert();
        cloud_pub.publish(converter.get_pointcloud());
    }

    return 0;
}
