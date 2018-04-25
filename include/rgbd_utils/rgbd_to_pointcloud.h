#ifndef _RGBD_TO_POINTCLOUD_H
#define _RGBD_TO_POINTCLOUD_H

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <sensor_msgs/image_encodings.h>
#include <image_geometry/pinhole_camera_model.h>
#include <Eigen/Geometry>
#include <eigen_conversions/eigen_msg.h>
#include <depth_image_proc/depth_traits.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>


using namespace message_filters::sync_policies;
namespace enc = sensor_msgs::image_encodings;
namespace rgbd_utils {


class RGBD_to_Pointcloud
{

public:
    /**
     * @brief default constructor
     */
    RGBD_to_Pointcloud(){
        _cloud_msg.reset(new sensor_msgs::PointCloud2);
    }

    /**
     * @brief construct the class and make the conversion
     * @param depth image msg
     * @param rgb image msg
     * @param rgb camera info msg
     */
    RGBD_to_Pointcloud(const sensor_msgs::ImageConstPtr& depth_msg,
                       const sensor_msgs::ImageConstPtr& rgb_msg,
                       const sensor_msgs::CameraInfoConstPtr& info_msg)
    {
        _cloud_msg.reset(new sensor_msgs::PointCloud2);
        _rgb_msg.reset(new sensor_msgs::Image(*rgb_msg));
        _depth_msg.reset(new sensor_msgs::Image(*depth_msg));
        _info_rgb.reset(new sensor_msgs::CameraInfo(*info_msg));

        convert();
    }



    ~RGBD_to_Pointcloud(){
        _cloud_msg.reset();
        _depth_msg.reset();
        _rgb_msg.reset();
    }

    /**
     * @brief make the conversion (is already called in the second constructor)
     */
    void convert();

//    void connectCb();

//    void imageCb(const sensor_msgs::ImageConstPtr& depth_msg,
//                 const sensor_msgs::ImageConstPtr& rgb_msg,
//                 const sensor_msgs::CameraInfoConstPtr& info_msg);


    //GETTERS & SETTERS
    void set_rgb(const sensor_msgs::Image& img){
        _rgb_msg.reset(new sensor_msgs::Image(img));
    }
    void set_depth(const sensor_msgs::Image& img){
        _depth_msg.reset(new sensor_msgs::Image(img));
    }
    void set_rgb_info(const sensor_msgs::CameraInfo& info){
        _info_rgb.reset(new sensor_msgs::CameraInfo(info));
    }

    const sensor_msgs::PointCloud2& get_pointcloud(){return *_cloud_msg;}
    const sensor_msgs::Image& get_rgb(){return *_rgb_msg;}
    const sensor_msgs::Image& get_depth(){return *_depth_msg;}

private:

    template<typename T>
    void _convert(const sensor_msgs::ImageConstPtr& depth_msg,
                 const sensor_msgs::ImageConstPtr& rgb_msg,
                 const sensor_msgs::PointCloud2Ptr& cloud_msg,
                 int red_offset = 0, int green_offset = 1, int blue_offset = 2, int color_step = 3);


//    ros::NodeHandlePtr rgb_nh_;
//    boost::shared_ptr<image_transport::ImageTransport> rgb_it_, depth_it_;

    // Subscriptions
//    image_transport::SubscriberFilter sub_depth_, sub_rgb_;
//    message_filters::Subscriber<sensor_msgs::CameraInfo> sub_info_;
//    typedef ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo> SyncPolicy;
//    typedef message_filters::Synchronizer<SyncPolicy> Synchronizer;
//    boost::shared_ptr<Synchronizer> sync_;

    // Publications
    boost::mutex connect_mutex_;
    ros::Publisher pub_point_cloud_;

    image_geometry::PinholeCameraModel model_;

    sensor_msgs::PointCloud2Ptr _cloud_msg;
    sensor_msgs::ImagePtr _rgb_msg;
    sensor_msgs::ImagePtr _depth_msg;
    sensor_msgs::CameraInfoPtr _info_rgb;

};

}




#endif //_RGBD_TO_POINTCLOUD_H
