#ifndef _RGBD_SUBSCRIBER_HPP
#define _RGBD_SUBSCRIBER_HPP

#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <memory>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

using namespace sensor_msgs;

namespace rgbd_utils {
    class RGBD_Subscriber {

    public:
        typedef std::shared_ptr<RGBD_Subscriber> Ptr;
        typedef const std::shared_ptr<RGBD_Subscriber> ConstPtr;

        typedef std::function<void(const ImageConstPtr& depth_msg, const ImageConstPtr& rgb_msg,
                                   const CameraInfoConstPtr& rgb_info_msg,
                                   const CameraInfoConstPtr& depth_info_msg)> image_cb_t;

        RGBD_Subscriber(const std::string& rgb_info, const std::string& rgb, const std::string& depth_info,
                        const std::string& depth, ros::NodeHandle& nodeHandle,
                        const image_cb_t callback = nullptr)
                : _rgb_info_topic(rgb_info), _rgb_topic(rgb), _depth_topic(depth),
                  _depth_info_topic(depth_info), _callback_complement(callback)
        {
            _rgb.reset(new Image);
            _depth.reset(new Image);
            _rgb_info.reset(new CameraInfo);
            _depth_info.reset(new CameraInfo);

            rgb_nh_.reset(new ros::NodeHandle(nodeHandle, "rgb"));
            depth_nh_.reset(new ros::NodeHandle(nodeHandle, "depth"));

            init();
            connect_callback();
        }

        ~RGBD_Subscriber()
        {
            _rgb_info.reset();
            _rgb.reset();
            _depth.reset();
            _depth_info.reset();
            rgb_it_.reset();
            depth_it_.reset();
            depth_nh_.reset();
            rgb_nh_.reset();
            sync_.reset();
        }

        void init();

        void image_callback(const ImageConstPtr& depth_msg, const ImageConstPtr& rgb_msg,
                            const CameraInfoConstPtr& rgb_info_msg, const CameraInfoConstPtr& depth_info_msg)
        {
            _rgb.reset(new Image(*rgb_msg));
            _depth.reset(new Image(*depth_msg));
            _rgb_info.reset(new CameraInfo(*rgb_info_msg));
            _depth_info.reset(new CameraInfo(*depth_info_msg));

            if (_callback_complement != nullptr) {
                _callback_complement(depth_msg, rgb_msg, rgb_info_msg, depth_info_msg);
            }
        }

        void connect_callback();

        const Image get_depth()
        { return *_depth; }

        const Image& get_rgb()
        { return *_rgb; }

        const CameraInfo& get_rgb_info()
        { return *_rgb_info; }

        const CameraInfo& get_depth_info()
        { return *_depth_info; }

        const ImagePtr get_depthConstPtr()
        { return _depth; }

        const ImagePtr get_rgbConstPtr()
        { return _rgb; }

        const CameraInfoPtr get_rgb_infoConstPtr()
        { return _rgb_info; }

        const CameraInfoPtr get_depth_infoConstPtr()
        { return _depth_info; }


    private:
        std::string _rgb_info_topic, _rgb_topic, _depth_topic, _depth_info_topic;
        std::function<void(const ImageConstPtr& depth_msg, const ImageConstPtr& rgb_msg,
                           const CameraInfoConstPtr& rgb_info_msg,
                           const CameraInfoConstPtr& depth_info_msg)> _callback_complement;

        ros::NodeHandlePtr rgb_nh_, depth_nh_;
        image_transport::SubscriberFilter sub_depth_, sub_rgb_;
        message_filters::Subscriber<CameraInfo> sub_rgb_info_, sub_depth_info_;


        boost::shared_ptr<image_transport::ImageTransport> rgb_it_, depth_it_;

        ImagePtr _rgb, _depth;
        CameraInfoPtr _rgb_info, _depth_info;

        using SyncPolicy = message_filters::sync_policies::ApproximateTime<Image,
                Image, CameraInfo, CameraInfo>;

        boost::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

        bool _is_init = false;
    };
}
#endif //_RGBD_SUBSCRIBER_HPP
