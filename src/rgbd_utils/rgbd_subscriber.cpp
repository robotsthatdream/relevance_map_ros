#include <rgbd_utils/rgbd_subscriber.hpp>


using namespace rgbd_utils;


void RGBD_Subscriber::init()
{
    int queue_size;
    ros::NodeHandle private_nh;

//  ros::NodeHandle depth_nh(nh, "depth_registered");
    rgb_it_.reset(new image_transport::ImageTransport(*rgb_nh_));
    depth_it_.reset(new image_transport::ImageTransport(*depth_nh_));

    // Read parameters
    private_nh.param("queue_size", queue_size, 5);

    // Synchronize inputs. Topic subscriptions happen on demand in the connection callback.
    sync_.reset(
            new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(queue_size), sub_depth_, sub_rgb_, sub_rgb_info_,
                                                          sub_depth_info_));
    sync_->registerCallback(boost::bind(&RGBD_Subscriber::image_callback, this, _1, _2, _3, _4));

    _is_init = true;
}

void RGBD_Subscriber::connect_callback()
{
    ros::NodeHandle private_nh;
    // parameter for depth_image_transport hint
    std::string depth_image_transport_param = "depth_image_transport";

    // depth image can use different transport.(e.g. compressedDepth)
    image_transport::TransportHints depth_hints("raw", ros::TransportHints(), private_nh, depth_image_transport_param);
    sub_depth_.subscribe(*depth_it_, _depth_topic, 1, depth_hints);

    // rgb uses normal ros transport hints.
    image_transport::TransportHints hints("raw", ros::TransportHints(), private_nh);
    sub_rgb_.subscribe(*rgb_it_, _rgb_topic, 1, hints);
    sub_rgb_info_.subscribe(*rgb_nh_, _rgb_info_topic, 1);
    sub_depth_info_.subscribe(*depth_nh_, _depth_info_topic, 1);
}
