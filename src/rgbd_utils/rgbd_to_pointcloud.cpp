#include <rgbd_utils/rgbd_to_pointcloud.h>
#include <sensor_msgs/point_cloud2_iterator.h>

using namespace rgbd_utils;

void RGBD_to_Pointcloud::convert()
{

  // Update camera model
  model_.fromCameraInfo(_info_rgb);


  // Check if the input image has to be resized
  sensor_msgs::ImageConstPtr rgb_msg = _rgb_msg;
  if (_depth_msg->width != rgb_msg->width || _depth_msg->height != rgb_msg->height)
  {
    sensor_msgs::CameraInfo info_msg_tmp = *_info_rgb;
    info_msg_tmp.width = _depth_msg->width;
    info_msg_tmp.height = _depth_msg->height;
    float ratio = float(_depth_msg->width)/float(rgb_msg->width);
    info_msg_tmp.K[0] *= ratio;
    info_msg_tmp.K[2] *= ratio;
    info_msg_tmp.K[4] *= ratio;
    info_msg_tmp.K[5] *= ratio;
    info_msg_tmp.P[0] *= ratio;
    info_msg_tmp.P[2] *= ratio;
    info_msg_tmp.P[5] *= ratio;
    info_msg_tmp.P[6] *= ratio;
    model_.fromCameraInfo(info_msg_tmp);

    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvShare(rgb_msg, rgb_msg->encoding);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    cv_bridge::CvImage cv_rsz;
    cv_rsz.header = cv_ptr->header;
    cv_rsz.encoding = cv_ptr->encoding;
    cv::resize(cv_ptr->image.rowRange(0,_depth_msg->height/ratio), cv_rsz.image, cv::Size(_depth_msg->width, _depth_msg->height));
    if ((rgb_msg->encoding == enc::RGB8) || (rgb_msg->encoding == enc::BGR8) || (rgb_msg->encoding == enc::MONO8))
      rgb_msg = cv_rsz.toImageMsg();
    else
      rgb_msg = cv_bridge::toCvCopy(cv_rsz.toImageMsg(), enc::RGB8)->toImageMsg();

    //NODELET_ERROR_THROTTLE(5, "Depth resolution (%ux%u) does not match RGB resolution (%ux%u)",
    //                       depth_msg->width, depth_msg->height, rgb_msg->width, rgb_msg->height);
    //return;
  } else
    rgb_msg = _rgb_msg;


  // Supported color encodings: RGB8, BGR8, MONO8
  int red_offset, green_offset, blue_offset, color_step;
  if (rgb_msg->encoding == enc::RGB8)
  {
    red_offset   = 0;
    green_offset = 1;
    blue_offset  = 2;
    color_step   = 3;
  }
  else if (rgb_msg->encoding == enc::BGR8)
  {
    red_offset   = 2;
    green_offset = 1;
    blue_offset  = 0;
    color_step   = 3;
  }
  else if (rgb_msg->encoding == enc::MONO8)
  {
    red_offset   = 0;
    green_offset = 0;
    blue_offset  = 0;
    color_step   = 1;
  }
  else
  {
    try
    {
      rgb_msg = cv_bridge::toCvCopy(rgb_msg, enc::RGB8)->toImageMsg();
    }
    catch (cv_bridge::Exception& e)
    {
      std::cerr << "Unsupported encoding" << rgb_msg->encoding.c_str() << " : " << e.what() << std::endl;
      return;
    }
    red_offset   = 0;
    green_offset = 1;
    blue_offset  = 2;
    color_step   = 3;
  }

  // Allocate new point cloud message
  _cloud_msg->header = _depth_msg->header; // Use depth image time stamp
  _cloud_msg->height = _depth_msg->height;
  _cloud_msg->width  = _depth_msg->width;
  _cloud_msg->is_dense = false;
  _cloud_msg->is_bigendian = false;

  sensor_msgs::PointCloud2Modifier pcd_modifier(*_cloud_msg);
  pcd_modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");

  if (_depth_msg->encoding == enc::TYPE_16UC1)
  {
    _convert<uint16_t>(_depth_msg, rgb_msg, _cloud_msg, red_offset, green_offset, blue_offset, color_step);
  }
  else if (_depth_msg->encoding == enc::TYPE_32FC1)
  {
    _convert<float>(_depth_msg, rgb_msg, _cloud_msg, red_offset, green_offset, blue_offset, color_step);
  }
  else
  {
    std::cerr <<  "Depth image has unsupported encoding " << _depth_msg->encoding.c_str() << std::endl;
    return;
  }

}

template<typename T>
void RGBD_to_Pointcloud::_convert(const sensor_msgs::ImageConstPtr& depth_msg,
                                      const sensor_msgs::ImageConstPtr& rgb_msg,
                                      const sensor_msgs::PointCloud2Ptr &cloud_msg,
                                      int red_offset, int green_offset, int blue_offset, int color_step)
{
  // Use correct principal point from calibration
  float center_x = model_.cx();
  float center_y = model_.cy();

  // Combine unit conversion (if necessary) with scaling by focal length for computing (X,Y)
  double unit_scaling = depth_image_proc::DepthTraits<T>::toMeters( T(1) );
  float constant_x = unit_scaling / model_.fx();
  float constant_y = unit_scaling / model_.fy();
  float bad_point = std::numeric_limits<float>::quiet_NaN ();

  const T* depth_row = reinterpret_cast<const T*>(&depth_msg->data[0]);
  int row_step = depth_msg->step / sizeof(T);
  const uint8_t* rgb = &rgb_msg->data[0];
  int rgb_skip = rgb_msg->step - rgb_msg->width * color_step;

  sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud_msg, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud_msg, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud_msg, "z");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(*cloud_msg, "r");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(*cloud_msg, "g");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(*cloud_msg, "b");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_a(*cloud_msg, "a");

  for (int v = 0; v < int(cloud_msg->height); ++v, depth_row += row_step, rgb += rgb_skip)
  {
    for (int u = 0; u < int(cloud_msg->width); ++u, rgb += color_step, ++iter_x, ++iter_y, ++iter_z, ++iter_a, ++iter_r, ++iter_g, ++iter_b)
    {
      T depth = depth_row[u];

      // Check for invalid measurements
      if (!depth_image_proc::DepthTraits<T>::valid(depth))
      {
        *iter_x = *iter_y = *iter_z = bad_point;
      }
      else
      {
        // Fill in XYZ
         if(_with_noise) //Noise would work only for float encoding
             depth = _noise<T>(depth,_std_dev);
        *iter_x = (u - center_x) * depth * constant_x;
        *iter_y = (v - center_y) * depth * constant_y;
        *iter_z = depth_image_proc::DepthTraits<T>::toMeters(depth);
      }

      // Fill in color
      *iter_a = 255;
      *iter_r = rgb[red_offset];
      *iter_g = rgb[green_offset];
      *iter_b = rgb[blue_offset];
    }
  }
}

