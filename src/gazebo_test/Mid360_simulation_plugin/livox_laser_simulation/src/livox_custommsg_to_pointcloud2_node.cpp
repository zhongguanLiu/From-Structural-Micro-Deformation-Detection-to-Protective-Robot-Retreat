#include <cstdint>
#include <string>

#include <livox_ros_driver/CustomMsg.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>

class LivoxCustomMsgToPointCloud2
{
public:
  LivoxCustomMsgToPointCloud2() : privateNode_("~")
  {
    privateNode_.param<std::string>("input_topic", inputTopic_, "/livox/lidar");
    privateNode_.param<std::string>(
        "output_topic", outputTopic_, "/livox/pointcloud2");
    publisher_ = node_.advertise<sensor_msgs::PointCloud2>(outputTopic_, 2);
    subscriber_ = node_.subscribe(
        inputTopic_, 2, &LivoxCustomMsgToPointCloud2::Callback, this);
    ROS_INFO_STREAM("RViz cloud bridge: " << inputTopic_ << " -> "
                    << outputTopic_);
  }

private:
  void Callback(const livox_ros_driver::CustomMsg::ConstPtr &input)
  {
    sensor_msgs::PointCloud2 output;
    output.header = input->header;
    output.height = 1;
    output.is_dense = false;
    sensor_msgs::PointCloud2Modifier modifier(output);
    modifier.setPointCloud2Fields(
        7,
        "x", 1, sensor_msgs::PointField::FLOAT32,
        "y", 1, sensor_msgs::PointField::FLOAT32,
        "z", 1, sensor_msgs::PointField::FLOAT32,
        "intensity", 1, sensor_msgs::PointField::FLOAT32,
        "time", 1, sensor_msgs::PointField::FLOAT32,
        "ring", 1, sensor_msgs::PointField::UINT16,
        "tag", 1, sensor_msgs::PointField::UINT8);
    modifier.resize(input->points.size());

    sensor_msgs::PointCloud2Iterator<float> x(output, "x");
    sensor_msgs::PointCloud2Iterator<float> y(output, "y");
    sensor_msgs::PointCloud2Iterator<float> z(output, "z");
    sensor_msgs::PointCloud2Iterator<float> intensity(output, "intensity");
    sensor_msgs::PointCloud2Iterator<float> time(output, "time");
    sensor_msgs::PointCloud2Iterator<std::uint16_t> ring(output, "ring");
    sensor_msgs::PointCloud2Iterator<std::uint8_t> tag(output, "tag");
    for (const auto &point : input->points)
    {
      *x = point.x;
      *y = point.y;
      *z = point.z;
      *intensity = static_cast<float>(point.reflectivity);
      *time = static_cast<float>(point.offset_time) * 1.0e-9F;
      *ring = static_cast<std::uint16_t>(point.line);
      *tag = point.tag;
      ++x; ++y; ++z; ++intensity; ++time; ++ring; ++tag;
    }
    publisher_.publish(output);
  }

  ros::NodeHandle node_;
  ros::NodeHandle privateNode_;
  ros::Subscriber subscriber_;
  ros::Publisher publisher_;
  std::string inputTopic_;
  std::string outputTopic_;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "livox_custommsg_to_pointcloud2");
  LivoxCustomMsgToPointCloud2 bridge;
  ros::spin();
  return 0;
}
