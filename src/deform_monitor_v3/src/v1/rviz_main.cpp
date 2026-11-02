#include "deform_monitor_v3/v1/rviz_adapter.hpp"

#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <std_msgs/String.h>

#include <string>
#include <cmath>
#include <stdexcept>

namespace deform_monitor_v3 {
namespace v1 {

class V1RvizNode {
 public:
  V1RvizNode() : private_node_("~") {
    private_node_.param("displacement_display_scale",
                        displacement_display_scale_, 50.0);
    private_node_.param("minimum_display_arrow_m",
                        minimum_display_arrow_m_, 0.005);
    private_node_.param("maximum_display_arrow_m",
                        maximum_display_arrow_m_, 1.0);
    private_node_.param("minimum_arrow_active_probability",
                        minimum_arrow_active_probability_, 0.0);
    if (!std::isfinite(minimum_arrow_active_probability_) ||
        minimum_arrow_active_probability_ < 0.0 ||
        minimum_arrow_active_probability_ > 1.0)
      throw std::invalid_argument("minimum_arrow_active_probability must be in [0,1]");
    if (minimum_arrow_active_probability_ > 0.0)
      ROS_WARN("[deform_v1] RViz arrows are filtered estimates (A+C lower bound >= %.2f), "
               "not calibrated deformation confirmations; raw detection_tiles are unchanged",
               minimum_arrow_active_probability_);
    private_node_.param("status_text_height_m", status_text_height_m_, 0.28);
    private_node_.param("status_position_x", status_position_.x, -4.0);
    private_node_.param("status_position_y", status_position_.y, -4.0);
    private_node_.param("status_position_z", status_position_.z, 3.0);
    private_node_.param("status_follow_odometry", status_follow_odometry_, true);
    private_node_.param("status_offset_x", status_offset_.x, -1.0);
    private_node_.param("status_offset_y", status_offset_.y, -1.0);
    private_node_.param("status_offset_z", status_offset_.z, 2.5);
    private_node_.param<std::string>(
        "status_odometry_topic", status_odometry_topic_,
        "/navigation/odometry");
    private_node_.param<std::string>(
        "status_detail_topic", status_detail_topic_, "");

    deformation_publisher_ = node_.advertise<visualization_msgs::MarkerArray>(
        "/deform/v1/rviz/deformation_markers", 1, true);
    status_publisher_ = node_.advertise<visualization_msgs::MarkerArray>(
        "/deform/v1/rviz/status_markers", 1, true);
    tile_subscriber_ = node_.subscribe(
        "/deform/v1/detection_tiles", 1, &V1RvizNode::TileCallback, this);
    status_subscriber_ = node_.subscribe(
        "/deform/v1/status", 1, &V1RvizNode::StatusCallback, this);
    if (status_follow_odometry_)
      odometry_subscriber_ = node_.subscribe(
          status_odometry_topic_, 1, &V1RvizNode::OdometryCallback, this);
    if (!status_detail_topic_.empty())
      detail_subscriber_ = node_.subscribe(
          status_detail_topic_, 1, &V1RvizNode::DetailCallback, this);
  }

 private:
  void TileCallback(
      const deform_monitor_v3::V1DetectionTile::ConstPtr& tile) {
    // LOD 2 is the complete local product and is published last for every
    // snapshot.  Rendering only this tile prevents three partial redraws.
    if (!tile || tile->lod != 2U) return;
    deformation_publisher_.publish(BuildV1DeformationMarkers(
        *tile, displacement_display_scale_, minimum_display_arrow_m_,
        maximum_display_arrow_m_, minimum_arrow_active_probability_));
  }

  void StatusCallback(const deform_monitor_v3::V1DetectionStatus::ConstPtr& status) {
    if (!status) return;
    latest_status_ = *status;
    latest_status_available_ = true;
    PublishStatus();
  }

  void DetailCallback(const std_msgs::String::ConstPtr& detail) {
    if (!detail) return;
    status_detail_ = detail->data;
    PublishStatus();
  }

  void PublishStatus() {
    if (!latest_status_available_) return;
    geometry_msgs::Point display_position = status_position_;
    if (status_follow_odometry_ && latest_odometry_available_ &&
        latest_odometry_frame_ == latest_status_.common.header.frame_id) {
      display_position = latest_odometry_position_;
    }
    status_publisher_.publish(BuildV1StatusMarkers(
        latest_status_, display_position, status_text_height_m_,
        displacement_display_scale_, status_detail_));
  }

  void OdometryCallback(const nav_msgs::Odometry::ConstPtr& odometry) {
    if (!odometry || odometry->header.frame_id.empty()) return;
    latest_odometry_position_.x =
        odometry->pose.pose.position.x + status_offset_.x;
    latest_odometry_position_.y =
        odometry->pose.pose.position.y + status_offset_.y;
    latest_odometry_position_.z =
        odometry->pose.pose.position.z + status_offset_.z;
    latest_odometry_frame_ = odometry->header.frame_id;
    latest_odometry_available_ = true;
  }

  ros::NodeHandle node_;
  ros::NodeHandle private_node_;
  ros::Subscriber tile_subscriber_;
  ros::Subscriber status_subscriber_;
  ros::Subscriber odometry_subscriber_;
  ros::Subscriber detail_subscriber_;
  ros::Publisher deformation_publisher_;
  ros::Publisher status_publisher_;
  double displacement_display_scale_ = 50.0;
  double minimum_display_arrow_m_ = 0.005;
  double maximum_display_arrow_m_ = 1.0;
  double minimum_arrow_active_probability_ = 0.0;
  double status_text_height_m_ = 0.28;
  geometry_msgs::Point status_position_;
  geometry_msgs::Point status_offset_;
  geometry_msgs::Point latest_odometry_position_;
  bool status_follow_odometry_ = true;
  bool latest_odometry_available_ = false;
  bool latest_status_available_ = false;
  deform_monitor_v3::V1DetectionStatus latest_status_;
  std::string status_detail_;
  std::string status_detail_topic_;
  std::string status_odometry_topic_;
  std::string latest_odometry_frame_;
};

}  // namespace v1
}  // namespace deform_monitor_v3

int main(int argc, char** argv) {
  ros::init(argc, argv, "deform_monitor_v1_rviz");
  deform_monitor_v3::v1::V1RvizNode node;
  ros::spin();
  return 0;
}
