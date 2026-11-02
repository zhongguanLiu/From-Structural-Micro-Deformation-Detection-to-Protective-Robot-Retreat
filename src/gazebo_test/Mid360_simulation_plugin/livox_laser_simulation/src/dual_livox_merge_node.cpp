#include "livox_laser_simulation/dual_livox_merge.hpp"
#include "livox_laser_simulation/point_time_model.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <livox_ros_driver/CustomMsg.h>
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>
#include <std_msgs/UInt64.h>

namespace livox_laser_simulation {
namespace {

template <std::size_t N>
bool ReadArray(ros::NodeHandle& node, const std::string& name,
               std::array<double, N>* output) {
  std::vector<double> values;
  if (!node.getParam(name, values) || values.size() != N) return false;
  std::copy(values.begin(), values.end(), output->begin());
  return true;
}

class DualLivoxMergeNode {
 public:
  DualLivoxMergeNode() : private_node_("~") {
    private_node_.param("primary_topic", primary_topic_,
                        std::string("/livox/front/lidar"));
    private_node_.param("secondary_topic", secondary_topic_,
                        std::string("/livox/rear/lidar"));
    private_node_.param("output_topic", output_topic_,
                        std::string("/livox/lidar"));
    private_node_.param("primary_frame", primary_frame_,
                        std::string("livox_frame"));
    private_node_.param("secondary_frame", secondary_frame_,
                        std::string("livox_rear_frame"));
    private_node_.param("secondary_enabled", secondary_enabled_, true);
    private_node_.param("diagnostics_topic", diagnostics_topic_,
                        std::string("/livox/merge/diagnostics"));
    private_node_.param("readiness_topic", readiness_topic_,
                        std::string("/livox/merge/ready"));
    private_node_.param(
        "primary_scheduler_catch_up_topic",
        primary_scheduler_catch_up_topic_,
        primary_topic_ + "/scheduler_catch_up_suppressed_count");
    private_node_.param("point_time_mode", point_time_mode_,
                        std::string("legacy_rolling"));
    private_node_.param("configured_sensor_update_rate_hz",
                        configured_sensor_update_rate_hz_, 10.0);
    if (!std::isfinite(configured_sensor_update_rate_hz_) ||
        configured_sensor_update_rate_hz_ <= 0.0 ||
        configured_sensor_update_rate_hz_ > 1000.0)
      throw std::runtime_error(
          "configured_sensor_update_rate_hz must be finite and in (0,1000]");
    private_node_.param("configured_min_range_m", configured_min_range_m_,
                        0.1);
    private_node_.param("configured_max_range_m", configured_max_range_m_,
                        100.0);
    if (!std::isfinite(configured_min_range_m_) ||
        !std::isfinite(configured_max_range_m_) ||
        configured_min_range_m_ < 0.0 ||
        configured_min_range_m_ >= configured_max_range_m_)
      throw std::runtime_error(
          "configured LiDAR range must be finite with 0 <= min < max");
    private_node_.param("configured_ray_samples_per_sensor",
                        configured_ray_samples_per_sensor_, 20000);
    if (configured_ray_samples_per_sensor_ < 1)
      throw std::runtime_error(
          "configured_ray_samples_per_sensor must be >= 1");
    private_node_.param("primary_downsample", primary_downsample_, 1);
    private_node_.param("secondary_downsample", secondary_downsample_, 1);
    if (primary_downsample_ < 1 || secondary_downsample_ < 1)
      throw std::runtime_error("LiDAR downsample strides must be >= 1");
    private_node_.param("overlap_voxel_size_m", overlap_voxel_size_m_, 0.20);
    private_node_.param("overlap_evaluation_pair_stride",
                        overlap_evaluation_pair_stride_, 10);
    if (!std::isfinite(overlap_voxel_size_m_) ||
        overlap_voxel_size_m_ <= 0.0 || overlap_voxel_size_m_ > 10.0)
      throw std::runtime_error(
          "overlap_voxel_size_m must be finite and in (0,10]");
    if (overlap_evaluation_pair_stride_ < 1)
      throw std::runtime_error(
          "overlap_evaluation_pair_stride must be >= 1");
    PointTimeMode parsed_time_mode;
    if (!ParsePointTimeMode(point_time_mode_, &parsed_time_mode))
      throw std::runtime_error(
          "point_time_mode must be snapshot or legacy_rolling");
    point_time_mode_ = PointTimeModeName(parsed_time_mode);
    double tolerance_seconds = 0.025;
    private_node_.param("maximum_sync_delta_s", tolerance_seconds, 0.025);
    if (!std::isfinite(tolerance_seconds) || tolerance_seconds < 0.0 ||
        tolerance_seconds > 1.0)
      throw std::runtime_error("maximum_sync_delta_s must be in [0,1]");
    maximum_sync_delta_ns_ = static_cast<uint64_t>(
        std::llround(tolerance_seconds * 1.0e9));
    if (!private_node_.getParam("calibration_revision",
                                calibration_.revision) ||
        !ReadArray(private_node_, "rotation_primary_secondary_row_major",
                   &calibration_.rotation_primary_secondary) ||
        !ReadArray(private_node_, "translation_primary_secondary_m",
                   &calibration_.translation_primary_secondary))
      throw std::runtime_error("dual-LiDAR calibration parameters are incomplete");
    std::string error;
    if (!ValidateDualLidarCalibration(calibration_, &error))
      throw std::runtime_error(error);

    publisher_ = node_.advertise<livox_ros_driver::CustomMsg>(output_topic_, 5);
    diagnostics_publisher_ =
        node_.advertise<std_msgs::String>(diagnostics_topic_, 2, true);
    readiness_publisher_ =
        node_.advertise<std_msgs::Bool>(readiness_topic_, 1, true);
    diagnostics_timer_ = node_.createTimer(
        ros::Duration(1.0), &DualLivoxMergeNode::PublishDiagnostics, this);
    primary_subscriber_ = node_.subscribe(
        primary_topic_, 10, &DualLivoxMergeNode::PrimaryCallback, this);
    primary_scheduler_catch_up_subscriber_ = node_.subscribe(
        primary_scheduler_catch_up_topic_, 1,
        &DualLivoxMergeNode::PrimarySchedulerCatchUpCallback, this);
    if (secondary_enabled_)
      secondary_subscriber_ = node_.subscribe(
          secondary_topic_, 10, &DualLivoxMergeNode::SecondaryCallback, this);
    // A ROS-time timer cannot run while Gazebo is intentionally paused.
    // This wall timer proves that both source publishers have completed their
    // transport handshake with the merger before the unpause controller lets
    // the first common acquisition occur.
    readiness_timer_ = node_.createWallTimer(
        ros::WallDuration(0.05),
        &DualLivoxMergeNode::PublishReadinessWhenConnected, this);
    ROS_INFO_STREAM("dual Livox front=" << primary_topic_
                    << " rear=" << (secondary_enabled_ ? secondary_topic_ : "disabled")
                    << " output=" << output_topic_
                    << " calibration=" << calibration_.revision
                    << " point_time_mode=" << point_time_mode_
                    << " configured_sensor_update_rate_hz="
                    << configured_sensor_update_rate_hz_
                    << " physical_range=[" << configured_min_range_m_ << ","
                    << configured_max_range_m_ << "]m"
                    << " ray_samples=" << configured_ray_samples_per_sensor_
                    << " downsample=" << primary_downsample_ << "/"
                    << secondary_downsample_
                    << " overlap_voxel=" << overlap_voxel_size_m_ << "m/"
                    << overlap_evaluation_pair_stride_ << "pairs"
                    << " zero_no_return_compaction="
                    << (secondary_enabled_ ? "dual-merge-only" : "disabled-relay")
                    << "; deskew uses only /livox/imu after rigid-frame merge");
  }

 private:
  using Message = livox_ros_driver::CustomMsg;
  using MessagePtr = livox_ros_driver::CustomMsgConstPtr;

  void PrimaryCallback(const MessagePtr& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    statistics_.RecordPrimaryReceived(message->points.size());
    if (!secondary_enabled_) {
      if (message->header.frame_id != primary_frame_ ||
          message->point_num != message->points.size()) {
        ROS_ERROR_THROTTLE(1.0, "invalid primary Livox message; frame not relayed");
        return;
      }
      publisher_.publish(*message);
      statistics_.RecordPrimaryRelayed(message->points.size());
      return;
    }
    primary_.push_back(message);
    Trim(&primary_, true);
    Match();
  }

  void PublishReadinessWhenConnected(const ros::WallTimerEvent&) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (readiness_published_) return;
    const bool primary_connected = primary_subscriber_.getNumPublishers() > 0U;
    const bool secondary_connected =
        !secondary_enabled_ || secondary_subscriber_.getNumPublishers() > 0U;
    if (!primary_connected || !secondary_connected) return;
    std_msgs::Bool message;
    message.data = true;
    readiness_publisher_.publish(message);
    readiness_published_ = true;
    readiness_timer_.stop();
    ROS_INFO("dual Livox merger transport ready before first acquisition");
  }

  void SecondaryCallback(const MessagePtr& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    statistics_.RecordSecondaryReceived(message->points.size());
    secondary_.push_back(message);
    Trim(&secondary_, false);
    Match();
  }

  void PrimarySchedulerCatchUpCallback(
      const std_msgs::UInt64ConstPtr& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    // The source publishes a latched, monotonic cumulative value.  Never let
    // a stale ROS callback make the audit counter move backwards.
    primary_scheduler_catch_up_suppressed_count_ = std::max(
        primary_scheduler_catch_up_suppressed_count_, message->data);
  }

  void Trim(std::deque<MessagePtr>* queue, bool primary_queue) {
    while (queue->size() > 5U) {
      queue->pop_front();
      if (primary_queue)
        statistics_.RecordPrimaryQueueDrop();
      else
        statistics_.RecordSecondaryQueueDrop();
      ROS_WARN_THROTTLE(1.0, "dual Livox synchronization queue overflow; oldest scan dropped");
    }
  }

  void Match() {
    while (!primary_.empty() && !secondary_.empty()) {
      const uint64_t primary_time = primary_.front()->timebase;
      const uint64_t secondary_time = secondary_.front()->timebase;
      const uint64_t delta = primary_time >= secondary_time
          ? primary_time - secondary_time
          : secondary_time - primary_time;
      if (delta <= maximum_sync_delta_ns_) {
        Message output;
        DualLivoxMergeAccounting accounting;
        std::string error;
        if (!MergeDualLivoxMessages(
                *primary_.front(), *secondary_.front(), calibration_,
                primary_frame_, secondary_frame_, maximum_sync_delta_ns_,
                &output, &error, &accounting)) {
          ROS_ERROR_THROTTLE(1.0, "dual Livox pair rejected: %s", error.c_str());
          statistics_.RecordPairRejected();
        } else {
          publisher_.publish(output);
          statistics_.RecordPairAccepted(
              output.timebase, delta, LivoxPointOffsetSpanNs(output),
              accounting);
          const uint64_t accepted_pair_index =
              statistics_.Snapshot().pairs_accepted;
          if (accepted_pair_index %
                  static_cast<uint64_t>(overlap_evaluation_pair_stride_) ==
              0U) {
            DualLidarVoxelOverlap overlap;
            if (ComputeDualLidarVoxelOverlap(
                    *primary_.front(), *secondary_.front(), calibration_,
                    primary_frame_, secondary_frame_, overlap_voxel_size_m_,
                    &overlap, &error)) {
              statistics_.RecordVoxelOverlapEvaluation(
                  accepted_pair_index, overlap);
            } else {
              statistics_.RecordVoxelOverlapEvaluationFailed();
              ROS_WARN_THROTTLE(
                  2.0, "dual Livox overlap diagnostic failed: %s",
                  error.c_str());
            }
          }
        }
        primary_.pop_front();
        secondary_.pop_front();
      } else if (primary_time < secondary_time) {
        primary_.pop_front();
        statistics_.RecordPrimaryUnpairedDrop();
        ROS_WARN_THROTTLE(1.0, "unpaired primary Livox scan dropped");
      } else {
        secondary_.pop_front();
        statistics_.RecordSecondaryUnpairedDrop();
        ROS_WARN_THROTTLE(1.0, "unpaired rear Livox scan dropped");
      }
    }
  }

  void PublishDiagnostics(const ros::TimerEvent&) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto value = statistics_.Snapshot();
    std::ostringstream json;
    const double configured_secondary_primary_ratio =
        static_cast<double>(primary_downsample_) /
        static_cast<double>(secondary_downsample_);
    json << std::setprecision(17)
         << "{\"schema_version\":5"
         << ",\"ros_time_s\":" << ros::Time::now().toSec()
         << ",\"secondary_enabled\":"
         << (secondary_enabled_ ? "true" : "false")
         << ",\"zero_no_return_compaction_enabled\":"
         << (secondary_enabled_ ? "true" : "false")
         << ",\"point_time_mode\":\"" << point_time_mode_ << "\""
         << ",\"configured_sensor_update_rate_hz\":"
         << configured_sensor_update_rate_hz_
         << ",\"configured_min_range_m\":" << configured_min_range_m_
         << ",\"configured_max_range_m\":" << configured_max_range_m_
         << ",\"configured_ray_samples_per_sensor\":"
         << configured_ray_samples_per_sensor_
         << ",\"primary_scheduler_catch_up_suppressed_count\":"
         << primary_scheduler_catch_up_suppressed_count_
         << ",\"primary_downsample\":" << primary_downsample_
         << ",\"secondary_downsample\":" << secondary_downsample_
         << ",\"configured_secondary_to_primary_point_ratio\":"
         << configured_secondary_primary_ratio
         << ",\"overlap_voxel_size_m\":" << overlap_voxel_size_m_
         << ",\"overlap_evaluation_pair_stride\":"
         << overlap_evaluation_pair_stride_
         << ",\"primary_received\":" << value.primary_received
         << ",\"secondary_received\":" << value.secondary_received
         << ",\"pairs_accepted\":" << value.pairs_accepted
         << ",\"pairs_rejected\":" << value.pairs_rejected
         << ",\"primary_unpaired_dropped\":"
         << value.primary_unpaired_dropped
         << ",\"secondary_unpaired_dropped\":"
         << value.secondary_unpaired_dropped
         << ",\"primary_queue_dropped\":"
         << value.primary_queue_dropped
         << ",\"secondary_queue_dropped\":"
         << value.secondary_queue_dropped
         << ",\"last_sync_delta_ns\":" << value.last_sync_delta_ns
         << ",\"maximum_sync_delta_ns\":"
         << value.maximum_sync_delta_ns
         << ",\"mean_sync_delta_ns\":" << value.mean_sync_delta_ns
         << ",\"first_accepted_measurement_time_ns\":"
         << value.first_accepted_measurement_time_ns
         << ",\"last_accepted_measurement_time_ns\":"
         << value.last_accepted_measurement_time_ns
         << ",\"accepted_measurement_interval_count\":"
         << value.accepted_measurement_interval_count
         << ",\"nonpositive_accepted_measurement_interval_count\":"
         << value.nonpositive_accepted_measurement_interval_count
         << ",\"minimum_accepted_measurement_interval_ns\":"
         << value.minimum_accepted_measurement_interval_ns
         << ",\"maximum_accepted_measurement_interval_ns\":"
         << value.maximum_accepted_measurement_interval_ns
         << ",\"mean_accepted_measurement_interval_ns\":"
         << value.mean_accepted_measurement_interval_ns
         << ",\"realized_measurement_hz\":"
         << value.realized_measurement_hz
         << ",\"last_merged_span_ns\":" << value.last_merged_span_ns
         << ",\"maximum_merged_span_ns\":"
         << value.maximum_merged_span_ns
         << ",\"last_primary_point_count\":"
         << value.last_primary_point_count
         << ",\"maximum_primary_point_count\":"
         << value.maximum_primary_point_count
         << ",\"last_secondary_point_count\":"
         << value.last_secondary_point_count
         << ",\"maximum_secondary_point_count\":"
         << value.maximum_secondary_point_count
         << ",\"last_merged_point_count\":"
         << value.last_merged_point_count
         << ",\"maximum_merged_point_count\":"
         << value.maximum_merged_point_count
         << ",\"last_secondary_to_primary_point_ratio\":"
         << value.last_secondary_to_primary_point_ratio
         << ",\"last_merged_to_primary_point_ratio\":"
         << value.last_merged_to_primary_point_ratio
         << ",\"last_total_raw_input_point_count\":"
         << value.last_total_raw_input_point_count
         << ",\"maximum_total_raw_input_point_count\":"
         << value.maximum_total_raw_input_point_count
         << ",\"last_compacted_output_point_count\":"
         << value.last_compacted_output_point_count
         << ",\"maximum_compacted_output_point_count\":"
         << value.maximum_compacted_output_point_count
         << ",\"last_removed_no_return_sentinel_count\":"
         << value.last_removed_no_return_sentinel_count
         << ",\"maximum_removed_no_return_sentinel_count\":"
         << value.maximum_removed_no_return_sentinel_count
         << ",\"last_compacted_output_to_total_raw_input_ratio\":"
         << value.last_compacted_output_to_total_raw_input_ratio
         << ",\"overlap_evaluations_attempted\":"
         << value.overlap_evaluations_attempted
         << ",\"overlap_evaluations_failed\":"
         << value.overlap_evaluations_failed
         << ",\"overlap_evaluation_sample_count\":"
         << value.overlap_evaluation_sample_count
         << ",\"last_overlap_pair_index\":"
         << value.last_overlap_pair_index
         << ",\"last_primary_valid_return_count\":"
         << value.last_primary_valid_return_count
         << ",\"last_secondary_valid_return_count\":"
         << value.last_secondary_valid_return_count
         << ",\"last_primary_occupied_voxel_count\":"
         << value.last_primary_occupied_voxel_count
         << ",\"last_secondary_occupied_voxel_count\":"
         << value.last_secondary_occupied_voxel_count
         << ",\"last_shared_occupied_voxel_count\":"
         << value.last_shared_occupied_voxel_count
         << ",\"last_union_occupied_voxel_count\":"
         << value.last_union_occupied_voxel_count
         << ",\"last_primary_overlap_fraction\":"
         << value.last_primary_overlap_fraction
         << ",\"last_secondary_overlap_fraction\":"
         << value.last_secondary_overlap_fraction
         << ",\"last_voxel_jaccard\":" << value.last_voxel_jaccard
         << ",\"mean_primary_overlap_fraction\":"
         << value.mean_primary_overlap_fraction
         << ",\"mean_secondary_overlap_fraction\":"
         << value.mean_secondary_overlap_fraction
         << ",\"mean_voxel_jaccard\":" << value.mean_voxel_jaccard
         << ",\"minimum_primary_overlap_fraction\":"
         << value.minimum_primary_overlap_fraction
         << ",\"minimum_secondary_overlap_fraction\":"
         << value.minimum_secondary_overlap_fraction
         << ",\"minimum_voxel_jaccard\":"
         << value.minimum_voxel_jaccard << "}";
    std_msgs::String message;
    message.data = json.str();
    diagnostics_publisher_.publish(message);
  }

  ros::NodeHandle node_;
  ros::NodeHandle private_node_;
  ros::Publisher publisher_;
  ros::Publisher diagnostics_publisher_;
  ros::Publisher readiness_publisher_;
  ros::Subscriber primary_subscriber_;
  ros::Subscriber secondary_subscriber_;
  ros::Subscriber primary_scheduler_catch_up_subscriber_;
  ros::Timer diagnostics_timer_;
  ros::WallTimer readiness_timer_;
  std::mutex mutex_;
  std::deque<MessagePtr> primary_;
  std::deque<MessagePtr> secondary_;
  DualLidarCalibration calibration_;
  DualLidarMergeStatistics statistics_;
  std::string primary_topic_;
  std::string secondary_topic_;
  std::string output_topic_;
  std::string diagnostics_topic_;
  std::string readiness_topic_;
  std::string primary_scheduler_catch_up_topic_;
  std::string point_time_mode_;
  std::string primary_frame_;
  std::string secondary_frame_;
  double configured_sensor_update_rate_hz_ = 10.0;
  double configured_min_range_m_ = 0.1;
  double configured_max_range_m_ = 100.0;
  int configured_ray_samples_per_sensor_ = 20000;
  bool secondary_enabled_ = true;
  bool readiness_published_ = false;
  int primary_downsample_ = 1;
  int secondary_downsample_ = 1;
  double overlap_voxel_size_m_ = 0.20;
  int overlap_evaluation_pair_stride_ = 10;
  uint64_t maximum_sync_delta_ns_ = 25000000U;
  uint64_t primary_scheduler_catch_up_suppressed_count_ = 0U;
};

}  // namespace
}  // namespace livox_laser_simulation

int main(int argc, char** argv) {
  ros::init(argc, argv, "dual_livox_merge");
  try {
    livox_laser_simulation::DualLivoxMergeNode node;
    ros::spin();
  } catch (const std::exception& error) {
    ROS_FATAL("dual Livox merge initialization failed: %s", error.what());
    return 1;
  }
  return 0;
}
