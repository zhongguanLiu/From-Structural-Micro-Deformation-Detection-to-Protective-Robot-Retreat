#ifndef DEFORM_MONITOR_V3_V1_NODE_HPP
#define DEFORM_MONITOR_V3_V1_NODE_HPP

#include "deform_monitor_v3/QueryV1Deformation.h"
#include "deform_monitor_v3/SetV1MonitoringPhase.h"
#include "deform_monitor_v3/V1DetectionTile.h"
#include "deform_monitor_v3/V1DeformationField.h"
#include "deform_monitor_v3/V1InterfaceMotionFrame.h"
#include "deform_monitor_v3/V1DetectionStatus.h"
#include "deform_monitor_v3/v1/pipeline.hpp"

#include <fast_lio/LioOdomCov.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <deque>
#include <memory>
#include <mutex>

namespace deform_monitor_v3 {
namespace v1 {

class V1Node {
 public:
  V1Node();
  bool Initialize();
  void Spin();

 private:
  struct TimedCloud { sensor_msgs::PointCloud2 message; };
  struct TimedCovariance { fast_lio::LioOdomCov message; };
  void CloudCallback(const sensor_msgs::PointCloud2ConstPtr& message);
  void CovarianceCallback(const fast_lio::LioOdomCovConstPtr& message);
  void TryProcess();
  bool PopSynchronized(sensor_msgs::PointCloud2* cloud,
                       fast_lio::LioOdomCov* covariance);
  bool BuildBundle(const sensor_msgs::PointCloud2& cloud,
                   const fast_lio::LioOdomCov& covariance,
                   FrameBundle* bundle, ReasonCode* reason,
                   std::string* error) const;
  void PublishSnapshot(const V1Snapshot& snapshot);
  void PublishAttemptStatus(const ros::Time& stamp, uint64_t frame_index,
                            bool published,
                            const std::vector<ReasonCode>& reasons);
  bool SetPhaseService(deform_monitor_v3::SetV1MonitoringPhase::Request& request,
                       deform_monitor_v3::SetV1MonitoringPhase::Response& response);
  bool QueryDeformationService(deform_monitor_v3::QueryV1Deformation::Request& request,
                               deform_monitor_v3::QueryV1Deformation::Response& response);
  bool TokenMatches(const deform_monitor_v3::V1StableId128& token,
                    ReasonCode* rejection_reason = nullptr,
                    ros::Time* evaluation_stamp = nullptr) const;

  ros::NodeHandle node_;
  ros::NodeHandle private_node_{"~"};
  V1Config config_;
  std::unique_ptr<V1Pipeline> pipeline_;
  ros::Subscriber cloud_subscriber_;
  ros::Subscriber covariance_subscriber_;
  ros::Publisher deformation_publisher_;
  ros::Publisher interface_publisher_;
  ros::Publisher tile_publisher_;
  ros::Publisher status_publisher_;
  ros::ServiceServer phase_service_;
  ros::ServiceServer deformation_query_service_;
  std::deque<TimedCloud> cloud_queue_;
  std::deque<TimedCovariance> covariance_queue_;
  mutable std::mutex mutex_;
  uint64_t next_accepted_frame_index_ = 0;
  double last_attempt_processing_time_ms_ = 0.0;
  uint64_t last_attempt_input_point_count_ = 0U;
};

}  // namespace v1
}  // namespace deform_monitor_v3

#endif  // DEFORM_MONITOR_V3_V1_NODE_HPP
