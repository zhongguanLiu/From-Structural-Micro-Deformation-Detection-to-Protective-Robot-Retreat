#ifndef DEFORM_MONITOR_V3_RISK_MAP_CHANNEL_ADAPTER_HPP
#define DEFORM_MONITOR_V3_RISK_MAP_CHANNEL_ADAPTER_HPP

#include "deform_monitor_v3/risk_map/risk_types.hpp"

namespace deform_monitor_v3 { namespace risk_map {

class ChannelAdapter {
 public:
  ChannelFrame BuildFrame(const AnchorReferenceVector& anchors,
                          const CurrentObservationVector& observations,
                          const PoseCov6D& pose_cov,
                          const Eigen::Vector3d& sensor_origin_R,
                          const ros::Time& stamp,
                          uint32_t reference_epoch,
                          const FrozenModeVector& modes) const;

  static FrozenModeVector FreezeModes(const AnchorReferenceVector& anchors,
                                      const AnchorStateVector& previous_states,
                                      const ros::Time& source_stamp,
                                      const Eigen::Vector3d& gravity_R);
};

} }
#endif
