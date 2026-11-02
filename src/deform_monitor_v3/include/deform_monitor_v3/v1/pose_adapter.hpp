#ifndef DEFORM_MONITOR_V3_V1_POSE_ADAPTER_HPP
#define DEFORM_MONITOR_V3_V1_POSE_ADAPTER_HPP

#include "deform_monitor_v3/v1/types.hpp"

#include <fast_lio/LioOdomCov.h>

namespace deform_monitor_v3 {
namespace v1 {

struct PoseAdapterResult {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool valid = false;
  PoseEstimate pose;
  ReasonCode reason = ReasonCode::INPUT_SCHEMA_INVALID;
  std::string detail;
};

// DEC-029: strict FAST-LIO 23D source convention to the canonical LiDAR
// convention. There is intentionally no odometry/default-sigma fallback.
PoseAdapterResult AdaptFastLioPose(const fast_lio::LioOdomCov& message,
                                   const ExtrinsicCalibration& extrinsic,
                                   const NumericalConfig& numerical);

Eigen::Matrix<double, 6, 6> FastLioToLidarJacobian(
    const Eigen::Matrix3d& R_RI, const Eigen::Vector3d& t_IL);

}  // namespace v1
}  // namespace deform_monitor_v3

#endif  // DEFORM_MONITOR_V3_V1_POSE_ADAPTER_HPP
