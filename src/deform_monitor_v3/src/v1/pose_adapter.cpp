#include "deform_monitor_v3/v1/pose_adapter.hpp"

#include "deform_monitor_v3/v1/math.hpp"

#include <cmath>

namespace deform_monitor_v3 {
namespace v1 {
namespace {

const char* kExpectedStateOrder =
    "pos,rot,offset_R_L_I,offset_T_L_I,vel,bg,ba,grav";

bool ValidRotation(const Eigen::Matrix3d& rotation, double tolerance) {
  return rotation.allFinite() &&
         (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() <= tolerance &&
         std::abs(rotation.determinant() - 1.0) <= tolerance;
}

}  // namespace

Eigen::Matrix<double, 6, 6> FastLioToLidarJacobian(
    const Eigen::Matrix3d& R_RI, const Eigen::Vector3d& t_IL) {
  Eigen::Matrix<double, 6, 6> jacobian = Eigen::Matrix<double, 6, 6>::Zero();
  jacobian.block<3, 3>(0, 3) = R_RI;
  jacobian.block<3, 3>(3, 0) = Eigen::Matrix3d::Identity();
  jacobian.block<3, 3>(3, 3) = -R_RI * Skew(t_IL);
  return jacobian;
}

PoseAdapterResult AdaptFastLioPose(const fast_lio::LioOdomCov& message,
                                   const ExtrinsicCalibration& extrinsic,
                                   const NumericalConfig& numerical) {
  PoseAdapterResult result;
  if (message.odom.header.stamp.isZero()) {
    result.reason = ReasonCode::INPUT_TIME_INVALID;
    result.detail = "LioOdomCov odometry stamp must be nonzero";
    return result;
  }
  if (extrinsic.online_estimation_enabled) {
    result.reason = ReasonCode::INPUT_EXTRINSIC_STATE_UNAVAILABLE;
    result.detail = "FAST-LIO online extrinsic estimation is incompatible with the V1 message contract";
    return result;
  }
  if (extrinsic.revision.empty() || extrinsic.digest.empty() ||
      !ValidRotation(extrinsic.R_IL, 1.0e-8) || !extrinsic.t_IL.allFinite()) {
    result.reason = ReasonCode::INPUT_EXTRINSIC_STATE_UNAVAILABLE;
    result.detail = "fixed LiDAR-to-IMU extrinsic is absent or invalid";
    return result;
  }
  if (message.state_dim != 23U || message.state_order != kExpectedStateOrder ||
      message.state_covariance.size() != 529U) {
    result.reason = ReasonCode::INPUT_SCHEMA_INVALID;
    result.detail = "LioOdomCov must contain the exact 23D IKFoM state schema";
    return result;
  }
  Eigen::Matrix<double, 23, 23> full;
  for (int row = 0; row < 23; ++row) {
    for (int column = 0; column < 23; ++column) {
      const double value = message.state_covariance[static_cast<size_t>(23 * row + column)];
      if (!std::isfinite(value)) {
        result.reason = ReasonCode::INPUT_POSE_COVARIANCE_INVALID;
        result.detail = "non-finite entry in FAST-LIO covariance";
        return result;
      }
      full(row, column) = value;
    }
  }
  if (!ValidateSymmetricPsd(full, numerical.psd_eigenvalue_tolerance)) {
    result.reason = ReasonCode::INPUT_POSE_COVARIANCE_INVALID;
    result.detail = "FAST-LIO covariance is not symmetric positive semidefinite";
    return result;
  }

  const geometry_msgs::Pose& pose = message.odom.pose.pose;
  const Eigen::Vector3d p_RI(pose.position.x, pose.position.y, pose.position.z);
  Eigen::Quaterniond q_RI(pose.orientation.w, pose.orientation.x,
                          pose.orientation.y, pose.orientation.z);
  if (!p_RI.allFinite() || !q_RI.coeffs().allFinite() ||
      std::abs(q_RI.norm() - 1.0) > 1.0e-6) {
    result.reason = ReasonCode::INPUT_SCHEMA_INVALID;
    result.detail = "FAST-LIO odometry pose is non-finite or quaternion is not normalized";
    return result;
  }
  q_RI.normalize();
  const Eigen::Matrix3d R_RI = q_RI.toRotationMatrix();
  result.pose.R_RL = R_RI * extrinsic.R_IL;
  result.pose.origin_L_R = p_RI + R_RI * extrinsic.t_IL;

  const Eigen::Matrix<double, 6, 6> source = full.block<6, 6>(0, 0);
  const Eigen::Matrix<double, 6, 6> jacobian =
      FastLioToLidarJacobian(R_RI, extrinsic.t_IL);
  result.pose.covariance = jacobian * source * jacobian.transpose();
  result.pose.covariance = 0.5 * (result.pose.covariance + result.pose.covariance.transpose());
  if (!ValidateSymmetricPsd(result.pose.covariance,
                            numerical.psd_eigenvalue_tolerance)) {
    result.reason = ReasonCode::INPUT_POSE_COVARIANCE_INVALID;
    result.detail = "transformed LiDAR covariance failed PSD validation";
    return result;
  }
  result.valid = true;
  result.reason = ReasonCode::NONE;
  return result;
}

}  // namespace v1
}  // namespace deform_monitor_v3
