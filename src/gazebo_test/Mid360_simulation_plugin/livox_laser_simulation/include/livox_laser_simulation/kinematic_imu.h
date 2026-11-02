#pragma once

#include <cmath>
#include <stdexcept>
#include <utility>
#include <ignition/math/Pose3.hh>

namespace livox_simulation {
// Velocity at the calibrated IMU origin, including the rotating lever arm.
inline ignition::math::Vector3d imuWorldVelocity(
    const ignition::math::Pose3d &model, const ignition::math::Vector3d &offset,
    double linear, double angular) {
  const double yaw = model.Rot().Yaw();
  return ignition::math::Vector3d(linear * std::cos(yaw), linear * std::sin(yaw), 0)
      + ignition::math::Vector3d(0, 0, angular).Cross(model.Rot().RotateVector(offset));
}

// Return (angular velocity, specific force), expressed in the IMU frame.
// Differencing endpoint velocities retains acceleration impulses between
// publication times; averaging physics-step angular increments avoids aliasing.
inline std::pair<ignition::math::Vector3d, ignition::math::Vector3d> imuSample(
    const ignition::math::Quaterniond &rotation,
    const ignition::math::Vector3d &previousVelocity,
    const ignition::math::Vector3d &velocity,
    const ignition::math::Vector3d &angularIntegral,
    const ignition::math::Vector3d &gravity, double dt) {
  if (!std::isfinite(dt) || dt <= 0)
    throw std::invalid_argument("IMU sample interval must be positive");
  return {rotation.RotateVectorReverse(angularIntegral / dt),
          rotation.RotateVectorReverse((velocity - previousVelocity) / dt - gravity)};
}
}  // namespace livox_simulation
