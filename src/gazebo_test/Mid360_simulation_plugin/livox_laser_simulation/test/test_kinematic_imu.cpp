#include <gtest/gtest.h>
#include <livox_laser_simulation/kinematic_imu.h>

using ignition::math::Pose3d;
using ignition::math::Vector3d;
using ignition::math::Quaterniond;

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

TEST(KinematicImu, GravityAndTranslationImpulse) {
  const Vector3d zero(0, 0, 0), gravity(0, 0, -9.8);
  const auto still = livox_simulation::imuSample(
      Quaterniond::Identity, zero, zero, zero, gravity, .005);
  EXPECT_NEAR(still.second.Z(), 9.8, 1.e-10);
  EXPECT_NEAR(still.first.Length(), 0, 1.e-10);
  const auto accelerated = livox_simulation::imuSample(
      Quaterniond::Identity, zero, Vector3d(1, 0, 0), zero, gravity, .005);
  EXPECT_NEAR(accelerated.second.X() * .005, 1, 1.e-10);
  EXPECT_THROW(livox_simulation::imuSample(
      Quaterniond::Identity, zero, zero, zero, gravity, 0), std::invalid_argument);
}

TEST(KinematicImu, RotatingLeverArmAndTiltedSensor) {
  const double dt = .005, omega = .8;
  const Vector3d offset(.2801, 0, .616928878), gravity(0, 0, -9.8);
  Pose3d before(0, 0, 0, 0, 0, 0), after(0, 0, 0, 0, 0, omega * dt);
  const auto v0 = livox_simulation::imuWorldVelocity(before, offset, 0, omega);
  const auto v1 = livox_simulation::imuWorldVelocity(after, offset, 0, omega);
  const auto sample = livox_simulation::imuSample(
      after.Rot(), v0, v1, Vector3d(0, 0, omega * dt), gravity, dt);
  EXPECT_NEAR(sample.first.Z(), omega, 1.e-10);
  EXPECT_NEAR(sample.second.X(), -omega * omega * offset.X(), 1.e-5);
  EXPECT_NEAR(sample.second.Z(), 9.8, 1.e-10);
  Quaterniond tilt(0, .3, 0);
  const auto tilted = livox_simulation::imuSample(
      tilt, Vector3d::Zero, Vector3d::Zero, Vector3d(0, 0, omega * dt), gravity, dt);
  EXPECT_NEAR(tilted.first.X(), -omega * std::sin(.3), 1.e-10);
  EXPECT_NEAR(tilted.second.X(), -9.8 * std::sin(.3), 1.e-10);
}
