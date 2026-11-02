#include <atomic>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <boost/bind/bind.hpp>
#include <gazebo/common/Events.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <livox_laser_simulation/kinematic_imu.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>

namespace gazebo
{
class Icra2027PlanarMovePlugin final : public ModelPlugin
{
public:
  ~Icra2027PlanarMovePlugin() override
  {
    this->alive_.store(false);
    this->callbackQueue_.disable();
    if (this->callbackThread_.joinable())
      this->callbackThread_.join();
    this->updateConnection_.reset();
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->velocitySubscriber_.shutdown();
    this->odometryPublisher_.shutdown();
    this->imuPublisher_.shutdown();
    this->tfBroadcaster_.reset();
    this->rosNode_.reset();
  }

  void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf) override
  {
    this->model_ = std::move(_model);
    if (!ros::isInitialized())
    {
      gzerr << "ICRA2027 planar move requires gazebo_ros.\n";
      return;
    }

    const std::string rosNamespace =
        this->SdfValue(_sdf, "robotNamespace", std::string("/"));
    this->commandTopic_ =
        this->SdfValue(_sdf, "commandTopic", std::string("/cmd_vel"));
    this->odometryTopic_ = this->SdfValue(
        _sdf, "odometryTopic", std::string("/ground_truth/odom"));
    this->odometryFrame_ =
        this->SdfValue(_sdf, "odometryFrame", std::string("world"));
    this->robotBaseFrame_ = this->SdfValue(
        _sdf, "robotBaseFrame", std::string("base_footprint"));
    this->odometryRate_ = this->SdfValue(_sdf, "odometryRate", 50.0);
    this->commandTimeout_ = this->SdfValue(_sdf, "cmdTimeout", 0.5);
    this->kinematicMode_ = this->SdfValue(_sdf, "kinematicMode", false);
    this->publishKinematicImu_ = this->SdfValue(_sdf, "publishKinematicImu", false);
    this->imuRate_ = this->SdfValue(_sdf, "imuRate", 200.0);
    this->imuFrame_ = this->SdfValue(_sdf, "imuFrame", std::string("imu_link"));
    this->imuPose_ = this->SdfValue(_sdf, "imuPose", ignition::math::Pose3d::Zero);
    if (this->publishKinematicImu_ && (!this->kinematicMode_
        || !std::isfinite(this->imuRate_) || this->imuRate_ <= 0
        || this->imuFrame_.empty() || !this->imuPose_.IsFinite()))
    {
      gzerr << "Kinematic IMU requires valid kinematic mode, pose, frame and rate.\n";
      return;
    }
    if (this->odometryRate_ <= 0.0 || this->commandTimeout_ < 0.0)
    {
      gzerr << "Invalid ICRA2027 planar move rate or timeout.\n";
      return;
    }

    this->rosNode_ = std::make_unique<ros::NodeHandle>(rosNamespace);
    ros::SubscribeOptions options =
        ros::SubscribeOptions::create<geometry_msgs::Twist>(
            this->commandTopic_, 1,
            boost::bind(&Icra2027PlanarMovePlugin::VelocityCallback, this,
                        boost::placeholders::_1),
            ros::VoidPtr(), &this->callbackQueue_);
    this->velocitySubscriber_ = this->rosNode_->subscribe(options);
    this->odometryPublisher_ =
        this->rosNode_->advertise<nav_msgs::Odometry>(this->odometryTopic_, 10);
    if (this->publishKinematicImu_)
      this->imuPublisher_ = this->rosNode_->advertise<sensor_msgs::Imu>(
          this->SdfValue(_sdf, "imuTopic", std::string("/livox/imu")), 200);
    this->tfBroadcaster_ = std::make_unique<tf::TransformBroadcaster>();

    this->alive_.store(true);
    this->callbackThread_ = std::thread([this]()
    {
      while (this->alive_.load() && ros::ok())
        this->callbackQueue_.callAvailable(ros::WallDuration(0.02));
    });
    this->lastCommandWallTime_ = ros::WallTime::now();
    this->updateConnection_ = event::Events::ConnectWorldUpdateBegin(
        std::bind(&Icra2027PlanarMovePlugin::OnUpdate, this,
                  std::placeholders::_1));
  }

private:
  template<typename T>
  T SdfValue(const sdf::ElementPtr &_sdf, const std::string &_name,
             const T &_fallback) const
  {
    return _sdf && _sdf->HasElement(_name) ?
        _sdf->Get<T>(_name) : _fallback;
  }

  void VelocityCallback(const geometry_msgs::Twist::ConstPtr &_message)
  {
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->linearCommand_ = std::isfinite(_message->linear.x) ?
        _message->linear.x : 0.0;
    this->angularCommand_ = std::isfinite(_message->angular.z) ?
        _message->angular.z : 0.0;
    this->lastCommandWallTime_ = ros::WallTime::now();
  }

  void OnUpdate(const common::UpdateInfo &_info)
  {
    if (!this->model_ || !this->alive_.load())
      return;
    double linear = 0.0;
    double angular = 0.0;
    {
      std::lock_guard<std::mutex> lock(this->mutex_);
      if ((ros::WallTime::now() - this->lastCommandWallTime_).toSec() <=
          this->commandTimeout_)
      {
        linear = this->linearCommand_;
        angular = this->angularCommand_;
      }
    }

    const double now = _info.simTime.Double();
    double step = 0.0;
    if (std::isfinite(this->lastUpdateTime_) && now >= this->lastUpdateTime_)
      step = now - this->lastUpdateTime_;
    this->lastUpdateTime_ = now;

    if (this->kinematicMode_)
    {
      ignition::math::Pose3d pose = this->model_->WorldPose();
      const ignition::math::Vector3d euler = pose.Rot().Euler();
      const double heading = euler.Z() + 0.5 * angular * step;
      pose.Pos().X() += linear * std::cos(heading) * step;
      pose.Pos().Y() += linear * std::sin(heading) * step;
      const double yaw = std::atan2(
          std::sin(euler.Z() + angular * step),
          std::cos(euler.Z() + angular * step));
      pose.Rot() = ignition::math::Quaterniond(euler.X(), euler.Y(), yaw);
      this->model_->SetWorldPose(pose);
    }
    else
    {
      const double yaw = this->model_->WorldPose().Rot().Euler().Z();
      const double verticalVelocity = this->model_->WorldLinearVel().Z();
      this->model_->SetLinearVel(ignition::math::Vector3d(
          linear * std::cos(yaw), linear * std::sin(yaw), verticalVelocity));
      this->model_->SetAngularVel(
          ignition::math::Vector3d(0.0, 0.0, angular));
    }

    if (this->publishKinematicImu_)
      this->PublishKinematicImu(_info.simTime, step, linear, angular);

    const double publishInterval = 1.0 / this->odometryRate_;
    if (std::isfinite(this->lastOdometryTime_) &&
        now - this->lastOdometryTime_ < publishInterval)
      return;
    this->lastOdometryTime_ = now;
    this->PublishOdometry(_info.simTime, linear, angular);
  }

  void PublishKinematicImu(const common::Time &stamp, double step,
                           double linear, double angular)
  {
    // Gazebo's physics IMU reads zero gyro on a static model moved by
    // SetWorldPose. Simulate the same calibrated sensor from the motion
    // applied above; no truth topic or pose is fed into FAST-LIO/navigation.
    const auto modelPose = this->model_->WorldPose();
    const auto velocity = livox_simulation::imuWorldVelocity(
        modelPose, this->imuPose_.Pos(), linear, angular);
    const double now = stamp.Double();
    if (!std::isfinite(this->lastImuTime_) || now <= this->lastImuTime_)
    {
      this->lastImuTime_ = now;
      this->lastImuVelocity_ = velocity;
      this->imuAngularIntegral_.Set(0, 0, 0);
      return;
    }
    this->imuAngularIntegral_ += ignition::math::Vector3d(0, 0, angular * step);
    const double dt = now - this->lastImuTime_;
    if (dt + 1.e-9 < 1.0 / this->imuRate_)
      return;
    const auto rotation = modelPose.Rot() * this->imuPose_.Rot();
    const auto measurement = livox_simulation::imuSample(
        rotation, this->lastImuVelocity_, velocity, this->imuAngularIntegral_,
        this->model_->GetWorld()->Gravity(), dt);
    sensor_msgs::Imu message;
    message.header.stamp = ros::Time(stamp.sec, stamp.nsec);
    message.header.frame_id = this->imuFrame_;
    message.orientation.x = rotation.X();
    message.orientation.y = rotation.Y();
    message.orientation.z = rotation.Z();
    message.orientation.w = rotation.W();
    message.angular_velocity.x = measurement.first.X();
    message.angular_velocity.y = measurement.first.Y();
    message.angular_velocity.z = measurement.first.Z();
    message.linear_acceleration.x = measurement.second.X();
    message.linear_acceleration.y = measurement.second.Y();
    message.linear_acceleration.z = measurement.second.Z();
    this->imuPublisher_.publish(message);
    this->lastImuTime_ = now;
    this->lastImuVelocity_ = velocity;
    this->imuAngularIntegral_.Set(0, 0, 0);
  }

  void PublishOdometry(const common::Time &_time,
                       const double _linear, const double _angular)
  {
    const ignition::math::Pose3d pose = this->model_->WorldPose();
    const ros::Time stamp(_time.sec, _time.nsec);
    nav_msgs::Odometry odometry;
    odometry.header.stamp = stamp;
    odometry.header.frame_id = this->odometryFrame_;
    odometry.child_frame_id = this->robotBaseFrame_;
    odometry.pose.pose.position.x = pose.Pos().X();
    odometry.pose.pose.position.y = pose.Pos().Y();
    odometry.pose.pose.position.z = pose.Pos().Z();
    odometry.pose.pose.orientation.x = pose.Rot().X();
    odometry.pose.pose.orientation.y = pose.Rot().Y();
    odometry.pose.pose.orientation.z = pose.Rot().Z();
    odometry.pose.pose.orientation.w = pose.Rot().W();
    odometry.twist.twist.linear.x = _linear;
    odometry.twist.twist.angular.z = _angular;
    this->odometryPublisher_.publish(odometry);

    tf::Transform transform;
    transform.setOrigin(tf::Vector3(
        pose.Pos().X(), pose.Pos().Y(), pose.Pos().Z()));
    transform.setRotation(tf::Quaternion(
        pose.Rot().X(), pose.Rot().Y(), pose.Rot().Z(), pose.Rot().W()));
    this->tfBroadcaster_->sendTransform(tf::StampedTransform(
        transform, stamp, this->odometryFrame_, this->robotBaseFrame_));
  }

  physics::ModelPtr model_;
  event::ConnectionPtr updateConnection_;
  std::unique_ptr<ros::NodeHandle> rosNode_;
  ros::CallbackQueue callbackQueue_;
  ros::Subscriber velocitySubscriber_;
  ros::Publisher odometryPublisher_;
  ros::Publisher imuPublisher_;
  bool publishKinematicImu_ = false;
  double imuRate_ = 200;
  std::string imuFrame_;
  ignition::math::Pose3d imuPose_;
  double lastImuTime_ = std::numeric_limits<double>::quiet_NaN();
  ignition::math::Vector3d lastImuVelocity_{0, 0, 0};
  ignition::math::Vector3d imuAngularIntegral_{0, 0, 0};
  std::unique_ptr<tf::TransformBroadcaster> tfBroadcaster_;
  std::thread callbackThread_;
  std::atomic<bool> alive_{false};
  std::mutex mutex_;
  std::string commandTopic_;
  std::string odometryTopic_;
  std::string odometryFrame_;
  std::string robotBaseFrame_;
  double odometryRate_ = 50.0;
  double commandTimeout_ = 0.5;
  double linearCommand_ = 0.0;
  double angularCommand_ = 0.0;
  bool kinematicMode_ = false;
  double lastUpdateTime_ = std::numeric_limits<double>::quiet_NaN();
  double lastOdometryTime_ = std::numeric_limits<double>::quiet_NaN();
  ros::WallTime lastCommandWallTime_;
};

GZ_REGISTER_MODEL_PLUGIN(Icra2027PlanarMovePlugin)
}  // namespace gazebo
