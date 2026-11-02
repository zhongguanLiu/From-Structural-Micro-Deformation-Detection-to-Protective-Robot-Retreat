#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <boost/bind/bind.hpp>
#include <gazebo/common/Events.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <ignition/math/AxisAlignedBox.hh>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector3.hh>
#include <ros/advertise_service_options.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <std_msgs/String.h>
#include <std_srvs/Trigger.h>
#include <XmlRpcValue.h>

namespace gazebo
{
namespace
{
constexpr int kFirstId = 1;
constexpr int kLastId = 6;
constexpr double kMillimetersToMeters = 0.001;
constexpr double kTolerance = 1.0e-9;

bool XmlNumber(const XmlRpc::XmlRpcValue &_value, double &_result)
{
  if (_value.getType() == XmlRpc::XmlRpcValue::TypeInt)
  {
    _result = static_cast<int>(_value);
    return true;
  }
  if (_value.getType() == XmlRpc::XmlRpcValue::TypeDouble)
  {
    _result = static_cast<double>(_value);
    return true;
  }
  return false;
}

bool XmlInteger(const XmlRpc::XmlRpcValue &_value, int &_result)
{
  double numeric = 0.0;
  if (!XmlNumber(_value, numeric) || !std::isfinite(numeric) ||
      std::fabs(numeric - std::round(numeric)) > kTolerance)
  {
    return false;
  }
  _result = static_cast<int>(std::llround(numeric));
  return true;
}

bool XmlVector3(const XmlRpc::XmlRpcValue &_value,
                ignition::math::Vector3d &_result)
{
  if (_value.getType() != XmlRpc::XmlRpcValue::TypeArray || _value.size() != 3)
    return false;
  double values[3] = {0.0, 0.0, 0.0};
  for (int index = 0; index < 3; ++index)
  {
    if (!XmlNumber(_value[index], values[index]) ||
        !std::isfinite(values[index]))
    {
      return false;
    }
  }
  _result.Set(values[0], values[1], values[2]);
  return true;
}

bool XmlVector3List(const XmlRpc::XmlRpcValue &_value,
                    std::vector<ignition::math::Vector3d> &_result)
{
  if (_value.getType() != XmlRpc::XmlRpcValue::TypeArray)
    return false;
  std::vector<ignition::math::Vector3d> parsed;
  parsed.reserve(_value.size());
  for (int index = 0; index < _value.size(); ++index)
  {
    ignition::math::Vector3d point;
    if (!XmlVector3(_value[index], point))
      return false;
    parsed.push_back(point);
  }
  _result = std::move(parsed);
  return true;
}

bool AllowedLinearSpeed(const double _value)
{
  const std::array<double, 4> allowed{{1.0, 3.0, 5.0, 10.0}};
  return std::any_of(allowed.begin(), allowed.end(), [_value](double candidate)
  {
    return std::fabs(_value - candidate) <= kTolerance;
  });
}

bool AllowedAngularSpeed(const double _value)
{
  const std::array<double, 2> allowed{{0.0015, 0.0025}};
  return std::any_of(allowed.begin(), allowed.end(), [_value](double candidate)
  {
    return std::fabs(_value - candidate) <= kTolerance;
  });
}

std::string JsonEscape(const std::string &_value)
{
  std::ostringstream output;
  for (const char character : _value)
  {
    switch (character)
    {
      case '\\': output << "\\\\"; break;
      case '"': output << "\\\""; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default: output << character; break;
    }
  }
  return output.str();
}

ignition::math::Vector3d TransformPoint(
    const ignition::math::Pose3d &_pose,
    const ignition::math::Vector3d &_point)
{
  return _pose.Pos() + _pose.Rot().RotateVector(_point);
}

double QuaternionDistance(const ignition::math::Quaterniond &_initial,
                          const ignition::math::Quaterniond &_current)
{
  ignition::math::Quaterniond relative = _initial.Inverse() * _current;
  relative.Normalize();
  const double boundedW = std::max(0.0, std::min(1.0,
      std::fabs(relative.W())));
  return 2.0 * std::acos(boundedW);
}
}  // namespace

class Icra2027MotionWorldPlugin final : public WorldPlugin
{
public:
  ~Icra2027MotionWorldPlugin() override
  {
    this->alive_.store(false);
    this->callbackQueue_.disable();
    if (this->callbackThread_.joinable())
      this->callbackThread_.join();
    this->updateConnection_.reset();
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->reloadService_.shutdown();
    this->startService_.shutdown();
    this->stopService_.shutdown();
    this->resetService_.shutdown();
    this->statusService_.shutdown();
    this->statusPublisher_.shutdown();
    this->truthPublisher_.shutdown();
    this->rosNode_.reset();
  }

  void Load(physics::WorldPtr _world, sdf::ElementPtr _sdf) override
  {
    this->world_ = std::move(_world);
    if (_sdf && _sdf->HasElement("ros_namespace"))
      this->rosNamespace_ = _sdf->Get<std::string>("ros_namespace");

    std::string error;
    this->environmentValid_ = this->DiscoverEnvironment(error);
    if (this->environmentValid_)
    {
      this->CaptureSceneInitialPoses();
      this->ApplyPhysicsOwnership();
      gzmsg << "ICRA2027 motion plugin owns m01-m06; all are gravity-free "
            << "kinematic objects while the plugin is loaded.\n";
    }
    else
    {
      this->lastError_ = error;
      gzerr << "ICRA2027 motion plugin disabled: " << error << "\n";
    }

    this->updateConnection_ = event::Events::ConnectWorldUpdateBegin(
        std::bind(&Icra2027MotionWorldPlugin::OnUpdate, this,
                  std::placeholders::_1));

    if (!ros::isInitialized())
    {
      this->environmentValid_ = false;
      this->lastError_ =
          "ROS is not initialized; start Gazebo through gazebo_ros";
      gzerr << this->lastError_ << "\n";
      return;
    }

    this->rosNode_ = std::make_unique<ros::NodeHandle>(this->rosNamespace_);
    this->statusPublisher_ =
        this->rosNode_->advertise<std_msgs::String>("status", 1, true);
    this->truthPublisher_ =
        this->rosNode_->advertise<std_msgs::String>("truth", 10, true);

    this->reloadService_ = this->AdvertiseService(
        "reload", &Icra2027MotionWorldPlugin::Reload);
    this->startService_ = this->AdvertiseService(
        "start", &Icra2027MotionWorldPlugin::Start);
    this->stopService_ = this->AdvertiseService(
        "stop", &Icra2027MotionWorldPlugin::Stop);
    this->resetService_ = this->AdvertiseService(
        "reset", &Icra2027MotionWorldPlugin::Reset);
    this->statusService_ = this->AdvertiseService(
        "get_status", &Icra2027MotionWorldPlugin::GetStatus);

    this->alive_.store(true);
    this->callbackThread_ = std::thread([this]()
    {
      while (this->alive_.load() && ros::ok())
        this->callbackQueue_.callAvailable(ros::WallDuration(0.05));
    });

    std::lock_guard<std::mutex> lock(this->mutex_);
    this->PublishStatus();
  }

private:
  enum class State
  {
    Unconfigured,
    Armed,
    Waiting,
    Running,
    Holding,
    Completed,
    Stopped,
    Error
  };

  struct ModelHandle
  {
    physics::ModelPtr model;
    physics::LinkPtr driveLink;
  };

  struct MotionCommand
  {
    int id = 0;
    std::string modelName;
    ignition::math::Vector3d linear;
    ignition::math::Vector3d angular;
    std::vector<ignition::math::Vector3d> surfacePoints;
  };

  using ServiceCallback = bool (Icra2027MotionWorldPlugin::*)(
      std_srvs::Trigger::Request &, std_srvs::Trigger::Response &);

  ros::ServiceServer AdvertiseService(const std::string &_name,
                                      ServiceCallback _callback)
  {
    ros::AdvertiseServiceOptions options =
        ros::AdvertiseServiceOptions::create<std_srvs::Trigger>(
            _name,
            boost::bind(_callback, this, boost::placeholders::_1,
                        boost::placeholders::_2),
            ros::VoidPtr(), &this->callbackQueue_);
    return this->rosNode_->advertiseService(options);
  }

  bool DiscoverEnvironment(std::string &_error)
  {
    for (int id = kFirstId; id <= kLastId; ++id)
    {
      const std::string expectedName = "m0" + std::to_string(id);
      physics::ModelPtr model;
      for (const physics::ModelPtr &candidate : this->world_->Models())
      {
        if (candidate->GetName() == expectedName)
        {
          model = candidate;
          break;
        }
      }
      if (!model)
      {
        _error = "missing controlled model " + expectedName;
        return false;
      }

      std::vector<physics::LinkPtr> collisionLinks;
      for (const physics::LinkPtr &link : model->GetLinks())
      {
        if (!link->GetCollisions().empty())
          collisionLinks.push_back(link);
        for (const physics::CollisionPtr &collision : link->GetCollisions())
        {
          const double retro = collision->GetLaserRetro();
          if (!std::isfinite(retro) ||
              std::fabs(retro - static_cast<double>(id)) > kTolerance)
          {
            _error = expectedName + " must use laser_retro " +
                     std::to_string(id) + " on every collision";
            return false;
          }
        }
      }
      if (collisionLinks.size() != 1)
      {
        _error = expectedName +
                 " must have exactly one collision-bearing drive link";
        return false;
      }
      this->controlledModels_[id] = {model, collisionLinks.front()};
    }
    return true;
  }

  void CaptureSceneInitialPoses()
  {
    this->sceneInitialPoses_.clear();
    this->sceneInitialLinkPoses_.clear();
    for (const auto &entry : this->controlledModels_)
    {
      this->sceneInitialPoses_[entry.first] = entry.second.model->WorldPose();
      this->sceneInitialLinkPoses_[entry.first] =
          entry.second.driveLink->WorldPose();
    }
  }

  void ApplyPhysicsOwnership()
  {
    const ignition::math::Vector3d zero = ignition::math::Vector3d::Zero;
    for (auto &entry : this->controlledModels_)
    {
      const ModelHandle &handle = entry.second;
      handle.model->SetStatic(false);
      handle.model->SetGravityMode(false);
      for (const physics::LinkPtr &link : handle.model->GetLinks())
      {
        link->SetGravityMode(false);
        link->SetForce(zero);
        link->SetTorque(zero);
        link->SetLinearVel(zero);
        link->SetAngularVel(zero);
      }
      // Gazebo 11 ODELink::SetKinematic(true) is not idempotent: calling
      // it on an already kinematic body can invoke dBodySetDynamic.
      // Both Load and reset reach here; query the actual ODE state first.
      if (!handle.driveLink->GetKinematic())
        handle.driveLink->SetKinematic(true);
      handle.driveLink->SetEnabled(true);
    }
  }

  bool ReadConfiguration(std::map<int, MotionCommand> &_commands,
                         std::string &_motionMode,
                         std::vector<int> &_sequenceOrder,
                         double &_startDelay,
                         double &_duration,
                         double &_endHold,
                         double &_truthRate,
                         double &_linearSpeed,
                         double &_angularSpeed,
                         std::string &_error) const
  {
    if (!this->rosNode_)
    {
      _error = "ROS node is unavailable";
      return false;
    }
    if (!this->rosNode_->getParam("start_delay", _startDelay) ||
        !std::isfinite(_startDelay) || _startDelay < 0.0)
    {
      _error = "start_delay must be finite and non-negative";
      return false;
    }
    if (!this->rosNode_->getParam("motion_mode", _motionMode) ||
        (_motionMode != "sequential" && _motionMode != "simultaneous"))
    {
      _error = "motion_mode must be sequential or simultaneous";
      return false;
    }
    std::vector<int> parsedSequence;
    if (_motionMode == "sequential")
    {
      XmlRpc::XmlRpcValue sequence;
      if (!this->rosNode_->getParam("sequence_order_ids", sequence) ||
          sequence.getType() != XmlRpc::XmlRpcValue::TypeArray ||
          sequence.size() != kLastId)
      {
        _error = "sequential mode requires six sequence_order_ids";
        return false;
      }
      std::set<int> sequenceIds;
      for (int index = 0; index < sequence.size(); ++index)
      {
        int id = 0;
        if (!XmlInteger(sequence[index], id) || id < kFirstId ||
            id > kLastId || !sequenceIds.insert(id).second)
        {
          _error = "sequence_order_ids must be a permutation of 1 through 6";
          return false;
        }
        parsedSequence.push_back(id);
      }
    }
    if (!this->rosNode_->getParam("duration", _duration) ||
        !std::isfinite(_duration) || _duration <= 0.0)
    {
      _error = "duration must be a finite positive number";
      return false;
    }
    if (!this->rosNode_->getParam("end_hold", _endHold) ||
        !std::isfinite(_endHold) || _endHold < 0.0)
    {
      _error = "end_hold must be finite and non-negative";
      return false;
    }
    if (!this->rosNode_->getParam("truth_rate_hz", _truthRate) ||
        !std::isfinite(_truthRate) || _truthRate <= 0.0)
    {
      _error = "truth_rate_hz must be a finite positive number";
      return false;
    }
    if (!this->rosNode_->getParam("linear_speed_mm_s", _linearSpeed) ||
        !AllowedLinearSpeed(_linearSpeed))
    {
      _error = "linear_speed_mm_s must be one of 1, 3, 5, 10";
      return false;
    }
    if (!this->rosNode_->getParam("angular_speed_rad_s", _angularSpeed) ||
        !AllowedAngularSpeed(_angularSpeed))
    {
      _error = "angular_speed_rad_s must be 0.0015 or 0.0025";
      return false;
    }

    XmlRpc::XmlRpcValue models;
    if (!this->rosNode_->getParam("models", models) ||
        models.getType() != XmlRpc::XmlRpcValue::TypeArray)
    {
      _error = "models must be a YAML list";
      return false;
    }

    std::map<int, MotionCommand> parsed;
    for (int index = 0; index < models.size(); ++index)
    {
      XmlRpc::XmlRpcValue &item = models[index];
      if (item.getType() != XmlRpc::XmlRpcValue::TypeStruct ||
          !item.hasMember("id") || !item.hasMember("model_name") ||
          !item.hasMember("command_frame") ||
          !item.hasMember("linear_direction") ||
          !item.hasMember("angular_axis") ||
          !item.hasMember("surface_points"))
      {
        _error = "each model needs id, model_name, command_frame, "
                 "linear_direction, angular_axis and surface_points";
        return false;
      }

      MotionCommand command;
      if (!XmlInteger(item["id"], command.id) ||
          command.id < kFirstId || command.id > kLastId ||
          parsed.count(command.id) != 0)
      {
        _error = "model IDs must be unique integers 1 through 6";
        return false;
      }
      if (item["model_name"].getType() !=
              XmlRpc::XmlRpcValue::TypeString ||
          item["command_frame"].getType() !=
              XmlRpc::XmlRpcValue::TypeString)
      {
        _error = "model_name and command_frame must be strings";
        return false;
      }
      command.modelName = static_cast<std::string>(item["model_name"]);
      const std::string frame =
          static_cast<std::string>(item["command_frame"]);
      if (frame != "world")
      {
        _error = "ICRA2027 commands must use fixed world axes";
        return false;
      }

      ignition::math::Vector3d linearDirection;
      ignition::math::Vector3d angularAxis;
      if (!XmlVector3(item["linear_direction"], linearDirection) ||
          !XmlVector3(item["angular_axis"], angularAxis) ||
          !XmlVector3List(item["surface_points"], command.surfacePoints))
      {
        _error = "invalid direction, axis or surface_points for ID " +
                 std::to_string(command.id);
        return false;
      }

      const std::string expectedName = "m0" + std::to_string(command.id);
      if (command.modelName != expectedName)
      {
        _error = "configured model name does not match ICRA2027 truth ID "
                 "for ID " + std::to_string(command.id);
        return false;
      }
      const bool translates =
          std::fabs(linearDirection.Length() - 1.0) <= kTolerance &&
          angularAxis.Length() <= kTolerance;
      const bool rotates =
          linearDirection.Length() <= kTolerance &&
          std::fabs(angularAxis.Length() - 1.0) <= kTolerance;
      if (!translates && !rotates)
      {
        _error = "each motion must define exactly one unit translation or "
                 "unit rotation axis for ID " + std::to_string(command.id);
        return false;
      }
      if (rotates && command.surfacePoints.empty())
      {
        _error = "rotating ID " + std::to_string(command.id) +
                 " needs surface truth points";
        return false;
      }
      if (!rotates && !command.surfacePoints.empty())
      {
        _error = "translating ID " + std::to_string(command.id) +
                 " must not define unnecessary surface points";
        return false;
      }

      command.linear = linearDirection *
          (_linearSpeed * kMillimetersToMeters);
      command.angular = angularAxis * _angularSpeed;
      const ModelHandle &handle = this->controlledModels_.at(command.id);
      if (command.modelName != handle.model->GetName())
      {
        _error = "ID " + std::to_string(command.id) + " resolves to " +
                 handle.model->GetName() + ", not " + command.modelName;
        return false;
      }
      parsed[command.id] = command;
    }
    if (parsed.size() != static_cast<std::size_t>(kLastId))
    {
      _error = "models must contain every ID 1 through 6";
      return false;
    }
    _commands = std::move(parsed);
    _sequenceOrder = std::move(parsedSequence);
    return true;
  }

  bool Reload(std_srvs::Trigger::Request &,
              std_srvs::Trigger::Response &_response)
  {
    std::lock_guard<std::mutex> lock(this->mutex_);
    if (!this->environmentValid_)
    {
      _response.success = false;
      _response.message = this->lastError_;
      return true;
    }
    if (this->state_ == State::Waiting || this->state_ == State::Running ||
        this->state_ == State::Holding)
    {
      _response.success = false;
      _response.message = "stop the active plan before reloading";
      return true;
    }

    std::map<int, MotionCommand> parsed;
    std::string error;
    if (!this->ReadConfiguration(parsed, this->motionMode_,
                                 this->sequenceOrder_, this->startDelay_,
                                 this->duration_,
                                 this->endHold_, this->truthRateHz_,
                                 this->linearSpeedMmS_,
                                 this->angularSpeedRadS_, error))
    {
      this->lastError_ = error;
      this->state_ = State::Error;
      _response.success = false;
      _response.message = error;
      this->PublishStatus();
      return true;
    }

    this->commands_ = std::move(parsed);
    this->state_ = State::Armed;
    this->ClearRunState();
    this->lastError_.clear();
    _response.success = true;
    _response.message = this->BuildStatusJson();
    this->PublishStatus();
    return true;
  }

  bool Start(std_srvs::Trigger::Request &,
             std_srvs::Trigger::Response &_response)
  {
    std::lock_guard<std::mutex> lock(this->mutex_);
    if (this->state_ != State::Armed && this->state_ != State::Completed &&
        this->state_ != State::Stopped)
    {
      _response.success = false;
      _response.message = "reload a valid plan before start";
      return true;
    }
    this->startRequested_ = true;
    this->stopRequested_ = false;
    _response.success = true;
    _response.message = "start queued for one atomic Gazebo update";
    return true;
  }

  bool Stop(std_srvs::Trigger::Request &,
            std_srvs::Trigger::Response &_response)
  {
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->stopRequested_ = true;
    this->startRequested_ = false;
    _response.success = true;
    _response.message = "stop queued for one atomic Gazebo update";
    return true;
  }

  bool Reset(std_srvs::Trigger::Request &,
             std_srvs::Trigger::Response &_response)
  {
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->resetRequested_ = true;
    this->startRequested_ = false;
    this->stopRequested_ = false;
    _response.success = true;
    _response.message = "scene reset queued for one atomic Gazebo update";
    return true;
  }

  bool GetStatus(std_srvs::Trigger::Request &,
                 std_srvs::Trigger::Response &_response)
  {
    std::lock_guard<std::mutex> lock(this->mutex_);
    _response.success = this->environmentValid_ && this->state_ != State::Error
        && this->PhysicsOwnershipValid();
    _response.message = this->BuildStatusJson();
    return true;
  }

  void ClearRunState()
  {
    this->startRequested_ = false;
    this->stopRequested_ = false;
    this->resetRequested_ = false;
    this->truthCaptured_ = false;
    this->runInitialPoses_.clear();
    this->runInitialLinkPoses_.clear();
    this->experimentStartTime_ = std::numeric_limits<double>::quiet_NaN();
    this->scheduledStartTime_ = std::numeric_limits<double>::quiet_NaN();
    this->actualStartTime_ = std::numeric_limits<double>::quiet_NaN();
    this->stopTime_ = std::numeric_limits<double>::quiet_NaN();
    this->holdStartTime_ = std::numeric_limits<double>::quiet_NaN();
    this->lastTruthPublishTime_ = std::numeric_limits<double>::quiet_NaN();
    this->lastStatusPublishTime_ = std::numeric_limits<double>::quiet_NaN();
  }

  void CaptureRunTruth()
  {
    this->runInitialPoses_.clear();
    this->runInitialLinkPoses_.clear();
    for (const auto &entry : this->controlledModels_)
    {
      this->runInitialPoses_[entry.first] = entry.second.model->WorldPose();
      this->runInitialLinkPoses_[entry.first] =
          entry.second.driveLink->WorldPose();
    }
    this->truthCaptured_ = true;
  }

  void RestoreSceneInitialPoses()
  {
    this->ZeroControlledModels();
    for (const auto &entry : this->controlledModels_)
    {
      entry.second.model->SetWorldPose(this->sceneInitialPoses_.at(entry.first));
      // A kinematic link is the ray-collision owner.  Setting only the model
      // pose can leave its collision broadphase at a different pose in ODE,
      // even though Model::WorldPose reports the requested truth.  Restore
      // both explicitly so reset truth and LiDAR geometry are identical.
      entry.second.driveLink->SetWorldPose(
          this->sceneInitialLinkPoses_.at(entry.first));
    }
    this->ApplyPhysicsOwnership();
    this->truthCaptured_ = false;
    this->runInitialPoses_.clear();
    this->runInitialLinkPoses_.clear();
  }

  double ExperimentElapsed(const double _now) const
  {
    if (!std::isfinite(this->experimentStartTime_))
      return 0.0;
    const double endTime = std::isfinite(this->stopTime_) ? this->stopTime_ : _now;
    return std::max(0.0, endTime - this->experimentStartTime_);
  }

  void OnUpdate(const common::UpdateInfo &_info)
  {
    std::lock_guard<std::mutex> lock(this->mutex_);
    const double now = _info.simTime.Double();
    if (!std::isfinite(now) || !this->environmentValid_)
      return;

    if (std::isfinite(this->lastSimTime_) && now < this->lastSimTime_)
    {
      this->ZeroControlledModels();
      this->state_ = this->commands_.empty() ? State::Unconfigured : State::Armed;
      this->ClearRunState();
      this->lastError_ =
          "simulation time moved backwards; run reset/reload before restart";
      this->PublishStatus();
    }
    this->lastSimTime_ = now;

    if (this->resetRequested_)
    {
      this->resetRequested_ = false;
      this->RestoreSceneInitialPoses();
      ++this->resetCount_;
      this->state_ = this->commands_.empty() ? State::Unconfigured : State::Armed;
      this->ClearRunState();
      this->PublishStatus();
      return;
    }
    if (this->stopRequested_)
    {
      this->stopRequested_ = false;
      this->startRequested_ = false;
      this->ZeroControlledModels();
      if (this->truthCaptured_ && !std::isfinite(this->stopTime_))
        this->stopTime_ = now;
      this->state_ = State::Stopped;
      this->PublishTruth(now, true);
      this->PublishStatus();
      return;
    }
    if (this->startRequested_)
    {
      this->startRequested_ = false;
      if (!this->truthCaptured_)
        this->CaptureRunTruth();
      this->experimentStartTime_ = now;
      this->scheduledStartTime_ = now + this->startDelay_;
      this->actualStartTime_ = std::numeric_limits<double>::quiet_NaN();
      this->stopTime_ = std::numeric_limits<double>::quiet_NaN();
      this->holdStartTime_ = std::numeric_limits<double>::quiet_NaN();
      this->state_ = State::Waiting;
      this->PublishTruth(now, true);
      this->PublishStatus();
    }

    if (this->state_ == State::Waiting)
    {
      this->ZeroControlledModels();
      this->PublishTruth(now, false);
      if (now + 1.0e-12 < this->scheduledStartTime_)
        return;
      this->actualStartTime_ = now;
      this->state_ = State::Running;
      this->PublishTruth(now, true);
      this->PublishStatus();
    }

    if (this->state_ == State::Running)
    {
      const double elapsed = now - this->actualStartTime_;
      if (elapsed >= this->TotalMotionDuration() - 1.0e-12)
      {
        this->ApplyCommands(this->TotalMotionDuration());
        this->ZeroControlledModels();
        this->state_ = this->endHold_ > 0.0 ? State::Holding : State::Completed;
        if (this->state_ == State::Holding)
          this->holdStartTime_ = now;
        this->PublishTruth(now, true);
        this->PublishStatus();
        return;
      }
      this->ApplyCommands(elapsed);
      this->PublishTruth(now, false);
      return;
    }

    if (this->state_ == State::Holding)
    {
      this->ZeroControlledModels();
      this->PublishTruth(now, false);
      if (now - this->holdStartTime_ >= this->endHold_ - 1.0e-12)
      {
        this->state_ = State::Completed;
        this->PublishTruth(now, true);
        this->PublishStatus();
      }
      return;
    }
    this->ZeroControlledModels();
  }

  void ApplyCommands(const double _elapsed)
  {
    for (const auto &entry : this->commands_)
    {
      const MotionCommand &command = entry.second;
      const ModelHandle &handle = this->controlledModels_.at(command.id);
      const ignition::math::Pose3d &initial =
          this->runInitialPoses_.at(command.id);
      const ignition::math::Pose3d &initialLink =
          this->runInitialLinkPoses_.at(command.id);
      ignition::math::Quaterniond orientation = initial.Rot();
      ignition::math::Quaterniond linkOrientation = initialLink.Rot();
      const double modelElapsed = this->ModelMotionElapsed(command.id, _elapsed);
      const double angularSpeed = command.angular.Length();
      ignition::math::Quaterniond worldRotation;
      if (angularSpeed > kTolerance)
      {
        worldRotation = ignition::math::Quaterniond(
            command.angular / angularSpeed, angularSpeed * modelElapsed);
        orientation = worldRotation * initial.Rot();
        orientation.Normalize();
        linkOrientation = worldRotation * initialLink.Rot();
        linkOrientation.Normalize();
      }
      const ignition::math::Vector3d translation =
          command.linear * modelElapsed;
      const ignition::math::Vector3d initialLinkOffset =
          initialLink.Pos() - initial.Pos();
      const ignition::math::Vector3d linkOffset = angularSpeed > kTolerance
          ? worldRotation.RotateVector(initialLinkOffset)
          : initialLinkOffset;
      handle.model->SetWorldPose(ignition::math::Pose3d(
          initial.Pos() + translation, orientation));
      // The drive link owns every target collision and visual.  Commanding it
      // explicitly is the sensor-ground-truth contract: LiDAR rays must see
      // the same displacement that the plugin publishes.
      handle.driveLink->SetWorldPose(ignition::math::Pose3d(
          initial.Pos() + translation + linkOffset, linkOrientation));
    }
  }

  double TotalMotionDuration() const
  {
    return this->motionMode_ == "sequential"
        ? this->duration_ * static_cast<double>(this->sequenceOrder_.size())
        : this->duration_;
  }

  double ModelMotionElapsed(const int _id, const double _globalElapsed) const
  {
    if (this->motionMode_ != "sequential")
      return std::max(0.0, std::min(this->duration_, _globalElapsed));
    const auto iterator = std::find(
        this->sequenceOrder_.begin(), this->sequenceOrder_.end(), _id);
    if (iterator == this->sequenceOrder_.end())
      return 0.0;
    const double start = this->duration_ * static_cast<double>(
        std::distance(this->sequenceOrder_.begin(), iterator));
    return std::max(0.0, std::min(this->duration_, _globalElapsed - start));
  }

  void ZeroControlledModels()
  {
    const ignition::math::Vector3d zero = ignition::math::Vector3d::Zero;
    for (auto &entry : this->controlledModels_)
    {
      const ModelHandle &handle = entry.second;
      handle.driveLink->SetForce(zero);
      handle.driveLink->SetTorque(zero);
      handle.driveLink->SetLinearVel(zero);
      handle.driveLink->SetAngularVel(zero);
    }
  }

  bool PhysicsOwnershipValid() const
  {
    if (this->controlledModels_.size() != static_cast<std::size_t>(kLastId))
      return false;
    for (const auto &entry : this->controlledModels_)
    {
      const auto &link = entry.second.driveLink;
      if (!link->GetKinematic() || link->GetGravityMode())
        return false;
    }
    return true;
  }

  std::string StateName() const
  {
    switch (this->state_)
    {
      case State::Unconfigured: return "UNCONFIGURED";
      case State::Armed: return "ARMED";
      case State::Waiting: return "WAITING";
      case State::Running: return "RUNNING";
      case State::Holding: return "HOLDING";
      case State::Completed: return "COMPLETED";
      case State::Stopped: return "STOPPED";
      case State::Error: return "ERROR";
    }
    return "ERROR";
  }

  double MotionElapsed(const double _now) const
  {
    if (!std::isfinite(this->actualStartTime_))
      return 0.0;
    const double endTime = std::isfinite(this->stopTime_) ? this->stopTime_ : _now;
    return std::max(0.0, std::min(this->TotalMotionDuration(),
        endTime - this->actualStartTime_));
  }

  void PublishTruth(const double _now, const bool _force)
  {
    // The online progress stream has its own fixed cadence and clock.  It is
    // invoked here because this method is reached on every active update, but
    // neither its data nor its rate is derived from the recorder-only truth
    // payload/rate below.
    this->PublishStatus(false);
    if (!this->truthPublisher_ || !this->truthCaptured_)
      return;
    const double interval = 1.0 / this->truthRateHz_;
    if (!_force && std::isfinite(this->lastTruthPublishTime_) &&
        _now - this->lastTruthPublishTime_ < interval - 1.0e-12)
    {
      return;
    }
    this->lastTruthPublishTime_ = _now;
    const double experimentElapsed = this->ExperimentElapsed(_now);
    const double elapsed = this->MotionElapsed(_now);
    const double waitingElapsed =
        std::max(0.0, std::min(this->startDelay_, experimentElapsed));

    std::ostringstream output;
    output.precision(12);
    output << "{\"schema_version\":2,\"state\":\""
           << this->StateName() << "\",\"sim_time\":" << _now
           << ",\"experiment_elapsed\":" << experimentElapsed
           << ",\"waiting_elapsed\":" << waitingElapsed
           << ",\"motion_elapsed\":" << elapsed
           << ",\"motion_mode\":\"" << this->motionMode_ << "\""
           << ",\"per_model_duration\":" << this->duration_
           << ",\"linear_speed_mm_s\":" << this->linearSpeedMmS_
           << ",\"angular_speed_rad_s\":" << this->angularSpeedRadS_
           << ",\"models\":[";

    bool first = true;
    for (const auto &entry : this->commands_)
    {
      const int id = entry.first;
      const MotionCommand &command = entry.second;
      const ignition::math::Pose3d initial =
          this->runInitialLinkPoses_.at(id);
      const ignition::math::Pose3d current =
          this->controlledModels_.at(id).driveLink->WorldPose();
      // Evaluation-only world AABB of the same collision-bearing drive link
      // whose pose defines the displacement truth.  It is published after
      // the atomic motion update and is never consumed by the detector.
      const ignition::math::AxisAlignedBox worldBounds =
          this->controlledModels_.at(id).driveLink->BoundingBox();
      const ignition::math::Vector3d translation =
          current.Pos() - initial.Pos();
      const double translationMagnitude = translation.Length();
      const double rotation = QuaternionDistance(initial.Rot(), current.Rot());
      const double modelElapsed = this->ModelMotionElapsed(id, elapsed);
      const double expectedTranslation = command.linear.Length() * modelElapsed;
      const double expectedRotation = command.angular.Length() * modelElapsed;
      const ignition::math::Vector3d expectedTranslationVector =
          command.linear * modelElapsed;

      double surfaceRms = translationMagnitude;
      double surfaceMax = translationMagnitude;
      if (!command.surfacePoints.empty())
      {
        double squaredSum = 0.0;
        surfaceMax = 0.0;
        for (const ignition::math::Vector3d &point : command.surfacePoints)
        {
          const double displacement =
              (TransformPoint(current, point) -
               TransformPoint(initial, point)).Length();
          squaredSum += displacement * displacement;
          surfaceMax = std::max(surfaceMax, displacement);
        }
        surfaceRms = std::sqrt(
            squaredSum / static_cast<double>(command.surfacePoints.size()));
      }

      if (!first)
        output << ',';
      first = false;
      output << "{\"laser_retro_id\":" << id
             << ",\"model_name\":\"" << JsonEscape(command.modelName) << "\""
             << ",\"motion_elapsed\":" << modelElapsed
             << ",\"position\":[" << current.Pos().X() << ','
             << current.Pos().Y() << ',' << current.Pos().Z() << ']'
             << ",\"orientation\":[" << current.Rot().X() << ','
             << current.Rot().Y() << ',' << current.Rot().Z() << ','
             << current.Rot().W() << ']'
             << ",\"aabb_min\":[" << worldBounds.Min().X() << ','
             << worldBounds.Min().Y() << ',' << worldBounds.Min().Z() << ']'
             << ",\"aabb_max\":[" << worldBounds.Max().X() << ','
             << worldBounds.Max().Y() << ',' << worldBounds.Max().Z() << ']'
             << ",\"translation\":[" << translation.X() << ','
             << translation.Y() << ',' << translation.Z() << ']'
             << ",\"expected_translation_vector\":["
             << expectedTranslationVector.X() << ','
             << expectedTranslationVector.Y() << ','
             << expectedTranslationVector.Z() << ']'
             << ",\"translation_magnitude\":" << translationMagnitude
             << ",\"translation_expected\":" << expectedTranslation
             << ",\"translation_error\":"
             << std::fabs(translationMagnitude - expectedTranslation)
             << ",\"rotation_angle\":" << rotation
             << ",\"rotation_expected\":" << expectedRotation
             << ",\"rotation_error\":"
             << std::fabs(rotation - expectedRotation)
             << ",\"surface_rms_displacement\":" << surfaceRms
             << ",\"surface_max_displacement\":" << surfaceMax
             << ",\"surface_point_count\":"
             << command.surfacePoints.size() << '}';
    }
    output << "]}";
    std_msgs::String message;
    message.data = output.str();
    this->truthPublisher_.publish(message);
  }

  std::string BuildStatusJson() const
  {
    std::ostringstream output;
    output.precision(12);
    output << "{\"state\":\"" << this->StateName() << "\""
           << ",\"environment_valid\":"
           << (this->environmentValid_ ? "true" : "false")
           << ",\"sim_time\":";
    if (std::isfinite(this->lastSimTime_))
      output << this->lastSimTime_;
    else
      output << "null";
    output << ",\"experiment_elapsed\":"
           << this->ExperimentElapsed(this->lastSimTime_)
           << ",\"motion_elapsed\":"
           << this->MotionElapsed(this->lastSimTime_);
    output << ",\"motion_mode\":\"" << this->motionMode_ << "\""
           << ",\"scheduled_start_time\":";
    if (std::isfinite(this->scheduledStartTime_))
      output << this->scheduledStartTime_;
    else
      output << "null";
    output << ",\"actual_start_time\":";
    if (std::isfinite(this->actualStartTime_))
      output << this->actualStartTime_;
    else
      output << "null";
    output << ",\"start_delay\":" << this->startDelay_
           << ",\"per_model_duration\":" << this->duration_
           << ",\"total_motion_duration\":" << this->TotalMotionDuration()
           << ",\"end_hold\":" << this->endHold_
           << ",\"linear_speed_mm_s\":" << this->linearSpeedMmS_
           << ",\"angular_speed_rad_s\":" << this->angularSpeedRadS_
           << ",\"reset_count\":" << this->resetCount_
           << ",\"physics_ownership_valid\":"
           << (this->PhysicsOwnershipValid() ? "true" : "false")
           << ",\"last_error\":\"" << JsonEscape(this->lastError_)
           << "\"}";
    return output.str();
  }

  void PublishStatus(const bool _force = true)
  {
    if (!this->statusPublisher_)
      return;
    constexpr double kStatusIntervalSeconds = 0.1;
    const double now = this->lastSimTime_;
    if (!_force && std::isfinite(now)
        && std::isfinite(this->lastStatusPublishTime_)
        && now - this->lastStatusPublishTime_
            < kStatusIntervalSeconds - 1.0e-12)
    {
      return;
    }
    if (std::isfinite(now))
      this->lastStatusPublishTime_ = now;
    std_msgs::String message;
    message.data = this->BuildStatusJson();
    this->statusPublisher_.publish(message);
  }

  physics::WorldPtr world_;
  event::ConnectionPtr updateConnection_;
  std::map<int, ModelHandle> controlledModels_;
  std::map<int, ignition::math::Pose3d> sceneInitialPoses_;
  std::map<int, ignition::math::Pose3d> sceneInitialLinkPoses_;
  std::map<int, ignition::math::Pose3d> runInitialPoses_;
  std::map<int, ignition::math::Pose3d> runInitialLinkPoses_;
  std::map<int, MotionCommand> commands_;

  std::string rosNamespace_ = "/icra2027_motion";
  std::unique_ptr<ros::NodeHandle> rosNode_;
  ros::CallbackQueue callbackQueue_;
  ros::ServiceServer reloadService_;
  ros::ServiceServer startService_;
  ros::ServiceServer stopService_;
  ros::ServiceServer resetService_;
  ros::ServiceServer statusService_;
  ros::Publisher statusPublisher_;
  ros::Publisher truthPublisher_;
  std::thread callbackThread_;
  std::atomic<bool> alive_{false};
  mutable std::mutex mutex_;

  State state_ = State::Unconfigured;
  bool environmentValid_ = false;
  bool startRequested_ = false;
  bool stopRequested_ = false;
  bool resetRequested_ = false;
  bool truthCaptured_ = false;
  std::uint64_t resetCount_ = 0;
  double startDelay_ = 0.0;
  std::string motionMode_ = "simultaneous";
  std::vector<int> sequenceOrder_{1, 2, 3, 4, 5, 6};
  double duration_ = 0.0;
  double endHold_ = 0.0;
  double truthRateHz_ = 10.0;
  double linearSpeedMmS_ = 0.0;
  double angularSpeedRadS_ = 0.0;
  double lastSimTime_ = std::numeric_limits<double>::quiet_NaN();
  double experimentStartTime_ = std::numeric_limits<double>::quiet_NaN();
  double scheduledStartTime_ = std::numeric_limits<double>::quiet_NaN();
  double actualStartTime_ = std::numeric_limits<double>::quiet_NaN();
  double stopTime_ = std::numeric_limits<double>::quiet_NaN();
  double holdStartTime_ = std::numeric_limits<double>::quiet_NaN();
  double lastTruthPublishTime_ = std::numeric_limits<double>::quiet_NaN();
  double lastStatusPublishTime_ = std::numeric_limits<double>::quiet_NaN();
  std::string lastError_;
};

GZ_REGISTER_WORLD_PLUGIN(Icra2027MotionWorldPlugin)
}  // namespace gazebo
