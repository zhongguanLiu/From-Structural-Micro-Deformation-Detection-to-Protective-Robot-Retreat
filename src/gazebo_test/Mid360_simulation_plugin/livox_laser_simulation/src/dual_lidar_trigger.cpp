#include "livox_laser_simulation/dual_lidar_trigger.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace livox_laser_simulation {
namespace {

void SetError(const std::string& value, std::string* error) {
  if (error) *error = value;
}

}  // namespace

bool ParseDualLidarTriggerRole(const std::string& value,
                               DualLidarTriggerRole* role) {
  if (!role) return false;
  if (value == "primary") {
    *role = DualLidarTriggerRole::PRIMARY;
    return true;
  }
  if (value == "secondary") {
    *role = DualLidarTriggerRole::SECONDARY;
    return true;
  }
  return false;
}

bool DualLidarTriggerRegistry::Register(
    const std::string& group, DualLidarTriggerRole role, Callback callback,
    uint64_t* token, std::string* error) {
  if (group.empty()) {
    SetError("dual-LiDAR synchronization group is empty", error);
    return false;
  }
  if (!callback || !token) {
    SetError("dual-LiDAR synchronization registration is incomplete", error);
    return false;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  Group& state = groups_[group];
  uint64_t* slot = role == DualLidarTriggerRole::PRIMARY
      ? &state.primary_token : &state.secondary_token;
  if (*slot != 0U) {
    SetError("duplicate role in dual-LiDAR synchronization group", error);
    return false;
  }
  if (next_token_ == 0U ||
      next_token_ == std::numeric_limits<uint64_t>::max()) {
    SetError("dual-LiDAR synchronization token space exhausted", error);
    return false;
  }
  const uint64_t assigned = next_token_++;
  registrations_.emplace(
      assigned, Registration{group, role, std::move(callback)});
  *slot = assigned;
  *token = assigned;
  return true;
}

void DualLidarTriggerRegistry::Unregister(uint64_t token) {
  if (token == 0U) return;
  std::lock_guard<std::mutex> lock(mutex_);
  const auto registration = registrations_.find(token);
  if (registration == registrations_.end()) return;
  const std::string group_name = registration->second.group;
  const auto group = groups_.find(group_name);
  if (group != groups_.end()) {
    if (group->second.primary_token == token)
      group->second.primary_token = 0U;
    if (group->second.secondary_token == token)
      group->second.secondary_token = 0U;
    if (group->second.primary_token == 0U &&
        group->second.secondary_token == 0U)
      groups_.erase(group);
  }
  registrations_.erase(registration);
}

bool DualLidarTriggerRegistry::GroupReady(const std::string& group) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto found = groups_.find(group);
  return found != groups_.end() && found->second.primary_token != 0U &&
         found->second.secondary_token != 0U;
}

bool DualLidarTriggerRegistry::TriggerPair(
    uint64_t primary_token, uint64_t acquisition_time_ns,
    std::string* error) {
  return TriggerPairWithMinimumInterval(
             primary_token, acquisition_time_ns, 0U, nullptr, error) ==
         DualLidarTriggerResult::TRIGGERED;
}

DualLidarTriggerResult
DualLidarTriggerRegistry::TriggerPairWithMinimumInterval(
    uint64_t primary_token, uint64_t acquisition_time_ns,
    uint64_t minimum_interval_ns, uint64_t* catch_up_suppressed_count,
    std::string* error) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto primary = registrations_.find(primary_token);
  if (primary == registrations_.end() ||
      primary->second.role != DualLidarTriggerRole::PRIMARY ||
      !primary->second.callback) {
    SetError("dual-LiDAR trigger caller is not a registered primary", error);
    return DualLidarTriggerResult::ERROR;
  }
  const auto group = groups_.find(primary->second.group);
  if (group == groups_.end() || group->second.secondary_token == 0U) {
    SetError("dual-LiDAR secondary is not registered", error);
    return DualLidarTriggerResult::ERROR;
  }
  const auto secondary = registrations_.find(group->second.secondary_token);
  if (secondary == registrations_.end() || !secondary->second.callback) {
    SetError("dual-LiDAR secondary callback is unavailable", error);
    return DualLidarTriggerResult::ERROR;
  }
  if (group->second.has_last_acquisition_time &&
      acquisition_time_ns <= group->second.last_acquisition_time_ns) {
    SetError("dual-LiDAR acquisition timestamp is not strictly increasing",
             error);
    return DualLidarTriggerResult::ERROR;
  }
  if (group->second.has_last_acquisition_time && minimum_interval_ns > 0U &&
      acquisition_time_ns - group->second.last_acquisition_time_ns <
          minimum_interval_ns) {
    if (group->second.scheduler_catch_up_suppressed_count <
        std::numeric_limits<uint64_t>::max()) {
      ++group->second.scheduler_catch_up_suppressed_count;
    }
    if (catch_up_suppressed_count) {
      *catch_up_suppressed_count =
          group->second.scheduler_catch_up_suppressed_count;
    }
    SetError("Gazebo scheduler catch-up epoch is earlier than the configured "
             "dual-LiDAR acquisition period", error);
    return DualLidarTriggerResult::SCHEDULER_CATCH_UP_SUPPRESSED;
  }

  // Reserve the epoch before invoking either publisher.  If a callback fails
  // the run is already invalid and this epoch must never be replayed into a
  // duplicate primary message.  Holding mutex_ makes the pair one ordered
  // transaction with respect to every overlapping primary callback.
  group->second.has_last_acquisition_time = true;
  group->second.last_acquisition_time_ns = acquisition_time_ns;
  try {
    if (!primary->second.callback(acquisition_time_ns)) {
      SetError("dual-LiDAR primary rejected the shared acquisition", error);
      return DualLidarTriggerResult::ERROR;
    }
    if (!secondary->second.callback(acquisition_time_ns)) {
      SetError("dual-LiDAR secondary rejected the shared acquisition", error);
      return DualLidarTriggerResult::ERROR;
    }
  } catch (const std::exception& exception) {
    SetError(std::string("dual-LiDAR paired callback failed: ") +
             exception.what(), error);
    return DualLidarTriggerResult::ERROR;
  } catch (...) {
    SetError("dual-LiDAR paired callback failed", error);
    return DualLidarTriggerResult::ERROR;
  }
  if (catch_up_suppressed_count) {
    *catch_up_suppressed_count =
        group->second.scheduler_catch_up_suppressed_count;
  }
  return DualLidarTriggerResult::TRIGGERED;
}

DualLidarTriggerRegistry& GlobalDualLidarTriggerRegistry() {
  static DualLidarTriggerRegistry registry;
  return registry;
}

}  // namespace livox_laser_simulation
