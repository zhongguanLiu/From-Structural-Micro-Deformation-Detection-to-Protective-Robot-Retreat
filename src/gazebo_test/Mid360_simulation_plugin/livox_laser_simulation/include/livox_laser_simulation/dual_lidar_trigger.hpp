#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>

namespace livox_laser_simulation {

enum class DualLidarTriggerRole {
  PRIMARY,
  SECONDARY,
};

// A scheduler catch-up callback is not a physical acquisition.  In
// particular, a saved Gazebo world can resume at a large simulation time and
// briefly emit callbacks closer together than the configured sensor period.
// Keep this result distinct from both a published pair and a real error so a
// caller can report the suppressed scheduler event without inventing a scan
// drop in the merger.
enum class DualLidarTriggerResult {
  TRIGGERED,
  SCHEDULER_CATCH_UP_SUPPRESSED,
  ERROR,
};

bool ParseDualLidarTriggerRole(const std::string& value,
                               DualLidarTriggerRole* role);

// Thread-safe registry used only inside one gzserver process.  The primary
// RaySensor owns the 10 Hz acquisition event; its callback evaluates both
// custom ray tables at one immutable simulation timestamp.  Keeping the
// secondary callback behind this registry prevents the two CPU-heavy Gazebo
// RaySensors from developing independent update phases under load.
class DualLidarTriggerRegistry {
 public:
  using Callback = std::function<bool(uint64_t acquisition_time_ns)>;

  bool Register(const std::string& group, DualLidarTriggerRole role,
                Callback callback, uint64_t* token, std::string* error);
  void Unregister(uint64_t token);
  bool GroupReady(const std::string& group) const;
  // Atomically invokes primary then secondary for one strictly increasing
  // epoch.  The registry mutex remains held across both callbacks so two
  // overlapping primary sensor callbacks cannot interleave their messages.
  bool TriggerPair(uint64_t primary_token, uint64_t acquisition_time_ns,
                   std::string* error);
  // Same atomic transaction with an accepted-epoch cadence gate.  The first
  // complete pair is always triggered.  Thereafter a strictly increasing
  // scheduler epoch less than minimum_interval_ns after the last triggered
  // pair is suppressed before either callback.  The original timestamp is
  // never changed, and suppressed epochs are not merger messages or drops.
  DualLidarTriggerResult TriggerPairWithMinimumInterval(
      uint64_t primary_token, uint64_t acquisition_time_ns,
      uint64_t minimum_interval_ns, uint64_t* catch_up_suppressed_count,
      std::string* error);

 private:
  struct Registration {
    std::string group;
    DualLidarTriggerRole role = DualLidarTriggerRole::PRIMARY;
    Callback callback;
  };

  struct Group {
    uint64_t primary_token = 0U;
    uint64_t secondary_token = 0U;
    bool has_last_acquisition_time = false;
    uint64_t last_acquisition_time_ns = 0U;
    uint64_t scheduler_catch_up_suppressed_count = 0U;
  };

  mutable std::mutex mutex_;
  uint64_t next_token_ = 1U;
  std::map<uint64_t, Registration> registrations_;
  std::map<std::string, Group> groups_;
};

DualLidarTriggerRegistry& GlobalDualLidarTriggerRegistry();

}  // namespace livox_laser_simulation
