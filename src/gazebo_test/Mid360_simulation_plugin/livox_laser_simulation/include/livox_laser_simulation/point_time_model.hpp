#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <string>

namespace livox_laser_simulation {

// Gazebo evaluates every ray in LivoxPointsPlugin::OnNewLaserScans after one
// sensor update and at one simulation pose.  SNAPSHOT therefore describes the
// physically represented simulation measurement.  LEGACY_ROLLING retains the
// historical synthetic 200 kpoint/s offsets for older worlds that may depend
// on them.
enum class PointTimeMode {
  LEGACY_ROLLING = 0,
  SNAPSHOT = 1,
};

inline bool ParsePointTimeMode(const std::string& value,
                               PointTimeMode* mode) {
  if (!mode) return false;
  std::string normalized = value;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  if (normalized == "legacy_rolling" || normalized == "rolling") {
    *mode = PointTimeMode::LEGACY_ROLLING;
    return true;
  }
  if (normalized == "snapshot") {
    *mode = PointTimeMode::SNAPSHOT;
    return true;
  }
  return false;
}

inline const char* PointTimeModeName(PointTimeMode mode) {
  return mode == PointTimeMode::SNAPSHOT ? "snapshot" : "legacy_rolling";
}

inline uint32_t SimulatedPointOffsetNs(PointTimeMode mode,
                                       std::size_t point_index) {
  if (mode == PointTimeMode::SNAPSHOT) return 0U;
  // Preserve the plugin's historical 200 kpoint/s timing exactly.  The
  // configured 20,000-ray scan remains below the uint32 offset limit.
  return static_cast<uint32_t>(5000U * point_index);
}

}  // namespace livox_laser_simulation
