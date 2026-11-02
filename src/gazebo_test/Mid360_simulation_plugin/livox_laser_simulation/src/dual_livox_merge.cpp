#include "livox_laser_simulation/dual_livox_merge.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace livox_laser_simulation {
namespace {

void SaturatingIncrement(uint64_t* value) {
  if (*value != std::numeric_limits<uint64_t>::max()) ++(*value);
}

void SetError(const std::string& value, std::string* error) {
  if (error) *error = value;
}

bool FinitePoint(const livox_ros_driver::CustomPoint& point) {
  return std::isfinite(point.x) && std::isfinite(point.y) &&
         std::isfinite(point.z);
}

bool NoReturnSentinel(const livox_ros_driver::CustomPoint& point) {
  return point.x == 0.0F && point.y == 0.0F && point.z == 0.0F;
}

using VoxelKey = std::array<int64_t, 3>;

bool MakeVoxelKey(double x, double y, double z, double voxel_size_m,
                  VoxelKey* key, std::string* error) {
  const std::array<double, 3> coordinates = {x, y, z};
  for (std::size_t axis = 0U; axis < coordinates.size(); ++axis) {
    const long double scaled =
        static_cast<long double>(coordinates[axis]) / voxel_size_m;
    const long double floored = std::floor(scaled);
    if (!std::isfinite(scaled) ||
        floored < static_cast<long double>(std::numeric_limits<int64_t>::min()) ||
        floored > static_cast<long double>(std::numeric_limits<int64_t>::max())) {
      SetError("LiDAR return is outside diagnostic voxel index range", error);
      return false;
    }
    (*key)[axis] = static_cast<int64_t>(floored);
  }
  return true;
}

bool OccupiedVoxels(const livox_ros_driver::CustomMsg& input,
                    bool transform,
                    const DualLidarCalibration& calibration,
                    double voxel_size_m,
                    uint64_t* valid_return_count,
                    std::vector<VoxelKey>* voxels,
                    std::string* error) {
  uint64_t returns = 0U;
  voxels->clear();
  voxels->reserve(input.points.size());
  for (const auto& point : input.points) {
    if (!FinitePoint(point)) {
      SetError("non-finite Livox point in overlap diagnostic", error);
      return false;
    }
    if (NoReturnSentinel(point)) continue;
    double x = point.x;
    double y = point.y;
    double z = point.z;
    if (transform) {
      x = calibration.rotation_primary_secondary[0] * point.x +
          calibration.rotation_primary_secondary[1] * point.y +
          calibration.rotation_primary_secondary[2] * point.z +
          calibration.translation_primary_secondary[0];
      y = calibration.rotation_primary_secondary[3] * point.x +
          calibration.rotation_primary_secondary[4] * point.y +
          calibration.rotation_primary_secondary[5] * point.z +
          calibration.translation_primary_secondary[1];
      z = calibration.rotation_primary_secondary[6] * point.x +
          calibration.rotation_primary_secondary[7] * point.y +
          calibration.rotation_primary_secondary[8] * point.z +
          calibration.translation_primary_secondary[2];
    }
    VoxelKey key{};
    if (!MakeVoxelKey(x, y, z, voxel_size_m, &key, error)) return false;
    voxels->push_back(key);
    SaturatingIncrement(&returns);
  }
  std::sort(voxels->begin(), voxels->end());
  voxels->erase(std::unique(voxels->begin(), voxels->end()), voxels->end());
  *valid_return_count = returns;
  return true;
}

bool AppendWithTimebase(const livox_ros_driver::CustomMsg& input,
                        uint64_t output_timebase,
                        bool transform,
                        const DualLidarCalibration& calibration,
                        std::vector<livox_ros_driver::CustomPoint>* points,
                        std::string* error) {
  for (const livox_ros_driver::CustomPoint& source : input.points) {
    if (!FinitePoint(source)) {
      SetError("non-finite Livox point", error);
      return false;
    }
    if (input.timebase >
        std::numeric_limits<uint64_t>::max() - source.offset_time) {
      SetError("point absolute timestamp overflow", error);
      return false;
    }
    const uint64_t absolute = input.timebase + source.offset_time;
    if (absolute < output_timebase ||
        absolute - output_timebase >
            std::numeric_limits<uint32_t>::max()) {
      SetError("merged point offset does not fit uint32", error);
      return false;
    }
    // Gazebo emits exactly (0,0,0) for a ray without a return.  Keep the raw
    // source messages untouched, but do not copy this non-measurement into the
    // formal dual-LiDAR stream consumed by FAST-LIO.  Timestamp validation is
    // deliberately performed first so malformed source messages are not
    // hidden by compaction.
    if (NoReturnSentinel(source)) continue;
    livox_ros_driver::CustomPoint point = source;
    point.offset_time = static_cast<uint32_t>(absolute - output_timebase);
    if (transform) {
      const double x = calibration.rotation_primary_secondary[0] * source.x +
                       calibration.rotation_primary_secondary[1] * source.y +
                       calibration.rotation_primary_secondary[2] * source.z +
                       calibration.translation_primary_secondary[0];
      const double y = calibration.rotation_primary_secondary[3] * source.x +
                       calibration.rotation_primary_secondary[4] * source.y +
                       calibration.rotation_primary_secondary[5] * source.z +
                       calibration.translation_primary_secondary[1];
      const double z = calibration.rotation_primary_secondary[6] * source.x +
                       calibration.rotation_primary_secondary[7] * source.y +
                       calibration.rotation_primary_secondary[8] * source.z +
                       calibration.translation_primary_secondary[2];
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
          std::abs(x) > std::numeric_limits<float>::max() ||
          std::abs(y) > std::numeric_limits<float>::max() ||
          std::abs(z) > std::numeric_limits<float>::max()) {
        SetError("transformed Livox point is invalid", error);
        return false;
      }
      point.x = static_cast<float>(x);
      point.y = static_cast<float>(y);
      point.z = static_cast<float>(z);
    }
    points->push_back(point);
  }
  return true;
}

}  // namespace

uint64_t LivoxPointOffsetSpanNs(
    const livox_ros_driver::CustomMsg& message) {
  if (message.points.empty()) return 0U;
  uint32_t minimum = std::numeric_limits<uint32_t>::max();
  uint32_t maximum = 0U;
  for (const auto& point : message.points) {
    minimum = std::min(minimum, point.offset_time);
    maximum = std::max(maximum, point.offset_time);
  }
  return static_cast<uint64_t>(maximum) - minimum;
}

void DualLidarMergeStatistics::RecordPrimaryReceived(uint64_t point_count) {
  SaturatingIncrement(&snapshot_.primary_received);
  snapshot_.last_primary_point_count = point_count;
  snapshot_.maximum_primary_point_count =
      std::max(snapshot_.maximum_primary_point_count, point_count);
}

void DualLidarMergeStatistics::RecordSecondaryReceived(uint64_t point_count) {
  SaturatingIncrement(&snapshot_.secondary_received);
  snapshot_.last_secondary_point_count = point_count;
  snapshot_.maximum_secondary_point_count =
      std::max(snapshot_.maximum_secondary_point_count, point_count);
}

void DualLidarMergeStatistics::RecordPairAccepted(
                                                  uint64_t accepted_measurement_time_ns,
                                                  uint64_t sync_delta_ns,
                                                  uint64_t merged_span_ns,
                                                  const DualLivoxMergeAccounting& accounting) {
  SaturatingIncrement(&snapshot_.pairs_accepted);
  if (snapshot_.pairs_accepted == 1U) {
    snapshot_.first_accepted_measurement_time_ns =
        accepted_measurement_time_ns;
    snapshot_.last_accepted_measurement_time_ns =
        accepted_measurement_time_ns;
  } else if (accepted_measurement_time_ns >
             snapshot_.last_accepted_measurement_time_ns) {
    const uint64_t interval_ns = accepted_measurement_time_ns -
        snapshot_.last_accepted_measurement_time_ns;
    SaturatingIncrement(&snapshot_.accepted_measurement_interval_count);
    if (snapshot_.accepted_measurement_interval_count == 1U) {
      snapshot_.minimum_accepted_measurement_interval_ns = interval_ns;
      snapshot_.maximum_accepted_measurement_interval_ns = interval_ns;
    } else {
      snapshot_.minimum_accepted_measurement_interval_ns = std::min(
          snapshot_.minimum_accepted_measurement_interval_ns, interval_ns);
      snapshot_.maximum_accepted_measurement_interval_ns = std::max(
          snapshot_.maximum_accepted_measurement_interval_ns, interval_ns);
    }
    const double interval_count = static_cast<double>(
        snapshot_.accepted_measurement_interval_count);
    snapshot_.mean_accepted_measurement_interval_ns +=
        (static_cast<double>(interval_ns) -
         snapshot_.mean_accepted_measurement_interval_ns) / interval_count;
    snapshot_.last_accepted_measurement_time_ns =
        accepted_measurement_time_ns;
    const uint64_t elapsed_ns =
        snapshot_.last_accepted_measurement_time_ns -
        snapshot_.first_accepted_measurement_time_ns;
    snapshot_.realized_measurement_hz = elapsed_ns > 0U
        ? interval_count * 1.0e9 / static_cast<double>(elapsed_ns)
        : 0.0;
  } else {
    // Do not silently fold duplicate or reversed measurement epochs into a
    // positive average.  Preserve the last valid endpoint and expose a
    // dedicated violation that the formal contract rejects.
    SaturatingIncrement(
        &snapshot_.nonpositive_accepted_measurement_interval_count);
  }
  snapshot_.last_sync_delta_ns = sync_delta_ns;
  snapshot_.maximum_sync_delta_ns =
      std::max(snapshot_.maximum_sync_delta_ns, sync_delta_ns);
  const double count = static_cast<double>(snapshot_.pairs_accepted);
  snapshot_.mean_sync_delta_ns +=
      (static_cast<double>(sync_delta_ns) - snapshot_.mean_sync_delta_ns) /
      count;
  snapshot_.last_merged_span_ns = merged_span_ns;
  snapshot_.maximum_merged_span_ns =
      std::max(snapshot_.maximum_merged_span_ns, merged_span_ns);
  const uint64_t primary_point_count = accounting.primary_raw_point_count;
  const uint64_t secondary_point_count = accounting.secondary_raw_point_count;
  const uint64_t merged_point_count = accounting.compacted_output_point_count;
  const uint64_t total_raw_point_count =
      primary_point_count + secondary_point_count;
  const uint64_t removed_sentinel_count =
      total_raw_point_count >= merged_point_count
          ? total_raw_point_count - merged_point_count
          : 0U;
  snapshot_.last_primary_point_count = primary_point_count;
  snapshot_.maximum_primary_point_count = std::max(
      snapshot_.maximum_primary_point_count, primary_point_count);
  snapshot_.last_secondary_point_count = secondary_point_count;
  snapshot_.maximum_secondary_point_count = std::max(
      snapshot_.maximum_secondary_point_count, secondary_point_count);
  snapshot_.last_merged_point_count = merged_point_count;
  snapshot_.maximum_merged_point_count = std::max(
      snapshot_.maximum_merged_point_count, merged_point_count);
  snapshot_.last_total_raw_input_point_count = total_raw_point_count;
  snapshot_.maximum_total_raw_input_point_count = std::max(
      snapshot_.maximum_total_raw_input_point_count, total_raw_point_count);
  snapshot_.last_compacted_output_point_count = merged_point_count;
  snapshot_.maximum_compacted_output_point_count = std::max(
      snapshot_.maximum_compacted_output_point_count, merged_point_count);
  snapshot_.last_removed_no_return_sentinel_count = removed_sentinel_count;
  snapshot_.maximum_removed_no_return_sentinel_count = std::max(
      snapshot_.maximum_removed_no_return_sentinel_count,
      removed_sentinel_count);
  snapshot_.last_compacted_output_to_total_raw_input_ratio =
      total_raw_point_count > 0U
          ? static_cast<double>(merged_point_count) /
                static_cast<double>(total_raw_point_count)
          : 0.0;
  if (primary_point_count > 0U) {
    snapshot_.last_secondary_to_primary_point_ratio =
        static_cast<double>(secondary_point_count) /
        static_cast<double>(primary_point_count);
    snapshot_.last_merged_to_primary_point_ratio =
        static_cast<double>(merged_point_count) /
        static_cast<double>(primary_point_count);
  } else {
    snapshot_.last_secondary_to_primary_point_ratio = 0.0;
    snapshot_.last_merged_to_primary_point_ratio = 0.0;
  }
}

void DualLidarMergeStatistics::RecordPrimaryRelayed(uint64_t point_count) {
  snapshot_.last_merged_point_count = point_count;
  snapshot_.maximum_merged_point_count = std::max(
      snapshot_.maximum_merged_point_count, point_count);
  snapshot_.last_secondary_to_primary_point_ratio = 0.0;
  snapshot_.last_merged_to_primary_point_ratio = point_count > 0U ? 1.0 : 0.0;
  snapshot_.last_total_raw_input_point_count = point_count;
  snapshot_.maximum_total_raw_input_point_count = std::max(
      snapshot_.maximum_total_raw_input_point_count, point_count);
  snapshot_.last_compacted_output_point_count = point_count;
  snapshot_.maximum_compacted_output_point_count = std::max(
      snapshot_.maximum_compacted_output_point_count, point_count);
  snapshot_.last_removed_no_return_sentinel_count = 0U;
  snapshot_.last_compacted_output_to_total_raw_input_ratio =
      point_count > 0U ? 1.0 : 0.0;
}

void DualLidarMergeStatistics::RecordVoxelOverlapEvaluation(
    uint64_t pair_index, const DualLidarVoxelOverlap& overlap) {
  SaturatingIncrement(&snapshot_.overlap_evaluations_attempted);
  SaturatingIncrement(&snapshot_.overlap_evaluation_sample_count);
  snapshot_.last_overlap_pair_index = pair_index;
  snapshot_.last_primary_valid_return_count =
      overlap.primary_valid_return_count;
  snapshot_.last_secondary_valid_return_count =
      overlap.secondary_valid_return_count;
  snapshot_.last_primary_occupied_voxel_count =
      overlap.primary_occupied_voxel_count;
  snapshot_.last_secondary_occupied_voxel_count =
      overlap.secondary_occupied_voxel_count;
  snapshot_.last_shared_occupied_voxel_count =
      overlap.shared_occupied_voxel_count;
  snapshot_.last_union_occupied_voxel_count =
      overlap.union_occupied_voxel_count;
  snapshot_.last_primary_overlap_fraction = overlap.primary_overlap_fraction;
  snapshot_.last_secondary_overlap_fraction = overlap.secondary_overlap_fraction;
  snapshot_.last_voxel_jaccard = overlap.voxel_jaccard;
  const double count =
      static_cast<double>(snapshot_.overlap_evaluation_sample_count);
  snapshot_.mean_primary_overlap_fraction +=
      (overlap.primary_overlap_fraction -
       snapshot_.mean_primary_overlap_fraction) / count;
  snapshot_.mean_secondary_overlap_fraction +=
      (overlap.secondary_overlap_fraction -
       snapshot_.mean_secondary_overlap_fraction) / count;
  snapshot_.mean_voxel_jaccard +=
      (overlap.voxel_jaccard - snapshot_.mean_voxel_jaccard) / count;
  if (snapshot_.overlap_evaluation_sample_count == 1U) {
    snapshot_.minimum_primary_overlap_fraction =
        overlap.primary_overlap_fraction;
    snapshot_.minimum_secondary_overlap_fraction =
        overlap.secondary_overlap_fraction;
    snapshot_.minimum_voxel_jaccard = overlap.voxel_jaccard;
  } else {
    snapshot_.minimum_primary_overlap_fraction = std::min(
        snapshot_.minimum_primary_overlap_fraction,
        overlap.primary_overlap_fraction);
    snapshot_.minimum_secondary_overlap_fraction = std::min(
        snapshot_.minimum_secondary_overlap_fraction,
        overlap.secondary_overlap_fraction);
    snapshot_.minimum_voxel_jaccard = std::min(
        snapshot_.minimum_voxel_jaccard, overlap.voxel_jaccard);
  }
}

void DualLidarMergeStatistics::RecordVoxelOverlapEvaluationFailed() {
  SaturatingIncrement(&snapshot_.overlap_evaluations_attempted);
  SaturatingIncrement(&snapshot_.overlap_evaluations_failed);
}

void DualLidarMergeStatistics::RecordPairRejected() {
  SaturatingIncrement(&snapshot_.pairs_rejected);
}

void DualLidarMergeStatistics::RecordPrimaryUnpairedDrop() {
  SaturatingIncrement(&snapshot_.primary_unpaired_dropped);
}

void DualLidarMergeStatistics::RecordSecondaryUnpairedDrop() {
  SaturatingIncrement(&snapshot_.secondary_unpaired_dropped);
}

void DualLidarMergeStatistics::RecordPrimaryQueueDrop() {
  SaturatingIncrement(&snapshot_.primary_queue_dropped);
}

void DualLidarMergeStatistics::RecordSecondaryQueueDrop() {
  SaturatingIncrement(&snapshot_.secondary_queue_dropped);
}

DualLidarMergeStatisticsSnapshot DualLidarMergeStatistics::Snapshot() const {
  return snapshot_;
}

bool ValidateDualLidarCalibration(const DualLidarCalibration& calibration,
                                  std::string* error) {
  if (calibration.revision.empty()) {
    SetError("dual-LiDAR calibration revision is empty", error);
    return false;
  }
  for (double value : calibration.rotation_primary_secondary)
    if (!std::isfinite(value)) {
      SetError("dual-LiDAR rotation contains a non-finite value", error);
      return false;
    }
  for (double value : calibration.translation_primary_secondary)
    if (!std::isfinite(value)) {
      SetError("dual-LiDAR translation contains a non-finite value", error);
      return false;
    }
  constexpr double kTolerance = 1.0e-8;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      double dot = 0.0;
      for (int axis = 0; axis < 3; ++axis)
        dot += calibration.rotation_primary_secondary[3 * row + axis] *
               calibration.rotation_primary_secondary[3 * column + axis];
      const double expected = row == column ? 1.0 : 0.0;
      if (std::abs(dot - expected) > kTolerance) {
        SetError("dual-LiDAR rotation is not orthonormal", error);
        return false;
      }
    }
  }
  const auto& r = calibration.rotation_primary_secondary;
  const double determinant =
      r[0] * (r[4] * r[8] - r[5] * r[7]) -
      r[1] * (r[3] * r[8] - r[5] * r[6]) +
      r[2] * (r[3] * r[7] - r[4] * r[6]);
  if (std::abs(determinant - 1.0) > kTolerance) {
    SetError("dual-LiDAR rotation determinant is not +1", error);
    return false;
  }
  return true;
}

bool ComputeDualLidarVoxelOverlap(
    const livox_ros_driver::CustomMsg& primary,
    const livox_ros_driver::CustomMsg& secondary,
    const DualLidarCalibration& calibration,
    const std::string& primary_frame,
    const std::string& secondary_frame,
    double voxel_size_m,
    DualLidarVoxelOverlap* overlap,
    std::string* error) {
  if (!overlap) {
    SetError("null voxel-overlap output", error);
    return false;
  }
  if (!std::isfinite(voxel_size_m) || voxel_size_m <= 0.0) {
    SetError("overlap voxel size must be finite and positive", error);
    return false;
  }
  if (!ValidateDualLidarCalibration(calibration, error)) return false;
  if (primary.header.frame_id != primary_frame ||
      secondary.header.frame_id != secondary_frame) {
    SetError("dual-LiDAR input frame does not match overlap calibration", error);
    return false;
  }
  if (primary.point_num != primary.points.size() ||
      secondary.point_num != secondary.points.size()) {
    SetError("Livox point_num does not match overlap input points", error);
    return false;
  }

  DualLidarVoxelOverlap result;
  std::vector<VoxelKey> primary_voxels;
  std::vector<VoxelKey> secondary_voxels;
  if (!OccupiedVoxels(primary, false, calibration, voxel_size_m,
                      &result.primary_valid_return_count, &primary_voxels,
                      error) ||
      !OccupiedVoxels(secondary, true, calibration, voxel_size_m,
                      &result.secondary_valid_return_count, &secondary_voxels,
                      error))
    return false;
  if (primary_voxels.empty() || secondary_voxels.empty()) {
    SetError("overlap diagnostic requires returns from both LiDARs", error);
    return false;
  }

  result.primary_occupied_voxel_count = primary_voxels.size();
  result.secondary_occupied_voxel_count = secondary_voxels.size();
  uint64_t shared = 0U;
  std::size_t primary_index = 0U;
  std::size_t secondary_index = 0U;
  while (primary_index < primary_voxels.size() &&
         secondary_index < secondary_voxels.size()) {
    if (primary_voxels[primary_index] < secondary_voxels[secondary_index]) {
      ++primary_index;
    } else if (secondary_voxels[secondary_index] <
               primary_voxels[primary_index]) {
      ++secondary_index;
    } else {
      SaturatingIncrement(&shared);
      ++primary_index;
      ++secondary_index;
    }
  }
  result.shared_occupied_voxel_count = shared;
  result.union_occupied_voxel_count =
      result.primary_occupied_voxel_count +
      result.secondary_occupied_voxel_count - shared;
  result.primary_overlap_fraction = static_cast<double>(shared) /
      static_cast<double>(result.primary_occupied_voxel_count);
  result.secondary_overlap_fraction = static_cast<double>(shared) /
      static_cast<double>(result.secondary_occupied_voxel_count);
  result.voxel_jaccard = static_cast<double>(shared) /
      static_cast<double>(result.union_occupied_voxel_count);
  *overlap = result;
  return true;
}

bool MergeDualLivoxMessages(
    const livox_ros_driver::CustomMsg& primary,
    const livox_ros_driver::CustomMsg& secondary,
    const DualLidarCalibration& calibration,
    const std::string& primary_frame,
    const std::string& secondary_frame,
    uint64_t maximum_sync_delta_ns,
    livox_ros_driver::CustomMsg* output,
    std::string* error,
    DualLivoxMergeAccounting* accounting) {
  if (!output) {
    SetError("null merged message output", error);
    return false;
  }
  if (!ValidateDualLidarCalibration(calibration, error)) return false;
  if (primary.header.frame_id != primary_frame ||
      secondary.header.frame_id != secondary_frame) {
    SetError("dual-LiDAR input frame does not match calibration", error);
    return false;
  }
  if (primary.point_num != primary.points.size() ||
      secondary.point_num != secondary.points.size()) {
    SetError("Livox point_num does not match points array", error);
    return false;
  }
  const uint64_t delta = primary.timebase >= secondary.timebase
      ? primary.timebase - secondary.timebase
      : secondary.timebase - primary.timebase;
  if (delta > maximum_sync_delta_ns) {
    SetError("dual-LiDAR scans exceed synchronization tolerance", error);
    return false;
  }
  const uint64_t output_timebase =
      std::min(primary.timebase, secondary.timebase);
  livox_ros_driver::CustomMsg merged;
  merged.header.frame_id = primary_frame;
  merged.header.stamp.fromNSec(output_timebase);
  merged.timebase = output_timebase;
  merged.lidar_id = primary.lidar_id;
  merged.rsvd = primary.rsvd;
  merged.points.reserve(primary.points.size() + secondary.points.size());
  DualLivoxMergeAccounting completed;
  completed.primary_raw_point_count = primary.points.size();
  completed.secondary_raw_point_count = secondary.points.size();
  if (!AppendWithTimebase(primary, output_timebase, false, calibration,
                          &merged.points, error))
    return false;
  completed.primary_valid_return_count = merged.points.size();
  if (!AppendWithTimebase(secondary, output_timebase, true, calibration,
                          &merged.points, error))
    return false;
  completed.secondary_valid_return_count =
      merged.points.size() - completed.primary_valid_return_count;
  std::stable_sort(merged.points.begin(), merged.points.end(),
                   [](const auto& lhs, const auto& rhs) {
                     return lhs.offset_time < rhs.offset_time;
                   });
  if (merged.points.size() > std::numeric_limits<uint32_t>::max()) {
    SetError("merged Livox point count exceeds uint32", error);
    return false;
  }
  merged.point_num = static_cast<uint32_t>(merged.points.size());
  completed.compacted_output_point_count = merged.points.size();
  *output = std::move(merged);
  if (accounting) *accounting = completed;
  return true;
}

}  // namespace livox_laser_simulation
