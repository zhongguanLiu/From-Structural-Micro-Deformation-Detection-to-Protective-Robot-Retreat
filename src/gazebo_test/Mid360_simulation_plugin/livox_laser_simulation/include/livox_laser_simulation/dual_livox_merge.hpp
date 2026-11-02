#pragma once

#include <array>
#include <cstdint>
#include <string>

#include <livox_ros_driver/CustomMsg.h>

namespace livox_laser_simulation {

struct DualLidarCalibration {
  // p_primary = rotation_primary_secondary * p_secondary
  //             + translation_primary_secondary.
  std::array<double, 9> rotation_primary_secondary{};
  std::array<double, 3> translation_primary_secondary{};
  std::string revision;
};

// Scene-dependent co-observation measured from one already synchronized scan
// pair.  Both occupied-voxel sets are formed in the primary LiDAR frame after
// excluding the Gazebo (0,0,0) no-return sentinel.  This diagnostic never
// changes, filters, or reorders the cloud delivered to FAST-LIO.
struct DualLidarVoxelOverlap {
  uint64_t primary_valid_return_count = 0U;
  uint64_t secondary_valid_return_count = 0U;
  uint64_t primary_occupied_voxel_count = 0U;
  uint64_t secondary_occupied_voxel_count = 0U;
  uint64_t shared_occupied_voxel_count = 0U;
  uint64_t union_occupied_voxel_count = 0U;
  double primary_overlap_fraction = 0.0;
  double secondary_overlap_fraction = 0.0;
  double voxel_jaccard = 0.0;
};

// Per-pair accounting at the exact formal merge boundary.  The two source
// counts are the immutable CustomMsg array sizes delivered on the front/rear
// topics.  A valid return is any finite point other than the Gazebo
// no-return sentinel whose coordinates are exactly (0,0,0).  The formal dual
// merge removes only that sentinel; it does not range-crop or voxelize.
struct DualLivoxMergeAccounting {
  uint64_t primary_raw_point_count = 0U;
  uint64_t secondary_raw_point_count = 0U;
  uint64_t primary_valid_return_count = 0U;
  uint64_t secondary_valid_return_count = 0U;
  uint64_t compacted_output_point_count = 0U;
};

// Fixed-memory synchronization telemetry.  These counters make scan pairing
// and simulated timing auditable without recording either raw point cloud.
struct DualLidarMergeStatisticsSnapshot {
  uint64_t primary_received = 0U;
  uint64_t secondary_received = 0U;
  uint64_t pairs_accepted = 0U;
  uint64_t pairs_rejected = 0U;
  uint64_t primary_unpaired_dropped = 0U;
  uint64_t secondary_unpaired_dropped = 0U;
  uint64_t primary_queue_dropped = 0U;
  uint64_t secondary_queue_dropped = 0U;
  uint64_t last_sync_delta_ns = 0U;
  uint64_t maximum_sync_delta_ns = 0U;
  double mean_sync_delta_ns = 0.0;
  // Effective measurement cadence is derived only from accepted merged-pair
  // timebases (Gazebo simulation time), never from callback wall time.  This
  // makes a slow sensor schedule visible even when pairing itself is perfect.
  uint64_t first_accepted_measurement_time_ns = 0U;
  uint64_t last_accepted_measurement_time_ns = 0U;
  uint64_t accepted_measurement_interval_count = 0U;
  uint64_t nonpositive_accepted_measurement_interval_count = 0U;
  uint64_t minimum_accepted_measurement_interval_ns = 0U;
  uint64_t maximum_accepted_measurement_interval_ns = 0U;
  double mean_accepted_measurement_interval_ns = 0.0;
  double realized_measurement_hz = 0.0;
  uint64_t last_merged_span_ns = 0U;
  uint64_t maximum_merged_span_ns = 0U;
  uint64_t last_primary_point_count = 0U;
  uint64_t maximum_primary_point_count = 0U;
  uint64_t last_secondary_point_count = 0U;
  uint64_t maximum_secondary_point_count = 0U;
  uint64_t last_merged_point_count = 0U;
  uint64_t maximum_merged_point_count = 0U;
  // Ratios are measured from the most recently accepted pair, after each
  // sensor plugin has applied its configured deterministic stride.
  double last_secondary_to_primary_point_ratio = 0.0;
  double last_merged_to_primary_point_ratio = 0.0;
  // Explicit schema-v5 aliases/accounting remove the historical ambiguity:
  // source counts are raw arrays, while the output count is after exact-zero
  // no-return compaction.  The older fields above remain compatible aliases
  // (primary/secondary=raw, merged=compacted output).
  uint64_t last_total_raw_input_point_count = 0U;
  uint64_t maximum_total_raw_input_point_count = 0U;
  uint64_t last_compacted_output_point_count = 0U;
  uint64_t maximum_compacted_output_point_count = 0U;
  uint64_t last_removed_no_return_sentinel_count = 0U;
  uint64_t maximum_removed_no_return_sentinel_count = 0U;
  double last_compacted_output_to_total_raw_input_ratio = 0.0;
  uint64_t overlap_evaluations_attempted = 0U;
  uint64_t overlap_evaluations_failed = 0U;
  uint64_t overlap_evaluation_sample_count = 0U;
  uint64_t last_overlap_pair_index = 0U;
  uint64_t last_primary_valid_return_count = 0U;
  uint64_t last_secondary_valid_return_count = 0U;
  uint64_t last_primary_occupied_voxel_count = 0U;
  uint64_t last_secondary_occupied_voxel_count = 0U;
  uint64_t last_shared_occupied_voxel_count = 0U;
  uint64_t last_union_occupied_voxel_count = 0U;
  double last_primary_overlap_fraction = 0.0;
  double last_secondary_overlap_fraction = 0.0;
  double last_voxel_jaccard = 0.0;
  double mean_primary_overlap_fraction = 0.0;
  double mean_secondary_overlap_fraction = 0.0;
  double mean_voxel_jaccard = 0.0;
  double minimum_primary_overlap_fraction = 0.0;
  double minimum_secondary_overlap_fraction = 0.0;
  double minimum_voxel_jaccard = 0.0;
};

class DualLidarMergeStatistics {
 public:
  void RecordPrimaryReceived(uint64_t point_count);
  void RecordSecondaryReceived(uint64_t point_count);
  void RecordPairAccepted(uint64_t accepted_measurement_time_ns,
                          uint64_t sync_delta_ns,
                          uint64_t merged_span_ns,
                          const DualLivoxMergeAccounting& accounting);
  void RecordPrimaryRelayed(uint64_t point_count);
  void RecordVoxelOverlapEvaluation(
      uint64_t pair_index, const DualLidarVoxelOverlap& overlap);
  void RecordVoxelOverlapEvaluationFailed();
  void RecordPairRejected();
  void RecordPrimaryUnpairedDrop();
  void RecordSecondaryUnpairedDrop();
  void RecordPrimaryQueueDrop();
  void RecordSecondaryQueueDrop();
  DualLidarMergeStatisticsSnapshot Snapshot() const;

 private:
  DualLidarMergeStatisticsSnapshot snapshot_;
};

uint64_t LivoxPointOffsetSpanNs(
    const livox_ros_driver::CustomMsg& message);

bool ValidateDualLidarCalibration(const DualLidarCalibration& calibration,
                                  std::string* error);

// Quantifies actual common occupied space, rather than nominal angular field
// of view.  A deterministic voxel set is used instead of point-to-point nearest
// neighbours to keep the periodically sampled diagnostic bounded and cheap.
bool ComputeDualLidarVoxelOverlap(
    const livox_ros_driver::CustomMsg& primary,
    const livox_ros_driver::CustomMsg& secondary,
    const DualLidarCalibration& calibration,
    const std::string& primary_frame,
    const std::string& secondary_frame,
    double voxel_size_m,
    DualLidarVoxelOverlap* overlap,
    std::string* error);

// The output timebase is the earlier scan time.  Every point keeps its
// absolute acquisition time through a recomputed uint32 offset, then the
// merged sequence is stably sorted by offset for FAST-LIO deskewing.  Only
// exact (0,0,0) Gazebo no-return sentinels are omitted from this formal dual
// output.  All input validation is transactional: on failure neither output
// nor optional accounting is modified.
bool MergeDualLivoxMessages(
    const livox_ros_driver::CustomMsg& primary,
    const livox_ros_driver::CustomMsg& secondary,
    const DualLidarCalibration& calibration,
    const std::string& primary_frame,
    const std::string& secondary_frame,
    uint64_t maximum_sync_delta_ns,
    livox_ros_driver::CustomMsg* output,
    std::string* error,
    DualLivoxMergeAccounting* accounting = nullptr);

}  // namespace livox_laser_simulation
