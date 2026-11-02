#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>

#include "livox_laser_simulation/dual_livox_merge.hpp"
#include "livox_laser_simulation/point_time_model.hpp"

namespace livox_laser_simulation {
namespace {

constexpr double kPi = 3.14159265358979323846;

DualLidarCalibration Calibration() {
  const double angle = 25.0 * kPi / 180.0;
  const double cosine = std::cos(angle);
  const double sine = std::sin(angle);
  DualLidarCalibration calibration;
  calibration.rotation_primary_secondary = {
      -cosine, 0.0, sine,
      0.0, -1.0, 0.0,
      sine, 0.0, cosine};
  calibration.translation_primary_secondary = {-0.6181, 0.0, 0.0};
  calibration.revision = "a2-mid360-frontlevel-tail-rearup25deg-v3";
  return calibration;
}

livox_ros_driver::CustomMsg Message(const std::string& frame,
                                    uint64_t timebase, float x) {
  livox_ros_driver::CustomMsg message;
  message.header.frame_id = frame;
  message.header.stamp.fromNSec(timebase);
  message.timebase = timebase;
  message.lidar_id = 0U;
  livox_ros_driver::CustomPoint point;
  point.offset_time = 100U;
  point.x = x;
  point.y = 0.0F;
  point.z = 0.0F;
  point.tag = 0x10U;
  message.points.push_back(point);
  message.point_num = 1U;
  return message;
}

void AddPoint(float x, float y, float z,
              livox_ros_driver::CustomMsg* message) {
  livox_ros_driver::CustomPoint point;
  point.x = x;
  point.y = y;
  point.z = z;
  point.tag = 0x10U;
  message->points.push_back(point);
  message->point_num = static_cast<uint32_t>(message->points.size());
}

DualLivoxMergeAccounting Accounting(uint64_t primary_raw,
                                    uint64_t secondary_raw,
                                    uint64_t compacted) {
  DualLivoxMergeAccounting accounting;
  accounting.primary_raw_point_count = primary_raw;
  accounting.secondary_raw_point_count = secondary_raw;
  accounting.compacted_output_point_count = compacted;
  return accounting;
}

TEST(DualLivoxMerge, PreservesAbsoluteTimeAndTransformsTailPoint) {
  const auto primary = Message("livox_frame", 1000000U, 2.0F);
  const auto secondary = Message("livox_rear_frame", 1000500U, 1.0F);
  livox_ros_driver::CustomMsg output;
  std::string error;
  ASSERT_TRUE(MergeDualLivoxMessages(
      primary, secondary, Calibration(), "livox_frame",
      "livox_rear_frame", 1000U, &output, &error)) << error;
  ASSERT_EQ(output.points.size(), 2U);
  EXPECT_EQ(output.timebase, 1000000U);
  EXPECT_EQ(output.points[0].offset_time, 100U);
  EXPECT_EQ(output.points[1].offset_time, 600U);
  EXPECT_FLOAT_EQ(output.points[0].x, 2.0F);
  EXPECT_NEAR(output.points[1].x, -std::cos(25.0 * kPi / 180.0) - 0.6181,
              1.0e-6);
  EXPECT_NEAR(output.points[1].y, 0.0, 1.0e-6);
  EXPECT_NEAR(output.points[1].z, std::sin(25.0 * kPi / 180.0),
              1.0e-6);
  EXPECT_EQ(output.header.frame_id, "livox_frame");
}

TEST(DualLivoxMerge, SnapshotPointsKeepOnlyRealInterSensorTimeShift) {
  auto primary = Message("livox_frame", 1000000U, 2.0F);
  auto secondary = Message("livox_rear_frame", 1015000U, 1.0F);
  primary.points[0].offset_time =
      SimulatedPointOffsetNs(PointTimeMode::SNAPSHOT, 19999U);
  secondary.points[0].offset_time =
      SimulatedPointOffsetNs(PointTimeMode::SNAPSHOT, 19999U);
  livox_ros_driver::CustomMsg output;
  std::string error;
  ASSERT_TRUE(MergeDualLivoxMessages(
      primary, secondary, Calibration(), "livox_frame",
      "livox_rear_frame", 25000000U, &output, &error)) << error;
  ASSERT_EQ(output.points.size(), 2U);
  EXPECT_EQ(output.timebase, 1000000U);
  EXPECT_EQ(output.points[0].offset_time, 0U);
  EXPECT_EQ(output.points[1].offset_time, 15000U);
}

TEST(PointTimeModel, SnapshotAndLegacyRollingAreDistinctAndParseable) {
  PointTimeMode mode = PointTimeMode::SNAPSHOT;
  ASSERT_TRUE(ParsePointTimeMode("legacy_rolling", &mode));
  EXPECT_EQ(mode, PointTimeMode::LEGACY_ROLLING);
  EXPECT_EQ(SimulatedPointOffsetNs(mode, 19999U), 99995000U);
  ASSERT_TRUE(ParsePointTimeMode("snapshot", &mode));
  EXPECT_EQ(mode, PointTimeMode::SNAPSHOT);
  EXPECT_EQ(SimulatedPointOffsetNs(mode, 19999U), 0U);
  EXPECT_FALSE(ParsePointTimeMode("deskew_guess", &mode));
}

TEST(DualLivoxMergeStatistics, IsFixedMemoryAndReportsTimingAndDrops) {
  DualLidarMergeStatistics statistics;
  statistics.RecordPrimaryReceived(20000U);
  statistics.RecordPrimaryReceived(19998U);
  statistics.RecordSecondaryReceived(10000U);
  statistics.RecordPairAccepted(
      1000000000U, 15000U, 15000U, Accounting(20000U, 10000U, 23000U));
  statistics.RecordPairAccepted(
      1100000000U, 25000U, 25000U, Accounting(19998U, 9999U, 22997U));
  statistics.RecordPairRejected();
  statistics.RecordPrimaryUnpairedDrop();
  statistics.RecordSecondaryQueueDrop();
  const auto value = statistics.Snapshot();
  EXPECT_EQ(value.primary_received, 2U);
  EXPECT_EQ(value.secondary_received, 1U);
  EXPECT_EQ(value.pairs_accepted, 2U);
  EXPECT_EQ(value.pairs_rejected, 1U);
  EXPECT_EQ(value.primary_unpaired_dropped, 1U);
  EXPECT_EQ(value.secondary_queue_dropped, 1U);
  EXPECT_EQ(value.last_sync_delta_ns, 25000U);
  EXPECT_EQ(value.maximum_sync_delta_ns, 25000U);
  EXPECT_DOUBLE_EQ(value.mean_sync_delta_ns, 20000.0);
  EXPECT_EQ(value.first_accepted_measurement_time_ns, 1000000000U);
  EXPECT_EQ(value.last_accepted_measurement_time_ns, 1100000000U);
  EXPECT_EQ(value.accepted_measurement_interval_count, 1U);
  EXPECT_EQ(value.nonpositive_accepted_measurement_interval_count, 0U);
  EXPECT_EQ(value.minimum_accepted_measurement_interval_ns, 100000000U);
  EXPECT_EQ(value.maximum_accepted_measurement_interval_ns, 100000000U);
  EXPECT_DOUBLE_EQ(value.mean_accepted_measurement_interval_ns, 100000000.0);
  EXPECT_DOUBLE_EQ(value.realized_measurement_hz, 10.0);
  EXPECT_EQ(value.maximum_merged_span_ns, 25000U);
  EXPECT_EQ(value.last_primary_point_count, 19998U);
  EXPECT_EQ(value.maximum_primary_point_count, 20000U);
  EXPECT_EQ(value.last_secondary_point_count, 9999U);
  EXPECT_EQ(value.maximum_secondary_point_count, 10000U);
  EXPECT_EQ(value.last_merged_point_count, 22997U);
  EXPECT_EQ(value.maximum_merged_point_count, 23000U);
  EXPECT_DOUBLE_EQ(value.last_secondary_to_primary_point_ratio, 0.5);
  EXPECT_NEAR(value.last_merged_to_primary_point_ratio,
              22997.0 / 19998.0, 1.0e-12);
  EXPECT_EQ(value.last_total_raw_input_point_count, 29997U);
  EXPECT_EQ(value.maximum_total_raw_input_point_count, 30000U);
  EXPECT_EQ(value.last_compacted_output_point_count, 22997U);
  EXPECT_EQ(value.maximum_compacted_output_point_count, 23000U);
  EXPECT_EQ(value.last_removed_no_return_sentinel_count, 7000U);
  EXPECT_EQ(value.maximum_removed_no_return_sentinel_count, 7000U);
  EXPECT_NEAR(value.last_compacted_output_to_total_raw_input_ratio,
              22997.0 / 29997.0, 1.0e-12);
}

TEST(DualLivoxMergeStatistics,
     ReportsEffectiveCadenceAndRejectableNonpositiveIntervals) {
  DualLidarMergeStatistics statistics;
  statistics.RecordPairAccepted(
      5000000000U, 0U, 0U, Accounting(10U, 10U, 16U));
  statistics.RecordPairAccepted(
      5200000000U, 0U, 0U, Accounting(10U, 10U, 16U));
  statistics.RecordPairAccepted(
      5400000000U, 0U, 0U, Accounting(10U, 10U, 16U));
  auto value = statistics.Snapshot();
  EXPECT_EQ(value.first_accepted_measurement_time_ns, 5000000000U);
  EXPECT_EQ(value.last_accepted_measurement_time_ns, 5400000000U);
  EXPECT_EQ(value.accepted_measurement_interval_count, 2U);
  EXPECT_EQ(value.nonpositive_accepted_measurement_interval_count, 0U);
  EXPECT_EQ(value.minimum_accepted_measurement_interval_ns, 200000000U);
  EXPECT_EQ(value.maximum_accepted_measurement_interval_ns, 200000000U);
  EXPECT_DOUBLE_EQ(value.mean_accepted_measurement_interval_ns, 200000000.0);
  EXPECT_DOUBLE_EQ(value.realized_measurement_hz, 5.0);

  // Duplicate and reversed accepted epochs are explicit contract violations;
  // neither is allowed to make the valid cadence appear faster or slower.
  statistics.RecordPairAccepted(
      5400000000U, 0U, 0U, Accounting(10U, 10U, 16U));
  statistics.RecordPairAccepted(
      5300000000U, 0U, 0U, Accounting(10U, 10U, 16U));
  value = statistics.Snapshot();
  EXPECT_EQ(value.accepted_measurement_interval_count, 2U);
  EXPECT_EQ(value.nonpositive_accepted_measurement_interval_count, 2U);
  EXPECT_EQ(value.last_accepted_measurement_time_ns, 5400000000U);
  EXPECT_EQ(value.minimum_accepted_measurement_interval_ns, 200000000U);
  EXPECT_EQ(value.maximum_accepted_measurement_interval_ns, 200000000U);
  EXPECT_DOUBLE_EQ(value.mean_accepted_measurement_interval_ns, 200000000.0);
  EXPECT_DOUBLE_EQ(value.realized_measurement_hz, 5.0);
}

TEST(DualLivoxMergeStatistics, ReportsMinimumAndMaximumAcceptedIntervals) {
  DualLidarMergeStatistics statistics;
  statistics.RecordPairAccepted(
      1000000000U, 0U, 0U, Accounting(1U, 1U, 2U));
  statistics.RecordPairAccepted(
      1100000000U, 0U, 0U, Accounting(1U, 1U, 2U));
  statistics.RecordPairAccepted(
      1450000000U, 0U, 0U, Accounting(1U, 1U, 2U));
  statistics.RecordPairAccepted(
      1650000000U, 0U, 0U, Accounting(1U, 1U, 2U));
  const auto value = statistics.Snapshot();
  EXPECT_EQ(value.accepted_measurement_interval_count, 3U);
  EXPECT_EQ(value.minimum_accepted_measurement_interval_ns, 100000000U);
  EXPECT_EQ(value.maximum_accepted_measurement_interval_ns, 350000000U);
  EXPECT_NEAR(value.mean_accepted_measurement_interval_ns,
              650000000.0 / 3.0, 1.0e-9);
  EXPECT_NEAR(value.realized_measurement_hz,
              3.0e9 / 650000000.0, 1.0e-12);
}

TEST(DualLivoxOverlap, MeasuresCommonOccupiedSpaceAfterRigidTransform) {
  DualLidarCalibration calibration;
  calibration.rotation_primary_secondary = {
      1.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0};
  calibration.translation_primary_secondary = {1.0, 0.0, 0.0};
  calibration.revision = "unit-test";
  auto primary = Message("livox_frame", 1000000U, 1.01F);
  AddPoint(1.21F, 0.01F, 0.01F, &primary);
  AddPoint(1.41F, 0.01F, 0.01F, &primary);
  // Duplicate return in the first occupied cell must not change set overlap.
  AddPoint(1.22F, 0.01F, 0.01F, &primary);
  auto secondary = Message("livox_rear_frame", 1000000U, 0.21F);
  AddPoint(0.41F, 0.01F, 0.01F, &secondary);
  AddPoint(0.61F, 0.01F, 0.01F, &secondary);
  // Gazebo's no-return sentinel is not an occupied cell.
  AddPoint(0.0F, 0.0F, 0.0F, &secondary);

  DualLidarVoxelOverlap overlap;
  std::string error;
  ASSERT_TRUE(ComputeDualLidarVoxelOverlap(
      primary, secondary, calibration, "livox_frame", "livox_rear_frame",
      0.20, &overlap, &error)) << error;
  EXPECT_EQ(overlap.primary_valid_return_count, 4U);
  EXPECT_EQ(overlap.secondary_valid_return_count, 3U);
  EXPECT_EQ(overlap.primary_occupied_voxel_count, 3U);
  EXPECT_EQ(overlap.secondary_occupied_voxel_count, 3U);
  EXPECT_EQ(overlap.shared_occupied_voxel_count, 2U);
  EXPECT_EQ(overlap.union_occupied_voxel_count, 4U);
  EXPECT_NEAR(overlap.primary_overlap_fraction, 2.0 / 3.0, 1.0e-12);
  EXPECT_NEAR(overlap.secondary_overlap_fraction, 2.0 / 3.0, 1.0e-12);
  EXPECT_NEAR(overlap.voxel_jaccard, 0.5, 1.0e-12);
}

TEST(DualLivoxOverlap, RejectsInvalidDiagnosticInputsWithoutMergeMutation) {
  const auto primary = Message("livox_frame", 1000000U, 2.0F);
  const auto secondary = Message("livox_rear_frame", 1000000U, 1.0F);
  DualLidarVoxelOverlap overlap;
  std::string error;
  EXPECT_FALSE(ComputeDualLidarVoxelOverlap(
      primary, secondary, Calibration(), "livox_frame", "livox_rear_frame",
      0.0, &overlap, &error));
  EXPECT_EQ(primary.points.size(), 1U);
  EXPECT_EQ(secondary.points.size(), 1U);
}

TEST(DualLivoxMergeStatistics, AggregatesPeriodicVoxelOverlapSamples) {
  DualLidarMergeStatistics statistics;
  DualLidarVoxelOverlap first;
  first.primary_valid_return_count = 100U;
  first.secondary_valid_return_count = 50U;
  first.primary_occupied_voxel_count = 80U;
  first.secondary_occupied_voxel_count = 40U;
  first.shared_occupied_voxel_count = 30U;
  first.union_occupied_voxel_count = 90U;
  first.primary_overlap_fraction = 0.375;
  first.secondary_overlap_fraction = 0.75;
  first.voxel_jaccard = 1.0 / 3.0;
  statistics.RecordVoxelOverlapEvaluation(10U, first);
  DualLidarVoxelOverlap second = first;
  second.primary_overlap_fraction = 0.5;
  second.secondary_overlap_fraction = 0.8;
  second.voxel_jaccard = 0.4;
  statistics.RecordVoxelOverlapEvaluation(20U, second);
  statistics.RecordVoxelOverlapEvaluationFailed();
  const auto value = statistics.Snapshot();
  EXPECT_EQ(value.overlap_evaluations_attempted, 3U);
  EXPECT_EQ(value.overlap_evaluations_failed, 1U);
  EXPECT_EQ(value.overlap_evaluation_sample_count, 2U);
  EXPECT_EQ(value.last_overlap_pair_index, 20U);
  EXPECT_DOUBLE_EQ(value.mean_primary_overlap_fraction, 0.4375);
  EXPECT_DOUBLE_EQ(value.mean_secondary_overlap_fraction, 0.775);
  EXPECT_NEAR(value.mean_voxel_jaccard, 11.0 / 30.0, 1.0e-12);
  EXPECT_DOUBLE_EQ(value.minimum_primary_overlap_fraction, 0.375);
  EXPECT_DOUBLE_EQ(value.minimum_secondary_overlap_fraction, 0.75);
  EXPECT_NEAR(value.minimum_voxel_jaccard, 1.0 / 3.0, 1.0e-12);
}

TEST(DualLivoxMergeStatistics, SingleSensorRelayStillReportsOutputDensity) {
  DualLidarMergeStatistics statistics;
  statistics.RecordPrimaryReceived(20000U);
  statistics.RecordPrimaryRelayed(20000U);
  const auto value = statistics.Snapshot();
  EXPECT_EQ(value.last_primary_point_count, 20000U);
  EXPECT_EQ(value.last_merged_point_count, 20000U);
  EXPECT_DOUBLE_EQ(value.last_secondary_to_primary_point_ratio, 0.0);
  EXPECT_DOUBLE_EQ(value.last_merged_to_primary_point_ratio, 1.0);
  EXPECT_EQ(value.last_total_raw_input_point_count, 20000U);
  EXPECT_EQ(value.last_compacted_output_point_count, 20000U);
  EXPECT_EQ(value.last_removed_no_return_sentinel_count, 0U);
  EXPECT_DOUBLE_EQ(value.last_compacted_output_to_total_raw_input_ratio, 1.0);
}

TEST(DualLivoxMerge, CompactsOnlyExactZeroNoReturnSentinels) {
  auto primary = Message("livox_frame", 1000000U, 0.0F);
  primary.points[0].offset_time = 50U;
  AddPoint(0.0F, 0.0F, std::numeric_limits<float>::min(), &primary);
  primary.points[1].offset_time = 80U;
  primary.points[1].tag = 0x21U;
  auto secondary = Message("livox_rear_frame", 1000100U, 0.0F);
  secondary.points[0].offset_time = 60U;
  AddPoint(1.0F, 2.0F, 3.0F, &secondary);
  secondary.points[1].offset_time = 90U;
  secondary.points[1].tag = 0x32U;
  const auto primary_before = primary;
  const auto secondary_before = secondary;
  livox_ros_driver::CustomMsg output;
  DualLivoxMergeAccounting accounting;
  std::string error;
  ASSERT_TRUE(MergeDualLivoxMessages(
      primary, secondary, Calibration(), "livox_frame",
      "livox_rear_frame", 1000U, &output, &error, &accounting)) << error;
  ASSERT_EQ(output.points.size(), 2U);
  EXPECT_EQ(output.point_num, 2U);
  EXPECT_EQ(output.points[0].offset_time, 80U);
  EXPECT_EQ(output.points[0].tag, 0x21U);
  EXPECT_FLOAT_EQ(output.points[0].z, std::numeric_limits<float>::min());
  EXPECT_EQ(output.points[1].offset_time, 190U);
  EXPECT_EQ(output.points[1].tag, 0x32U);
  EXPECT_EQ(accounting.primary_raw_point_count, 2U);
  EXPECT_EQ(accounting.secondary_raw_point_count, 2U);
  EXPECT_EQ(accounting.primary_valid_return_count, 1U);
  EXPECT_EQ(accounting.secondary_valid_return_count, 1U);
  EXPECT_EQ(accounting.compacted_output_point_count, 2U);
  // The const source messages represent the raw front/rear topics and must not
  // be compacted or transformed in-place.
  EXPECT_EQ(primary.points.size(), primary_before.points.size());
  EXPECT_EQ(secondary.points.size(), secondary_before.points.size());
  EXPECT_FLOAT_EQ(secondary.points[1].x, secondary_before.points[1].x);
  EXPECT_EQ(secondary.points[1].offset_time,
            secondary_before.points[1].offset_time);
}

TEST(DualLivoxMerge, RejectsNonFinitePointWithoutPartialOutputOrAccounting) {
  auto primary = Message("livox_frame", 1000000U, 2.0F);
  auto secondary = Message("livox_rear_frame", 1000000U, 0.0F);
  AddPoint(std::numeric_limits<float>::quiet_NaN(), 0.0F, 1.0F, &secondary);
  auto output = Message("preserve_me", 42U, 7.0F);
  DualLivoxMergeAccounting accounting;
  accounting.primary_raw_point_count = 123U;
  std::string error;
  EXPECT_FALSE(MergeDualLivoxMessages(
      primary, secondary, Calibration(), "livox_frame",
      "livox_rear_frame", 1000U, &output, &error, &accounting));
  EXPECT_EQ(output.header.frame_id, "preserve_me");
  EXPECT_EQ(output.timebase, 42U);
  ASSERT_EQ(output.points.size(), 1U);
  EXPECT_FLOAT_EQ(output.points[0].x, 7.0F);
  EXPECT_EQ(accounting.primary_raw_point_count, 123U);
}

TEST(DualLivoxMerge, RejectsFrameAndSynchronizationMismatchAtomically) {
  auto primary = Message("wrong", 1000000U, 1.0F);
  auto secondary = Message("livox_rear_frame", 1005000U, 1.0F);
  livox_ros_driver::CustomMsg output;
  std::string error;
  EXPECT_FALSE(MergeDualLivoxMessages(
      primary, secondary, Calibration(), "livox_frame",
      "livox_rear_frame", 1000U, &output, &error));
  primary.header.frame_id = "livox_frame";
  EXPECT_FALSE(MergeDualLivoxMessages(
      primary, secondary, Calibration(), "livox_frame",
      "livox_rear_frame", 1000U, &output, &error));
}

TEST(DualLivoxMerge, RejectsNonRigidCalibration) {
  DualLidarCalibration calibration = Calibration();
  calibration.rotation_primary_secondary[0] = 2.0;
  std::string error;
  EXPECT_FALSE(ValidateDualLidarCalibration(calibration, &error));
}

}  // namespace
}  // namespace livox_laser_simulation

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
