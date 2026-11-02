#!/usr/bin/env python3
"""Bounded, analysis-ready recorder for the ICRA2027 deformation experiment.

This node deliberately does not record rosbag files.  It preserves continuous
numeric streams as CSV, configuration/provenance as JSON, and only bounded
phase snapshots of high-volume maps and registered clouds as compressed NPZ.
Gazebo truth is written to evaluation tables and is never published back into
the detector, evolution map, avoidance map, or navigation stack.
"""

import csv
import bisect
import copy
import hashlib
import json
import math
import os
import glob
import threading
import time
from collections import defaultdict

import numpy as np
import rospy
import tf.transformations as transformations
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import PerformanceMetrics
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from move_base_msgs.msg import MoveBaseActionGoal, MoveBaseActionResult
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, Imu
from std_msgs.msg import String

from deform_avoidance_map.msg import DeformationAvoidanceMap
from deform_evolution_map.msg import DeformationEvolutionFrame
from deform_monitor_v3.msg import V1DetectionStatus, V1DetectionTile


def yaw_from_quaternion(quaternion):
    return transformations.euler_from_quaternion((
        quaternion.x, quaternion.y, quaternion.z, quaternion.w))[2]


def stable_id_string(stable_id):
    return "{:016x}{:016x}".format(stable_id.high, stable_id.low)


def rotate_xy(vector, yaw):
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return (cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1])


def compose_planar_body_pose(model_pose, calibration):
    offset = rotate_xy(calibration[:2], model_pose[3])
    return (model_pose[0] + offset[0], model_pose[1] + offset[1],
            model_pose[2] + calibration[2], model_pose[3] + calibration[3])


def normalized_quaternion(quaternion):
    values = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(values))
    if values.shape != (4,) or not math.isfinite(norm) or norm <= 1.0e-15:
        raise ValueError("invalid quaternion")
    return tuple(float(value) for value in values / norm)


def multiply_quaternions(first, second):
    """Hamilton product for ROS xyzw quaternions."""
    x1, y1, z1, w1 = normalized_quaternion(first)
    x2, y2, z2, w2 = normalized_quaternion(second)
    return normalized_quaternion((
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2))


def rotate_vector_quaternion(vector, quaternion):
    x, y, z, w = normalized_quaternion(quaternion)
    rotation = np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)]], dtype=float)
    return tuple(float(value) for value in rotation.dot(
        np.asarray(vector, dtype=float)))


def compose_body_pose_se3(model_position, model_quaternion, calibration):
    """Compose model->FAST-LIO-body rigid transform without planar loss."""
    rotated_offset = rotate_vector_quaternion(calibration[:3], model_quaternion)
    position = tuple(float(model_position[index]) + rotated_offset[index]
                     for index in range(3))
    half_yaw = 0.5 * float(calibration[3])
    calibration_quaternion = (0.0, 0.0, math.sin(half_yaw),
                              math.cos(half_yaw))
    orientation = multiply_quaternions(model_quaternion,
                                       calibration_quaternion)
    return position, orientation


def map_from_world_anchor(map_pose, world_pose):
    yaw = map_pose[2] - world_pose[2]
    rotated = rotate_xy(world_pose[:2], yaw)
    return (map_pose[0] - rotated[0], map_pose[1] - rotated[1], yaw)


def transform_point(transform, point):
    rotated = rotate_xy(point[:2], transform[2])
    return (rotated[0] + transform[0], rotated[1] + transform[1])


def inverse_transform_point(transform, point):
    shifted = (point[0] - transform[0], point[1] - transform[1])
    return rotate_xy(shifted, -transform[2])


def vector_metrics(estimate, truth):
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    estimate_norm = float(np.linalg.norm(estimate))
    truth_norm = float(np.linalg.norm(truth))
    error = float(np.linalg.norm(estimate - truth))
    cosine = ""
    if estimate_norm > 1.0e-12 and truth_norm > 1.0e-12:
        cosine = float(np.dot(estimate, truth) / (estimate_norm * truth_norm))
    return estimate_norm, truth_norm, error, cosine


def trajectory_analysis_segment(stamp_s, capture_cutoff_stamp_s,
                                post_capture_is_return=True):
    """Assign odometry without mislabelling stationary post-capture data."""
    if capture_cutoff_stamp_s is None or stamp_s < capture_cutoff_stamp_s:
        return "PRIMARY_CAPTURE"
    return ("RETURN_RECORDED_ONLY" if post_capture_is_return
            else "POST_PRIMARY_CAPTURE")


def _avoidance_grid_geometry(grid):
    try:
        width = int(grid.info.width)
        height = int(grid.info.height)
        resolution = float(grid.info.resolution)
        origin = grid.info.origin.position
        values = (resolution, float(origin.x), float(origin.y), float(origin.z))
    except (AttributeError, TypeError, ValueError, OverflowError):
        raise ValueError("invalid avoidance grid geometry")
    if (width <= 0 or height <= 0 or resolution <= 0.0
            or len(grid.data) != width * height
            or not all(math.isfinite(value) for value in values)):
        raise ValueError("invalid avoidance grid geometry or data length")
    return (str(grid.header.frame_id), width, height) + values


def avoidance_forecast_layer_name(horizon_s):
    horizon = float(horizon_s)
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("forecast horizon must be finite and positive")
    milliseconds = int(round(1000.0 * horizon))
    if milliseconds <= 0 or abs(milliseconds / 1000.0 - horizon) > 1.0e-9:
        raise ValueError("forecast horizon must have exact millisecond identity")
    return "avoidance_forecast_{:04d}ms".format(milliseconds)


def avoidance_snapshot_layers(message):
    """Expand one avoidance message into an atomic current+forecast bundle."""
    snapshot_id = str(message.source_snapshot_id)
    if not snapshot_id:
        raise ValueError("avoidance snapshot_id is empty")
    frame_index = int(message.source_accepted_frame_index)
    revision = int(message.map_revision)
    if frame_index < 0 or revision < 0:
        raise ValueError("avoidance snapshot index/revision is invalid")
    horizons = list(message.forecast_horizons_s)
    overlays = list(message.forecast_overlays)
    if len(horizons) != len(overlays):
        raise ValueError("avoidance forecast horizon/grid cardinality mismatch")
    current_geometry = _avoidance_grid_geometry(message.current_overlay)
    maps = {"avoidance_overlay": message.current_overlay}
    source_stamp_s = float(message.header.stamp.to_sec())
    if not math.isfinite(source_stamp_s):
        raise ValueError("avoidance source stamp is invalid")
    current_stamp_s = float(message.current_overlay.header.stamp.to_sec())
    if (not math.isfinite(current_stamp_s)
            or abs(current_stamp_s - source_stamp_s) > 1.0e-9):
        raise ValueError("avoidance current grid stamp differs from source")
    common = {
        "source_snapshot_id": snapshot_id,
        "source_accepted_frame_index": frame_index,
        "map_revision": revision,
        "source_stamp_s": source_stamp_s,
    }
    metadata = {
        "avoidance_overlay": dict(
            common,
            frame_id=current_geometry[0],
            stamp_s=current_stamp_s,
            forecast_horizon_s=""),
    }
    for horizon, overlay in zip(horizons, overlays):
        name = avoidance_forecast_layer_name(horizon)
        if name in maps:
            raise ValueError("duplicate avoidance forecast horizon")
        if _avoidance_grid_geometry(overlay) != current_geometry:
            raise ValueError("avoidance forecast geometry differs from current grid")
        overlay_stamp_s = float(overlay.header.stamp.to_sec())
        if (not math.isfinite(overlay_stamp_s)
                or abs(overlay_stamp_s - source_stamp_s) > 1.0e-9):
            raise ValueError("avoidance forecast grid stamp differs from source")
        maps[name] = overlay
        metadata[name] = dict(
            common, frame_id=current_geometry[0], stamp_s=overlay_stamp_s,
            forecast_horizon_s=float(horizon))
    return maps, metadata


def replace_avoidance_snapshot_layers(latest_maps, latest_metadata,
                                      bundle_maps, bundle_metadata):
    """Atomically replace current and forecast keys, removing stale horizons."""
    if set(bundle_maps) != set(bundle_metadata):
        raise ValueError("avoidance map/metadata bundle keys differ")
    for name in set(latest_maps) | set(latest_metadata):
        if name == "avoidance_overlay" or name.startswith(
                "avoidance_forecast_"):
            latest_maps.pop(name, None)
            latest_metadata.pop(name, None)
    latest_maps.update(bundle_maps)
    latest_metadata.update(bundle_metadata)


STATIONARY_FORECAST_REQUIRED_PATHS = (
    "analysis/deformation_route_maps/03_forecast_anchor_current_overlay.png",
    "analysis/deformation_route_maps/04_forecast_0.5s.png",
    "analysis/deformation_route_maps/05_forecast_1.0s.png",
    "analysis/deformation_route_maps/06_forecast_2.0s.png",
    "analysis/deformation_route_maps/14_runtime_forecast_self_consistency.png",
    "analysis/deformation_route_maps/forecast_anchor_current_overlay.pgm",
    "analysis/deformation_route_maps/forecast_0.5s.pgm",
    "analysis/deformation_route_maps/forecast_1.0s.pgm",
    "analysis/deformation_route_maps/forecast_2.0s.pgm",
    "analysis/deformation_route_maps/all_map_arrays.npz",
)


# A timeline checkpoint is useful only when the data product which the paper
# compares can be reconstructed from that same recorder snapshot.  Forecasts
# remain optional at an individual checkpoint (the producer may legitimately
# publish current-only while warming up), but the current avoidance layer and
# the exact detector-input cloud are mandatory.
CHECKPOINT_REQUIRED_MAP_NAMES = ("avoidance_overlay",)
CHECKPOINT_REQUIRED_CLOUD_NAMES = ("detector_input",)


MOTION_TRUTH_SCHEMA_VERSION = 2
MOTION_TRUTH_MODEL_IDS = {
    "m01": 1, "m02": 2, "m03": 3,
    "m04": 4, "m05": 5, "m06": 6,
}
MOTION_TRUTH_STATIONARY_CHECKPOINTS_S = (0.0, 10.0, 50.0, 60.0)
MOTION_TRUTH_CHECKPOINT_TOLERANCE_S = 0.25
MOTION_TRUTH_STATES = {
    "WAITING", "RUNNING", "HOLDING", "COMPLETED", "STOPPED",
}


DUAL_LIDAR_SUCCESS_THRESHOLDS = {
    "schema_version": 5,
    "point_time_mode": "snapshot",
    "configured_sensor_update_rate_hz": 10.0,
    "configured_min_range_m": 0.5,
    "configured_max_range_m": 20.0,
    "configured_ray_samples_per_sensor": 20000,
    "minimum_realized_measurement_hz": 4.0,
    "maximum_realized_measurement_hz": 5.5,
    "minimum_accepted_measurement_interval_s": 0.10,
    "maximum_accepted_measurement_interval_s": 0.35,
    "primary_downsample": 1,
    "secondary_downsample": 2,
    "overlap_voxel_size_m": 0.20,
    "overlap_evaluation_pair_stride": 10,
    "minimum_overlap_sample_count": 10,
    "maximum_sync_delta_ms": 25.0,
    "minimum_pairing_rate": 0.90,
    "minimum_mean_primary_overlap_fraction": 0.20,
    "minimum_mean_secondary_overlap_fraction": 0.20,
    "minimum_mean_voxel_jaccard": 0.10,
}


def configured_dual_lidar_thresholds(sensor_update_rate_hz, min_range_m,
                                     max_range_m, ray_samples_per_sensor,
                                     primary_downsample,
                                     secondary_downsample):
    """Return a validated copy of the formal, launch-specific contract."""
    rate = float(sensor_update_rate_hz)
    minimum = float(min_range_m)
    maximum = float(max_range_m)
    samples = int(ray_samples_per_sensor)
    primary = int(primary_downsample)
    secondary = int(secondary_downsample)
    if (not math.isfinite(rate) or rate <= 0.0
            or not math.isfinite(minimum) or not math.isfinite(maximum)
            or minimum < 0.0 or minimum >= maximum
            or samples < 1 or primary < 1 or secondary < 1):
        raise ValueError("invalid configured dual-LiDAR contract")
    configured = dict(DUAL_LIDAR_SUCCESS_THRESHOLDS)
    configured.update({
        "configured_sensor_update_rate_hz": rate,
        "configured_min_range_m": minimum,
        "configured_max_range_m": maximum,
        "configured_ray_samples_per_sensor": samples,
        "primary_downsample": primary,
        "secondary_downsample": secondary,
    })
    return configured


def dual_lidar_success_contract(latest, thresholds=None):
    """Evaluate one explicitly configured dual-LiDAR E2E contract.

    The merge diagnostic counters form one internally consistent cumulative
    snapshot.  In particular, evaluating every configured Nth accepted pair
    implies ``attempted == floor(pairs_accepted / N)``; allowing an arbitrary
    end-of-run tolerance would make a failed or skipped overlap computation
    indistinguishable from a timer boundary.

    ``thresholds`` is copied before use.  This lets a launch override the two
    deterministic ray strides (for example 1/2 versus 2/2) without the
    recorder silently continuing to judge point counts against 1/2.  The
    formal defaults remain the measured 1/2 profile.
    """
    latest = latest if isinstance(latest, dict) else {}
    expected = dict(DUAL_LIDAR_SUCCESS_THRESHOLDS)
    if thresholds is not None:
        expected.update(dict(thresholds))

    def number(key):
        return _optional_float(latest, (key,))

    def integer(key):
        value = number(key)
        if value is None or abs(value - round(value)) > 1.0e-9:
            return None
        return int(round(value))

    pairs = integer("pairs_accepted")
    schema_version = integer("schema_version")
    configured_sensor_update_rate_hz = number(
        "configured_sensor_update_rate_hz")
    configured_min_range_m = number("configured_min_range_m")
    configured_max_range_m = number("configured_max_range_m")
    configured_ray_samples = integer(
        "configured_ray_samples_per_sensor")
    primary_downsample = integer("primary_downsample")
    secondary_downsample = integer("secondary_downsample")
    primary_received = integer("primary_received")
    secondary_received = integer("secondary_received")
    stride = integer("overlap_evaluation_pair_stride")
    attempted = integer("overlap_evaluations_attempted")
    failed = integer("overlap_evaluations_failed")
    samples = integer("overlap_evaluation_sample_count")
    primary_queue_dropped = integer("primary_queue_dropped")
    secondary_queue_dropped = integer("secondary_queue_dropped")
    maximum_sync_delta_ns = number("maximum_sync_delta_ns")
    first_measurement_time_ns = number(
        "first_accepted_measurement_time_ns")
    last_measurement_time_ns = number(
        "last_accepted_measurement_time_ns")
    measurement_interval_count = integer(
        "accepted_measurement_interval_count")
    nonpositive_measurement_interval_count = integer(
        "nonpositive_accepted_measurement_interval_count")
    minimum_measurement_interval_ns = number(
        "minimum_accepted_measurement_interval_ns")
    maximum_measurement_interval_ns = number(
        "maximum_accepted_measurement_interval_ns")
    mean_measurement_interval_ns = number(
        "mean_accepted_measurement_interval_ns")
    realized_measurement_hz = number("realized_measurement_hz")
    mean_primary = number("mean_primary_overlap_fraction")
    mean_secondary = number("mean_secondary_overlap_fraction")
    mean_jaccard = number("mean_voxel_jaccard")
    overlap_voxel_size_m = number("overlap_voxel_size_m")
    last_primary_point_count = integer("last_primary_point_count")
    last_secondary_point_count = integer("last_secondary_point_count")
    last_merged_point_count = integer("last_merged_point_count")
    last_total_raw_input_point_count = integer(
        "last_total_raw_input_point_count")
    last_compacted_output_point_count = integer(
        "last_compacted_output_point_count")
    last_removed_no_return_sentinel_count = integer(
        "last_removed_no_return_sentinel_count")
    last_compacted_raw_ratio = number(
        "last_compacted_output_to_total_raw_input_ratio")
    last_secondary_primary_ratio = number(
        "last_secondary_to_primary_point_ratio")
    last_merged_primary_ratio = number("last_merged_to_primary_point_ratio")
    primary_rate = (pairs / float(primary_received)
                    if pairs is not None and primary_received else None)
    secondary_rate = (pairs / float(secondary_received)
                      if pairs is not None and secondary_received else None)
    expected_overlap_stride = int(expected["overlap_evaluation_pair_stride"])
    expected_attempts = (
        pairs // expected_overlap_stride if pairs is not None else None)
    expected_primary_count = (
        (int(expected["configured_ray_samples_per_sensor"])
         + int(expected["primary_downsample"]) - 1)
        // int(expected["primary_downsample"]))
    expected_secondary_count = (
        (int(expected["configured_ray_samples_per_sensor"])
         + int(expected["secondary_downsample"]) - 1)
        // int(expected["secondary_downsample"]))
    expected_total_raw_input_count = (
        expected_primary_count + expected_secondary_count)
    expected_secondary_primary_ratio = (
        expected_secondary_count / float(expected_primary_count))
    derived_compacted_raw_ratio = (
        last_compacted_output_point_count /
        float(last_total_raw_input_point_count)
        if (last_compacted_output_point_count is not None
            and last_total_raw_input_point_count is not None
            and last_total_raw_input_point_count > 0) else None)
    derived_compacted_primary_ratio = (
        last_compacted_output_point_count / float(last_primary_point_count)
        if (last_compacted_output_point_count is not None
            and last_primary_point_count is not None
            and last_primary_point_count > 0) else None)
    expected_measurement_intervals = (
        pairs - 1 if pairs is not None and pairs >= 1 else None)
    measurement_elapsed_ns = (
        last_measurement_time_ns - first_measurement_time_ns
        if (first_measurement_time_ns is not None
            and last_measurement_time_ns is not None) else None)
    derived_mean_measurement_interval_ns = (
        measurement_elapsed_ns / float(measurement_interval_count)
        if (measurement_elapsed_ns is not None
            and measurement_elapsed_ns > 0.0
            and measurement_interval_count is not None
            and measurement_interval_count > 0) else None)
    derived_realized_measurement_hz = (
        1.0e9 / derived_mean_measurement_interval_ns
        if (derived_mean_measurement_interval_ns is not None
            and derived_mean_measurement_interval_ns > 0.0) else None)

    def close_frequency_value(value, expected):
        if value is None or expected is None:
            return False
        return abs(value - expected) <= max(
            1.0e-6, 1.0e-9 * max(abs(value), abs(expected)))

    checks = {
        "diagnostic_available": bool(latest),
        "schema_version_matches_expected": (
            schema_version == int(expected["schema_version"])),
        "secondary_enabled": (_csv_bool(latest.get("secondary_enabled", ""))
                              if latest else False),
        "zero_no_return_compaction_enabled": (
            _csv_bool(latest.get("zero_no_return_compaction_enabled", ""))
            if latest else False),
        "point_time_mode_matches_expected": (
            str(latest.get("point_time_mode", "")) ==
            str(expected["point_time_mode"])),
        "configured_sensor_update_rate_matches_expected": (
            configured_sensor_update_rate_hz is not None
            and abs(configured_sensor_update_rate_hz - float(
                expected["configured_sensor_update_rate_hz"])) <= 1.0e-12),
        "configured_min_range_matches_expected": (
            configured_min_range_m is not None
            and abs(configured_min_range_m - float(
                expected["configured_min_range_m"])) <= 1.0e-12),
        "configured_max_range_matches_expected": (
            configured_max_range_m is not None
            and abs(configured_max_range_m - float(
                expected["configured_max_range_m"])) <= 1.0e-12),
        "configured_ray_samples_match_expected": (
            configured_ray_samples == int(
                expected["configured_ray_samples_per_sensor"])),
        "primary_downsample_matches_expected": (
            primary_downsample == int(expected["primary_downsample"])),
        "secondary_downsample_matches_expected": (
            secondary_downsample == int(expected["secondary_downsample"])),
        "overlap_voxel_size_matches_expected": (
            overlap_voxel_size_m is not None
            and abs(overlap_voxel_size_m - float(
                expected["overlap_voxel_size_m"])) <= 1.0e-12),
        "overlap_stride_matches_expected": stride == expected_overlap_stride,
        "last_primary_point_count_matches_stride": (
            last_primary_point_count == expected_primary_count),
        "last_secondary_point_count_matches_stride": (
            last_secondary_point_count == expected_secondary_count),
        "last_total_raw_input_point_count_matches_stride": (
            last_total_raw_input_point_count ==
            expected_total_raw_input_count),
        "last_secondary_primary_point_ratio_matches_stride": (
            last_secondary_primary_ratio is not None
            and abs(last_secondary_primary_ratio -
                    expected_secondary_primary_ratio) <= 1.0e-12),
        # Compacted output is scene-dependent.  It must never be compared to
        # the configured ray-table size as if every simulated ray returned.
        "compacted_output_count_positive_and_bounded_by_raw": (
            last_compacted_output_point_count is not None
            and last_total_raw_input_point_count is not None
            and 0 < last_compacted_output_point_count
            <= last_total_raw_input_point_count),
        "legacy_merged_count_matches_explicit_compacted_output": (
            last_merged_point_count == last_compacted_output_point_count),
        "removed_sentinel_count_accounting_exact": (
            last_removed_no_return_sentinel_count is not None
            and last_total_raw_input_point_count is not None
            and last_compacted_output_point_count is not None
            and last_removed_no_return_sentinel_count ==
            last_total_raw_input_point_count -
            last_compacted_output_point_count),
        "compacted_raw_ratio_accounting_exact": (
            last_compacted_raw_ratio is not None
            and derived_compacted_raw_ratio is not None
            and abs(last_compacted_raw_ratio -
                    derived_compacted_raw_ratio) <= 1.0e-12),
        "legacy_merged_primary_ratio_matches_compacted_output": (
            last_merged_primary_ratio is not None
            and derived_compacted_primary_ratio is not None
            and abs(last_merged_primary_ratio -
                    derived_compacted_primary_ratio) <= 1.0e-12),
        "overlap_sample_count_meets_minimum": (
            samples is not None and samples >= int(
                expected["minimum_overlap_sample_count"])),
        "overlap_evaluations_failed_zero": failed == 0,
        "overlap_attempt_accounting_exact": (
            attempted is not None and samples is not None and failed is not None
            and attempted == samples + failed),
        "overlap_schedule_accounting_exact": (
            attempted is not None and expected_attempts is not None
            and attempted == expected_attempts),
        "maximum_sync_delta_within_limit": (
            maximum_sync_delta_ns is not None
            and maximum_sync_delta_ns <= float(
                expected["maximum_sync_delta_ms"]) * 1.0e6 + 1.0e-6),
        # The formal run must contain enough accepted measurements to define a
        # cadence.  Frequency is computed from Gazebo measurement epochs, not
        # recorder wall time or ROS callback latency.
        "effective_frequency_has_at_least_two_measurements": (
            pairs is not None and pairs >= 2),
        "effective_interval_accounting_exact": (
            measurement_interval_count is not None
            and expected_measurement_intervals is not None
            and measurement_interval_count == expected_measurement_intervals),
        "effective_intervals_strictly_positive": (
            nonpositive_measurement_interval_count == 0),
        "minimum_effective_interval_within_limit": (
            minimum_measurement_interval_ns is not None
            and minimum_measurement_interval_ns >= float(
                expected["minimum_accepted_measurement_interval_s"])
            * 1.0e9 - 1.0e-6),
        "maximum_effective_interval_within_limit": (
            maximum_measurement_interval_ns is not None
            and maximum_measurement_interval_ns <= float(
                expected["maximum_accepted_measurement_interval_s"])
            * 1.0e9 + 1.0e-6),
        "effective_interval_extrema_contain_mean": (
            minimum_measurement_interval_ns is not None
            and maximum_measurement_interval_ns is not None
            and mean_measurement_interval_ns is not None
            and 0.0 < minimum_measurement_interval_ns
            <= mean_measurement_interval_ns
            <= maximum_measurement_interval_ns),
        "accepted_measurement_epoch_range_positive": (
            measurement_elapsed_ns is not None
            and measurement_elapsed_ns > 0.0),
        "mean_effective_measurement_interval_consistent": (
            mean_measurement_interval_ns is not None
            and mean_measurement_interval_ns > 0.0
            and close_frequency_value(
                mean_measurement_interval_ns,
                derived_mean_measurement_interval_ns)),
        "realized_measurement_hz_positive_and_consistent": (
            realized_measurement_hz is not None
            and realized_measurement_hz > 0.0
            and close_frequency_value(
                realized_measurement_hz,
                derived_realized_measurement_hz)),
        "realized_measurement_hz_within_configured_bounds": (
            realized_measurement_hz is not None
            and float(expected["minimum_realized_measurement_hz"])
            - 1.0e-12 <= realized_measurement_hz
            <= float(expected["maximum_realized_measurement_hz"])
            + 1.0e-12),
        "primary_queue_drop_zero": primary_queue_dropped == 0,
        "secondary_queue_drop_zero": secondary_queue_dropped == 0,
        "primary_pairing_rate_meets_minimum": (
            primary_rate is not None and primary_rate >= float(
                expected["minimum_pairing_rate"]) - 1.0e-12),
        "secondary_pairing_rate_meets_minimum": (
            secondary_rate is not None and secondary_rate >= float(
                expected["minimum_pairing_rate"]) - 1.0e-12),
        "mean_primary_overlap_meets_minimum": (
            mean_primary is not None and mean_primary >= float(
                expected["minimum_mean_primary_overlap_fraction"])
            - 1.0e-12),
        "mean_secondary_overlap_meets_minimum": (
            mean_secondary is not None and mean_secondary >= float(
                expected["minimum_mean_secondary_overlap_fraction"])
            - 1.0e-12),
        "mean_voxel_jaccard_meets_minimum": (
            mean_jaccard is not None and mean_jaccard >= float(
                expected["minimum_mean_voxel_jaccard"]) - 1.0e-12),
    }
    errors = [name for name, satisfied in checks.items() if not satisfied]
    reported_thresholds = dict(expected)
    reported_thresholds.update({
        "expected_primary_point_count": expected_primary_count,
        "expected_secondary_point_count": expected_secondary_count,
        "expected_total_raw_input_point_count": (
            expected_total_raw_input_count),
        "maximum_compacted_output_point_count": (
            expected_total_raw_input_count),
        "expected_secondary_to_primary_point_ratio": (
            expected_secondary_primary_ratio),
    })
    return {
        "status": "COMPLETED" if not errors else "FAILED",
        "checks": checks,
        "errors": errors,
        "thresholds": reported_thresholds,
        "observed": {
            "schema_version": schema_version,
            "zero_no_return_compaction_enabled": (
                _csv_bool(latest.get(
                    "zero_no_return_compaction_enabled", ""))
                if latest else False),
            "point_time_mode": latest.get("point_time_mode", ""),
            "configured_sensor_update_rate_hz": (
                configured_sensor_update_rate_hz),
            "configured_min_range_m": configured_min_range_m,
            "configured_max_range_m": configured_max_range_m,
            "configured_ray_samples_per_sensor": configured_ray_samples,
            "primary_downsample": primary_downsample,
            "secondary_downsample": secondary_downsample,
            "overlap_voxel_size_m": overlap_voxel_size_m,
            "pairs_accepted": pairs,
            "primary_received": primary_received,
            "secondary_received": secondary_received,
            "primary_pairing_rate": primary_rate,
            "secondary_pairing_rate": secondary_rate,
            "overlap_evaluation_pair_stride": stride,
            "overlap_evaluations_attempted": attempted,
            "overlap_evaluations_failed": failed,
            "overlap_evaluation_sample_count": samples,
            "expected_overlap_attempts_floor_pairs_div_stride": (
                expected_attempts),
            "expected_primary_point_count": expected_primary_count,
            "expected_secondary_point_count": expected_secondary_count,
            "expected_total_raw_input_point_count": (
                expected_total_raw_input_count),
            "maximum_sync_delta_ms": (
                maximum_sync_delta_ns / 1.0e6
                if maximum_sync_delta_ns is not None else None),
            "first_accepted_measurement_time_ns": first_measurement_time_ns,
            "last_accepted_measurement_time_ns": last_measurement_time_ns,
            "accepted_measurement_interval_count": (
                measurement_interval_count),
            "expected_measurement_interval_count": (
                expected_measurement_intervals),
            "nonpositive_accepted_measurement_interval_count": (
                nonpositive_measurement_interval_count),
            "minimum_accepted_measurement_interval_ns": (
                minimum_measurement_interval_ns),
            "maximum_accepted_measurement_interval_ns": (
                maximum_measurement_interval_ns),
            "mean_accepted_measurement_interval_ns": (
                mean_measurement_interval_ns),
            "derived_mean_accepted_measurement_interval_ns": (
                derived_mean_measurement_interval_ns),
            "realized_measurement_hz": realized_measurement_hz,
            "derived_realized_measurement_hz": (
                derived_realized_measurement_hz),
            "primary_queue_dropped": primary_queue_dropped,
            "secondary_queue_dropped": secondary_queue_dropped,
            "mean_primary_overlap_fraction": mean_primary,
            "mean_secondary_overlap_fraction": mean_secondary,
            "mean_voxel_jaccard": mean_jaccard,
            "last_primary_point_count": last_primary_point_count,
            "last_secondary_point_count": last_secondary_point_count,
            "last_merged_point_count": last_merged_point_count,
            "last_total_raw_input_point_count": (
                last_total_raw_input_point_count),
            "last_compacted_output_point_count": (
                last_compacted_output_point_count),
            "last_removed_no_return_sentinel_count": (
                last_removed_no_return_sentinel_count),
            "last_compacted_output_to_total_raw_input_ratio": (
                last_compacted_raw_ratio),
            "last_secondary_to_primary_point_ratio": (
                last_secondary_primary_ratio),
            "last_merged_to_primary_point_ratio": last_merged_primary_ratio,
        },
    }


def checkpoint_artifact_contract(output_directory, requested_checkpoints_s):
    """Resolve and verify every required timeline map/cloud artifact."""
    def scalar(stored, key):
        if key not in stored.files:
            raise ValueError("missing NPZ field " + key)
        values = np.asarray(stored[key])
        if values.size != 1:
            raise ValueError("non-scalar NPZ field " + key)
        value = values.reshape(-1)[0].item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return value

    def same_float(actual, expected):
        try:
            left = float(actual)
            right = float(expected)
        except (TypeError, ValueError):
            return False
        return (math.isfinite(left) and math.isfinite(right)
                and abs(left - right) <= 1.0e-9)

    requested = sorted(set(float(value) for value in requested_checkpoints_s))
    if not requested:
        return {"status": "NOT_REQUIRED", "required_paths": [],
                "checkpoints": [], "errors": []}
    relative_timeline = "events/timeline_checkpoints.csv"
    required_paths = [relative_timeline]
    errors = []
    rows = []
    timeline = os.path.join(output_directory, relative_timeline)
    if os.path.isfile(timeline):
        try:
            with open(timeline, newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        except (OSError, UnicodeError, csv.Error) as error:
            errors.append("timeline_unreadable: {}".format(error))
    else:
        errors.append("timeline_missing")
    checkpoint_results = []
    for checkpoint in requested:
        matches = []
        for row in rows:
            value = _optional_float(row, ("target_experiment_elapsed_s",))
            if value is not None and abs(value - checkpoint) <= 1.0e-9:
                matches.append(row)
        expected_label = "CHECKPOINT_{:06.2f}s".format(checkpoint)
        result = {"target_experiment_elapsed_s": checkpoint,
                  "expected_snapshot_label": expected_label,
                  "timeline_row_count": len(matches), "artifacts": {}}
        if len(matches) != 1:
            errors.append("checkpoint_{:.2f}_timeline_row_count={}".format(
                checkpoint, len(matches)))
        elif matches[0].get("snapshot_label", "") != expected_label:
            errors.append("checkpoint_{:.2f}_snapshot_label_mismatch".format(
                checkpoint))
        timeline_row = matches[0] if len(matches) == 1 else {}
        declared_forecast_horizons = []
        declared_forecast_names = {}
        raw_forecast_horizons = str(timeline_row.get(
            "avoidance_forecast_horizons_s", "")).strip()
        if raw_forecast_horizons:
            for token in raw_forecast_horizons.split("|"):
                try:
                    horizon = float(token)
                    name = avoidance_forecast_layer_name(horizon)
                except (TypeError, ValueError) as error:
                    errors.append(
                        "checkpoint_{:.2f}_forecast_declaration_invalid={}".format(
                            checkpoint, error))
                    continue
                if name in declared_forecast_names:
                    errors.append(
                        "checkpoint_{:.2f}_forecast_declaration_duplicate={}".format(
                            checkpoint, token))
                    continue
                declared_forecast_names[name] = horizon
                declared_forecast_horizons.append(horizon)
        result["declared_forecast_horizons_s"] = (
            declared_forecast_horizons)
        safe_label = "".join(
            character if character.isalnum() else "_"
            for character in expected_label)[:48]
        required_checkpoint_map_names = (
            tuple(CHECKPOINT_REQUIRED_MAP_NAMES)
            + tuple(name for name, _ in sorted(
                declared_forecast_names.items(), key=lambda item: item[1])))
        for category, names in (
                ("maps", required_checkpoint_map_names),
                ("clouds", CHECKPOINT_REQUIRED_CLOUD_NAMES)):
            for name in names:
                pattern = os.path.join(
                    output_directory, "snapshots", category,
                    "*_{}_{}.npz".format(safe_label, name))
                paths = sorted(glob.glob(pattern))
                key = "{}/{}".format(category, name)
                result["artifacts"][key] = [
                    os.path.relpath(path, output_directory) for path in paths]
                if len(paths) != 1:
                    errors.append(
                        "checkpoint_{:.2f}_{}_artifact_count={}".format(
                            checkpoint, key.replace("/", "_"), len(paths)))
                else:
                    relative_path = os.path.relpath(paths[0], output_directory)
                    required_paths.append(relative_path)
                    try:
                        with np.load(paths[0], allow_pickle=False) as stored:
                            if category == "maps":
                                data = stored["data"]
                                if data.ndim != 2 or data.size == 0:
                                    raise ValueError(
                                        "map data is not a nonempty 2-D array")
                                expected_horizon = (
                                    timeline_row.get(
                                        "avoidance_current_forecast_horizon_s",
                                        "")
                                    if name == "avoidance_overlay" else
                                    declared_forecast_names[name])
                                expected_stamp = (
                                    timeline_row.get(
                                        "avoidance_current_map_stamp_s", "")
                                    if name == "avoidance_overlay" else
                                    timeline_row.get(
                                        "avoidance_source_stamp_s", ""))
                                comparisons = {
                                    "source_snapshot_id": (
                                        scalar(stored, "source_snapshot_id"),
                                        timeline_row.get(
                                            "avoidance_source_snapshot_id", ""),
                                        "text"),
                                    "source_accepted_frame_index": (
                                        scalar(stored,
                                               "source_accepted_frame_index"),
                                        timeline_row.get(
                                            "avoidance_source_accepted_frame_index",
                                            ""), "float"),
                                    "map_revision": (
                                        scalar(stored, "map_revision"),
                                        timeline_row.get(
                                            "avoidance_map_revision", ""),
                                        "float"),
                                    "forecast_horizon_s": (
                                        scalar(stored, "forecast_horizon_s"),
                                        expected_horizon,
                                        ("text" if name == "avoidance_overlay"
                                         else "float")),
                                    "source_stamp_s": (
                                        scalar(stored, "source_stamp_s"),
                                        timeline_row.get(
                                            "avoidance_source_stamp_s", ""),
                                        "float"),
                                    "stamp_s": (
                                        scalar(stored, "stamp_s"),
                                        expected_stamp,
                                        "float"),
                                    "frame_id": (
                                        scalar(stored, "frame_id"),
                                        timeline_row.get(
                                            "avoidance_current_map_frame_id", ""),
                                        "text"),
                                }
                            else:
                                points = stored["points"]
                                if (points.ndim != 2 or points.shape[0] == 0
                                        or points.shape[1] < 3):
                                    raise ValueError(
                                        "cloud points are not nonempty N-by-3+")
                                comparisons = {
                                    "stamp_s": (
                                        scalar(stored, "stamp_s"),
                                        timeline_row.get(
                                            "detector_input_cloud_stamp_s", ""),
                                        "float"),
                                    "frame_id": (
                                        scalar(stored, "frame_id"),
                                        timeline_row.get(
                                            "detector_input_cloud_frame_id", ""),
                                        "text"),
                                }
                            identities = {}
                            for field, (actual, expected, value_type) in (
                                    comparisons.items()):
                                identities[field] = {
                                    "npz": actual, "timeline": expected}
                                matches_identity = (
                                    same_float(actual, expected)
                                    if value_type == "float"
                                    else str(actual) == str(expected))
                                if not matches_identity:
                                    errors.append(
                                        "checkpoint_{:.2f}_{}_{}_mismatch".format(
                                            checkpoint,
                                            key.replace("/", "_"), field))
                            result["artifacts"][key + "_identity"] = identities
                    except (KeyError, OSError, ValueError, EOFError) as error:
                        errors.append(
                            "checkpoint_{:.2f}_{}_invalid={}".format(
                                checkpoint, key.replace("/", "_"), error))
        checkpoint_results.append(result)
    return {
        "status": "COMPLETED" if not errors else "FAILED",
        "required_paths": required_paths,
        "checkpoints": checkpoint_results,
        "errors": errors,
    }


def slerp_quaternion(first, second, fraction):
    """Shortest-arc normalized quaternion interpolation in ROS xyzw order."""
    first = np.asarray(normalized_quaternion(first), dtype=float)
    second = np.asarray(normalized_quaternion(second), dtype=float)
    alpha = min(1.0, max(0.0, float(fraction)))
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return normalized_quaternion(first + alpha * (second - first))
    angle = math.acos(dot)
    sine = math.sin(angle)
    return normalized_quaternion(
        math.sin((1.0 - alpha) * angle) / sine * first
        + math.sin(alpha * angle) / sine * second)


def _prepare_model_truth_history(history):
    """Sort one truth history once and retain its bisect key array.

    The public alignment helper accepts arbitrary iterables and therefore
    still prepares on each independent call.  Final analysis, however, joins
    many spatial estimates from the same detector frame against the same six
    histories.  It uses this immutable prepared pair once per model instead of
    repeatedly filtering, sorting and rebuilding timestamps for every sample.
    """
    valid = [entry for entry in history
             if math.isfinite(float(entry.get("sim_time_s", float("nan"))))]
    valid.sort(key=lambda entry: float(entry["sim_time_s"]))
    return tuple(valid), tuple(float(entry["sim_time_s"]) for entry in valid)


def _align_prepared_model_truth_at_time(
        prepared_history, target_time_s, max_zoh_residual_s=0.25):
    """Align against an immutable ``(history, timestamps)`` prepared pair."""
    valid, times = prepared_history
    try:
        target = float(target_time_s)
        maximum_residual = float(max_zoh_residual_s)
    except (TypeError, ValueError):
        return {"available": False, "method": "INVALID_TIME"}
    if (not math.isfinite(target) or not math.isfinite(maximum_residual)
            or maximum_residual < 0.0):
        return {"available": False, "method": "INVALID_TIME"}
    if not valid:
        return {"available": False, "method": "NO_TRUTH_HISTORY"}
    right = bisect.bisect_left(times, target)
    if right < len(times) and abs(times[right] - target) <= 1.0e-12:
        result = dict(valid[right])
        result.update({
            "available": True, "method": "EXACT",
            "time_offset_s": times[right] - target,
            "before_time_s": times[right], "after_time_s": times[right],
            "bracket_span_s": 0.0, "zoh_bound_s": maximum_residual})
        return result
    if 0 < right < len(valid):
        before = valid[right - 1]
        after = valid[right]
        before_time = times[right - 1]
        after_time = times[right]
        span = after_time - before_time
        if span <= 0.0:
            return {"available": False, "method": "INVALID_TRUTH_ORDER"}
        alpha = (target - before_time) / span
        result = {
            "available": True, "method": "LINEAR_SE3",
            "sim_time_s": target, "time_offset_s": 0.0,
            "before_time_s": before_time, "after_time_s": after_time,
            "bracket_span_s": span, "zoh_bound_s": maximum_residual,
        }
        for key in ("position", "translation", "aabb_min", "aabb_max"):
            first = before.get(key)
            second = after.get(key)
            if _finite_vector3(first) and _finite_vector3(second):
                result[key] = tuple(
                    (1.0 - alpha) * float(first[index])
                    + alpha * float(second[index]) for index in range(3))
        first_orientation = before.get("orientation")
        second_orientation = after.get("orientation")
        if (isinstance(first_orientation, (list, tuple))
                and isinstance(second_orientation, (list, tuple))
                and len(first_orientation) == 4
                and len(second_orientation) == 4):
            result["orientation"] = slerp_quaternion(
                first_orientation, second_orientation, alpha)
        for key in ("experiment_elapsed_s", "model_motion_elapsed_s"):
            first = before.get(key)
            second = after.get(key)
            try:
                result[key] = ((1.0 - alpha) * float(first)
                               + alpha * float(second))
            except (TypeError, ValueError):
                pass
        return result
    selected = valid[0] if right == 0 else valid[-1]
    selected_time = float(selected["sim_time_s"])
    offset = selected_time - target
    if abs(offset) > maximum_residual + 1.0e-12:
        return {
            "available": False, "method": "OUTSIDE_ZOH_BOUND",
            "time_offset_s": offset, "before_time_s": (
                selected_time if selected_time <= target else ""),
            "after_time_s": (selected_time if selected_time >= target else ""),
            "bracket_span_s": "", "zoh_bound_s": maximum_residual}
    result = dict(selected)
    result.update({
        "available": True,
        "method": "ZOH_BEFORE" if selected_time < target else "ZOH_AFTER",
        "time_offset_s": offset,
        "before_time_s": selected_time if selected_time < target else "",
        "after_time_s": selected_time if selected_time > target else "",
        "bracket_span_s": "", "zoh_bound_s": maximum_residual})
    return result


def align_model_truth_at_time(history, target_time_s, max_zoh_residual_s=0.25):
    """Align one model's truth history to a detector measurement timestamp.

    Bracketed samples use linear translation/position/AABB interpolation and
    quaternion SLERP. Outside the available interval, zero-order hold is
    allowed only within the explicit ``max_zoh_residual_s`` bound. Returned
    ``time_offset_s`` is aligned-sample time minus requested message time; it
    is exactly zero for interpolation and signed for ZOH.

    This compatibility entry point prepares an arbitrary history itself.
    Bulk final analysis uses the private prepared helper and a per-measurement-
    time cache, preserving these equations while avoiding repeated sorting.
    """
    return _align_prepared_model_truth_at_time(
        _prepare_model_truth_history(history), target_time_s,
        max_zoh_residual_s)


def associate_model(point_world, model_positions, tolerances):
    """Nearest normalized ellipsoid; labels are evaluation metadata only."""
    best = None
    best_score = float("inf")
    for name, position in model_positions.items():
        if name not in tolerances:
            continue
        xy_radius, z_radius = tolerances[name]
        planar = math.hypot(point_world[0] - position[0],
                            point_world[1] - position[1])
        vertical = abs(point_world[2] - position[2])
        if planar > xy_radius or vertical > z_radius:
            continue
        score = (planar / xy_radius) ** 2 + (vertical / z_radius) ** 2
        if score < best_score:
            best = name
            best_score = score
    return best


def _finite_vector3(values):
    return (isinstance(values, (list, tuple)) and len(values) == 3
            and all(math.isfinite(float(value)) for value in values))


def _point_aabb_distance(point, lower, upper):
    """Euclidean distance from a point to a closed axis-aligned box."""
    squared = 0.0
    for coordinate, minimum, maximum in zip(point, lower, upper):
        if coordinate < minimum:
            squared += (minimum - coordinate) ** 2
        elif coordinate > maximum:
            squared += (coordinate - maximum) ** 2
    return math.sqrt(squared)


def associate_model_for_evaluation(point_world, model_geometries, tolerances,
                                   aabb_margin_m=0.25):
    """Conservative truth association used only by the offline evaluator.

    Exact containment in a valid world AABB is the only AABB-based condition
    that may become ``MATCHED``.  The enlarged ``aabb_margin_m`` shell is kept
    solely to report nearby samples as ``MARGIN_ONLY_UNCERTAIN``; it must not
    turn a neighbouring static surface into a model hit.  A model without AABB
    metadata falls back to its bounded legacy ellipsoid.  Ambiguous support is
    deliberately never assigned to the nearest model.  The detector and both
    downstream maps never read this result.
    """
    exact_candidates = []
    margin_candidates = []
    fallback_candidates = []
    for name in sorted(model_geometries):
        geometry = model_geometries[name]
        position = geometry.get("position")
        lower = geometry.get("aabb_min")
        upper = geometry.get("aabb_max")
        if _finite_vector3(lower) and _finite_vector3(upper):
            lower = tuple(float(value) for value in lower)
            upper = tuple(float(value) for value in upper)
            if any(lower[index] > upper[index] for index in range(3)):
                continue
            distance = _point_aabb_distance(point_world, lower, upper)
            if distance <= 1.0e-12:
                exact_candidates.append((name, distance, "WORLD_AABB"))
            elif distance <= aabb_margin_m:
                margin_candidates.append(
                    (name, distance, "WORLD_AABB_MARGIN"))
            continue
        if not _finite_vector3(position) or name not in tolerances:
            continue
        xy_radius, z_radius = tolerances[name]
        if xy_radius <= 0.0 or z_radius <= 0.0:
            continue
        planar = math.hypot(point_world[0] - float(position[0]),
                            point_world[1] - float(position[1]))
        vertical = abs(point_world[2] - float(position[2]))
        if planar <= xy_radius and vertical <= z_radius:
            score = (planar / xy_radius) ** 2 + (vertical / z_radius) ** 2
            fallback_candidates.append(
                (name, score, "BOUNDED_ELLIPSOID"))

    # An exact physical-box containment is stronger than proximity to another
    # model's margin or an approximate fallback ellipsoid.  Multiple exact
    # boxes remain ambiguous because the recorder cannot invent ownership.
    candidates = exact_candidates if exact_candidates else fallback_candidates
    candidates.sort(key=lambda item: (item[1], item[0]))
    margin_candidates.sort(key=lambda item: (item[1], item[0]))
    names = [candidate[0] for candidate in candidates]
    if not candidates:
        if margin_candidates:
            return {
                "status": "MARGIN_ONLY_UNCERTAIN", "model_name": "",
                "candidates": [candidate[0]
                               for candidate in margin_candidates],
                "method": "WORLD_AABB_MARGIN",
                "score": margin_candidates[0][1]}
        return {
            "status": "UNMATCHED", "model_name": "", "candidates": [],
            "method": "NONE", "score": ""}
    if len(candidates) > 1:
        return {
            "status": "AMBIGUOUS", "model_name": "",
            "candidates": names, "method": "MIXED" if len({
                candidate[2] for candidate in candidates}) > 1 else candidates[0][2],
            "score": candidates[0][1]}
    return {
        "status": "MATCHED", "model_name": candidates[0][0],
        "candidates": names, "method": candidates[0][2],
        "score": candidates[0][1]}


def timeline_checkpoints_crossed(previous_elapsed_s, current_elapsed_s,
                                 pending_checkpoints_s):
    """Return pending experiment-time checkpoints reached by this truth row.

    The pending set is authoritative.  Returning every pending checkpoint not
    later than the current sample permits a failed NPZ write to be retried on
    the next truth callback instead of being silently declared complete.
    """
    if current_elapsed_s is None or not math.isfinite(float(current_elapsed_s)):
        return []
    current = float(current_elapsed_s)
    return [float(checkpoint) for checkpoint in sorted(pending_checkpoints_s)
            if float(checkpoint) <= current]


def experiment_elapsed_at_message(message_time_s, latest_truth_sim_time_s,
                                  latest_experiment_elapsed_s,
                                  latest_motion_state=""):
    """Map a source ROS/simulation stamp onto the experiment timeline.

    The motion truth topic supplies the affine anchor
    ``(sim_time, experiment_elapsed)``.  Using the detector callback receive
    time would add queueing and CPU latency to confirmation latency.  A
    completed/stopped experiment is clamped so late publications cannot move
    beyond the terminal 60 s timeline.
    """
    values = (message_time_s, latest_truth_sim_time_s,
              latest_experiment_elapsed_s)
    try:
        message_time, truth_time, elapsed = (float(value) for value in values)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value)
               for value in (message_time, truth_time, elapsed)):
        return None
    mapped = max(0.0, elapsed + message_time - truth_time)
    if str(latest_motion_state).upper() in ("COMPLETED", "STOPPED"):
        mapped = min(mapped, elapsed)
    return mapped


def summarize_map_ages(truth_sim_time_s, metadata_by_name):
    """Describe the asynchronous map snapshots used at a timeline checkpoint."""
    try:
        truth_time = float(truth_sim_time_s)
    except (TypeError, ValueError):
        truth_time = None
    if truth_time is not None and not math.isfinite(truth_time):
        truth_time = None
    stamps = []
    ages = []
    for name in sorted(metadata_by_name):
        metadata = metadata_by_name[name]
        try:
            stamp = float(metadata.get("stamp_s", ""))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(stamp):
            continue
        stamps.append("{}:{:.9f}".format(name, stamp))
        if truth_time is not None:
            age = truth_time - stamp
            ages.append((name, age))
    avoidance = metadata_by_name.get("avoidance_overlay", {})
    forecast_horizons = sorted(
        float(metadata.get("forecast_horizon_s"))
        for name, metadata in metadata_by_name.items()
        if name.startswith("avoidance_forecast_")
        and metadata.get("forecast_horizon_s") not in (None, ""))
    return {
        "map_stamp_summary_s": "|".join(stamps),
        "map_age_summary_s": "|".join(
            "{}:{:.9f}".format(name, age) for name, age in ages),
        "maximum_map_age_s": max((age for _, age in ages), default=""),
        "avoidance_source_accepted_frame_index": avoidance.get(
            "source_accepted_frame_index", ""),
        "avoidance_source_snapshot_id": avoidance.get(
            "source_snapshot_id", ""),
        "avoidance_map_revision": avoidance.get("map_revision", ""),
        "avoidance_forecast_horizons_s": "|".join(
            "{:.9g}".format(value) for value in forecast_horizons),
    }


def summarize_checkpoint_products(snapshot_metadata):
    """Expose exact required NPZ identity in the checkpoint timeline row."""
    snapshot_metadata = (snapshot_metadata
                         if isinstance(snapshot_metadata, dict) else {})
    maps = snapshot_metadata.get("maps", {})
    clouds = snapshot_metadata.get("clouds", {})
    current = maps.get("avoidance_overlay", {})
    detector_cloud = clouds.get("detector_input", {})
    return {
        "avoidance_source_stamp_s": current.get("source_stamp_s", ""),
        "avoidance_current_map_stamp_s": current.get("stamp_s", ""),
        "avoidance_current_map_frame_id": current.get("frame_id", ""),
        "avoidance_current_forecast_horizon_s": current.get(
            "forecast_horizon_s", ""),
        "detector_input_cloud_stamp_s": detector_cloud.get("stamp_s", ""),
        "detector_input_cloud_frame_id": detector_cloud.get("frame_id", ""),
    }


def _csv_bool(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def _optional_float(row, keys):
    for key in keys:
        value = row.get(key, "")
        if value not in (None, ""):
            try:
                result = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(result):
                return result
    return None


def summarize_gazebo_performance(rows):
    """Summarize recorder-only Gazebo telemetry without making a gate."""
    def summary(values):
        finite = [float(value) for value in values
                  if value is not None and math.isfinite(float(value))]
        return {
            "sample_count": len(finite),
            "minimum": min(finite) if finite else None,
            "mean": float(np.mean(finite)) if finite else None,
            "median": float(np.median(finite)) if finite else None,
            "p95": float(np.percentile(finite, 95)) if finite else None,
            "maximum": max(finite) if finite else None,
        }

    factors_by_message = {}
    per_sensor = defaultdict(lambda: {
        "sim_update_rate_hz": [], "real_update_rate_hz": [], "fps": []})
    for row in rows:
        factor = _optional_float(row, ("real_time_factor",))
        sequence = str(row.get("message_sequence", ""))
        stamp = str(row.get("message_time_s", ""))
        if factor is not None:
            factors_by_message[(sequence, stamp)] = factor
        name = str(row.get("sensor_name", ""))
        if not name:
            continue
        for field in ("sim_update_rate_hz", "real_update_rate_hz", "fps"):
            value = _optional_float(row, (field,))
            if value is not None:
                per_sensor[name][field].append(value)
    return {
        "message_sample_count": len(factors_by_message),
        "real_time_factor": summary(list(factors_by_message.values())),
        "sensors": {
            name: {field: summary(values)
                   for field, values in fields.items()}
            for name, fields in sorted(per_sensor.items())
        },
        "acceptance_gate": False,
        "interpretation": {
            "real_time_factor": (
                "simulation-time advance divided by wall-time advance; below "
                "one means Gazebo cannot run in real time"),
            "sim_update_rate_hz": "sensor updates per simulation second",
            "real_update_rate_hz": "sensor updates per wall-clock second",
            "gui_attribution": (
                "compare otherwise identical gui:=true and gui:=false runs; "
                "one run alone cannot uniquely attribute rendering cost"),
        },
    }


def _truth_float(container, key, errors, prefix="", minimum=None,
                 strictly_positive=False):
    label = prefix + key
    try:
        value = float(container[key])
    except (KeyError, TypeError, ValueError, OverflowError):
        errors.append(label + "_missing_or_non_numeric")
        return None
    if not math.isfinite(value):
        errors.append(label + "_non_finite")
        return None
    if strictly_positive and value <= 0.0:
        errors.append(label + "_not_positive")
    elif minimum is not None and value < minimum:
        errors.append(label + "_below_minimum")
    return value


def _truth_integer(container, key, errors, prefix="", minimum=None):
    value = _truth_float(container, key, errors, prefix)
    if value is None:
        return None
    if abs(value - round(value)) > 1.0e-9:
        errors.append(prefix + key + "_not_integer")
        return None
    result = int(round(value))
    if minimum is not None and result < minimum:
        errors.append(prefix + key + "_below_minimum")
    return result


def _truth_vector(container, key, length, errors, prefix=""):
    label = prefix + key
    values = container.get(key, None) if isinstance(container, dict) else None
    if not isinstance(values, (list, tuple)) or len(values) != length:
        errors.append(label + "_invalid_cardinality")
        return None
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError):
        errors.append(label + "_non_numeric")
        return None
    if not all(math.isfinite(value) for value in vector):
        errors.append(label + "_non_finite")
        return None
    return vector


def _truth_close(first, second, absolute=1.0e-9, relative=1.0e-8):
    return math.isclose(float(first), float(second), abs_tol=absolute,
                        rel_tol=relative)


def validate_motion_truth_batch(document):
    """Validate one evaluation-only payload from the Gazebo motion plugin.

    The formulas below deliberately mirror ``icra2027_motion_world_plugin``:
    translation error is the absolute difference of the two magnitudes and
    rotation error is the absolute difference of the two angles.  They are
    not vector error or a detector residual.  No value from this validator is
    sent back to deformation inference, mapping, or navigation.
    """
    errors = []
    if not isinstance(document, dict):
        return {"status": "FAILED", "errors": ["payload_not_object"]}

    schema_version = _truth_integer(document, "schema_version", errors,
                                    minimum=0)
    if (schema_version is not None
            and schema_version != MOTION_TRUTH_SCHEMA_VERSION):
        errors.append("schema_version_not_{}".format(
            MOTION_TRUTH_SCHEMA_VERSION))
    state = str(document.get("state", ""))
    if state not in MOTION_TRUTH_STATES:
        errors.append("state_invalid")
    motion_mode = str(document.get("motion_mode", ""))
    if motion_mode not in ("sequential", "simultaneous"):
        errors.append("motion_mode_invalid")

    sim_time = _truth_float(document, "sim_time", errors, minimum=0.0)
    experiment_elapsed = _truth_float(
        document, "experiment_elapsed", errors, minimum=0.0)
    waiting_elapsed = _truth_float(
        document, "waiting_elapsed", errors, minimum=0.0)
    motion_elapsed = _truth_float(
        document, "motion_elapsed", errors, minimum=0.0)
    duration = _truth_float(
        document, "per_model_duration", errors, strictly_positive=True)
    linear_speed = _truth_float(
        document, "linear_speed_mm_s", errors, minimum=0.0)
    angular_speed = _truth_float(
        document, "angular_speed_rad_s", errors, minimum=0.0)
    if (experiment_elapsed is not None and waiting_elapsed is not None
            and waiting_elapsed > experiment_elapsed + 1.0e-9):
        errors.append("waiting_elapsed_exceeds_experiment_elapsed")
    if (experiment_elapsed is not None and motion_elapsed is not None
            and motion_elapsed > experiment_elapsed + 1.0e-9):
        errors.append("motion_elapsed_exceeds_experiment_elapsed")
    if duration is not None and motion_elapsed is not None:
        multiplier = 6.0 if motion_mode == "sequential" else 1.0
        if motion_elapsed > multiplier * duration + 1.0e-9:
            errors.append("motion_elapsed_exceeds_plan_duration")

    models = document.get("models", None)
    if not isinstance(models, list):
        errors.append("models_not_array")
        models = []
    if len(models) != len(MOTION_TRUTH_MODEL_IDS):
        errors.append("models_cardinality_{}_expected_6".format(len(models)))
    seen_names = set()
    seen_ids = set()
    for index, model in enumerate(models):
        prefix = "model_{}_".format(index)
        if not isinstance(model, dict):
            errors.append(prefix + "not_object")
            continue
        name = str(model.get("model_name", ""))
        if not name:
            errors.append(prefix + "model_name_missing")
        elif name in seen_names:
            errors.append(prefix + "model_name_duplicate_" + name)
        seen_names.add(name)
        retro_id = _truth_integer(model, "laser_retro_id", errors, prefix,
                                  minimum=1)
        if retro_id is not None:
            if retro_id in seen_ids:
                errors.append(prefix + "laser_retro_id_duplicate_{}".format(
                    retro_id))
            seen_ids.add(retro_id)
        expected_id = MOTION_TRUTH_MODEL_IDS.get(name)
        if expected_id is None:
            errors.append(prefix + "unexpected_model_name_" + name)
        elif retro_id is not None and retro_id != expected_id:
            errors.append(prefix + "model_id_mismatch_{}_expected_{}".format(
                retro_id, expected_id))

        model_elapsed = _truth_float(
            model, "motion_elapsed", errors, prefix, minimum=0.0)
        if (model_elapsed is not None and duration is not None
                and model_elapsed > duration + 1.0e-9):
            errors.append(prefix + "motion_elapsed_exceeds_duration")
        position = _truth_vector(model, "position", 3, errors, prefix)
        orientation = _truth_vector(model, "orientation", 4, errors, prefix)
        lower = _truth_vector(model, "aabb_min", 3, errors, prefix)
        upper = _truth_vector(model, "aabb_max", 3, errors, prefix)
        translation = _truth_vector(model, "translation", 3, errors, prefix)
        expected_vector = _truth_vector(
            model, "expected_translation_vector", 3, errors, prefix)
        del position  # Finite presence is the contract; no semantic shortcut.
        if orientation is not None:
            quaternion_norm = math.sqrt(sum(value * value
                                            for value in orientation))
            if (quaternion_norm <= 1.0e-12
                    or abs(quaternion_norm - 1.0) > 1.0e-6):
                errors.append(prefix + "orientation_not_unit_quaternion")
        if lower is not None and upper is not None:
            if any(lower[axis] > upper[axis] for axis in range(3)):
                errors.append(prefix + "aabb_not_ordered")

        magnitude = _truth_float(
            model, "translation_magnitude", errors, prefix, minimum=0.0)
        expected_magnitude = _truth_float(
            model, "translation_expected", errors, prefix, minimum=0.0)
        translation_error = _truth_float(
            model, "translation_error", errors, prefix, minimum=0.0)
        rotation = _truth_float(
            model, "rotation_angle", errors, prefix, minimum=0.0)
        expected_rotation = _truth_float(
            model, "rotation_expected", errors, prefix, minimum=0.0)
        rotation_error = _truth_float(
            model, "rotation_error", errors, prefix, minimum=0.0)
        surface_rms = _truth_float(
            model, "surface_rms_displacement", errors, prefix, minimum=0.0)
        surface_max = _truth_float(
            model, "surface_max_displacement", errors, prefix, minimum=0.0)
        surface_count = _truth_integer(
            model, "surface_point_count", errors, prefix, minimum=0)

        if translation is not None and magnitude is not None:
            actual_norm = math.sqrt(sum(value * value for value in translation))
            if not _truth_close(magnitude, actual_norm):
                errors.append(prefix + "translation_magnitude_inconsistent")
        if expected_vector is not None and expected_magnitude is not None:
            expected_norm = math.sqrt(sum(value * value
                                          for value in expected_vector))
            if not _truth_close(expected_magnitude, expected_norm):
                errors.append(prefix + "translation_expected_inconsistent")
        if (magnitude is not None and expected_magnitude is not None
                and translation_error is not None
                and not _truth_close(
                    translation_error,
                    abs(magnitude - expected_magnitude))):
            errors.append(prefix + "translation_error_formula_inconsistent")
        if (rotation is not None and expected_rotation is not None
                and rotation_error is not None
                and not _truth_close(
                    rotation_error, abs(rotation - expected_rotation))):
            errors.append(prefix + "rotation_error_formula_inconsistent")
        if (surface_rms is not None and surface_max is not None
                and surface_rms > surface_max + 1.0e-9):
            errors.append(prefix + "surface_rms_exceeds_maximum")
        if (surface_count == 0 and magnitude is not None
                and surface_rms is not None and surface_max is not None
                and (not _truth_close(surface_rms, magnitude)
                     or not _truth_close(surface_max, magnitude))):
            errors.append(prefix + "zero_point_surface_not_translation")

    if seen_names != set(MOTION_TRUTH_MODEL_IDS):
        errors.append("model_name_set_mismatch")
    if seen_ids != set(MOTION_TRUTH_MODEL_IDS.values()):
        errors.append("laser_retro_id_set_mismatch")
    return {
        "status": "COMPLETED" if not errors else "FAILED",
        "errors": errors,
        "observed": {
            "schema_version": schema_version,
            "sim_time_s": sim_time,
            "experiment_elapsed_s": experiment_elapsed,
            "waiting_elapsed_s": waiting_elapsed,
            "motion_elapsed_s": motion_elapsed,
            "per_model_duration_s": duration,
            "linear_speed_mm_s": linear_speed,
            "angular_speed_rad_s": angular_speed,
            "model_names": sorted(seen_names),
            "laser_retro_ids": sorted(seen_ids),
        },
    }


def _truth_document_from_csv_batch(rows):
    first = rows[0]
    models = []
    for row in rows:
        models.append({
            "laser_retro_id": row.get("laser_retro_id", ""),
            "model_name": row.get("model_name", ""),
            "motion_elapsed": row.get("model_motion_elapsed_s", ""),
            "position": [row.get("position_x_m", ""),
                         row.get("position_y_m", ""),
                         row.get("position_z_m", "")],
            "orientation": [row.get("orientation_x", ""),
                            row.get("orientation_y", ""),
                            row.get("orientation_z", ""),
                            row.get("orientation_w", "")],
            "aabb_min": [row.get("world_aabb_min_x_m", ""),
                         row.get("world_aabb_min_y_m", ""),
                         row.get("world_aabb_min_z_m", "")],
            "aabb_max": [row.get("world_aabb_max_x_m", ""),
                         row.get("world_aabb_max_y_m", ""),
                         row.get("world_aabb_max_z_m", "")],
            "translation": [row.get("translation_x_m", ""),
                            row.get("translation_y_m", ""),
                            row.get("translation_z_m", "")],
            "expected_translation_vector": [
                row.get("expected_translation_x_m", ""),
                row.get("expected_translation_y_m", ""),
                row.get("expected_translation_z_m", "")],
            "translation_magnitude": row.get(
                "translation_magnitude_m", ""),
            "translation_expected": row.get("translation_expected_m", ""),
            "translation_error": row.get("translation_error_m", ""),
            "rotation_angle": row.get("rotation_angle_rad", ""),
            "rotation_expected": row.get("rotation_expected_rad", ""),
            "rotation_error": row.get("rotation_error_rad", ""),
            "surface_rms_displacement": row.get(
                "surface_rms_displacement_m", ""),
            "surface_max_displacement": row.get(
                "surface_max_displacement_m", ""),
            "surface_point_count": row.get("surface_point_count", ""),
        })
    return {
        "schema_version": first.get("truth_schema_version", ""),
        "state": first.get("motion_state", ""),
        "sim_time": first.get("truth_sim_time_s", ""),
        "experiment_elapsed": first.get("experiment_elapsed_s", ""),
        "waiting_elapsed": first.get("waiting_elapsed_s", ""),
        "motion_elapsed": first.get("motion_elapsed_s", ""),
        "motion_mode": first.get("motion_mode", ""),
        "per_model_duration": first.get("per_model_duration_s", ""),
        "linear_speed_mm_s": first.get("linear_speed_mm_s", ""),
        "angular_speed_rad_s": first.get("angular_speed_rad_s", ""),
        "models": models,
    }


def gazebo_truth_artifact_contract(output_directory,
                                   required_motion_experiment_elapsed_s,
                                   stationary_mode,
                                   input_errors=(),
                                   rejected_input_count=None,
                                   checkpoint_tolerance_s=(
                                       MOTION_TRUTH_CHECKPOINT_TOLERANCE_S)):
    """Audit complete six-model truth batches and formal endpoint coverage."""
    input_errors = tuple(input_errors)
    observed_rejected_input_count = (
        int(rejected_input_count) if rejected_input_count is not None
        else len(input_errors))
    relative = "truth/m01_m06_motion_truth.csv"
    path = os.path.join(output_directory, relative)
    errors = ["input_" + str(error) for error in input_errors]
    if observed_rejected_input_count > 0 and not input_errors:
        errors.append("input_rejected_batch_count_{}".format(
            observed_rejected_input_count))
    rows = []
    try:
        with open(path, newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, UnicodeError, csv.Error) as error:
        errors.append("truth_csv_unreadable_{}".format(error))

    batches = defaultdict(list)
    for row_index, row in enumerate(rows, 1):
        value = _optional_float(row, ("truth_batch_index",))
        if (value is None or value < 0.0
                or abs(value - round(value)) > 1.0e-9):
            errors.append("row_{}_truth_batch_index_invalid".format(row_index))
            continue
        batches[int(round(value))].append(row)

    valid_batches = []
    batch_reports = []
    batch_fields = (
        "truth_frame_id", "truth_schema_version", "motion_state",
        "truth_sim_time_s",
        "experiment_elapsed_s", "waiting_elapsed_s", "motion_elapsed_s",
        "per_model_duration_s", "motion_mode", "linear_speed_mm_s",
        "angular_speed_rad_s")
    for batch_index in sorted(batches):
        group = batches[batch_index]
        batch_errors = []
        first = group[0]
        for field in batch_fields:
            values = {str(row.get(field, "")) for row in group}
            if len(values) != 1:
                batch_errors.append("batch_scalar_{}_not_identical".format(
                    field))
        if {str(row.get("truth_frame_id", "")) for row in group} != {"world"}:
            batch_errors.append("truth_frame_id_not_world")
        validation = validate_motion_truth_batch(
            _truth_document_from_csv_batch(group))
        batch_errors.extend(validation["errors"])
        if not batch_errors:
            valid_batches.append((batch_index, group))
        else:
            errors.extend("batch_{}_{}".format(batch_index, error)
                          for error in batch_errors)
        batch_reports.append({
            "truth_batch_index": batch_index,
            "row_count": len(group),
            "status": "COMPLETED" if not batch_errors else "FAILED",
            "errors": batch_errors,
        })
    if not rows:
        errors.append("truth_csv_has_no_rows")
    if not valid_batches:
        errors.append("no_valid_complete_truth_batch")

    try:
        required_end = float(required_motion_experiment_elapsed_s)
    except (TypeError, ValueError, OverflowError):
        required_end = None
    if required_end is None or not math.isfinite(required_end) or required_end < 0.0:
        errors.append("required_motion_experiment_elapsed_s_invalid")
        required_end = None

    per_model = {}
    for name, expected_id in sorted(MOTION_TRUTH_MODEL_IDS.items()):
        samples = []
        for batch_index, group in valid_batches:
            for row in group:
                if str(row.get("model_name", "")) != name:
                    continue
                elapsed = _optional_float(row, ("experiment_elapsed_s",))
                if elapsed is not None:
                    samples.append({
                        "truth_batch_index": batch_index,
                        "experiment_elapsed_s": elapsed,
                        "motion_state": str(row.get("motion_state", "")),
                    })
        maximum = max((sample["experiment_elapsed_s"] for sample in samples),
                      default=None)
        endpoint_satisfied = (
            required_end is not None and maximum is not None
            and maximum + 1.0e-9 >= required_end)
        if not endpoint_satisfied:
            errors.append("{}_required_endpoint_not_covered".format(name))
        checkpoint_reports = {}
        if stationary_mode:
            for checkpoint in MOTION_TRUTH_STATIONARY_CHECKPOINTS_S:
                nearby = [sample for sample in samples
                          if abs(sample["experiment_elapsed_s"] - checkpoint)
                          <= checkpoint_tolerance_s + 1.0e-12]
                checkpoint_reports["{:.0f}".format(checkpoint)] = {
                    "nearby_sample_count": len(nearby),
                    "states": sorted({sample["motion_state"]
                                      for sample in nearby}),
                    "nearest_residual_s": min(
                        (abs(sample["experiment_elapsed_s"] - checkpoint)
                         for sample in samples), default=None),
                }
                if not nearby:
                    errors.append("{}_checkpoint_{:.0f}s_not_covered".format(
                        name, checkpoint))
            transition_states = {
                10.0: ({"WAITING"}, {"RUNNING"}),
                50.0: ({"RUNNING"}, {"HOLDING", "COMPLETED"}),
                60.0: ({"HOLDING"}, {"COMPLETED"}),
            }
            for boundary, (before_states, after_states) in transition_states.items():
                nearby_states = {
                    sample["motion_state"] for sample in samples
                    if abs(sample["experiment_elapsed_s"] - boundary)
                    <= checkpoint_tolerance_s + 1.0e-12}
                if not (nearby_states & before_states):
                    errors.append("{}_boundary_{:.0f}s_pre_state_missing".format(
                        name, boundary))
                if not (nearby_states & after_states):
                    errors.append("{}_boundary_{:.0f}s_post_state_missing".format(
                        name, boundary))
        per_model[name] = {
            "expected_laser_retro_id": expected_id,
            "valid_sample_count": len(samples),
            "maximum_experiment_elapsed_s": maximum,
            "required_endpoint_satisfied": endpoint_satisfied,
            "stationary_checkpoint_coverage": checkpoint_reports,
        }

    return {
        "schema_version": 1,
        "status": "COMPLETED" if not errors else "FAILED",
        "errors": errors,
        "truth_is_evaluation_only": True,
        "expected_model_ids": dict(MOTION_TRUTH_MODEL_IDS),
        "required_motion_experiment_elapsed_s": required_end,
        "stationary_mode": bool(stationary_mode),
        "stationary_checkpoint_tolerance_s": checkpoint_tolerance_s,
        "stationary_required_checkpoints_s": (
            list(MOTION_TRUTH_STATIONARY_CHECKPOINTS_S)
            if stationary_mode else []),
        "csv_row_count": len(rows),
        "observed_batch_count": len(batches),
        "valid_complete_batch_count": len(valid_batches),
        "rejected_input_count": observed_rejected_input_count,
        "batches": batch_reports,
        "per_model": per_model,
        "recall_denominator_policy": (
            "registered mission required_detection_models; truth contract "
            "failure cannot remove a registered model"),
    }


def truth_histories_from_rows(truth_rows):
    """Rebuild timestamped per-model truth histories from the recorded CSV."""
    histories = defaultdict(list)
    for row in truth_rows:
        model = str(row.get("model_name", ""))
        stamp = _optional_float(row, ("truth_sim_time_s",))
        if not model or stamp is None:
            continue
        entry = {
            "sim_time_s": stamp,
            "experiment_elapsed_s": _optional_float(
                row, ("experiment_elapsed_s",)),
            "model_motion_elapsed_s": _optional_float(
                row, ("model_motion_elapsed_s",)),
        }
        vector_fields = {
            "position": ("position_x_m", "position_y_m", "position_z_m"),
            "translation": (
                "translation_x_m", "translation_y_m", "translation_z_m"),
            "aabb_min": ("world_aabb_min_x_m", "world_aabb_min_y_m",
                         "world_aabb_min_z_m"),
            "aabb_max": ("world_aabb_max_x_m", "world_aabb_max_y_m",
                         "world_aabb_max_z_m"),
        }
        for key, fields in vector_fields.items():
            values = tuple(_optional_float(row, (field,)) for field in fields)
            if all(value is not None for value in values):
                entry[key] = values
        orientation = tuple(_optional_float(row, (field,)) for field in (
            "orientation_x", "orientation_y", "orientation_z", "orientation_w"))
        if all(value is not None for value in orientation):
            try:
                entry["orientation"] = normalized_quaternion(orientation)
            except ValueError:
                pass
        history = histories[model]
        if history and abs(float(history[-1]["sim_time_s"]) - stamp) <= 1.0e-12:
            history[-1] = entry
        else:
            history.append(entry)
    for history in histories.values():
        history.sort(key=lambda entry: float(entry["sim_time_s"]))
    return histories


def evaluate_algorithm_truth_at_time(point_world, estimate, message_time_s,
                                     truth_histories, tolerances,
                                     aabb_margin_m, map_from_world_yaw,
                                     max_zoh_residual_s,
                                     prepared_truth_histories=None,
                                     aligned_truth_cache=None):
    """Return evaluation-only, timestamp-aligned truth fields for one sample."""
    result = {
        "model_name": "", "association_status": "UNMATCHED",
        "association_method": "NONE", "association_candidate_models": "",
        "association_score": "", "association_detail": "NO_WORLD_ALIGNMENT",
        "association_truth_alignment_method": "",
        "association_truth_before_time_s": "",
        "association_truth_after_time_s": "",
        "association_truth_bracket_span_s": "",
        "association_truth_zoh_bound_s": max_zoh_residual_s,
        "association_truth_time_offset_s": "",
        "associated_truth_available": False,
        "truth_model_position_world_x_m": "",
        "truth_model_position_world_y_m": "",
        "truth_model_position_world_z_m": "",
        "truth_model_orientation_x": "", "truth_model_orientation_y": "",
        "truth_model_orientation_z": "", "truth_model_orientation_w": "",
        "truth_dx_map_m": "", "truth_dy_map_m": "", "truth_dz_map_m": "",
        "truth_magnitude_m": "", "vector_error_m": "",
        "magnitude_error_m": "", "direction_cosine": "",
        "aligned_experiment_elapsed_s": "",
    }
    if not _finite_vector3(point_world):
        return result
    try:
        cache_key = float(message_time_s)
    except (TypeError, ValueError):
        cache_key = None
    if (aligned_truth_cache is not None and cache_key is not None
            and cache_key in aligned_truth_cache):
        aligned_truth = aligned_truth_cache[cache_key]
    else:
        aligned_truth = {}
        for name, history in truth_histories.items():
            if (prepared_truth_histories is not None
                    and name in prepared_truth_histories):
                aligned_truth[name] = _align_prepared_model_truth_at_time(
                    prepared_truth_histories[name], message_time_s,
                    max_zoh_residual_s)
            else:
                aligned_truth[name] = align_model_truth_at_time(
                    history, message_time_s, max_zoh_residual_s)
        if aligned_truth_cache is not None and cache_key is not None:
            aligned_truth_cache[cache_key] = aligned_truth
    geometries = {}
    for name, aligned in aligned_truth.items():
        if not aligned.get("available", False):
            continue
        position = aligned.get("position")
        if not _finite_vector3(position):
            continue
        geometries[name] = {"position": tuple(position)}
        if (_finite_vector3(aligned.get("aabb_min"))
                and _finite_vector3(aligned.get("aabb_max"))):
            geometries[name]["aabb_min"] = tuple(aligned["aabb_min"])
            geometries[name]["aabb_max"] = tuple(aligned["aabb_max"])
    if not geometries:
        result["association_detail"] = "NO_TIME_ALIGNED_MODEL_TRUTH"
        return result
    association = associate_model_for_evaluation(
        tuple(float(value) for value in point_world), geometries, tolerances,
        aabb_margin_m)
    result.update({
        "model_name": association["model_name"],
        "association_status": association["status"],
        "association_method": association["method"],
        "association_candidate_models": "|".join(association["candidates"]),
        "association_score": association["score"],
        "association_detail": "GEOMETRIC_EVALUATION_ONLY",
    })
    model = association["model_name"]
    if association["status"] != "MATCHED" or not model:
        return result
    aligned = aligned_truth[model]
    position = aligned.get("position")
    orientation = aligned.get("orientation")
    result.update({
        "association_truth_alignment_method": aligned.get("method", ""),
        "association_truth_before_time_s": aligned.get("before_time_s", ""),
        "association_truth_after_time_s": aligned.get("after_time_s", ""),
        "association_truth_bracket_span_s": aligned.get("bracket_span_s", ""),
        "association_truth_zoh_bound_s": aligned.get(
            "zoh_bound_s", max_zoh_residual_s),
        "association_truth_time_offset_s": aligned.get("time_offset_s", ""),
        "aligned_experiment_elapsed_s": aligned.get(
            "experiment_elapsed_s", ""),
    })
    if _finite_vector3(position):
        result.update({
            "truth_model_position_world_x_m": position[0],
            "truth_model_position_world_y_m": position[1],
            "truth_model_position_world_z_m": position[2],
        })
    if (isinstance(orientation, (list, tuple)) and len(orientation) == 4):
        result.update({
            "truth_model_orientation_x": orientation[0],
            "truth_model_orientation_y": orientation[1],
            "truth_model_orientation_z": orientation[2],
            "truth_model_orientation_w": orientation[3],
        })
    translation = aligned.get("translation")
    if not _finite_vector3(translation):
        result["association_detail"] = "TRUTH_VECTOR_UNAVAILABLE"
        return result
    truth_xy = rotate_xy(translation[:2], map_from_world_yaw)
    truth_map = (truth_xy[0], truth_xy[1], float(translation[2]))
    estimate_magnitude, truth_magnitude, vector_error, cosine = vector_metrics(
        estimate, truth_map)
    result.update({
        "associated_truth_available": True,
        "truth_dx_map_m": truth_map[0], "truth_dy_map_m": truth_map[1],
        "truth_dz_map_m": truth_map[2],
        "truth_magnitude_m": truth_magnitude,
        "vector_error_m": vector_error,
        "magnitude_error_m": abs(estimate_magnitude - truth_magnitude),
        "direction_cosine": cosine,
    })
    return result


def realign_algorithm_truth_rows(estimates, truth_rows, tolerances,
                                 aabb_margin_m, map_from_world_yaw,
                                 max_zoh_residual_s):
    """Recompute evaluation/truth-clock columns from complete truth history.

    Detector estimates, probabilities, identifiers and timestamps are copied
    byte-for-byte from their parsed CSV values.  A row is eligible only when it
    contains a detector measurement timestamp, world-space sample position and
    displacement estimate.  This deliberately makes truth a final-analysis
    join and never a detector or map input.
    """
    histories = truth_histories_from_rows(truth_rows)
    prepared_histories = {
        name: _prepare_model_truth_history(history)
        for name, history in histories.items()}
    # A detector tile contains many physical samples with the same source
    # timestamp.  Alignment is a property of that source timestamp and truth
    # history, not of the sample position, so compute it once and reuse it for
    # every sample in the tile.  Spatial association and vector-error formulas
    # remain per sample below.
    aligned_truth_cache = {}
    output = []
    statistics = {
        "raw_row_count": len(estimates),
        "truth_history_model_count": len(histories),
        "prepared_truth_history_count": len(prepared_histories),
        "eligible_row_count": 0,
        "realigned_row_count": 0,
        "skipped_missing_map_world_anchor": 0,
        "skipped_missing_measurement_fields": 0,
        "fixed_map_world_yaw_available": False,
    }
    try:
        yaw = float(map_from_world_yaw)
    except (TypeError, ValueError):
        yaw = None
    if yaw is not None and not math.isfinite(yaw):
        yaw = None
    statistics["fixed_map_world_yaw_available"] = yaw is not None
    for source in estimates:
        row = dict(source)
        message_time = _optional_float(row, ("message_time_s",))
        point_world = tuple(_optional_float(row, (field,)) for field in (
            "position_world_x_m", "position_world_y_m", "position_world_z_m"))
        estimate = tuple(_optional_float(row, (field,)) for field in (
            "estimated_dx_m", "estimated_dy_m", "estimated_dz_m"))
        if yaw is None:
            statistics["skipped_missing_map_world_anchor"] += 1
            output.append(row)
            continue
        if (message_time is None or any(value is None for value in point_world)
                or any(value is None for value in estimate)):
            statistics["skipped_missing_measurement_fields"] += 1
            output.append(row)
            continue
        statistics["eligible_row_count"] += 1
        evaluation = evaluate_algorithm_truth_at_time(
            point_world, estimate, message_time, histories, tolerances,
            aabb_margin_m, yaw, max_zoh_residual_s,
            prepared_truth_histories=prepared_histories,
            aligned_truth_cache=aligned_truth_cache)
        aligned_elapsed = evaluation.pop("aligned_experiment_elapsed_s", "")
        row.update(evaluation)
        if aligned_elapsed not in (None, ""):
            row["experiment_elapsed_s_at_message"] = aligned_elapsed
        statistics["realigned_row_count"] += 1
        output.append(row)
    statistics["unique_aligned_measurement_time_count"] = len(
        aligned_truth_cache)
    statistics["prepared_alignment_evaluation_count"] = (
        len(aligned_truth_cache) * len(prepared_histories))
    return output, statistics


def registered_primary_model_recall(model_rows, primary_models):
    """Use the pre-registered target set as the immutable recall denominator."""
    registered = tuple(dict.fromkeys(str(model) for model in primary_models))
    by_name = {str(row.get("model_name", "")): row for row in model_rows}
    detected = []
    false_negative = []
    for model in registered:
        row = by_name.get(model)
        if (row is not None and _csv_bool(row.get("truth_active", False))
                and _csv_bool(row.get("detected", False))):
            detected.append(model)
        else:
            # Missing/inactive truth does not erase a registered target. The
            # independent truth contract will also fail such an incomplete run.
            false_negative.append(model)
    denominator = len(registered)
    return {
        "denominator_models": list(registered),
        "denominator_model_count": denominator,
        "detected_models": detected,
        "detected_model_count": len(detected),
        "false_negative_models": false_negative,
        "false_negative_model_count": len(false_negative),
        "object_recall": (len(detected) / float(denominator)
                          if denominator else None),
    }


def finalization_high_water_checks(required, recorded,
                                   capture_freeze_required,
                                   capture_freeze_committed,
                                   pending_checkpoint_count):
    """Pure contract check used before closing any recorder stream."""
    def reached(name, tolerance=1.0e-9):
        target = required.get(name)
        actual = recorded.get(name)
        return (target is None
                or (actual is not None and actual + tolerance >= target))

    return {
        "capture_freeze": (not capture_freeze_required
                           or bool(capture_freeze_committed)),
        "motion_truth": reached("motion_experiment_elapsed_s"),
        "detector": reached("detector_accepted_frame_index", 0.0),
        "evolution": reached("evolution_accepted_frame_index", 0.0),
        "avoidance": reached("avoidance_accepted_frame_index", 0.0),
        "navigation": reached("navigation_stamp_s"),
        "robot_truth": reached("robot_truth_stamp_s"),
        "checkpoints": int(pending_checkpoint_count) == 0,
    }


def validate_required_artifacts(output_directory, required_paths):
    """Validate required products by type, not merely by path existence."""
    root = os.path.realpath(os.path.abspath(output_directory))
    errors = []
    for relative in sorted(set(str(value) for value in required_paths)):
        if (not relative or os.path.isabs(relative)
                or relative == ".." or relative.startswith("../")):
            errors.append("invalid required path: " + relative)
            continue
        path = os.path.realpath(os.path.join(root, relative))
        try:
            if os.path.commonpath((root, path)) != root:
                errors.append("required path escapes output: " + relative)
                continue
        except ValueError:
            errors.append("required path escapes output: " + relative)
            continue
        if not os.path.isfile(path):
            errors.append("missing: " + relative)
            continue
        try:
            if os.path.getsize(path) <= 0:
                raise ValueError("empty file")
            extension = os.path.splitext(relative)[1].lower()
            if extension == ".json":
                with open(path, encoding="utf-8") as stream:
                    document = json.load(stream)
                if not isinstance(document, (dict, list)):
                    raise ValueError("JSON root is not an object or array")
            elif extension == ".csv":
                with open(path, newline="", encoding="utf-8") as stream:
                    reader = csv.reader(stream)
                    header = next(reader, None)
                    data = next(reader, None)
                if not header or any(not str(field).strip() for field in header):
                    raise ValueError("missing/invalid CSV header")
                if data is None:
                    raise ValueError("CSV has no research data row")
            elif extension == ".png":
                with open(path, "rb") as stream:
                    if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                        raise ValueError("invalid PNG signature")
            elif extension == ".npz":
                with np.load(path, allow_pickle=False) as stored:
                    keys = set(stored.files)
                    if "snapshots/maps/" in relative.replace("\\", "/"):
                        if "data" not in keys or stored["data"].ndim != 2:
                            raise ValueError("map NPZ lacks a 2-D data array")
                    elif "snapshots/clouds/" in relative.replace("\\", "/"):
                        if ("points" not in keys or stored["points"].ndim != 2
                                or stored["points"].shape[1] < 3):
                            raise ValueError(
                                "cloud NPZ lacks an N-by-at-least-3 point array")
        except (OSError, UnicodeError, ValueError, csv.Error, EOFError) as error:
            errors.append("invalid: {} ({})".format(relative, error))
    return errors


TRACK_ASSOCIATION_MINIMUM_MATCHED_FRAMES = 3
TRACK_ASSOCIATION_MINIMUM_MATCHED_FRAME_PURITY = 0.80


def _track_frame_association(estimates):
    """Aggregate spatial samples into independent accepted-frame evidence.

    One accepted detector frame contributes at most one temporal vote to a
    stable GDU.  A frame is a clean match for model ``m`` when it contains at
    least one MATCHED sample for ``m``, no MATCHED sample for another model,
    and no AMBIGUOUS sample.  UNMATCHED spatial samples in that same frame do
    not contradict the unique positive association.  Every identifiable frame
    containing estimates remains in the purity denominator.

    Rows without an integer accepted-frame index are reported explicitly and
    make the track ineligible for MATCHED, because their independence cannot
    be established without inventing temporal evidence.
    """
    frames = defaultdict(list)
    missing_frame_count = 0
    sample_model_votes = defaultdict(int)
    ambiguous_sample_count = 0
    unmatched_sample_count = 0
    for row in estimates:
        status = str(row.get("association_status", "UNMATCHED")).strip().upper()
        model = str(row.get("model_name", "")).strip()
        if status == "MATCHED" and model:
            sample_model_votes[model] += 1
        elif status == "AMBIGUOUS":
            ambiguous_sample_count += 1
        else:
            unmatched_sample_count += 1

        frame_value = _optional_float(
            row, ("accepted_frame_index", "source_accepted_frame_index"))
        if (frame_value is None or frame_value < 0.0
                or abs(frame_value - round(frame_value)) > 1.0e-9):
            missing_frame_count += 1
            continue
        frames[int(round(frame_value))].append(row)

    model_frame_support = defaultdict(int)
    candidate_models = set()
    ambiguous_frame_count = 0
    unmatched_frame_count = 0
    multi_model_frame_count = 0
    for frame_rows in frames.values():
        matched_models = {
            str(row.get("model_name", "")).strip()
            for row in frame_rows
            if str(row.get("association_status", "")).strip().upper()
            == "MATCHED" and str(row.get("model_name", "")).strip()
        }
        candidate_models.update(matched_models)
        has_ambiguous_sample = any(
            str(row.get("association_status", "")).strip().upper()
            == "AMBIGUOUS" for row in frame_rows)
        has_invalid_matched_sample = any(
            str(row.get("association_status", "")).strip().upper()
            == "MATCHED" and not str(row.get("model_name", "")).strip()
            for row in frame_rows)
        if len(matched_models) > 1:
            multi_model_frame_count += 1
            ambiguous_frame_count += 1
        elif len(matched_models) == 1 and not (
                has_ambiguous_sample or has_invalid_matched_sample):
            model_frame_support[next(iter(matched_models))] += 1
        elif matched_models or has_ambiguous_sample or has_invalid_matched_sample:
            ambiguous_frame_count += 1
        else:
            unmatched_frame_count += 1

    candidate_models.update(sample_model_votes)
    candidate_models = sorted(candidate_models)
    ordered_support = sorted(
        ((model, model_frame_support.get(model, 0))
         for model in candidate_models),
        key=lambda item: (-item[1], item[0]))
    best_model, best_support = ordered_support[0] if ordered_support else ("", 0)
    independent_frame_count = len(frames)
    matched_frame_purity = (
        best_support / float(independent_frame_count)
        if independent_frame_count else 0.0)
    frame_contract_satisfied = missing_frame_count == 0

    if not candidate_models:
        association_status = (
            "AMBIGUOUS" if ambiguous_frame_count else "UNMATCHED")
        accepted_model = ""
    elif (len(candidate_models) == 1 and frame_contract_satisfied
          and best_support >= TRACK_ASSOCIATION_MINIMUM_MATCHED_FRAMES
          and matched_frame_purity
          >= TRACK_ASSOCIATION_MINIMUM_MATCHED_FRAME_PURITY):
        association_status = "MATCHED"
        accepted_model = best_model
    else:
        # Any multi-model hit, unknown frame identity, insufficient temporal
        # support, or insufficient purity remains conservative ambiguity.
        association_status = "AMBIGUOUS"
        accepted_model = ""

    return {
        "association_status": association_status,
        "model_name": accepted_model,
        "candidate_models": candidate_models,
        "candidate_model_matched_frame_counts": "|".join(
            "{}:{}".format(model, support)
            for model, support in sorted(ordered_support)),
        "best_candidate_model": best_model,
        "independent_estimate_frame_count": independent_frame_count,
        "matched_frame_count": best_support,
        "total_clean_matched_frame_count": sum(model_frame_support.values()),
        "ambiguous_frame_count": ambiguous_frame_count,
        "unmatched_frame_count": unmatched_frame_count,
        "multi_model_frame_count": multi_model_frame_count,
        "estimate_rows_without_accepted_frame_count": missing_frame_count,
        "independent_frame_contract_satisfied": frame_contract_satisfied,
        "matched_frame_purity": matched_frame_purity,
        "minimum_matched_frame_support": (
            TRACK_ASSOCIATION_MINIMUM_MATCHED_FRAMES),
        "minimum_matched_frame_purity": (
            TRACK_ASSOCIATION_MINIMUM_MATCHED_FRAME_PURITY),
        "sample_model_votes": sample_model_votes,
        "ambiguous_sample_count": ambiguous_sample_count,
        "unmatched_sample_count": unmatched_sample_count,
    }


def build_unique_matched_track_relations(track_inventory_rows):
    """Return the sole permitted GDU-to-truth-model comparison relation.

    ``detection_track_inventory.csv`` is the authoritative output of
    :func:`build_track_inventory`.  Derived evolution/truth comparisons may
    consume a GDU only when the inventory contains exactly one row for it and
    that row passed the independent-frame rule as MATCHED with one non-empty
    model.  Confirmation remains an independent lifecycle property copied from
    that same row; neither raw spatial-sample majorities nor mission-region
    bookkeeping can create or overwrite a relation.
    """
    grouped = defaultdict(list)
    for row in track_inventory_rows:
        gdu_id = str(row.get("gdu_id", "")).strip()
        if gdu_id:
            grouped[gdu_id].append(row)
    relations = {}
    for gdu_id, rows in grouped.items():
        if len(rows) != 1:
            continue
        row = rows[0]
        model = str(row.get("model_name", "")).strip()
        if (str(row.get("association_status", "")).strip().upper()
                != "MATCHED" or not model):
            continue
        relations[gdu_id] = {
            "model_name": model,
            "confirmed": _csv_bool(row.get("confirmed", False)),
            "confirmation_time_available": _csv_bool(
                row.get("confirmation_time_available", False)),
            "evaluation_category": str(
                row.get("evaluation_category", "")),
        }
    return relations


def build_track_inventory(estimates, evidence_rows, evolution_rows, truth_rows,
                          model_names, displacement_evaluation_floor_m=0.005,
                          primary_target_models=None):
    """Build a complete, one-row-per-stable-GDU evaluation inventory.

    The function never edits detector output.  It reports every GDU seen in
    any raw detector/evolution table, including unconfirmed, unmatched and
    ambiguous tracks.  Association is pre-registered at independent accepted-
    frame level, never by counting multiple spatial samples from the same GDU
    and frame as independent evidence.  Every uniquely associated confirmed
    GDU with an observed NEWLY_DETECTED transition is a track-level true
    positive, even when several GDUs lie on the same target; only structural-
    target recall deduplicates by target.
    """
    primary_target_models = set(
        model_names if primary_target_models is None else primary_target_models)
    tracks = defaultdict(lambda: {"estimate": [], "evidence": [], "evolution": []})
    for source_name, rows in (("estimate", estimates),
                              ("evidence", evidence_rows),
                              ("evolution", evolution_rows)):
        for row in rows:
            gdu_id = str(row.get("gdu_id", ""))
            if gdu_id:
                tracks[gdu_id][source_name].append(row)

    truth_by_model = defaultdict(list)
    for row in truth_rows:
        model = str(row.get("model_name", ""))
        if model:
            truth_by_model[model].append(row)
    truth_activation = {}
    for model in model_names:
        onset_candidates = []
        fallback_candidates = []
        for row in truth_by_model[model]:
            experiment_elapsed = _optional_float(
                row, ("experiment_elapsed_s",))
            model_elapsed = _optional_float(
                row, ("model_motion_elapsed_s",))
            if (experiment_elapsed is not None and model_elapsed is not None
                    and model_elapsed > 0.0):
                onset_candidates.append(experiment_elapsed - model_elapsed)
            expected = [_optional_float(row, (key,)) or 0.0 for key in (
                "expected_translation_x_m", "expected_translation_y_m",
                "expected_translation_z_m")]
            actual = _optional_float(
                row, ("surface_max_displacement_m",
                      "translation_magnitude_m")) or 0.0
            rotation_expected = _optional_float(
                row, ("rotation_expected_rad",)) or 0.0
            if (experiment_elapsed is not None
                    and (np.linalg.norm(expected) > 1.0e-12
                         or actual > 1.0e-12
                         or abs(rotation_expected) > 1.0e-12)):
                fallback_candidates.append(experiment_elapsed)
        if onset_candidates:
            onset = min(onset_candidates)
            onset_source = "EXPERIMENT_ELAPSED_MINUS_MODEL_MOTION_ELAPSED"
        elif fallback_candidates:
            onset = min(fallback_candidates)
            onset_source = "FIRST_NONZERO_TRUTH_SAMPLE_APPROXIMATION"
        else:
            onset = None
            onset_source = "NO_MOTION_TRUTH"
        truth_activation[model] = {
            "active": onset is not None,
            "experiment_elapsed_s": onset,
            "onset_source": onset_source,
            "maximum_displacement_m": max(
                (_optional_float(
                    row, ("surface_max_displacement_m",
                          "translation_magnitude_m")) or 0.0)
                for row in truth_by_model[model])
            if truth_by_model[model] else None,
            "displacement_evaluation_floor_m": displacement_evaluation_floor_m,
        }

    inventory = []
    for gdu_id in sorted(tracks):
        grouped = tracks[gdu_id]
        all_rows = (grouped["estimate"] + grouped["evidence"]
                    + grouped["evolution"])
        receive_times = [_optional_float(
            row, ("receive_time_s", "ros_time_s")) for row in all_rows]
        receive_times = [value for value in receive_times if value is not None]
        experiment_times = [_optional_float(
            row, ("experiment_elapsed_s_at_message",
                  "experiment_elapsed_s_at_receive")) for row in all_rows]
        experiment_times = [value for value in experiment_times if value is not None]
        accepted_indices = [_optional_float(
            row, ("accepted_frame_index", "source_accepted_frame_index"))
                            for row in all_rows]
        accepted_indices = [int(value) for value in accepted_indices
                            if value is not None]
        retained_rows = [row for row in grouped["evolution"]
                         if _csv_bool(row.get("retained_for_avoidance", ""))
                         and int(float(row.get("lifecycle_state", 0) or 0))
                         in (1, 2, 3)]
        retained_rows.sort(key=lambda row: (
            _optional_float(row, ("experiment_elapsed_s_at_message",
                                  "experiment_elapsed_s_at_receive"))
            if _optional_float(row, ("experiment_elapsed_s_at_message",
                                     "experiment_elapsed_s_at_receive"))
            is not None else float("inf"),
            _optional_float(row, ("receive_time_s", "ros_time_s"))
            if _optional_float(row, ("receive_time_s", "ros_time_s"))
            is not None else float("inf")))
        # NEWLY_DETECTED is the only explicit final-confirmation transition in
        # the V3 lifecycle.  PERSISTENT/CLEAR_CANDIDATE prove that a track was
        # confirmed earlier, but cannot reconstruct its exact confirmation
        # time if the NEWLY_DETECTED row was not recorded.
        confirmation_rows = [
            row for row in retained_rows
            if int(float(row.get("lifecycle_state", 0) or 0)) == 1]
        confirmation = confirmation_rows[0] if confirmation_rows else None
        first_retained = retained_rows[0] if retained_rows else None
        frame_association = _track_frame_association(grouped["estimate"])
        association_status = frame_association["association_status"]
        model_name = frame_association["model_name"]
        matched_models = frame_association["candidate_models"]
        votes = frame_association["sample_model_votes"]
        first_observation_exp = min(experiment_times) if experiment_times else None
        confirmed_exp = (_optional_float(
            confirmation, ("experiment_elapsed_s_at_message",
                           "experiment_elapsed_s_at_receive"))
                         if confirmation is not None else None)
        inventory.append({
            "gdu_id": gdu_id,
            "association_status": association_status,
            "model_name": model_name,
            "included_in_primary_metrics": model_name in primary_target_models,
            "candidate_models": "|".join(matched_models),
            "candidate_model_matched_frame_counts": frame_association[
                "candidate_model_matched_frame_counts"],
            "best_candidate_model": frame_association["best_candidate_model"],
            "estimate_sample_count": len(grouped["estimate"]),
            "evidence_sample_count": len(grouped["evidence"]),
            "evolution_sample_count": len(grouped["evolution"]),
            "matched_sample_count": sum(votes.values()),
            "ambiguous_sample_count": frame_association[
                "ambiguous_sample_count"],
            "unmatched_sample_count": frame_association[
                "unmatched_sample_count"],
            "independent_estimate_frame_count": frame_association[
                "independent_estimate_frame_count"],
            "matched_frame_count": frame_association["matched_frame_count"],
            "total_clean_matched_frame_count": frame_association[
                "total_clean_matched_frame_count"],
            "ambiguous_frame_count": frame_association[
                "ambiguous_frame_count"],
            "unmatched_frame_count": frame_association[
                "unmatched_frame_count"],
            "multi_model_frame_count": frame_association[
                "multi_model_frame_count"],
            "estimate_rows_without_accepted_frame_count": frame_association[
                "estimate_rows_without_accepted_frame_count"],
            "independent_frame_contract_satisfied": frame_association[
                "independent_frame_contract_satisfied"],
            "matched_frame_purity": frame_association[
                "matched_frame_purity"],
            "minimum_matched_frame_support": frame_association[
                "minimum_matched_frame_support"],
            "minimum_matched_frame_purity": frame_association[
                "minimum_matched_frame_purity"],
            "first_accepted_frame_index": min(accepted_indices)
            if accepted_indices else "",
            "last_accepted_frame_index": max(accepted_indices)
            if accepted_indices else "",
            "first_receive_time_s": min(receive_times) if receive_times else "",
            "last_receive_time_s": max(receive_times) if receive_times else "",
            "first_observation_experiment_elapsed_s": (
                first_observation_exp
                if first_observation_exp is not None else ""),
            "candidate_time_available": False,
            "candidate_experiment_elapsed_s": "",
            "confirmed": bool(retained_rows),
            "confirmation_time_available": confirmation is not None,
            "confirmation_time_source": (
                "LIFECYCLE_NEWLY_DETECTED" if confirmation is not None
                else ("RETAINED_WITHOUT_NEWLY_DETECTED_EVENT"
                      if retained_rows else "NOT_CONFIRMED")),
            "confirmation_experiment_elapsed_s": confirmed_exp
            if confirmed_exp is not None else "",
            "confirmation_receive_time_s": (
                _optional_float(confirmation, ("receive_time_s", "ros_time_s"))
                if confirmation is not None else ""),
            "first_retained_observation_experiment_elapsed_s": (
                _optional_float(
                    first_retained,
                    ("experiment_elapsed_s_at_message",
                     "experiment_elapsed_s_at_receive"))
                if first_retained is not None else ""),
            "maximum_task_scale_probability": max(
                (_optional_float(row, ("task_scale_probability",)) or 0.0)
                for row in all_rows) if all_rows else "",
            "evaluation_category": "PENDING",
        })

    confirmed_order = sorted(
        [row for row in inventory if row["confirmed"]],
        key=lambda row: (
            float(row["confirmation_experiment_elapsed_s"])
            if row["confirmation_experiment_elapsed_s"] != "" else float("inf"),
            float(row["confirmation_receive_time_s"])
            if row["confirmation_receive_time_s"] not in ("", None)
            else float("inf"), row["gdu_id"]))
    for row in confirmed_order:
        if not row["confirmation_time_available"]:
            # A retained PERSISTENT/CLEAR_CANDIDATE row proves that a
            # confirmation happened, but without the NEWLY_DETECTED transition
            # neither its temporal ordering nor its confirmation displacement
            # is measurable.  It remains in the precision/FAR denominator and
            # can never be promoted to a true positive.
            row["evaluation_category"] = (
                "FALSE_POSITIVE_CONFIRMATION_TIME_UNAVAILABLE")
        elif row["association_status"] == "UNMATCHED":
            row["evaluation_category"] = "FALSE_POSITIVE_UNMATCHED"
        elif row["association_status"] == "AMBIGUOUS":
            # The paper's pooled precision denominator contains every unique
            # confirmed track.  A track that cannot be associated uniquely is
            # therefore a false confirmed track, not a silently omitted row.
            row["evaluation_category"] = "FALSE_POSITIVE_AMBIGUOUS"
        elif row["model_name"] not in primary_target_models:
            # m04 is intentionally retained as a distractor/optional target.
            # A confirmation on an excluded model remains visible and counts
            # in the pooled precision denominator, but it cannot improve the
            # registered structural-target recall.
            row["evaluation_category"] = "FALSE_POSITIVE_EXCLUDED_MODEL"
        elif not truth_activation[row["model_name"]]["active"]:
            row["evaluation_category"] = "FALSE_POSITIVE_STATIC_MODEL"
        elif (row["confirmation_experiment_elapsed_s"] != ""
              and truth_activation[row["model_name"]][
                  "experiment_elapsed_s"] is not None
              and float(row["confirmation_experiment_elapsed_s"])
              < truth_activation[row["model_name"]][
                  "experiment_elapsed_s"]):
            row["evaluation_category"] = "FALSE_POSITIVE_PRE_ACTIVATION"
        else:
            row["evaluation_category"] = "TRUE_POSITIVE"
    for row in inventory:
        if not row["confirmed"]:
            row["evaluation_category"] = "UNCONFIRMED_OBSERVATION_TRACK"
    return inventory, truth_activation


CSV_SCHEMAS = {
    "sensors/imu.csv": [
        "ros_time_s", "message_time_s", "wall_elapsed_s", "mission_state",
        "frame_id", "wx_rad_s", "wy_rad_s", "wz_rad_s",
        "ax_mps2", "ay_mps2", "az_mps2"],
    "navigation/global_costmaps.csv": [
        "ros_time_s", "message_time_s", "wall_elapsed_s", "mission_state",
        "frame_id", "map_id", "relative_path"],
    "navigation/global_plans.csv": [
        "ros_time_s", "message_time_s", "wall_elapsed_s", "mission_state",
        "plan_id", "frame_id", "point_index", "point_count",
        "x_m", "y_m", "yaw_rad"],
    "navigation/action_events.csv": [
        "ros_time_s", "message_time_s", "wall_elapsed_s", "mission_state",
        "event", "goal_id", "frame_id", "x_m", "y_m", "status", "text"],
    "trajectories/slam_odometry.csv": [
        "ros_time_s", "message_time_s", "receive_time_s", "wall_elapsed_s",
        "mission_state", "frame_id",
        "child_frame_id", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw",
        "yaw_rad", "vx_mps", "vy_mps", "vz_mps", "wz_rad_s"],
    "trajectories/navigation_base_odometry.csv": [
        "ros_time_s", "message_time_s", "receive_time_s", "wall_elapsed_s",
        "mission_state", "frame_id",
        "child_frame_id", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw",
        "yaw_rad", "vx_mps", "vy_mps", "vz_mps", "wz_rad_s"],
    "trajectories/gazebo_robot_truth.csv": [
        "ros_time_s", "message_time_s", "receive_time_s", "wall_elapsed_s",
        "mission_state", "frame_id", "child_frame_id",
        "model_x_m", "model_y_m", "model_z_m", "model_yaw_rad",
        "model_qx", "model_qy", "model_qz", "model_qw",
        "body_x_m", "body_y_m", "body_z_m", "body_yaw_rad",
        "body_qx", "body_qy", "body_qz", "body_qw",
        "truth_map_frame_id", "truth_map_x_m", "truth_map_y_m",
        "truth_map_yaw_rad",
        "vx_mps", "vy_mps", "vz_mps", "wz_rad_s"],
    "trajectories/cmd_vel.csv": [
        "ros_time_s", "receive_time_s", "wall_elapsed_s", "mission_state",
        "vx_mps", "vy_mps", "vz_mps", "wx_rad_s", "wy_rad_s", "wz_rad_s"],
    "truth/m01_m06_motion_truth.csv": [
        "ros_time_s", "truth_sim_time_s", "receive_time_s", "wall_elapsed_s",
        "mission_state", "truth_frame_id", "truth_batch_index",
        "truth_schema_version",
        "motion_state",
        "experiment_elapsed_s", "waiting_elapsed_s", "motion_elapsed_s",
        "model_motion_elapsed_s", "motion_mode", "per_model_duration_s",
        "linear_speed_mm_s", "angular_speed_rad_s", "model_name",
        "laser_retro_id",
        "position_x_m", "position_y_m", "position_z_m",
        "orientation_x", "orientation_y", "orientation_z", "orientation_w",
        "world_aabb_min_x_m", "world_aabb_min_y_m", "world_aabb_min_z_m",
        "world_aabb_max_x_m", "world_aabb_max_y_m", "world_aabb_max_z_m",
        "translation_x_m", "translation_y_m", "translation_z_m",
        "expected_translation_x_m", "expected_translation_y_m",
        "expected_translation_z_m",
        "translation_magnitude_m", "translation_expected_m",
        "translation_error_m", "rotation_angle_rad", "rotation_expected_rad",
        "rotation_error_rad", "surface_rms_displacement_m",
        "surface_max_displacement_m", "surface_point_count"],
    "deformation/algorithm_displacement.csv": [
        "ros_time_s", "message_time_s", "publish_time_s", "receive_time_s",
        "truth_sim_time_s_at_receive", "experiment_elapsed_s_at_receive",
        "experiment_elapsed_s_at_message",
        "wall_elapsed_s", "mission_state", "frame_id",
        "accepted_frame_index",
        "reference_epoch", "snapshot_id", "tile_key", "gdu_id", "sample_index",
        "position_map_x_m", "position_map_y_m", "position_map_z_m",
        "position_world_x_m", "position_world_y_m", "position_world_z_m",
        "estimated_dx_m", "estimated_dy_m", "estimated_dz_m",
        "estimated_magnitude_m", "estimated_vx_mps", "estimated_vy_mps",
        "estimated_vz_mps", "displacement_covariance_row_major",
        "velocity_covariance_row_major", "observable_projector_row_major",
        "active_probability_lower",
        "active_probability_mean", "active_probability_upper",
        "active_probability_mean_identified", "observable_rank",
        "task_scale_probability", "detection_allowed", "detection_admission",
        "validity", "minimum_detectable_effect_m", "other_mass",
        "interpolation_error_upper_m", "model_name",
        "association_status", "association_method",
        "association_candidate_models", "association_score",
        "association_detail", "association_truth_alignment_method",
        "association_truth_before_time_s", "association_truth_after_time_s",
        "association_truth_bracket_span_s", "association_truth_zoh_bound_s",
        "association_truth_time_offset_s",
        "associated_truth_available",
        "truth_model_position_world_x_m", "truth_model_position_world_y_m",
        "truth_model_position_world_z_m", "truth_model_orientation_x",
        "truth_model_orientation_y", "truth_model_orientation_z",
        "truth_model_orientation_w",
        "truth_dx_map_m", "truth_dy_map_m", "truth_dz_map_m",
        "truth_magnitude_m", "vector_error_m", "magnitude_error_m",
        "direction_cosine"],
    "deformation/detection_evidence.csv": [
        "ros_time_s", "message_time_s", "publish_time_s", "receive_time_s",
        "truth_sim_time_s_at_receive", "experiment_elapsed_s_at_receive",
        "experiment_elapsed_s_at_message",
        "wall_elapsed_s", "mission_state", "frame_id",
        "accepted_frame_index",
        "reference_epoch", "gdu_id", "position_x_m", "position_y_m", "position_z_m",
        "p_u", "p_a", "p_c", "p_other", "task_scale_probability_lower",
        "task_scale_probability_mean", "task_scale_probability_upper",
        "task_scale_probability_mean_identified", "task_displacement_probability",
        "current_velocity_exceedance_probability",
        "continuous_valid_observation_duration_s", "velocity_duration_requirement_met",
        "observable_rank", "minimum_detectable_effect_m",
        "any_task_direction_detectable", "all_task_directions_detectable",
        "detection_allowed", "clearing_allowed", "effective_spatial_support",
        "effective_temporal_support", "detection_admission", "validity"],
    "deformation/reference_support.csv": [
        "ros_time_s", "message_time_s", "publish_time_s", "receive_time_s",
        "truth_sim_time_s_at_receive", "experiment_elapsed_s_at_receive",
        "experiment_elapsed_s_at_message",
        "wall_elapsed_s", "mission_state", "frame_id",
        "accepted_frame_index",
        "reference_epoch", "gdu_id", "surfel_id", "position_x_m", "position_y_m",
        "position_z_m", "reference_covariance_row_major",
        "observable_projector_row_major", "geometry_probability",
        "support_radius_m", "exposure_count", "point_count", "validity"],
    "deformation/interface_motion.csv": [
        "ros_time_s", "message_time_s", "publish_time_s", "receive_time_s",
        "truth_sim_time_s_at_receive", "experiment_elapsed_s_at_receive",
        "experiment_elapsed_s_at_message",
        "wall_elapsed_s", "mission_state", "frame_id",
        "accepted_frame_index",
        "reference_epoch", "interface_id", "left_gdu_id", "right_gdu_id",
        "center_x_m", "center_y_m", "center_z_m", "normal_x", "normal_y", "normal_z",
        "scale_m", "joint_probability", "jump_dx_m", "jump_dy_m", "jump_dz_m",
        "jump_covariance_row_major", "other_mass",
        "connectivity_coherence_confounding", "validity"],
    "deformation/detector_status.csv": [
        "ros_time_s", "message_time_s", "publish_time_s", "receive_time_s",
        "truth_sim_time_s_at_receive", "experiment_elapsed_s_at_receive",
        "experiment_elapsed_s_at_message",
        "wall_elapsed_s", "mission_state", "frame_id", "phase",
        "reference_epoch", "accepted_frame_index", "reference_frozen",
        "gdu_count", "interface_count", "validity", "active_gates",
        "deadline_failure_count", "resource_failure_count",
        "last_attempt_frame_index", "last_attempt_published",
        "last_attempt_reason_codes", "last_attempt_processing_time_ms",
        "last_attempt_input_point_count", "last_attempt_time_s"],
    "runtime/per_cycle.csv": [
        "ros_time_s", "message_time_s", "publish_time_s", "receive_time_s",
        "truth_sim_time_s_at_receive", "experiment_elapsed_s_at_receive",
        "experiment_elapsed_s_at_message",
        "wall_elapsed_s", "mission_state", "frame_id",
        "last_attempt_frame_index",
        "last_attempt_published", "last_attempt_reason_codes",
        "last_attempt_time_s", "processing_time_ms", "input_point_count"],
    "evolution/gdu_evolution.csv": [
        "ros_time_s", "message_time_s", "receive_time_s",
        "truth_sim_time_s_at_receive", "experiment_elapsed_s_at_receive",
        "experiment_elapsed_s_at_message",
        "wall_elapsed_s", "mission_state", "frame_id",
        "source_accepted_frame_index",
        "source_snapshot_id",
        "reference_epoch", "gdu_id", "active_now", "retained_for_avoidance",
        "retained_birth_streak", "retained_birth_confirmations",
        "lifecycle_state", "magnitude_trend", "task_scale_probability",
        "displacement_rms_m", "previous_displacement_rms_m",
        "displacement_change_m", "observation_interval_s", "displacement_rate_mps",
        "mean_dx_m", "mean_dy_m", "mean_dz_m", "mean_vx_mps", "mean_vy_mps",
        "mean_vz_mps", "support_radius_m", "forecast_count", "validity"],
    "maps/avoidance_frames.csv": [
        "ros_time_s", "message_time_s", "receive_time_s",
        "truth_sim_time_s_at_receive", "experiment_elapsed_s_at_receive",
        "experiment_elapsed_s_at_message",
        "wall_elapsed_s", "mission_state", "frame_id",
        "source_accepted_frame_index",
        "source_snapshot_id",
        "map_revision", "policy", "hard_avoidance_radius_m", "warning_radius_m",
        "elevated_source_min_z_m", "elevated_hard_projection_radius_m",
        "elevated_warning_projection_radius_m",
        "active_source_count", "active_source_ids",
        "ordinary_projection_source_ids", "elevated_projection_source_ids",
        "forecast_horizons_s",
        "current_occupied_cell_count", "current_unknown_cell_count",
        "all_current_sources_within_map", "all_forecast_sources_within_map"],
    "sensors/dual_lidar_merge_diagnostics.csv": [
        "ros_time_s", "diagnostic_time_s", "receive_time_s",
        "receive_monotonic_s", "receive_wall_time_s", "wall_elapsed_s",
        "mission_state", "schema_version", "secondary_enabled",
        "zero_no_return_compaction_enabled",
        "point_time_mode", "configured_sensor_update_rate_hz",
        "configured_min_range_m", "configured_max_range_m",
        "configured_ray_samples_per_sensor",
        "primary_scheduler_catch_up_suppressed_count",
        "primary_downsample", "secondary_downsample",
        "configured_secondary_to_primary_point_ratio",
        "overlap_voxel_size_m", "overlap_evaluation_pair_stride",
        "primary_received", "secondary_received",
        "pairs_accepted", "pairs_rejected", "primary_unpaired_dropped",
        "secondary_unpaired_dropped", "primary_queue_dropped",
        "secondary_queue_dropped", "last_sync_delta_ns",
        "maximum_sync_delta_ns", "mean_sync_delta_ns",
        "first_accepted_measurement_time_ns",
        "last_accepted_measurement_time_ns",
        "accepted_measurement_interval_count",
        "nonpositive_accepted_measurement_interval_count",
        "minimum_accepted_measurement_interval_ns",
        "maximum_accepted_measurement_interval_ns",
        "mean_accepted_measurement_interval_ns", "realized_measurement_hz",
        "last_merged_span_ns", "maximum_merged_span_ns",
        "last_primary_point_count", "maximum_primary_point_count",
        "last_secondary_point_count", "maximum_secondary_point_count",
        "last_merged_point_count", "maximum_merged_point_count",
        "last_secondary_to_primary_point_ratio",
        "last_merged_to_primary_point_ratio",
        "last_total_raw_input_point_count",
        "maximum_total_raw_input_point_count",
        "last_compacted_output_point_count",
        "maximum_compacted_output_point_count",
        "last_removed_no_return_sentinel_count",
        "maximum_removed_no_return_sentinel_count",
        "last_compacted_output_to_total_raw_input_ratio",
        "overlap_evaluations_attempted", "overlap_evaluations_failed",
        "overlap_evaluation_sample_count", "last_overlap_pair_index",
        "last_primary_valid_return_count",
        "last_secondary_valid_return_count",
        "last_primary_occupied_voxel_count",
        "last_secondary_occupied_voxel_count",
        "last_shared_occupied_voxel_count",
        "last_union_occupied_voxel_count",
        "last_primary_overlap_fraction", "last_secondary_overlap_fraction",
        "last_voxel_jaccard", "mean_primary_overlap_fraction",
        "mean_secondary_overlap_fraction", "mean_voxel_jaccard",
        "minimum_primary_overlap_fraction",
        "minimum_secondary_overlap_fraction", "minimum_voxel_jaccard",
        "primary_pair_ratio", "secondary_pair_ratio"],
    "sensors/gazebo_performance_metrics.csv": [
        "ros_time_s", "message_time_s", "receive_time_s",
        "receive_monotonic_s", "receive_wall_time_s", "wall_elapsed_s",
        "mission_state", "frame_id", "message_sequence",
        "real_time_factor", "sensor_name", "sim_update_rate_hz",
        "real_update_rate_hz", "fps"],
    "events/mission_states.csv": [
        "ros_time_s", "message_time_s", "receive_time_s", "wall_elapsed_s",
        "state", "detail"],
    "events/timeline_checkpoints.csv": [
        "ros_time_s", "truth_sim_time_s", "receive_time_s", "wall_elapsed_s",
        "mission_state", "motion_state", "target_experiment_elapsed_s",
        "actual_experiment_elapsed_s", "checkpoint_lateness_s",
        "snapshot_label", "map_stamp_summary_s", "map_age_summary_s",
        "maximum_map_age_s", "avoidance_source_accepted_frame_index",
        "avoidance_source_snapshot_id", "avoidance_map_revision",
        "avoidance_forecast_horizons_s",
        "avoidance_source_stamp_s",
        "avoidance_current_map_stamp_s",
        "avoidance_current_map_frame_id",
        "avoidance_current_forecast_horizon_s",
        "detector_input_cloud_stamp_s",
        "detector_input_cloud_frame_id"],
}


class ExperimentRecorder:
    def __init__(self):
        self.lock = threading.RLock()
        self.dual_lidar_condition = threading.Condition(self.lock)
        self.start_wall = time.monotonic()
        self.closed = False
        self.finalization_started = False
        self.finalized = False
        self.finalization_error = ""
        self.finalization_thread = None
        self.finalization_done = threading.Event()
        self.finalization_request = None
        self.required_artifact_errors = []
        self.finalization_high_water_timeout_s = float(rospy.get_param(
            "~finalization_high_water_timeout_s", 60.0))
        if (not math.isfinite(self.finalization_high_water_timeout_s)
                or self.finalization_high_water_timeout_s <= 0.0):
            raise ValueError("finalization_high_water_timeout_s must be positive")
        self.state = "STARTING"
        self.counts = defaultdict(int)
        self.latest_lio = None
        self.latest_truth = None
        self.initial_map_from_world = None
        self.initial_z_map_from_world = None
        self.initial_map_frame_id = ""
        self.latest_motion_truth_sim_time_s = None
        self.latest_motion_experiment_elapsed_s = None
        self.latest_motion_state = ""
        self.previous_motion_experiment_elapsed_s = None
        self.recorded_motion_experiment_elapsed_s = None
        self.truth_next_batch_index = 0
        self.truth_accepted_batch_count = 0
        self.truth_rejected_batch_count = 0
        self.truth_contract_input_errors = []
        self.recorded_lio_stamp_s = None
        self.recorded_navigation_stamp_s = None
        self.recorded_robot_truth_stamp_s = None
        self.recorded_detector_frame_index = -1
        self.recorded_evolution_frame_index = -1
        self.recorded_avoidance_frame_index = -1
        self.latest_maps = {}
        self.latest_map_metadata = {}
        self.latest_clouds = {}
        self.snapshotted_states = set()
        self.snapshot_in_progress_states = set()
        self.snapshot_metadata_by_state = {}
        self.dual_lidar_diagnostic_sequence = 0
        self.dual_lidar_request_sequence = None
        self.dual_lidar_final_sequence = None
        self.dual_lidar_request_monotonic_s = None
        self.dual_lidar_request_wall_time_s = None
        self.dual_lidar_latest_receive_monotonic_s = None
        self.dual_lidar_latest_receive_wall_time_s = None
        self.dual_lidar_final_receive_monotonic_s = None
        self.dual_lidar_final_receive_wall_time_s = None
        self.recorded_reference_keys = set()
        self.recorded_runtime_attempt_keys = set()
        self.monitoring_capture_open = True
        self.capture_freeze_state = ""
        self.capture_freeze_ros_time_s = None
        self.capture_cutoff_frame_index = None
        self.capture_cutoff_experiment_elapsed_s = None
        self.capture_freeze_committed = False
        self.freeze_on_detection_state = bool(rospy.get_param(
            "~freeze_on_detection_state", True))
        self.freeze_on_monitoring_complete = bool(rospy.get_param(
            "~freeze_on_monitoring_complete", False))
        self.return_after_detection = bool(rospy.get_param(
            "~mission/return_after_detection", True))
        # Disabled by default for isolated recorder/unit use.  The two formal
        # ICRA2027 dual-LiDAR launch modes set this explicitly to true.
        self.require_dual_lidar_success_contract = bool(rospy.get_param(
            "~require_dual_lidar_success_contract", False))
        self.dual_lidar_success_thresholds = configured_dual_lidar_thresholds(
            rospy.get_param(
                "~dual_lidar_contract/configured_sensor_update_rate_hz",
                DUAL_LIDAR_SUCCESS_THRESHOLDS[
                    "configured_sensor_update_rate_hz"]),
            rospy.get_param(
                "~dual_lidar_contract/configured_min_range_m",
                DUAL_LIDAR_SUCCESS_THRESHOLDS["configured_min_range_m"]),
            rospy.get_param(
                "~dual_lidar_contract/configured_max_range_m",
                DUAL_LIDAR_SUCCESS_THRESHOLDS["configured_max_range_m"]),
            rospy.get_param(
                "~dual_lidar_contract/configured_ray_samples_per_sensor",
                DUAL_LIDAR_SUCCESS_THRESHOLDS[
                    "configured_ray_samples_per_sensor"]),
            rospy.get_param(
                "~dual_lidar_contract/primary_downsample",
                DUAL_LIDAR_SUCCESS_THRESHOLDS["primary_downsample"]),
            rospy.get_param(
                "~dual_lidar_contract/secondary_downsample",
                DUAL_LIDAR_SUCCESS_THRESHOLDS["secondary_downsample"]))
        known_models = {"m01", "m02", "m03", "m04", "m05", "m06"}
        configured_primary = rospy.get_param(
            "~mission/required_detection_models",
            ["m01", "m02", "m03", "m05", "m06"])
        if isinstance(configured_primary, str):
            configured_primary = [value.strip()
                                  for value in configured_primary.split(",")
                                  if value.strip()]
        self.primary_metric_models = tuple(dict.fromkeys(
            str(value) for value in configured_primary))
        invalid_primary = set(self.primary_metric_models) - known_models
        if not self.primary_metric_models or invalid_primary:
            raise ValueError(
                "mission/required_detection_models must be a non-empty "
                "subset of m01..m06; invalid={}".format(
                    sorted(invalid_primary)))
        self.output = os.path.abspath(os.path.expanduser(rospy.get_param(
            "~output_directory", "/tmp/icra2027_experiment")))
        self.max_snapshot_points = int(rospy.get_param(
            "~max_snapshot_points", 500000))
        calibration = rospy.get_param(
            "~mission/truth_model_to_fastlio_body_xyzyaw",
            [0.2801, 0.0, 0.616928878, 0.0])
        self.calibration = tuple(float(value) for value in calibration)
        if (len(self.calibration) != 4
                or not all(math.isfinite(value) for value in self.calibration)):
            raise ValueError(
                "mission/truth_model_to_fastlio_body_xyzyaw must contain four "
                "finite [x,y,z,yaw] values")
        self.truth_association_aabb_margin_m = float(rospy.get_param(
            "~truth_association_aabb_margin_m", 0.25))
        if (not math.isfinite(self.truth_association_aabb_margin_m)
                or self.truth_association_aabb_margin_m < 0.0):
            raise ValueError(
                "truth_association_aabb_margin_m must be finite and nonnegative")
        self.truth_alignment_max_zoh_residual_s = float(rospy.get_param(
            "~truth_alignment_max_zoh_residual_s", 0.25))
        if (not math.isfinite(self.truth_alignment_max_zoh_residual_s)
                or self.truth_alignment_max_zoh_residual_s < 0.0):
            raise ValueError(
                "truth_alignment_max_zoh_residual_s must be finite and nonnegative")
        raw_checkpoints = rospy.get_param("~timeline_checkpoints_s", [])
        if isinstance(raw_checkpoints, str):
            raw_checkpoints = [value.strip()
                               for value in raw_checkpoints.split(",")
                               if value.strip()]
        self.requested_timeline_checkpoints_s = {
            float(value) for value in raw_checkpoints
            if math.isfinite(float(value)) and float(value) >= 0.0}
        self.pending_timeline_checkpoints_s = set(
            self.requested_timeline_checkpoints_s)
        configured = rospy.get_param("~mission/model_association_tolerance_m", {})
        self.tolerances = {
            str(name): (float(value[0]), float(value[1]))
            for name, value in configured.items() if len(value) == 2}
        if any(not all(math.isfinite(component) and component > 0.0
                       for component in tolerance)
               for tolerance in self.tolerances.values()):
            raise ValueError(
                "mission/model_association_tolerance_m values must be finite "
                "positive [xy_radius,z_radius] pairs")

        self.streams = {}
        self.writers = {}
        for relative, fields in CSV_SCHEMAS.items():
            path = os.path.join(self.output, relative)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Large detector tiles can contain thousands of physical samples.
            # Line buffering would issue one flush per CSV row and can steal a
            # substantial fraction of the simulation CPU budget.  Formal
            # finalization flushes and fsyncs every stream before analysis and
            # before run_manifest.json is committed, so a bounded userspace
            # buffer preserves the artifact contract without perturbing the
            # online detector cadence.
            stream = open(path, "w", newline="", buffering=1024 * 1024)
            writer = csv.DictWriter(stream, fieldnames=fields,
                                    extrasaction="ignore")
            writer.writeheader()
            self.streams[relative] = stream
            self.writers[relative] = writer
        os.makedirs(os.path.join(self.output, "snapshots", "maps"), exist_ok=True)
        os.makedirs(os.path.join(self.output, "snapshots", "clouds"), exist_ok=True)
        os.makedirs(os.path.join(self.output, "analysis"), exist_ok=True)
        self.timeline_status_publisher = rospy.Publisher(
            "/deform/experiment_recorder/status", String,
            queue_size=1, latch=True)

        self.navigation_plan_sequence = 0
        self.navigation_map_ids = set()
        rospy.Subscriber("/move_base/global_costmap/costmap", OccupancyGrid,
                         self._navigation_costmap_callback, queue_size=2)
        rospy.Subscriber("/move_base/NavfnROS/plan", Path,
                         self._global_plan_callback, queue_size=20)
        rospy.Subscriber("/move_base/goal", MoveBaseActionGoal,
                         self._navigation_goal_callback, queue_size=20)
        rospy.Subscriber("/move_base/result", MoveBaseActionResult,
                         self._navigation_result_callback, queue_size=20)

        rospy.Subscriber("/Odometry", Odometry, self._lio_callback, queue_size=50)
        rospy.Subscriber("/livox/imu", Imu, self._imu_callback, queue_size=1000)
        rospy.Subscriber("/navigation/odometry", Odometry,
                         self._navigation_callback, queue_size=50)
        rospy.Subscriber("/ground_truth/odom", Odometry, self._truth_callback,
                         queue_size=50)
        rospy.Subscriber("/cmd_vel", Twist, self._cmd_callback, queue_size=100)
        rospy.Subscriber("/icra2027_motion/truth", String,
                         self._object_truth_callback, queue_size=100)
        rospy.Subscriber("/livox/merge/diagnostics", String,
                         self._dual_lidar_diagnostics_callback, queue_size=20)
        # Recorder-only simulator telemetry.  It is never joined with truth,
        # never published back to the online chain, and is not an acceptance
        # gate; it distinguishes a slow physics/sensor schedule from GUI lag.
        rospy.Subscriber("/gazebo/performance_metrics", PerformanceMetrics,
                         self._gazebo_performance_callback, queue_size=10)
        rospy.Subscriber("/deform/mission/status", String,
                         self._mission_callback, queue_size=20)
        rospy.Subscriber("/deform/v1/status", V1DetectionStatus,
                         self._status_callback, queue_size=20)
        rospy.Subscriber("/deform/v1/detection_tiles", V1DetectionTile,
                         self._tile_callback, queue_size=30)
        rospy.Subscriber("/deform/evolution/frame", DeformationEvolutionFrame,
                         self._evolution_callback, queue_size=10)
        rospy.Subscriber("/deform/avoidance/frame", DeformationAvoidanceMap,
                         self._avoidance_callback, queue_size=10)
        rospy.Subscriber("/map", OccupancyGrid,
                         lambda msg: self._map_callback("slam_map", msg), queue_size=1)
        rospy.Subscriber("/mission/navigation_map", OccupancyGrid,
                         lambda msg: self._map_callback("navigation_map", msg), queue_size=1)
        rospy.Subscriber("/cloud_registered", PointCloud2,
                         lambda msg: self._cloud_callback("registered", msg),
                         queue_size=1, buff_size=64 * 1024 * 1024)
        rospy.Subscriber("/monitor/cloud_registered", PointCloud2,
                         lambda msg: self._cloud_callback("detector_input", msg),
                         queue_size=1, buff_size=64 * 1024 * 1024)
        rospy.on_shutdown(self.close)
        self._write_provenance()
        rospy.loginfo("ICRA2027 structured experiment output: %s", self.output)

    def _navigation_costmap_callback(self, message):
        # The frozen deformation layer is unchanged, but real LiDAR obstacles
        # remain live. Preserve published global costmaps independently of the
        # detector cutoff; publication time is NOT an exact per-plan binding.
        data = np.asarray(message.data, dtype=np.int16).reshape(
            message.info.height, message.info.width)
        origin = message.info.origin
        metadata = dict(frame_id=message.header.frame_id,
                        resolution=message.info.resolution,
                        origin=[origin.position.x, origin.position.y],
                        orientation=[origin.orientation.x, origin.orientation.y,
                                     origin.orientation.z, origin.orientation.w])
        encoded = json.dumps(metadata, sort_keys=True)
        digest = hashlib.sha256(encoded.encode() + data.tobytes()).hexdigest()
        relative = "navigation/costmaps/" + digest + ".npz"
        with self.lock:
            if self.closed:
                return
            if digest not in self.navigation_map_ids:
                path = os.path.join(self.output, relative)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                self._atomic_savez(path, data=data, metadata=encoded)
                self.navigation_map_ids.add(digest)
            row = self._base(message.header.stamp.to_sec())
            row.update(frame_id=message.header.frame_id, map_id=digest,
                       relative_path=relative)
            self._write("navigation/global_costmaps.csv", row)

    def _global_plan_callback(self, message):
        # Navigation continues after monitoring_capture_open becomes false.
        # A Path has no action goal ID: do not invent an exact association.
        with self.lock:
            self.navigation_plan_sequence += 1
            common = self._base(message.header.stamp.to_sec())
            common.update(plan_id=self.navigation_plan_sequence,
                          frame_id=message.header.frame_id,
                          point_count=len(message.poses))
            rows = [dict(common, point_index=index, x_m=pose.pose.position.x,
                         y_m=pose.pose.position.y,
                         yaw_rad=yaw_from_quaternion(pose.pose.orientation))
                    for index, pose in enumerate(message.poses)]
            if not rows:
                rows = [dict(common, point_index=-1)]
            self._write_batch("navigation/global_plans.csv", rows)

    def _navigation_goal_callback(self, message):
        pose = message.goal.target_pose
        row = self._base(message.header.stamp.to_sec())
        row.update(event="goal", goal_id=message.goal_id.id,
                   frame_id=pose.header.frame_id,
                   x_m=pose.pose.position.x, y_m=pose.pose.position.y)
        self._write("navigation/action_events.csv", row)

    def _navigation_result_callback(self, message):
        row = self._base(message.header.stamp.to_sec())
        row.update(event="result", goal_id=message.status.goal_id.id,
                   status=message.status.status, text=message.status.text)
        self._write("navigation/action_events.csv", row)

    def _base(self, message_time_s=None, publish_time_s=None):
        receive_time_s = rospy.Time.now().to_sec()
        with self.lock:
            truth_sim_time_s = self.latest_motion_truth_sim_time_s
            experiment_elapsed_s = self.latest_motion_experiment_elapsed_s
            motion_state = self.latest_motion_state
        elapsed_at_message = experiment_elapsed_at_message(
            message_time_s, truth_sim_time_s, experiment_elapsed_s,
            motion_state)
        return {"ros_time_s": receive_time_s,
                "receive_time_s": receive_time_s,
                "message_time_s": (message_time_s
                                     if message_time_s is not None else ""),
                "publish_time_s": (publish_time_s
                                     if publish_time_s is not None else ""),
                "truth_sim_time_s_at_receive": (
                    truth_sim_time_s if truth_sim_time_s is not None else ""),
                "experiment_elapsed_s_at_receive": (
                    experiment_elapsed_s
                    if experiment_elapsed_s is not None else ""),
                "experiment_elapsed_s_at_message": (
                    elapsed_at_message if elapsed_at_message is not None else ""),
                "wall_elapsed_s": time.monotonic() - self.start_wall,
                "mission_state": self.state}

    def _write(self, relative, row):
        with self.lock:
            if self.closed:
                return False
            self.writers[relative].writerow(row)
            self.counts[relative] += 1
            return True

    def _write_batch(self, relative, rows):
        """Commit one validated logical batch while holding the stream lock."""
        with self.lock:
            if self.closed:
                return False
            self.writers[relative].writerows(rows)
            self.counts[relative] += len(rows)
            return True

    def _record_truth_contract_error(self, error):
        with self.lock:
            self.truth_rejected_batch_count += 1
            # Retain a bounded diagnostic sample while the independent count
            # preserves the fact that every rejected input invalidates a run.
            if len(self.truth_contract_input_errors) < 256:
                self.truth_contract_input_errors.append(str(error))

    def _publish_timeline_status(self):
        with self.lock:
            latest_elapsed = self.latest_motion_experiment_elapsed_s
            latest_sim = self.latest_motion_truth_sim_time_s
            completed = sorted(self.requested_timeline_checkpoints_s
                               - self.pending_timeline_checkpoints_s)
            pending = sorted(self.pending_timeline_checkpoints_s)
            truth_rows = int(self.counts["truth/m01_m06_motion_truth.csv"])
            truth_accepted_batches = int(self.truth_accepted_batch_count)
            truth_rejected_batches = int(self.truth_rejected_batch_count)
            finalization_started = bool(self.finalization_started)
            finalized = bool(self.finalized)
            finalization_error = str(self.finalization_error)
            required_artifact_errors = list(self.required_artifact_errors)
            request = copy.deepcopy(self.finalization_request)
            recorded_high_water = {
                "motion_experiment_elapsed_s": (
                    self.recorded_motion_experiment_elapsed_s),
                "navigation_stamp_s": self.recorded_navigation_stamp_s,
                "robot_truth_stamp_s": self.recorded_robot_truth_stamp_s,
                "detector_accepted_frame_index": (
                    self.recorded_detector_frame_index),
                "evolution_accepted_frame_index": (
                    self.recorded_evolution_frame_index),
                "avoidance_accepted_frame_index": (
                    self.recorded_avoidance_frame_index),
            }
        message = String()
        message.data = json.dumps({
            "schema_version": 2,
            "latest_experiment_elapsed_s": latest_elapsed,
            "latest_truth_sim_time_s": latest_sim,
            "completed_checkpoints_s": completed,
            "pending_checkpoints_s": pending,
            "truth_row_count": truth_rows,
            "truth_accepted_batch_count": truth_accepted_batches,
            "truth_rejected_batch_count": truth_rejected_batches,
            "finalization_started": finalization_started,
            "finalized": finalized,
            "finalization_error": finalization_error,
            "required_artifact_errors": required_artifact_errors,
            "run_manifest": ("run_manifest.json" if finalized else ""),
            "finalization_request_id": (
                request.get("request_id", "") if request else ""),
            "recorded_high_water": recorded_high_water,
        }, sort_keys=True)
        self.timeline_status_publisher.publish(message)

    @staticmethod
    def _pose(message):
        pose = message.pose.pose
        return (float(pose.position.x), float(pose.position.y),
                float(pose.position.z), yaw_from_quaternion(pose.orientation))

    @staticmethod
    def _pose_se3(message):
        pose = message.pose.pose
        return ((float(pose.position.x), float(pose.position.y),
                 float(pose.position.z)),
                normalized_quaternion((pose.orientation.x, pose.orientation.y,
                                       pose.orientation.z, pose.orientation.w)))

    def _truth_body_se3(self, message):
        position, orientation = self._pose_se3(message)
        return compose_body_pose_se3(position, orientation, self.calibration)

    def _truth_body(self, message):
        position, orientation = self._truth_body_se3(message)
        yaw = transformations.euler_from_quaternion(orientation)[2]
        return (position[0], position[1], position[2], yaw)

    def _current_transform(self):
        if self.latest_lio is None or self.latest_truth is None:
            return None
        map_pose = self._pose(self.latest_lio)
        world_body = self._truth_body(self.latest_truth)
        return map_from_world_anchor(
            (map_pose[0], map_pose[1], map_pose[3]),
            (world_body[0], world_body[1], world_body[3]))

    def _lio_callback(self, message):
        with self.lock:
            self.latest_lio = message
            if self.initial_map_from_world is None and self.latest_truth is not None:
                self.initial_map_from_world = self._current_transform()
                self.initial_z_map_from_world = (
                    self._pose(message)[2]
                    - self._truth_body(self.latest_truth)[2])
                self.initial_map_frame_id = str(message.header.frame_id)
        pose = message.pose.pose
        twist = message.twist.twist
        row = self._base(message.header.stamp.to_sec())
        row.update({
            "frame_id": message.header.frame_id,
            "child_frame_id": message.child_frame_id,
            "x_m": pose.position.x, "y_m": pose.position.y, "z_m": pose.position.z,
            "qx": pose.orientation.x, "qy": pose.orientation.y,
            "qz": pose.orientation.z, "qw": pose.orientation.w,
            "yaw_rad": yaw_from_quaternion(pose.orientation),
            "vx_mps": twist.linear.x, "vy_mps": twist.linear.y,
            "vz_mps": twist.linear.z, "wz_rad_s": twist.angular.z})
        if self._write("trajectories/slam_odometry.csv", row):
            with self.lock:
                self.recorded_lio_stamp_s = message.header.stamp.to_sec()

    def _imu_callback(self, message):
        row = self._base(message.header.stamp.to_sec())
        row.update(frame_id=message.header.frame_id,
                   wx_rad_s=message.angular_velocity.x,
                   wy_rad_s=message.angular_velocity.y,
                   wz_rad_s=message.angular_velocity.z,
                   ax_mps2=message.linear_acceleration.x,
                   ay_mps2=message.linear_acceleration.y,
                   az_mps2=message.linear_acceleration.z)
        self._write("sensors/imu.csv", row)

    def _navigation_callback(self, message):
        """Record the calibrated footprint-centre pose used by navigation."""
        pose = message.pose.pose
        twist = message.twist.twist
        row = self._base(message.header.stamp.to_sec())
        row.update({
            "frame_id": message.header.frame_id,
            "child_frame_id": message.child_frame_id,
            "x_m": pose.position.x, "y_m": pose.position.y,
            "z_m": pose.position.z,
            "qx": pose.orientation.x, "qy": pose.orientation.y,
            "qz": pose.orientation.z, "qw": pose.orientation.w,
            "yaw_rad": yaw_from_quaternion(pose.orientation),
            "vx_mps": twist.linear.x, "vy_mps": twist.linear.y,
            "vz_mps": twist.linear.z, "wz_rad_s": twist.angular.z})
        if self._write("trajectories/navigation_base_odometry.csv", row):
            with self.lock:
                self.recorded_navigation_stamp_s = message.header.stamp.to_sec()

    def _truth_callback(self, message):
        with self.lock:
            self.latest_truth = message
            if self.initial_map_from_world is None and self.latest_lio is not None:
                self.initial_map_from_world = self._current_transform()
                self.initial_z_map_from_world = (
                    self._pose(self.latest_lio)[2]
                    - self._truth_body(message)[2])
                self.initial_map_frame_id = str(
                    self.latest_lio.header.frame_id)
            anchor = self.initial_map_from_world
            map_frame_id = self.initial_map_frame_id
        model = self._pose(message)
        body = self._truth_body(message)
        model_position, model_orientation = self._pose_se3(message)
        body_position, body_orientation = self._truth_body_se3(message)
        truth_map = transform_point(anchor, body[:2]) if anchor else ("", "")
        twist = message.twist.twist
        row = self._base(message.header.stamp.to_sec())
        row.update({
            "frame_id": message.header.frame_id,
            "child_frame_id": message.child_frame_id,
            "model_x_m": model[0], "model_y_m": model[1],
            "model_z_m": model[2], "model_yaw_rad": model[3],
            "model_qx": model_orientation[0], "model_qy": model_orientation[1],
            "model_qz": model_orientation[2], "model_qw": model_orientation[3],
            "body_x_m": body[0], "body_y_m": body[1],
            "body_z_m": body[2], "body_yaw_rad": body[3],
            "body_qx": body_orientation[0], "body_qy": body_orientation[1],
            "body_qz": body_orientation[2], "body_qw": body_orientation[3],
            "truth_map_frame_id": map_frame_id if anchor else "",
            "truth_map_x_m": truth_map[0], "truth_map_y_m": truth_map[1],
            "truth_map_yaw_rad": body[3] + anchor[2] if anchor else "",
            "vx_mps": twist.linear.x, "vy_mps": twist.linear.y,
            "vz_mps": twist.linear.z, "wz_rad_s": twist.angular.z})
        if self._write("trajectories/gazebo_robot_truth.csv", row):
            with self.lock:
                self.recorded_robot_truth_stamp_s = message.header.stamp.to_sec()

    def _cmd_callback(self, message):
        row = self._base()
        row.update({"vx_mps": message.linear.x, "vy_mps": message.linear.y,
                    "vz_mps": message.linear.z, "wx_rad_s": message.angular.x,
                    "wy_rad_s": message.angular.y, "wz_rad_s": message.angular.z})
        self._write("trajectories/cmd_vel.csv", row)

    def _object_truth_callback(self, message):
        if not self.monitoring_capture_open:
            return
        try:
            document = json.loads(message.data)
        except (TypeError, ValueError):
            self._record_truth_contract_error("invalid_json")
            rospy.logerr_throttle(2.0, "experiment recorder rejected invalid truth JSON")
            return
        validation = validate_motion_truth_batch(document)
        if validation["status"] != "COMPLETED":
            diagnostic = "invalid_batch:" + "|".join(validation["errors"])
            self._record_truth_contract_error(diagnostic)
            rospy.logerr_throttle(
                2.0, "experiment recorder rejected invalid truth batch: %s",
                diagnostic)
            self._publish_timeline_status()
            return

        truth_sim_time_s = float(document["sim_time"])
        experiment_elapsed_s = float(document["experiment_elapsed"])
        with self.lock:
            previous_elapsed_s = self.previous_motion_experiment_elapsed_s
            truth_batch_index = self.truth_next_batch_index
            self.truth_next_batch_index += 1
        truth_rows = []
        for model in document["models"]:
            name = str(model["model_name"])
            position = tuple(float(value) for value in model["position"])
            orientation = tuple(float(value) for value in model["orientation"])
            translation = tuple(float(value) for value in model["translation"])
            expected_translation = tuple(
                float(value) for value in model["expected_translation_vector"])
            aabb_min = tuple(float(value) for value in model["aabb_min"])
            aabb_max = tuple(float(value) for value in model["aabb_max"])
            row = self._base()
            row.update({
                "truth_sim_time_s": truth_sim_time_s,
                # The motion-plugin contract accepts only command_frame=world;
                # make that coordinate convention explicit in every raw row
                # instead of relying on model names or an external note.
                "truth_frame_id": "world",
                "truth_batch_index": truth_batch_index,
                "truth_schema_version": document["schema_version"],
                "motion_state": document["state"],
                "experiment_elapsed_s": experiment_elapsed_s,
                "waiting_elapsed_s": document["waiting_elapsed"],
                "motion_elapsed_s": document["motion_elapsed"],
                "model_motion_elapsed_s": model["motion_elapsed"],
                "motion_mode": document["motion_mode"],
                "per_model_duration_s": document["per_model_duration"],
                "linear_speed_mm_s": document["linear_speed_mm_s"],
                "angular_speed_rad_s": document["angular_speed_rad_s"],
                "model_name": name,
                "laser_retro_id": model["laser_retro_id"],
                "position_x_m": position[0],
                "position_y_m": position[1],
                "position_z_m": position[2],
                "orientation_x": orientation[0],
                "orientation_y": orientation[1],
                "orientation_z": orientation[2],
                "orientation_w": orientation[3],
                "world_aabb_min_x_m": aabb_min[0],
                "world_aabb_min_y_m": aabb_min[1],
                "world_aabb_min_z_m": aabb_min[2],
                "world_aabb_max_x_m": aabb_max[0],
                "world_aabb_max_y_m": aabb_max[1],
                "world_aabb_max_z_m": aabb_max[2],
                "translation_x_m": translation[0],
                "translation_y_m": translation[1],
                "translation_z_m": translation[2],
                "expected_translation_x_m": expected_translation[0],
                "expected_translation_y_m": expected_translation[1],
                "expected_translation_z_m": expected_translation[2],
                "translation_magnitude_m": model["translation_magnitude"],
                "translation_expected_m": model["translation_expected"],
                "translation_error_m": model["translation_error"],
                "rotation_angle_rad": model["rotation_angle"],
                "rotation_expected_rad": model["rotation_expected"],
                "rotation_error_rad": model["rotation_error"],
                "surface_rms_displacement_m": model[
                    "surface_rms_displacement"],
                "surface_max_displacement_m": model[
                    "surface_max_displacement"],
                "surface_point_count": model["surface_point_count"]})
            truth_rows.append(row)
        write_error_recorded = False
        try:
            truth_batch_committed = self._write_batch(
                "truth/m01_m06_motion_truth.csv", truth_rows)
        except (OSError, ValueError, TypeError) as error:
            truth_batch_committed = False
            write_error_recorded = True
            self._record_truth_contract_error(
                "batch_{}_write_failed:{}".format(truth_batch_index, error))
        if not truth_batch_committed:
            if not write_error_recorded:
                self._record_truth_contract_error(
                    "batch_{}_write_not_committed".format(truth_batch_index))
            self._publish_timeline_status()
            return

        with self.lock:
            self.truth_accepted_batch_count += 1
            self.latest_motion_truth_sim_time_s = truth_sim_time_s
            self.latest_motion_experiment_elapsed_s = experiment_elapsed_s
            self.latest_motion_state = str(document["state"])
            self.previous_motion_experiment_elapsed_s = experiment_elapsed_s
            checkpoints = timeline_checkpoints_crossed(
                previous_elapsed_s, experiment_elapsed_s,
                self.pending_timeline_checkpoints_s)
        checkpoints_committed = True
        for checkpoint in checkpoints:
            label = "CHECKPOINT_{:06.2f}s".format(checkpoint)
            receive_time_s = rospy.Time.now().to_sec()
            # Snapshot first and use the exact metadata copy returned by that
            # operation.  This prevents an asynchronous map callback between
            # the CSV row and NPZ save from giving them different revisions.
            snapshot_map_metadata, snapshot_committed = self._snapshot_phase(label)
            if not snapshot_committed:
                checkpoints_committed = False
                # A checkpoint can be crossed before the first complete map
                # callback of that same simulation instant.  The checkpoint
                # remains pending and is retried on the next truth batch, so
                # this is expected startup back-pressure rather than data
                # loss or a recorder failure.
                rospy.logwarn_throttle(
                    2.0, "checkpoint %.2f s snapshot not committed; retrying",
                    checkpoint)
                continue
            map_metrics = summarize_map_ages(
                truth_sim_time_s, snapshot_map_metadata)
            with self.lock:
                snapshot_product_metadata = copy.deepcopy(
                    self.snapshot_metadata_by_state.get(label, {}))
            checkpoint_row = {
                "ros_time_s": receive_time_s,
                "truth_sim_time_s": (truth_sim_time_s
                                     if truth_sim_time_s is not None else ""),
                "receive_time_s": receive_time_s,
                "wall_elapsed_s": time.monotonic() - self.start_wall,
                "mission_state": self.state,
                "motion_state": document.get("state", ""),
                "target_experiment_elapsed_s": checkpoint,
                "actual_experiment_elapsed_s": (experiment_elapsed_s
                                                  if experiment_elapsed_s
                                                  is not None else ""),
                "checkpoint_lateness_s": (
                    experiment_elapsed_s - checkpoint
                    if experiment_elapsed_s is not None else ""),
                "snapshot_label": label}
            checkpoint_row.update(map_metrics)
            checkpoint_row.update(summarize_checkpoint_products(
                snapshot_product_metadata))
            checkpoint_committed = self._write(
                "events/timeline_checkpoints.csv", checkpoint_row)
            if checkpoint_committed:
                with self.lock:
                    self.pending_timeline_checkpoints_s.discard(checkpoint)
            else:
                checkpoints_committed = False
        if checkpoints_committed:
            with self.lock:
                self.recorded_motion_experiment_elapsed_s = max(
                    float(experiment_elapsed_s),
                    (self.recorded_motion_experiment_elapsed_s
                     if self.recorded_motion_experiment_elapsed_s is not None
                     else float("-inf")))
            self._maybe_commit_capture_freeze()
        # Publish after every accepted six-model batch. The stationary mission
        # can advance only when each crossed checkpoint bundle also committed.
        self._publish_timeline_status()

    def _mission_callback(self, message):
        try:
            document = json.loads(message.data)
        except (TypeError, ValueError):
            return
        state = str(document.get("state", "UNKNOWN"))
        with self.lock:
            changed = state != self.state
            self.state = state
        if changed:
            receive_time_s = rospy.Time.now().to_sec()
            self._write("events/mission_states.csv", {
                "ros_time_s": document.get("sim_time_s", rospy.Time.now().to_sec()),
                "message_time_s": document.get("sim_time_s", ""),
                "receive_time_s": receive_time_s,
                "wall_elapsed_s": time.monotonic() - self.start_wall,
                "state": state, "detail": document.get("detail", "")})
            if self.monitoring_capture_open:
                self._snapshot_phase(state)
            freeze_requested = (
                (state == "DEFORMATION_DETECTED"
                 and self.freeze_on_detection_state)
                or (state == "MONITORING_COMPLETE"
                    and self.freeze_on_monitoring_complete))
            if freeze_requested:
                cutoff = document.get("capture_cutoff", {})
                try:
                    cutoff_frame = int(cutoff["accepted_frame_index"])
                    cutoff_elapsed = float(
                        cutoff["motion_experiment_elapsed_s"])
                    if (cutoff_frame < 0
                            or not math.isfinite(cutoff_elapsed)
                            or cutoff_elapsed < 0.0):
                        raise ValueError("invalid capture cutoff")
                except (KeyError, TypeError, ValueError):
                    rospy.logerr(
                        "recorder rejected %s without a valid capture cutoff",
                        state)
                    return
                with self.lock:
                    self.capture_freeze_state = state
                    self.capture_freeze_ros_time_s = float(
                        document.get("sim_time_s", rospy.Time.now().to_sec()))
                    self.capture_cutoff_frame_index = cutoff_frame
                    self.capture_cutoff_experiment_elapsed_s = cutoff_elapsed
                self._maybe_commit_capture_freeze()
            if state == "COMPLETED":
                self._start_finalization(document.get(
                    "finalization_request", {}))

    def _maybe_commit_capture_freeze(self):
        with self.lock:
            if (not self.monitoring_capture_open
                    or self.capture_cutoff_frame_index is None
                    or self.capture_cutoff_experiment_elapsed_s is None):
                return self.capture_freeze_committed
            frame = self.capture_cutoff_frame_index
            elapsed = self.capture_cutoff_experiment_elapsed_s
            ready = (
                self.recorded_detector_frame_index >= frame
                and self.recorded_evolution_frame_index >= frame
                and self.recorded_avoidance_frame_index >= frame
                and self.recorded_motion_experiment_elapsed_s is not None
                and self.recorded_motion_experiment_elapsed_s + 1.0e-9
                >= elapsed)
            if ready:
                self.monitoring_capture_open = False
                self.capture_freeze_committed = True
            return self.capture_freeze_committed

    def _validate_finalization_request(self, request):
        if not isinstance(request, dict):
            raise ValueError("missing finalization request")
        required = (
            "request_id", "run_session_id", "mission_outcome",
            "output_directory", "decision_summary_relative_path",
            "decision_summary_sha256")
        if any(not request.get(key) for key in required):
            raise ValueError("incomplete finalization request")
        if request["request_id"] != request["run_session_id"]:
            raise ValueError("finalization request/session mismatch")
        if request["mission_outcome"] != "COMPLETED":
            raise ValueError("finalization requested for non-completed mission")
        if os.path.abspath(request["output_directory"]) != self.output:
            raise ValueError("finalization output directory mismatch")
        relative = str(request["decision_summary_relative_path"])
        if (os.path.isabs(relative) or os.path.dirname(relative)
                or relative != "mission_decision_summary.json"):
            raise ValueError("invalid decision summary path")
        path = os.path.join(self.output, relative)
        if not os.path.isfile(path):
            raise ValueError("decision summary is missing")
        if self._sha256(path) != str(request["decision_summary_sha256"]):
            raise ValueError("decision summary digest mismatch")
        with open(path, "r", encoding="utf-8") as stream:
            summary = json.load(stream)
        if (summary.get("run_session_id") != request["run_session_id"]
                or summary.get("outcome") != "COMPLETED"):
            raise ValueError("decision summary identity/outcome mismatch")

    @staticmethod
    def _required_number(request, key, integer=False):
        value = request.get(key, None)
        if value is None:
            return None
        result = int(value) if integer else float(value)
        if ((integer and result < 0)
                or (not integer and (not math.isfinite(result)
                                     or result < 0.0))):
            raise ValueError("invalid finalization high-water " + key)
        return result

    def _finalization_high_water(self, request):
        motion = self._required_number(
            request, "required_motion_experiment_elapsed_s")
        detector = self._required_number(
            request, "required_detector_accepted_frame_index", integer=True)
        evolution = self._required_number(
            request, "required_evolution_accepted_frame_index", integer=True)
        avoidance = self._required_number(
            request, "required_avoidance_accepted_frame_index", integer=True)
        navigation = self._required_number(
            request, "required_navigation_stamp_s")
        robot_truth = self._required_number(
            request, "required_robot_truth_stamp_s")
        # The online mission deliberately never subscribes robot truth.  Both
        # odometry streams use the same ROS simulation clock, so the recorder
        # independently requires its truth stream to reach the mission's final
        # navigation stamp when no explicit legacy truth high-water is given.
        if robot_truth is None:
            robot_truth = navigation
        with self.lock:
            required = {
                "motion_experiment_elapsed_s": motion,
                "detector_accepted_frame_index": detector,
                "evolution_accepted_frame_index": evolution,
                "avoidance_accepted_frame_index": avoidance,
                "navigation_stamp_s": navigation,
                "robot_truth_stamp_s": robot_truth,
            }
            recorded = {
                "motion_experiment_elapsed_s": (
                    self.recorded_motion_experiment_elapsed_s),
                "detector_accepted_frame_index": (
                    self.recorded_detector_frame_index),
                "evolution_accepted_frame_index": (
                    self.recorded_evolution_frame_index),
                "avoidance_accepted_frame_index": (
                    self.recorded_avoidance_frame_index),
                "navigation_stamp_s": self.recorded_navigation_stamp_s,
                "robot_truth_stamp_s": self.recorded_robot_truth_stamp_s,
            }
            checks = finalization_high_water_checks(
                required, recorded,
                self.freeze_on_detection_state
                or self.freeze_on_monitoring_complete,
                self.capture_freeze_committed,
                len(self.pending_timeline_checkpoints_s))
        return all(checks.values()), [
            name for name, satisfied in checks.items() if not satisfied]

    def _start_finalization(self, request):
        """Finalize off-callback while the mission keeps roslaunch alive."""
        with self.lock:
            if self.finalization_started:
                return
            self.finalization_started = True
            self.finalization_error = ""
            self.finalization_request = copy.deepcopy(request)
            self.dual_lidar_request_monotonic_s = time.monotonic()
            self.dual_lidar_request_wall_time_s = time.time()
            self.dual_lidar_request_sequence = int(
                self.dual_lidar_diagnostic_sequence)
        self._publish_timeline_status()
        thread = threading.Thread(
            target=self._finalize_completed_run,
            name="icra2027_recorder_finalization", daemon=False)
        with self.lock:
            self.finalization_thread = thread
        thread.start()

    def _wait_for_post_request_dual_lidar_diagnostic(self, deadline):
        """Require one fully written 1 Hz diagnostic after the close request."""
        if not self.require_dual_lidar_success_contract:
            return
        with self.dual_lidar_condition:
            request_sequence = self.dual_lidar_request_sequence
            request_monotonic_s = self.dual_lidar_request_monotonic_s
            if request_sequence is None or request_monotonic_s is None:
                raise RuntimeError(
                    "dual-LiDAR request sequence/time was not captured")

            def fresh():
                return (
                    self.dual_lidar_diagnostic_sequence > request_sequence
                    and self.dual_lidar_latest_receive_monotonic_s is not None
                    and self.dual_lidar_latest_receive_monotonic_s
                    > request_monotonic_s)

            while not fresh() and not rospy.is_shutdown():
                remaining = float(deadline) - time.monotonic()
                if remaining <= 0.0:
                    break
                self.dual_lidar_condition.wait(timeout=min(remaining, 0.5))
            if not fresh():
                raise RuntimeError(
                    "finalization high-water timeout: dual_lidar_fresh_diagnostic")
            self.dual_lidar_final_sequence = int(
                self.dual_lidar_diagnostic_sequence)
            self.dual_lidar_final_receive_monotonic_s = float(
                self.dual_lidar_latest_receive_monotonic_s)
            self.dual_lidar_final_receive_wall_time_s = (
                self.dual_lidar_latest_receive_wall_time_s)

    def _finalize_completed_run(self):
        try:
            with self.lock:
                request = copy.deepcopy(self.finalization_request)
            self._validate_finalization_request(request)
            deadline = time.monotonic() + self.finalization_high_water_timeout_s
            missing = []
            while time.monotonic() < deadline and not rospy.is_shutdown():
                ready, missing = self._finalization_high_water(request)
                if ready:
                    break
                time.sleep(0.02)
            else:
                raise RuntimeError(
                    "finalization high-water timeout: " + ",".join(missing))
            # The 1 Hz diagnostic is itself part of the formal E2E contract.
            # Requiring sequence(request)+1 *and* callback-receive monotonic
            # time strictly after the request rejects a callback which began
            # before the boundary but wrote afterwards. It shares the existing
            # 60 s wall-clock high-water deadline because /clock may run well
            # below real time.
            self._wait_for_post_request_dual_lidar_diagnostic(deadline)
            self._finalize_files(request)
            manifest = os.path.join(self.output, "run_manifest.json")
            if not os.path.isfile(manifest):
                raise OSError("run_manifest.json was not committed")
            with self.lock:
                self.finalized = True
        except Exception as error:  # terminal status is consumed by mission
            with self.lock:
                self.finalization_error = str(error)
            rospy.logerr("experiment recorder finalization failed: %s", error)
        self.finalization_done.set()
        self._publish_timeline_status()

    def _dual_lidar_diagnostics_callback(self, message):
        receive_monotonic_s = time.monotonic()
        receive_wall_time_s = time.time()
        try:
            document = json.loads(message.data)
        except (TypeError, ValueError):
            rospy.logerr_throttle(
                2.0, "experiment recorder rejected invalid dual-LiDAR diagnostics")
            return
        receive_time_s = rospy.Time.now().to_sec()
        primary_received = int(document.get("primary_received", 0))
        secondary_received = int(document.get("secondary_received", 0))
        pairs_accepted = int(document.get("pairs_accepted", 0))
        row = {
            "ros_time_s": receive_time_s,
            "diagnostic_time_s": document.get("ros_time_s", ""),
            "receive_time_s": receive_time_s,
            "receive_monotonic_s": receive_monotonic_s,
            "receive_wall_time_s": receive_wall_time_s,
            "wall_elapsed_s": time.monotonic() - self.start_wall,
            "mission_state": self.state,
            "schema_version": document.get("schema_version", ""),
            "secondary_enabled": document.get("secondary_enabled", ""),
            "zero_no_return_compaction_enabled": document.get(
                "zero_no_return_compaction_enabled", ""),
            "point_time_mode": document.get("point_time_mode", ""),
            "configured_sensor_update_rate_hz": document.get(
                "configured_sensor_update_rate_hz", ""),
            "configured_min_range_m": document.get(
                "configured_min_range_m", ""),
            "configured_max_range_m": document.get(
                "configured_max_range_m", ""),
            "configured_ray_samples_per_sensor": document.get(
                "configured_ray_samples_per_sensor", ""),
            "primary_scheduler_catch_up_suppressed_count": document.get(
                "primary_scheduler_catch_up_suppressed_count", ""),
            "primary_downsample": document.get("primary_downsample", ""),
            "secondary_downsample": document.get("secondary_downsample", ""),
            "configured_secondary_to_primary_point_ratio": document.get(
                "configured_secondary_to_primary_point_ratio", ""),
            "overlap_voxel_size_m": document.get(
                "overlap_voxel_size_m", ""),
            "overlap_evaluation_pair_stride": document.get(
                "overlap_evaluation_pair_stride", ""),
            "primary_received": primary_received,
            "secondary_received": secondary_received,
            "pairs_accepted": pairs_accepted,
            "pairs_rejected": document.get("pairs_rejected", 0),
            "primary_unpaired_dropped": document.get(
                "primary_unpaired_dropped", 0),
            "secondary_unpaired_dropped": document.get(
                "secondary_unpaired_dropped", 0),
            "primary_queue_dropped": document.get("primary_queue_dropped", 0),
            "secondary_queue_dropped": document.get(
                "secondary_queue_dropped", 0),
            "last_sync_delta_ns": document.get("last_sync_delta_ns", ""),
            "maximum_sync_delta_ns": document.get("maximum_sync_delta_ns", ""),
            "mean_sync_delta_ns": document.get("mean_sync_delta_ns", ""),
            "first_accepted_measurement_time_ns": document.get(
                "first_accepted_measurement_time_ns", ""),
            "last_accepted_measurement_time_ns": document.get(
                "last_accepted_measurement_time_ns", ""),
            "accepted_measurement_interval_count": document.get(
                "accepted_measurement_interval_count", ""),
            "nonpositive_accepted_measurement_interval_count": document.get(
                "nonpositive_accepted_measurement_interval_count", ""),
            "minimum_accepted_measurement_interval_ns": document.get(
                "minimum_accepted_measurement_interval_ns", ""),
            "maximum_accepted_measurement_interval_ns": document.get(
                "maximum_accepted_measurement_interval_ns", ""),
            "mean_accepted_measurement_interval_ns": document.get(
                "mean_accepted_measurement_interval_ns", ""),
            "realized_measurement_hz": document.get(
                "realized_measurement_hz", ""),
            "last_merged_span_ns": document.get("last_merged_span_ns", ""),
            "maximum_merged_span_ns": document.get(
                "maximum_merged_span_ns", ""),
            "last_primary_point_count": document.get(
                "last_primary_point_count", ""),
            "maximum_primary_point_count": document.get(
                "maximum_primary_point_count", ""),
            "last_secondary_point_count": document.get(
                "last_secondary_point_count", ""),
            "maximum_secondary_point_count": document.get(
                "maximum_secondary_point_count", ""),
            "last_merged_point_count": document.get(
                "last_merged_point_count", ""),
            "maximum_merged_point_count": document.get(
                "maximum_merged_point_count", ""),
            "last_secondary_to_primary_point_ratio": document.get(
                "last_secondary_to_primary_point_ratio", ""),
            "last_merged_to_primary_point_ratio": document.get(
                "last_merged_to_primary_point_ratio", ""),
            "last_total_raw_input_point_count": document.get(
                "last_total_raw_input_point_count", ""),
            "maximum_total_raw_input_point_count": document.get(
                "maximum_total_raw_input_point_count", ""),
            "last_compacted_output_point_count": document.get(
                "last_compacted_output_point_count", ""),
            "maximum_compacted_output_point_count": document.get(
                "maximum_compacted_output_point_count", ""),
            "last_removed_no_return_sentinel_count": document.get(
                "last_removed_no_return_sentinel_count", ""),
            "maximum_removed_no_return_sentinel_count": document.get(
                "maximum_removed_no_return_sentinel_count", ""),
            "last_compacted_output_to_total_raw_input_ratio": document.get(
                "last_compacted_output_to_total_raw_input_ratio", ""),
            "overlap_evaluations_attempted": document.get(
                "overlap_evaluations_attempted", 0),
            "overlap_evaluations_failed": document.get(
                "overlap_evaluations_failed", 0),
            "overlap_evaluation_sample_count": document.get(
                "overlap_evaluation_sample_count", 0),
            "last_overlap_pair_index": document.get(
                "last_overlap_pair_index", ""),
            "last_primary_valid_return_count": document.get(
                "last_primary_valid_return_count", ""),
            "last_secondary_valid_return_count": document.get(
                "last_secondary_valid_return_count", ""),
            "last_primary_occupied_voxel_count": document.get(
                "last_primary_occupied_voxel_count", ""),
            "last_secondary_occupied_voxel_count": document.get(
                "last_secondary_occupied_voxel_count", ""),
            "last_shared_occupied_voxel_count": document.get(
                "last_shared_occupied_voxel_count", ""),
            "last_union_occupied_voxel_count": document.get(
                "last_union_occupied_voxel_count", ""),
            "last_primary_overlap_fraction": document.get(
                "last_primary_overlap_fraction", ""),
            "last_secondary_overlap_fraction": document.get(
                "last_secondary_overlap_fraction", ""),
            "last_voxel_jaccard": document.get("last_voxel_jaccard", ""),
            "mean_primary_overlap_fraction": document.get(
                "mean_primary_overlap_fraction", ""),
            "mean_secondary_overlap_fraction": document.get(
                "mean_secondary_overlap_fraction", ""),
            "mean_voxel_jaccard": document.get("mean_voxel_jaccard", ""),
            "minimum_primary_overlap_fraction": document.get(
                "minimum_primary_overlap_fraction", ""),
            "minimum_secondary_overlap_fraction": document.get(
                "minimum_secondary_overlap_fraction", ""),
            "minimum_voxel_jaccard": document.get(
                "minimum_voxel_jaccard", ""),
            "primary_pair_ratio": (pairs_accepted / float(primary_received)
                                   if primary_received else ""),
            "secondary_pair_ratio": (pairs_accepted / float(secondary_received)
                                     if secondary_received else ""),
        }
        if self._write("sensors/dual_lidar_merge_diagnostics.csv", row):
            with self.dual_lidar_condition:
                self.dual_lidar_diagnostic_sequence += 1
                if (self.dual_lidar_latest_receive_monotonic_s is None
                        or receive_monotonic_s
                        > self.dual_lidar_latest_receive_monotonic_s):
                    self.dual_lidar_latest_receive_monotonic_s = (
                        receive_monotonic_s)
                    self.dual_lidar_latest_receive_wall_time_s = (
                        receive_wall_time_s)
                self.dual_lidar_condition.notify_all()

    def _gazebo_performance_callback(self, message):
        """Record simulator performance without affecting online sensing."""
        receive_monotonic_s = time.monotonic()
        receive_wall_time_s = time.time()
        message_time_s = message.header.stamp.to_sec()
        base = self._base(message_time_s=message_time_s)
        common = dict(base)
        common.update({
            "receive_monotonic_s": receive_monotonic_s,
            "receive_wall_time_s": receive_wall_time_s,
            "frame_id": str(message.header.frame_id),
            "message_sequence": int(message.header.seq),
            "real_time_factor": float(message.real_time_factor),
        })
        metrics = list(message.sensors)
        if not metrics:
            row = dict(common)
            row.update({"sensor_name": "", "sim_update_rate_hz": "",
                        "real_update_rate_hz": "", "fps": ""})
            self._write("sensors/gazebo_performance_metrics.csv", row)
            return
        rows = []
        for metric in metrics:
            row = dict(common)
            row.update({
                "sensor_name": str(metric.name),
                "sim_update_rate_hz": float(metric.sim_update_rate),
                "real_update_rate_hz": float(metric.real_update_rate),
                "fps": float(metric.fps),
            })
            rows.append(row)
        self._write_batch("sensors/gazebo_performance_metrics.csv", rows)

    def _status_callback(self, message):
        if not self.monitoring_capture_open:
            return
        row = self._base(message.common.header.stamp.to_sec(),
                         message.common.publish_stamp.to_sec())
        row.update({
            "frame_id": message.common.header.frame_id,
            "phase": message.phase, "reference_epoch": message.common.reference_epoch,
            "accepted_frame_index": message.common.accepted_frame_index,
            "reference_frozen": message.reference_frozen,
            "gdu_count": message.gdu_count, "interface_count": message.interface_count,
            "validity": message.common.validity,
            "active_gates": "|".join(message.active_gates),
            "deadline_failure_count": message.deadline_failure_count,
            "resource_failure_count": message.resource_failure_count,
            "last_attempt_frame_index": message.last_attempt_frame_index,
            "last_attempt_published": message.last_attempt_published,
            "last_attempt_reason_codes": "|".join(
                str(value) for value in message.last_attempt_reason_codes),
            "last_attempt_processing_time_ms": (
                message.last_attempt_processing_time_ms),
            "last_attempt_input_point_count": (
                message.last_attempt_input_point_count),
            "last_attempt_time_s": message.last_attempt_stamp.to_sec()})
        self._write("deformation/detector_status.csv", row)
        runtime_row = dict(row)
        runtime_row.update({
            "message_time_s": message.last_attempt_stamp.to_sec(),
            "processing_time_ms": message.last_attempt_processing_time_ms,
            "input_point_count": message.last_attempt_input_point_count})
        runtime_key = (
            message.last_attempt_stamp.to_nsec(),
            int(message.last_attempt_frame_index),
            int(message.last_attempt_input_point_count),
            float(message.last_attempt_processing_time_ms),
            bool(message.last_attempt_published))
        with self.lock:
            new_runtime_attempt = runtime_key not in self.recorded_runtime_attempt_keys
            if message.last_attempt_input_point_count > 0 and new_runtime_attempt:
                self.recorded_runtime_attempt_keys.add(runtime_key)
        if message.last_attempt_input_point_count > 0 and new_runtime_attempt:
            self._write("runtime/per_cycle.csv", runtime_row)

    def _tile_callback(self, message):
        if not self.monitoring_capture_open:
            return
        if int(message.lod) != 2:
            return
        evidence_by_gdu = {
            stable_id_string(item.gdu_id): item
            for item in message.detection_evidence}
        common = message.common
        now = self._base(common.header.stamp.to_sec(),
                         common.publish_stamp.to_sec())
        join = lambda values: "|".join("{:.17g}".format(float(value))
                                       for value in values)
        tile_committed = True
        reference_rows = []
        for support in message.reference_support:
            support_key = (int(common.reference_epoch),
                           stable_id_string(support.gdu_id),
                           stable_id_string(support.surfel_id))
            with self.lock:
                if support_key in self.recorded_reference_keys:
                    continue
                self.recorded_reference_keys.add(support_key)
            row = dict(now)
            row.update({
                "frame_id": common.header.frame_id,
                "accepted_frame_index": common.accepted_frame_index,
                "reference_epoch": common.reference_epoch,
                "gdu_id": stable_id_string(support.gdu_id),
                "surfel_id": stable_id_string(support.surfel_id),
                "position_x_m": support.position.x,
                "position_y_m": support.position.y,
                "position_z_m": support.position.z,
                "reference_covariance_row_major": join(
                    support.reference_covariance_row_major),
                "observable_projector_row_major": join(
                    support.observable_projector_row_major),
                "geometry_probability": join(support.geometry_probability),
                "support_radius_m": support.support_radius_m,
                "exposure_count": support.exposure_count,
                "point_count": support.point_count,
                "validity": support.validity})
            reference_rows.append(row)
        if reference_rows:
            tile_committed = (
                self._write_batch("deformation/reference_support.csv",
                                  reference_rows)
                and tile_committed)
        interface_rows = []
        for interface in message.interface_motion:
            row = dict(now)
            row.update({
                "frame_id": common.header.frame_id,
                "accepted_frame_index": common.accepted_frame_index,
                "reference_epoch": common.reference_epoch,
                "interface_id": stable_id_string(interface.interface_id),
                "left_gdu_id": stable_id_string(interface.left_gdu_id),
                "right_gdu_id": stable_id_string(interface.right_gdu_id),
                "center_x_m": interface.reference_center.x,
                "center_y_m": interface.reference_center.y,
                "center_z_m": interface.reference_center.z,
                "normal_x": interface.reference_normal.x,
                "normal_y": interface.reference_normal.y,
                "normal_z": interface.reference_normal.z,
                "scale_m": interface.scale_m,
                "joint_probability": join(interface.joint_probability),
                "jump_dx_m": interface.jump_mean.x,
                "jump_dy_m": interface.jump_mean.y,
                "jump_dz_m": interface.jump_mean.z,
                "jump_covariance_row_major": join(
                    interface.jump_covariance_row_major),
                "other_mass": interface.other_mass,
                "connectivity_coherence_confounding": (
                    interface.connectivity_coherence_confounding),
                "validity": interface.validity})
            interface_rows.append(row)
        if interface_rows:
            tile_committed = (
                self._write_batch("deformation/interface_motion.csv",
                                  interface_rows)
                and tile_committed)
        evidence_rows = []
        for evidence in message.detection_evidence:
            row = dict(now)
            probability = (evidence.task_scale_probability_mean
                           if evidence.task_scale_probability_mean_identified
                           else evidence.task_scale_probability_upper)
            row.update({
                "frame_id": common.header.frame_id,
                "accepted_frame_index": common.accepted_frame_index,
                "reference_epoch": common.reference_epoch,
                "gdu_id": stable_id_string(evidence.gdu_id),
                "position_x_m": evidence.position.x,
                "position_y_m": evidence.position.y,
                "position_z_m": evidence.position.z,
                "p_u": evidence.p_u, "p_a": evidence.p_a,
                "p_c": evidence.p_c, "p_other": evidence.p_other,
                "task_scale_probability_lower": evidence.task_scale_probability_lower,
                "task_scale_probability_mean": evidence.task_scale_probability_mean,
                "task_scale_probability_upper": evidence.task_scale_probability_upper,
                "task_scale_probability_mean_identified": evidence.task_scale_probability_mean_identified,
                "task_displacement_probability": evidence.task_displacement_probability,
                "current_velocity_exceedance_probability": evidence.current_velocity_exceedance_probability,
                "continuous_valid_observation_duration_s": evidence.continuous_valid_observation_duration_s,
                "velocity_duration_requirement_met": evidence.velocity_duration_requirement_met,
                "observable_rank": evidence.observable_rank,
                "minimum_detectable_effect_m": (evidence.minimum_detectable_effect_m
                                                if evidence.minimum_detectable_effect_defined else ""),
                "any_task_direction_detectable": evidence.any_task_direction_detectable,
                "all_task_directions_detectable": evidence.all_task_directions_detectable,
                "detection_allowed": evidence.detection_allowed,
                "clearing_allowed": evidence.clearing_allowed,
                "effective_spatial_support": evidence.effective_spatial_support,
                "effective_temporal_support": evidence.effective_temporal_support,
                "detection_admission": evidence.detection_admission,
                "validity": evidence.validity,
                "task_scale_probability": probability})
            evidence_rows.append(row)
        if evidence_rows:
            tile_committed = (
                self._write_batch("deformation/detection_evidence.csv",
                                  evidence_rows)
                and tile_committed)

        with self.lock:
            # A fixed startup map/world anchor avoids using current robot truth
            # to erase SLAM drift during evaluation association.  Only this
            # coordinate conversion is needed while recording.  Model/AABB
            # association and truth-vector errors are deliberately postponed
            # until finalization, where the complete truth history is joined
            # once per detector timestamp.  This both prevents truth-driven
            # online work and avoids an O(samples x models) callback cost.
            transform = self.initial_map_from_world
            z_map_from_world = self.initial_z_map_from_world
        snapshot_id = stable_id_string(common.snapshot_id)
        algorithm_rows = []
        for index, sample in enumerate(message.physical_deformation):
            gdu_id = stable_id_string(sample.gdu_id)
            evidence = evidence_by_gdu.get(gdu_id)
            point_map = (sample.position.x, sample.position.y, sample.position.z)
            point_world = ("", "", "")
            estimate = (sample.displacement_mean.x,
                        sample.displacement_mean.y,
                        sample.displacement_mean.z)
            estimate_magnitude = float(np.linalg.norm(estimate))
            if transform is not None and z_map_from_world is not None:
                point_world_xy = inverse_transform_point(transform, point_map[:2])
                point_world = (point_world_xy[0], point_world_xy[1],
                               point_map[2] - z_map_from_world)
            probability = ""
            if evidence is not None:
                probability = (evidence.task_scale_probability_mean
                               if evidence.task_scale_probability_mean_identified
                               else evidence.task_scale_probability_upper)
            row = dict(now)
            row.update({
                "frame_id": common.header.frame_id,
                "accepted_frame_index": common.accepted_frame_index,
                "reference_epoch": common.reference_epoch,
                "snapshot_id": snapshot_id, "tile_key": message.tile_key,
                "gdu_id": gdu_id, "sample_index": index,
                "position_map_x_m": point_map[0], "position_map_y_m": point_map[1],
                "position_map_z_m": point_map[2],
                "position_world_x_m": point_world[0],
                "position_world_y_m": point_world[1],
                "position_world_z_m": point_world[2],
                "estimated_dx_m": estimate[0], "estimated_dy_m": estimate[1],
                "estimated_dz_m": estimate[2],
                "estimated_magnitude_m": estimate_magnitude,
                "estimated_vx_mps": sample.velocity_mean.x,
                "estimated_vy_mps": sample.velocity_mean.y,
                "estimated_vz_mps": sample.velocity_mean.z,
                "displacement_covariance_row_major": join(
                    sample.displacement_covariance_row_major),
                "velocity_covariance_row_major": join(
                    sample.velocity_covariance_row_major),
                "observable_projector_row_major": join(
                    sample.observable_projector_row_major),
                "active_probability_lower": sample.active_probability_lower,
                "active_probability_mean": sample.active_probability_mean,
                "active_probability_upper": sample.active_probability_upper,
                "active_probability_mean_identified": sample.active_probability_mean_identified,
                "observable_rank": evidence.observable_rank if evidence else "",
                "task_scale_probability": probability,
                "detection_allowed": evidence.detection_allowed if evidence else "",
                "detection_admission": sample.detection_admission,
                "validity": sample.validity,
                "minimum_detectable_effect_m": (sample.minimum_detectable_effect_m
                                                if sample.minimum_detectable_effect_defined else ""),
                "other_mass": sample.other_mass,
                "interpolation_error_upper_m": sample.interpolation_error_upper_m,
                "model_name": "",
                "association_status": "PENDING_FINALIZATION",
                "association_method": "NONE",
                "association_candidate_models": "",
                "association_score": "",
                "association_detail": "OFFLINE_TRUTH_JOIN_PENDING",
                "association_truth_alignment_method": "",
                "association_truth_before_time_s": "",
                "association_truth_after_time_s": "",
                "association_truth_bracket_span_s": "",
                "association_truth_zoh_bound_s": (
                    self.truth_alignment_max_zoh_residual_s),
                "association_truth_time_offset_s": "",
                "associated_truth_available": False,
                "truth_model_position_world_x_m": "",
                "truth_model_position_world_y_m": "",
                "truth_model_position_world_z_m": "",
                "truth_model_orientation_x": "",
                "truth_model_orientation_y": "",
                "truth_model_orientation_z": "",
                "truth_model_orientation_w": "",
                "truth_dx_map_m": "", "truth_dy_map_m": "",
                "truth_dz_map_m": "", "truth_magnitude_m": "",
                "vector_error_m": "", "magnitude_error_m": "",
                "direction_cosine": ""})
            algorithm_rows.append(row)
        if algorithm_rows:
            tile_committed = (
                self._write_batch("deformation/algorithm_displacement.csv",
                                  algorithm_rows)
                and tile_committed)
        with self.lock:
            if not self.closed and tile_committed:
                self.recorded_detector_frame_index = max(
                    self.recorded_detector_frame_index,
                    int(common.accepted_frame_index))
        self._maybe_commit_capture_freeze()

    def _evolution_callback(self, message):
        if not self.monitoring_capture_open:
            return
        now = self._base(message.header.stamp.to_sec())
        evolution_rows = []
        for gdu in message.gdus:
            row = dict(now)
            row.update({
                "frame_id": message.header.frame_id,
                "source_accepted_frame_index": message.source_accepted_frame_index,
                "source_snapshot_id": message.source_snapshot_id,
                "reference_epoch": message.reference_epoch, "gdu_id": gdu.gdu_id,
                "active_now": gdu.active_now,
                "retained_for_avoidance": gdu.retained_for_avoidance,
                "retained_birth_streak": getattr(
                    gdu, "retained_birth_streak", 0),
                "retained_birth_confirmations": getattr(
                    gdu, "retained_birth_confirmations", 0),
                "lifecycle_state": gdu.lifecycle_state,
                "magnitude_trend": gdu.magnitude_trend,
                "task_scale_probability": gdu.task_scale_probability,
                "displacement_rms_m": gdu.displacement_rms_m,
                "previous_displacement_rms_m": gdu.previous_displacement_rms_m,
                "displacement_change_m": gdu.displacement_change_m,
                "observation_interval_s": gdu.observation_interval_s,
                "displacement_rate_mps": gdu.displacement_rate_mps,
                "mean_dx_m": gdu.mean_displacement.x,
                "mean_dy_m": gdu.mean_displacement.y,
                "mean_dz_m": gdu.mean_displacement.z,
                "mean_vx_mps": gdu.mean_velocity.x,
                "mean_vy_mps": gdu.mean_velocity.y,
                "mean_vz_mps": gdu.mean_velocity.z,
                "support_radius_m": gdu.support_radius_m,
                "forecast_count": len(gdu.forecasts), "validity": gdu.validity})
            evolution_rows.append(row)
        evolution_committed = (not evolution_rows or self._write_batch(
            "evolution/gdu_evolution.csv", evolution_rows))
        with self.lock:
            if not self.closed and evolution_committed:
                self.recorded_evolution_frame_index = max(
                    self.recorded_evolution_frame_index,
                    int(message.source_accepted_frame_index))
        self._maybe_commit_capture_freeze()

    def _avoidance_callback(self, message):
        if not self.monitoring_capture_open:
            return
        try:
            bundle_maps, bundle_metadata = avoidance_snapshot_layers(message)
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            rospy.logerr_throttle(
                2.0, "invalid avoidance snapshot bundle: %s", error)
            return
        grid = message.current_overlay
        row = self._base(message.header.stamp.to_sec())
        row.update({
            "frame_id": message.current_overlay.header.frame_id,
            "source_accepted_frame_index": message.source_accepted_frame_index,
            "source_snapshot_id": message.source_snapshot_id,
            "map_revision": message.map_revision, "policy": message.policy,
            "hard_avoidance_radius_m": message.hard_avoidance_radius_m,
            "warning_radius_m": message.warning_radius_m,
            "elevated_source_min_z_m": message.elevated_source_min_z_m,
            "elevated_hard_projection_radius_m": (
                message.elevated_hard_projection_radius_m),
            "elevated_warning_projection_radius_m": (
                message.elevated_warning_projection_radius_m),
            "active_source_count": len(message.active_source_ids),
            "active_source_ids": "|".join(message.active_source_ids),
            "ordinary_projection_source_ids": "|".join(
                message.ordinary_projection_source_ids),
            "elevated_projection_source_ids": "|".join(
                message.elevated_projection_source_ids),
            "forecast_horizons_s": "|".join(str(value) for value in message.forecast_horizons_s),
            "current_occupied_cell_count": sum(value >= 50 for value in grid.data),
            "current_unknown_cell_count": sum(value < 0 for value in grid.data),
            "all_current_sources_within_map": message.all_current_sources_within_map,
            "all_forecast_sources_within_map": message.all_forecast_sources_within_map})
        if not self._write("maps/avoidance_frames.csv", row):
            return
        with self.lock:
            replace_avoidance_snapshot_layers(
                self.latest_maps, self.latest_map_metadata,
                bundle_maps, bundle_metadata)
            self.recorded_avoidance_frame_index = max(
                self.recorded_avoidance_frame_index,
                int(message.source_accepted_frame_index))
        self._maybe_commit_capture_freeze()

    def _map_callback(self, name, message):
        if not self.monitoring_capture_open:
            return
        with self.lock:
            self.latest_maps[name] = message
            self.latest_map_metadata[name] = {
                "stamp_s": float(message.header.stamp.to_sec()),
                "frame_id": str(message.header.frame_id),
                "source_accepted_frame_index": "",
                "source_snapshot_id": "",
                "map_revision": "",
                "forecast_horizon_s": "",
            }

    def _cloud_callback(self, name, message):
        if not self.monitoring_capture_open:
            return
        with self.lock:
            self.latest_clouds[name] = message

    def _safe_state(self, state):
        return "".join(character if character.isalnum() else "_"
                       for character in state)[:48]

    @staticmethod
    def _atomic_savez(path, **arrays):
        temporary = path + ".tmp"
        try:
            with open(temporary, "wb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    def _save_grid(self, prefix, name, message, metadata=None):
        data = np.asarray(message.data, dtype=np.int16).reshape(
            (message.info.height, message.info.width))
        path = os.path.join(self.output, "snapshots", "maps",
                            "{}_{}.npz".format(prefix, name))
        metadata = metadata or {}
        self._atomic_savez(
            path, data=data, resolution_m=float(message.info.resolution),
            origin=np.asarray([message.info.origin.position.x,
                               message.info.origin.position.y,
                               message.info.origin.position.z], dtype=float),
            frame_id=np.asarray(message.header.frame_id),
            stamp_s=float(message.header.stamp.to_sec()),
            source_accepted_frame_index=np.asarray(
                metadata.get("source_accepted_frame_index", "")),
            source_snapshot_id=np.asarray(
                metadata.get("source_snapshot_id", "")),
            source_stamp_s=np.asarray(metadata.get("source_stamp_s", "")),
            map_revision=np.asarray(metadata.get("map_revision", "")),
            forecast_horizon_s=np.asarray(
                metadata.get("forecast_horizon_s", "")))
        self.counts["map_snapshots"] += 1
        return path

    def _save_cloud(self, prefix, name, message):
        available = {field.name for field in message.fields}
        fields = [field for field in ("x", "y", "z", "intensity")
                  if field in available]
        if not all(field in fields for field in ("x", "y", "z")):
            return None
        points = np.asarray(list(point_cloud2.read_points(
            message, field_names=fields, skip_nans=True)), dtype=np.float32)
        if points.size == 0:
            return None
        if points.ndim == 1 and points.size:
            points = points.reshape((1, -1))
        if len(points) > self.max_snapshot_points:
            stride = int(math.ceil(len(points) / float(self.max_snapshot_points)))
            points = points[::stride]
        path = os.path.join(self.output, "snapshots", "clouds",
                            "{}_{}.npz".format(prefix, name))
        self._atomic_savez(path, points=points,
                           fields=np.asarray(fields),
                           frame_id=np.asarray(message.header.frame_id),
                           stamp_s=float(message.header.stamp.to_sec()))
        self.counts["cloud_snapshots"] += 1
        return path

    def _snapshot_phase(self, state):
        with self.lock:
            if state in self.snapshotted_states:
                cached = self.snapshot_metadata_by_state.get(state, {})
                return copy.deepcopy(cached.get("maps", {})), True
            if state in self.snapshot_in_progress_states:
                return {}, False
            maps = dict(self.latest_maps)
            map_metadata = {
                name: dict(metadata)
                for name, metadata in self.latest_map_metadata.items()}
            clouds = dict(self.latest_clouds)
            for name, message in maps.items():
                map_metadata.setdefault(name, {})
                map_metadata[name].setdefault(
                    "stamp_s", float(message.header.stamp.to_sec()))
                map_metadata[name].setdefault(
                    "frame_id", str(message.header.frame_id))
            cloud_metadata = {
                name: {
                    "stamp_s": float(message.header.stamp.to_sec()),
                    "frame_id": str(message.header.frame_id),
                }
                for name, message in clouds.items()}
            is_checkpoint = str(state).startswith("CHECKPOINT_")
            missing_maps = [name for name in CHECKPOINT_REQUIRED_MAP_NAMES
                            if name not in maps] if is_checkpoint else []
            missing_clouds = [name for name in CHECKPOINT_REQUIRED_CLOUD_NAMES
                              if name not in clouds] if is_checkpoint else []
            if missing_maps or missing_clouds:
                rospy.logwarn_throttle(
                    2.0,
                    "checkpoint %s awaits current map/cloud: maps=%s clouds=%s",
                    state, missing_maps, missing_clouds)
                return {}, False
            self.snapshot_in_progress_states.add(state)
            prefix = "{:02d}_{}".format(
                                        len(self.snapshotted_states)
                                        + len(self.snapshot_in_progress_states),
                                        self._safe_state(state))
        committed = bool(maps)
        saved_maps = set()
        saved_clouds = set()
        saved_paths = []
        try:
            for name, message in maps.items():
                path = self._save_grid(prefix, name, message,
                                       map_metadata.get(name, {}))
                if path:
                    saved_maps.add(name)
                    saved_paths.append(path)
            for name, message in clouds.items():
                path = self._save_cloud(prefix, name, message)
                if path:
                    saved_clouds.add(name)
                    saved_paths.append(path)
            if is_checkpoint:
                committed = (
                    set(CHECKPOINT_REQUIRED_MAP_NAMES).issubset(saved_maps)
                    and set(CHECKPOINT_REQUIRED_CLOUD_NAMES).issubset(
                        saved_clouds))
        except (OSError, ValueError, TypeError) as error:
            committed = False
            rospy.logerr("could not save phase snapshot %s: %s", state, error)
        if not committed:
            for path in saved_paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            with self.lock:
                self.counts["map_snapshots"] -= len(saved_maps)
                self.counts["cloud_snapshots"] -= len(saved_clouds)
        with self.lock:
            self.snapshot_in_progress_states.discard(state)
            if committed:
                self.snapshotted_states.add(state)
                self.snapshot_metadata_by_state[state] = {
                    "maps": copy.deepcopy(map_metadata),
                    "clouds": copy.deepcopy(cloud_metadata),
                }
        return map_metadata, committed

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _write_provenance(self):
        parameter_names = rospy.get_param_names()
        prefixes = (
            "/icra2027", "/laserMapping", "/common", "/preprocess",
            "/mapping", "/publish", "/dual_livox_merge", "/deform",
            "/move_base", "/stationary_monitor_cloud_adapter",
            "/lio_twist_odometry", "/mission_octomap",
            "/mission_map_proxy", "/icra2027_closed_loop_mission",
            rospy.get_name())
        exact_names = {
            "/use_sim_time", "/robot_description", "/filter_size_surf",
            "/filter_size_map", "/point_filter_num",
            "/feature_extract_enable", "/max_iteration",
            "/cube_side_length", "/runtime_pos_log_enable",
            "/runtime_pcd_save_enable", "/runtime_scan_publish_enable",
            "/runtime_dense_publish_enable",
            "/runtime_scan_bodyframe_publish_enable",
        }
        parameters = {}
        for name in parameter_names:
            if name in exact_names or name.startswith(prefixes):
                try:
                    parameters[name] = rospy.get_param(name)
                except KeyError:
                    pass
        with open(os.path.join(self.output, "ros_parameters.json"), "w",
                  encoding="utf-8") as stream:
            json.dump(parameters, stream, indent=2, sort_keys=True,
                      ensure_ascii=False)
            stream.write("\n")

    def _generate_track_and_runtime_analysis(self):
        """Generate non-visual research tables without requiring matplotlib."""
        analysis = os.path.join(self.output, "analysis")

        def read_rows(relative):
            path = os.path.join(self.output, relative)
            if not os.path.exists(path):
                return []
            with open(path, newline="") as stream:
                return list(csv.DictReader(stream))

        estimates = read_rows("deformation/algorithm_displacement.csv")
        evidences = read_rows("deformation/detection_evidence.csv")
        evolutions = read_rows("evolution/gdu_evolution.csv")
        truth = read_rows("truth/m01_m06_motion_truth.csv")
        startup_anchor = getattr(self, "initial_map_from_world", None)
        map_from_world_yaw = (
            startup_anchor[2]
            if isinstance(startup_anchor, (list, tuple))
            and len(startup_anchor) >= 3 else None)
        estimates, final_truth_realign = realign_algorithm_truth_rows(
            estimates, truth, getattr(self, "tolerances", {}),
            getattr(self, "truth_association_aabb_margin_m", 0.25),
            map_from_world_yaw,
            getattr(self, "truth_alignment_max_zoh_residual_s", 0.25))
        # The raw detector table is an immutable observation record.  Truth is
        # joined once in memory for evaluation products below; rewriting this
        # very large CSV added no information and delayed every batch trial.
        cutoff_frame = getattr(self, "capture_cutoff_frame_index", None)
        cutoff_elapsed = getattr(
            self, "capture_cutoff_experiment_elapsed_s", None)
        if cutoff_frame is not None:
            def within_frame_cutoff(row):
                value = _optional_float(
                    row, ("accepted_frame_index",
                          "source_accepted_frame_index"))
                return value is None or value <= cutoff_frame
            estimates = [row for row in estimates if within_frame_cutoff(row)]
            evidences = [row for row in evidences if within_frame_cutoff(row)]
            evolutions = [row for row in evolutions if within_frame_cutoff(row)]
        if cutoff_elapsed is not None:
            truth = [
                row for row in truth
                if (_optional_float(row, ("experiment_elapsed_s",)) is None
                    or _optional_float(row, ("experiment_elapsed_s",))
                    <= cutoff_elapsed + 1.0e-9)]
        models = ("m01", "m02", "m03", "m04", "m05", "m06")
        primary_models = tuple(getattr(
            self, "primary_metric_models",
            ("m01", "m02", "m03", "m05", "m06")))
        displacement_evaluation_floor_m = 0.005
        inventory, activation = build_track_inventory(
            estimates, evidences, evolutions, truth, models,
            displacement_evaluation_floor_m, primary_models)

        truth_by_model = defaultdict(list)
        for row in truth:
            model = str(row.get("model_name", ""))
            elapsed = _optional_float(row, ("experiment_elapsed_s",))
            # Eq. (confirmation-displacement) in the reference paper is a
            # surface maximum.  Translation magnitude is only the backwards-
            # compatible fallback for legacy truth rows without surface data.
            magnitude = _optional_float(
                row, ("surface_max_displacement_m",
                      "translation_magnitude_m"))
            if model and elapsed is not None and magnitude is not None:
                truth_by_model[model].append((elapsed, magnitude))
        for values in truth_by_model.values():
            values.sort()

        def truth_magnitude_at(model, elapsed):
            values = truth_by_model.get(model, [])
            if not values or elapsed in (None, ""):
                return ""
            times = np.asarray([item[0] for item in values], dtype=float)
            magnitudes = np.asarray([item[1] for item in values], dtype=float)
            return float(np.interp(float(elapsed), times, magnitudes))

        for row in inventory:
            model = row["model_name"]
            model_activation = activation.get(model, {})
            row["truth_active"] = bool(model_activation.get("active", False))
            row["truth_activation_experiment_elapsed_s"] = (
                model_activation.get("experiment_elapsed_s")
                if model else "")
            if row["evaluation_category"] == "TRUE_POSITIVE":
                confirmation = row["confirmation_experiment_elapsed_s"]
                activation_time = model_activation.get("experiment_elapsed_s")
                row["confirmation_latency_s"] = (
                    float(confirmation) - float(activation_time)
                    if confirmation != "" and activation_time is not None else "")
                row["truth_displacement_at_confirmation_m"] = (
                    truth_magnitude_at(model, confirmation))
            else:
                row["confirmation_latency_s"] = ""
                row["truth_displacement_at_confirmation_m"] = ""

        track_fields = [
            "gdu_id", "association_status", "model_name", "candidate_models",
            "candidate_model_matched_frame_counts", "best_candidate_model",
            "included_in_primary_metrics",
            "evaluation_category", "confirmed", "estimate_sample_count",
            "evidence_sample_count", "evolution_sample_count",
            "matched_sample_count", "ambiguous_sample_count",
            "unmatched_sample_count", "independent_estimate_frame_count",
            "matched_frame_count", "total_clean_matched_frame_count",
            "ambiguous_frame_count", "unmatched_frame_count",
            "multi_model_frame_count",
            "estimate_rows_without_accepted_frame_count",
            "independent_frame_contract_satisfied", "matched_frame_purity",
            "minimum_matched_frame_support", "minimum_matched_frame_purity",
            "first_accepted_frame_index",
            "last_accepted_frame_index", "first_receive_time_s",
            "last_receive_time_s", "first_observation_experiment_elapsed_s",
            "candidate_time_available", "candidate_experiment_elapsed_s",
            "confirmation_time_available", "confirmation_time_source",
            "confirmation_experiment_elapsed_s", "confirmation_receive_time_s",
            "first_retained_observation_experiment_elapsed_s",
            "maximum_task_scale_probability", "truth_active",
            "truth_activation_experiment_elapsed_s", "confirmation_latency_s",
            "truth_displacement_at_confirmation_m"]
        with open(os.path.join(analysis, "detection_track_inventory.csv"),
                  "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=track_fields,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(inventory)

        model_fields = [
            "model_name", "included_in_primary_metrics", "truth_active",
            "truth_activation_experiment_elapsed_s",
            "truth_motion_onset_source", "maximum_truth_displacement_m",
            "associated_observation_track_count", "confirmed_track_count",
            "true_positive_track_count", "true_positive_track_ids",
            "detected", "first_observation_experiment_elapsed_s",
            "candidate_time_available", "candidate_latency_s",
            "confirmation_experiment_elapsed_s", "confirmation_latency_s",
            "truth_displacement_at_confirmation_m"]
        model_rows = []
        for model in models:
            candidates = [row for row in inventory
                          if row["association_status"] == "MATCHED"
                          and row["model_name"] == model]
            confirmed = [row for row in candidates if row["confirmed"]]
            true_positive_tracks = [row for row in candidates
                                    if row["evaluation_category"]
                                    == "TRUE_POSITIVE"]
            true_positive_tracks.sort(key=lambda row: (
                float(row["confirmation_experiment_elapsed_s"])
                if row["confirmation_experiment_elapsed_s"] != ""
                else float("inf"), row["gdu_id"]))
            first_true_positive = (true_positive_tracks[0]
                                   if true_positive_tracks else None)
            observation_times = [
                float(row["first_observation_experiment_elapsed_s"])
                for row in candidates
                if row["first_observation_experiment_elapsed_s"] != ""]
            first_observation = (min(observation_times)
                                 if observation_times else None)
            activation_time = activation[model]["experiment_elapsed_s"]
            model_rows.append({
                "model_name": model,
                "included_in_primary_metrics": model in primary_models,
                "truth_active": activation[model]["active"],
                "truth_activation_experiment_elapsed_s": (
                    activation_time if activation_time is not None else ""),
                "truth_motion_onset_source": activation[model]["onset_source"],
                "maximum_truth_displacement_m": (
                    activation[model]["maximum_displacement_m"]
                    if activation[model]["maximum_displacement_m"] is not None
                    else ""),
                "associated_observation_track_count": len(candidates),
                "confirmed_track_count": len(confirmed),
                "true_positive_track_count": len(true_positive_tracks),
                "true_positive_track_ids": "|".join(
                    row["gdu_id"] for row in true_positive_tracks),
                "detected": bool(true_positive_tracks),
                "first_observation_experiment_elapsed_s": (
                    first_observation if first_observation is not None else ""),
                "candidate_time_available": False,
                "candidate_latency_s": "",
                "confirmation_experiment_elapsed_s": (
                    first_true_positive["confirmation_experiment_elapsed_s"]
                    if first_true_positive else ""),
                "confirmation_latency_s": (
                    first_true_positive["confirmation_latency_s"]
                    if first_true_positive else ""),
                "truth_displacement_at_confirmation_m": (
                    first_true_positive["truth_displacement_at_confirmation_m"]
                    if first_true_positive else ""),
            })
        with open(os.path.join(analysis, "model_detection_event_metrics.csv"),
                  "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=model_fields)
            writer.writeheader()
            writer.writerows(model_rows)

        active_models = [row for row in model_rows
                         if row["truth_active"]
                         and row["included_in_primary_metrics"]]
        recall_contract = registered_primary_model_recall(
            model_rows, primary_models)
        true_positives = [row for row in inventory
                          if row["evaluation_category"] == "TRUE_POSITIVE"]
        false_positives = [row for row in inventory if row["evaluation_category"] in (
            "FALSE_POSITIVE_UNMATCHED", "FALSE_POSITIVE_STATIC_MODEL",
            "FALSE_POSITIVE_PRE_ACTIVATION", "FALSE_POSITIVE_AMBIGUOUS",
            "FALSE_POSITIVE_EXCLUDED_MODEL",
            "FALSE_POSITIVE_CONFIRMATION_TIME_UNAVAILABLE")]
        ambiguous = [row for row in inventory
                     if row["evaluation_category"]
                     == "FALSE_POSITIVE_AMBIGUOUS"]
        denominator = len(true_positives) + len(false_positives)
        all_truth_elapsed = [_optional_float(
            row, ("experiment_elapsed_s",)) for row in truth]
        all_truth_elapsed = [value for value in all_truth_elapsed
                             if value is not None]
        activation_times = [value["experiment_elapsed_s"]
                            for value in activation.values()
                            if value["experiment_elapsed_s"] is not None]
        baseline_start = min(all_truth_elapsed) if all_truth_elapsed else None
        baseline_end = min(activation_times) if activation_times else None
        baseline_duration = (
            max(0.0, baseline_end - baseline_start)
            if baseline_start is not None and baseline_end is not None else None)
        effective_monitoring_start = (
            min(all_truth_elapsed) if all_truth_elapsed else None)
        effective_monitoring_end = (
            max(all_truth_elapsed) if all_truth_elapsed else None)
        effective_monitoring_duration = (
            max(0.0, effective_monitoring_end - effective_monitoring_start)
            if effective_monitoring_start is not None
            and effective_monitoring_end is not None else None)
        baseline_false = [row for row in inventory
                          if row["confirmed"]
                          and row["confirmation_experiment_elapsed_s"] != ""
                          and baseline_start is not None and baseline_end is not None
                          and baseline_start <= float(
                              row["confirmation_experiment_elapsed_s"])
                          < baseline_end]
        alignment_method_counts = defaultdict(int)
        for row in estimates:
            method = row.get("association_truth_alignment_method", "")
            if method:
                alignment_method_counts[method] += 1
        alignment_offsets = [_optional_float(
            row, ("association_truth_time_offset_s",)) for row in estimates]
        alignment_offsets = [value for value in alignment_offsets
                             if value is not None]
        alignment_spans = [_optional_float(
            row, ("association_truth_bracket_span_s",)) for row in estimates]
        alignment_spans = [value for value in alignment_spans
                           if value is not None]
        truth_alignment_summary = {
            "target_timestamp": "detector message_time_s",
            "final_full_history_realign": final_truth_realign,
            "experiment_elapsed_s_at_message_recomputed_when_aligned": True,
            "detector_estimate_probability_id_or_timestamp_modified": False,
            "method_counts": dict(sorted(alignment_method_counts.items())),
            "aligned_truth_sample_count": sum(
                _csv_bool(row.get("associated_truth_available", ""))
                for row in estimates),
            "signed_time_offset_s_mean": (
                float(np.mean(alignment_offsets))
                if alignment_offsets else None),
            "absolute_time_offset_s_p95": (
                float(np.percentile(np.abs(alignment_offsets), 95))
                if alignment_offsets else None),
            "absolute_time_offset_s_maximum": (
                max((abs(value) for value in alignment_offsets), default=None)),
            "interpolation_bracket_span_s_p95": (
                float(np.percentile(alignment_spans, 95))
                if alignment_spans else None),
            "maximum_zoh_residual_s": getattr(
                self, "truth_alignment_max_zoh_residual_s", None),
            "truth_feedback_to_detector_or_maps": False,
        }
        run_summary = {
            "association_is_evaluation_only": True,
            "association_feedback_to_detector_or_maps": False,
            "association_statuses": ["MATCHED", "UNMATCHED", "AMBIGUOUS"],
            "displacement_error_evaluation_floor_m": (
                displacement_evaluation_floor_m),
            "primary_metric_models": list(primary_models),
            "excluded_evaluation_models": sorted(set(models)
                                                 - set(primary_models)),
            "truth_motion_onset_definition": (
                "experiment_elapsed_s - model_motion_elapsed_s for the first "
                "positive per-model motion elapsed sample; nonzero truth is "
                "only a legacy fallback and 5 mm is never used as onset"),
            "raw_algorithm_sample_count": len(estimates),
            "raw_unmatched_algorithm_sample_count": sum(
                row.get("association_status", "") == "UNMATCHED"
                for row in estimates),
            "raw_ambiguous_algorithm_sample_count": sum(
                row.get("association_status", "") == "AMBIGUOUS"
                for row in estimates),
            "truth_time_alignment": truth_alignment_summary,
            "track_count": len(inventory),
            "confirmed_track_count": sum(row["confirmed"] for row in inventory),
            "confirmation_time_unavailable_track_count": sum(
                row["confirmed"] and not row["confirmation_time_available"]
                for row in inventory),
            "confirmation_time_unavailable_confirmed_tracks_counted_as_false": sum(
                row["evaluation_category"]
                == "FALSE_POSITIVE_CONFIRMATION_TIME_UNAVAILABLE"
                for row in inventory),
            "confirmation_event_chain_complete": not any(
                row["evaluation_category"]
                == "FALSE_POSITIVE_CONFIRMATION_TIME_UNAVAILABLE"
                for row in inventory),
            "track_association_minimum_matched_independent_frames": (
                TRACK_ASSOCIATION_MINIMUM_MATCHED_FRAMES),
            "track_association_minimum_matched_frame_purity": (
                TRACK_ASSOCIATION_MINIMUM_MATCHED_FRAME_PURITY),
            "tracks_with_unidentifiable_estimate_frame_rows": sum(
                row["estimate_rows_without_accepted_frame_count"] > 0
                for row in inventory),
            "true_positive_track_count": len(true_positives),
            "false_positive_track_count_per_run": len(false_positives),
            "ambiguous_confirmed_track_count_counted_as_false": len(ambiguous),
            "active_truth_model_count": len(active_models),
            "detected_active_truth_model_count": sum(
                _csv_bool(row["detected"]) for row in active_models),
            "object_recall_denominator_policy": (
                "fixed_pre_registered_primary_models"),
            "object_recall_denominator_models": recall_contract[
                "denominator_models"],
            "object_recall_denominator_model_count": recall_contract[
                "denominator_model_count"],
            "detected_registered_primary_models": recall_contract[
                "detected_models"],
            "detected_registered_primary_model_count": recall_contract[
                "detected_model_count"],
            "false_negative_registered_primary_models": recall_contract[
                "false_negative_models"],
            "false_negative_registered_primary_model_count": recall_contract[
                "false_negative_model_count"],
            "object_recall": recall_contract["object_recall"],
            "unique_confirmed_track_precision": (
                len(true_positives) / float(denominator)
                if denominator else None),
            "effective_monitoring_start_experiment_elapsed_s": (
                effective_monitoring_start),
            "effective_monitoring_end_experiment_elapsed_s": (
                effective_monitoring_end),
            "effective_monitoring_duration_s": effective_monitoring_duration,
            "false_alarm_rate_tracks_per_min": (
                len(false_positives) * 60.0 / effective_monitoring_duration
                if effective_monitoring_duration
                and effective_monitoring_duration > 0.0 else None),
            "zero_displacement_baseline_start_experiment_elapsed_s": baseline_start,
            "zero_displacement_baseline_end_experiment_elapsed_s": baseline_end,
            "zero_displacement_baseline_duration_s": baseline_duration,
            "zero_displacement_baseline_false_confirmations": len(baseline_false),
            "zero_displacement_false_alarm_rate_per_min": (
                len(baseline_false) * 60.0 / baseline_duration
                if baseline_duration and baseline_duration > 0.0 else None),
            "track_definition": (
                "one stable gdu_id across raw detector evidence, deformation "
                "samples and evolution frames"),
            "track_association_independent_frame_definition": (
                "all spatial estimate samples sharing one integer "
                "accepted_frame_index and gdu_id form one temporal frame; "
                "a clean model frame has at least one MATCHED sample for "
                "exactly one model and no AMBIGUOUS sample; UNMATCHED samples "
                "in that frame do not create extra votes"),
            "track_matched_frame_purity_definition": (
                "matched frames for the sole candidate model divided by all "
                "unique accepted frames containing an estimate for the GDU, "
                "including unmatched and ambiguous frames"),
            "track_match_acceptance_rule": (
                "MATCHED iff exactly one candidate model is observed, every "
                "estimate row has an identifiable accepted frame, at least 3 "
                "independent frames cleanly match that model, and matched-frame "
                "purity is at least 0.80; otherwise model evidence is AMBIGUOUS, "
                "while tracks without model evidence are UNMATCHED"),
            "first_observation_time_definition": (
                "first recorded row for that gdu_id; this is not a candidate"),
            "candidate_time_available": False,
            "candidate_time_unavailable_reason": (
                "v3 exposes no distinct candidate lifecycle/flag; raw evidence "
                "is retained for a future pre-registered candidate protocol"),
            "confirmation_time_definition": (
                "first lifecycle NEWLY_DETECTED row; later PERSISTENT or "
                "CLEAR_CANDIDATE rows prove prior confirmation but do not "
                "reconstruct its timestamp; a confirmed track without this "
                "event is FALSE_POSITIVE_CONFIRMATION_TIME_UNAVAILABLE and "
                "cannot be a true positive"),
            "precision_denominator_note": (
                "every unique confirmed track is counted once; unmatched, "
                "ambiguous, excluded-model, static-model and pre-activation "
                "confirmations, plus confirmations with unavailable confirmation "
                "time, are false tracks; excluded models never improve registered "
                "structural-target recall"),
        }
        with open(os.path.join(analysis, "detection_run_summary.json"), "w",
                  encoding="utf-8") as stream:
            json.dump(run_summary, stream, indent=2, sort_keys=True,
                      ensure_ascii=False)
            stream.write("\n")
        with open(os.path.join(analysis, "truth_alignment_summary.json"), "w",
                  encoding="utf-8") as stream:
            json.dump(truth_alignment_summary, stream, indent=2,
                      sort_keys=True, ensure_ascii=False)
            stream.write("\n")

        runtime_rows = read_rows("runtime/per_cycle.csv")
        processing_ms = [_optional_float(row, ("processing_time_ms",))
                         for row in runtime_rows]
        processing_ms = [value for value in processing_ms if value is not None]
        input_points = [_optional_float(row, ("input_point_count",))
                        for row in runtime_rows]
        input_points = [value for value in input_points if value is not None]
        def numeric_summary(values):
            return {
                "sample_count": len(values),
                "mean": float(np.mean(values)) if values else None,
                "median": float(np.median(values)) if values else None,
                "p95": float(np.percentile(values, 95)) if values else None,
                "maximum": max(values) if values else None,
            }
        with open(os.path.join(analysis, "detector_runtime_summary.json"), "w",
                  encoding="utf-8") as stream:
            json.dump({
                "cycle_count": len(runtime_rows),
                "published_cycle_count": sum(
                    _csv_bool(row.get("last_attempt_published", ""))
                    for row in runtime_rows),
                "processing_time_ms": numeric_summary(processing_ms),
                "input_point_count": numeric_summary(input_points),
                "timing_boundary": (
                    "wall-clock synchronized callback work: BuildBundle plus "
                    "V1Pipeline::AcceptFrame and, for a published success, "
                    "construction of deformation/interface/tile/status "
                    "products plus the synchronous roscpp handoff of "
                    "deformation/interface/tile messages"),
                "excluded_from_timing": (
                    "pre-synchronization waiting, transport/network delivery, "
                    "subscriber processing and the final status-message handoff"),
                "stage_breakdown_available": False,
            }, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")

        gazebo_performance_rows = read_rows(
            "sensors/gazebo_performance_metrics.csv")
        with open(os.path.join(
                analysis, "gazebo_performance_summary.json"), "w",
                encoding="utf-8") as stream:
            json.dump(summarize_gazebo_performance(gazebo_performance_rows),
                      stream, indent=2, sort_keys=True,
                      ensure_ascii=False)
            stream.write("\n")

        def write_tum(filename, rows, position_fields, quaternion_fields):
            samples = {}
            for row in rows:
                stamp = _optional_float(row, ("message_time_s", "ros_time_s"))
                values = [_optional_float(row, (field,))
                          for field in position_fields + quaternion_fields]
                if stamp is None or any(value is None for value in values):
                    continue
                samples[stamp] = values
            with open(os.path.join(analysis, filename), "w",
                      encoding="utf-8") as stream:
                for stamp in sorted(samples):
                    stream.write("{:.9f} {}\n".format(
                        stamp, " ".join("{:.17g}".format(value)
                                        for value in samples[stamp])))
            return len(samples)

        slam_rows = read_rows("trajectories/slam_odometry.csv")
        robot_truth_rows = read_rows("trajectories/gazebo_robot_truth.csv")
        slam_tum_count = write_tum(
            "slam_trajectory_tum.txt", slam_rows,
            ("x_m", "y_m", "z_m"), ("qx", "qy", "qz", "qw"))
        truth_tum_count = write_tum(
            "gazebo_body_truth_tum.txt", robot_truth_rows,
            ("body_x_m", "body_y_m", "body_z_m"),
            ("body_qx", "body_qy", "body_qz", "body_qw"))
        with open(os.path.join(analysis, "se3_trajectory_export.json"), "w",
                  encoding="utf-8") as stream:
            json.dump({
                "format": "TUM: timestamp tx ty tz qx qy qz qw",
                "slam_file": "slam_trajectory_tum.txt",
                "slam_frame": "FAST-LIO odometry frame as published",
                "slam_sample_count": slam_tum_count,
                "truth_file": "gazebo_body_truth_tum.txt",
                "truth_frame": "Gazebo world, calibrated FAST-LIO body pose",
                "truth_sample_count": truth_tum_count,
                "body_pose_composition": (
                    "full SE(3): Gazebo model quaternion rotates the calibrated "
                    "xyz offset and composes the calibration yaw quaternion"),
                "ate_rpe_computed": False,
                "reason_not_computed": (
                    "the recorder exports timestamped trajectories; evo alignment "
                    "and ATE/RPE are an explicit downstream evaluation step"),
            }, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")

        diagnostics = read_rows("sensors/dual_lidar_merge_diagnostics.csv")
        latest = diagnostics[-1] if diagnostics else {}
        dual_success = dual_lidar_success_contract(
            latest, getattr(self, "dual_lidar_success_thresholds", None))
        dual_success["required_for_this_run"] = bool(getattr(
            self, "require_dual_lidar_success_contract", False))
        dual_success["fresh_diagnostic_after_finalization_request"] = {
            "request_sequence": getattr(
                self, "dual_lidar_request_sequence", None),
            "final_sequence": getattr(self, "dual_lidar_final_sequence", None),
            "request_monotonic_s": getattr(
                self, "dual_lidar_request_monotonic_s", None),
            "request_wall_time_s": getattr(
                self, "dual_lidar_request_wall_time_s", None),
            "final_receive_monotonic_s": getattr(
                self, "dual_lidar_final_receive_monotonic_s", None),
            "final_receive_wall_time_s": getattr(
                self, "dual_lidar_final_receive_wall_time_s", None),
            "satisfied": (
                not bool(getattr(
                    self, "require_dual_lidar_success_contract", False))
                or (getattr(self, "dual_lidar_request_sequence", None)
                    is not None
                    and getattr(self, "dual_lidar_final_sequence", None)
                    is not None
                    and self.dual_lidar_final_sequence
                    > self.dual_lidar_request_sequence
                    and getattr(self, "dual_lidar_request_monotonic_s", None)
                    is not None
                    and getattr(
                        self, "dual_lidar_final_receive_monotonic_s", None)
                    is not None
                    and self.dual_lidar_final_receive_monotonic_s
                    > self.dual_lidar_request_monotonic_s)),
        }
        def latest_number(key):
            return _optional_float(latest, (key,))
        with open(os.path.join(analysis, "dual_lidar_timing_summary.json"), "w",
                  encoding="utf-8") as stream:
            json.dump({
                "formal_success_contract": dual_success,
                "diagnostic_sample_count": len(diagnostics),
                "point_time_modes_observed": sorted({
                    row.get("point_time_mode", "") for row in diagnostics
                    if row.get("point_time_mode", "")}),
                "secondary_enabled_latest": (
                    _csv_bool(latest.get("secondary_enabled", ""))
                    if latest else None),
                "zero_no_return_compaction_enabled_latest": (
                    _csv_bool(latest.get(
                        "zero_no_return_compaction_enabled", ""))
                    if latest else None),
                "raw_input_point_count_definition": (
                    "front/rear CustomMsg slots after configured sensor "
                    "stride, before formal dual-merge compaction"),
                "compacted_output_point_count_definition": (
                    "finite non-(0,0,0) returns published on /livox/lidar; "
                    "no range crop or voxelization is applied by merger"),
                "configured_sensor_update_rate_hz": latest_number(
                    "configured_sensor_update_rate_hz"),
                "configured_min_range_m": latest_number(
                    "configured_min_range_m"),
                "configured_max_range_m": latest_number(
                    "configured_max_range_m"),
                "configured_ray_samples_per_sensor": latest_number(
                    "configured_ray_samples_per_sensor"),
                "primary_scheduler_catch_up_suppressed_count": latest_number(
                    "primary_scheduler_catch_up_suppressed_count"),
                "primary_downsample_configured": latest_number(
                    "primary_downsample"),
                "secondary_downsample_configured": latest_number(
                    "secondary_downsample"),
                "configured_secondary_to_primary_point_ratio": latest_number(
                    "configured_secondary_to_primary_point_ratio"),
                "overlap_voxel_size_m": latest_number(
                    "overlap_voxel_size_m"),
                "overlap_evaluation_pair_stride": latest_number(
                    "overlap_evaluation_pair_stride"),
                "overlap_definition": (
                    "unique occupied voxels in the primary LiDAR frame; "
                    "rho_primary=intersection/primary, "
                    "rho_secondary=intersection/secondary, "
                    "J=intersection/union"),
                "overlap_input_stage": (
                    "accepted synchronized per-sensor plugin-downsampled "
                    "returns before FAST-LIO deskew; secondary is rigidly "
                    "transformed into the primary frame"),
                "overlap_no_return_policy": (
                    "exact Gazebo (0,0,0) sentinels are excluded"),
                "primary_received_latest": latest_number("primary_received"),
                "secondary_received_latest": latest_number("secondary_received"),
                "pairs_accepted_latest": latest_number("pairs_accepted"),
                "pairs_rejected_latest": latest_number("pairs_rejected"),
                "primary_unpaired_dropped_latest": latest_number(
                    "primary_unpaired_dropped"),
                "secondary_unpaired_dropped_latest": latest_number(
                    "secondary_unpaired_dropped"),
                "primary_queue_dropped_latest": latest_number(
                    "primary_queue_dropped"),
                "secondary_queue_dropped_latest": latest_number(
                    "secondary_queue_dropped"),
                "maximum_sync_delta_ms": (
                    latest_number("maximum_sync_delta_ns") / 1.0e6
                    if latest_number("maximum_sync_delta_ns") is not None
                    else None),
                "mean_sync_delta_ms": (
                    latest_number("mean_sync_delta_ns") / 1.0e6
                    if latest_number("mean_sync_delta_ns") is not None else None),
                "first_accepted_measurement_time_s": (
                    latest_number("first_accepted_measurement_time_ns") / 1.0e9
                    if latest_number("first_accepted_measurement_time_ns")
                    is not None else None),
                "last_accepted_measurement_time_s": (
                    latest_number("last_accepted_measurement_time_ns") / 1.0e9
                    if latest_number("last_accepted_measurement_time_ns")
                    is not None else None),
                "accepted_measurement_interval_count": latest_number(
                    "accepted_measurement_interval_count"),
                "nonpositive_accepted_measurement_interval_count": (
                    latest_number(
                        "nonpositive_accepted_measurement_interval_count")),
                "minimum_accepted_measurement_interval_s": (
                    latest_number("minimum_accepted_measurement_interval_ns") /
                    1.0e9
                    if latest_number("minimum_accepted_measurement_interval_ns")
                    is not None else None),
                "maximum_accepted_measurement_interval_s": (
                    latest_number("maximum_accepted_measurement_interval_ns") /
                    1.0e9
                    if latest_number("maximum_accepted_measurement_interval_ns")
                    is not None else None),
                "mean_accepted_measurement_interval_s": (
                    latest_number("mean_accepted_measurement_interval_ns") /
                    1.0e9
                    if latest_number("mean_accepted_measurement_interval_ns")
                    is not None else None),
                "realized_measurement_hz": latest_number(
                    "realized_measurement_hz"),
                "maximum_merged_span_ms": (
                    latest_number("maximum_merged_span_ns") / 1.0e6
                    if latest_number("maximum_merged_span_ns") is not None
                    else None),
                "last_primary_point_count": latest_number(
                    "last_primary_point_count"),
                "maximum_primary_point_count": latest_number(
                    "maximum_primary_point_count"),
                "last_secondary_point_count": latest_number(
                    "last_secondary_point_count"),
                "maximum_secondary_point_count": latest_number(
                    "maximum_secondary_point_count"),
                "last_merged_point_count": latest_number(
                    "last_merged_point_count"),
                "maximum_merged_point_count": latest_number(
                    "maximum_merged_point_count"),
                "last_secondary_to_primary_point_ratio": latest_number(
                    "last_secondary_to_primary_point_ratio"),
                "last_merged_to_primary_point_ratio": latest_number(
                    "last_merged_to_primary_point_ratio"),
                "last_total_raw_input_point_count": latest_number(
                    "last_total_raw_input_point_count"),
                "maximum_total_raw_input_point_count": latest_number(
                    "maximum_total_raw_input_point_count"),
                "last_compacted_output_point_count": latest_number(
                    "last_compacted_output_point_count"),
                "maximum_compacted_output_point_count": latest_number(
                    "maximum_compacted_output_point_count"),
                "last_removed_no_return_sentinel_count": latest_number(
                    "last_removed_no_return_sentinel_count"),
                "maximum_removed_no_return_sentinel_count": latest_number(
                    "maximum_removed_no_return_sentinel_count"),
                "last_compacted_output_to_total_raw_input_ratio": (
                    latest_number(
                        "last_compacted_output_to_total_raw_input_ratio")),
                "overlap_evaluations_attempted_latest": latest_number(
                    "overlap_evaluations_attempted"),
                "overlap_evaluations_failed_latest": latest_number(
                    "overlap_evaluations_failed"),
                "overlap_evaluation_sample_count_latest": latest_number(
                    "overlap_evaluation_sample_count"),
                "last_overlap_pair_index": latest_number(
                    "last_overlap_pair_index"),
                "last_primary_valid_return_count": latest_number(
                    "last_primary_valid_return_count"),
                "last_secondary_valid_return_count": latest_number(
                    "last_secondary_valid_return_count"),
                "last_primary_occupied_voxel_count": latest_number(
                    "last_primary_occupied_voxel_count"),
                "last_secondary_occupied_voxel_count": latest_number(
                    "last_secondary_occupied_voxel_count"),
                "last_shared_occupied_voxel_count": latest_number(
                    "last_shared_occupied_voxel_count"),
                "last_union_occupied_voxel_count": latest_number(
                    "last_union_occupied_voxel_count"),
                "last_primary_overlap_fraction": latest_number(
                    "last_primary_overlap_fraction"),
                "last_secondary_overlap_fraction": latest_number(
                    "last_secondary_overlap_fraction"),
                "last_voxel_jaccard": latest_number("last_voxel_jaccard"),
                "mean_primary_overlap_fraction": latest_number(
                    "mean_primary_overlap_fraction"),
                "mean_secondary_overlap_fraction": latest_number(
                    "mean_secondary_overlap_fraction"),
                "mean_voxel_jaccard": latest_number("mean_voxel_jaccard"),
                "minimum_primary_overlap_fraction": latest_number(
                    "minimum_primary_overlap_fraction"),
                "minimum_secondary_overlap_fraction": latest_number(
                    "minimum_secondary_overlap_fraction"),
                "minimum_voxel_jaccard": latest_number(
                    "minimum_voxel_jaccard"),
                "primary_pair_ratio_latest": latest_number(
                    "primary_pair_ratio"),
                "secondary_pair_ratio_latest": latest_number(
                    "secondary_pair_ratio"),
            }, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")

        return estimates

    def _generate_analysis(self, aligned_algorithm_rows=None):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return
        analysis = os.path.join(self.output, "analysis")
        # Robot trajectory: Gazebo body truth is transformed by the fixed
        # startup anchor, making both curves directly comparable in camera_init.
        # The capture cutoff is the immutable primary-analysis boundary.
        # Its post-capture interval is called RETURN only in the closed loop;
        # stationary mode uses the neutral POST_PRIMARY_CAPTURE label.
        def read_rows(relative):
            with open(os.path.join(self.output, relative), newline="") as stream:
                return list(csv.DictReader(stream))

        slam = read_rows("trajectories/slam_odometry.csv")
        truth = read_rows("trajectories/gazebo_robot_truth.csv")
        if slam and truth:
            fig, axis = plt.subplots(figsize=(8, 7))
            truth_valid = [row for row in truth if row["truth_map_x_m"]]
            cutoff = self.capture_freeze_ros_time_s
            post_segment = ("RETURN_RECORDED_ONLY" if self.return_after_detection
                            else "POST_PRIMARY_CAPTURE")
            for segment, line_style, alpha in (
                    ("PRIMARY_CAPTURE", "-", 1.0),
                    (post_segment, "--", 0.65)):
                slam_segment = [
                    row for row in slam
                    if trajectory_analysis_segment(
                        float(row.get("message_time_s", "")
                              or row["ros_time_s"]), cutoff,
                        self.return_after_detection) == segment]
                truth_segment = [
                    row for row in truth_valid
                    if trajectory_analysis_segment(
                        float(row.get("message_time_s", "")
                              or row["ros_time_s"]), cutoff,
                        self.return_after_detection) == segment]
                label_suffix = ("primary capture"
                                if segment == "PRIMARY_CAPTURE"
                                else ("return (recorded only)"
                                      if self.return_after_detection
                                      else "post-primary capture"))
                if slam_segment:
                    axis.plot([float(row["x_m"]) for row in slam_segment],
                              [float(row["y_m"]) for row in slam_segment],
                              line_style, alpha=alpha,
                              label="FAST-LIO " + label_suffix)
                if truth_segment:
                    axis.plot(
                        [float(row["truth_map_x_m"]) for row in truth_segment],
                        [float(row["truth_map_y_m"]) for row in truth_segment],
                        line_style, alpha=alpha,
                        label="Gazebo truth " + label_suffix)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("camera_init x [m]")
            axis.set_ylabel("camera_init y [m]")
            axis.grid(True, alpha=0.3)
            axis.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(analysis, "robot_trajectory_comparison.png"), dpi=180)
            plt.close(fig)
            truth_valid = [row for row in truth if row["truth_map_x_m"]]
            if truth_valid:
                truth_time = np.asarray(
                    [float(row.get("message_time_s", "")
                           or row["ros_time_s"]) for row in truth_valid])
                truth_x = np.asarray(
                    [float(row["truth_map_x_m"]) for row in truth_valid])
                truth_y = np.asarray(
                    [float(row["truth_map_y_m"]) for row in truth_valid])
                trajectory_fields = [
                    "ros_time_s", "analysis_segment", "slam_x_m", "slam_y_m",
                    "gazebo_truth_map_x_m", "gazebo_truth_map_y_m",
                    "planar_position_error_m"]
                trajectory_rows = []
                for row in slam:
                    stamp = float(row.get("message_time_s", "")
                                  or row["ros_time_s"])
                    if stamp < truth_time[0] or stamp > truth_time[-1]:
                        continue
                    estimated_x = float(row["x_m"])
                    estimated_y = float(row["y_m"])
                    actual_x = float(np.interp(stamp, truth_time, truth_x))
                    actual_y = float(np.interp(stamp, truth_time, truth_y))
                    trajectory_rows.append({
                        "ros_time_s": stamp,
                        "analysis_segment": trajectory_analysis_segment(
                            stamp, cutoff, self.return_after_detection),
                        "slam_x_m": estimated_x,
                        "slam_y_m": estimated_y,
                        "gazebo_truth_map_x_m": actual_x,
                        "gazebo_truth_map_y_m": actual_y,
                        "planar_position_error_m": math.hypot(
                            estimated_x - actual_x,
                            estimated_y - actual_y)})
                primary_rows = [
                    row for row in trajectory_rows
                    if row["analysis_segment"] == "PRIMARY_CAPTURE"]
                post_rows = [
                    row for row in trajectory_rows
                    if row["analysis_segment"] == post_segment]
                post_filename = ("robot_trajectory_error_return.csv"
                                 if self.return_after_detection else
                                 "robot_trajectory_error_post_primary_capture.csv")
                for filename, rows in (
                        ("robot_trajectory_error.csv", primary_rows),
                        (post_filename, post_rows),
                        ("robot_trajectory_error_all_segments.csv",
                         trajectory_rows)):
                    trajectory_path = os.path.join(analysis, filename)
                    with open(trajectory_path, "w", newline="") as stream:
                        writer = csv.DictWriter(
                            stream, fieldnames=trajectory_fields)
                        writer.writeheader()
                        writer.writerows(rows)
                def error_summary(rows):
                    errors = [float(row["planar_position_error_m"])
                              for row in rows]
                    return {
                        "sample_count": len(errors),
                        "rmse_m": (float(math.sqrt(np.mean(np.square(errors))))
                                   if errors else None),
                        "median_m": (float(np.median(errors))
                                     if errors else None),
                        "p95_m": (float(np.percentile(errors, 95))
                                  if errors else None),
                        "maximum_m": max(errors) if errors else None,
                    }
                with open(os.path.join(
                        analysis, "robot_trajectory_error_summary.json"),
                        "w", encoding="utf-8") as stream:
                    json.dump({
                        "primary_metric_segment": "PRIMARY_CAPTURE",
                        "primary_interval_end_ros_time_s": cutoff,
                        "post_primary_segment": post_segment,
                        "post_primary_segment_counted_in_primary_metric": False,
                        "post_primary_segment_file": post_filename,
                        "primary": error_summary(primary_rows),
                        "post_primary_recorded_only": error_summary(post_rows),
                    }, stream, indent=2, sort_keys=True, ensure_ascii=False)
                    stream.write("\n")

        object_truth = read_rows("truth/m01_m06_motion_truth.csv")
        estimates = (aligned_algorithm_rows
                     if aligned_algorithm_rows is not None
                     else read_rows("deformation/algorithm_displacement.csv"))
        evolutions = read_rows("evolution/gdu_evolution.csv")
        track_inventory = read_rows("analysis/detection_track_inventory.csv")
        matched_track_relations = build_unique_matched_track_relations(
            track_inventory)
        truth_floor_m = 0.005

        # Produce the object-motion comparison from the exact downstream
        # source that triggers avoidance: each retained GDU's evolution mean.
        # The only GDU-to-model relation is the independent-frame decision in
        # detection_track_inventory.  Raw spatial rows remain available, but
        # their sample majority cannot force an ambiguous track onto a model.
        association_truth = {}
        for row in estimates:
            gdu_id = str(row.get("gdu_id", ""))
            relation = matched_track_relations.get(gdu_id)
            if relation is None:
                continue
            model = relation["model_name"]
            if (str(row.get("association_status", "")).upper() != "MATCHED"
                    or str(row.get("model_name", "")) != model):
                continue
            key = (row["accepted_frame_index"], gdu_id)
            truth_vector = tuple(_optional_float(row, (field,)) for field in (
                "truth_dx_map_m", "truth_dy_map_m", "truth_dz_map_m"))
            if all(value is not None for value in truth_vector):
                association_truth[(key, model)] = (
                    truth_vector[0], truth_vector[1], truth_vector[2])
        truth_translation_series = defaultdict(list)
        for row in object_truth:
            stamp = _optional_float(row, ("truth_sim_time_s", "ros_time_s"))
            if stamp is None:
                continue
            vector = tuple(_optional_float(row, (key,)) for key in (
                "translation_x_m", "translation_y_m", "translation_z_m"))
            if all(value is not None for value in vector):
                truth_translation_series[row["model_name"]].append(
                    (stamp, vector))
        for values in truth_translation_series.values():
            values.sort(key=lambda item: item[0])

        def synchronized_truth_vector(model, stamp):
            series = truth_translation_series.get(model, [])
            if stamp is None or not series or stamp < series[0][0] or stamp > series[-1][0]:
                return None
            times = np.asarray([item[0] for item in series], dtype=float)
            world = tuple(float(np.interp(
                stamp, times, np.asarray([item[1][axis]
                                         for item in series], dtype=float)))
                          for axis in range(3))
            yaw = self.initial_map_from_world[2] if self.initial_map_from_world else 0.0
            mapped_xy = rotate_xy(world[:2], yaw)
            return (mapped_xy[0], mapped_xy[1], world[2])
        object_fields = [
            "ros_time_s", "message_time_s", "receive_time_s",
            "experiment_elapsed_s_at_receive", "source_accepted_frame_index",
            "gdu_id", "model_name",
            "track_association_status", "confirmed_detection_source",
            "track_confirmation_time_available", "track_evaluation_category",
            "retained_for_avoidance",
            "task_scale_probability",
            "estimated_dx_m", "estimated_dy_m", "estimated_dz_m",
            "estimated_magnitude_m", "truth_dx_map_m", "truth_dy_map_m",
            "truth_dz_map_m", "truth_magnitude_m", "vector_error_m",
            "magnitude_error_m", "direction_cosine"]
        object_rows = []
        for row in evolutions:
            key = (row["source_accepted_frame_index"], row["gdu_id"])
            relation = matched_track_relations.get(str(row["gdu_id"]))
            if relation is None:
                continue
            model = relation["model_name"]
            measurement_stamp = _optional_float(
                row, ("message_time_s", "ros_time_s"))
            truth_vector = synchronized_truth_vector(model, measurement_stamp)
            if truth_vector is None:
                truth_vector = association_truth.get((key, model))
            if truth_vector is None:
                continue
            estimate_vector = (
                float(row["mean_dx_m"]), float(row["mean_dy_m"]),
                float(row["mean_dz_m"]))
            estimate_magnitude, truth_magnitude, vector_error, cosine = (
                vector_metrics(estimate_vector, truth_vector))
            object_rows.append({
                "ros_time_s": row["ros_time_s"],
                "message_time_s": row.get("message_time_s", ""),
                "receive_time_s": row.get("receive_time_s", row["ros_time_s"]),
                "experiment_elapsed_s_at_receive": row.get(
                    "experiment_elapsed_s_at_receive", ""),
                "source_accepted_frame_index": row["source_accepted_frame_index"],
                "gdu_id": row["gdu_id"], "model_name": model,
                "track_association_status": "MATCHED",
                "confirmed_detection_source": relation["confirmed"],
                "track_confirmation_time_available": relation[
                    "confirmation_time_available"],
                "track_evaluation_category": relation[
                    "evaluation_category"],
                "retained_for_avoidance": row["retained_for_avoidance"],
                "task_scale_probability": row["task_scale_probability"],
                "estimated_dx_m": estimate_vector[0],
                "estimated_dy_m": estimate_vector[1],
                "estimated_dz_m": estimate_vector[2],
                "estimated_magnitude_m": estimate_magnitude,
                "truth_dx_map_m": truth_vector[0],
                "truth_dy_map_m": truth_vector[1],
                "truth_dz_map_m": truth_vector[2],
                "truth_magnitude_m": truth_magnitude,
                "vector_error_m": vector_error,
                "magnitude_error_m": abs(estimate_magnitude - truth_magnitude),
                "direction_cosine": cosine})
        object_path = os.path.join(analysis, "object_motion_comparison.csv")
        with open(object_path, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=object_fields)
            writer.writeheader()
            writer.writerows(object_rows)
        if object_truth:
            fig, axis = plt.subplots(figsize=(10, 6))
            for name in ("m01", "m02", "m03", "m04", "m05", "m06"):
                selected = [row for row in object_truth if row["model_name"] == name]
                if selected:
                    axis.plot([float(row.get("truth_sim_time_s", "")
                                           or row["ros_time_s"])
                               for row in selected],
                              [1000.0 * float(
                                  row.get("surface_max_displacement_m", "")
                                  or row["translation_magnitude_m"])
                               for row in selected], label=name + " truth")
                matched = [row for row in object_rows
                           if row["model_name"] == name
                           and str(row["confirmed_detection_source"]).lower()
                           in ("true", "1")
                           and str(row["retained_for_avoidance"]).lower()
                           in ("true", "1")]
                if matched:
                    axis.scatter([float(row.get("message_time_s", "")
                                        or row["ros_time_s"])
                                  for row in matched],
                                 [1000.0 * float(row["estimated_magnitude_m"])
                                  for row in matched], s=5, alpha=0.35,
                                 label=name + " estimate")
            axis.set_xlabel("simulation time [s]")
            axis.set_ylabel("displacement magnitude [mm]")
            axis.grid(True, alpha=0.3)
            axis.legend(ncol=2, fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(analysis, "m01_m06_displacement_comparison.png"),
                        dpi=180)
            plt.close(fig)

        summary_path = os.path.join(analysis, "model_detection_summary.csv")
        fields = [
            "model_name", "truth_active_sample_count",
            "associated_estimate_sample_count", "retained_estimate_sample_count",
            "confirmed_source_sample_count", "evaluation_sample_count",
            "evaluation_truth_floor_m", "maximum_estimated_magnitude_m",
            "maximum_truth_magnitude_m", "maximum_task_scale_probability",
            "evaluation_median_vector_error_m", "evaluation_p95_vector_error_m",
            "evaluation_rmse_vector_error_m",
            "evaluation_median_magnitude_error_m",
            "evaluation_p95_magnitude_error_m", "unmatched_estimate_sample_count",
            "ambiguous_estimate_sample_count",
            "evaluation_median_direction_cosine"]
        with open(summary_path, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for name in ("m01", "m02", "m03", "m04", "m05", "m06"):
                model_truth = [row for row in object_truth if row["model_name"] == name]
                truth_active = [row for row in model_truth
                                if float(
                                    row.get("surface_max_displacement_m", "")
                                    or row.get("translation_magnitude_m", 0.0)
                                    or 0.0) >= truth_floor_m]
                associated_gdu_ids = {
                    gdu_id for gdu_id, relation
                    in matched_track_relations.items()
                    if relation["model_name"] == name}
                associated_estimates = [
                    row for row in estimates
                    if str(row.get("gdu_id", "")) in associated_gdu_ids]
                matched = [row for row in object_rows
                           if row["model_name"] == name
                           and str(row["confirmed_detection_source"]).lower()
                           in ("true", "1")
                           and str(row["retained_for_avoidance"]).lower()
                           in ("true", "1")]
                retained_estimates = [row for row in object_rows
                                      if row["model_name"] == name
                                      and str(row["retained_for_avoidance"]).lower()
                                      in ("true", "1")]
                evaluation = [row for row in matched
                              if row.get("truth_magnitude_m", "")
                              and float(row["truth_magnitude_m"]) >= truth_floor_m]
                numeric = lambda rows, key: [float(row[key]) for row in rows
                                             if row.get(key, "")]
                vector_errors = numeric(evaluation, "vector_error_m")
                magnitude_errors = numeric(evaluation, "magnitude_error_m")
                direction_cosines = numeric(evaluation, "direction_cosine")
                writer.writerow({
                    "model_name": name,
                    "truth_active_sample_count": len(truth_active),
                    "associated_estimate_sample_count": len(associated_estimates),
                    "retained_estimate_sample_count": len(retained_estimates),
                    "confirmed_source_sample_count": len(matched),
                    "evaluation_sample_count": len(evaluation),
                    "evaluation_truth_floor_m": truth_floor_m,
                    "maximum_estimated_magnitude_m": max(
                        numeric(matched, "estimated_magnitude_m"), default=""),
                    "maximum_truth_magnitude_m": max(
                        (float(row.get("surface_max_displacement_m", "")
                               or row["translation_magnitude_m"])
                         for row in model_truth),
                        default=""),
                    "maximum_task_scale_probability": max(
                        numeric(matched, "task_scale_probability"), default=""),
                    "evaluation_median_vector_error_m": (
                        float(np.median(vector_errors)) if vector_errors else ""),
                    "evaluation_p95_vector_error_m": (
                        float(np.percentile(vector_errors, 95)) if vector_errors else ""),
                    "evaluation_rmse_vector_error_m": (
                        float(math.sqrt(np.mean(np.square(vector_errors))))
                        if vector_errors else ""),
                    "evaluation_median_magnitude_error_m": (
                        float(np.median(magnitude_errors)) if magnitude_errors else ""),
                    "evaluation_p95_magnitude_error_m": (
                        float(np.percentile(magnitude_errors, 95))
                        if magnitude_errors else ""),
                    "unmatched_estimate_sample_count": "",
                    "ambiguous_estimate_sample_count": "",
                    "evaluation_median_direction_cosine": (
                        float(np.median(direction_cosines))
                        if direction_cosines else "")})
            unmatched = [row for row in estimates
                         if row.get("association_status", "") == "UNMATCHED"]
            writer.writerow({
                "model_name": "UNMATCHED",
                "truth_active_sample_count": "",
                "associated_estimate_sample_count": "",
                "retained_estimate_sample_count": "",
                "confirmed_source_sample_count": "",
                "evaluation_sample_count": "",
                "evaluation_truth_floor_m": truth_floor_m,
                "maximum_estimated_magnitude_m": max(
                    (float(row["estimated_magnitude_m"]) for row in unmatched),
                    default=""),
                "maximum_truth_magnitude_m": "",
                "maximum_task_scale_probability": max(
                    (float(row["task_scale_probability"]) for row in unmatched
                     if row.get("task_scale_probability", "")),
                    default=""),
                "evaluation_median_vector_error_m": "",
                "evaluation_p95_vector_error_m": "",
                "evaluation_rmse_vector_error_m": "",
                "evaluation_median_magnitude_error_m": "",
                "evaluation_p95_magnitude_error_m": "",
                "unmatched_estimate_sample_count": len(unmatched),
                "ambiguous_estimate_sample_count": "",
                "evaluation_median_direction_cosine": "",
            })
            ambiguous_samples = [row for row in estimates
                                 if row.get("association_status", "")
                                 == "AMBIGUOUS"]
            writer.writerow({
                "model_name": "AMBIGUOUS",
                "truth_active_sample_count": "",
                "associated_estimate_sample_count": "",
                "retained_estimate_sample_count": "",
                "confirmed_source_sample_count": "",
                "evaluation_sample_count": "",
                "evaluation_truth_floor_m": truth_floor_m,
                "maximum_estimated_magnitude_m": max(
                    (float(row["estimated_magnitude_m"])
                     for row in ambiguous_samples), default=""),
                "maximum_truth_magnitude_m": "",
                "maximum_task_scale_probability": max(
                    (float(row["task_scale_probability"])
                     for row in ambiguous_samples
                     if row.get("task_scale_probability", "")), default=""),
                "evaluation_median_vector_error_m": "",
                "evaluation_p95_vector_error_m": "",
                "evaluation_rmse_vector_error_m": "",
                "evaluation_median_magnitude_error_m": "",
                "evaluation_p95_magnitude_error_m": "",
                "unmatched_estimate_sample_count": "",
                "ambiguous_estimate_sample_count": len(ambiguous_samples),
                "evaluation_median_direction_cosine": "",
            })

        # Human-readable previews are derived from the exact NPZ snapshots;
        # the NPZ remains the quantitative map/cloud product used in analysis.
        for path in glob.glob(os.path.join(
                self.output, "snapshots", "maps", "*.npz")):
            with np.load(path, allow_pickle=False) as stored:
                grid = stored["data"]
            fig, axis = plt.subplots(figsize=(9, 7))
            image = axis.imshow(grid, origin="lower", cmap="gray_r",
                                vmin=-1, vmax=100)
            axis.set_title(os.path.basename(path).replace(".npz", ""))
            axis.set_xlabel("grid column")
            axis.set_ylabel("grid row")
            fig.colorbar(image, ax=axis, label="occupancy value")
            fig.tight_layout()
            fig.savefig(os.path.join(
                analysis, os.path.basename(path).replace(".npz", ".png")),
                dpi=180)
            plt.close(fig)
        for path in glob.glob(os.path.join(
                self.output, "snapshots", "clouds", "*.npz")):
            with np.load(path, allow_pickle=False) as stored:
                points = stored["points"]
            if points.size == 0:
                continue
            stride = max(1, int(math.ceil(len(points) / 120000.0)))
            shown = points[::stride]
            fig, axes = plt.subplots(1, 2, figsize=(13, 6))
            axes[0].scatter(shown[:, 0], shown[:, 1], c=shown[:, 2],
                            s=0.2, cmap="viridis")
            axes[0].set_aspect("equal", adjustable="box")
            axes[0].set_title("top view")
            axes[0].set_xlabel("x [m]")
            axes[0].set_ylabel("y [m]")
            axes[1].scatter(shown[:, 0], shown[:, 2], c=shown[:, 1],
                            s=0.2, cmap="viridis")
            axes[1].set_aspect("equal", adjustable="box")
            axes[1].set_title("side view")
            axes[1].set_xlabel("x [m]")
            axes[1].set_ylabel("z [m]")
            fig.tight_layout()
            fig.savefig(os.path.join(
                analysis, os.path.basename(path).replace(".npz", ".png")),
                dpi=180)
            plt.close(fig)

    def _finalize_files(self, request=None):
        with self.lock:
            if self.closed:
                raise RuntimeError("recorder streams were already closed")
            _, snapshot_committed = self._snapshot_phase("FINAL")
            if not snapshot_committed:
                raise OSError("FINAL bounded snapshot was not committed")
            for stream in self.streams.values():
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
            self.closed = True
        analysis_errors = []
        request_document = copy.deepcopy(request) if request else {}
        with self.lock:
            truth_input_errors = tuple(self.truth_contract_input_errors)
            truth_rejected_count = int(self.truth_rejected_batch_count)
        truth_contract = gazebo_truth_artifact_contract(
            self.output,
            request_document.get("required_motion_experiment_elapsed_s"),
            stationary_mode=not self.return_after_detection,
            input_errors=truth_input_errors,
            rejected_input_count=truth_rejected_count)
        truth_contract_path = os.path.join(
            self.output, "analysis", "gazebo_truth_contract.json")
        truth_contract_temporary = truth_contract_path + ".tmp"
        try:
            with open(truth_contract_temporary, "w", encoding="utf-8") as stream:
                json.dump(truth_contract, stream, indent=2, sort_keys=True,
                          ensure_ascii=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(truth_contract_temporary, truth_contract_path)
        except OSError as error:
            analysis_errors.append("gazebo_truth_contract_write: {}".format(
                error))
            try:
                os.unlink(truth_contract_temporary)
            except OSError:
                pass
        analysis_errors.extend(
            "gazebo_truth_contract: " + error
            for error in truth_contract["errors"])
        aligned_algorithm_rows = None
        try:
            aligned_algorithm_rows = self._generate_track_and_runtime_analysis()
        except (OSError, ValueError, TypeError) as error:
            rospy.logerr("could not generate track/runtime analysis: %s", error)
            analysis_errors.append("track_runtime: {}".format(error))
        try:
            self._generate_analysis(aligned_algorithm_rows)
        except (OSError, ValueError) as error:
            rospy.logerr("could not generate experiment analysis: %s", error)
            analysis_errors.append("experiment_analysis: {}".format(error))
        required_paths = [
            "mission_summary.json",
            "mission_decision_summary.json",
            "ros_parameters.json",
            "truth/m01_m06_motion_truth.csv",
            "deformation/algorithm_displacement.csv",
            "deformation/detection_evidence.csv",
            "evolution/gdu_evolution.csv",
            "maps/avoidance_frames.csv",
            "sensors/dual_lidar_merge_diagnostics.csv",
            "sensors/gazebo_performance_metrics.csv",
            "analysis/detection_run_summary.json",
            "analysis/truth_alignment_summary.json",
            "analysis/gazebo_truth_contract.json",
            "analysis/detection_track_inventory.csv",
            "analysis/model_detection_event_metrics.csv",
            "analysis/detector_runtime_summary.json",
            "analysis/dual_lidar_timing_summary.json",
            "analysis/gazebo_performance_summary.json",
            "analysis/robot_trajectory_comparison.png",
            "analysis/m01_m06_displacement_comparison.png",
            "analysis/deformation_route_maps/metrics.json",
            "analysis/deformation_route_maps/12_complete_dashboard.png",
            "analysis/deformation_route_maps/13_fused_planning_map.png",
        ]
        checkpoint_contract = checkpoint_artifact_contract(
            self.output, self.requested_timeline_checkpoints_s)
        # Scene verification is read-only and runs before robot spawning.
        # It records the saved state actually loaded, never a detector ROI.
        if rospy.get_param("/spawn_mobile_mid360/scene_world", ""):
            required_paths.extend(["scene/scene_startup.json", "scene/loaded_scene.world",
                                   "scene/motion_config.yaml"])
            try:
                with open(os.path.join(self.output, "scene/scene_startup.json")) as stream:
                    scene = json.load(stream)
                if (scene.get("status") != "VERIFIED"
                        or scene["world_sha256"] != self._sha256(os.path.join(self.output, "scene/loaded_scene.world"))
                        or scene["motion_config_sha256"] != self._sha256(os.path.join(self.output, "scene/motion_config.yaml"))):
                    raise ValueError("startup scene verification or snapshot hash mismatch")
            except (OSError, ValueError, KeyError) as error:
                analysis_errors.append("scene_verification: " + str(error))
        required_paths.extend(checkpoint_contract["required_paths"])
        analysis_errors.extend(
            "checkpoint_contract: " + error
            for error in checkpoint_contract["errors"])

        latest_dual_diagnostic = {}
        dual_diagnostic_path = os.path.join(
            self.output, "sensors", "dual_lidar_merge_diagnostics.csv")
        try:
            with open(dual_diagnostic_path, newline="", encoding="utf-8") as stream:
                dual_rows = list(csv.DictReader(stream))
            latest_dual_diagnostic = dual_rows[-1] if dual_rows else {}
        except (OSError, UnicodeError, csv.Error):
            latest_dual_diagnostic = {}
        dual_lidar_contract = dual_lidar_success_contract(
            latest_dual_diagnostic,
            getattr(self, "dual_lidar_success_thresholds", None))
        dual_lidar_contract["required_for_this_run"] = bool(
            self.require_dual_lidar_success_contract)
        dual_lidar_contract["fresh_diagnostic_after_finalization_request"] = {
            "request_sequence": self.dual_lidar_request_sequence,
            "final_sequence": self.dual_lidar_final_sequence,
            "request_monotonic_s": self.dual_lidar_request_monotonic_s,
            "request_wall_time_s": self.dual_lidar_request_wall_time_s,
            "final_receive_monotonic_s": (
                self.dual_lidar_final_receive_monotonic_s),
            "final_receive_wall_time_s": (
                self.dual_lidar_final_receive_wall_time_s),
            "satisfied": (
                not self.require_dual_lidar_success_contract
                or (self.dual_lidar_request_sequence is not None
                    and self.dual_lidar_final_sequence is not None
                    and self.dual_lidar_final_sequence
                    > self.dual_lidar_request_sequence
                    and self.dual_lidar_request_monotonic_s is not None
                    and self.dual_lidar_final_receive_monotonic_s is not None
                    and self.dual_lidar_final_receive_monotonic_s
                    > self.dual_lidar_request_monotonic_s)),
            "timeout_budget_s": self.finalization_high_water_timeout_s,
        }
        if (self.require_dual_lidar_success_contract
                and not dual_lidar_contract[
                    "fresh_diagnostic_after_finalization_request"][
                        "satisfied"]):
            dual_lidar_contract["status"] = "FAILED"
            dual_lidar_contract["errors"].append(
                "fresh_diagnostic_after_finalization_request")
        if (self.require_dual_lidar_success_contract
                and dual_lidar_contract["status"] != "COMPLETED"):
            analysis_errors.extend(
                "dual_lidar_contract: " + error
                for error in dual_lidar_contract["errors"])
        if not self.return_after_detection:
            # The stationary full-process experiment promises retrospective
            # runtime forecasts.  Require the real arrays and figures only in
            # that mode; the closed-loop run must not substitute an older
            # forecast for a live decision snapshot.
            required_paths.extend(STATIONARY_FORECAST_REQUIRED_PATHS)
        artifact_errors = validate_required_artifacts(
            self.output, required_paths)
        missing_paths = [error[len("missing: "):]
                         for error in artifact_errors
                         if error.startswith("missing: ")]
        analysis_errors.extend(artifact_errors)
        with self.lock:
            self.required_artifact_errors = list(analysis_errors)
        request = request_document
        recorder_contract_completed = (
            request.get("mission_outcome") == "COMPLETED"
            and not analysis_errors)
        manifest = {
            "schema_version": 3,
            "experiment": "ICRA2027 direct detection / closed-loop",
            "run_session_id": request.get("run_session_id"),
            "finalization_request": request,
            "finalization_outcome": (
                "COMPLETED" if recorder_contract_completed else "FAILED"),
            "required_artifact_errors": list(analysis_errors),
            "required_artifact_contract": {
                "phase": "RECORDER_FINALIZATION",
                "required_paths": list(required_paths),
                "missing_paths": list(missing_paths),
                "status": ("COMPLETED" if recorder_contract_completed
                           else "FAILED"),
                "errors": list(analysis_errors),
            },
            "timeline_checkpoint_artifact_contract": checkpoint_contract,
            "gazebo_motion_truth_contract": truth_contract,
            "dual_lidar_success_contract": dual_lidar_contract,
            # ROS log files remain open until roslaunch exits.  The two run
            # wrappers scrub them afterwards and atomically advance this
            # second-stage contract without changing the recorder outcome.
            "post_shutdown_artifact_contract": {
                "phase": "POST_ROSLAUNCH",
                "required_paths": [
                    "ros_log_scrub_report.json", "mission_summary.json"],
                "status": ("PENDING" if recorder_contract_completed
                           else "BLOCKED"),
                "errors": [],
            },
            "final_output_contract_status": (
                "PENDING_POST_SHUTDOWN" if recorder_contract_completed
                else "FAILED"),
            "mission_state_at_close": self.state,
            "rosbag_recorded": False,
            "truth_used_by_detector": False,
            "freeze_on_detection_state": self.freeze_on_detection_state,
            "freeze_on_monitoring_complete": self.freeze_on_monitoring_complete,
            "deformation_capture_frozen_at_state": self.capture_freeze_state,
            "deformation_capture_freeze_ros_time_s": (
                self.capture_freeze_ros_time_s),
            "deformation_capture_cutoff_accepted_frame_index": (
                self.capture_cutoff_frame_index),
            "deformation_capture_cutoff_experiment_elapsed_s": (
                self.capture_cutoff_experiment_elapsed_s),
            "deformation_capture_freeze_committed": bool(
                self.capture_freeze_committed),
            "return_trajectory_recorded_after_capture_freeze": (
                self.return_after_detection),
            "primary_slam_analysis_interval": (
                "START_TO_{}".format(self.capture_freeze_state)
                if self.capture_freeze_state else "START_TO_RUN_END"),
            "post_primary_segment_semantics": (
                "RETURN_RECORDED_ONLY" if self.return_after_detection
                else "POST_PRIMARY_CAPTURE"),
            "post_primary_segment_counted_in_primary_slam_analysis": False,
            "post_primary_segment_analysis_file": (
                "analysis/robot_trajectory_error_return.csv"
                if self.return_after_detection else
                "analysis/robot_trajectory_error_post_primary_capture.csv"),
            "return_segment_analysis_file": (
                "analysis/robot_trajectory_error_return.csv"
                if self.return_after_detection else None),
            "primary_capture_end_state": self.capture_freeze_state or "RUN_END",
            "output_contract": "CSV continuous streams; JSON provenance; bounded NPZ snapshots; PNG analysis",
            "time_semantics": {
                "message_time_s": "source message header measurement time",
                "publish_time_s": "detector product publication time when exposed",
                "receive_time_s": "recorder callback ROS time",
                "truth_sim_time_s": "Gazebo motion-plugin simulation time",
                "experiment_elapsed_s": "motion experiment relative time",
                "experiment_elapsed_s_at_message": (
                    "detector source time mapped through time-aligned per-model "
                    "Gazebo truth when associated; affine clock fallback is "
                    "used for non-model product rows"),
                "experiment_elapsed_s_at_receive": (
                    "latest experiment elapsed observed by the recorder at "
                    "callback time; retained only for queue/CPU diagnostics"),
                "wall_elapsed_s": "recorder monotonic wall-clock elapsed time",
            },
            "coordinate_and_unit_semantics": {
                "si_policy": (
                    "positions/displacements/distances are metres; linear "
                    "velocities are metres per second; angles are radians; "
                    "angular velocities are radians per second; covariance "
                    "entries use the corresponding squared SI units"),
                "truth_command_speed_exception": (
                    "linear_speed_mm_s is the commanded fixture speed in "
                    "millimetres per second; all realized translation and "
                    "surface-displacement columns are metres"),
                "quaternion_order": "x,y,z,w",
                "truth_object_frame": "world",
                "detector_vector_frame_column": "frame_id",
                "gazebo_robot_pose_frame_column": "frame_id",
                "gazebo_robot_twist_frame_column": "child_frame_id",
                "truth_map_pose_frame_column": "truth_map_frame_id",
                "fixed_truth_map_frame_id": self.initial_map_frame_id or None,
            },
            "primary_metric_models": list(self.primary_metric_models),
            "excluded_evaluation_models": sorted(
                {"m01", "m02", "m03", "m04", "m05", "m06"}
                - set(self.primary_metric_models)),
            "checkpoint_map_semantics": (
                "nearest latest asynchronously received map at each checkpoint; "
                "NPZ stamp/revision and CSV map age quantify staleness"),
            "algorithm_truth_alignment": {
                "target_time": "detector message_time_s",
                "bracketed_method": (
                    "linear position/displacement/AABB interpolation plus "
                    "shortest-arc normalized quaternion SLERP"),
                "unbracketed_method": "bounded zero-order hold",
                "maximum_zoh_residual_s": (
                    self.truth_alignment_max_zoh_residual_s),
                "residual_field": "association_truth_time_offset_s",
                "finalization_policy": (
                    "keep the raw detector CSV immutable; join the complete "
                    "m01-m06 truth history once in memory and write only the "
                    "derived evaluation tables"),
                "online_csv_policy": (
                    "detector-owned fields and fixed-anchor world coordinates "
                    "are buffered online; raw association/truth-error columns "
                    "remain pending because evaluation-only associations are "
                    "stored in analysis tables"),
                "detector_owned_columns_modified_at_finalization": False,
                "truth_feedback_to_detector_or_maps": False,
                "map_world_alignment": (
                    "fixed startup SE(2)+z anchor; Gazebo world and FAST-LIO "
                    "map roll/pitch are assumed aligned in this ground-robot sim"),
            },
            "se3_trajectory_evaluation": {
                "tum_exported": True,
                "ate_rpe_computed": False,
                "alignment_required_downstream": True,
            },
            "truth_model_to_fastlio_body_xyzyaw": list(self.calibration),
            "truth_association": {
                "evaluation_only": True,
                "feedback_to_detector_or_maps": False,
                "preferred_method": "WORLD_AABB",
                "fallback_method": "BOUNDED_ELLIPSOID",
                "aabb_margin_m": self.truth_association_aabb_margin_m,
                "statuses": ["MATCHED", "UNMATCHED", "AMBIGUOUS"],
            },
            "requested_timeline_checkpoints_s": sorted(
                self.requested_timeline_checkpoints_s),
            "unreached_timeline_checkpoints_s": sorted(
                self.pending_timeline_checkpoints_s),
            "initial_map_from_world_se2": (list(self.initial_map_from_world)
                                           if self.initial_map_from_world else None),
            "initial_z_map_from_world_m": self.initial_z_map_from_world,
            "initial_map_frame_id": self.initial_map_frame_id or None,
            "record_counts": dict(sorted(self.counts.items())),
            "csv_schemas": CSV_SCHEMAS,
        }
        temporary = os.path.join(self.output, "run_manifest.json.tmp")
        final = os.path.join(self.output, "run_manifest.json")
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True,
                      ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, final)
        if not recorder_contract_completed:
            raise RuntimeError("; ".join(analysis_errors)
                               if analysis_errors else
                               "missing completed finalization request")

    def close(self):
        """Shutdown hook; normal completed runs have already finalized."""
        with self.lock:
            worker = self.finalization_thread
            started = self.finalization_started
            terminal = self.finalized or bool(self.finalization_error)
        if started:
            if (worker is not None and worker is not threading.current_thread()
                    and not terminal):
                self.finalization_done.wait(
                    self.finalization_high_water_timeout_s + 180.0)
            return
        try:
            self._finalize_files(None)
        except Exception as error:
            rospy.logerr("incomplete recorder shutdown finalization: %s", error)


def main():
    rospy.init_node("icra2027_experiment_recorder")
    ExperimentRecorder()
    rospy.spin()


if __name__ == "__main__":
    main()
