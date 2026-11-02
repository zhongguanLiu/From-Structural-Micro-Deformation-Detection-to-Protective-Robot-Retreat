#!/usr/bin/env python3

import importlib.util
import inspect
import csv
import json
import math
import os
import sys
import tempfile
import threading
import time
import types
import unittest

import numpy as np


SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))


def _stub_module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


_stub_module(
    "rospy",
    Time=types.SimpleNamespace(
        now=lambda: types.SimpleNamespace(to_sec=lambda: 0.0)),
    logerr_throttle=lambda *args, **kwargs: None,
    logwarn_throttle=lambda *args, **kwargs: None,
    is_shutdown=lambda: False)
_stub_module("tf")
_stub_module("tf.transformations", euler_from_quaternion=lambda quat: (0.0, 0.0, 0.0))
_stub_module("geometry_msgs")
_stub_module("geometry_msgs.msg", Twist=type("Twist", (), {}))
_stub_module("nav_msgs")
_stub_module("nav_msgs.msg",
             Path=type("Path", (), {}),
             OccupancyGrid=type("OccupancyGrid", (), {}),
             Odometry=type("Odometry", (), {}))
_stub_module("move_base_msgs.msg",
             MoveBaseActionGoal=type("MoveBaseActionGoal", (), {}),
             MoveBaseActionResult=type("MoveBaseActionResult", (), {}))
_stub_module("sensor_msgs")
_stub_module("sensor_msgs.point_cloud2", read_points=lambda *args, **kwargs: [])
_stub_module("sensor_msgs.msg", PointCloud2=type("PointCloud2", (), {}), Imu=type("Imu", (), {}))
_stub_module("std_msgs")
_stub_module("std_msgs.msg", String=type("String", (), {}))
_stub_module("deform_avoidance_map")
_stub_module("deform_avoidance_map.msg",
             DeformationAvoidanceMap=type("DeformationAvoidanceMap", (), {}))
_stub_module("deform_evolution_map")
_stub_module("deform_evolution_map.msg",
             DeformationEvolutionFrame=type("DeformationEvolutionFrame", (), {}))
_stub_module("deform_monitor_v3")
_stub_module("deform_monitor_v3.msg",
             V1DetectionStatus=type("V1DetectionStatus", (), {}),
             V1DetectionTile=type("V1DetectionTile", (), {}))

SPEC = importlib.util.spec_from_file_location(
    "icra2027_experiment_recorder",
    os.path.join(SCRIPTS, "icra2027_experiment_recorder.py"))
RECORDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECORDER)
TRUTH_SPEC = importlib.util.spec_from_file_location(
    "icra2027_truth_recorder",
    os.path.join(
        SCRIPTS, "..", "..", "gazebo_test", "Mid360_simulation_plugin",
        "livox_laser_simulation", "scripts", "icra2027_truth_recorder.py"))
TRUTH_RECORDER = importlib.util.module_from_spec(TRUTH_SPEC)
TRUTH_SPEC.loader.exec_module(TRUTH_RECORDER)


def _fake_stamp(value):
    return types.SimpleNamespace(to_sec=lambda: float(value))


def _fake_grid(value, stamp_s=12.0):
    return types.SimpleNamespace(
        header=types.SimpleNamespace(
            frame_id="camera_init", stamp=_fake_stamp(stamp_s)),
        info=types.SimpleNamespace(
            width=2, height=2, resolution=0.1,
            origin=types.SimpleNamespace(position=types.SimpleNamespace(
                x=-0.1, y=-0.1, z=0.0))),
        data=[int(value)] * 4)


def _fake_avoidance(snapshot_id="snapshot-17", frame_index=17,
                    horizons=(0.5, 1.0, 2.0)):
    return types.SimpleNamespace(
        source_snapshot_id=snapshot_id,
        source_accepted_frame_index=frame_index,
        map_revision=4,
        header=types.SimpleNamespace(stamp=_fake_stamp(12.0)),
        current_overlay=_fake_grid(10),
        forecast_horizons_s=list(horizons),
        forecast_overlays=[_fake_grid(20 + index)
                           for index, _ in enumerate(horizons)])


def _valid_motion_truth_batch(experiment_elapsed=0.0, state="WAITING"):
    motion_elapsed = max(0.0, min(40.0, float(experiment_elapsed) - 10.0))
    models = []
    for model_id in range(1, 7):
        model_elapsed = motion_elapsed
        translation = [0.001 * model_elapsed, 0.0, 0.0]
        magnitude = abs(translation[0])
        models.append({
            "laser_retro_id": model_id,
            "model_name": "m{:02d}".format(model_id),
            "motion_elapsed": model_elapsed,
            "position": [float(model_id), 2.0, 0.5],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "aabb_min": [float(model_id) - 0.5, 1.5, 0.0],
            "aabb_max": [float(model_id) + 0.5, 2.5, 1.0],
            "translation": translation,
            "expected_translation_vector": list(translation),
            "translation_magnitude": magnitude,
            "translation_expected": magnitude,
            "translation_error": 0.0,
            "rotation_angle": 0.0,
            "rotation_expected": 0.0,
            "rotation_error": 0.0,
            "surface_rms_displacement": magnitude,
            "surface_max_displacement": magnitude,
            "surface_point_count": 0,
        })
    return {
        "schema_version": 2,
        "state": state,
        "sim_time": 100.0 + float(experiment_elapsed),
        "experiment_elapsed": float(experiment_elapsed),
        "waiting_elapsed": min(10.0, float(experiment_elapsed)),
        "motion_elapsed": motion_elapsed,
        "motion_mode": "simultaneous",
        "per_model_duration": 40.0,
        "linear_speed_mm_s": 1.0,
        "angular_speed_rad_s": 0.0015,
        "models": models,
    }


def _truth_csv_rows(document, batch_index):
    rows = []
    for model in document["models"]:
        row = {field: "" for field in
               RECORDER.CSV_SCHEMAS["truth/m01_m06_motion_truth.csv"]}
        row.update({
            "truth_frame_id": "world",
            "truth_batch_index": batch_index,
            "truth_schema_version": document["schema_version"],
            "motion_state": document["state"],
            "truth_sim_time_s": document["sim_time"],
            "experiment_elapsed_s": document["experiment_elapsed"],
            "waiting_elapsed_s": document["waiting_elapsed"],
            "motion_elapsed_s": document["motion_elapsed"],
            "model_motion_elapsed_s": model["motion_elapsed"],
            "motion_mode": document["motion_mode"],
            "per_model_duration_s": document["per_model_duration"],
            "linear_speed_mm_s": document["linear_speed_mm_s"],
            "angular_speed_rad_s": document["angular_speed_rad_s"],
            "model_name": model["model_name"],
            "laser_retro_id": model["laser_retro_id"],
            "position_x_m": model["position"][0],
            "position_y_m": model["position"][1],
            "position_z_m": model["position"][2],
            "orientation_x": model["orientation"][0],
            "orientation_y": model["orientation"][1],
            "orientation_z": model["orientation"][2],
            "orientation_w": model["orientation"][3],
            "world_aabb_min_x_m": model["aabb_min"][0],
            "world_aabb_min_y_m": model["aabb_min"][1],
            "world_aabb_min_z_m": model["aabb_min"][2],
            "world_aabb_max_x_m": model["aabb_max"][0],
            "world_aabb_max_y_m": model["aabb_max"][1],
            "world_aabb_max_z_m": model["aabb_max"][2],
            "translation_x_m": model["translation"][0],
            "translation_y_m": model["translation"][1],
            "translation_z_m": model["translation"][2],
            "expected_translation_x_m": model[
                "expected_translation_vector"][0],
            "expected_translation_y_m": model[
                "expected_translation_vector"][1],
            "expected_translation_z_m": model[
                "expected_translation_vector"][2],
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
            "surface_point_count": model["surface_point_count"],
        })
        rows.append(row)
    return rows


def _write_truth_csv(output, documents):
    directory = os.path.join(output, "truth")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "m01_m06_motion_truth.csv")
    fields = RECORDER.CSV_SCHEMAS["truth/m01_m06_motion_truth.csv"]
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for batch_index, document in enumerate(documents):
            writer.writerows(_truth_csv_rows(document, batch_index))


class ExperimentRecorderCoreTests(unittest.TestCase):
    def test_imu_keeps_sensor_timestamp_after_detection_cutoff(self):
        recorder = object.__new__(RECORDER.ExperimentRecorder)
        recorder.monitoring_capture_open = False
        recorder._base = lambda stamp: {"message_time_s": stamp}
        written = []
        recorder._write = lambda path, row: written.append((path, row))
        message = types.SimpleNamespace(
            header=types.SimpleNamespace(stamp=_fake_stamp(42), frame_id="livox_imu_link"),
            angular_velocity=types.SimpleNamespace(x=0, y=0, z=.8),
            linear_acceleration=types.SimpleNamespace(x=-.2, y=0, z=9.8))
        recorder._imu_callback(message)
        self.assertEqual(written[0][0], "sensors/imu.csv")
        self.assertEqual(written[0][1]["message_time_s"], 42)
        self.assertEqual(written[0][1]["wz_rad_s"], .8)
        self.assertEqual(written[0][1]["az_mps2"], 9.8)

    def test_navigation_costmap_deduplicates_content_but_keeps_timestamps(self):
        recorder = object.__new__(RECORDER.ExperimentRecorder)
        recorder.lock = threading.RLock()
        recorder.closed = False
        recorder.monitoring_capture_open = False
        recorder.navigation_map_ids = set()
        recorder._base = lambda stamp: {"message_time_s": stamp}
        written = []
        recorder._write = lambda path, row: written.append(row)
        message = _fake_grid(10)
        message.info.origin.orientation = types.SimpleNamespace(x=0, y=0, z=0, w=1)
        with tempfile.TemporaryDirectory() as output:
            recorder.output = output
            recorder._navigation_costmap_callback(message)
            message.header.stamp = _fake_stamp(13)
            recorder._navigation_costmap_callback(message)
            self.assertEqual(len(recorder.navigation_map_ids), 1)
            self.assertEqual(len(written), 2)
            self.assertEqual(written[1]["message_time_s"], 13)
            with np.load(os.path.join(output, written[0]["relative_path"])) as saved:
                self.assertTrue(np.all(saved["data"] == 10))
            message.data = [100] * 4
            recorder._navigation_costmap_callback(message)
            self.assertEqual(len(recorder.navigation_map_ids), 2)

    def test_navigation_records_continue_after_monitoring_cutoff(self):
        recorder = object.__new__(RECORDER.ExperimentRecorder)
        recorder.lock = threading.RLock()
        recorder.monitoring_capture_open = False
        recorder.navigation_plan_sequence = 0
        recorder._base = lambda stamp: {"message_time_s": stamp}
        written = []
        recorder._write_batch = lambda path, rows: written.extend(rows)
        pose = types.SimpleNamespace(pose=types.SimpleNamespace(
            position=types.SimpleNamespace(x=1.0, y=2.0),
            orientation=types.SimpleNamespace(x=0, y=0, z=0, w=1)))
        message = types.SimpleNamespace(
            header=types.SimpleNamespace(stamp=_fake_stamp(15), frame_id="map"),
            poses=[pose])
        recorder._global_plan_callback(message)
        self.assertEqual(written[0]["x_m"], 1.0)
        self.assertEqual(written[0]["point_count"], 1)
        message.poses = []
        recorder._global_plan_callback(message)
        self.assertEqual(written[1]["point_index"], -1)

    @staticmethod
    def _passing_dual_lidar_diagnostic():
        return {
            "schema_version": "5",
            "secondary_enabled": "true",
            "zero_no_return_compaction_enabled": "true",
            "point_time_mode": "snapshot",
            "configured_sensor_update_rate_hz": "10.0",
            "configured_min_range_m": "0.5",
            "configured_max_range_m": "20.0",
            "configured_ray_samples_per_sensor": "20000",
            "primary_downsample": "1",
            "secondary_downsample": "2",
            "overlap_voxel_size_m": "0.20",
            "primary_received": "130",
            "secondary_received": "130",
            "pairs_accepted": "120",
            "overlap_evaluation_pair_stride": "10",
            "overlap_evaluations_attempted": "12",
            "overlap_evaluations_failed": "0",
            "overlap_evaluation_sample_count": "12",
            "maximum_sync_delta_ns": "25000000",
            "first_accepted_measurement_time_ns": "1000000000",
            "last_accepted_measurement_time_ns": "24800000000",
            "accepted_measurement_interval_count": "119",
            "nonpositive_accepted_measurement_interval_count": "0",
            "minimum_accepted_measurement_interval_ns": "200000000",
            "maximum_accepted_measurement_interval_ns": "200000000",
            "mean_accepted_measurement_interval_ns": "200000000",
            "realized_measurement_hz": "5.0",
            "primary_queue_dropped": "0",
            "secondary_queue_dropped": "0",
            "mean_primary_overlap_fraction": "0.20",
            "mean_secondary_overlap_fraction": "0.21",
            "mean_voxel_jaccard": "0.10",
            "last_primary_point_count": "20000",
            "last_secondary_point_count": "10000",
            "last_merged_point_count": "23000",
            "last_secondary_to_primary_point_ratio": "0.5",
            "last_merged_to_primary_point_ratio": str(23000.0 / 20000.0),
            "last_total_raw_input_point_count": "30000",
            "last_compacted_output_point_count": "23000",
            "last_removed_no_return_sentinel_count": "7000",
            "last_compacted_output_to_total_raw_input_ratio": str(
                23000.0 / 30000.0),
        }

    def test_motion_truth_contract_accepts_one_complete_legal_batch(self):
        contract = RECORDER.validate_motion_truth_batch(
            _valid_motion_truth_batch())
        self.assertEqual(contract["status"], "COMPLETED")
        self.assertEqual(contract["errors"], [])
        self.assertEqual(contract["observed"]["model_names"], [
            "m01", "m02", "m03", "m04", "m05", "m06"])
        self.assertEqual(contract["observed"]["laser_retro_ids"],
                         [1, 2, 3, 4, 5, 6])

    def test_motion_truth_contract_rejects_empty_models(self):
        document = _valid_motion_truth_batch()
        document["models"] = []
        contract = RECORDER.validate_motion_truth_batch(document)
        self.assertEqual(contract["status"], "FAILED")
        self.assertIn("models_cardinality_0_expected_6", contract["errors"])
        self.assertIn("model_name_set_mismatch", contract["errors"])

    def test_motion_truth_contract_rejects_missing_model(self):
        document = _valid_motion_truth_batch()
        document["models"].pop()
        contract = RECORDER.validate_motion_truth_batch(document)
        self.assertEqual(contract["status"], "FAILED")
        self.assertIn("models_cardinality_5_expected_6", contract["errors"])
        self.assertIn("laser_retro_id_set_mismatch", contract["errors"])

    def test_motion_truth_contract_rejects_duplicate_name_or_id(self):
        document = _valid_motion_truth_batch()
        document["models"][5]["model_name"] = "m05"
        document["models"][5]["laser_retro_id"] = 5
        contract = RECORDER.validate_motion_truth_batch(document)
        self.assertEqual(contract["status"], "FAILED")
        self.assertTrue(any("model_name_duplicate_m05" in error
                            for error in contract["errors"]))
        self.assertTrue(any("laser_retro_id_duplicate_5" in error
                            for error in contract["errors"]))

    def test_motion_truth_contract_rejects_nonfinite_scalar_and_vector(self):
        document = _valid_motion_truth_batch()
        document["sim_time"] = float("nan")
        document["models"][0]["position"][2] = float("inf")
        contract = RECORDER.validate_motion_truth_batch(document)
        self.assertEqual(contract["status"], "FAILED")
        self.assertIn("sim_time_non_finite", contract["errors"])
        self.assertIn("model_0_position_non_finite", contract["errors"])

    def test_motion_truth_contract_rejects_invalid_aabb_and_quaternion(self):
        document = _valid_motion_truth_batch()
        document["models"][0]["aabb_min"][0] = 100.0
        document["models"][1]["orientation"] = [0.0, 0.0, 0.0, 2.0]
        contract = RECORDER.validate_motion_truth_batch(document)
        self.assertEqual(contract["status"], "FAILED")
        self.assertIn("model_0_aabb_not_ordered", contract["errors"])
        self.assertIn("model_1_orientation_not_unit_quaternion",
                      contract["errors"])

    def test_motion_truth_contract_uses_plugin_error_definitions(self):
        document = _valid_motion_truth_batch(20.0, "RUNNING")
        document["models"][0]["translation_error"] = 0.1
        document["models"][1]["rotation_error"] = 0.1
        contract = RECORDER.validate_motion_truth_batch(document)
        self.assertEqual(contract["status"], "FAILED")
        self.assertIn(
            "model_0_translation_error_formula_inconsistent",
            contract["errors"])
        self.assertIn(
            "model_1_rotation_error_formula_inconsistent",
            contract["errors"])

    def test_invalid_motion_truth_does_not_advance_high_water(self):
        recorder = object.__new__(RECORDER.ExperimentRecorder)
        recorder.lock = threading.RLock()
        recorder.monitoring_capture_open = True
        recorder.truth_rejected_batch_count = 0
        recorder.truth_contract_input_errors = []
        recorder.recorded_motion_experiment_elapsed_s = 7.0
        recorder._publish_timeline_status = lambda: None
        document = _valid_motion_truth_batch(10.0, "RUNNING")
        document["models"] = []
        recorder._object_truth_callback(types.SimpleNamespace(
            data=json.dumps(document)))
        self.assertEqual(recorder.recorded_motion_experiment_elapsed_s, 7.0)
        self.assertEqual(recorder.truth_rejected_batch_count, 1)

    def test_motion_truth_artifact_contract_rejects_missing_endpoint(self):
        with tempfile.TemporaryDirectory() as output:
            _write_truth_csv(output, [
                _valid_motion_truth_batch(0.0, "WAITING"),
                _valid_motion_truth_batch(20.0, "RUNNING")])
            contract = RECORDER.gazebo_truth_artifact_contract(
                output, 25.0, stationary_mode=False)
            self.assertEqual(contract["status"], "FAILED")
            for model in RECORDER.MOTION_TRUTH_MODEL_IDS:
                self.assertIn(
                    model + "_required_endpoint_not_covered",
                    contract["errors"])

    def test_motion_truth_artifact_contract_rejects_implicit_or_wrong_frame(self):
        document = _valid_motion_truth_batch(60.0, "COMPLETED")
        with tempfile.TemporaryDirectory() as output:
            _write_truth_csv(output, [document])
            path = os.path.join(output, "truth", "m01_m06_motion_truth.csv")
            with open(path, newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["truth_frame_id"] = "camera_init"
            with open(path, "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=RECORDER.CSV_SCHEMAS[
                        "truth/m01_m06_motion_truth.csv"])
                writer.writeheader()
                writer.writerows(rows)
            contract = RECORDER.gazebo_truth_artifact_contract(
                output, 60.0, stationary_mode=False)
            self.assertEqual(contract["status"], "FAILED")
            self.assertIn("batch_0_batch_scalar_truth_frame_id_not_identical",
                          contract["errors"])
            self.assertIn("batch_0_truth_frame_id_not_world",
                          contract["errors"])

    def test_missing_truth_cannot_shrink_registered_recall_denominator(self):
        primary = ("m01", "m02", "m03", "m05", "m06")
        model_rows = [{
            "model_name": model, "truth_active": True, "detected": True,
        } for model in primary if model != "m06"]
        recall = RECORDER.registered_primary_model_recall(
            model_rows, primary)
        self.assertEqual(recall["denominator_model_count"], 5)
        self.assertEqual(recall["false_negative_models"], ["m06"])
        self.assertAlmostEqual(recall["object_recall"], 4.0 / 5.0)

        incomplete = _valid_motion_truth_batch(60.0, "COMPLETED")
        incomplete["models"] = [model for model in incomplete["models"]
                                if model["model_name"] != "m06"]
        with tempfile.TemporaryDirectory() as output:
            _write_truth_csv(output, [incomplete])
            contract = RECORDER.gazebo_truth_artifact_contract(
                output, 60.0, stationary_mode=False)
            self.assertEqual(contract["status"], "FAILED")
            self.assertIn("m06_required_endpoint_not_covered",
                          contract["errors"])

    def test_stationary_motion_truth_artifact_covers_all_boundaries(self):
        documents = [
            _valid_motion_truth_batch(0.0, "WAITING"),
            _valid_motion_truth_batch(9.9, "WAITING"),
            _valid_motion_truth_batch(10.0, "RUNNING"),
            _valid_motion_truth_batch(49.9, "RUNNING"),
            _valid_motion_truth_batch(50.0, "HOLDING"),
            _valid_motion_truth_batch(59.9, "HOLDING"),
            _valid_motion_truth_batch(60.0, "COMPLETED"),
        ]
        with tempfile.TemporaryDirectory() as output:
            _write_truth_csv(output, documents)
            contract = RECORDER.gazebo_truth_artifact_contract(
                output, 60.0, stationary_mode=True)
            self.assertEqual(contract["status"], "COMPLETED")
            self.assertEqual(contract["errors"], [])
            self.assertEqual(contract["valid_complete_batch_count"], 7)
            for model in RECORDER.MOTION_TRUTH_MODEL_IDS:
                self.assertTrue(contract["per_model"][model][
                    "required_endpoint_satisfied"])
                self.assertEqual(set(contract["per_model"][model][
                    "stationary_checkpoint_coverage"]),
                    {"0", "10", "50", "60"})

    def test_dual_lidar_formal_success_contract_is_strict_and_complete(self):
        contract = RECORDER.dual_lidar_success_contract(
            self._passing_dual_lidar_diagnostic())
        self.assertEqual(contract["status"], "COMPLETED")
        self.assertEqual(contract["errors"], [])
        self.assertEqual(
            contract["observed"][
                "expected_overlap_attempts_floor_pairs_div_stride"], 12)
        self.assertAlmostEqual(
            contract["observed"]["primary_pairing_rate"], 120.0 / 130.0)
        self.assertEqual(
            contract["observed"]["accepted_measurement_interval_count"],
            119)
        self.assertAlmostEqual(
            contract["observed"]["realized_measurement_hz"], 5.0)
        self.assertAlmostEqual(
            contract["observed"]["derived_realized_measurement_hz"], 5.0)
        self.assertAlmostEqual(
            contract["thresholds"]["configured_sensor_update_rate_hz"],
            10.0)
        self.assertAlmostEqual(
            contract["thresholds"]["minimum_realized_measurement_hz"], 4.0)
        self.assertAlmostEqual(
            contract["thresholds"]["maximum_realized_measurement_hz"], 5.5)
        self.assertAlmostEqual(
            contract["thresholds"][
                "minimum_accepted_measurement_interval_s"], 0.10)
        self.assertAlmostEqual(
            contract["thresholds"][
                "maximum_accepted_measurement_interval_s"], 0.35)

        failures = {
            "schema_version": "4",
            "zero_no_return_compaction_enabled": "false",
            "point_time_mode": "legacy_rolling",
            "configured_sensor_update_rate_hz": "9.9",
            "configured_min_range_m": "0.4",
            "configured_max_range_m": "20.1",
            "configured_ray_samples_per_sensor": "19999",
            "primary_downsample": "2",
            "secondary_downsample": "1",
            "overlap_voxel_size_m": "0.21",
            "overlap_evaluation_sample_count": "9",
            "overlap_evaluations_failed": "1",
            "overlap_evaluations_attempted": "11",
            "pairs_accepted": "119",
            "maximum_sync_delta_ns": "25000001",
            "first_accepted_measurement_time_ns": "24800000000",
            "last_accepted_measurement_time_ns": "1000000000",
            "accepted_measurement_interval_count": "118",
            "nonpositive_accepted_measurement_interval_count": "1",
            "minimum_accepted_measurement_interval_ns": "99999999",
            "maximum_accepted_measurement_interval_ns": "350000001",
            "mean_accepted_measurement_interval_ns": "200000001",
            "realized_measurement_hz": "3.9",
            "primary_queue_dropped": "1",
            "secondary_queue_dropped": "1",
            "primary_received": "134",
            "secondary_received": "134",
            "mean_primary_overlap_fraction": "0.199",
            "mean_secondary_overlap_fraction": "0.199",
            "mean_voxel_jaccard": "0.099",
            "last_primary_point_count": "19999",
            "last_secondary_point_count": "9999",
            "last_merged_point_count": "22999",
            "last_secondary_to_primary_point_ratio": "0.49",
            "last_merged_to_primary_point_ratio": "1.14",
            "last_total_raw_input_point_count": "29999",
            "last_compacted_output_point_count": "23001",
            "last_removed_no_return_sentinel_count": "6999",
            "last_compacted_output_to_total_raw_input_ratio": "0.7",
        }
        for key, value in failures.items():
            diagnostic = self._passing_dual_lidar_diagnostic()
            diagnostic[key] = value
            self.assertEqual(
                RECORDER.dual_lidar_success_contract(diagnostic)["status"],
                "FAILED", key)
        for mean_interval_ns, last_time_ns, realized_measurement_hz in (
                (300000000, 36700000000, 10.0 / 3.0),
                (160000000, 20040000000, 6.25)):
            diagnostic = self._passing_dual_lidar_diagnostic()
            diagnostic.update({
                "last_accepted_measurement_time_ns": str(last_time_ns),
                "mean_accepted_measurement_interval_ns": str(
                    mean_interval_ns),
                "minimum_accepted_measurement_interval_ns": str(
                    mean_interval_ns),
                "maximum_accepted_measurement_interval_ns": str(
                    mean_interval_ns),
                "realized_measurement_hz": str(realized_measurement_hz),
            })
            contract = RECORDER.dual_lidar_success_contract(diagnostic)
            self.assertTrue(contract["checks"][
                "realized_measurement_hz_positive_and_consistent"])
            self.assertFalse(contract["checks"][
                "realized_measurement_hz_within_configured_bounds"])
            self.assertEqual(contract["status"], "FAILED")

    def test_dual_lidar_contract_tracks_configurable_two_two_stride(self):
        diagnostic = self._passing_dual_lidar_diagnostic()
        diagnostic.update({
            "primary_downsample": "2",
            "secondary_downsample": "2",
            "last_primary_point_count": "10000",
            "last_secondary_point_count": "10000",
            "last_merged_point_count": "16000",
            "last_secondary_to_primary_point_ratio": "1.0",
            "last_merged_to_primary_point_ratio": "1.6",
            "last_total_raw_input_point_count": "20000",
            "last_compacted_output_point_count": "16000",
            "last_removed_no_return_sentinel_count": "4000",
            "last_compacted_output_to_total_raw_input_ratio": "0.8",
        })
        thresholds = RECORDER.configured_dual_lidar_thresholds(
            10.0, 0.5, 20.0, 20000, 2, 2)
        contract = RECORDER.dual_lidar_success_contract(
            diagnostic, thresholds)
        self.assertEqual(contract["status"], "COMPLETED")
        self.assertEqual(
            contract["thresholds"]["expected_primary_point_count"], 10000)
        self.assertEqual(
            contract["thresholds"]["expected_secondary_point_count"], 10000)
        self.assertEqual(
            contract["thresholds"]["expected_total_raw_input_point_count"],
            20000)
        self.assertEqual(
            contract["thresholds"]["maximum_compacted_output_point_count"],
            20000)

    def test_gazebo_performance_summary_is_observational_not_a_gate(self):
        rows = [
            {"message_sequence": "1", "message_time_s": "10.0",
             "real_time_factor": "0.40", "sensor_name": "front",
             "sim_update_rate_hz": "5.0", "real_update_rate_hz": "2.0",
             "fps": "2.0"},
            {"message_sequence": "1", "message_time_s": "10.0",
             "real_time_factor": "0.40", "sensor_name": "rear",
             "sim_update_rate_hz": "5.0", "real_update_rate_hz": "2.0",
             "fps": "2.0"},
            {"message_sequence": "2", "message_time_s": "11.0",
             "real_time_factor": "0.60", "sensor_name": "front",
             "sim_update_rate_hz": "5.0", "real_update_rate_hz": "3.0",
             "fps": "3.0"},
        ]
        summary = RECORDER.summarize_gazebo_performance(rows)
        self.assertEqual(summary["message_sample_count"], 2)
        self.assertAlmostEqual(summary["real_time_factor"]["mean"], 0.5)
        self.assertAlmostEqual(
            summary["sensors"]["front"]["real_update_rate_hz"]["mean"],
            2.5)
        self.assertFalse(summary["acceptance_gate"])

    def test_finalization_waits_for_diagnostic_sequence_after_request(self):
        recorder = object.__new__(RECORDER.ExperimentRecorder)
        recorder.lock = threading.RLock()
        recorder.dual_lidar_condition = threading.Condition(recorder.lock)
        recorder.require_dual_lidar_success_contract = True
        recorder.dual_lidar_diagnostic_sequence = 7
        recorder.dual_lidar_request_sequence = 7
        recorder.dual_lidar_final_sequence = None
        recorder.dual_lidar_request_monotonic_s = time.monotonic()
        recorder.dual_lidar_latest_receive_monotonic_s = (
            recorder.dual_lidar_request_monotonic_s - 1.0)
        recorder.dual_lidar_latest_receive_wall_time_s = time.time() - 1.0
        recorder.dual_lidar_final_receive_monotonic_s = None
        recorder.dual_lidar_final_receive_wall_time_s = None

        def publish_fresh_diagnostic():
            time.sleep(0.01)
            with recorder.dual_lidar_condition:
                recorder.dual_lidar_diagnostic_sequence = 8
                recorder.dual_lidar_latest_receive_monotonic_s = (
                    time.monotonic())
                recorder.dual_lidar_latest_receive_wall_time_s = time.time()
                recorder.dual_lidar_condition.notify_all()

        worker = threading.Thread(target=publish_fresh_diagnostic)
        worker.start()
        recorder._wait_for_post_request_dual_lidar_diagnostic(
            time.monotonic() + 0.5)
        worker.join()
        self.assertEqual(recorder.dual_lidar_final_sequence, 8)
        self.assertGreater(
            recorder.dual_lidar_final_receive_monotonic_s,
            recorder.dual_lidar_request_monotonic_s)

    def test_pre_request_callback_writing_late_does_not_satisfy_freshness(self):
        recorder = object.__new__(RECORDER.ExperimentRecorder)
        recorder.lock = threading.RLock()
        recorder.dual_lidar_condition = threading.Condition(recorder.lock)
        recorder.require_dual_lidar_success_contract = True
        # Sequence advanced after the request, but the callback's receive time
        # proves it began before the request and merely completed its write
        # late. This was the sequence-only race.
        recorder.dual_lidar_diagnostic_sequence = 8
        recorder.dual_lidar_request_sequence = 7
        recorder.dual_lidar_final_sequence = None
        recorder.dual_lidar_request_monotonic_s = time.monotonic()
        recorder.dual_lidar_latest_receive_monotonic_s = (
            recorder.dual_lidar_request_monotonic_s - 0.1)
        recorder.dual_lidar_latest_receive_wall_time_s = time.time() - 0.1
        recorder.dual_lidar_final_receive_monotonic_s = None
        recorder.dual_lidar_final_receive_wall_time_s = None
        with self.assertRaisesRegex(
                RuntimeError, "dual_lidar_fresh_diagnostic"):
            recorder._wait_for_post_request_dual_lidar_diagnostic(
                time.monotonic() + 0.01)

    def test_dual_diagnostic_freshness_advances_only_after_successful_write(self):
        recorder = object.__new__(RECORDER.ExperimentRecorder)
        recorder.lock = threading.RLock()
        recorder.dual_lidar_condition = threading.Condition(recorder.lock)
        recorder.dual_lidar_diagnostic_sequence = 0
        recorder.dual_lidar_latest_receive_monotonic_s = None
        recorder.dual_lidar_latest_receive_wall_time_s = None
        recorder.start_wall = time.monotonic()
        recorder.state = "MONITORING"
        message = types.SimpleNamespace(data=json.dumps(
            self._passing_dual_lidar_diagnostic()))
        recorder._write = lambda *args, **kwargs: False
        recorder._dual_lidar_diagnostics_callback(message)
        self.assertEqual(recorder.dual_lidar_diagnostic_sequence, 0)
        self.assertIsNone(recorder.dual_lidar_latest_receive_monotonic_s)
        recorder._write = lambda *args, **kwargs: True
        recorder._dual_lidar_diagnostics_callback(message)
        self.assertEqual(recorder.dual_lidar_diagnostic_sequence, 1)
        self.assertIsNotNone(recorder.dual_lidar_latest_receive_monotonic_s)

    def test_checkpoint_contract_requires_one_current_map_and_cloud_each(self):
        with tempfile.TemporaryDirectory() as output:
            event_directory = os.path.join(output, "events")
            map_directory = os.path.join(output, "snapshots", "maps")
            cloud_directory = os.path.join(output, "snapshots", "clouds")
            os.makedirs(event_directory)
            os.makedirs(map_directory)
            os.makedirs(cloud_directory)
            timeline = os.path.join(event_directory,
                                    "timeline_checkpoints.csv")
            with open(timeline, "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=(
                        "target_experiment_elapsed_s", "snapshot_label",
                        "avoidance_source_snapshot_id",
                        "avoidance_source_accepted_frame_index",
                        "avoidance_map_revision",
                        "avoidance_forecast_horizons_s",
                        "avoidance_source_stamp_s",
                        "avoidance_current_forecast_horizon_s",
                        "avoidance_current_map_stamp_s",
                        "avoidance_current_map_frame_id",
                        "detector_input_cloud_stamp_s",
                        "detector_input_cloud_frame_id"))
                writer.writeheader()
                for checkpoint in (0.0, 10.0, 50.0, 60.0):
                    index = (0.0, 10.0, 50.0, 60.0).index(checkpoint) + 1
                    writer.writerow({
                        "target_experiment_elapsed_s": checkpoint,
                        "snapshot_label": "CHECKPOINT_{:06.2f}s".format(
                            checkpoint),
                        "avoidance_source_snapshot_id": "snap-{}".format(index),
                        "avoidance_source_accepted_frame_index": 100 + index,
                        "avoidance_map_revision": 7,
                        "avoidance_forecast_horizons_s": (
                            "0.5|1|2" if checkpoint == 50.0 else ""),
                        "avoidance_source_stamp_s": 20.0 + checkpoint,
                        "avoidance_current_forecast_horizon_s": "",
                        "avoidance_current_map_stamp_s": 20.0 + checkpoint,
                        "avoidance_current_map_frame_id": "camera_init",
                        "detector_input_cloud_stamp_s": 20.01 + checkpoint,
                        "detector_input_cloud_frame_id": "camera_init",
                    })
            for index, checkpoint in enumerate((0.0, 10.0, 50.0, 60.0), 1):
                safe = "CHECKPOINT_{:06.2f}s".format(checkpoint).replace(
                    ".", "_")
                np.savez_compressed(os.path.join(
                    map_directory,
                    "{:02d}_{}_avoidance_overlay.npz".format(index, safe)),
                    data=np.zeros((2, 2), dtype=np.int16),
                    source_snapshot_id=np.asarray("snap-{}".format(index)),
                    source_accepted_frame_index=np.asarray(100 + index),
                    map_revision=np.asarray(7),
                    forecast_horizon_s=np.asarray(""),
                    source_stamp_s=np.asarray(20.0 + checkpoint),
                    stamp_s=np.asarray(20.0 + checkpoint),
                    frame_id=np.asarray("camera_init"))
                np.savez_compressed(os.path.join(
                    cloud_directory,
                    "{:02d}_{}_detector_input.npz".format(index, safe)),
                    points=np.zeros((1, 3), dtype=np.float32),
                    stamp_s=np.asarray(20.01 + checkpoint),
                    frame_id=np.asarray("camera_init"))
                if checkpoint == 50.0:
                    for horizon, name in (
                            (0.5, "avoidance_forecast_0500ms"),
                            (1.0, "avoidance_forecast_1000ms"),
                            (2.0, "avoidance_forecast_2000ms")):
                        np.savez_compressed(os.path.join(
                            map_directory,
                            "{:02d}_{}_{}.npz".format(index, safe, name)),
                            data=np.zeros((2, 2), dtype=np.int16),
                            source_snapshot_id=np.asarray(
                                "snap-{}".format(index)),
                            source_accepted_frame_index=np.asarray(100 + index),
                            map_revision=np.asarray(7),
                            forecast_horizon_s=np.asarray(horizon),
                            source_stamp_s=np.asarray(20.0 + checkpoint),
                            stamp_s=np.asarray(20.0 + checkpoint),
                            frame_id=np.asarray("camera_init"))
            contract = RECORDER.checkpoint_artifact_contract(
                output, (0.0, 10.0, 50.0, 60.0))
            self.assertEqual(contract["status"], "COMPLETED")
            self.assertEqual(len(contract["required_paths"]), 12)
            self.assertFalse(any(
                "forecast" in key
                for key in contract["checkpoints"][0]["artifacts"]))
            mismatched_map = os.path.join(
                map_directory,
                "02_CHECKPOINT_010_00s_avoidance_overlay.npz")
            with np.load(mismatched_map, allow_pickle=False) as stored:
                arrays = {key: stored[key] for key in stored.files}
            arrays["map_revision"] = np.asarray(8)
            np.savez_compressed(mismatched_map, **arrays)
            mismatch = RECORDER.checkpoint_artifact_contract(
                output, (0.0, 10.0, 50.0, 60.0))
            self.assertIn(
                "checkpoint_10.00_maps_avoidance_overlay_map_revision_mismatch",
                mismatch["errors"])
            arrays["map_revision"] = np.asarray(7)
            np.savez_compressed(mismatched_map, **arrays)
            forecast = os.path.join(
                map_directory,
                "03_CHECKPOINT_050_00s_avoidance_forecast_1000ms.npz")
            with np.load(forecast, allow_pickle=False) as stored:
                forecast_arrays = {key: stored[key] for key in stored.files}
            forecast_arrays["forecast_horizon_s"] = np.asarray(1.1)
            np.savez_compressed(forecast, **forecast_arrays)
            horizon_mismatch = RECORDER.checkpoint_artifact_contract(
                output, (0.0, 10.0, 50.0, 60.0))
            self.assertIn(
                "checkpoint_50.00_maps_avoidance_forecast_1000ms_"
                "forecast_horizon_s_mismatch",
                horizon_mismatch["errors"])
            forecast_arrays["forecast_horizon_s"] = np.asarray(1.0)
            forecast_arrays["source_stamp_s"] = np.asarray(69.5)
            np.savez_compressed(forecast, **forecast_arrays)
            forecast_mismatch = RECORDER.checkpoint_artifact_contract(
                output, (0.0, 10.0, 50.0, 60.0))
            self.assertIn(
                "checkpoint_50.00_maps_avoidance_forecast_1000ms_"
                "source_stamp_s_mismatch",
                forecast_mismatch["errors"])
            forecast_arrays["source_stamp_s"] = np.asarray(70.0)
            np.savez_compressed(forecast, **forecast_arrays)
            os.unlink(forecast)
            forecast_missing = RECORDER.checkpoint_artifact_contract(
                output, (0.0, 10.0, 50.0, 60.0))
            self.assertIn(
                "checkpoint_50.00_maps_avoidance_forecast_1000ms_"
                "artifact_count=0",
                forecast_missing["errors"])
            np.savez_compressed(forecast, **forecast_arrays)
            missing = os.path.join(
                cloud_directory,
                "03_CHECKPOINT_050_00s_detector_input.npz")
            os.unlink(missing)
            failed = RECORDER.checkpoint_artifact_contract(
                output, (0.0, 10.0, 50.0, 60.0))
            self.assertEqual(failed["status"], "FAILED")
            self.assertIn(
                "checkpoint_50.00_clouds_detector_input_artifact_count=0",
                failed["errors"])

    def test_checkpoint_missing_cloud_remains_pending_for_retry(self):
        recorder = object.__new__(RECORDER.ExperimentRecorder)
        recorder.lock = __import__("threading").RLock()
        recorder.snapshotted_states = set()
        recorder.snapshot_in_progress_states = set()
        recorder.snapshot_metadata_by_state = {}
        recorder.latest_maps = {"avoidance_overlay": _fake_grid(10)}
        recorder.latest_map_metadata = {"avoidance_overlay": {}}
        recorder.latest_clouds = {}
        metadata, committed = recorder._snapshot_phase(
            "CHECKPOINT_010.00s")
        self.assertEqual(metadata, {})
        self.assertFalse(committed)
        self.assertNotIn("CHECKPOINT_010.00s", recorder.snapshotted_states)

    def test_checkpoint_timeline_retry_reuses_original_product_metadata(self):
        recorder = object.__new__(RECORDER.ExperimentRecorder)
        recorder.lock = __import__("threading").RLock()
        recorder.snapshotted_states = set()
        recorder.snapshot_in_progress_states = set()
        recorder.snapshot_metadata_by_state = {}
        recorder.latest_maps = {"avoidance_overlay": _fake_grid(10, 12.0)}
        recorder.latest_map_metadata = {"avoidance_overlay": {
            "stamp_s": 12.0, "frame_id": "camera_init",
            "source_snapshot_id": "snapshot-17",
            "source_accepted_frame_index": 17,
            "map_revision": 4, "forecast_horizon_s": "",
        }}
        cloud = types.SimpleNamespace(header=types.SimpleNamespace(
            stamp=_fake_stamp(12.01), frame_id="camera_init"))
        recorder.latest_clouds = {"detector_input": cloud}
        recorder._save_grid = lambda *args, **kwargs: "/tmp/grid.npz"
        recorder._save_cloud = lambda *args, **kwargs: "/tmp/cloud.npz"
        first, committed = recorder._snapshot_phase("CHECKPOINT_010.00s")
        self.assertTrue(committed)
        # This second call models retry after the first timeline CSV _write
        # failed. It must not return the old empty-metadata shortcut.
        second, committed = recorder._snapshot_phase("CHECKPOINT_010.00s")
        self.assertTrue(committed)
        self.assertEqual(second, first)
        products = recorder.snapshot_metadata_by_state[
            "CHECKPOINT_010.00s"]
        self.assertEqual(
            products["maps"]["avoidance_overlay"]["source_snapshot_id"],
            "snapshot-17")
        self.assertAlmostEqual(
            products["clouds"]["detector_input"]["stamp_s"], 12.01)
        self.assertEqual(
            RECORDER.summarize_checkpoint_products(products)[
                "detector_input_cloud_frame_id"], "camera_init")

    def test_formal_launch_enables_dual_lidar_success_contract(self):
        launch_path = os.path.abspath(os.path.join(
            SCRIPTS, "..", "launch", "icra2027_closed_loop.launch"))
        with open(launch_path, encoding="utf-8") as stream:
            launch = stream.read()
        self.assertIn(
            '<param name="require_dual_lidar_success_contract" value="true"',
            launch)

    def test_stationary_forecast_artifact_contract_requires_real_products(self):
        paths = RECORDER.STATIONARY_FORECAST_REQUIRED_PATHS
        self.assertIn(
            "analysis/deformation_route_maps/03_forecast_anchor_current_overlay.png",
            paths)
        self.assertIn(
            "analysis/deformation_route_maps/14_runtime_forecast_self_consistency.png",
            paths)
        self.assertIn(
            "analysis/deformation_route_maps/all_map_arrays.npz", paths)
        self.assertFalse(any("unavailable" in path for path in paths))
        self.assertEqual(len(paths), len(set(paths)))

    def test_avoidance_snapshot_expands_atomic_forecast_bundle(self):
        maps, metadata = RECORDER.avoidance_snapshot_layers(
            _fake_avoidance())
        self.assertEqual(set(maps), {
            "avoidance_overlay",
            "avoidance_forecast_0500ms",
            "avoidance_forecast_1000ms",
            "avoidance_forecast_2000ms",
        })
        self.assertEqual(set(maps), set(metadata))
        identities = {
            (entry["source_snapshot_id"],
             entry["source_accepted_frame_index"],
             entry["map_revision"])
            for entry in metadata.values()}
        self.assertEqual(identities, {("snapshot-17", 17, 4)})
        self.assertEqual(
            metadata["avoidance_forecast_1000ms"]["forecast_horizon_s"],
            1.0)

    def test_avoidance_snapshot_rejects_cardinality_and_geometry_mismatch(self):
        message = _fake_avoidance()
        message.forecast_overlays.pop()
        with self.assertRaises(ValueError):
            RECORDER.avoidance_snapshot_layers(message)
        message = _fake_avoidance()
        message.forecast_overlays[1].info.width = 3
        with self.assertRaises(ValueError):
            RECORDER.avoidance_snapshot_layers(message)
        message = _fake_avoidance()
        message.current_overlay.header.stamp = _fake_stamp(12.01)
        with self.assertRaises(ValueError):
            RECORDER.avoidance_snapshot_layers(message)
        message = _fake_avoidance()
        message.forecast_overlays[0].header.stamp = _fake_stamp(12.01)
        with self.assertRaises(ValueError):
            RECORDER.avoidance_snapshot_layers(message)

    def test_invalid_avoidance_bundle_commits_nothing(self):
        recorder = object.__new__(RECORDER.ExperimentRecorder)
        recorder.monitoring_capture_open = True
        recorder.latest_maps = {"sentinel": object()}
        recorder.latest_map_metadata = {"sentinel": {}}
        recorder.recorded_avoidance_frame_index = 16
        writes = []
        recorder._write = lambda *args, **kwargs: writes.append(args) or True
        invalid = _fake_avoidance(frame_index=17)
        invalid.forecast_overlays.pop()
        recorder._avoidance_callback(invalid)
        self.assertEqual(writes, [])
        self.assertEqual(set(recorder.latest_maps), {"sentinel"})
        self.assertEqual(set(recorder.latest_map_metadata), {"sentinel"})
        self.assertEqual(recorder.recorded_avoidance_frame_index, 16)

    def test_current_only_snapshot_removes_stale_forecast_layers(self):
        latest_maps, latest_metadata = RECORDER.avoidance_snapshot_layers(
            _fake_avoidance(snapshot_id="old", frame_index=16))
        current_maps, current_metadata = RECORDER.avoidance_snapshot_layers(
            _fake_avoidance(snapshot_id="new", frame_index=17,
                            horizons=()))
        RECORDER.replace_avoidance_snapshot_layers(
            latest_maps, latest_metadata, current_maps, current_metadata)
        self.assertEqual(set(latest_maps), {"avoidance_overlay"})
        self.assertEqual(set(latest_metadata), {"avoidance_overlay"})
        self.assertEqual(
            latest_metadata["avoidance_overlay"]["source_snapshot_id"],
            "new")

    def test_saved_grid_preserves_snapshot_identity_and_horizon(self):
        with tempfile.TemporaryDirectory() as directory:
            os.makedirs(os.path.join(directory, "snapshots", "maps"))
            recorder = object.__new__(RECORDER.ExperimentRecorder)
            recorder.output = directory
            recorder.counts = {"map_snapshots": 0}
            grid = _fake_grid(25)
            metadata = {
                "source_snapshot_id": "snapshot-17",
                "source_accepted_frame_index": 17,
                "source_stamp_s": 12.0,
                "map_revision": 4,
                "forecast_horizon_s": 1.0,
            }
            recorder._save_grid("01_TEST", "avoidance_forecast_1000ms",
                                grid, metadata)
            path = os.path.join(
                directory, "snapshots", "maps",
                "01_TEST_avoidance_forecast_1000ms.npz")
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(str(saved["source_snapshot_id"]),
                                 "snapshot-17")
                self.assertEqual(int(saved["source_accepted_frame_index"]), 17)
                self.assertEqual(float(saved["forecast_horizon_s"]), 1.0)

    def test_checkpoint_summary_exposes_forecast_bundle_identity(self):
        _, metadata = RECORDER.avoidance_snapshot_layers(_fake_avoidance())
        summary = RECORDER.summarize_map_ages(12.5, metadata)
        self.assertEqual(summary["avoidance_source_snapshot_id"],
                         "snapshot-17")
        self.assertEqual(summary["avoidance_forecast_horizons_s"],
                         "0.5|1|2")
        self.assertEqual(summary["avoidance_source_accepted_frame_index"], 17)

    def test_required_artifact_contract_validates_content_and_containment(self):
        with tempfile.TemporaryDirectory() as directory:
            valid_json = os.path.join(directory, "summary.json")
            valid_csv = os.path.join(directory, "samples.csv")
            valid_png = os.path.join(directory, "plot.png")
            valid_npz = os.path.join(directory, "snapshots", "maps",
                                     "checkpoint.npz")
            os.makedirs(os.path.dirname(valid_npz))
            with open(valid_json, "w", encoding="utf-8") as stream:
                json.dump({"status": "COMPLETED"}, stream)
            with open(valid_csv, "w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("stamp", "value"))
                writer.writerow((1, 2))
            with open(valid_png, "wb") as stream:
                stream.write(b"\x89PNG\r\n\x1a\ncontent")
            np.savez_compressed(valid_npz,
                                data=np.zeros((2, 2), dtype=np.int16))
            self.assertEqual(RECORDER.validate_required_artifacts(
                directory, ("summary.json", "samples.csv", "plot.png",
                            "snapshots/maps/checkpoint.npz")), [])

            with open(valid_json, "w", encoding="utf-8") as stream:
                stream.write("not JSON")
            with open(valid_csv, "w", encoding="utf-8") as stream:
                stream.write("header_only\n")
            with open(valid_png, "wb") as stream:
                stream.write(b"not-a-png")
            errors = RECORDER.validate_required_artifacts(directory, (
                "summary.json", "samples.csv", "plot.png", "missing.csv",
                "../escape.json"))
            joined = "\n".join(errors)
            self.assertIn("invalid required path: ../escape.json", joined)
            self.assertIn("missing: missing.csv", joined)
            self.assertIn("invalid: summary.json", joined)
            self.assertIn("CSV has no research data row", joined)
            self.assertIn("invalid PNG signature", joined)

    def test_finalization_high_water_waits_for_every_required_stream(self):
        required = {
            "motion_experiment_elapsed_s": 60.0,
            "detector_accepted_frame_index": 450,
            "evolution_accepted_frame_index": 450,
            "avoidance_accepted_frame_index": 450,
            "navigation_stamp_s": 1200.0,
            "robot_truth_stamp_s": 1200.0,
        }
        recorded = dict(required)
        checks = RECORDER.finalization_high_water_checks(
            required, recorded, True, True, 0)
        self.assertTrue(all(checks.values()))
        for key in required:
            delayed = dict(recorded)
            delayed[key] = required[key] - 1.0
            checks = RECORDER.finalization_high_water_checks(
                required, delayed, True, True, 0)
            self.assertFalse(all(checks.values()), key)
        self.assertFalse(all(RECORDER.finalization_high_water_checks(
            required, recorded, True, False, 0).values()))
        self.assertFalse(all(RECORDER.finalization_high_water_checks(
            required, recorded, True, True, 1).values()))

    def test_recorder_derives_robot_truth_high_water_from_navigation(self):
        recorder = object.__new__(RECORDER.ExperimentRecorder)
        recorder.lock = threading.RLock()
        recorder.recorded_motion_experiment_elapsed_s = None
        recorder.recorded_detector_frame_index = -1
        recorder.recorded_evolution_frame_index = -1
        recorder.recorded_avoidance_frame_index = -1
        recorder.recorded_navigation_stamp_s = 120.0
        recorder.recorded_robot_truth_stamp_s = 119.0
        recorder.freeze_on_detection_state = False
        recorder.freeze_on_monitoring_complete = False
        recorder.capture_freeze_committed = False
        recorder.pending_timeline_checkpoints_s = set()
        ready, missing = recorder._finalization_high_water({
            "required_navigation_stamp_s": 120.0,
        })
        self.assertFalse(ready)
        self.assertIn("robot_truth", missing)
        recorder.recorded_robot_truth_stamp_s = 120.0
        ready, missing = recorder._finalization_high_water({
            "required_navigation_stamp_s": 120.0,
        })
        self.assertTrue(ready)
        self.assertEqual(missing, [])

    def test_finalization_request_rejects_stale_or_corrupt_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = object.__new__(RECORDER.ExperimentRecorder)
            recorder.output = directory
            summary_path = os.path.join(
                directory, "mission_decision_summary.json")
            with open(summary_path, "w", encoding="utf-8") as stream:
                json.dump({"run_session_id": "run-a",
                           "outcome": "COMPLETED"}, stream)
            request = {
                "request_id": "run-a",
                "run_session_id": "run-a",
                "mission_outcome": "COMPLETED",
                "output_directory": directory,
                "decision_summary_relative_path": (
                    "mission_decision_summary.json"),
                "decision_summary_sha256": RECORDER.ExperimentRecorder._sha256(
                    summary_path),
            }
            recorder._validate_finalization_request(request)
            corrupt = dict(request)
            corrupt["decision_summary_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                recorder._validate_finalization_request(corrupt)
            stale = dict(request)
            stale["request_id"] = "run-old"
            with self.assertRaises(ValueError):
                recorder._validate_finalization_request(stale)

    def test_world_map_transform_round_trip(self):
        transform = RECORDER.map_from_world_anchor(
            (22.2, 6.7, 0.0), (-3.9, 12.4, 2.07))
        point = (-2.1, 9.4)
        mapped = RECORDER.transform_point(transform, point)
        recovered = RECORDER.inverse_transform_point(transform, mapped)
        np.testing.assert_allclose(recovered, point, atol=1.0e-12)

    def test_vector_metrics_preserve_direction_and_magnitude_error(self):
        estimate_magnitude, truth_magnitude, error, cosine = (
            RECORDER.vector_metrics((0.0, 0.009, 0.0),
                                    (0.0, 0.010, 0.0)))
        self.assertAlmostEqual(estimate_magnitude, 0.009)
        self.assertAlmostEqual(truth_magnitude, 0.010)
        self.assertAlmostEqual(error, 0.001)
        self.assertAlmostEqual(cosine, 1.0)

    def test_body_calibration_is_composed_in_full_se3(self):
        half_yaw = 0.25 * math.pi
        position, orientation = RECORDER.compose_body_pose_se3(
            (1.0, 2.0, 3.0),
            (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)),
            (0.3, 0.0, 0.6, 0.0))
        np.testing.assert_allclose(position, (1.0, 2.3, 3.6), atol=1.0e-12)
        np.testing.assert_allclose(
            orientation,
            (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)),
            atol=1.0e-12)

    def test_model_association_is_unique_and_height_aware(self):
        models = {"m02": (0.0, 9.0, 3.0), "m06": (-2.5, 9.0, 5.3)}
        tolerances = {"m02": (2.6, 1.8), "m06": (3.2, 1.4)}
        self.assertEqual(RECORDER.associate_model(
            (-2.0, 9.0, 5.2), models, tolerances), "m06")
        self.assertEqual(RECORDER.associate_model(
            (0.1, 9.1, 3.1), models, tolerances), "m02")
        self.assertIsNone(RECORDER.associate_model(
            (8.0, 8.0, 8.0), models, tolerances))

    def test_evaluation_association_prefers_aabb_and_withholds_ambiguity(self):
        geometries = {
            "m01": {"position": (0.0, 0.0, 0.0),
                    "aabb_min": (-1.0, -1.0, -1.0),
                    "aabb_max": (1.0, 1.0, 1.0)},
            "m02": {"position": (2.0, 0.0, 0.0),
                    "aabb_min": (1.1, -1.0, -1.0),
                    "aabb_max": (3.0, 1.0, 1.0)},
        }
        tolerances = {"m01": (5.0, 5.0), "m02": (5.0, 5.0)}
        matched = RECORDER.associate_model_for_evaluation(
            (-0.8, 0.0, 0.0), geometries, tolerances, 0.15)
        self.assertEqual(matched["status"], "MATCHED")
        self.assertEqual(matched["model_name"], "m01")
        self.assertEqual(matched["method"], "WORLD_AABB")
        ambiguous = RECORDER.associate_model_for_evaluation(
            (1.05, 0.0, 0.0), geometries, tolerances, 0.15)
        self.assertEqual(ambiguous["status"], "MARGIN_ONLY_UNCERTAIN")
        self.assertEqual(ambiguous["candidates"], ["m01", "m02"])
        unmatched = RECORDER.associate_model_for_evaluation(
            (20.0, 0.0, 0.0), geometries, tolerances, 0.15)
        self.assertEqual(unmatched["status"], "UNMATCHED")

        margin_only = RECORDER.associate_model_for_evaluation(
            (1.08, 1.08, 0.0), {"m01": geometries["m01"]},
            tolerances, 0.15)
        self.assertEqual(margin_only["status"], "MARGIN_ONLY_UNCERTAIN")
        self.assertEqual(margin_only["model_name"], "")
        self.assertEqual(margin_only["candidates"], ["m01"])
        self.assertEqual(margin_only["method"], "WORLD_AABB_MARGIN")

        # Exact containment is not made ambiguous by the soft margin of an
        # adjacent box.
        exact_over_margin = RECORDER.associate_model_for_evaluation(
            (0.99, 0.0, 0.0), geometries, tolerances, 0.15)
        self.assertEqual(exact_over_margin["status"], "MATCHED")
        self.assertEqual(exact_over_margin["model_name"], "m01")

    def test_evaluation_association_has_explicit_bounded_fallback(self):
        result = RECORDER.associate_model_for_evaluation(
            (0.1, 0.2, 0.3), {"m03": {"position": (0.0, 0.0, 0.0)}},
            {"m03": (1.0, 1.0)}, 0.1)
        self.assertEqual(result["status"], "MATCHED")
        self.assertEqual(result["method"], "BOUNDED_ELLIPSOID")

    def test_model_truth_alignment_interpolates_se3_at_message_time(self):
        half_turn = (0.0, 0.0, 1.0, 0.0)
        history = [
            {"sim_time_s": 10.0, "experiment_elapsed_s": 0.0,
             "model_motion_elapsed_s": 0.0,
             "position": (0.0, 0.0, 1.0),
             "orientation": (0.0, 0.0, 0.0, 1.0),
             "translation": (0.0, 0.0, 0.0),
             "aabb_min": (-1.0, -1.0, 0.0),
             "aabb_max": (1.0, 1.0, 2.0)},
            {"sim_time_s": 12.0, "experiment_elapsed_s": 2.0,
             "model_motion_elapsed_s": 2.0,
             "position": (2.0, 0.0, 1.0), "orientation": half_turn,
             "translation": (0.002, 0.0, 0.0),
             "aabb_min": (1.0, -1.0, 0.0),
             "aabb_max": (3.0, 1.0, 2.0)},
        ]
        aligned = RECORDER.align_model_truth_at_time(history, 11.0, 0.25)
        self.assertTrue(aligned["available"])
        self.assertEqual(aligned["method"], "LINEAR_SE3")
        self.assertEqual(aligned["time_offset_s"], 0.0)
        self.assertEqual(aligned["bracket_span_s"], 2.0)
        np.testing.assert_allclose(aligned["position"], (1.0, 0.0, 1.0))
        np.testing.assert_allclose(aligned["translation"], (0.001, 0.0, 0.0))
        self.assertAlmostEqual(np.linalg.norm(aligned["orientation"]), 1.0)
        np.testing.assert_allclose(
            aligned["orientation"],
            (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)), atol=1.0e-12)

    def test_model_truth_alignment_has_explicit_bounded_zoh(self):
        history = [{"sim_time_s": 10.0, "position": (0.0, 0.0, 0.0),
                    "orientation": (0.0, 0.0, 0.0, 1.0),
                    "translation": (0.0, 0.0, 0.0)}]
        held = RECORDER.align_model_truth_at_time(history, 10.1, 0.25)
        self.assertTrue(held["available"])
        self.assertEqual(held["method"], "ZOH_BEFORE")
        self.assertAlmostEqual(held["time_offset_s"], -0.1)
        self.assertEqual(held["zoh_bound_s"], 0.25)
        rejected = RECORDER.align_model_truth_at_time(history, 10.3, 0.25)
        self.assertFalse(rejected["available"])
        self.assertEqual(rejected["method"], "OUTSIDE_ZOH_BOUND")

    def test_final_truth_join_uses_detector_time_and_preserves_estimate(self):
        truth = [
            {"model_name": "m01", "truth_sim_time_s": "10.0",
             "experiment_elapsed_s": "0.0", "model_motion_elapsed_s": "0.0",
             "position_x_m": "0", "position_y_m": "0", "position_z_m": "0",
             "orientation_x": "0", "orientation_y": "0",
             "orientation_z": "0", "orientation_w": "1",
             "translation_x_m": "0", "translation_y_m": "0",
             "translation_z_m": "0", "world_aabb_min_x_m": "-0.5",
             "world_aabb_min_y_m": "-0.5", "world_aabb_min_z_m": "-0.5",
             "world_aabb_max_x_m": "0.5", "world_aabb_max_y_m": "0.5",
             "world_aabb_max_z_m": "0.5"},
            {"model_name": "m01", "truth_sim_time_s": "12.0",
             "experiment_elapsed_s": "2.0", "model_motion_elapsed_s": "2.0",
             "position_x_m": "2", "position_y_m": "0", "position_z_m": "0",
             "orientation_x": "0", "orientation_y": "0",
             "orientation_z": "1", "orientation_w": "0",
             "translation_x_m": "0.002", "translation_y_m": "0",
             "translation_z_m": "0", "world_aabb_min_x_m": "1.5",
             "world_aabb_min_y_m": "-0.5", "world_aabb_min_z_m": "-0.5",
             "world_aabb_max_x_m": "2.5", "world_aabb_max_y_m": "0.5",
             "world_aabb_max_z_m": "0.5"},
        ]
        raw = [{
            "gdu_id": "detector-owned-id", "task_scale_probability": "0.73",
            "message_time_s": "11.0", "position_world_x_m": "1.0",
            "position_world_y_m": "0", "position_world_z_m": "0",
            "estimated_dx_m": "0", "estimated_dy_m": "0.001",
            "estimated_dz_m": "0"}, {
            "gdu_id": "held", "message_time_s": "12.1",
            "position_world_x_m": "2.0", "position_world_y_m": "0",
            "position_world_z_m": "0", "estimated_dx_m": "0",
            "estimated_dy_m": "0.002", "estimated_dz_m": "0"}]
        aligned, statistics = RECORDER.realign_algorithm_truth_rows(
            raw, truth, {"m01": (2.0, 2.0)}, 0.1, 0.5 * math.pi, 0.25)
        self.assertEqual(statistics["realigned_row_count"], 2)
        self.assertEqual(aligned[0]["association_status"], "MATCHED")
        self.assertEqual(aligned[0]["association_truth_alignment_method"],
                         "LINEAR_SE3")
        self.assertEqual(aligned[0]["association_truth_time_offset_s"], 0.0)
        self.assertAlmostEqual(
            float(aligned[0]["experiment_elapsed_s_at_message"]), 1.0)
        self.assertAlmostEqual(float(aligned[0]["truth_dx_map_m"]), 0.0,
                               places=12)
        self.assertAlmostEqual(float(aligned[0]["truth_dy_map_m"]), 0.001)
        quaternion = [float(aligned[0][field]) for field in (
            "truth_model_orientation_x", "truth_model_orientation_y",
            "truth_model_orientation_z", "truth_model_orientation_w")]
        self.assertAlmostEqual(np.linalg.norm(quaternion), 1.0)
        self.assertEqual(aligned[1]["association_truth_alignment_method"],
                         "ZOH_BEFORE")
        self.assertAlmostEqual(
            float(aligned[1]["association_truth_time_offset_s"]), -0.1)
        # Evaluation may replace association/truth columns only; detector data
        # and the caller's source rows remain untouched.
        self.assertEqual(aligned[0]["gdu_id"], "detector-owned-id")
        self.assertEqual(aligned[0]["task_scale_probability"], "0.73")
        self.assertEqual(aligned[0]["estimated_dy_m"], "0.001")
        self.assertNotIn("association_status", raw[0])

    def test_final_truth_join_prepares_once_and_caches_each_frame_time(self):
        truth = []
        for sample_index in range(20):
            stamp = 10.0 + 0.1 * sample_index
            for model_index in range(1, 7):
                center = float(model_index * 10)
                truth.append({
                    "model_name": "m{:02d}".format(model_index),
                    "truth_sim_time_s": str(stamp),
                    "experiment_elapsed_s": str(stamp - 10.0),
                    "model_motion_elapsed_s": str(stamp - 10.0),
                    "position_x_m": str(center),
                    "position_y_m": "0", "position_z_m": "0",
                    "orientation_x": "0", "orientation_y": "0",
                    "orientation_z": "0", "orientation_w": "1",
                    "translation_x_m": str(0.001 * (stamp - 10.0)),
                    "translation_y_m": "0", "translation_z_m": "0",
                    "world_aabb_min_x_m": str(center - 1.0),
                    "world_aabb_min_y_m": "-1",
                    "world_aabb_min_z_m": "-1",
                    "world_aabb_max_x_m": str(center + 1.0),
                    "world_aabb_max_y_m": "1",
                    "world_aabb_max_z_m": "1",
                })
        frame_times = (10.25, 10.75, 11.25)
        estimates = [{
            "gdu_id": "g{}".format(index),
            "message_time_s": str(frame_times[index % len(frame_times)]),
            "position_world_x_m": "10", "position_world_y_m": "0",
            "position_world_z_m": "0", "estimated_dx_m": "0.001",
            "estimated_dy_m": "0", "estimated_dz_m": "0",
        } for index in range(240)]

        original = RECORDER._align_prepared_model_truth_at_time
        calls = []

        def counted(prepared, target, bound=0.25):
            calls.append(float(target))
            return original(prepared, target, bound)

        RECORDER._align_prepared_model_truth_at_time = counted
        try:
            aligned, statistics = RECORDER.realign_algorithm_truth_rows(
                estimates, truth,
                {"m{:02d}".format(index): (2.0, 2.0)
                 for index in range(1, 7)},
                0.1, 0.0, 0.25)
        finally:
            RECORDER._align_prepared_model_truth_at_time = original

        self.assertEqual(len(aligned), len(estimates))
        self.assertTrue(all(row["association_status"] == "MATCHED"
                            for row in aligned))
        self.assertEqual(statistics["prepared_truth_history_count"], 6)
        self.assertEqual(
            statistics["unique_aligned_measurement_time_count"], 3)
        self.assertEqual(statistics["prepared_alignment_evaluation_count"], 18)
        self.assertEqual(len(calls), 18)

    def test_prepared_truth_alignment_is_numerically_identical(self):
        history = [
            {"sim_time_s": 12.0, "experiment_elapsed_s": 2.0,
             "model_motion_elapsed_s": 2.0,
             "position": (2.0, 0.0, 1.0),
             "orientation": (0.0, 0.0, 1.0, 0.0),
             "translation": (0.002, 0.0, 0.0),
             "aabb_min": (1.0, -1.0, 0.0),
             "aabb_max": (3.0, 1.0, 2.0)},
            {"sim_time_s": 10.0, "experiment_elapsed_s": 0.0,
             "model_motion_elapsed_s": 0.0,
             "position": (0.0, 0.0, 1.0),
             "orientation": (0.0, 0.0, 0.0, 1.0),
             "translation": (0.0, 0.0, 0.0),
             "aabb_min": (-1.0, -1.0, 0.0),
             "aabb_max": (1.0, 1.0, 2.0)},
        ]
        prepared = RECORDER._prepare_model_truth_history(history)
        for target in (9.8, 10.0, 11.0, 12.0, 12.2, 12.4):
            expected = RECORDER.align_model_truth_at_time(
                history, target, 0.25)
            actual = RECORDER._align_prepared_model_truth_at_time(
                prepared, target, 0.25)
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                if isinstance(expected[key], tuple):
                    np.testing.assert_allclose(actual[key], expected[key],
                                               atol=1.0e-12)
                else:
                    self.assertEqual(actual[key], expected[key])

    def test_tile_callback_defers_all_model_truth_join_to_finalization(self):
        source = inspect.getsource(
            RECORDER.ExperimentRecorder._tile_callback)
        self.assertIn("self.initial_map_from_world", source)
        self.assertIn('"PENDING_FINALIZATION"', source)
        self.assertIn("algorithm_rows", source)
        self.assertIn("_write_batch", source)
        self.assertNotIn("align_model_truth_at_time", source)
        self.assertNotIn("self.motion_truth_history", source)
        self.assertNotIn("self.latest_model_translations", source)
        self.assertNotIn("self.latest_model_geometries", source)
        self.assertNotIn("associate_model_for_evaluation", source)

    def test_tile_callback_records_every_raw_sample_regardless_of_gate(self):
        recorder = object.__new__(RECORDER.ExperimentRecorder)
        recorder.monitoring_capture_open = True
        recorder.lock = threading.RLock()
        recorder.closed = False
        recorder.recorded_reference_keys = set()
        recorder.recorded_detector_frame_index = -1
        recorder.initial_map_from_world = None
        recorder.initial_z_map_from_world = None
        recorder.truth_alignment_max_zoh_residual_s = 0.25
        recorder._base = lambda *args, **kwargs: {
            "message_time_s": args[0], "publish_time_s": args[1]}
        captured = {}

        def write_batch(relative, rows):
            captured.setdefault(relative, []).extend(rows)
            return True

        recorder._write_batch = write_batch
        recorder._maybe_commit_capture_freeze = lambda: None

        def stable(high, low):
            return types.SimpleNamespace(high=high, low=low)

        def vector(x, y, z):
            return types.SimpleNamespace(x=x, y=y, z=z)

        evidence_rows = []
        samples = []
        for index, validity in enumerate((0, 4)):
            gdu = stable(0, index + 1)
            evidence_rows.append(types.SimpleNamespace(
                gdu_id=gdu, position=vector(index, 0, 0),
                task_scale_probability_mean=0.2 + 0.1 * index,
                task_scale_probability_mean_identified=True,
                p_u=0.5, p_a=0.2, p_c=0.2, p_other=0.1,
                task_scale_probability_lower=0.1,
                task_scale_probability_upper=0.4,
                task_displacement_probability=0.3,
                current_velocity_exceedance_probability=0.2,
                continuous_valid_observation_duration_s=1.0,
                velocity_duration_requirement_met=False,
                observable_rank=1,
                minimum_detectable_effect_m=0.001,
                minimum_detectable_effect_defined=True,
                any_task_direction_detectable=True,
                all_task_directions_detectable=False,
                detection_allowed=False, clearing_allowed=False,
                effective_spatial_support=2.0,
                effective_temporal_support=1.0,
                detection_admission=validity, validity=validity))
            samples.append(types.SimpleNamespace(
                gdu_id=gdu, position=vector(index, 0, 0),
                displacement_mean=vector(0.001 * (index + 1), 0, 0),
                velocity_mean=vector(0, 0, 0),
                displacement_covariance_row_major=[1.0] * 9,
                velocity_covariance_row_major=[1.0] * 9,
                observable_projector_row_major=[1.0] * 9,
                active_probability_lower=0.1,
                active_probability_mean=0.2,
                active_probability_upper=0.3,
                active_probability_mean_identified=True,
                detection_admission=validity, validity=validity,
                minimum_detectable_effect_m=0.001,
                minimum_detectable_effect_defined=True,
                other_mass=0.1, interpolation_error_upper_m=0.0))
        common = types.SimpleNamespace(
            header=types.SimpleNamespace(
                frame_id="camera_init", stamp=_fake_stamp(12.0)),
            publish_stamp=_fake_stamp(12.1), reference_epoch=3,
            accepted_frame_index=7, snapshot_id=stable(1, 2))
        message = types.SimpleNamespace(
            lod=2, common=common, tile_key="roi-root/lod-2",
            reference_support=[], interface_motion=[],
            detection_evidence=evidence_rows,
            physical_deformation=samples)

        recorder._tile_callback(message)
        raw = captured["deformation/algorithm_displacement.csv"]
        self.assertEqual(len(raw), 2)
        self.assertEqual([row["validity"] for row in raw], [0, 4])
        self.assertEqual([row["detection_admission"] for row in raw], [0, 4])
        self.assertTrue(all(row["frame_id"] == "camera_init" for row in raw))
        self.assertTrue(all(row["association_status"] ==
                            "PENDING_FINALIZATION" for row in raw))
        self.assertEqual(len(captured[
            "deformation/detection_evidence.csv"]), 2)
        self.assertEqual(recorder.recorded_detector_frame_index, 7)

    def test_final_analysis_reuses_in_memory_truth_join_without_raw_rewrite(self):
        source = inspect.getsource(
            RECORDER.ExperimentRecorder._generate_track_and_runtime_analysis)
        self.assertIn("realign_algorithm_truth_rows", source)
        self.assertNotIn("os.replace", source)
        self.assertIn("return estimates", source)
        self.assertIn("final_full_history_realign", source)
        visual = inspect.getsource(
            RECORDER.ExperimentRecorder._generate_analysis)
        self.assertIn("aligned_algorithm_rows", visual)

    def test_timeline_checkpoints_use_experiment_elapsed_crossings(self):
        self.assertEqual(RECORDER.timeline_checkpoints_crossed(
            None, 0.02, {0.0, 10.0, 50.0, 60.0}), [0.0])
        self.assertEqual(RECORDER.timeline_checkpoints_crossed(
            9.95, 10.03, {10.0, 50.0, 60.0}), [10.0])
        self.assertEqual(RECORDER.timeline_checkpoints_crossed(
            49.9, 60.1, {50.0, 60.0}), [50.0, 60.0])

    def test_source_stamp_is_mapped_to_experiment_time_without_callback_delay(self):
        self.assertAlmostEqual(RECORDER.experiment_elapsed_at_message(
            112.25, 112.0, 12.0, "MOVING"), 12.25)
        # A publication after the terminal truth sample cannot extend the
        # predefined 60 s stationary experiment timeline.
        self.assertAlmostEqual(RECORDER.experiment_elapsed_at_message(
            160.4, 160.0, 60.0, "COMPLETED"), 60.0)

    def test_checkpoint_map_age_is_explicit_not_assumed_synchronous(self):
        summary = RECORDER.summarize_map_ages(120.0, {
            "slam_map": {"stamp_s": 119.5},
            "avoidance_overlay": {
                "stamp_s": 119.8,
                "source_accepted_frame_index": 17,
                "map_revision": 4,
            },
        })
        self.assertAlmostEqual(summary["maximum_map_age_s"], 0.5)
        self.assertEqual(summary["avoidance_source_accepted_frame_index"], 17)
        self.assertEqual(summary["avoidance_map_revision"], 4)

    def test_complete_track_inventory_retains_false_and_ambiguous_tracks(self):
        truth = [
            {"model_name": "m01", "experiment_elapsed_s": "0",
             "translation_magnitude_m": "0"},
            {"model_name": "m01", "experiment_elapsed_s": "10",
             "translation_magnitude_m": "0.005"},
            {"model_name": "m02", "experiment_elapsed_s": "10",
             "translation_magnitude_m": "0"},
        ]
        estimates = [
            {"gdu_id": gdu_id, "association_status": "MATCHED",
             "model_name": "m01", "accepted_frame_index": str(frame),
             "experiment_elapsed_s_at_receive": str(10.0 + frame / 10.0)}
            for gdu_id, base in (("g1", 0), ("g2", 10))
            for frame in range(base + 1, base + 4)
        ] + [
            {"gdu_id": "g3", "association_status": "UNMATCHED",
             "model_name": "", "accepted_frame_index": "21",
             "experiment_elapsed_s_at_receive": "12"},
            {"gdu_id": "g4", "association_status": "AMBIGUOUS",
             "model_name": "", "accepted_frame_index": "22",
             "experiment_elapsed_s_at_receive": "12"},
            {"gdu_id": "g5", "association_status": "UNMATCHED",
             "model_name": "", "accepted_frame_index": "23",
             "experiment_elapsed_s_at_receive": "13"},
        ]
        evolutions = [
            {"gdu_id": "g1", "retained_for_avoidance": "true",
             "lifecycle_state": "1", "experiment_elapsed_s_at_receive": "12"},
            {"gdu_id": "g2", "retained_for_avoidance": "true",
             "lifecycle_state": "1", "experiment_elapsed_s_at_receive": "13"},
            {"gdu_id": "g3", "retained_for_avoidance": "true",
             "lifecycle_state": "1", "experiment_elapsed_s_at_receive": "14"},
            {"gdu_id": "g4", "retained_for_avoidance": "true",
             "lifecycle_state": "1", "experiment_elapsed_s_at_receive": "15"},
        ]
        inventory, activation = RECORDER.build_track_inventory(
            estimates, [], evolutions, truth, ("m01", "m02"), 0.005)
        by_id = {row["gdu_id"]: row for row in inventory}
        self.assertEqual(set(by_id), {"g1", "g2", "g3", "g4", "g5"})
        self.assertEqual(by_id["g1"]["evaluation_category"], "TRUE_POSITIVE")
        self.assertEqual(by_id["g1"]["independent_estimate_frame_count"], 3)
        self.assertEqual(by_id["g1"]["matched_frame_count"], 3)
        self.assertAlmostEqual(by_id["g1"]["matched_frame_purity"], 1.0)
        # The paper's track precision counts each stable, correctly associated
        # GDU once even when multiple local tracks lie on the same object.
        self.assertEqual(by_id["g2"]["evaluation_category"], "TRUE_POSITIVE")
        self.assertEqual(sum(row["evaluation_category"] == "TRUE_POSITIVE"
                             for row in inventory), 2)
        self.assertEqual({row["model_name"] for row in inventory
                          if row["evaluation_category"] == "TRUE_POSITIVE"},
                         {"m01"})
        self.assertEqual(by_id["g3"]["evaluation_category"],
                         "FALSE_POSITIVE_UNMATCHED")
        self.assertEqual(by_id["g4"]["evaluation_category"],
                         "FALSE_POSITIVE_AMBIGUOUS")
        self.assertEqual(by_id["g5"]["evaluation_category"],
                         "UNCONFIRMED_OBSERVATION_TRACK")
        self.assertTrue(activation["m01"]["active"])
        self.assertFalse(activation["m02"]["active"])

    def test_track_association_uses_unique_frames_support_and_purity(self):
        def estimate(frame, status, model="", sample=0):
            return {
                "gdu_id": "g", "accepted_frame_index": str(frame),
                "sample_index": str(sample), "association_status": status,
                "model_name": model,
            }

        # Ten spatial samples in frame 1 still contribute exactly one vote.
        rows = [estimate(1, "MATCHED", "m01", sample)
                for sample in range(10)]
        rows.append(estimate(1, "UNMATCHED", sample=10))
        rows += [estimate(frame, "MATCHED", "m01")
                 for frame in (2, 3, 4)]
        rows += [estimate(5, "AMBIGUOUS")]
        association = RECORDER._track_frame_association(rows)
        self.assertEqual(association["association_status"], "MATCHED")
        self.assertEqual(association["independent_estimate_frame_count"], 5)
        self.assertEqual(association["matched_frame_count"], 4)
        self.assertEqual(association["ambiguous_frame_count"], 1)
        self.assertAlmostEqual(association["matched_frame_purity"], 0.8)
        self.assertEqual(association["minimum_matched_frame_support"], 3)
        self.assertAlmostEqual(
            association["minimum_matched_frame_purity"], 0.8)

        # Perfect purity cannot replace the pre-registered three-frame support.
        insufficient_support = RECORDER._track_frame_association([
            estimate(1, "MATCHED", "m01"),
            estimate(2, "MATCHED", "m01"),
        ])
        self.assertEqual(insufficient_support["association_status"],
                         "AMBIGUOUS")
        self.assertAlmostEqual(
            insufficient_support["matched_frame_purity"], 1.0)

        # Three of four frames is below 0.80 even though support reaches three.
        insufficient_purity = RECORDER._track_frame_association([
            estimate(1, "MATCHED", "m01"),
            estimate(2, "MATCHED", "m01"),
            estimate(3, "MATCHED", "m01"),
            estimate(4, "UNMATCHED"),
        ])
        self.assertEqual(insufficient_purity["association_status"],
                         "AMBIGUOUS")
        self.assertAlmostEqual(
            insufficient_purity["matched_frame_purity"], 0.75)

        # Any second-model hit remains ambiguous even at 4/5 = 0.80 purity.
        multiple_models = RECORDER._track_frame_association([
            estimate(frame, "MATCHED", "m01") for frame in (1, 2, 3, 4)
        ] + [estimate(5, "MATCHED", "m02")])
        self.assertEqual(multiple_models["association_status"], "AMBIGUOUS")
        self.assertEqual(multiple_models["candidate_models"], ["m01", "m02"])
        self.assertAlmostEqual(multiple_models["matched_frame_purity"], 0.8)

    def test_missing_accepted_frame_identity_disqualifies_track_match(self):
        rows = [{
            "gdu_id": "g", "association_status": "MATCHED",
            "model_name": "m01", "accepted_frame_index": str(frame),
        } for frame in (1, 2, 3)]
        rows.append({
            "gdu_id": "g", "association_status": "MATCHED",
            "model_name": "m01", "accepted_frame_index": "",
        })
        association = RECORDER._track_frame_association(rows)
        self.assertEqual(association["association_status"], "AMBIGUOUS")
        self.assertFalse(association["independent_frame_contract_satisfied"])
        self.assertEqual(
            association["estimate_rows_without_accepted_frame_count"], 1)

    def test_object_comparison_relations_use_only_frame_gated_inventory(self):
        estimates = []
        # Forty m01 spatial rows cannot override the independent fifth-frame
        # hit on m02: the confirmed track is still multi-model AMBIGUOUS.
        for frame in (1, 2, 3, 4):
            for sample in range(10):
                estimates.append({
                    "gdu_id": "majority_but_ambiguous",
                    "accepted_frame_index": str(frame),
                    "sample_index": str(sample),
                    "association_status": "MATCHED", "model_name": "m01",
                })
        estimates.append({
            "gdu_id": "majority_but_ambiguous",
            "accepted_frame_index": "5", "association_status": "MATCHED",
            "model_name": "m02",
        })
        estimates.extend({
            "gdu_id": "unique_confirmed", "accepted_frame_index": str(frame),
            "association_status": "MATCHED", "model_name": "m03",
        } for frame in (10, 11, 12))
        estimates.extend({
            "gdu_id": "unique_unconfirmed",
            "accepted_frame_index": str(frame),
            "association_status": "MATCHED", "model_name": "m05",
        } for frame in (20, 21, 22))
        evolutions = [{
            "gdu_id": "majority_but_ambiguous",
            "retained_for_avoidance": "true", "lifecycle_state": "1",
            "source_accepted_frame_index": "5",
        }, {
            "gdu_id": "unique_confirmed", "retained_for_avoidance": "true",
            "lifecycle_state": "1", "source_accepted_frame_index": "12",
        }]
        inventory, _ = RECORDER.build_track_inventory(
            estimates, [], evolutions, [],
            ("m01", "m02", "m03", "m05"), 0.005)
        by_id = {row["gdu_id"]: row for row in inventory}
        self.assertTrue(by_id["majority_but_ambiguous"]["confirmed"])
        self.assertEqual(
            by_id["majority_but_ambiguous"]["association_status"],
            "AMBIGUOUS")
        relations = RECORDER.build_unique_matched_track_relations(inventory)
        self.assertNotIn("majority_but_ambiguous", relations)
        self.assertEqual(relations["unique_confirmed"]["model_name"], "m03")
        self.assertTrue(relations["unique_confirmed"]["confirmed"])
        self.assertEqual(relations["unique_unconfirmed"]["model_name"], "m05")
        self.assertFalse(relations["unique_unconfirmed"]["confirmed"])

        # A malformed duplicate inventory cannot silently create a relation.
        duplicated = [dict(by_id["unique_confirmed"]),
                      dict(by_id["unique_confirmed"])]
        self.assertEqual(
            RECORDER.build_unique_matched_track_relations(duplicated), {})

    def test_visual_object_comparison_has_no_raw_vote_or_mission_override(self):
        source = inspect.getsource(RECORDER.ExperimentRecorder._generate_analysis)
        self.assertIn("build_unique_matched_track_relations", source)
        self.assertNotIn("association_votes", source)
        self.assertNotIn("detected_model_sources", source)
        self.assertNotIn("confirmed_gdu_model", source)

    def test_truth_motion_onset_uses_per_model_elapsed_not_five_mm(self):
        truth = [
            {"model_name": "m06", "experiment_elapsed_s": "10.1",
             "model_motion_elapsed_s": "0.1",
             "translation_magnitude_m": "0.0001"},
            {"model_name": "m06", "experiment_elapsed_s": "15.0",
             "model_motion_elapsed_s": "5.0",
             "translation_magnitude_m": "0.005"},
        ]
        _, activation = RECORDER.build_track_inventory(
            [], [], [], truth, ("m06",), 0.005)
        self.assertTrue(activation["m06"]["active"])
        self.assertAlmostEqual(activation["m06"]["experiment_elapsed_s"], 10.0)
        self.assertEqual(activation["m06"]["onset_source"],
                         "EXPERIMENT_ELAPSED_MINUS_MODEL_MOTION_ELAPSED")

    def test_confirmation_time_requires_newly_detected_transition(self):
        truth = [{
            "model_name": "m01", "experiment_elapsed_s": "10.1",
            "model_motion_elapsed_s": "0.1",
            "surface_max_displacement_m": "0.0001"}]
        estimates = [{
            "gdu_id": "persistent_only", "association_status": "MATCHED",
            "model_name": "m01", "accepted_frame_index": str(frame),
            "experiment_elapsed_s_at_receive": "12"}
            for frame in (1, 2, 3)]
        evolutions = [{
            "gdu_id": "persistent_only", "retained_for_avoidance": "true",
            "lifecycle_state": "2", "experiment_elapsed_s_at_receive": "12"}]
        inventory, _ = RECORDER.build_track_inventory(
            estimates, [], evolutions, truth, ("m01",), 0.005)
        self.assertTrue(inventory[0]["confirmed"])
        self.assertFalse(inventory[0]["confirmation_time_available"])
        self.assertEqual(inventory[0]["confirmation_experiment_elapsed_s"], "")
        self.assertEqual(inventory[0]["confirmation_time_source"],
                         "RETAINED_WITHOUT_NEWLY_DETECTED_EVENT")
        self.assertEqual(inventory[0]["association_status"], "MATCHED")
        self.assertEqual(
            inventory[0]["evaluation_category"],
            "FALSE_POSITIVE_CONFIRMATION_TIME_UNAVAILABLE")

    def test_m04_is_recorded_but_excluded_from_primary_recall(self):
        truth = [{
            "model_name": "m04", "experiment_elapsed_s": "10.1",
            "model_motion_elapsed_s": "0.1",
            "surface_max_displacement_m": "0.0001"}]
        estimates = [{
            "gdu_id": "person_track", "association_status": "MATCHED",
            "model_name": "m04", "accepted_frame_index": str(frame),
            "experiment_elapsed_s_at_message": "11"}
            for frame in (1, 2, 3)]
        evolutions = [{
            "gdu_id": "person_track", "retained_for_avoidance": "true",
            "lifecycle_state": "1",
            "experiment_elapsed_s_at_message": "12"}]
        inventory, _ = RECORDER.build_track_inventory(
            estimates, [], evolutions, truth, ("m04",), 0.005,
            primary_target_models=())
        self.assertTrue(inventory[0]["confirmed"])
        self.assertFalse(inventory[0]["included_in_primary_metrics"])
        self.assertEqual(inventory[0]["evaluation_category"],
                         "FALSE_POSITIVE_EXCLUDED_MODEL")

    def test_capture_cutoff_labels_return_and_stationary_segments_correctly(self):
        self.assertEqual(RECORDER.trajectory_analysis_segment(
            9.999, 10.0, True), "PRIMARY_CAPTURE")
        self.assertEqual(RECORDER.trajectory_analysis_segment(
            10.0, 10.0, True), "RETURN_RECORDED_ONLY")
        self.assertEqual(RECORDER.trajectory_analysis_segment(
            10.0, 10.0, False), "POST_PRIMARY_CAPTURE")
        self.assertEqual(RECORDER.trajectory_analysis_segment(
            50.0, None, False), "PRIMARY_CAPTURE")

    def test_output_contract_contains_required_research_streams(self):
        schemas = RECORDER.CSV_SCHEMAS
        for relative, fields in schemas.items():
            self.assertEqual(
                len(fields), len(set(fields)),
                "duplicate CSV columns in {}".format(relative))
        self.assertIn("trajectories/slam_odometry.csv", schemas)
        self.assertIn("trajectories/gazebo_robot_truth.csv", schemas)
        self.assertIn("truth/m01_m06_motion_truth.csv", schemas)
        self.assertIn("deformation/algorithm_displacement.csv", schemas)
        self.assertIn("deformation/reference_support.csv", schemas)
        self.assertIn("deformation/detection_evidence.csv", schemas)
        fields = schemas["deformation/algorithm_displacement.csv"]
        self.assertIn("frame_id", fields)
        self.assertIn("truth_magnitude_m", fields)
        self.assertIn("vector_error_m", fields)
        self.assertIn("displacement_covariance_row_major", fields)
        self.assertIn("association_status", fields)
        self.assertIn("association_candidate_models", fields)
        self.assertIn("message_time_s", fields)
        self.assertIn("receive_time_s", fields)
        self.assertIn("experiment_elapsed_s_at_message", fields)
        self.assertIn("association_truth_alignment_method", fields)
        self.assertIn("association_truth_time_offset_s", fields)
        self.assertIn("association_truth_zoh_bound_s", fields)
        self.assertIn("truth_model_orientation_w", fields)
        truth_fields = schemas["truth/m01_m06_motion_truth.csv"]
        self.assertIn("truth_frame_id", truth_fields)
        self.assertIn("experiment_elapsed_s", truth_fields)
        self.assertIn("model_motion_elapsed_s", truth_fields)
        self.assertIn("expected_translation_x_m", truth_fields)
        self.assertIn("orientation_w", truth_fields)
        self.assertIn("truth_sim_time_s", truth_fields)
        self.assertIn("truth_batch_index", truth_fields)
        self.assertIn("per_model_duration_s", truth_fields)
        self.assertIn("world_aabb_min_x_m", truth_fields)
        self.assertIn("runtime/per_cycle.csv", schemas)
        self.assertIn("events/timeline_checkpoints.csv", schemas)
        checkpoint_fields = schemas["events/timeline_checkpoints.csv"]
        self.assertIn("maximum_map_age_s", checkpoint_fields)
        self.assertIn("avoidance_map_revision", checkpoint_fields)
        self.assertIn("avoidance_source_snapshot_id", checkpoint_fields)
        self.assertIn("avoidance_forecast_horizons_s", checkpoint_fields)
        self.assertIn("avoidance_current_map_stamp_s", checkpoint_fields)
        self.assertIn("avoidance_source_stamp_s", checkpoint_fields)
        self.assertIn("avoidance_current_map_frame_id", checkpoint_fields)
        self.assertIn("detector_input_cloud_stamp_s", checkpoint_fields)
        self.assertIn("detector_input_cloud_frame_id", checkpoint_fields)
        self.assertIn(
            "source_snapshot_id",
            schemas["evolution/gdu_evolution.csv"])
        self.assertIn(
            "retained_birth_streak",
            schemas["evolution/gdu_evolution.csv"])
        self.assertIn(
            "retained_birth_confirmations",
            schemas["evolution/gdu_evolution.csv"])
        self.assertIn(
            "source_snapshot_id", schemas["maps/avoidance_frames.csv"])
        robot_truth_fields = schemas["trajectories/gazebo_robot_truth.csv"]
        self.assertIn("child_frame_id", robot_truth_fields)
        self.assertIn("truth_map_frame_id", robot_truth_fields)
        self.assertIn("model_qx", robot_truth_fields)
        self.assertIn("body_qw", robot_truth_fields)
        for relative in (
                "deformation/detection_evidence.csv",
                "deformation/reference_support.csv",
                "deformation/interface_motion.csv",
                "deformation/detector_status.csv",
                "runtime/per_cycle.csv",
                "evolution/gdu_evolution.csv",
                "maps/avoidance_frames.csv"):
            self.assertIn("frame_id", schemas[relative], relative)
        lidar_fields = schemas["sensors/dual_lidar_merge_diagnostics.csv"]
        self.assertIn("point_time_mode", lidar_fields)
        self.assertIn("configured_sensor_update_rate_hz", lidar_fields)
        self.assertIn("configured_min_range_m", lidar_fields)
        self.assertIn("configured_max_range_m", lidar_fields)
        self.assertIn("configured_ray_samples_per_sensor", lidar_fields)
        self.assertIn(
            "primary_scheduler_catch_up_suppressed_count", lidar_fields)
        self.assertIn("receive_monotonic_s", lidar_fields)
        self.assertIn("receive_wall_time_s", lidar_fields)
        self.assertIn("maximum_merged_span_ns", lidar_fields)
        self.assertIn("first_accepted_measurement_time_ns", lidar_fields)
        self.assertIn("last_accepted_measurement_time_ns", lidar_fields)
        self.assertIn("accepted_measurement_interval_count", lidar_fields)
        self.assertIn(
            "nonpositive_accepted_measurement_interval_count", lidar_fields)
        self.assertIn("minimum_accepted_measurement_interval_ns", lidar_fields)
        self.assertIn("maximum_accepted_measurement_interval_ns", lidar_fields)
        self.assertIn("mean_accepted_measurement_interval_ns", lidar_fields)
        self.assertIn("realized_measurement_hz", lidar_fields)
        self.assertIn("primary_downsample", lidar_fields)
        self.assertIn("secondary_downsample", lidar_fields)
        self.assertIn("last_primary_point_count", lidar_fields)
        self.assertIn("maximum_secondary_point_count", lidar_fields)
        self.assertIn("last_merged_point_count", lidar_fields)
        self.assertIn("last_total_raw_input_point_count", lidar_fields)
        self.assertIn("last_compacted_output_point_count", lidar_fields)
        self.assertIn("last_removed_no_return_sentinel_count", lidar_fields)
        self.assertIn(
            "last_compacted_output_to_total_raw_input_ratio", lidar_fields)
        self.assertIn("last_secondary_to_primary_point_ratio", lidar_fields)
        self.assertIn("overlap_voxel_size_m", lidar_fields)
        performance_fields = schemas[
            "sensors/gazebo_performance_metrics.csv"]
        for field in ("real_time_factor", "sensor_name",
                      "sim_update_rate_hz", "real_update_rate_hz", "fps"):
            self.assertIn(field, performance_fields)
        self.assertIn("overlap_evaluation_sample_count", lidar_fields)
        self.assertIn("last_primary_overlap_fraction", lidar_fields)
        self.assertIn("last_secondary_overlap_fraction", lidar_fields)
        self.assertIn("mean_voxel_jaccard", lidar_fields)

    def test_analysis_uses_post_threshold_errors_not_zero_motion_median(self):
        path = os.path.join(SCRIPTS, "icra2027_experiment_recorder.py")
        with open(path, "r", encoding="utf-8") as stream:
            source = stream.read()
        self.assertIn("truth_floor_m = 0.005", source)
        self.assertIn("freeze_on_monitoring_complete", source)
        self.assertIn('"evaluation_p95_vector_error_m"', source)
        self.assertIn('"evaluation_rmse_vector_error_m"', source)
        self.assertIn('"robot_trajectory_error.csv"', source)
        self.assertIn('"robot_trajectory_error_return.csv"', source)
        self.assertIn('"robot_trajectory_error_all_segments.csv"', source)
        self.assertIn('"post_primary_segment_counted_in_primary_metric": False', source)
        self.assertIn('"robot_trajectory_error_post_primary_capture.csv"', source)
        self.assertIn('"object_motion_comparison.csv"', source)
        self.assertIn('row["mean_dx_m"]', source)
        self.assertIn('row["retained_for_avoidance"]', source)
        self.assertIn('"confirmed_detection_source"', source)
        self.assertIn('"UNMATCHED"', source)

    def test_nonvisual_analysis_exports_tracks_runtime_tum_and_lidar_summary(self):
        with tempfile.TemporaryDirectory() as output:
            for relative, fields in RECORDER.CSV_SCHEMAS.items():
                path = os.path.join(output, relative)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", newline="") as stream:
                    csv.DictWriter(stream, fieldnames=fields).writeheader()

            def append(relative, row):
                with open(os.path.join(output, relative), "a", newline="") as stream:
                    writer = csv.DictWriter(
                        stream, fieldnames=RECORDER.CSV_SCHEMAS[relative],
                        extrasaction="ignore")
                    writer.writerow(row)

            append("truth/m01_m06_motion_truth.csv", {
                "model_name": "m01", "experiment_elapsed_s": 0.0,
                "model_motion_elapsed_s": 0.0,
                "surface_max_displacement_m": 0.0,
                "translation_magnitude_m": 0.0,
                "truth_sim_time_s": 90.0})
            append("truth/m01_m06_motion_truth.csv", {
                "model_name": "m01", "experiment_elapsed_s": 10.1,
                "model_motion_elapsed_s": 0.1,
                "surface_max_displacement_m": 0.0001,
                "translation_magnitude_m": 0.0001,
                "truth_sim_time_s": 100.1})
            append("truth/m01_m06_motion_truth.csv", {
                "model_name": "m01", "experiment_elapsed_s": 60.0,
                "model_motion_elapsed_s": 40.0,
                "surface_max_displacement_m": 0.04,
                "translation_magnitude_m": 0.04,
                "truth_sim_time_s": 150.0})
            for frame in (7, 8, 9):
                append("deformation/algorithm_displacement.csv", {
                    "gdu_id": "g1", "association_status": "MATCHED",
                    "model_name": "m01",
                    "experiment_elapsed_s_at_receive": 10.2,
                    "experiment_elapsed_s_at_message": 10.15,
                    "receive_time_s": 100.2,
                    "accepted_frame_index": frame,
                    "task_scale_probability": 0.9})
            append("evolution/gdu_evolution.csv", {
                "gdu_id": "g1", "retained_for_avoidance": True,
                "lifecycle_state": 1, "experiment_elapsed_s_at_receive": 10.3,
                "experiment_elapsed_s_at_message": 10.25,
                "receive_time_s": 100.3, "source_accepted_frame_index": 9,
                "task_scale_probability": 0.9})
            append("deformation/algorithm_displacement.csv", {
                "gdu_id": "g2", "association_status": "AMBIGUOUS",
                "model_name": "", "experiment_elapsed_s_at_receive": 20.0,
                "receive_time_s": 110.0, "accepted_frame_index": 10,
                "task_scale_probability": 0.8})
            append("evolution/gdu_evolution.csv", {
                "gdu_id": "g2", "retained_for_avoidance": True,
                "lifecycle_state": 1, "experiment_elapsed_s_at_receive": 20.1,
                "receive_time_s": 110.1, "source_accepted_frame_index": 10,
                "task_scale_probability": 0.8})
            for frame in (11, 12, 13):
                append("deformation/algorithm_displacement.csv", {
                    "gdu_id": "g3", "association_status": "MATCHED",
                    "model_name": "m01", "accepted_frame_index": frame,
                    "experiment_elapsed_s_at_receive": 30.0,
                    "receive_time_s": 120.0,
                    "task_scale_probability": 0.9})
            append("evolution/gdu_evolution.csv", {
                "gdu_id": "g3", "retained_for_avoidance": True,
                "lifecycle_state": 2, "experiment_elapsed_s_at_receive": 30.1,
                "receive_time_s": 120.1, "source_accepted_frame_index": 13,
                "task_scale_probability": 0.9})
            append("runtime/per_cycle.csv", {
                "last_attempt_frame_index": 7, "last_attempt_published": True,
                "processing_time_ms": 12.5, "input_point_count": 40000})
            append("trajectories/slam_odometry.csv", {
                "message_time_s": 100.0, "x_m": 1.0, "y_m": 2.0,
                "z_m": 3.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0})
            append("trajectories/gazebo_robot_truth.csv", {
                "message_time_s": 100.0, "body_x_m": 4.0, "body_y_m": 5.0,
                "body_z_m": 6.0, "body_qx": 0.0, "body_qy": 0.0,
                "body_qz": 0.0, "body_qw": 1.0})
            append("sensors/dual_lidar_merge_diagnostics.csv", {
                "schema_version": 5,
                "point_time_mode": "snapshot", "secondary_enabled": True,
                "zero_no_return_compaction_enabled": True,
                "configured_sensor_update_rate_hz": 10.0,
                "configured_min_range_m": 0.5,
                "configured_max_range_m": 20.0,
                "configured_ray_samples_per_sensor": 20000,
                "primary_scheduler_catch_up_suppressed_count": 1,
                "primary_downsample": 1, "secondary_downsample": 2,
                "configured_secondary_to_primary_point_ratio": 0.5,
                "overlap_voxel_size_m": 0.2,
                "overlap_evaluation_pair_stride": 10,
                "primary_received": 10, "secondary_received": 10,
                "pairs_accepted": 9, "pairs_rejected": 1,
                "maximum_sync_delta_ns": 5000000,
                "mean_sync_delta_ns": 2000000,
                "first_accepted_measurement_time_ns": 1000000000,
                "last_accepted_measurement_time_ns": 2600000000,
                "accepted_measurement_interval_count": 8,
                "nonpositive_accepted_measurement_interval_count": 0,
                "minimum_accepted_measurement_interval_ns": 200000000,
                "maximum_accepted_measurement_interval_ns": 200000000,
                "mean_accepted_measurement_interval_ns": 200000000,
                "realized_measurement_hz": 5.0,
                "maximum_merged_span_ns": 5000000,
                "last_primary_point_count": 20000,
                "maximum_primary_point_count": 20000,
                "last_secondary_point_count": 10000,
                "maximum_secondary_point_count": 10000,
                "last_merged_point_count": 23000,
                "maximum_merged_point_count": 23000,
                "last_secondary_to_primary_point_ratio": 0.5,
                "last_merged_to_primary_point_ratio": 1.15,
                "last_total_raw_input_point_count": 30000,
                "maximum_total_raw_input_point_count": 30000,
                "last_compacted_output_point_count": 23000,
                "maximum_compacted_output_point_count": 23000,
                "last_removed_no_return_sentinel_count": 7000,
                "maximum_removed_no_return_sentinel_count": 7000,
                "last_compacted_output_to_total_raw_input_ratio": (
                    23.0 / 30.0),
                "overlap_evaluations_attempted": 1,
                "overlap_evaluations_failed": 0,
                "overlap_evaluation_sample_count": 1,
                "last_overlap_pair_index": 10,
                "last_primary_valid_return_count": 15000,
                "last_secondary_valid_return_count": 8000,
                "last_primary_occupied_voxel_count": 12000,
                "last_secondary_occupied_voxel_count": 6500,
                "last_shared_occupied_voxel_count": 5000,
                "last_union_occupied_voxel_count": 13500,
                "last_primary_overlap_fraction": 5.0 / 12.0,
                "last_secondary_overlap_fraction": 10.0 / 13.0,
                "last_voxel_jaccard": 10.0 / 27.0,
                "mean_primary_overlap_fraction": 5.0 / 12.0,
                "mean_secondary_overlap_fraction": 10.0 / 13.0,
                "mean_voxel_jaccard": 10.0 / 27.0,
                "minimum_primary_overlap_fraction": 5.0 / 12.0,
                "minimum_secondary_overlap_fraction": 10.0 / 13.0,
                "minimum_voxel_jaccard": 10.0 / 27.0,
                "primary_pair_ratio": 0.9, "secondary_pair_ratio": 0.9})
            os.makedirs(os.path.join(output, "analysis"), exist_ok=True)
            recorder = object.__new__(RECORDER.ExperimentRecorder)
            recorder.output = output
            recorder._generate_track_and_runtime_analysis()
            with open(os.path.join(
                    output, "analysis", "detection_run_summary.json"),
                    encoding="utf-8") as stream:
                summary = json.load(stream)
            self.assertEqual(summary["true_positive_track_count"], 1)
            self.assertEqual(summary["false_positive_track_count_per_run"], 2)
            self.assertEqual(
                summary["ambiguous_confirmed_track_count_counted_as_false"], 1)
            self.assertEqual(
                summary[
                    "confirmation_time_unavailable_confirmed_tracks_counted_as_false"],
                1)
            self.assertFalse(summary["confirmation_event_chain_complete"])
            self.assertEqual(
                summary["track_association_minimum_matched_independent_frames"],
                3)
            self.assertAlmostEqual(
                summary["track_association_minimum_matched_frame_purity"], 0.8)
            self.assertAlmostEqual(summary["unique_confirmed_track_precision"],
                                   1.0 / 3.0)
            self.assertAlmostEqual(summary["effective_monitoring_duration_s"],
                                   60.0)
            self.assertAlmostEqual(summary["false_alarm_rate_tracks_per_min"],
                                   2.0)
            self.assertFalse(summary["candidate_time_available"])
            with open(os.path.join(
                    output, "analysis", "detection_track_inventory.csv"),
                    newline="") as stream:
                tracks = {row["gdu_id"]: row for row in csv.DictReader(stream)}
            self.assertAlmostEqual(
                float(tracks["g1"]["confirmation_experiment_elapsed_s"]),
                10.25)
            self.assertAlmostEqual(float(tracks["g1"]["confirmation_latency_s"]),
                                   0.25)
            self.assertEqual(
                int(tracks["g1"]["independent_estimate_frame_count"]), 3)
            self.assertEqual(int(tracks["g1"]["matched_frame_count"]), 3)
            self.assertAlmostEqual(
                float(tracks["g1"]["matched_frame_purity"]), 1.0)
            self.assertEqual(
                tracks["g3"]["evaluation_category"],
                "FALSE_POSITIVE_CONFIRMATION_TIME_UNAVAILABLE")
            with open(os.path.join(
                    output, "analysis", "dual_lidar_timing_summary.json"),
                    encoding="utf-8") as stream:
                lidar = json.load(stream)
            self.assertEqual(lidar["point_time_modes_observed"], ["snapshot"])
            self.assertAlmostEqual(lidar["maximum_sync_delta_ms"], 5.0)
            self.assertEqual(lidar["primary_downsample_configured"], 1)
            self.assertEqual(lidar["secondary_downsample_configured"], 2)
            self.assertAlmostEqual(
                lidar["configured_sensor_update_rate_hz"], 10.0)
            self.assertEqual(
                lidar["primary_scheduler_catch_up_suppressed_count"], 1)
            self.assertAlmostEqual(lidar["realized_measurement_hz"], 5.0)
            self.assertAlmostEqual(
                lidar["minimum_accepted_measurement_interval_s"], 0.2)
            self.assertAlmostEqual(
                lidar["maximum_accepted_measurement_interval_s"], 0.2)
            self.assertAlmostEqual(
                lidar["mean_accepted_measurement_interval_s"], 0.2)
            self.assertEqual(lidar["maximum_primary_point_count"], 20000)
            self.assertEqual(lidar["maximum_secondary_point_count"], 10000)
            self.assertEqual(lidar["maximum_merged_point_count"], 23000)
            self.assertAlmostEqual(
                lidar["last_secondary_to_primary_point_ratio"], 0.5)
            self.assertAlmostEqual(
                lidar["last_merged_to_primary_point_ratio"], 1.15)
            self.assertEqual(
                lidar["maximum_total_raw_input_point_count"], 30000)
            self.assertEqual(
                lidar["maximum_compacted_output_point_count"], 23000)
            self.assertEqual(
                lidar["last_removed_no_return_sentinel_count"], 7000)
            self.assertAlmostEqual(lidar["overlap_voxel_size_m"], 0.2)
            self.assertEqual(
                lidar["overlap_evaluation_pair_stride"], 10)
            self.assertEqual(
                lidar["overlap_evaluation_sample_count_latest"], 1)
            self.assertIn("intersection/union", lidar["overlap_definition"])
            self.assertIn("before FAST-LIO deskew", lidar["overlap_input_stage"])
            self.assertAlmostEqual(
                lidar["mean_primary_overlap_fraction"], 5.0 / 12.0)
            self.assertAlmostEqual(
                lidar["mean_secondary_overlap_fraction"], 10.0 / 13.0)
            self.assertAlmostEqual(
                lidar["mean_voxel_jaccard"], 10.0 / 27.0)
            with open(os.path.join(
                    output, "analysis", "slam_trajectory_tum.txt"),
                    encoding="utf-8") as stream:
                self.assertEqual(len(stream.readlines()), 1)

    def test_truth_rows_preserve_experiment_and_model_elapsed(self):
        rows = TRUTH_RECORDER.rows_from_truth({
            "schema_version": 1,
            "state": "WAITING",
            "sim_time": 12.0,
            "experiment_elapsed": 2.0,
            "waiting_elapsed": 2.0,
            "motion_elapsed": 0.0,
            "motion_mode": "simultaneous",
            "linear_speed_mm_s": 1.0,
            "angular_speed_rad_s": 0.0015,
            "models": [{
                "laser_retro_id": 6,
                "model_name": "m06",
                "motion_elapsed": 0.0,
                "position": [1.0, 2.0, 3.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "translation": [0.0, 0.0, 0.0],
                "expected_translation_vector": [0.0, 0.0, 0.0],
            }],
        })
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["model_name"], "m06")
        self.assertEqual(row["experiment_elapsed_s"], 2.0)
        self.assertEqual(row["waiting_elapsed_s"], 2.0)
        self.assertEqual(row["motion_elapsed_s"], 0.0)
        self.assertEqual(row["model_motion_elapsed_s"], 0.0)
        self.assertEqual(row["motion_mode"], "simultaneous")
        self.assertEqual(row["orientation_w"], 1.0)


if __name__ == "__main__":
    unittest.main()
