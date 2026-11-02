#!/usr/bin/env python3

import csv
import json
import math
import pathlib
import subprocess
import tempfile
import unittest

from analyze_script.analyze_stationary_speed_sweep import (
    build_displacement_timeseries,
    displacement_accuracy_rows,
    temporal_accuracy_profiles,
    positive_log_scale,
    load_run,
    flatten_run,
)


SCRIPT = pathlib.Path(__file__).with_name("analyze_stationary_speed_sweep.py")
SPEEDS = (1.0, 3.0, 5.0, 10.0)
MODELS = ("m01", "m02", "m03", "m04", "m05", "m06")
PRIMARY = ("m01", "m02", "m03", "m05", "m06")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class AnalyzerTest(unittest.TestCase):
    def test_soft_and_historical_hard_iou_are_not_pooled(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary) / "run"
            self.make_run(directory, 1.)
            metrics = directory / "analysis/deformation_route_maps/metrics.json"
            metrics.parent.mkdir()
            record = dict(run_dir=directory, speed_mm_s=1., run_id="001",
                          repeat=1, scenario_id="synthetic", attempt_count=1)
            for key, expected in (("hard_avoidance_iou", (.6, "")),
                                  ("high_cost_core_iou", ("", .6))):
                metrics.write_text(json.dumps({"forecast_self_consistency": {key: .6}}))
                result = flatten_run(load_run(record, 100.))
                self.assertEqual((result["risk_forecast_hard_iou"],
                                  result["risk_forecast_soft_core_iou"]), expected)

    def test_zero_and_missing_errors_are_not_forced_positive(self):
        self.assertEqual(positive_log_scale([0, 0]), "linear")
        self.assertEqual(positive_log_scale([0, 2]), "linear")
        self.assertEqual(positive_log_scale([None, float("nan")]), "linear")
        self.assertEqual(positive_log_scale([1e-6, 2e-6]), "log")

    def test_timeseries_weights_each_confirmed_gdu_once_per_frame(self):
        base = {
            "model_name": "m01", "confirmed_detection_source": True,
            "retained_for_avoidance": True,
            "track_evaluation_category": "TRUE_POSITIVE",
            "source_accepted_frame_index": 7,
            "experiment_elapsed_s_at_receive": 12.2,
            "message_time_s": 101.0, "receive_time_s": 101.2,
            "truth_magnitude_m": .010,
        }
        comparisons = [
            dict(base, gdu_id="a", estimated_magnitude_m=.006),
            dict(base, gdu_id="a", estimated_magnitude_m=.006),
            dict(base, gdu_id="b", estimated_magnitude_m=.008),
        ]
        estimates, truth = build_displacement_timeseries(comparisons, [{
            "model_name": "m01", "experiment_elapsed_s": 10,
            "surface_max_displacement_m": .01,
        }])
        self.assertEqual(len(estimates), 1)
        self.assertEqual(estimates[0]["confirmed_gdu_count"], 2)
        self.assertAlmostEqual(
            estimates[0]["estimated_magnitude_median_mm"], 7.0)
        self.assertAlmostEqual(estimates[0]["experiment_elapsed_s"], 12.0)
        self.assertEqual(truth["m01"][0]["truth_magnitude_mm"], 10.0)

    def test_accuracy_separates_confirmation_and_tracking_error(self):
        events = [{"model_name": model, "included_in_primary_metrics": True,
                   "detected": model == "m01",
                   "confirmation_experiment_elapsed_s": 12.0,
                   "truth_displacement_at_confirmation_m": .010}
                  for model in MODELS]
        run = {
            "model_events": events,
            "model_accuracy": {"m01": {
                "evaluation_rmse_vector_error_m": .003}},
            "displacement_series": [
                {"model_name": "m01", "series_type": "FORMAL_TRUE_POSITIVE",
                 "experiment_elapsed_s": 12.1, "truth_magnitude_mm": 10.0,
                 "estimated_magnitude_median_mm": 8.0},
                {"model_name": "m01", "series_type": "FORMAL_TRUE_POSITIVE",
                 "experiment_elapsed_s": 13.0, "truth_magnitude_mm": 12.0,
                 "estimated_magnitude_median_mm": 16.0},
            ],
        }
        metric = displacement_accuracy_rows(run)[0]
        self.assertAlmostEqual(metric["confirmation_absolute_error_mm"], 2.0)
        self.assertAlmostEqual(metric["tracking_bias_mm"], 1.0)
        self.assertAlmostEqual(metric["tracking_mae_mm"], 3.0)
        self.assertAlmostEqual(metric["tracking_rmse_mm"], math.sqrt(10.0))
        self.assertAlmostEqual(metric["raw_vector_rmse_mm"], 3.0)

    def test_temporal_profile_respects_truth_floor_and_formal_series(self):
        runs = []
        for speed in SPEEDS:
            runs.append({
                "speed_mm_s": speed,
                "primary_models": ("m01",),
                "accuracy_truth_floor_mm": 5.0,
                "displacement_series": [
                    {"model_name": "m01", "series_type": "FORMAL_TRUE_POSITIVE",
                     "experiment_elapsed_s": 11.0, "truth_magnitude_mm": 4.0,
                     "absolute_magnitude_error_mm": 99.0},
                    {"model_name": "m01", "series_type": "FORMAL_TRUE_POSITIVE",
                     "experiment_elapsed_s": 13.0, "truth_magnitude_mm": 6.0,
                     "absolute_magnitude_error_mm": 2.0},
                    {"model_name": "m01", "series_type": "CONFIRMED_DIAGNOSTIC",
                     "experiment_elapsed_s": 13.0, "truth_magnitude_mm": 6.0,
                     "absolute_magnitude_error_mm": 88.0},
                ],
            })
        profile = temporal_accuracy_profiles(runs)[1.0]
        self.assertTrue(math.isnan(profile["error_median"][0]))
        self.assertEqual(profile["coverage_mean"][0], 0.0)
        self.assertEqual(profile["error_median"][1], 2.0)
        self.assertEqual(profile["coverage_mean"][1], 1.0)

    def test_four_speeds_stay_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            batch = pathlib.Path(temporary) / "batch"
            batch.mkdir()
            matrix, status = [], []
            run_id = 0
            for repeat in range(1, 11):
                for speed in SPEEDS:
                    run_id += 1
                    scenario = "stationary_l{}mmps_r{:02d}".format(
                        str(speed).replace(".", "p"), repeat)
                    run = batch / "runs" / scenario
                    self.make_run(run, speed)
                    matrix.append({"run_id": "{:03d}".format(run_id),
                                   "speed_mm_s": speed, "repeat": "{:02d}".format(repeat),
                                   "block_position": 1, "scenario_id": scenario})
                    status.append({"timestamp": "x", "run_id": "{:03d}".format(run_id),
                                   "speed_mm_s": speed, "repeat": "{:02d}".format(repeat),
                                   "scenario_id": scenario, "attempt_count": 1,
                                   "status": "SUCCESS", "run_directory": str(run)})
            write_csv(batch / "matrix.csv", matrix)
            write_csv(batch / "batch_status.csv", status)
            (batch / "batch_summary.json").write_text(json.dumps({
                "status": "COMPLETED", "successful_run_count": 40}), encoding="utf-8")
            subprocess.check_call([
                "python3", "-B", str(SCRIPT), "--batch-dir", str(batch),
                "--bootstrap-samples", "100"])
            output = batch / "analysis_icra"
            with (output / "speed_condition_index.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([float(row["speed_mm_s"]) for row in rows], list(SPEEDS))
            self.assertTrue(all(int(row["n_runs"]) == 10 for row in rows))
            for speed in SPEEDS:
                directory = output / "by_speed" / (
                    "speed_{}_mmps".format(("{:.1f}".format(speed)).replace(".", "p")))
                with (directory / "run_metrics.csv").open(newline="") as stream:
                    run_rows = list(csv.DictReader(stream))
                self.assertEqual({float(row["speed_mm_s"]) for row in run_rows}, {speed})
                self.assertEqual(len(run_rows), 10)
                self.assertTrue(all(int(row["runtime_cycle_count"]) == 1
                                    for row in run_rows))
                with (directory / "model_summary.csv").open(newline="") as stream:
                    model_summary = list(csv.DictReader(stream))
                m04 = next(row for row in model_summary if row["model_name"] == "m04")
                self.assertEqual(m04["rate_definition"],
                                 "diagnostic_any_confirmed_track_rate")
            manifest = json.loads((output / "analysis_manifest.json").read_text())
            self.assertFalse(manifest["cross_speed_pooled_metric_emitted"])
            self.assertTrue(manifest["per_run_displacement_timeseries_emitted"])
            self.assertTrue(manifest["per_run_displacement_accuracy_emitted"])
            self.assertTrue((output / "measurement_accuracy_summary.csv").is_file())
            self.assertTrue((output / "fig_measurement_accuracy.png").is_file())
            self.assertTrue((output / "fig_accuracy_object_speed_matrix.pdf").is_file())
            self.assertTrue((output / "fig_measurement_agreement.pdf").is_file())
            self.assertTrue((output / "run_level_accuracy.csv").is_file())
            with (output / "run_displacement_timeseries_index.csv").open(
                    newline="") as stream:
                timeseries = list(csv.DictReader(stream))
            self.assertEqual(len(timeseries), 40)
            for row in timeseries:
                self.assertTrue((output / row["comparison_csv"]).is_file())
                self.assertTrue((output / row["figure_png"]).is_file())
                self.assertTrue((output / row["figure_pdf"]).is_file())
                self.assertTrue((output / row["accuracy_csv"]).is_file())
                self.assertTrue((output / row["error_figure_png"]).is_file())

    def make_run(self, run, speed):
        (run / "analysis").mkdir(parents=True)
        (run / "runtime").mkdir()
        (run / "deformation").mkdir()
        (run / "truth").mkdir()
        (run / "run_manifest.json").write_text(json.dumps({
            "final_output_contract_status": "COMPLETED",
            "finalization_outcome": "COMPLETED", "required_artifact_errors": [],
            "rosbag_recorded": False,
            "deformation_capture_cutoff_accepted_frame_index": 1}), encoding="utf-8")
        (run / "ros_parameters.json").write_text(json.dumps({
            "/deform_monitor_v1/v1/calibration/test": 1}), encoding="utf-8")
        (run / "analysis/detection_run_summary.json").write_text(json.dumps({
            "primary_metric_models": list(PRIMARY),
            "detected_registered_primary_model_count": 5,
            "false_negative_registered_primary_model_count": 0,
            "object_recall_denominator_model_count": 5,
            "true_positive_track_count": 5,
            "false_positive_track_count_per_run": 1,
            "effective_monitoring_duration_s": 60,
            "zero_displacement_baseline_duration_s": 10,
            "zero_displacement_baseline_false_confirmations": 0,
            "displacement_error_evaluation_floor_m": .005}), encoding="utf-8")
        events = []
        for model in MODELS:
            events.append({"model_name": model,
                           "included_in_primary_metrics": model in PRIMARY,
                           "truth_active": True, "detected": model in PRIMARY,
                           "associated_observation_track_count": 1,
                           "confirmed_track_count": 1,
                           "true_positive_track_count": 1 if model in PRIMARY else 0,
                           "confirmation_latency_s": 8 if model in PRIMARY else "",
                           "truth_displacement_at_confirmation_m": .008 if model in PRIMARY else "",
                           "maximum_truth_displacement_m": .04})
        write_csv(run / "analysis/model_detection_event_metrics.csv", events)
        write_csv(run / "analysis/model_detection_summary.csv", [
            {"model_name": model, "evaluation_sample_count": 1,
             "evaluation_rmse_vector_error_m": .001} for model in MODELS])
        write_csv(run / "analysis/object_motion_comparison.csv", [{
            "track_evaluation_category": "TRUE_POSITIVE",
            "confirmed_detection_source": True,
            "retained_for_avoidance": True,
            "truth_magnitude_m": .01,
            "vector_error_m": .001,
            "magnitude_error_m": .0005,
            "direction_cosine": .99}])
        write_csv(run / "runtime/per_cycle.csv", [
            {"processing_time_ms": 100, "last_attempt_published": True,
             "last_attempt_frame_index": 1},
            {"processing_time_ms": 999, "last_attempt_published": True,
             "last_attempt_frame_index": 2}])
        write_csv(run / "deformation/detection_evidence.csv", [
            {"minimum_detectable_effect_m": .002,
             "any_task_direction_detectable": True,
             "all_task_directions_detectable": False,
             "detection_allowed": True, "accepted_frame_index": 1},
            {"minimum_detectable_effect_m": .999,
             "any_task_direction_detectable": False,
             "all_task_directions_detectable": False,
             "detection_allowed": False, "accepted_frame_index": 2}])
        write_csv(run / "truth/m01_m06_motion_truth.csv", [
            {"linear_speed_mm_s": speed, "model_name": model,
             "rotation_expected_rad": 0} for model in MODELS])


if __name__ == "__main__":
    unittest.main()
