#!/usr/bin/env python3
"""ICRA-style analysis for the V3 stationary 1/3/5/10 mm/s sweep.

Each speed is analyzed independently.  The top-level files only place the four
independent condition summaries side by side; no metric is pooled across speed.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import random
import statistics

os.environ.setdefault("MPLCONFIGDIR", "/tmp/deform_monitor_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


EXPECTED_SPEEDS = (1.0, 3.0, 5.0, 10.0)
EXPECTED_REPEATS = 10
EXPECTED_MODELS = ("m01", "m02", "m03", "m04", "m05", "m06")
SPEED_COLORS = {
    1.0: "#0072B2", 3.0: "#E69F00", 5.0: "#009E73", 10.0: "#D55E00",
}

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Tinos", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.35,
    "grid.alpha": 0.35,
    "lines.linewidth": 1.0,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def read_csv(path):
    with pathlib.Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def file_sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows, fieldnames=None):
    path = pathlib.Path(path)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def boolean(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def number(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def finite(values):
    return [float(value) for value in values if value is not None
            and math.isfinite(float(value))]


def speed_token(speed):
    return ("{:.1f}".format(speed)).replace(".", "p")


def percentile(values, q):
    values = finite(values)
    return float(np.percentile(values, q)) if values else None


def mean(values):
    values = finite(values)
    return float(np.mean(values)) if values else None


def sample_sd(values):
    values = finite(values)
    return float(statistics.stdev(values)) if len(values) > 1 else None


def build_displacement_timeseries(comparison_rows, truth_rows):
    """Compact support-point rows into one estimate per GDU and source frame."""
    truth = {model: {} for model in EXPECTED_MODELS}
    for row in truth_rows:
        model = row.get("model_name")
        elapsed = number(row.get("experiment_elapsed_s"))
        magnitude = number(row.get("surface_max_displacement_m"))
        if magnitude is None:
            magnitude = number(row.get("translation_magnitude_m"))
        if model in truth and elapsed is not None and magnitude is not None:
            truth[model][elapsed] = 1000.0 * magnitude

    per_gdu = {}
    for row in comparison_rows:
        model = row.get("model_name")
        if (model not in truth or not boolean(row.get("confirmed_detection_source"))
                or not boolean(row.get("retained_for_avoidance"))):
            continue
        estimate = number(row.get("estimated_magnitude_m"))
        actual = number(row.get("truth_magnitude_m"))
        frame = number(row.get("source_accepted_frame_index"))
        elapsed = number(row.get("experiment_elapsed_s_at_receive"))
        message_time = number(row.get("message_time_s"))
        receive_time = number(row.get("receive_time_s"))
        if (elapsed is not None and message_time is not None
                and receive_time is not None):
            elapsed -= receive_time - message_time
        if None in (estimate, actual, frame, elapsed):
            continue
        kind = ("FORMAL_TRUE_POSITIVE"
                if row.get("track_evaluation_category") == "TRUE_POSITIVE"
                else "CONFIRMED_DIAGNOSTIC")
        key = (model, kind, int(frame), row.get("gdu_id", ""))
        per_gdu.setdefault(key, (elapsed, estimate, actual))

    grouped = {}
    for (model, kind, frame, _gdu), sample in per_gdu.items():
        grouped.setdefault((model, kind, frame), []).append(sample)
    estimates = []
    for (model, kind, frame), samples in sorted(grouped.items()):
        times = [sample[0] for sample in samples]
        values = [1000.0 * sample[1] for sample in samples]
        actual = 1000.0 * float(np.median([sample[2] for sample in samples]))
        estimate = float(np.median(values))
        estimates.append({
            "model_name": model,
            "series_type": kind,
            "source_accepted_frame_index": frame,
            "experiment_elapsed_s": float(np.median(times)),
            "truth_magnitude_mm": actual,
            "estimated_magnitude_median_mm": estimate,
            "estimated_magnitude_q25_mm": percentile(values, 25),
            "estimated_magnitude_q75_mm": percentile(values, 75),
            "absolute_magnitude_error_mm": abs(estimate - actual),
            "confirmed_gdu_count": len(samples),
        })
    truth_series = {
        model: [{"experiment_elapsed_s": elapsed,
                 "truth_magnitude_mm": magnitude}
                for elapsed, magnitude in sorted(samples.items())]
        for model, samples in truth.items()
    }
    return estimates, truth_series


def confirmation_time(event):
    value = number(event.get("confirmation_experiment_elapsed_s"))
    if value is not None:
        return value
    onset = number(event.get("truth_activation_experiment_elapsed_s"))
    delay = number(event.get("confirmation_latency_s"))
    return onset + delay if onset is not None and delay is not None else None


def displacement_accuracy_rows(run, truth_floor_mm=5.0):
    """One readable magnitude-accuracy row per model and run."""
    events = {row["model_name"]: row for row in run["model_events"]}
    results = []
    for model in EXPECTED_MODELS:
        event = events[model]
        when = confirmation_time(event)
        samples = [row for row in run["displacement_series"]
                   if row["model_name"] == model
                   and row["series_type"] == "FORMAL_TRUE_POSITIVE"]
        nearest = min(samples,
                      key=lambda row: abs(row["experiment_elapsed_s"] - when)) \
            if samples and when is not None else None
        tracking = [row for row in samples
                    if row["truth_magnitude_mm"] >= truth_floor_mm
                    and (when is None or row["experiment_elapsed_s"] >= when)]
        errors = [row["estimated_magnitude_median_mm"]
                  - row["truth_magnitude_mm"] for row in tracking]
        absolute = [abs(value) for value in errors]
        source_accuracy = run["model_accuracy"].get(model, {})
        truth_at_confirmation = number(
            event.get("truth_displacement_at_confirmation_m"))
        truth_at_confirmation = (1000.0 * truth_at_confirmation
                                 if truth_at_confirmation is not None else None)
        confirmation_error = (
            nearest["estimated_magnitude_median_mm"] - truth_at_confirmation
            if nearest is not None and truth_at_confirmation is not None else None)
        raw_vector_rmse = number(source_accuracy.get(
            "evaluation_rmse_vector_error_m"))
        results.append({
            "model_name": model,
            "included_in_primary_metrics": event["included_in_primary_metrics"],
            "formally_detected": boolean(event.get("detected")),
            "confirmation_time_s": when,
            "truth_at_confirmation_mm": truth_at_confirmation,
            "estimate_at_confirmation_mm": (
                nearest["estimated_magnitude_median_mm"] if nearest else None),
            "confirmation_sample_time_offset_s": (
                nearest["experiment_elapsed_s"] - when
                if nearest is not None and when is not None else None),
            "confirmation_signed_error_mm": confirmation_error,
            "confirmation_absolute_error_mm": (
                abs(confirmation_error) if confirmation_error is not None else None),
            "tracking_sample_count": len(errors),
            "tracking_truth_floor_mm": truth_floor_mm,
            "tracking_bias_mm": mean(errors),
            "tracking_mae_mm": mean(absolute),
            "tracking_rmse_mm": (
                math.sqrt(float(np.mean(np.square(errors)))) if errors else None),
            "tracking_p95_absolute_error_mm": percentile(absolute, 95),
            "raw_vector_rmse_mm": (1000.0 * raw_vector_rmse
                                   if raw_vector_rmse is not None else None),
        })
    return results


def bootstrap_ci(runs, estimator, samples, seed):
    """Run-cluster bootstrap; target and cycle rows stay inside their run."""
    observed = estimator(runs)
    if observed is None or not runs:
        return observed, None, None
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        selected = [runs[rng.randrange(len(runs))] for _ in runs]
        value = estimator(selected)
        if value is not None and math.isfinite(value):
            values.append(value)
    return observed, percentile(values, 2.5), percentile(values, 97.5)


def wilson_interval(successes, total, z=1.959963984540054):
    if total <= 0:
        return None, None
    p = successes / float(total)
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total
                         + z * z / (4.0 * total * total)) / denominator
    return center - half, center + half


def protocol_fingerprint(parameters):
    """Compare detector parameters only; motion speed and output paths are absent."""
    selected = {key: value for key, value in parameters.items()
                if key.startswith("/deform_monitor_v1/")}
    encoded = json.dumps(selected, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_batch(batch_dir):
    summary_path = batch_dir / "batch_summary.json"
    if not summary_path.is_file():
        raise SystemExit("Batch is still running or incomplete: missing {}"
                         .format(summary_path))
    summary = read_json(summary_path)
    if summary.get("status") != "COMPLETED":
        raise SystemExit("Formal analysis requires batch_summary status COMPLETED")
    matrix = read_csv(batch_dir / "matrix.csv")
    status = read_csv(batch_dir / "batch_status.csv")
    if len(matrix) != len(EXPECTED_SPEEDS) * EXPECTED_REPEATS:
        raise SystemExit("Expected 40 matrix rows, found {}".format(len(matrix)))
    by_id = {}
    for row in status:
        run_id = int(row["run_id"])
        if row.get("status") != "SUCCESS" or run_id in by_id:
            raise SystemExit("Status ledger must contain one SUCCESS per run_id")
        by_id[run_id] = row
    if set(by_id) != set(range(1, len(matrix) + 1)):
        raise SystemExit("Status ledger does not cover the complete matrix")
    counts = {speed: 0 for speed in EXPECTED_SPEEDS}
    records = []
    for row in matrix:
        speed = float(row["speed_mm_s"])
        if speed not in counts:
            raise SystemExit("Unexpected speed {}".format(speed))
        counts[speed] += 1
        status_row = by_id[int(row["run_id"])]
        run_dir = pathlib.Path(status_row["run_directory"]).resolve()
        records.append({
            "run_id": int(row["run_id"]),
            "speed_mm_s": speed,
            "repeat": int(row["repeat"]),
            "scenario_id": row["scenario_id"],
            "attempt_count": int(status_row["attempt_count"]),
            "run_dir": run_dir,
        })
    if any(count != EXPECTED_REPEATS for count in counts.values()):
        raise SystemExit("Each speed must have exactly 10 completed repeats")
    return records


def load_run(record, cycle_budget_ms):
    run_dir = record["run_dir"]
    required = [
        "run_manifest.json", "ros_parameters.json",
        "analysis/detection_run_summary.json",
        "analysis/model_detection_event_metrics.csv",
        "analysis/model_detection_summary.csv",
        "analysis/object_motion_comparison.csv",
        "runtime/per_cycle.csv", "deformation/detection_evidence.csv",
        "truth/m01_m06_motion_truth.csv",
    ]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise SystemExit("{} missing required analysis inputs: {}".format(
            run_dir, ", ".join(missing)))
    manifest = read_json(run_dir / "run_manifest.json")
    if (manifest.get("final_output_contract_status") != "COMPLETED"
            or manifest.get("finalization_outcome") != "COMPLETED"
            or manifest.get("required_artifact_errors")
            or manifest.get("rosbag_recorded") is not False):
        raise SystemExit("Run contract is not complete: {}".format(run_dir))

    truth_rows = read_csv(run_dir / "truth/m01_m06_motion_truth.csv")
    truth_speeds = {number(row.get("linear_speed_mm_s")) for row in truth_rows}
    truth_models = {row.get("model_name") for row in truth_rows}
    if truth_speeds != {record["speed_mm_s"]} or truth_models != set(EXPECTED_MODELS):
        raise SystemExit("Truth speed/model contract mismatch: {}".format(run_dir))
    if any(abs(number(row.get("rotation_expected_rad"), 0.0)) > 1.0e-12
           for row in truth_rows):
        raise SystemExit("Rotation is active in pure-translation batch: {}".format(run_dir))

    detection = read_json(run_dir / "analysis/detection_run_summary.json")
    model_events = read_csv(run_dir / "analysis/model_detection_event_metrics.csv")
    model_accuracy = read_csv(run_dir / "analysis/model_detection_summary.csv")
    if {row["model_name"] for row in model_events} != set(EXPECTED_MODELS):
        raise SystemExit("Incomplete model event table: {}".format(run_dir))
    primary = tuple(detection.get("primary_metric_models", []))
    event_by_model = {row["model_name"]: row for row in model_events}
    delays = [number(event_by_model[name].get("confirmation_latency_s"))
              for name in primary if boolean(event_by_model[name].get("detected"))]
    displacements = [number(event_by_model[name].get(
        "truth_displacement_at_confirmation_m")) for name in primary
        if boolean(event_by_model[name].get("detected"))]

    cutoff_frame = number(manifest.get(
        "deformation_capture_cutoff_accepted_frame_index"))
    runtime_rows = read_csv(run_dir / "runtime/per_cycle.csv")
    if cutoff_frame is not None:
        runtime_rows = [row for row in runtime_rows
                        if number(row.get("last_attempt_frame_index"),
                                  cutoff_frame) <= cutoff_frame]
    cycle_times = finite(number(row.get("processing_time_ms"))
                         for row in runtime_rows)
    evidence_rows = read_csv(run_dir / "deformation/detection_evidence.csv")
    if cutoff_frame is not None:
        evidence_rows = [row for row in evidence_rows
                         if number(row.get("accepted_frame_index"),
                                   cutoff_frame) <= cutoff_frame]
    mde_values = finite(number(row.get("minimum_detectable_effect_m"))
                        for row in evidence_rows)
    evidence_count = len(evidence_rows)

    comparisons_path = run_dir / "analysis/object_motion_comparison.csv"
    comparison_rows = read_csv(comparisons_path)
    displacement_series, truth_series = build_displacement_timeseries(
        comparison_rows, truth_rows)
    floor = number(detection.get("displacement_error_evaluation_floor_m"), 0.005)
    accuracy_rows = [row for row in comparison_rows
                     if row.get("track_evaluation_category") == "TRUE_POSITIVE"
                     and boolean(row.get("confirmed_detection_source"))
                     and boolean(row.get("retained_for_avoidance"))
                     and number(row.get("truth_magnitude_m"), -1.0) >= floor]
    vector_errors = finite(number(row.get("vector_error_m")) for row in accuracy_rows)
    magnitude_errors = finite(number(row.get("magnitude_error_m")) for row in accuracy_rows)
    direction_cosines = finite(number(row.get("direction_cosine")) for row in accuracy_rows)

    trajectory_path = run_dir / "analysis/robot_trajectory_error_summary.json"
    trajectory = read_json(trajectory_path) if trajectory_path.is_file() else {}
    map_path = run_dir / "analysis/deformation_route_maps/metrics.json"
    map_metrics = read_json(map_path) if map_path.is_file() else {}
    lidar_path = run_dir / "analysis/dual_lidar_timing_summary.json"
    lidar = read_json(lidar_path) if lidar_path.is_file() else {}

    run = dict(record)
    run.update({
        "run_dir": str(run_dir),
        "run_manifest_sha256": file_sha256(run_dir / "run_manifest.json"),
        "primary_models": primary,
        "parameter_fingerprint": protocol_fingerprint(
            read_json(run_dir / "ros_parameters.json")),
        "tp_objects": int(detection["detected_registered_primary_model_count"]),
        "fn_objects": int(detection["false_negative_registered_primary_model_count"]),
        "object_denominator": int(detection["object_recall_denominator_model_count"]),
        "tp_tracks": int(detection["true_positive_track_count"]),
        "fp_tracks": int(detection["false_positive_track_count_per_run"]),
        "duration_s": number(detection["effective_monitoring_duration_s"]),
        "baseline_duration_s": number(detection["zero_displacement_baseline_duration_s"]),
        "baseline_fp_tracks": int(detection["zero_displacement_baseline_false_confirmations"]),
        "confirmation_delays_s": finite(delays),
        "confirmation_displacements_m": finite(displacements),
        "cycle_times_ms": cycle_times,
        "runtime_cycle_count": len(cycle_times),
        "runtime_mean_ms": mean(cycle_times),
        "runtime_p95_ms": percentile(cycle_times, 95),
        "runtime_max_ms": max(cycle_times) if cycle_times else None,
        "deadline_miss_count": sum(value > cycle_budget_ms for value in cycle_times),
        "accuracy_sample_count": len(vector_errors),
        "accuracy_truth_floor_mm": 1000.0 * floor,
        "vector_rmse_m": (math.sqrt(float(np.mean(np.square(vector_errors))))
                          if vector_errors else None),
        "vector_median_m": percentile(vector_errors, 50),
        "vector_p95_m": percentile(vector_errors, 95),
        "magnitude_median_m": percentile(magnitude_errors, 50),
        "magnitude_p95_m": percentile(magnitude_errors, 95),
        "direction_cosine_median": percentile(direction_cosines, 50),
        "evidence_row_count": evidence_count,
        "mde_median_m": percentile(mde_values, 50),
        "any_direction_detectable_fraction": (sum(boolean(row.get(
            "any_task_direction_detectable")) for row in evidence_rows)
            / float(evidence_count) if evidence_count else None),
        "all_directions_detectable_fraction": (sum(boolean(row.get(
            "all_task_directions_detectable")) for row in evidence_rows)
            / float(evidence_count) if evidence_count else None),
        "detection_allowed_fraction": (sum(boolean(row.get("detection_allowed"))
            for row in evidence_rows) / float(evidence_count)
            if evidence_count else None),
        "slam_primary_rmse_m": number(trajectory.get("primary", {}).get("rmse_m")),
        "lidar_realized_hz": number(lidar.get("realized_measurement_hz")),
        "risk_map_available": bool(map_metrics),
        "risk_active_source_count": len(map_metrics.get("active_source_ids", [])),
        "risk_local_route_affected": bool(map_metrics.get(
            "local_route_affected_by_deformation", False)),
        "risk_paths_differ": bool(map_metrics.get("local_paths_differ", False)),
        "risk_forecast_iou": number(map_metrics.get(
            "forecast_self_consistency", {}).get("hard_avoidance_iou")),
        # Do not silently pool historical hard-core and new soft-core metrics.
        "risk_forecast_soft_core_iou": number(map_metrics.get(
            "forecast_self_consistency", {}).get("high_cost_core_iou")),
        "model_events": model_events,
        "model_accuracy": {row["model_name"]: row for row in model_accuracy},
        "displacement_series": displacement_series,
        "truth_series": truth_series,
    })
    run["displacement_accuracy"] = displacement_accuracy_rows(
        run, 1000.0 * floor)
    return run


def flatten_run(run):
    denominator = run["tp_tracks"] + run["fp_tracks"]
    cycles = run["runtime_cycle_count"]
    return {
        "run_id": run["run_id"], "repeat": run["repeat"],
        "scenario_id": run["scenario_id"], "speed_mm_s": run["speed_mm_s"],
        "attempt_count": run["attempt_count"], "run_directory": run["run_dir"],
        "object_recall": run["tp_objects"] / float(run["object_denominator"]),
        "detected_primary_targets": run["tp_objects"],
        "missed_primary_targets": run["fn_objects"],
        "true_positive_tracks": run["tp_tracks"],
        "false_positive_tracks": run["fp_tracks"],
        "track_precision": run["tp_tracks"] / float(denominator) if denominator else "",
        "false_alarm_rate_tracks_per_min": (run["fp_tracks"] * 60.0 / run["duration_s"]),
        "baseline_false_confirmations": run["baseline_fp_tracks"],
        "confirmation_delay_mean_s": mean(run["confirmation_delays_s"]),
        "confirmation_delay_fastest_s": (min(run["confirmation_delays_s"])
                                           if run["confirmation_delays_s"] else ""),
        "confirmation_displacement_min_mm": (
            1000.0 * min(run["confirmation_displacements_m"])
            if run["confirmation_displacements_m"] else ""),
        "runtime_cycle_count": cycles,
        "runtime_mean_ms": run["runtime_mean_ms"],
        "runtime_p95_ms": run["runtime_p95_ms"],
        "runtime_max_ms": run["runtime_max_ms"],
        "deadline_miss_fraction": (run["deadline_miss_count"] / float(cycles)
                                    if cycles else ""),
        "accuracy_sample_count": run["accuracy_sample_count"],
        "vector_rmse_mm": (1000.0 * run["vector_rmse_m"]
                           if run["vector_rmse_m"] is not None else ""),
        "vector_p95_mm": (1000.0 * run["vector_p95_m"]
                          if run["vector_p95_m"] is not None else ""),
        "magnitude_median_error_mm": (1000.0 * run["magnitude_median_m"]
                                      if run["magnitude_median_m"] is not None else ""),
        "direction_cosine_median": run["direction_cosine_median"],
        "mde_median_mm": (1000.0 * run["mde_median_m"]
                          if run["mde_median_m"] is not None else ""),
        "any_direction_detectable_fraction": run["any_direction_detectable_fraction"],
        "all_directions_detectable_fraction": run["all_directions_detectable_fraction"],
        "detection_allowed_fraction": run["detection_allowed_fraction"],
        "slam_primary_rmse_mm": (1000.0 * run["slam_primary_rmse_m"]
                                 if run["slam_primary_rmse_m"] is not None else ""),
        "lidar_realized_hz": run["lidar_realized_hz"],
        "risk_map_available": run["risk_map_available"],
        "risk_active_source_count": run["risk_active_source_count"],
        "risk_local_route_affected": run["risk_local_route_affected"],
        "risk_paths_differ": run["risk_paths_differ"],
        "risk_forecast_hard_iou": (run["risk_forecast_iou"]
                                   if run["risk_forecast_iou"] is not None else ""),
        "risk_forecast_soft_core_iou": (run.get("risk_forecast_soft_core_iou")
                                       if run.get("risk_forecast_soft_core_iou") is not None else ""),
    }


def model_rows(runs):
    rows = []
    for run in runs:
        for event in run["model_events"]:
            accuracy = run["model_accuracy"].get(event["model_name"], {})
            rows.append({
                "run_id": run["run_id"], "repeat": run["repeat"],
                "speed_mm_s": run["speed_mm_s"],
                "model_name": event["model_name"],
                "included_in_primary_metrics": event["included_in_primary_metrics"],
                "truth_active": event["truth_active"],
                "detected": event["detected"],
                "associated_observation_track_count": event[
                    "associated_observation_track_count"],
                "confirmed_track_count": event["confirmed_track_count"],
                "true_positive_track_count": event["true_positive_track_count"],
                "confirmation_latency_s": event["confirmation_latency_s"],
                "truth_displacement_at_confirmation_mm": (
                    1000.0 * number(event["truth_displacement_at_confirmation_m"])
                    if number(event["truth_displacement_at_confirmation_m"]) is not None
                    else ""),
                "maximum_truth_displacement_mm": (
                    1000.0 * number(event["maximum_truth_displacement_m"])
                    if number(event["maximum_truth_displacement_m"]) is not None else ""),
                "evaluation_sample_count": accuracy.get("evaluation_sample_count", ""),
                "vector_error_median_mm": (1000.0 * number(
                    accuracy.get("evaluation_median_vector_error_m"))
                    if number(accuracy.get("evaluation_median_vector_error_m")) is not None
                    else ""),
                "vector_error_p95_mm": (1000.0 * number(
                    accuracy.get("evaluation_p95_vector_error_m"))
                    if number(accuracy.get("evaluation_p95_vector_error_m")) is not None
                    else ""),
                "vector_error_rmse_mm": (1000.0 * number(
                    accuracy.get("evaluation_rmse_vector_error_m"))
                    if number(accuracy.get("evaluation_rmse_vector_error_m")) is not None
                    else ""),
                "magnitude_error_median_mm": (1000.0 * number(
                    accuracy.get("evaluation_median_magnitude_error_m"))
                    if number(accuracy.get("evaluation_median_magnitude_error_m")) is not None
                    else ""),
                "magnitude_error_p95_mm": (1000.0 * number(
                    accuracy.get("evaluation_p95_magnitude_error_m"))
                    if number(accuracy.get("evaluation_p95_magnitude_error_m")) is not None
                    else ""),
                "direction_cosine_median": accuracy.get(
                    "evaluation_median_direction_cosine", ""),
            })
    return rows


def speed_summary(speed, runs, bootstrap_samples, seed, cycle_budget_ms):
    def ratio(selected, numerator, denominator):
        den = sum(item[denominator] for item in selected)
        return sum(item[numerator] for item in selected) / float(den) if den else None

    def precision(selected):
        tp = sum(item["tp_tracks"] for item in selected)
        fp = sum(item["fp_tracks"] for item in selected)
        return tp / float(tp + fp) if tp + fp else None

    def far(selected):
        duration = sum(item["duration_s"] for item in selected)
        return 60.0 * sum(item["fp_tracks"] for item in selected) / duration if duration else None

    def pooled_event_mean(selected, key):
        return mean(value for item in selected for value in item[key])

    estimates = {}
    for key, estimator, offset in (
            ("object_recall", lambda selected: ratio(
                selected, "tp_objects", "object_denominator"), 1),
            ("pooled_track_precision", precision, 2),
            ("false_alarm_rate_tracks_per_min", far, 3),
            ("confirmation_delay_mean_s", lambda selected: pooled_event_mean(
                selected, "confirmation_delays_s"), 4),
            ("confirmation_displacement_mean_mm", lambda selected: (
                1000.0 * pooled_event_mean(selected, "confirmation_displacements_m")
                if pooled_event_mean(selected, "confirmation_displacements_m") is not None
                else None), 5),
            ("runtime_mean_ms", lambda selected: mean(
                item["runtime_mean_ms"] for item in selected), 6),
            ("vector_rmse_mean_mm", lambda selected: mean(
                1000.0 * item["vector_rmse_m"] for item in selected
                if item["vector_rmse_m"] is not None), 7)):
        value, low, high = bootstrap_ci(
            runs, estimator, bootstrap_samples, seed + offset)
        estimates[key] = {"estimate": value, "ci95_low": low, "ci95_high": high}

    all_delays = finite(value for run in runs for value in run["confirmation_delays_s"])
    all_displacements = finite(
        1000.0 * value for run in runs for value in run["confirmation_displacements_m"])
    all_cycles = finite(value for run in runs for value in run["cycle_times_ms"])
    model_summary = {}
    for model in EXPECTED_MODELS:
        events = [next(row for row in run["model_events"]
                       if row["model_name"] == model) for run in runs]
        included = boolean(events[0]["included_in_primary_metrics"])
        confirmed_runs = sum(
            boolean(row["detected"]) if included
            else int(row["confirmed_track_count"]) > 0
            for row in events)
        low, high = wilson_interval(confirmed_runs, len(events))
        model_delays = finite(number(row["confirmation_latency_s"]) for row in events)
        model_displacements = finite(
            1000.0 * number(row["truth_displacement_at_confirmation_m"])
            for row in events if number(row["truth_displacement_at_confirmation_m"]) is not None)
        run_accuracy = [run["model_accuracy"].get(model, {}) for run in runs]
        vector_rmse = finite(
            1000.0 * number(row.get("evaluation_rmse_vector_error_m"))
            for row in run_accuracy
            if number(row.get("evaluation_rmse_vector_error_m")) is not None)
        model_summary[model] = {
            "included_in_primary_metrics": included,
            "rate_definition": ("primary_target_recall" if included
                                else "diagnostic_any_confirmed_track_rate"),
            "confirmed_runs": confirmed_runs, "total_runs": len(events),
            "rate": confirmed_runs / float(len(events)),
            "rate_wilson_ci95_low": low, "rate_wilson_ci95_high": high,
            "confirmation_delay_mean_s": mean(model_delays),
            "confirmation_delay_median_s": percentile(model_delays, 50),
            "confirmation_displacement_median_mm": percentile(model_displacements, 50),
            "accuracy_valid_run_count": len(vector_rmse),
            "vector_error_rmse_mm_mean_across_runs": mean(vector_rmse),
            "vector_error_rmse_mm_sd_across_runs": sample_sd(vector_rmse),
        }
    return {
        "schema_version": 1,
        "speed_mm_s": speed,
        "run_count": len(runs),
        "primary_models": list(runs[0]["primary_models"]),
        "diagnostic_models": sorted(set(EXPECTED_MODELS) - set(runs[0]["primary_models"])),
        "run_cluster_bootstrap_samples": bootstrap_samples,
        "estimates": estimates,
        "counts": {
            "primary_target_run_units": sum(run["object_denominator"] for run in runs),
            "detected_primary_target_run_units": sum(run["tp_objects"] for run in runs),
            "missed_primary_target_run_units": sum(run["fn_objects"] for run in runs),
            "true_positive_tracks": sum(run["tp_tracks"] for run in runs),
            "false_positive_tracks": sum(run["fp_tracks"] for run in runs),
            "baseline_false_confirmations": sum(run["baseline_fp_tracks"] for run in runs),
            "confirmation_event_count": len(all_delays),
            "runtime_cycle_count": len(all_cycles),
            "runtime_deadline_miss_count": sum(value > cycle_budget_ms for value in all_cycles),
        },
        "descriptive": {
            "false_positive_tracks_per_run_mean": mean(run["fp_tracks"] for run in runs),
            "zero_displacement_baseline_far_tracks_per_min": (
                60.0 * sum(run["baseline_fp_tracks"] for run in runs)
                / sum(run["baseline_duration_s"] for run in runs)
                if sum(run["baseline_duration_s"] for run in runs) else None),
            "confirmation_delay_fastest_s": min(all_delays) if all_delays else None,
            "confirmation_delay_median_s": percentile(all_delays, 50),
            "confirmation_delay_iqr_s": [percentile(all_delays, 25), percentile(all_delays, 75)],
            "confirmation_displacement_minimum_mm": min(all_displacements) if all_displacements else None,
            "confirmation_displacement_median_mm": percentile(all_displacements, 50),
            "runtime_p95_ms_pooled_cycles": percentile(all_cycles, 95),
            "runtime_maximum_ms": max(all_cycles) if all_cycles else None,
            "runtime_deadline_miss_fraction": (sum(value > cycle_budget_ms for value in all_cycles)
                                                / float(len(all_cycles)) if all_cycles else None),
            "slam_primary_rmse_mm_mean": mean(
                1000.0 * run["slam_primary_rmse_m"] for run in runs
                if run["slam_primary_rmse_m"] is not None),
            "lidar_realized_hz_mean": mean(run["lidar_realized_hz"] for run in runs),
            "mde_median_mm_across_run_medians": percentile(
                [1000.0 * run["mde_median_m"] for run in runs
                 if run["mde_median_m"] is not None], 50),
            "risk_map_available_run_fraction": mean(run["risk_map_available"] for run in runs),
            "risk_local_route_affected_run_fraction": mean(
                run["risk_local_route_affected"] for run in runs),
            "risk_paths_differ_run_fraction": mean(run["risk_paths_differ"] for run in runs),
        },
        "per_model": model_summary,
        "candidate_metrics": {
            "available": False,
            "reason": "V3 does not publish a distinct Candidate lifecycle event; first observation is not substituted.",
        },
    }


def fmt(value, digits=3):
    return "--" if value is None else ("{:.%df}" % digits).format(value)


def condition_row(summary):
    e = summary["estimates"]
    d = summary["descriptive"]
    return {
        "speed_mm_s": summary["speed_mm_s"], "n_runs": summary["run_count"],
        "target_recall": e["object_recall"]["estimate"],
        "target_recall_ci95_low": e["object_recall"]["ci95_low"],
        "target_recall_ci95_high": e["object_recall"]["ci95_high"],
        "pooled_track_precision": e["pooled_track_precision"]["estimate"],
        "pooled_track_precision_ci95_low": e["pooled_track_precision"]["ci95_low"],
        "pooled_track_precision_ci95_high": e["pooled_track_precision"]["ci95_high"],
        "false_positive_tracks_per_run": d["false_positive_tracks_per_run_mean"],
        "false_alarm_rate_tracks_per_min": e["false_alarm_rate_tracks_per_min"]["estimate"],
        "zero_displacement_baseline_far_tracks_per_min": d[
            "zero_displacement_baseline_far_tracks_per_min"],
        "confirmation_delay_mean_s": e["confirmation_delay_mean_s"]["estimate"],
        "confirmation_delay_fastest_s": d["confirmation_delay_fastest_s"],
        "confirmation_displacement_minimum_mm": d[
            "confirmation_displacement_minimum_mm"],
        "vector_rmse_mean_mm": e["vector_rmse_mean_mm"]["estimate"],
        "runtime_mean_ms": e["runtime_mean_ms"]["estimate"],
        "runtime_p95_ms": d["runtime_p95_ms_pooled_cycles"],
        "deadline_miss_fraction": d["runtime_deadline_miss_fraction"],
        "slam_primary_rmse_mm_mean": d["slam_primary_rmse_mm_mean"],
        "lidar_realized_hz_mean": d["lidar_realized_hz_mean"],
        "risk_map_available_run_fraction": d["risk_map_available_run_fraction"],
    }


def save_figure(fig, stem):
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_speed_summary(rows, runs, out_dir, cycle_budget_ms):
    speeds = [row["speed_mm_s"] for row in rows]
    run_rows = []
    for run in runs:
        run_rows.append({
            "speed_mm_s": run["speed_mm_s"],
            "false_tracks": run["fp_tracks"],
            "confirmation_delay_s": mean(run["confirmation_delays_s"]),
            "confirmation_displacement_mm": percentile(
                [1000.0 * value for value in run["confirmation_displacements_m"]],
                50),
            "runtime_mean_ms": run["runtime_mean_ms"],
            "runtime_p95_ms": run["runtime_p95_ms"],
        })

    fig = plt.figure(figsize=(7.16, 4.75))
    grid = fig.add_gridspec(2, 3, height_ratios=(0.95, 1.05))
    detection = fig.add_subplot(grid[0, :2])
    false_alarm = fig.add_subplot(grid[0, 2])
    delay = fig.add_subplot(grid[1, 0])
    displacement = fig.add_subplot(grid[1, 1])
    runtime = fig.add_subplot(grid[1, 2])

    for label, key, low_key, high_key, color, marker in (
            ("Primary-target recall", "target_recall",
             "target_recall_ci95_low", "target_recall_ci95_high",
             "#0072B2", "o"),
            ("Confirmed-track precision", "pooled_track_precision",
             "pooled_track_precision_ci95_low", "pooled_track_precision_ci95_high",
             "#D55E00", "s")):
        values = np.asarray([row[key] for row in rows])
        lower = np.asarray([row[low_key] for row in rows])
        upper = np.asarray([row[high_key] for row in rows])
        detection.fill_between(speeds, lower, upper, color=color, alpha=0.12)
        detection.plot(speeds, values, marker=marker, color=color,
                       ms=4.0, label=label)
    detection.set_ylim(0.68, 1.015)
    detection.set_xticks(speeds)
    detection.set_xlabel("Translation speed [mm/s]")
    detection.set_ylabel("Rate")
    detection.set_title("Detection reliability with run-cluster 95% CI")
    detection.legend(frameon=False, loc="lower left")
    detection.grid(True, ls="--")
    detection.spines[["top", "right"]].set_visible(False)
    detection.text(-0.07, 1.04, "(a)", transform=detection.transAxes,
                   fontweight="bold", fontsize=9)

    distribution_panel(false_alarm, run_rows, "false_tracks",
                       "False confirmed tracks / run", "(b)")
    false_alarm.set_title("False confirmations")
    distribution_panel(delay, run_rows, "confirmation_delay_s",
                       "Mean delay / run [s]", "(c)")
    delay.set_title("Confirmation latency")
    distribution_panel(displacement, run_rows, "confirmation_displacement_mm",
                       "Median displacement / run [mm]", "(d)")
    displacement.set_title("Displacement at confirmation")

    positions = np.arange(len(EXPECTED_SPEEDS)) + 1.0
    for index, speed in enumerate(EXPECTED_SPEEDS):
        selected = [row for row in run_rows if row["speed_mm_s"] == speed]
        for offset, key, marker, label in (
                (-0.08, "runtime_mean_ms", "o", "run mean"),
                (0.08, "runtime_p95_ms", "^", "run P95")):
            values = finite(row[key] for row in selected)
            runtime.scatter(np.full(len(values), positions[index] + offset),
                            values, s=13, marker=marker,
                            facecolor=(SPEED_COLORS[speed] if key == "runtime_mean_ms"
                                       else "white"),
                            edgecolor=SPEED_COLORS[speed], linewidth=0.7,
                            label=label if index == 0 else None, zorder=3)
    runtime.axhline(cycle_budget_ms, color="#BA0C2F", ls="--", lw=0.9,
                    label="{:.0f} ms budget".format(cycle_budget_ms))
    runtime.set_xticks(positions)
    runtime.set_xticklabels(["{:.0f}".format(speed) for speed in EXPECTED_SPEEDS])
    runtime.set_xlabel("Translation speed [mm/s]")
    runtime.set_ylabel("Synchronous detector time [ms]")
    runtime.set_title("Runtime distribution")
    runtime.grid(True, axis="y", ls="--")
    runtime.spines[["top", "right"]].set_visible(False)
    runtime.legend(frameon=False, fontsize=5.8, loc="lower left")
    runtime.text(-0.10, 1.035, "(e)", transform=runtime.transAxes,
                 fontweight="bold", fontsize=9)

    fig.tight_layout(pad=0.45, w_pad=0.9, h_pad=0.9)
    save_figure(fig, out_dir / "fig_speed_summary")


def plot_model_recall(summaries, out_dir):
    matrix = np.asarray([[summary["per_model"][model]["rate"]
                          for summary in summaries] for model in EXPECTED_MODELS])
    fig, axis = plt.subplots(figsize=(6.6, 3.0))
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(summaries)))
    axis.set_xticklabels(["{} mm/s".format(summary["speed_mm_s"])
                          for summary in summaries])
    axis.set_yticks(range(len(EXPECTED_MODELS)))
    axis.set_yticklabels(EXPECTED_MODELS)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, "{:.0%}".format(matrix[row, column]),
                      ha="center", va="center", color=("white" if matrix[row, column] < .55 else "black"),
                      fontsize=8)
    axis.set_title("Primary-target recall; m04 diagnostic confirmation rate")
    fig.colorbar(image, ax=axis, label="Detected runs / 10")
    fig.tight_layout()
    save_figure(fig, out_dir / "fig_model_recall")


def plot_condition(summary, run_rows, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.8))
    recall = summary["estimates"]["object_recall"]["estimate"]
    precision = summary["estimates"]["pooled_track_precision"]["estimate"]
    axes[0].bar(["Recall", "Precision"], [
        recall if recall is not None else np.nan,
        precision if precision is not None else np.nan],
        color=["#4c78a8", "#59a14f"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Detection")
    delays = [number(row["confirmation_delay_mean_s"]) for row in run_rows]
    axes[1].boxplot([finite(delays)], labels=["10 runs"], showmeans=True)
    axes[1].set_ylabel("Confirmation delay [s]")
    axes[1].set_title("Run-level response")
    runtime = [number(row["runtime_mean_ms"]) for row in run_rows]
    axes[2].boxplot([finite(runtime)], labels=["10 runs"], showmeans=True)
    axes[2].set_ylabel("Mean detector time [ms]")
    axes[2].set_title("Runtime")
    for axis in axes:
        axis.grid(True, axis="y", ls=":", lw=0.5, alpha=0.7)
    fig.suptitle("{} mm/s condition".format(summary["speed_mm_s"]), fontsize=10)
    fig.tight_layout()
    save_figure(fig, out_dir / "fig_condition_summary")


def plot_run_displacement(run, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "model_name", "series_type", "source_accepted_frame_index",
        "experiment_elapsed_s", "truth_magnitude_mm",
        "estimated_magnitude_median_mm", "estimated_magnitude_q25_mm",
        "estimated_magnitude_q75_mm", "absolute_magnitude_error_mm",
        "confirmed_gdu_count",
    ]
    write_csv(out_dir / "displacement_timeseries.csv",
              run["displacement_series"], fields)
    write_csv(out_dir / "displacement_accuracy.csv",
              run["displacement_accuracy"])
    events = {row["model_name"]: row for row in run["model_events"]}
    accuracy = {row["model_name"]: row
                for row in run["displacement_accuracy"]}
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 8.2), sharex=True)
    for axis, model in zip(axes.flat, EXPECTED_MODELS):
        truth = run["truth_series"].get(model, [])
        if truth:
            axis.plot([row["experiment_elapsed_s"] for row in truth],
                      [row["truth_magnitude_mm"] for row in truth],
                      color="black", lw=1.5, label="Gazebo truth")
        for kind, color, label in (
                ("FORMAL_TRUE_POSITIVE", "#1f77b4", "confirmed estimate"),
                ("CONFIRMED_DIAGNOSTIC", "#ff7f0e", "diagnostic estimate")):
            rows = [row for row in run["displacement_series"]
                    if row["model_name"] == model and row["series_type"] == kind]
            if not rows:
                continue
            times = [row["experiment_elapsed_s"] for row in rows]
            medians = [row["estimated_magnitude_median_mm"] for row in rows]
            axis.plot(times, medians, color=color, lw=1.1, label=label)
            axis.fill_between(
                times,
                [row["estimated_magnitude_q25_mm"] for row in rows],
                [row["estimated_magnitude_q75_mm"] for row in rows],
                color=color, alpha=0.18, linewidth=0)
        event = events.get(model, {})
        confirmation = confirmation_time(event)
        if confirmation is not None:
            axis.axvline(confirmation, color="#d62728", ls="--", lw=1.0,
                         label="confirmation")
            confirmed_truth = number(event.get(
                "truth_displacement_at_confirmation_m"))
            if confirmed_truth is not None:
                axis.scatter([confirmation], [1000.0 * confirmed_truth],
                             color="#d62728", s=18, zorder=4)
        axis.axvspan(10.0, 50.0, color="#f2b134", alpha=0.06)
        rmse = accuracy[model]["tracking_rmse_mm"]
        axis.set_title("{} | tracking RMSE {} mm".format(
            model, "--" if rmse is None else "{:.2f}".format(rmse)))
        axis.set_ylabel("displacement [mm]")
        axis.grid(True, ls=":", lw=0.5, alpha=0.7)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, loc="upper left", fontsize=7)
    for axis in axes[-1]:
        axis.set_xlabel("experiment simulation time [s]")
    fig.suptitle("{}: aligned displacement time series".format(
        run["scenario_id"]), fontsize=11)
    fig.tight_layout()
    save_figure(fig, out_dir / "fig_displacement_timeseries")

    fig, axes = plt.subplots(3, 2, figsize=(10.5, 8.2), sharex=True)
    for axis, model in zip(axes.flat, EXPECTED_MODELS):
        rows = [row for row in run["displacement_series"]
                if row["model_name"] == model
                and row["series_type"] == "FORMAL_TRUE_POSITIVE"]
        axis.axhline(0.0, color="black", lw=0.8)
        if rows:
            axis.plot([row["experiment_elapsed_s"] for row in rows],
                      [row["estimated_magnitude_median_mm"]
                       - row["truth_magnitude_mm"] for row in rows],
                      color="#1f77b4", lw=1.1)
        metric = accuracy[model]
        when = metric["confirmation_time_s"]
        if when is not None:
            axis.axvline(when, color="#d62728", ls="--", lw=1.0)
        axis.axvspan(10.0, 50.0, color="#f2b134", alpha=0.06)
        axis.set_title("{} | MAE {} | RMSE {} mm".format(
            model,
            "--" if metric["tracking_mae_mm"] is None else
            "{:.2f}".format(metric["tracking_mae_mm"]),
            "--" if metric["tracking_rmse_mm"] is None else
            "{:.2f}".format(metric["tracking_rmse_mm"])))
        axis.set_ylabel("estimate - truth [mm]")
        axis.grid(True, ls=":", lw=0.5, alpha=0.7)
    for axis in axes[-1]:
        axis.set_xlabel("experiment simulation time [s]")
    fig.suptitle("{}: confirmed displacement error".format(
        run["scenario_id"]), fontsize=11)
    fig.tight_layout()
    save_figure(fig, out_dir / "fig_displacement_error_timeseries")


def accuracy_summary_rows(runs):
    rows = []
    for speed in EXPECTED_SPEEDS:
        selected = [run for run in runs if run["speed_mm_s"] == speed]
        for model in EXPECTED_MODELS:
            metrics = [next(row for row in run["displacement_accuracy"]
                            if row["model_name"] == model)
                       for run in selected]
            confirmation = finite(row["confirmation_absolute_error_mm"]
                                  for row in metrics)
            bias = finite(row["tracking_bias_mm"] for row in metrics)
            mae = finite(row["tracking_mae_mm"] for row in metrics)
            rmse = finite(row["tracking_rmse_mm"] for row in metrics)
            p95 = finite(row["tracking_p95_absolute_error_mm"]
                         for row in metrics)
            vector = finite(row["raw_vector_rmse_mm"] for row in metrics)
            rows.append({
                "speed_mm_s": speed,
                "model_name": model,
                "run_count": len(metrics),
                "formally_detected_run_count": sum(
                    row["formally_detected"] for row in metrics),
                "tracking_valid_run_count": len(rmse),
                "confirmation_absolute_error_mean_mm": mean(confirmation),
                "confirmation_absolute_error_median_mm": percentile(
                    confirmation, 50),
                "confirmation_absolute_error_p95_mm": percentile(
                    confirmation, 95),
                "tracking_bias_mean_mm": mean(bias),
                "tracking_mae_mean_mm": mean(mae),
                "tracking_rmse_mean_mm": mean(rmse),
                "tracking_rmse_sd_mm": sample_sd(rmse),
                "tracking_p95_absolute_error_mean_mm": mean(p95),
                "raw_vector_valid_run_count": len(vector),
                "raw_vector_rmse_mean_mm": mean(vector),
            })
    return rows


def run_level_accuracy(runs):
    rows = []
    for run in runs:
        primary = set(run["primary_models"])
        metrics = [row for row in run["displacement_accuracy"]
                   if row["model_name"] in primary]
        confirmation = finite(row["confirmation_absolute_error_mm"]
                              for row in metrics)
        rmse = finite(row["tracking_rmse_mm"] for row in metrics)
        bias = finite(row["tracking_bias_mm"] for row in metrics)
        rows.append({
            "run_id": run["run_id"], "speed_mm_s": run["speed_mm_s"],
            "recall": run["tp_objects"] / float(run["object_denominator"]),
            "confirmation_mae_mm": mean(confirmation),
            "tracking_model_balanced_rmse_mm": (
                math.sqrt(float(np.mean(np.square(rmse)))) if rmse else None),
            "tracking_bias_mm": mean(bias),
            "valid_model_count": len(rmse),
        })
    return rows


def mean_ci(values, seed):
    values = finite(values)
    if not values:
        return None, None, None
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
    return mean(values), percentile(samples, 2.5), percentile(samples, 97.5)


def positive_log_scale(values):
    """Zero/missing error is not a tiny positive error; retain it on a linear axis."""
    values = finite(values)
    return "log" if values and min(values) > 0 else "linear"


def distribution_panel(axis, run_rows, key, ylabel, panel, log=False):
    grouped = [[row[key] for row in run_rows
                if row["speed_mm_s"] == speed and row[key] is not None]
               for speed in EXPECTED_SPEEDS]
    positions = np.arange(len(EXPECTED_SPEEDS)) + 1.0
    boxes = axis.boxplot(grouped, positions=positions, widths=0.48,
                         patch_artist=True, showfliers=False,
                         medianprops={"color": "black", "lw": 1.0},
                         whiskerprops={"color": "#555555", "lw": 0.7},
                         capprops={"color": "#555555", "lw": 0.7})
    rng = np.random.default_rng(2027)
    for index, (patch, values, speed) in enumerate(
            zip(boxes["boxes"], grouped, EXPECTED_SPEEDS)):
        patch.set_facecolor(SPEED_COLORS[speed])
        patch.set_alpha(0.28)
        jitter = rng.uniform(-0.11, 0.11, len(values))
        axis.scatter(positions[index] + jitter, values, s=13,
                     color=SPEED_COLORS[speed], edgecolor="white",
                     linewidth=0.35, alpha=0.9, zorder=3)
        center, low, high = mean_ci(values, 2100 + index)
        if center is not None:
            axis.errorbar(positions[index], center,
                          yerr=[[center - low], [high - center]], fmt="D",
                          ms=3.2, color="black", capsize=2.0, lw=0.8, zorder=4)
    axis.set_xticks(positions)
    axis.set_xticklabels(["{:.0f}".format(speed) for speed in EXPECTED_SPEEDS])
    axis.set_xlabel("Translation speed [mm/s]")
    axis.set_ylabel(ylabel)
    axis.set_yscale(positive_log_scale(v for group in grouped for v in group) if log else "linear")
    axis.grid(True, axis="y", ls="--")
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(-0.10, 1.035, panel, transform=axis.transAxes,
              fontweight="bold", fontsize=9)


def horizontal_distribution_panel(axis, run_rows, key, xlabel, title,
                                  panel, log=False):
    grouped = [[row[key] for row in run_rows
                if row["speed_mm_s"] == speed and row[key] is not None]
               for speed in EXPECTED_SPEEDS]
    positions = np.arange(len(EXPECTED_SPEEDS)) + 1.0
    boxes = axis.boxplot(grouped, positions=positions, widths=0.48,
                         vert=False, patch_artist=True, showfliers=False,
                         medianprops={"color": "black", "lw": 1.0},
                         whiskerprops={"color": "#555555", "lw": 0.7},
                         capprops={"color": "#555555", "lw": 0.7})
    rng = np.random.default_rng(2027)
    for index, (patch, values, speed) in enumerate(
            zip(boxes["boxes"], grouped, EXPECTED_SPEEDS)):
        patch.set_facecolor(SPEED_COLORS[speed])
        patch.set_alpha(0.25)
        jitter = rng.uniform(-0.11, 0.11, len(values))
        axis.scatter(values, positions[index] + jitter, s=14,
                     color=SPEED_COLORS[speed], edgecolor="white",
                     linewidth=0.35, alpha=0.92, zorder=3)
        center, low, high = mean_ci(values, 2300 + index)
        if center is not None:
            axis.errorbar(center, positions[index],
                          xerr=[[center - low], [high - center]], fmt="D",
                          ms=3.2, color="black", capsize=2.0, lw=0.8, zorder=4)
    axis.set_yticks(positions)
    axis.set_yticklabels(["{:.0f}".format(speed) for speed in EXPECTED_SPEEDS])
    axis.set_ylabel("Speed [mm/s]")
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.set_xscale(positive_log_scale(v for group in grouped for v in group) if log else "linear")
    axis.grid(True, axis="x", ls="--")
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(-0.10, 1.035, panel, transform=axis.transAxes,
              fontweight="bold", fontsize=9)


def temporal_accuracy_profiles(runs):
    edges = np.arange(10.0, 62.0, 2.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    profiles = {}
    for speed in EXPECTED_SPEEDS:
        per_run_error, per_run_coverage = [], []
        for run in (item for item in runs if item["speed_mm_s"] == speed):
            primary = set(run["primary_models"])
            error_curve, coverage_curve = [], []
            for low, high in zip(edges[:-1], edges[1:]):
                model_errors = []
                for model in primary:
                    errors = [row["absolute_magnitude_error_mm"]
                              for row in run["displacement_series"]
                              if row["model_name"] == model
                              and row["series_type"] == "FORMAL_TRUE_POSITIVE"
                              and row["truth_magnitude_mm"] >=
                              run["accuracy_truth_floor_mm"]
                              and low <= row["experiment_elapsed_s"] < high]
                    if errors:
                        model_errors.append(percentile(errors, 50))
                error_curve.append(mean(model_errors))
                coverage_curve.append(len(model_errors) / float(len(primary)))
            per_run_error.append(error_curve)
            per_run_coverage.append(coverage_curve)
        error = np.asarray([[np.nan if value is None else value for value in row]
                            for row in per_run_error])
        coverage = np.asarray(per_run_coverage)
        def error_percentile(q):
            values = [percentile(error[:, column], q)
                      for column in range(error.shape[1])]
            return np.asarray([np.nan if value is None else value
                               for value in values])
        profiles[speed] = {
            "time_s": centers,
            "error_median": error_percentile(50),
            "error_q25": error_percentile(25),
            "error_q75": error_percentile(75),
            "coverage_mean": np.mean(coverage, axis=0),
            "coverage_q25": np.percentile(coverage, 25, axis=0),
            "coverage_q75": np.percentile(coverage, 75, axis=0),
        }
    return profiles


def plot_measurement_accuracy(_rows, runs, out_dir):
    run_rows = run_level_accuracy(runs)
    write_csv(out_dir / "run_level_accuracy.csv", run_rows)
    profiles = temporal_accuracy_profiles(runs)
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.9))
    horizontal_distribution_panel(
        axes[0, 0], run_rows, "confirmation_mae_mm",
        "Run-mean absolute error [mm]", "Accuracy at first confirmation", "(a)")
    axes[0, 0].scatter([], [], s=13, color="#777777", edgecolor="white",
                       linewidth=0.35, label="independent run")
    axes[0, 0].errorbar([], [], yerr=[], fmt="D", ms=3.2, color="black",
                        capsize=2.0, lw=0.8, label="mean and 95% CI")
    axes[0, 0].legend(frameon=False, loc="lower right", fontsize=6.2)
    horizontal_distribution_panel(
        axes[0, 1], run_rows, "tracking_model_balanced_rmse_mm",
        "Model-balanced RMSE [mm]", "Post-confirmation tracking", "(b)",
        log=True)

    for speed in EXPECTED_SPEEDS:
        profile = profiles[speed]
        axes[1, 0].plot(profile["time_s"], profile["error_median"],
                        color=SPEED_COLORS[speed],
                        label="{:.0f} mm/s".format(speed))
        axes[1, 0].fill_between(profile["time_s"], profile["error_q25"],
                                profile["error_q75"],
                                color=SPEED_COLORS[speed], alpha=0.12,
                                linewidth=0)
        axes[1, 1].plot(profile["time_s"], profile["coverage_mean"],
                        color=SPEED_COLORS[speed],
                        label="{:.0f} mm/s".format(speed))
        axes[1, 1].fill_between(profile["time_s"], profile["coverage_q25"],
                                profile["coverage_q75"],
                                color=SPEED_COLORS[speed], alpha=0.12,
                                linewidth=0)
    for axis in axes[1]:
        axis.axvspan(10.0, 50.0, color="#E69F00", alpha=0.035)
        axis.axvline(50.0, color="#666666", ls=":", lw=0.7)
        axis.set_xlim(10.0, 60.0)
        axis.set_xlabel("Experiment simulation time [s]")
        axis.grid(True, ls="--")
        axis.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_yscale(positive_log_scale(
        value for profile in profiles.values() for key in ("error_q25", "error_median", "error_q75")
        for value in profile[key]))
    axes[1, 0].set_ylabel("Run-balanced median $|e_d|$ [mm]")
    axes[1, 0].set_title("Error accumulation during motion and hold")
    axes[1, 0].legend(frameon=False, ncol=2, fontsize=6.2)
    axes[1, 0].text(-0.10, 1.035, "(c)", transform=axes[1, 0].transAxes,
                    fontweight="bold", fontsize=9)
    axes[1, 1].set_ylim(-0.02, 1.04)
    truth_floor = runs[0]["accuracy_truth_floor_mm"]
    axes[1, 1].set_ylabel("Eligible formal trajectories / registered targets")
    axes[1, 1].set_title("Coverage above {:.0f} mm truth floor".format(
        truth_floor))
    axes[1, 1].text(-0.10, 1.035, "(d)", transform=axes[1, 1].transAxes,
                    fontweight="bold", fontsize=9)
    fig.tight_layout(pad=0.5, w_pad=1.0, h_pad=0.9)
    save_figure(fig, out_dir / "fig_measurement_accuracy")


def plot_accuracy_heatmaps(rows, out_dir):
    panels = [
        ("confirmation_absolute_error_mean_mm", "Confirmation $|e_d|$ [mm]",
         "formally_detected_run_count", None),
        ("tracking_rmse_mean_mm", "Tracking RMSE [mm]",
         "tracking_valid_run_count", "log"),
        ("raw_vector_rmse_mean_mm", "3-D vector RMSE [mm]",
         "raw_vector_valid_run_count", "log"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.65))
    for panel_index, (key, title, count_key, scale) in enumerate(panels):
        matrix = np.full((len(EXPECTED_MODELS), len(EXPECTED_SPEEDS)), np.nan)
        counts = np.zeros_like(matrix)
        for row in rows:
            i = EXPECTED_MODELS.index(row["model_name"])
            j = EXPECTED_SPEEDS.index(row["speed_mm_s"])
            if row[key] is not None:
                matrix[i, j] = row[key]
            counts[i, j] = row[count_key]
        valid = matrix[np.isfinite(matrix)]
        cmap = matplotlib.colormaps["viridis"].copy()
        cmap.set_bad("#E6E6E6")
        norm = (matplotlib.colors.LogNorm(vmin=valid.min(), vmax=valid.max())
                if scale == "log" and positive_log_scale(valid) == "log"
                and valid.min() < valid.max() else None)
        image = axes[panel_index].imshow(matrix, cmap=cmap, norm=norm,
                                         aspect="auto")
        axes[panel_index].set_title(title)
        axes[panel_index].set_xticks(range(len(EXPECTED_SPEEDS)))
        axes[panel_index].set_xticklabels(
            ["{:.0f}".format(speed) for speed in EXPECTED_SPEEDS])
        axes[panel_index].set_xlabel("Speed [mm/s]")
        axes[panel_index].set_yticks(range(len(EXPECTED_MODELS)))
        axes[panel_index].set_yticklabels(
            ["m04$^{\dagger}$" if model == "m04" else model
             for model in EXPECTED_MODELS])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value_format = "{:.2f}" if panel_index == 0 else "{:.1f}"
                text_value = ("--" if not np.isfinite(matrix[i, j]) else
                              (value_format + "\n({:.0f}/10)").format(
                                  matrix[i, j], counts[i, j]))
                axes[panel_index].text(j, i, text_value, ha="center", va="center",
                                       fontsize=5.7,
                                       color=("white" if np.isfinite(matrix[i, j])
                                              and image.norm(matrix[i, j]) > 0.58
                                              else "black"))
        axes[panel_index].text(-0.16, 1.06, "({})".format(chr(97 + panel_index)),
                               transform=axes[panel_index].transAxes,
                               fontweight="bold", fontsize=9)
        fig.colorbar(image, ax=axes[panel_index], fraction=0.046, pad=0.025)
    fig.text(0.5, -0.01,
             "$^{\dagger}$m04 is diagnostic-only; blank cells mean no formal trajectory.",
             ha="center", fontsize=6.5)
    fig.tight_layout(pad=0.45, w_pad=0.7)
    save_figure(fig, out_dir / "fig_accuracy_object_speed_matrix")


def plot_measurement_agreement(runs, out_dir):
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 4.25))
    rng = np.random.default_rng(2027)
    for column, speed in enumerate(EXPECTED_SPEEDS):
        selected_runs = [run for run in runs if run["speed_mm_s"] == speed]
        samples = [(row["truth_magnitude_mm"],
                    row["estimated_magnitude_median_mm"])
                   for run in selected_runs for row in run["displacement_series"]
                   if row["series_type"] == "FORMAL_TRUE_POSITIVE"
                   and row["truth_magnitude_mm"] >= 5.0]
        if not samples:
            for axis in axes[:, column]:
                axis.text(.5, .5, "No eligible measurements", transform=axis.transAxes,
                          ha="center", va="center", fontsize=6)
                axis.set_title("{:.0f} mm/s".format(speed))
            continue
        if len(samples) > 5000:
            indices = rng.choice(len(samples), 5000, replace=False)
            samples = [samples[index] for index in indices]
        truth = np.asarray([sample[0] for sample in samples])
        estimate = np.asarray([sample[1] for sample in samples])
        upper = max(float(np.max(truth)), float(np.max(estimate))) * 1.03
        axes[0, column].hexbin(truth, estimate, gridsize=32, mincnt=1,
                               bins="log", cmap="Blues", rasterized=True)
        axes[0, column].plot([0, upper], [0, upper], color="#D55E00",
                             ls="--", lw=0.8)
        axes[0, column].set_xlim(0, upper)
        axes[0, column].set_ylim(0, upper)
        axes[0, column].set_aspect("equal", adjustable="box")
        axes[0, column].set_title("{:.0f} mm/s".format(speed))
        average = 0.5 * (truth + estimate)
        difference = estimate - truth
        axes[1, column].hexbin(average, difference, gridsize=32, mincnt=1,
                               bins="log", cmap="Blues", rasterized=True)
        bias = float(np.mean(difference))
        low, high = np.percentile(difference, [2.5, 97.5])
        axes[1, column].axhline(bias, color="#D55E00", lw=0.9)
        axes[1, column].axhline(low, color="#555555", ls="--", lw=0.7)
        axes[1, column].axhline(high, color="#555555", ls="--", lw=0.7)
        axes[1, column].text(0.03, 0.96,
                             "bias={:.1f}\nemp. LoA [{:.1f}, {:.1f}]".format(
            bias, low, high), transform=axes[1, column].transAxes,
            va="top", fontsize=5.8)
        for row in range(2):
            axes[row, column].grid(True, ls=":")
            axes[row, column].spines[["top", "right"]].set_visible(False)
            axes[row, column].text(-0.22, 1.05,
                                   "({})".format(chr(97 + row * 4 + column)),
                                   transform=axes[row, column].transAxes,
                                   fontweight="bold", fontsize=8)
    axes[0, 0].set_ylabel("Estimated displacement [mm]")
    axes[1, 0].set_ylabel("Estimate - truth [mm]")
    fig.text(0.5, 0.015, "Mean of estimated and true displacement [mm]",
             ha="center", fontsize=8)
    fig.text(0.5, 0.985,
             "Hexagon darkness represents aligned frame-sample density",
             ha="center", va="top", fontsize=6.3)
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.97),
                     pad=0.45, w_pad=0.65, h_pad=0.85)
    save_figure(fig, out_dir / "fig_measurement_agreement")


def write_condition_report(path, summary):
    e, d, c = summary["estimates"], summary["descriptive"], summary["counts"]
    lines = [
        "# {} mm/s independent condition".format(summary["speed_mm_s"]), "",
        "- Valid runs: {}".format(summary["run_count"]),
        "- Primary targets: {}".format(", ".join(summary["primary_models"])),
        "- Diagnostic-only targets: {}".format(", ".join(summary["diagnostic_models"])),
        "- Target recall: {:.3f} (run-cluster bootstrap 95% CI {:.3f}--{:.3f})".format(
            e["object_recall"]["estimate"], e["object_recall"]["ci95_low"],
            e["object_recall"]["ci95_high"]),
        "- Pooled confirmed-track precision: {}".format(
            fmt(e["pooled_track_precision"]["estimate"])),
        "- False-positive tracks/run: {:.3f}".format(
            d["false_positive_tracks_per_run_mean"]),
        "- FAR: {:.3f} tracks/min".format(
            e["false_alarm_rate_tracks_per_min"]["estimate"]),
        "- Zero-displacement baseline FAR: {} tracks/min".format(
            fmt(d["zero_displacement_baseline_far_tracks_per_min"])),
        "- Confirmation delay: mean {} s, fastest {} s".format(
            fmt(e["confirmation_delay_mean_s"]["estimate"]),
            fmt(d["confirmation_delay_fastest_s"])),
        "- Minimum observed confirmation surface displacement: {} mm".format(
            fmt(d["confirmation_displacement_minimum_mm"])),
        "- Detector runtime: mean {} ms, pooled-cycle P95 {} ms".format(
            fmt(e["runtime_mean_ms"]["estimate"]),
            fmt(d["runtime_p95_ms_pooled_cycles"])),
        "- Confirmation events: {}; TP tracks: {}; FP tracks: {}".format(
            c["confirmation_event_count"], c["true_positive_tracks"],
            c["false_positive_tracks"]), "",
        "> Candidate delay is unavailable because V3 has no distinct Candidate event. "
        "First observation is not used as a substitute.", "",
        "> Confidence intervals resample complete runs, preserving correlations among "
        "targets and cycles within the same run.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_table(path, rows):
    lines = [
        "% Each row is an independent 10-run condition; no cross-speed pooling.",
        "\\begin{tabular}{c c c c c c c}",
        "\\toprule",
        "Speed & $R_{obj}$ & $P_{pooled}$ & FP/run & $\\tau_{conf}$ & $d_{conf,min}$ & Runtime \\\\",
        "(mm/s) & & & & (s) & (mm) & (ms) \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append("{:.0f} & {} & {} & {} & {} & {} & {} \\\\".format(
            row["speed_mm_s"], fmt(row["target_recall"]),
            fmt(row["pooled_track_precision"]),
            fmt(row["false_positive_tracks_per_run"], 2),
            fmt(row["confirmation_delay_mean_s"], 2),
            fmt(row["confirmation_displacement_minimum_mm"], 2),
            fmt(row["runtime_mean_ms"], 1)))
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--cycle-budget-ms", type=float, default=500.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise SystemExit("--bootstrap-samples must be at least 100")
    batch_dir = args.batch_dir.resolve()
    records = validate_batch(batch_dir)
    runs = [load_run(record, args.cycle_budget_ms) for record in records]
    fingerprints = {run["parameter_fingerprint"] for run in runs}
    primary_sets = {run["primary_models"] for run in runs}
    if len(fingerprints) != 1 or len(primary_sets) != 1:
        raise SystemExit("Detector parameters or primary metric target set changed within batch")

    grouped = {speed: [run for run in runs if run["speed_mm_s"] == speed]
               for speed in EXPECTED_SPEEDS}
    summaries = [speed_summary(speed, grouped[speed], args.bootstrap_samples,
                               args.seed + index * 100, args.cycle_budget_ms)
                 for index, speed in enumerate(EXPECTED_SPEEDS)]
    output_dir = (args.output_dir.resolve() if args.output_dir
                  else batch_dir / "analysis_icra")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "by_speed").mkdir(exist_ok=True)

    condition_rows = []
    timeseries_index = []
    for summary, speed in zip(summaries, EXPECTED_SPEEDS):
        condition_dir = output_dir / "by_speed" / ("speed_{}_mmps".format(
            speed_token(speed)))
        condition_dir.mkdir(parents=True, exist_ok=True)
        flattened = [flatten_run(run) for run in grouped[speed]]
        models = model_rows(grouped[speed])
        write_csv(condition_dir / "run_metrics.csv", flattened)
        write_csv(condition_dir / "model_metrics.csv", models)
        write_csv(condition_dir / "displacement_accuracy_metrics.csv", [
            dict({"run_id": run["run_id"], "repeat": run["repeat"],
                  "scenario_id": run["scenario_id"],
                  "speed_mm_s": speed}, **row)
            for run in grouped[speed] for row in run["displacement_accuracy"]])
        write_csv(condition_dir / "model_summary.csv", [
            dict({"model_name": model}, **summary["per_model"][model])
            for model in EXPECTED_MODELS])
        (condition_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8")
        write_condition_report(condition_dir / "report.md", summary)
        plot_condition(summary, flattened, condition_dir)
        for run in grouped[speed]:
            run_dir = condition_dir / "runs" / run["scenario_id"]
            plot_run_displacement(run, run_dir)
            timeseries_index.append({
                "run_id": run["run_id"], "repeat": run["repeat"],
                "speed_mm_s": speed, "scenario_id": run["scenario_id"],
                "comparison_csv": str((run_dir / "displacement_timeseries.csv").relative_to(output_dir)),
                "figure_png": str((run_dir / "fig_displacement_timeseries.png").relative_to(output_dir)),
                "figure_pdf": str((run_dir / "fig_displacement_timeseries.pdf").relative_to(output_dir)),
                "accuracy_csv": str((run_dir / "displacement_accuracy.csv").relative_to(output_dir)),
                "error_figure_png": str((run_dir / "fig_displacement_error_timeseries.png").relative_to(output_dir)),
                "error_figure_pdf": str((run_dir / "fig_displacement_error_timeseries.pdf").relative_to(output_dir)),
            })
        condition_rows.append(condition_row(summary))

    write_csv(output_dir / "speed_condition_index.csv", condition_rows)
    write_csv(output_dir / "run_displacement_timeseries_index.csv",
              timeseries_index)
    accuracy_rows = accuracy_summary_rows(runs)
    write_csv(output_dir / "measurement_accuracy_summary.csv", accuracy_rows)
    write_latex_table(output_dir / "table_speed_results.tex", condition_rows)
    plot_speed_summary(condition_rows, runs, output_dir,
                       args.cycle_budget_ms)
    plot_model_recall(summaries, output_dir)
    plot_measurement_accuracy(accuracy_rows, runs, output_dir)
    plot_accuracy_heatmaps(accuracy_rows, output_dir)
    plot_measurement_agreement(runs, output_dir)
    script_path = pathlib.Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "batch_directory": str(batch_dir),
        "analysis_scope": "FOUR_INDEPENDENT_SPEED_CONDITIONS_NO_CROSS_SPEED_POOLING",
        "speed_conditions_mm_s": list(EXPECTED_SPEEDS),
        "runs_per_condition": EXPECTED_REPEATS,
        "cycle_budget_ms": args.cycle_budget_ms,
        "bootstrap": {
            "unit": "complete run cluster", "samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "detector_parameter_fingerprint_sha256": next(iter(fingerprints)),
        "analysis_script_sha256": file_sha256(script_path),
        "matrix_sha256": file_sha256(batch_dir / "matrix.csv"),
        "input_runs": [{
            "run_id": run["run_id"], "speed_mm_s": run["speed_mm_s"],
            "repeat": run["repeat"], "run_directory": run["run_dir"],
            "run_manifest_sha256": run["run_manifest_sha256"],
        } for run in runs],
        "primary_models": list(next(iter(primary_sets))),
        "diagnostic_only_models": sorted(set(EXPECTED_MODELS) - set(next(iter(primary_sets)))),
        "metric_authority": "per-run recorder finalization products from the frozen 0-10-50-60 s interval",
        "candidate_metrics_available": False,
        "candidate_metrics_reason": "No distinct V3 Candidate lifecycle event is recorded.",
        "truth_feedback_to_online_algorithm": False,
        "cross_speed_pooled_metric_emitted": False,
        "per_run_displacement_timeseries_emitted": True,
        "per_run_displacement_accuracy_emitted": True,
        "publication_accuracy_figures_emitted": [
            "fig_measurement_accuracy", "fig_accuracy_object_speed_matrix",
            "fig_measurement_agreement", "fig_speed_summary"],
        "rosbag_used": False,
        "metric_definitions": {
            "target_recall": "sum detected registered primary targets / sum registered primary targets within one speed",
            "pooled_track_precision": "sum true-positive unique confirmed tracks / sum all unique confirmed tracks within one speed",
            "false_alarm_rate": "false-positive unique confirmed tracks / effective monitoring minutes within one speed",
            "confirmation_delay": "first NEWLY_DETECTED time minus model truth motion onset; detected primary targets only",
            "confirmation_displacement": "truth maximum surface displacement at first confirmation",
            "runtime": "BuildBundle + V1Pipeline::AcceptFrame and documented synchronous product handoff, truncated at capture cutoff",
            "confirmation_magnitude_error": "nearest formal confirmed estimate minus Gazebo truth displacement at first confirmation",
            "tracking_magnitude_error": "one median confirmed-GDU estimate per source frame after first confirmation and above the frozen truth floor",
            "raw_vector_error": "existing point/GDU-level three-dimensional vector error from object_motion_comparison.csv",
        },
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
