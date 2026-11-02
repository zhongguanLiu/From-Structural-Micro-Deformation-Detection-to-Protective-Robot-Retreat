#!/usr/bin/env python3
"""Reproducible, speed-stratified analysis of ALERT and dynamic-detection baselines."""

import argparse
import csv
import gzip
import json
import math
import os
import pathlib
import statistics
import hashlib
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/deform_monitor_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import compare_figures
import analyze_stationary_speed_sweep as stationary


ALL_METHODS = compare_figures.METHODS
METHODS = ALL_METHODS[:-1]
METHOD_LABELS = compare_figures.LABELS
SPEEDS = compare_figures.SPEEDS
MODELS = compare_figures.MODELS
PRIMARY_MODELS = ("m01", "m02", "m03", "m05", "m06")
SOURCES = set()

def number(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def read_csv_gz(path):
    SOURCES.add(pathlib.Path(path).resolve())
    with gzip.open(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def read_csv(path):
    SOURCES.add(pathlib.Path(path).resolve())
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def boolean(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, q):
    values = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.percentile(values, q)) if values else None


def mean(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(values)) if values else None


def bootstrap_ci(records, statistic, rng, samples=10000):
    if not records:
        return None, None
    estimates = []
    for _ in range(samples):
        selected = [records[index] for index in rng.integers(0, len(records), len(records))]
        value = statistic(selected)
        if value is not None and math.isfinite(value):
            estimates.append(value)
    return (percentile(estimates, 2.5), percentile(estimates, 97.5))


def wilson_interval(successes, trials, z=1.959963984540054):
    if trials <= 0:
        return None, None
    proportion = successes / float(trials)
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    low = 0.0 if successes == 0 else max(0.0, center - half)
    high = 1.0 if successes == trials else min(1.0, center + half)
    return low, high


def confirmation(frames, minimum_points, consecutive_frames, maximum_gap_s):
    streak = 0
    previous_stamp = None
    streak_start = None
    for frame in frames:
        stamp = frame["source_stamp_s"]
        if previous_stamp is not None and stamp - previous_stamp > maximum_gap_s:
            streak = 0
            streak_start = None
        if frame["point_count"] >= minimum_points:
            if streak == 0:
                streak_start = frame
            streak += 1
            if streak >= consecutive_frames:
                return streak_start, frame
        else:
            streak = 0
            streak_start = None
        previous_stamp = stamp
    return None, None


def quaternion_angle(a, b):
    dot = abs(sum(x * y for x, y in zip(a, b)))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def load_frames(path):
    frames = {}
    for row in read_csv_gz(path):
        elapsed = number(row.get("truth_experiment_elapsed_s"))
        if elapsed is None or not 0.0 <= elapsed <= 60.5:
            continue
        stamp = float(row["source_stamp_s"])
        frame = frames.setdefault(stamp, {
            "source_stamp_s": stamp,
            "elapsed_s": elapsed,
            "motion_elapsed_s": number(row.get("truth_motion_elapsed_s"), 0.0),
            "state": row.get("truth_state", ""),
            "counts": {},
        })
        if (not math.isfinite(stamp) or row["group_name"] in frame["counts"]
                or abs(frame["elapsed_s"] - elapsed) > 1e-6
                or int(row["point_count"]) < 0):
            raise ValueError("invalid or duplicated frame row in {}".format(path))
        frame["counts"][row["group_name"]] = int(row["point_count"])
    if not frames or any(set(frame["counts"]) != set(MODELS) | {"OTHER"}
                         for frame in frames.values()):
        raise ValueError("incomplete target/OTHER frame groups in {}".format(path))
    return [frames[key] for key in sorted(frames)]


def analyze_run(run_dir, ledger_row, minimum_points, consecutive_frames,
                maximum_gap_s):
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    SOURCES.add(run_dir / "run_manifest.json")
    detections = load_frames(run_dir / "detections/detection_frames.csv.gz")
    inputs = load_frames(run_dir / "runtime/input_frames.csv.gz")
    for frame in detections:
        frame["point_count"] = sum(frame["counts"].values())

    model_metrics = []
    for model in MODELS:
        motion = []
        for frame in detections:
            if 10.0 <= frame["elapsed_s"] < 50.0:
                motion.append(dict(frame, point_count=frame["counts"].get(model, 0)))
        start, confirmed = confirmation(
            motion, minimum_points, consecutive_frames, maximum_gap_s)
        model_metrics.append({
            "model_name": model,
            "confirmed": confirmed is not None,
            "confirmation_latency_s": (None if confirmed is None else
                                         confirmed["elapsed_s"] - 10.0),
            "confirmation_displacement_mm": None,  # Filled from recorded truth, not commanded speed.
            "raw_positive_frame_fraction": (
                sum(frame["point_count"] >= minimum_points for frame in motion) /
                len(motion) if motion else 0.0),
            "maximum_dynamic_points": max(
                (frame["point_count"] for frame in motion), default=0),
        })

    primary = [row for row in model_metrics if row["model_name"] in PRIMARY_MODELS]
    baseline = [frame for frame in detections if frame["elapsed_s"] < 10.0]
    moving = [frame for frame in detections if 10.0 <= frame["elapsed_s"] < 50.0]
    hold = [frame for frame in detections if 50.0 <= frame["elapsed_s"] <= 60.5]
    target_points = sum(sum(frame["counts"].get(name, 0) for name in MODELS)
                        for frame in moving)
    all_dynamic_points = sum(frame["point_count"] for frame in moving)
    other_points = sum(frame["counts"].get("OTHER", 0) for frame in moving)

    runtime = read_csv_gz(run_dir / "runtime/algorithm_runtime.csv.gz")
    elapsed_by_stamp = {frame["source_stamp_s"]: frame["elapsed_s"]
                        for frame in detections}
    runtime_ms = []
    for row in runtime:
        stamp = float(row["source_stamp_s"])
        if stamp in elapsed_by_stamp and 0.0 <= elapsed_by_stamp[stamp] <= 60.0:
            runtime_ms.append(float(row["processing_time_ms"]))

    odometry = read_csv_gz(run_dir / "runtime/fastlio_odometry.csv.gz")
    truth_start = statistics.median(
        frame["source_stamp_s"] - frame["elapsed_s"] for frame in inputs)
    odometry = [dict(row, experiment_elapsed_s=
                     float(row["source_stamp_s"]) - truth_start)
                for row in odometry
                if truth_start <= float(row["source_stamp_s"]) <= truth_start + 60.5]
    odom_translation = []
    odom_rotation = []
    if odometry:
        reference = min(odometry,
                        key=lambda row: abs(row["experiment_elapsed_s"] - 10.0))
        first_p = tuple(float(reference[key]) for key in
                        ("position_x", "position_y", "position_z"))
        first_q = tuple(float(reference[key]) for key in
                        ("orientation_x", "orientation_y", "orientation_z", "orientation_w"))
        for row in (item for item in odometry
                    if item["experiment_elapsed_s"] >= reference["experiment_elapsed_s"]):
            p = tuple(float(row[key]) for key in
                      ("position_x", "position_y", "position_z"))
            q = tuple(float(row[key]) for key in
                      ("orientation_x", "orientation_y", "orientation_z", "orientation_w"))
            odom_translation.append(math.sqrt(sum((x - y) ** 2 for x, y in zip(p, first_p))))
            odom_rotation.append(quaternion_angle(q, first_q))

    input_motion = [frame for frame in inputs if 10.0 <= frame["elapsed_s"] < 50.0]
    visibility = {
        model: (sum(frame["counts"].get(model, 0) > 0 for frame in input_motion) /
                len(input_motion) if input_motion else 0.0)
        for model in MODELS}
    return {
        "run_id": int(ledger_row["run_id"]),
        "scenario_id": ledger_row["scenario_id"],
        "method": manifest["method"],
        "speed_mm_s": float(manifest["configured_speed_mm_s"]),
        "repeat": int(ledger_row["repeat"]),
        "primary_recall": sum(row["confirmed"] for row in primary) / len(primary),
        "all_six_recall": sum(row["confirmed"] for row in model_metrics) / len(model_metrics),
        "confirmed_primary_count": sum(row["confirmed"] for row in primary),
        "confirmed_all_count": sum(row["confirmed"] for row in model_metrics),
        "target_dynamic_points": target_points,
        "other_dynamic_points": other_points,
        "all_dynamic_points": all_dynamic_points,
        "dynamic_point_precision": (target_points / all_dynamic_points
                                    if all_dynamic_points else None),
        "baseline_false_frame_fraction": (
            sum(frame["point_count"] >= minimum_points for frame in baseline) /
            len(baseline) if baseline else None),
        "baseline_false_points_per_min": (
            sum(frame["point_count"] for frame in baseline) * 6.0),
        "motion_other_points_per_min": other_points * 1.5,
        "hold_positive_frame_fraction": (
            sum(frame["point_count"] >= minimum_points for frame in hold) /
            len(hold) if hold else None),
        "runtime_mean_ms": mean(runtime_ms),
        "runtime_p95_ms": percentile(runtime_ms, 95),
        "runtime_max_ms": max(runtime_ms) if runtime_ms else None,
        "runtime_frame_count": len(runtime_ms),
        "odometry_max_translation_drift_m": max(odom_translation) if odom_translation else None,
        "odometry_final_translation_drift_m": odom_translation[-1] if odom_translation else None,
        "odometry_max_rotation_drift_deg": (
            math.degrees(max(odom_rotation)) if odom_rotation else None),
        "minimum_target_visibility_fraction": min(visibility.values()),
        "run_directory": str(run_dir),
        "runtime_samples_ms": runtime_ms,
        "model_metrics": model_metrics,
        "detection_frames": detections,
    }


def ratio(records, numerator, denominator):
    den = sum(record[denominator] for record in records)
    return sum(record[numerator] for record in records) / den if den else None


def summarize(runs, bootstrap_samples, seed):
    rng = np.random.default_rng(seed)
    summaries = []
    for method in METHODS:
        for speed in SPEEDS:
            selected = [run for run in runs
                        if run["method"] == method and run["speed_mm_s"] == speed]
            if len(selected) != 10:
                raise SystemExit("expected 10 runs for {} {} mm/s".format(method, speed))
            metrics = {
                "primary_recall": lambda rows: mean(r["primary_recall"] for r in rows),
                "all_six_recall": lambda rows: mean(r["all_six_recall"] for r in rows),
                "dynamic_point_precision": lambda rows: ratio(
                    rows, "target_dynamic_points", "all_dynamic_points"),
                "baseline_false_frame_fraction": lambda rows: mean(
                    r["baseline_false_frame_fraction"] for r in rows),
                "runtime_mean_ms": lambda rows: mean(r["runtime_mean_ms"] for r in rows),
            }
            summary = {"method": method, "method_label": METHOD_LABELS[method],
                       "speed_mm_s": speed, "n_runs": len(selected)}
            for name, function in metrics.items():
                estimate = function(selected)
                low, high = bootstrap_ci(selected, function, rng, bootstrap_samples)
                summary[name] = estimate
                summary[name + "_ci95_low"] = low
                summary[name + "_ci95_high"] = high
            delays = [metric["confirmation_latency_s"]
                      for run in selected for metric in run["model_metrics"]
                      if metric["model_name"] in PRIMARY_MODELS and metric["confirmed"]]
            summary.update({
                "confirmation_latency_median_s": percentile(delays, 50),
                "confirmation_latency_p95_s": percentile(delays, 95),
                "runtime_p95_ms_pooled": percentile(
                    [value for run in selected for value in run["runtime_samples_ms"]], 95),
                "runtime_max_ms": max(run["runtime_max_ms"] for run in selected),
                "baseline_false_points_per_min_mean": mean(
                    run["baseline_false_points_per_min"] for run in selected),
                "motion_other_points_per_min_mean": mean(
                    run["motion_other_points_per_min"] for run in selected),
                "hold_positive_frame_fraction_mean": mean(
                    run["hold_positive_frame_fraction"] for run in selected),
                "odometry_max_translation_drift_m_mean": mean(
                    run["odometry_max_translation_drift_m"] for run in selected),
                "minimum_target_visibility_fraction": min(
                    run["minimum_target_visibility_fraction"] for run in selected),
            })
            summaries.append(summary)
    return summaries


def object_summaries(runs):
    rows = []
    for method in METHODS:
        for speed in SPEEDS:
            selected = [run for run in runs
                        if run["method"] == method and run["speed_mm_s"] == speed]
            for model in MODELS:
                metrics = [next(item for item in run["model_metrics"]
                                if item["model_name"] == model) for run in selected]
                delays = [row["confirmation_latency_s"] for row in metrics
                          if row["confirmed"]]
                successes = sum(row["confirmed"] for row in metrics)
                wilson_low, wilson_high = wilson_interval(successes, len(metrics))
                rows.append({
                    "method": method, "speed_mm_s": speed, "model_name": model,
                    "confirmed_run_count": successes,
                    "detection_probability": successes / 10.0,
                    "detection_probability_wilson_low": wilson_low,
                    "detection_probability_wilson_high": wilson_high,
                    "raw_positive_frame_fraction_mean": mean(
                        row["raw_positive_frame_fraction"] for row in metrics),
                    "confirmation_latency_median_s": percentile(delays, 50),
                    "confirmation_displacement_median_mm": percentile(
                        [row["confirmation_displacement_mm"] for row in metrics
                         if row["confirmed"]], 50),
                    "maximum_dynamic_points": max(row["maximum_dynamic_points"]
                                                  for row in metrics),
                })
    return rows


def load_alert_batch(batch, bootstrap_samples, seed):
    """Load ALERT's native confirmed-track results without redefining its detector."""
    batch = batch.resolve()
    summary = json.loads((batch / "batch_summary.json").read_text())
    ledger = read_csv(batch / "batch_status.csv")
    if (summary.get("status") != "COMPLETED"
            or summary.get("successful_run_count") != 40
            or len(ledger) != 40
            or any(row["status"] != "SUCCESS" for row in ledger)):
        raise SystemExit("ALERT batch is not 40/40 complete: {}".format(batch))
    published = {float(row["speed_mm_s"]): row for row in read_csv(
        batch / "analysis_icra/speed_condition_index.csv")}
    events = []
    runtime_rows = []
    runtime_by_speed = {speed: [] for speed in SPEEDS}
    for record in ledger:
        speed = float(record["speed_mm_s"])
        run_dir = pathlib.Path(record["run_directory"])
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        SOURCES.add(run_dir / "run_manifest.json")
        checks = manifest.get("dual_lidar_success_contract", {}).get("checks", {})
        if not checks.get("secondary_enabled") or manifest.get("truth_used_by_detector"):
            raise SystemExit("ALERT sensor/truth contract mismatch: {}".format(run_dir))
        model_events = read_csv(run_dir / "analysis/model_detection_event_metrics.csv")
        primary = [row for row in model_events
                   if boolean(row["included_in_primary_metrics"])]
        if {row["model_name"] for row in primary} != set(PRIMARY_MODELS):
            raise SystemExit("ALERT primary targets differ from baselines: {}".format(run_dir))
        if len(model_events) != 6 or {row["model_name"] for row in model_events} != set(MODELS):
            raise ValueError("ALERT model event rows are incomplete: {}".format(run_dir))
        for row in model_events:
            latency = number(row.get("confirmation_latency_s"))
            detected = boolean(row.get("detected"))
            if detected and latency is None:
                raise ValueError("confirmed ALERT event has no finite time: {}".format(run_dir))
            events.append({"method": "alert", "speed_mm_s": speed,
                           "model_name": row["model_name"], "detected": detected,
                           "repeat": int(record["repeat"]),
                           "scenario_id": record["scenario_id"],
                           "source_file": str(run_dir / "analysis/model_detection_event_metrics.csv"),
                           "included_in_primary_metrics": row["model_name"] in PRIMARY_MODELS,
                           "confirmation_latency_s": latency if detected else None})
        cutoff = number(manifest.get("deformation_capture_cutoff_accepted_frame_index"))
        truth = read_csv(run_dir / "truth/m01_m06_motion_truth.csv")
        truth_origins = [float(row["truth_sim_time_s"]) - float(row["experiment_elapsed_s"])
                         for row in truth if row["model_name"] == "m01"]
        origin = statistics.median(truth_origins)
        if max(abs(value - origin) for value in truth_origins) > 0.002:
            raise ValueError("inconsistent ALERT experiment clock: {}".format(run_dir))
        cycle_times = []
        for row in read_csv(run_dir / "runtime/per_cycle.csv"):
            stamp = number(row.get("last_attempt_time_s"))
            frame_index = number(row.get("last_attempt_frame_index"))
            elapsed = stamp - origin if stamp is not None else None
            if (elapsed is None or not 0.0 <= elapsed <= 60.0
                    or (cutoff is not None and (frame_index is None or frame_index > cutoff))):
                continue
            value = number(row.get("processing_time_ms"))
            if value is None or value <= 0:
                raise ValueError("invalid ALERT runtime: {}".format(run_dir))
            cycle_times.append(value)
        if not cycle_times:
            raise ValueError("no ALERT runtime in the experiment window: {}".format(run_dir))
        run_runtime = mean(cycle_times)
        runtime_by_speed[speed].append(run_runtime)
        runtime_rows.append({"method": "alert", "speed_mm_s": speed,
                             "repeat": int(record["repeat"]),
                             "scenario_id": record["scenario_id"],
                             "source_file": str(run_dir / "runtime/per_cycle.csv"),
                             "runtime_mean_ms": run_runtime,
                             "runtime_p95_ms": percentile(cycle_times, 95),
                             "runtime_frame_count": len(cycle_times),
                             "runtime_samples_ms": cycle_times})

    summaries = []
    rng = np.random.default_rng(seed + 100)
    for speed in SPEEDS:
        selected = [row for row in events if row["speed_mm_s"] == speed
                    and row["included_in_primary_metrics"]]
        if len(selected) != 50 or len(runtime_by_speed[speed]) != 10:
            raise SystemExit("ALERT condition is not 5 targets x 10 runs: {} mm/s".format(speed))
        successes = sum(row["detected"] for row in selected)
        recall = successes / len(selected)
        latencies = [row["confirmation_latency_s"] for row in selected
                     if row["confirmation_latency_s"] is not None]
        runtime_records = [{"value": value} for value in runtime_by_speed[speed]]
        runtime_estimate = mean(row["value"] for row in runtime_records)
        runtime_low, runtime_high = bootstrap_ci(
            runtime_records, lambda rows: mean(row["value"] for row in rows),
            rng, bootstrap_samples)
        published_row = published[speed]
        if not math.isclose(recall, float(published_row["target_recall"]), abs_tol=1e-12):
            raise SystemExit("ALERT recall disagrees with its published analysis: {} mm/s".format(speed))
        summaries.append({
            "method": "alert", "method_label": METHOD_LABELS["alert"],
            "speed_mm_s": speed,
            "n_runs": 10, "primary_recall": recall,
            "confirmation_latency_median_s": percentile(latencies, 50),
            "confirmation_latency_p95_s": percentile(latencies, 95),
            "runtime_mean_ms": runtime_estimate,
            "runtime_mean_ms_ci95_low": runtime_low,
            "runtime_mean_ms_ci95_high": runtime_high,
            "native_precision": number(published_row.get("pooled_track_precision")),
            "native_precision_definition": "confirmed-track precision",
            "native_false_alarm_rate": number(
                published_row.get("false_alarm_rate_tracks_per_min")),
            "native_false_alarm_definition": "false confirmed tracks/min",
        })
    return summaries, events, runtime_rows


def temporal_rows(runs, minimum_points):
    rows = []
    for method in METHODS:
        for speed in SPEEDS:
            selected = [run for run in runs
                        if run["method"] == method and run["speed_mm_s"] == speed]
            for second in range(60):
                positive = total = 0
                for run in selected:
                    for frame in run["detection_frames"]:
                        if second <= frame["elapsed_s"] < second + 1:
                            for model in PRIMARY_MODELS:
                                positive += frame["counts"].get(model, 0) >= minimum_points
                                total += 1
                rows.append({"method": method, "speed_mm_s": speed,
                             "time_s": second + 0.5,
                             "positive_target_frame_fraction": positive / total if total else None})
    return rows


def confirmation_event_rows(runs, alert_events):
    events = list(alert_events)
    for run in runs:
        for metric in run["model_metrics"]:
            events.append({
                    "method": run["method"], "speed_mm_s": run["speed_mm_s"],
                    "model_name": metric["model_name"],
                    "repeat": run["repeat"],
                    "scenario_id": run["scenario_id"],
                    "source_file": str(pathlib.Path(run["run_directory"]) /
                                       "detections/detection_frames.csv.gz"),
                    "included_in_primary_metrics": metric["model_name"] in PRIMARY_MODELS,
                    "detected": metric["confirmed"],
                    "confirmation_latency_s": metric["confirmation_latency_s"],
                })
    keys = [(row["method"], row["speed_mm_s"], row["repeat"], row["model_name"])
            for row in events]
    expected = {(method, speed, repeat, model) for method in ALL_METHODS
                for speed in SPEEDS for repeat in range(1, 11) for model in MODELS}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("events must contain exactly 4 methods x 4 speeds x 10 runs x 6 targets")
    for row in events:
        row["native_detected"] = row["detected"]
        row["native_confirmation_latency_s"] = row["confirmation_latency_s"]
        latency = row["confirmation_latency_s"]
        if row["detected"] and (latency is None or not math.isfinite(latency)):
            raise ValueError("confirmation without a finite latency")
        row["detected"] = bool(row["detected"] and 0.0 <= latency < 40.0)
        row["confirmation_latency_s"] = latency if row["detected"] else None
        row["observation_horizon_s"] = 40.0
    return events


def cluster_mean_interval(values, samples, seed):
    """Resample complete runs; columns may be metrics or exact event times."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] != 10 or not np.isfinite(values).all():
        raise ValueError("cluster statistics require 10 finite run-level observations")
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    weights = np.random.default_rng(seed).multinomial(
        10, np.full(10, 0.1), size=samples) / 10.0
    estimates = weights @ values
    low, high = np.percentile(estimates, (2.5, 97.5), axis=0)
    return values.mean(axis=0), low, high


def shared_statistics(events, runtime_rows, bootstrap_samples, seed):
    """Run-level inference and exact right-continuous confirmation curves."""
    summaries, curves, run_metrics = [], [], []
    for method_index, method in enumerate(ALL_METHODS):
        for speed_index, speed in enumerate(SPEEDS):
            selected = [row for row in events if row["method"] == method
                        and row["speed_mm_s"] == speed
                        and row["included_in_primary_metrics"]]
            runtime = sorted([row for row in runtime_rows if row["method"] == method
                              and row["speed_mm_s"] == speed], key=lambda row: row["repeat"])
            if len(selected) != 50 or [row["repeat"] for row in runtime] != list(range(1, 11)):
                raise ValueError("incomplete experiment condition: {} {}".format(method, speed))
            latencies = [row["confirmation_latency_s"] for row in selected if row["detected"]]
            times = np.unique([0.0, 40.0] + latencies)
            trajectories, values = [], []
            for repeat in range(1, 11):
                items = [row for row in selected if row["repeat"] == repeat]
                if len(items) != 5:
                    raise ValueError("each run must contain five primary targets")
                delays = np.asarray([row["confirmation_latency_s"] if row["detected"]
                                     else np.inf for row in items])
                trajectories.append(np.mean(delays[:, None] <= times, axis=0))
                rt = runtime[repeat - 1]
                row = {"method": method, "speed_mm_s": speed, "repeat": repeat,
                       "scenario_id": rt["scenario_id"],
                       "primary_recall": float(np.isfinite(delays).mean()),
                       "restricted_mean_wait_s": float(np.minimum(delays, 40.0).mean()),
                       "runtime_mean_ms": rt["runtime_mean_ms"],
                       "runtime_p95_ms": rt["runtime_p95_ms"],
                       "runtime_frame_count": rt["runtime_frame_count"]}
                run_metrics.append(row)
                values.append([row[key] for key in (
                    "primary_recall", "restricted_mean_wait_s", "runtime_mean_ms")])
            joined = np.column_stack((values, trajectories))
            estimate, low, high = cluster_mean_interval(
                joined, bootstrap_samples, seed + 100 * method_index + speed_index)
            summary = {"method": method, "method_label": METHOD_LABELS[method],
                       "speed_mm_s": speed, "n_runs": 10, "primary_target_run_count": 50,
                       "confirmed_count": len(latencies), "unconfirmed_count": 50 - len(latencies),
                       "confirmation_latency_median_s": percentile(latencies, 50),
                       "confirmation_latency_p95_s": percentile(latencies, 95),
                       "runtime_p95_ms_pooled": percentile(
                           [v for r in runtime for v in r["runtime_samples_ms"]], 95),
                       "runtime_frame_count": sum(r["runtime_frame_count"] for r in runtime)}
            for column, name in enumerate(("primary_recall", "restricted_mean_wait_s",
                                            "runtime_mean_ms")):
                summary[name] = float(estimate[column])
                summary[name + "_ci95_low"] = float(low[column])
                summary[name + "_ci95_high"] = float(high[column])
            summaries.append(summary)
            for column, time in enumerate(times, 3):
                count = sum(delay <= time for delay in latencies)
                curves.append({"method": method, "speed_mm_s": speed,
                               "motion_elapsed_s": float(time), "confirmed_count": count,
                               "target_run_count": 50,
                               "cumulative_confirmation_fraction": count / 50.0,
                               "cluster_ci95_low": float(low[column]),
                               "cluster_ci95_high": float(high[column])})
    return summaries, curves, run_metrics


def object_detection_rows(events):
    rows = []
    for method in ALL_METHODS:
        for speed in SPEEDS:
            for model in MODELS:
                selected = [row for row in events if row["method"] == method
                            and row["speed_mm_s"] == speed
                            and row["model_name"] == model]
                if len(selected) != 10:
                    raise SystemExit("expected 10 event units for {} {} {}".format(
                        method, speed, model))
                successes = sum(row["detected"] for row in selected)
                low, high = wilson_interval(successes, len(selected))
                rows.append({
                    "method": method, "speed_mm_s": speed, "model_name": model,
                    "included_in_primary_metrics": model in PRIMARY_MODELS,
                    "confirmed_run_count": successes,
                    "detection_probability": successes / len(selected),
                    "wilson_low": low, "wilson_high": high,
                    "confirmation_latency_median_s": percentile(
                        [row["confirmation_latency_s"] for row in selected
                         if row["confirmation_latency_s"] is not None], 50),
                })
    return rows


def interpolate_truth(times, values, when):
    """Interpolate recorded truth only, never extrapolate or substitute a command."""
    times, values = np.asarray(times), np.asarray(values)
    if (len(times) < 2 or not np.isfinite(times).all() or not np.isfinite(values).all()
            or np.any(np.diff(times) <= 0) or not times[0] <= when <= times[-1]):
        raise ValueError("invalid truth samples or event outside recorded support")
    return float(np.interp(when, times, values))


def evidence_data(events, runs, output, samples, seed):
    """Physical-scale events and full-process observations; all runs, no case selection."""
    groups = {}
    for event in events:
        groups.setdefault((event["method"], event["speed_mm_s"], event["repeat"]), []).append(event)
    response, tracking, accuracy = [], [], []
    by_run = {(r["method"], r["speed_mm_s"], r["repeat"]): r for r in runs}
    for (method, speed, repeat), items in groups.items():
        run_dir = pathlib.Path(items[0]["source_file"]).parent.parent
        curves = {model: {} for model in MODELS}
        rotations = {model: [] for model in MODELS}
        if method == "alert":
            truth_path = run_dir / "truth/m01_m06_motion_truth.csv"
            truth = read_csv(truth_path)
            model_events = read_csv(run_dir / "analysis/model_detection_event_metrics.csv")
            for row in truth:
                model = row["model_name"]
                curves[model][float(row["experiment_elapsed_s"])] = 1000 * float(row["translation_magnitude_m"])
                rotations[model].append(float(row["rotation_angle_rad"]))
        else:
            truth_path = run_dir / "truth/motion_truth.jsonl.gz"
            SOURCES.add(truth_path)
            with gzip.open(truth_path, "rt") as stream:
                for line in stream:
                    row = json.loads(line)
                    for model in row["models"]:
                        name = model["model_name"]
                        curves[name][float(row["experiment_elapsed"])] = 1000 * float(model["translation_magnitude"])
                        rotations[name].append(float(model["rotation_angle"]))
        for event in items:
            model = event["model_name"]
            pairs = sorted(curves[model].items())
            times, values = zip(*pairs)
            onset = 10.0
            if method == "alert":
                # Use the recorder's model-specific source-clock motion origin.
                native = next(r for r in model_events if r["model_name"] == model)
                onset = float(native["truth_activation_experiment_elapsed_s"])
            event.update({
                "truth_source_file": str(truth_path),
                "motion_onset_experiment_s": onset,
                "maximum_recorded_rotation_rad": max(rotations[model]),
                "truth_translation_at_confirmation_mm": (
                    interpolate_truth(times, values, onset + event["confirmation_latency_s"])
                    if event["detected"] else None),
                "truth_translation_at_window_end_mm": interpolate_truth(times, values, onset + 40),
                "pre_hold_translation_mm": interpolate_truth(times, values, onset + 39.9),
                "post_hold_translation_mm": interpolate_truth(times, values, onset + 40.1),
            })
            event["hold_transition_change_mm"] = event["post_hold_translation_mm"] - event["pre_hold_translation_mm"]
            if method == "alert":
                curves[model] = (times, values)
            else:
                metric = next(m for m in by_run[(method, speed, repeat)]["model_metrics"]
                              if m["model_name"] == model)
                metric["confirmation_displacement_mm"] = event["truth_translation_at_confirmation_mm"]
        if method != "alert":
            for frame in by_run[(method, speed, repeat)]["detection_frames"]:
                if not 0 <= frame["elapsed_s"] <= 60:
                    continue
                for model in MODELS:
                    if frame["counts"][model] > 0:
                        response.append({"method": method, "speed_mm_s": speed, "repeat": repeat,
                                         "model_name": model, "experiment_elapsed_s": frame["elapsed_s"],
                                         "signal": "dynamic_point_output", "count": frame["counts"][model],
                                         "source_file": items[0]["source_file"]})
            continue
        comparison_path = run_dir / "analysis/object_motion_comparison.csv"
        series, _ = stationary.build_displacement_timeseries(read_csv(comparison_path), truth)
        # Reuse the existing GDU/frame aggregation, including diagnostic rows. No truth-guided
        # selection of the best GDU, and no interpolation of missing detector estimates.
        formal = [r for r in series if r["series_type"] == "FORMAL_TRUE_POSITIVE"
                  and 0 <= r["experiment_elapsed_s"] <= 60]
        for row in formal:
            response.append({"method": method, "speed_mm_s": speed, "repeat": repeat,
                             "model_name": row["model_name"],
                             "experiment_elapsed_s": row["experiment_elapsed_s"],
                             "signal": "formal_displacement_estimate", "count": row["confirmed_gdu_count"],
                             "source_file": str(comparison_path)})
        # Half-second display bins: aggregate within each run FIRST. Empty bins stay missing.
        for model in MODELS:
            event = next(e for e in items if e["model_name"] == model)
            selected = [r for r in formal if r["model_name"] == model]
            confirmation_time = (event["motion_onset_experiment_s"] + event["confirmation_latency_s"]
                                 if event["detected"] else None)
            valid = [r for r in selected if r["truth_magnitude_mm"] >= 5
                     and confirmation_time is not None and r["experiment_elapsed_s"] >= confirmation_time]
            errors = [r["estimated_magnitude_median_mm"] - r["truth_magnitude_mm"] for r in valid]
            accuracy.append({"speed_mm_s": speed, "repeat": repeat, "model_name": model,
                             "detected": event["detected"], "tracking_samples": len(errors),
                             "magnitude_rmse_mm": math.sqrt(mean(v*v for v in errors)) if errors else None,
                             "magnitude_bias_mm": mean(errors), "truth_floor_mm": 5,
                             "source_file": str(comparison_path)})
            for k in range(120):
                t = .5 * k + .25
                observed = [r for r in selected if .5*k <= r["experiment_elapsed_s"] < .5*(k+1)]
                tracking.append({"speed_mm_s": speed, "repeat": repeat, "model_name": model,
                                 "experiment_elapsed_s": t,
                                 "truth_translation_mm": interpolate_truth(*curves[model], t),
                                 "estimate_mm": percentile([r["estimated_magnitude_median_mm"] for r in observed], 50),
                                 "matched_truth_mm": percentile([r["truth_magnitude_mm"] for r in observed], 50),
                                 "sample_count": len(observed), "source_file": str(comparison_path)})
    displacement_curves, displacement_summary = [], []
    for mi, method in enumerate(ALL_METHODS):
        for si, speed in enumerate(SPEEDS):
            selected = [e for e in events if e["method"] == method and e["speed_mm_s"] == speed
                        and e["included_in_primary_metrics"]]
            values = [e["truth_translation_at_confirmation_mm"] for e in selected if e["detected"]]
            end = max(e["truth_translation_at_window_end_mm"] for e in selected)
            grid = np.unique([0., end] + values)
            run_values = [np.mean([[e["detected"] and e["truth_translation_at_confirmation_mm"] <= d
                                   for d in grid] for e in selected if e["repeat"] == repeat], axis=0)
                          for repeat in range(1, 11)]
            estimate, low, high = cluster_mean_interval(run_values, samples, seed + mi*100 + si)
            for j, d in enumerate(grid):
                displacement_curves.append({"method": method, "speed_mm_s": speed,
                                            "translation_mm": float(d), "confirmed_fraction": float(estimate[j]),
                                            "ci95_low": float(low[j]), "ci95_high": float(high[j])})
            displacement_summary.append({"method": method, "speed_mm_s": speed,
                                         "confirmed_count": len(values), "event_count": 50,
                                         "translation_median_mm": percentile(values, 50),
                                         "translation_q25_mm": percentile(values, 25),
                                         "translation_q75_mm": percentile(values, 75),
                                         "window_end_translation_min_mm": min(e["truth_translation_at_window_end_mm"] for e in selected),
                                         "window_end_translation_max_mm": end})
    for name, rows in (("confirmation_displacement_curves", displacement_curves),
                       ("confirmation_displacement_summary", displacement_summary),
                       ("response_samples", response), ("alert_tracking_bins", tracking),
                       ("alert_accuracy_runs", accuracy)):
        write_csv(output / (name + ".csv"), rows)
    return displacement_curves, displacement_summary, response, tracking, accuracy




def latex_table(summaries, path):
    def fmt(value, digits=2):
        return "--" if value is None else ("{:.%df}" % digits).format(value)
    lines = [
        r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Speed & Method & Recall (\%) & Wait (s) & Latency (s) & Mean (ms) & P95 (ms) \\",
        r"\midrule",
    ]
    for speed in SPEEDS:
        for row in (r for r in summaries if r["speed_mm_s"] == speed):
            lines.append("{} & {} & {} & {} & {} & {} & {} \\\\".format(
                "{:g}".format(speed), row["method_label"],
                fmt(100 * row["primary_recall"], 1), fmt(row["restricted_mean_wait_s"]),
                fmt(row["confirmation_latency_median_s"]), fmt(row["runtime_mean_ms"], 1),
                fmt(row["runtime_p95_ms_pooled"], 1)))
        lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def sensitivity(runs, output, minimum_points, maximum_gap_s):
    # Reuse the decoded frames; changing persistence does not require reading odometry/truth again.
    rows = []
    for persistence in (1, 3, 5):
        for method in METHODS:
            for speed in SPEEDS:
                selected = [r for r in runs if r["method"] == method and r["speed_mm_s"] == speed]
                primary, all_six = [], []
                for run in selected:
                    confirmed = {}
                    for model in MODELS:
                        frames = [dict(f, point_count=f["counts"][model])
                                  for f in run["detection_frames"] if 10.0 <= f["elapsed_s"] < 50.0]
                        confirmed[model] = confirmation(
                            frames, minimum_points, persistence, maximum_gap_s)[1] is not None
                    primary.append(sum(confirmed[m] for m in PRIMARY_MODELS) / 5)
                    all_six.append(sum(confirmed.values()) / 6)
                rows.append({"method": method, "speed_mm_s": speed,
                             "consecutive_frames": persistence, "n_runs": len(selected),
                             "primary_recall": mean(primary), "all_six_recall": mean(all_six)})
    write_csv(output / "confirmation_sensitivity.csv", rows)
    return rows


def report(summaries, output, args, evidence):
    physical, accuracy = evidence[1], evidence[4]
    def f(v, digits=2):
        return "N/A" if v is None else ("{:.%df}" % digits).format(v)
    lines = [
        "# 微小形变：检测尺度、响应过程与测量能力", "",
        "4方法 × 4速度 × 10次运行；所有速度独立分析，原始实验记录保持不变。", "",
        "## 阅读顺序", "",
        "1. [毫米尺度确认](fig_deformation_scale.pdf)：真实位移达到什么量级时形成确认。",
        "2. [原始响应与持续确认](fig_response_persistence.pdf)：零星响应不等于完全无响应。",
        "3. 各速度的逐次响应时间线与六目标位移过程：完整保留失败、缺测和后期误差。",
        "4. [计算代价](fig_computation_cost.pdf)：保留基线效率优势，不构造综合总分。", "",
        "打开[11页完整图册](FIGURE_GALLERY.pdf)，或使用[逐图解释与英文图注](FIGURE_CAPTIONS.md)。",
        "其中毫米尺度、响应持续性和计算代价为比较总览，其余8页为四速度分别展开的全程证据。", "",
        "## 同速度结果", "",
        "| 速度 mm/s | 方法 | 确认/50 | 确认真值平移中位 mm | 成功确认延迟中位 s | 限制平均等待 s | 检测均值 ms | 合并P95 ms |",
        "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for speed in SPEEDS:
        for r in (r for r in summaries if r["speed_mm_s"] == speed):
            d = next(d for d in physical if d["method"] == r["method"] and d["speed_mm_s"] == speed)
            lines.append("| {:g} | {} | {} | {} | {} | {} | {} | {} |".format(
                speed, r["method_label"], r["confirmed_count"], f(d["translation_median_mm"]),
                f(r["confirmation_latency_median_s"]), f(r["restricted_mean_wait_s"]),
                f(r["runtime_mean_ms"],1), f(r["runtime_p95_ms_pooled"],1)))
    lines += ["", "## 不能混淆的统计含义", "",
        "- 主目标M01/M02/M03/M05/M06共800事件，M04的160事件单列辅助诊断；不因漏检删除目标。",
        "- 基线正式确认：至少{}个动态点、连续{}帧、帧间隔不超过{} s；ALERT使用原生确认。原生确认与三帧规则不是相同滤波器。".format(args.minimum_points,args.consecutive_frames,args.maximum_gap_s),
        "- 1/3/5帧对照仅改变离线评价规则，不是算法参数消融。ST normals的单帧响应必须展示。",
        "- 确认窗口为运动开始后[0,40) s。未确认仍在50事件分母中，位移/成功延迟中位数只针对成功事件。",
        "- 首次确认位移使用记录的translation_magnitude按实际事件时刻插值，不使用设定速度乘时间。",
        "- 该量是模型参考点平移，不是最小可检测形变MDE，不是测量误差，也不是旋转时所有表面点的共同位移。",
        "- 部分低速记录含非零实际转动；confirmation_events.csv保留每目标最大转角，不能声称这些记录严格纯平移。",
        "- 三阶段底色标注的是指令流程，并非实测真值一定严格遵守。事件CSV另存49.9/50.1 s附近的参考点平移及其差值，保留阶段交界的实际跳变。",
        "- 位移图黑虚线为参考点平移；棕虚线为记录器关联测量点真值。存在转动时两者不强制相等。",
        "- 位移尺度图是首次确认平移的经验累计分布，不是坍塌概率或任意运动轨迹的检测概率模型。",
        "- 位移曲线止于该条件记录的最大窗口末平移，空白量程不外推。最小/最大末平移同时写入CSV。",
        "- 位移分布置信带以完整运行为簇进行{}次bootstrap，保持同次五目标相关性；为逐点区间。".format(args.bootstrap_samples),
        "- 全成功/全失败时经验区间可能退化，不意味着总体概率已被精确确定。逐目标Wilson区间仍在target_response_summary.csv。",
        "- 限制平均等待对未确认计40 s，但绝不把它们解释为40 s检出；成功确认延迟另列。",
        "- 时间线刻痕为基线实际动态点输出或ALERT正式位移估计，两种输出语义不同。无刻痕表示没有该类记录，不自动证明环境静止。",
        "- 50–60 s指令保持阶段的偏移仍存在；ALERT保留偏移估计不能直接算成误报。部分记录真值在50 s附近发生跳变，因此不能把全部后期误差都归因为匀速跟踪失败。",
        "- 估计先按GDU/源帧聚合，再在每次运行内按0.5 s分箱取中位数；缺测不补零、不跨空箱插值。",
        "- 细线覆盖全部十次运行，粗线和阴影为可用运行的中位数与IQR，不是置信带；下方n/10条带显示逐时有效运行比例。",
        "- 位移精度仍用未分箱源帧计算：正式关联、确认后、匹配真值幅值≥5 mm；只评价有效估计，缺失不是零误差。",
        "- 检测耗时按源时间选择0–60 s；ALERT遵守采集冻结截止。总体P95来自实际周期样本，非各次P95的P95。",
        "", "## ALERT连续测量的局限也保留", "",
        "下面是每目标每次运行的幅值RMSE，再在同速度同目标的有效运行间求均值；不是全目标RMSE，也不是三维向量误差。",
        "确认时的毫米级位移不能替代全过程精度；M04和未确认运行保持缺失状态。", "",
        "| 速度 mm/s | 目标 | 有效运行/10 | 确认后幅值RMSE均值 mm |",
        "|---:|---|---:|---:|"]
    for speed in SPEEDS:
        for model in MODELS:
            vals = [r["magnitude_rmse_mm"] for r in accuracy if r["speed_mm_s"] == speed
                    and r["model_name"] == model and r["magnitude_rmse_mm"] is not None]
            lines.append("| {:g} | {} | {} | {} |".format(speed,model,len(vals),f(mean(vals))))
    lines += ["", "## 原生指标分开解释", "",
        "动态点精确率与确认轨迹精确率不可直接相互比较，不计算共同PR/F1或基线位移RMSE。", "",
        "| 方法 | 速度 mm/s | 原生精确率 | 定义 | 原生误报量 | 定义 |",
        "|---|---:|---:|---|---:|---|"]
    for r in summaries:
        lines.append("| {} | {:g} | {} | {} | {} | {} |".format(r["method_label"],
            r["speed_mm_s"],f(r["native_precision"],4),r["native_precision_definition"],
            f(r["native_false_alarm_rate"],4),r["native_false_alarm_definition"]))
    lines += ["", "## 完整来源", "",
        "source_data_manifest.csv记录实际读取文件的SHA-256；confirmation_events.csv逐事件列出检测与Gazebo真值源文件。",
        "alert_tracking_bins.csv及alert_accuracy_runs.csv列出测量比较源文件。未筛选最佳运行，未更改检测阈值。",
        "source hash证明本次读取内容，不替代采集时未记录的代码版本信息。",
        "analysis_manifest.json记录统计定义、生成器哈希和完整面板检查；RAW_DATA_AUDIT.md保留此前原始记录审计。",
        "旧重复图仅按明确文件名清理；CSV、原始检测、Gazebo真值和审计记录保留。", ""]
    (output / "REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    captions = {
        "fig_deformation_scale": ("confirmation_displacement_curves.csv; confirmation_displacement_summary.csv; confirmation_events.csv",
            "Cumulative distribution of recorded model-reference translation at first confirmation, separately by speed. All 50 primary target-run events remain in the denominator, including misses. Curves terminate at the largest recorded end-of-window translation. The logarithmic axis displays both millimetre-scale and late responses without truncation. Shading is a pointwise 95% complete-run bootstrap interval. Annotated medians are conditional on confirmation; translation is neither measurement error nor a calibrated minimum detectable deformation."),
        "fig_response_persistence": ("confirmation_sensitivity.csv; integrated_method_speed_summary.csv",
            "Raw target-event response versus persistence of baseline outputs. Open circles, filled circles and ticks denote at least one, three and five consecutive positive frames, respectively. Text lists the three percentages in that order. Diamonds show ALERT native confirmations separately; its decision rule is not replaced by the baseline persistence rule."),
        "fig_computation_cost": ("runtime_run_distribution.csv",
            "Detector computation by speed. Each dot is one run, horizontal ranges span all ten runs, and method symbols mark medians. Panel (a) uses within-run means; panel (b) uses within-run P95 values, not a pooled percentile. Source timestamps select the 0--60 s experiment window. No speeds are pooled."),
    }
    text = ["# 论文图件：阅读顺序与可复算数据", "", "各页完整铺满1×2、2×2或2×3，无空置面板。PDF固定7.16英寸宽并嵌入字体；PNG为600 dpi。", ""]
    latex = ["% Requires graphicx; paths relative to this analysis directory.", ""]
    for name, (_, _, description) in compare_figures.FIGURES.items():
        if name.startswith("fig_response_timeline_"):
            src = "response_samples.csv; confirmation_events.csv"
            caption = "Full-process response at the speed stated in the figure. Each target has ten fixed lanes in repeat order. Ticks show actual baseline dynamic-point output or ALERT formal displacement estimates, not interchangeable scores. Open diamonds mark first confirmation; counts show confirmations out of ten runs. Shading denotes the commanded static, moving and hold phases, not a guarantee of ideal ground-truth motion. M04 is auxiliary. Missing ticks do not imply a cleared environment."
        elif name.startswith("fig_displacement_tracking_"):
            src = "alert_tracking_bins.csv; alert_accuracy_runs.csv; confirmation_events.csv"
            caption = "Full-process ALERT displacement measurements at the stated speed. Thin lines retain all ten runs; the median and interquartile band use available runs after within-run half-second binning. Missing bins remain missing. Dashed black curves show recorded reference translation; dotted brown curves show matched-point truth, which may differ under rotation. Shading denotes commanded motion phases; recorded phase-boundary jumps are retained. The lower strip encodes available runs from zero to ten, not negative displacement. Title counts indicate confirmed runs. RMSE annotations average per-run unbinned magnitude RMSE after confirmation with matched truth at least 5 mm; these are conditional on available estimates. M04 remains auxiliary without fabricated estimates."
        else:
            src, caption = captions[name]
        text += ["## " + name, "", description, "", "源数据：" + src, "", "**English caption.** " + caption,
                 "", "![]({}.png)".format(name), ""]
        latex += [r"\begin{figure*}[t]", r"\centering",
                  r"\includegraphics[width=\textwidth]{" + name + ".pdf}",
                  r"\caption{" + caption.replace("%", r"\%") + "}", r"\label{fig:" + name[4:] + "}",
                  r"\end{figure*}", ""]
    (output / "FIGURE_CAPTIONS.md").write_text("\n".join(text),encoding="utf-8")
    (output / "figures_ieee.tex").write_text("\n".join(latex),encoding="utf-8")


def source_manifest(output):
    rows = []
    for path in sorted(SOURCES):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        rows.append({"path": str(path), "size_bytes": path.stat().st_size,
                     "sha256": digest.hexdigest()})
    write_csv(output / "source_data_manifest.csv", rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=pathlib.Path)
    parser.add_argument("--alert-batch", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--minimum-points", type=int, default=1)
    parser.add_argument("--consecutive-frames", type=int, default=3)
    parser.add_argument("--maximum-gap-s", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    if args.minimum_points < 1 or args.consecutive_frames < 1:
        raise SystemExit("minimum-points and consecutive-frames must be positive")
    if args.bootstrap_samples < 1 or not math.isfinite(args.maximum_gap_s) or args.maximum_gap_s <= 0:
        raise SystemExit("bootstrap-samples and maximum-gap-s must be positive and finite")
    batch = args.batch.resolve()
    summary = json.loads((batch / "batch_summary.json").read_text())
    if summary.get("status") != "COMPLETED" or summary.get("successful_run_count") != 120:
        raise SystemExit("comparison batch is not complete")
    ledger = read_csv(batch / "batch_status.csv")
    SOURCES.update((batch / "batch_summary.json", args.alert_batch.resolve() / "batch_summary.json"))
    if len(ledger) != 120 or any(row["status"] != "SUCCESS" for row in ledger):
        raise SystemExit("batch ledger is not 120/120 SUCCESS")
    output = (args.output or batch / "analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = [analyze_run(pathlib.Path(row["run_directory"]), row,
                        args.minimum_points, args.consecutive_frames,
                        args.maximum_gap_s) for row in ledger]
    summaries = summarize(runs, args.bootstrap_samples, args.seed)
    alert_summaries, alert_events, alert_runtime = load_alert_batch(
        args.alert_batch, args.bootstrap_samples, args.seed)
    shared_summaries = []
    for row in summaries:
        shared_summaries.append(dict(
            row,
            native_precision=row["dynamic_point_precision"],
            native_precision_definition="dynamic-point precision",
            native_false_alarm_rate=row["baseline_false_frame_fraction"],
            native_false_alarm_definition="baseline positive-frame fraction"))
    shared_summaries.extend(alert_summaries)
    temporal = temporal_rows(runs, args.minimum_points)
    confirmation_events = confirmation_event_rows(runs, alert_events)
    print("Recomputing physical-scale evidence and all 40 ALERT trajectories...", flush=True)
    evidence = evidence_data(confirmation_events, runs, output, args.bootstrap_samples, args.seed)
    objects = object_summaries(runs)
    target_response = object_detection_rows(confirmation_events)
    runtime_rows = [{"method": run["method"], "speed_mm_s": run["speed_mm_s"],
                     "repeat": run["repeat"], "scenario_id": run["scenario_id"],
                     "source_file": str(pathlib.Path(run["run_directory"]) / "runtime/algorithm_runtime.csv.gz"),
                     "runtime_mean_ms": run["runtime_mean_ms"],
                     "runtime_p95_ms": run["runtime_p95_ms"],
                     "runtime_frame_count": run["runtime_frame_count"],
                     "runtime_samples_ms": run["runtime_samples_ms"]}
                    for run in runs] + alert_runtime
    common, confirmation_dynamics, integrated_runs = shared_statistics(
        confirmation_events, runtime_rows, args.bootstrap_samples, args.seed)
    for row in shared_summaries:
        row.update(next(r for r in common if r["method"] == row["method"]
                        and r["speed_mm_s"] == row["speed_mm_s"]))
    # The curve area is computed independently from the per-target waiting-time mean.
    for row in common:
        curve = [r for r in confirmation_dynamics if r["method"] == row["method"]
                 and r["speed_mm_s"] == row["speed_mm_s"]]
        area = sum((b["motion_elapsed_s"] - a["motion_elapsed_s"])
                   * (1 - a["cumulative_confirmation_fraction"])
                   for a, b in zip(curve, curve[1:]))
        if not (math.isclose(area, row["restricted_mean_wait_s"], abs_tol=1e-10)
                and math.isclose(curve[-1]["cumulative_confirmation_fraction"],
                                 row["primary_recall"], abs_tol=1e-12)):
            raise ValueError("curve and summary statistics disagree")
    flat_runs = [{key: value for key, value in run.items()
                  if key not in ("model_metrics", "detection_frames", "runtime_samples_ms")}
                 for run in runs]
    write_csv(output / "run_metrics.csv", flat_runs)
    write_csv(output / "method_speed_summary.csv", summaries)
    write_csv(output / "object_speed_summary.csv", objects)
    write_csv(output / "temporal_response.csv", temporal)
    write_csv(output / "integrated_method_speed_summary.csv", shared_summaries)
    write_csv(output / "confirmation_events.csv", confirmation_events)
    write_csv(output / "confirmation_dynamics.csv", confirmation_dynamics)
    write_csv(output / "target_response_summary.csv", target_response)
    write_csv(output / "runtime_run_distribution.csv", [
        {k: v for k, v in r.items() if k != "runtime_samples_ms"} for r in runtime_rows])
    write_csv(output / "integrated_run_metrics.csv", integrated_runs)
    sensitivity_rows = sensitivity(runs, output, args.minimum_points, args.maximum_gap_s)
    latex_table(shared_summaries, output / "table_comparison_results.tex")
    layout_checks = compare_figures.render_all(
        output, shared_summaries, confirmation_events, runtime_rows, sensitivity_rows, evidence)
    for obsolete in ("fig_object_detection_matrix.pdf",
                     "fig_object_detection_matrix.png",
                     "fig_temporal_response.pdf", "fig_temporal_response.png",
                     "fig_capability_radar.pdf", "fig_capability_radar.png", "capability_profile.csv",
                     *[stem + suffix for stem in (
                         "fig_comparison_overview", "fig_confirmation_dynamics", "fig_target_response_profiles",
                         "fig_confirmation_outcomes", "fig_runtime_distribution", "fig_confirmation_sensitivity",
                         "fig_detection_cost_tradeoff") for suffix in (".pdf", ".png")]):
        (output / obsolete).unlink(missing_ok=True)
    report(shared_summaries, output, args, evidence)
    source_count = source_manifest(output)
    manifest = {
        "schema_version": 4, "baseline_batch": str(batch),
        "alert_batch": str(args.alert_batch.resolve()),
        "baseline_run_count": len(runs), "alert_run_count": 40,
        "methods": list(ALL_METHODS), "speeds_mm_s": list(SPEEDS),
        "minimum_dynamic_points": args.minimum_points,
        "consecutive_confirmation_frames": args.consecutive_frames,
        "maximum_interframe_gap_s": args.maximum_gap_s,
        "bootstrap_samples": args.bootstrap_samples, "bootstrap_seed": args.seed,
        "displacement_accuracy_reported_for_baselines": False,
        "cross_method_native_score_comparison": False,
        "unique_track_precision_reported": False,
        "native_precision_metrics_combined_in_one_figure": False,
        "figures": list(compare_figures.FIGURES), "figure_layout_checks": layout_checks,
        "primary_recall_interval": "run-cluster percentile bootstrap; 10 complete runs per speed",
        "target_recall_interval": "Wilson over 10 runs per target and speed",
        "curve_intervals": "pointwise run-cluster bootstrap; exact event-time grid",
        "runtime_window": "source timestamp in experiment [0,60] s; ALERT freeze cutoff also enforced",
        "runtime_p95_definition": "95th percentile of actual pooled cycle durations",
        "restricted_mean_wait_horizon_s": 40.0,
        "source_file_count": source_count,
        "event_count": len(confirmation_events), "primary_event_count": 800,
        "curve_area_identity_verified": True,
        "cross_speed_aggregation": False,
        "confirmation_translation": "interpolated recorded model-reference translation; not commanded speed*time",
        "tracking_display": "0.5 s within-run bins; missing bins unfilled; all ten runs retained",
        "tracking_accuracy": "unbinned formal matched-point magnitude errors after confirmation, truth >=5 mm",
        "response_sample_count": len(evidence[2]), "tracking_bin_count": len(evidence[3]),
        "rotation_event_count": sum(r["maximum_recorded_rotation_rad"] > 1e-5 for r in confirmation_events),
        "generator_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (pathlib.Path(__file__), pathlib.Path(compare_figures.__file__),
                                        pathlib.Path(stationary.__file__))},
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("ANALYSIS_COMPLETE {} baseline + 40 ALERT runs -> {}".format(
        len(runs), output))


if __name__ == "__main__":
    main()
