#!/usr/bin/env python3
"""Reproduce paper runtime statistics; write only paper-local runtime_audit.json."""
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True
PAPER = Path(__file__).resolve().parent
SOURCE = PAPER.parent
OUTPUT = SOURCE.parent / "experiment_output"
STATIC = OUTPUT / "stationary_speed_sweep/batch_20260901_012253"
BASELINE = OUTPUT / "compare_methods_single_lidar/batch_20260901_235707"
MOBILE = OUTPUT / "evacuation_paper_run/closed_loop/icra2027_20260911_235806"
sys.path.insert(0, str(SOURCE / "analyze_script"))
import analyze_compare_methods as compare
from analyze_stationary_speed_sweep import file_sha256

PARAMETERS = (
    "/use_sim_time",
    "/deform_monitor_v1/v1/io/cloud_topic",
    "/deform_monitor_v1/v1/io/covariance_topic",
    "/deform_monitor_v1/v1/algorithm/support_partition_roles",
    "/deform_monitor_v1/v1/algorithm/support_partition_percent",
    "/deform_monitor_v1/v1/algorithm/reference_update_during_monitoring",
    "/deform_monitor_v1/v1/resource/upstream_stationary_pose_freeze_verified",
    "/deform_monitor_v1/v1/resource/frame_sync_tolerance_s",
    "/deform_monitor_v1/v1/resource/max_points_per_frame",
    "/deform_monitor_v1/v1/resource/max_raw_buffer_points",
    "/deform_monitor_v1/v1/resource/max_update_gap_s",
    "/deform_monitor_v1/v1/resource/deadline_ms",
    "/deform_monitor_v1/v1/resource/reference_min_frames",
    "/deform_monitor_v1/v1/resource/reference_min_duration_s",
    "/deform_monitor_v1/v1/resource/reference_min_surfel_exposures",
    "/deform_monitor_v1/v1/resource/pose_control_min_patches",
    "/deform_monitor_v1/v1/resource/pose_control_exclusion_radius_m",
    "/stationary_monitor_cloud_adapter/monitor_voxel_size_m",
    "/stationary_monitor_cloud_adapter/monitor_apply_radial_range_filter",
    "/icra2027_closed_loop_mission/mission/stable_dwell_s",
    "/icra2027_closed_loop_mission/mission/reference_preparation_dwell_s",
)


def read_json(path):
    compare.SOURCES.add(path)
    return json.loads(path.read_text())


def summarize(rows):
    assert len(rows) > 1
    cycles = [float(row["processing_time_ms"]) for row in rows]
    points = [float(row["input_point_count"]) for row in rows]
    assert all(math.isfinite(v) and v > 0 for v in cycles + points)
    result = {
        "cycle_count": len(rows),
        "published_cycle_count": sum(compare.boolean(r["last_attempt_published"]) for r in rows),
        "cycle_mean_ms": statistics.mean(cycles),
        "cycle_p95_ms": compare.percentile(cycles, 95),
        "input_point_count_mean": statistics.mean(points),
        "input_point_count_min": min(points),
        "input_point_count_max": max(points),
    }
    for clock, column in (("source", "last_attempt_time_s"),
                          ("receive_sim", "receive_time_s"),
                          ("wall", "wall_elapsed_s")):
        times = [float(row[column]) for row in rows]
        assert all(math.isfinite(t) for t in times)
        assert all(b > a for a, b in zip(times, times[1:])), (clock, "nonmonotone time")
        result[clock + "_first_s"] = times[0]
        result[clock + "_last_s"] = times[-1]
        result[clock + "_span_s"] = times[-1] - times[0]
        result[clock + "_rate_hz"] = (len(rows) - 1) / (times[-1] - times[0])
    delays = [float(r["receive_time_s"]) - float(r["last_attempt_time_s"]) for r in rows]
    assert min(delays) >= 0
    result["source_to_receive_sim_s"] = {
        "median": statistics.median(delays),
        "p95": compare.percentile(delays, 95), "maximum": max(delays)}
    return result


def source_evidence(relative, needles):
    path = SOURCE / relative
    compare.SOURCES.add(path)
    lines = path.read_text().splitlines()
    matches = []
    for needle in needles:
        found = [(i, line) for i, line in enumerate(lines, 1) if needle in line]
        assert found, (relative, needle)
        matches.extend({"line": i, "text": line.strip()} for i, line in found)
    return {"path": str(path), "lines": matches}


def build_audit():
    compare.SOURCES.clear()
    # Reuse the published loader as the selection oracle; no bootstrap is needed.
    _, _, original_runtime = compare.load_alert_batch(STATIC, bootstrap_samples=0, seed=0)
    compare.SOURCES.add(STATIC / "batch_summary.json")
    parameter_groups = {}
    runs, pooled = [], {speed: [] for speed in compare.SPEEDS}

    def record_parameters(directory):
        path = directory / "ros_parameters.json"
        parameters = read_json(path)
        values = {key: parameters[key] for key in PARAMETERS if key in parameters}
        missing = [key for key in PARAMETERS if key not in parameters]
        key = json.dumps(values, sort_keys=True)
        group = parameter_groups.setdefault(key, {
            "recorded_values": values, "unrecorded_parameters": missing,
            "source_paths": []})
        group["source_paths"].append(str(path))
        assert values["/use_sim_time"] is True
        assert values["/deform_monitor_v1/v1/resource/upstream_stationary_pose_freeze_verified"] is True
        return str(path)

    for original in original_runtime:
        directory = Path(original["source_file"]).parent.parent
        manifest = read_json(directory / "run_manifest.json")
        truth = compare.read_csv(directory / "truth/m01_m06_motion_truth.csv")
        origins = [float(r["truth_sim_time_s"]) - float(r["experiment_elapsed_s"])
                   for r in truth if r["model_name"] == "m01"]
        origin = statistics.median(origins)
        assert max(abs(t - origin) for t in origins) <= 0.002
        cutoff = compare.number(manifest.get("deformation_capture_cutoff_accepted_frame_index"))
        selected = []
        # Same source-time/capture-cutoff predicate as load_alert_batch.
        for row in compare.read_csv(directory / "runtime/per_cycle.csv"):
            stamp = compare.number(row.get("last_attempt_time_s"))
            index = compare.number(row.get("last_attempt_frame_index"))
            elapsed = stamp - origin if stamp is not None else None
            if (elapsed is None or not 0.0 <= elapsed <= 60.0
                    or (cutoff is not None and (index is None or index > cutoff))):
                continue
            selected.append(row)
        # This runnable check detects drift from the existing analysis selection.
        assert [float(r["processing_time_ms"]) for r in selected] == original["runtime_samples_ms"]
        result = summarize(selected)
        assert result["published_cycle_count"] == result["cycle_count"]
        result.update(run_id=directory.name, speed_mm_s=original["speed_mm_s"],
                      truth_origin_s=origin, capture_cutoff_frame_index=cutoff,
                      parameters_source=record_parameters(directory),
                      runtime_source=original["source_file"])
        runs.append(result)
        pooled[original["speed_mm_s"]].extend(selected)

    speeds = []
    for speed in compare.SPEEDS:
        group = [r for r in runs if r["speed_mm_s"] == speed]
        assert len(group) == 10
        rows = pooled[speed]
        speeds.append({
            "speed_mm_s": speed, "run_count": len(group), "cycle_count": len(rows),
            "cycle_mean_ms": statistics.mean(r["cycle_mean_ms"] for r in group),
            "cycle_p95_ms": compare.percentile([float(r["processing_time_ms"]) for r in rows], 95),
            "input_point_count_mean": statistics.mean(float(r["input_point_count"]) for r in rows),
            "input_point_count_min": min(r["input_point_count_min"] for r in group),
            "input_point_count_max": max(r["input_point_count_max"] for r in group),
            **{clock + "_rate_hz": statistics.mean(r[clock + "_rate_hz"] for r in group)
               for clock in ("source", "receive_sim", "wall")}})
    mobile = summarize([r for r in compare.read_csv(MOBILE / "runtime/per_cycle.csv")
                        if compare.boolean(r["last_attempt_published"])])
    mobile.update(run_id=MOBILE.name, parameters_source=record_parameters(MOBILE),
                  runtime_source=str(MOBILE / "runtime/per_cycle.csv"))
    # Reproduce the baseline loader's actual cycle selection, not its old five-target metrics.
    baseline_runs = []
    for row in compare.read_csv(BASELINE / "analysis/runtime_run_distribution.csv"):
        if row["method"] == "alert":
            continue
        path = Path(row["source_file"])
        frames = compare.load_frames(path.parent.parent / "detections/detection_frames.csv.gz")
        elapsed = {frame["source_stamp_s"]: frame["elapsed_s"] for frame in frames}
        durations = [float(r["processing_time_ms"]) for r in compare.read_csv_gz(path)
                     if 0 <= elapsed.get(float(r["source_stamp_s"]), -1) <= 60]
        assert len(durations) == int(row["runtime_frame_count"])
        assert all(math.isfinite(v) and v > 0 for v in durations)
        assert math.isclose(statistics.mean(durations), float(row["runtime_mean_ms"]), rel_tol=1e-12)
        baseline_runs.append(dict(method=row["method"], speed_mm_s=float(row["speed_mm_s"]),
                                  repeat=int(row["repeat"]), cycle_count=len(durations),
                                  cycle_mean_ms=statistics.mean(durations), runtime_source=str(path)))
    assert len(baseline_runs) == 120
    baseline_speeds = []
    for method in compare.METHODS:
        for speed in compare.SPEEDS:
            group = [r for r in baseline_runs if r["method"] == method and r["speed_mm_s"] == speed]
            assert sorted(r["repeat"] for r in group) == list(range(1, 11))
            baseline_speeds.append(dict(method=method, speed_mm_s=speed, run_count=10,
                                       cycle_mean_ms=statistics.mean(r["cycle_mean_ms"] for r in group),
                                       cycle_count=sum(r["cycle_count"] for r in group)))
    recorded_summary = read_json(MOBILE / "analysis/detector_runtime_summary.json")
    assert mobile["cycle_count"] == recorded_summary["published_cycle_count"] == 88
    assert len(runs) == 40 and sum(r["cycle_count"] for r in runs) == 9677

    evidence = {
        "queue": source_evidence("deform_monitor_v3/src/v1/node.cpp", [
            "subscribe(cloud_topic, 2", "subscribe(covariance_topic, 2",
            "cloud_queue_.size() > 4U", "covariance_queue_.size() > 4U",
            "max_raw_buffer_points)", "void V1Node::Spin()"]),
        "frozen_pose": source_evidence("deform_monitor_v3/src/v1/pipeline.cpp", [
            "if (config_.resource.upstream_stationary_pose_freeze_verified)",
            "nuisance.pose_covariance.setZero()", "nuisance.pose_correction.setZero()"]),
        "stationary_adapter": source_evidence("deform_monitor_bringup/scripts/stationary_monitor_cloud_adapter.py", [
            'state == "BUILDING_REFERENCE"', "self.frozen_matrix = odometry_matrix"]),
        "dwell_wall_clock": source_evidence("deform_monitor_bringup/scripts/icra2027_closed_loop_mission.py", [
            "time.sleep(self.stable_dwell)", "time.sleep(self.reference_preparation_dwell)"]),
        "role_implementation": source_evidence("deform_monitor_v3/src/v1/deterministic.cpp", ["if (bucket <"]),
        "role_parameter_contract": source_evidence("deform_monitor_v3/src/v1/ros_config.cpp", [
            "partition != std::vector<int>"]),
    }
    thresholds = [int(re.search(r"bucket < (\d+)ULL", r["text"]).group(1))
                  for r in evidence["role_implementation"]["lines"]]
    assert len(thresholds) == 3 and thresholds == sorted(thresholds)
    split = [b - a for a, b in zip([0] + thresholds, thresholds + [100])]
    for group in parameter_groups.values():
        group["role_percent_matches_current_implementation"] = (
            group["recorded_values"]["/deform_monitor_v1/v1/algorithm/support_partition_percent"] == split)
    compare.SOURCES.update((Path(__file__).resolve(), Path(compare.__file__),
                            Path(compare.stationary.__file__), Path(compare.compare_figures.__file__)))
    return {
        "selection": {"static": "Existing load_alert_batch source-time 0 <= t <= 60 s and capture cutoff; selected durations cross-checked exactly.",
                      "mobile": "All last_attempt_published=true runtime rows; no pre-reference mapping attempts.",
                      "baseline": "Existing analyze_run selection: join algorithm_runtime to detection source stamps and select their recorder truth_elapsed in [0,60] s; not recomputed source-origin cropping. Means/counts exactly checked against runtime_run_distribution. Workloads and timing boundaries differ across methods."},
        "aggregation": {"cycle_mean_ms": "Arithmetic mean of the ten run means; single-run mean for mobile.",
                        "cycle_p95_ms": "95th percentile of pooled selected cycle durations, not mean of run P95s.",
                        "input_point_count_mean": "Pooled selected-cycle mean.",
                        "rates": "(N-1)/(last-first) in each run, then arithmetic mean across ten runs; source and receive_sim use simulation seconds, wall uses recorder monotonic seconds.",
                        "source_to_receive_sim_s": "Difference of ROS source and receive times; not wall-clock end-to-end latency."},
        "timing_boundary": recorded_summary["timing_boundary"],
        "excluded_from_timing": recorded_summary["excluded_from_timing"],
        "stage_breakdown_available": False,
        "static_by_speed": speeds, "static_runs": runs, "mobile_published": mobile,
        "baseline_by_speed": baseline_speeds, "baseline_runs": baseline_runs,
        "recorded_parameter_groups": list(parameter_groups.values()),
        "current_source_evidence": evidence,
        "current_implemented_role_percent": split,
        "limitations": [
            "Current source evidence is not an archived executable attestation for the recorded runs.",
            "ROS ingress queues and internal queues are bounded in current source; complete historical ingress/drop counts are not recorded.",
            "Recorded role-percent parameters and current hardcoded role allocation disagree; neither is evidence of the historical executable allocation.",
            "The frozen-pose mode bypasses local control registration; these runs do not validate deformation estimation while the robot moves.",
            "No internal stage durations or matched-workload baseline efficiency are inferred."],
        "source_sha256": {str(path): file_sha256(path) for path in sorted(compare.SOURCES)},
    }


if __name__ == "__main__":
    audit = build_audit()
    target = PAPER / "runtime_audit.json"
    target.write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print("PASS: 40 stationary runs / 9,677 cycles; 88 mobile published cycles; " + str(target))
