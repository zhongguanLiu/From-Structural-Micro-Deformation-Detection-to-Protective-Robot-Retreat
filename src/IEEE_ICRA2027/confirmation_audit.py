#!/usr/bin/env python3
"""Rebuild six-target confirmations from recorded source times and native outputs."""
import bisect
import gzip
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
PAPER = Path(__file__).resolve().parent
SOURCE = PAPER.parent
OUTPUT = SOURCE.parent / "experiment_output"
BASELINE = OUTPUT / "compare_methods_single_lidar/batch_20260901_235707"
ALERT = OUTPUT / "stationary_speed_sweep/batch_20260901_012253"
DESTINATION = PAPER / "confirmation_analysis"
sys.path.insert(0, str(SOURCE / "analyze_script"))
import analyze_compare_methods as compare
from response_audit import audit_native_confirmations

MODELS = tuple("m%02d" % i for i in range(1, 7))
METHODS = tuple(compare.ALL_METHODS)
SPEEDS = tuple(compare.SPEEDS)
ONSET, END, BOOTSTRAPS, SEED = 10.0, 50.0, 10000, 2027


def truth_data(directory, method):
    curves = {model: {} for model in MODELS}
    origins = []
    if method == "alert":
        path = directory / "truth/m01_m06_motion_truth.csv"
        for row in compare.read_csv(path):
            stamp, elapsed = float(row["truth_sim_time_s"]), float(row["experiment_elapsed_s"])
            vector = tuple(float(row["translation_%s_m" % axis]) for axis in "xyz")
            assert stamp not in curves[row["model_name"]] or curves[row["model_name"]][stamp] == vector
            curves[row["model_name"]][stamp] = vector
            origins.append(stamp - elapsed)
    else:
        path = directory / "truth/motion_truth.jsonl.gz"
        compare.SOURCES.add(path)
        with gzip.open(path, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                stamp = float(row["sim_time"])
                origins.append(stamp - float(row["experiment_elapsed"]))
                for model in row["models"]:
                    vector = tuple(map(float, model["translation"]))
                    assert stamp not in curves[model["model_name"]] or curves[model["model_name"]][stamp] == vector
                    curves[model["model_name"]][stamp] = vector
    origin = statistics.median(origins)
    assert max(abs(value - origin) for value in origins) <= .002, directory
    output = {}
    for model, points in curves.items():
        times = np.array(sorted(points))
        vectors = np.array([points[stamp] for stamp in times])
        assert len(times) > 1 and np.all(np.diff(times) > 0), (directory, model)
        assert np.isfinite(vectors).all() and times[0] <= origin + ONSET < origin + END <= times[-1]
        output[model] = times, vectors
    return origin, output, path


def truth_at(curve, stamp):
    times, vectors = curve
    assert times[0] <= stamp <= times[-1], (stamp, times[0], times[-1])
    upper = min(len(times) - 1, bisect.bisect_left(times, stamp))
    lower = upper if times[upper] == stamp else upper - 1
    assert lower >= 0
    fraction = 0.0 if lower == upper else (stamp - times[lower]) / (times[upper] - times[lower])
    vector = vectors[lower] + fraction * (vectors[upper] - vectors[lower])
    return dict(truth_translation_at_confirmation_mm=float(np.linalg.norm(vector) * 1000),
                truth_translation_x_mm=float(vector[0] * 1000),
                truth_translation_y_mm=float(vector[1] * 1000),
                truth_translation_z_mm=float(vector[2] * 1000),
                truth_bracket_lower_source_s=float(times[lower]),
                truth_bracket_upper_source_s=float(times[upper]),
                truth_interpolation_fraction=float(fraction))


def event_row(record, method, model, directory, origin, curve, truth_path):
    return dict(method=method, speed_mm_s=float(record["speed_mm_s"]),
                model_name=model, repeat=int(record["repeat"]), scenario_id=record["scenario_id"],
                detected=False, confirmation_latency_s=None,
                confirmation_experiment_elapsed_s=None, confirmation_source_stamp_s=None,
                truth_translation_at_confirmation_mm=None, included_in_primary_metrics=True,
                event_population="ALL_SIX_TARGETS", observation_horizon_s=END - ONSET,
                motion_onset_experiment_s=ONSET, source_clock_origin_s=origin,
                source_file="", truth_source_file=str(truth_path),
                truth_translation_at_window_end_mm=truth_at(curve, origin + END)["truth_translation_at_confirmation_mm"])


def alert_events(record, directory, origin, truths, truth_path, excluded):
    path = directory / "analysis/detection_track_inventory.csv"
    inventory = compare.read_csv(path)
    assert len({row["gdu_id"] for row in inventory}) == len(inventory), path
    tracks = {row["gdu_id"]: row for row in inventory}
    candidates = {model: [] for model in MODELS}
    for birth in audit_native_confirmations(directory, origin):
        track = tracks.get(birth["gdu_id"], {})
        elapsed = birth["confirmation_elapsed_s"]
        reason = None
        if track.get("association_status") != "MATCHED" or track.get("model_name") not in MODELS:
            reason = "NO_UNIQUE_MATCHED_TARGET"
        elif not ONSET <= elapsed < END:
            reason = "BEFORE_MOTION_WINDOW" if elapsed < ONSET else "AT_OR_AFTER_WINDOW_END"
        elif not birth["evidence_complete"]:
            reason = "CONFIRMATION_SUPPORT_UNRESOLVED"
        elif not birth["post_onset_only"]:
            reason = "BASELINE_ASSISTED_CONFIRMATION"
        else:
            candidates[track["model_name"]].append((birth, track))
        if reason:
            excluded.append(dict(method="alert", speed_mm_s=float(record["speed_mm_s"]),
                                 repeat=int(record["repeat"]), scenario_id=record["scenario_id"],
                                 gdu_id=birth["gdu_id"], model_name=track.get("model_name", ""),
                                 confirmation_experiment_elapsed_s=elapsed,
                                 evidence_complete=birth["evidence_complete"],
                                 first_counted_source_stamp_s=birth["first_counted_source_stamp_s"],
                                 counted_source_frames=json.dumps(birth["counted_source_frames"], sort_keys=True),
                                 exclusion_reason=reason, original_evaluation_category=track.get("evaluation_category", ""),
                                 source_file=str(path)))
    events = []
    for model in MODELS:
        row = event_row(record, "alert", model, directory, origin, truths[model], truth_path)
        row.update(source_file=str(path), confirmation_policy="POST_ONSET_NATIVE_MAP_RETENTION",
                   matched_confirmed_tracks_in_window=len(candidates[model]))
        if candidates[model]:
            first, track = min(candidates[model], key=lambda pair: (
                pair[0]["confirmation_source_stamp_s"], pair[0]["gdu_id"]))
            elapsed = first["confirmation_elapsed_s"]
            stamp = first["confirmation_source_stamp_s"]
            row.update(detected=True, confirmation_latency_s=elapsed - ONSET,
                       confirmation_experiment_elapsed_s=elapsed, confirmation_source_stamp_s=stamp,
                       gdu_id=first["gdu_id"], confirmation_time_source="NATIVE_RETAINED_BIRTH_SOURCE",
                       evidence_complete=True, post_onset_only=True,
                       first_counted_source_stamp_s=first["first_counted_source_stamp_s"],
                       counted_source_frames=json.dumps(first["counted_source_frames"], sort_keys=True),
                       original_evaluation_category=track["evaluation_category"])
            row.update(truth_at(truths[model], stamp))
        events.append(row)
    return events


def baseline_events(record, directory, origin, truths, truth_path):
    path = directory / "detections/detection_frames.csv.gz"
    original = compare.load_frames(path)
    frames = [dict(frame, nearest_truth_elapsed_s=frame["elapsed_s"],
                   elapsed_s=frame["source_stamp_s"] - origin) for frame in original]
    events = []
    for model in MODELS:
        row = event_row(record, record["method"], model, directory, origin, truths[model], truth_path)
        row.update(source_file=str(path), confirmation_policy="THREE_SOURCE_FRAMES_MIN1POINT_MAXGAP0.35S")
        motion = [dict(f, point_count=f["counts"][model]) for f in frames if ONSET <= f["elapsed_s"] < END]
        assert motion, (directory, model)
        for count in (1, 3, 5):
            start, confirmed = compare.confirmation(motion, 1, count, .35)
            row["detected_%d_frame" % count] = confirmed is not None
            row["confirmation_latency_%d_frame_s" % count] = None if confirmed is None else confirmed["elapsed_s"] - ONSET
            row["confirmation_source_%d_frame_s" % count] = None if confirmed is None else confirmed["source_stamp_s"]
            row["truth_translation_%d_frame_mm" % count] = None if confirmed is None else truth_at(
                truths[model], confirmed["source_stamp_s"])["truth_translation_at_confirmation_mm"]
            if count == 3 and confirmed:
                row.update(detected=True, confirmation_latency_s=confirmed["elapsed_s"] - ONSET,
                           confirmation_experiment_elapsed_s=confirmed["elapsed_s"],
                           confirmation_source_stamp_s=confirmed["source_stamp_s"],
                           confirmation_nearest_truth_elapsed_s=confirmed["nearest_truth_elapsed_s"],
                           source_minus_nearest_truth_s=confirmed["elapsed_s"] - confirmed["nearest_truth_elapsed_s"],
                           first_positive_source_stamp_s=start["source_stamp_s"])
                row.update(truth_at(truths[model], confirmed["source_stamp_s"]))
        old_motion = [dict(f, point_count=f["counts"][model]) for f in original if ONSET <= f["elapsed_s"] < END]
        _, previous = compare.confirmation(old_motion, 1, 3, .35)
        row.update(previous_nearest_truth_detected=previous is not None,
                   previous_nearest_truth_latency_s=None if previous is None else previous["elapsed_s"] - ONSET)
        row.update(positive_frame_count=sum(f["point_count"] > 0 for f in motion),
                   maximum_dynamic_points=max(f["point_count"] for f in motion),
                   source_window_frame_count=len(motion))
        assert not row["detected_5_frame"] or row["detected_3_frame"]
        assert not row["detected_3_frame"] or row["detected_1_frame"]
        events.append(row)
    return events


def statistics_rows(events, methods=METHODS, labels=None):
    labels = compare.METHOD_LABELS if labels is None else labels
    summaries, curves = [], []
    for mi, method in enumerate(methods):
        for si, speed in enumerate(SPEEDS):
            selected = [r for r in events if r["method"] == method and r["speed_mm_s"] == speed]
            assert len(selected) == 60
            hits = [r for r in selected if r["detected"]]
            translations = [r["truth_translation_at_confirmation_mm"] for r in hits]
            delays = [r["confirmation_latency_s"] for r in hits]
            end = max(r["truth_translation_at_window_end_mm"] for r in selected)
            grid = np.unique([0.0, end] + translations)
            rows = []
            for repeat in range(1, 11):
                run = [r for r in selected if r["repeat"] == repeat]
                assert {r["model_name"] for r in run} == set(MODELS) and len(run) == 6
                outcomes = [r["confirmation_latency_s"] if r["detected"] else END - ONSET for r in run]
                values = [sum(r["detected"] for r in run) / 6.0, statistics.mean(outcomes)]
                values += [sum(r["detected"] and r["truth_translation_at_confirmation_mm"] <= d for r in run) / 6.0 for d in grid]
                rows.append(values)
            estimate, low, high = compare.cluster_mean_interval(rows, BOOTSTRAPS, SEED + mi * 100 + si)
            summary = dict(method=method, method_label=labels[method], speed_mm_s=speed,
                           n_runs=10, event_count=60, confirmed_count=len(hits), unconfirmed_count=60 - len(hits),
                           translation_median_mm=compare.percentile(translations, 50),
                           translation_q25_mm=compare.percentile(translations, 25),
                           translation_q75_mm=compare.percentile(translations, 75),
                           confirmation_latency_median_s=compare.percentile(delays, 50),
                           restricted_mean_wait_s=float(estimate[1]), recall=float(estimate[0]),
                           recall_ci95_low=float(low[0]), recall_ci95_high=float(high[0]),
                           restricted_mean_wait_ci95_low_s=float(low[1]), restricted_mean_wait_ci95_high_s=float(high[1]),
                           m04_confirmed_count=sum(r["model_name"] == "m04" for r in hits),
                           window_end_translation_min_mm=min(r["truth_translation_at_window_end_mm"] for r in selected),
                           window_end_translation_max_mm=end)
            for count in (1, 3, 5):
                summary["confirmed_count_%d_frame" % count] = None if method == "alert" else sum(r["detected_%d_frame" % count] for r in selected)
            summaries.append(summary)
            for index, displacement in enumerate(grid, 2):
                curves.append(dict(method=method, speed_mm_s=speed, translation_mm=float(displacement),
                                   confirmed_fraction=float(estimate[index]), ci95_low=float(low[index]),
                                   ci95_high=float(high[index]), event_count=60, n_runs=10))
            assert math.isclose(curves[-1]["confirmed_fraction"], len(hits) / 60.0)
    return summaries, curves


def main():
    compare.SOURCES.clear()
    events, excluded = [], []
    for batch, method, expected_runs in ((BASELINE, None, 120), (ALERT, "alert", 40)):
        ledger = compare.read_csv(batch / "batch_status.csv")
        assert len(ledger) == expected_runs and all(r["status"] == "SUCCESS" for r in ledger)
        for record in ledger:
            actual_method = method or record["method"]
            directory = batch / "runs" / record["scenario_id"]
            origin, truths, truth_path = truth_data(directory, actual_method)
            if actual_method == "alert":
                events.extend(alert_events(record, directory, origin, truths, truth_path, excluded))
            else:
                events.extend(baseline_events(record, directory, origin, truths, truth_path))
    keys = [(r["method"], r["speed_mm_s"], r["repeat"], r["model_name"]) for r in events]
    expected = {(m, s, r, g) for m in METHODS for s in SPEEDS for r in range(1, 11) for g in MODELS}
    assert len(keys) == 960 and len(set(keys)) == 960 and set(keys) == expected
    assert all(not r["detected"] or 0 <= r["confirmation_latency_s"] < 40 for r in events)
    # No identity exclusion: three supported M04 events remain, while a
    # baseline-assisted birth and a pre-motion birth stay diagnostic.
    assert {(r["speed_mm_s"], r["repeat"]) for r in events
            if r["method"] == "alert" and r["model_name"] == "m04" and r["detected"]} == {
                (1.0, 9), (5.0, 6), (10.0, 5)}
    assert [(r["speed_mm_s"], r["repeat"], r["model_name"]) for r in excluded
            if r["exclusion_reason"] == "BASELINE_ASSISTED_CONFIRMATION"] == [(3., 5, "m04")]
    assert any(r["model_name"] == "m04" and r["speed_mm_s"] == 5 and r["repeat"] == 4
               and r["exclusion_reason"] == "BEFORE_MOTION_WINDOW" for r in excluded)
    events.sort(key=lambda r: (METHODS.index(r["method"]), r["speed_mm_s"], r["repeat"], r["model_name"]))
    summaries, curves = statistics_rows(events)
    DESTINATION.mkdir(exist_ok=True)
    for name, rows in (("events", events), ("summary", summaries), ("curves", curves), ("excluded_tracks", excluded)):
        compare.write_csv(DESTINATION / (name + ".csv"), rows)
    compare.SOURCES.update({Path(__file__).resolve(), Path(compare.__file__).resolve(),
                            PAPER / "response_audit.py"})
    sources = [dict(path=str(path), bytes=path.stat().st_size,
                    sha256=compare.stationary.file_sha256(path)) for path in sorted(compare.SOURCES)]
    manifest = dict(event_count=len(events), target_names=MODELS, n_run_clusters_per_condition=10,
                    bootstrap_samples=BOOTSTRAPS, bootstrap_seed=SEED,
                    time_policy="source_stamp minus median(truth_source_stamp - truth_elapsed); [10,50), latency from 10",
                    baseline_policy="at least one recorded target-associated point for 3 consecutive source frames, gap <=0.35s; 1/5-frame sensitivity retained",
                    alert_policy="first MATCHED native retained birth in [10,50) with complete counted-active support entirely at/after onset; search all births; no new detector threshold",
                    truth_policy="linear interpolation of recorded translation vectors at source stamp, then Euclidean norm; no extrapolation or command integration",
                    exclusion_policy="No target identity exclusions. Unmatched/out-of-window births, incomplete support and baseline-assisted confirmations remain diagnostic, not primary successes. Accuracy retains its separate native-confirmed cohort.",
                    sources=sources)
    (DESTINATION / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for row in summaries:
        print(row["method"], row["speed_mm_s"], row["confirmed_count"], row["m04_confirmed_count"],
              row["translation_median_mm"], row["confirmation_latency_median_s"], row["restricted_mean_wait_s"],
              row["confirmed_count_1_frame"], row["confirmed_count_5_frame"])
    print("PASS: 960 events, 16 conditions, six targets per run, 10-cluster bootstrap, source-clock bounds.")


if __name__ == "__main__":
    main()
