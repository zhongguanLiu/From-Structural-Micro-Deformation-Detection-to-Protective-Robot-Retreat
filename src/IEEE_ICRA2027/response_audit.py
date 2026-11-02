#!/usr/bin/env python3
"""Audit recorded native confirmation support and response timing; no replay."""
import collections
import json
import math
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True
PAPER = Path(__file__).resolve().parent
SOURCE = PAPER.parent
sys.path.insert(0, str(SOURCE / "analyze_script"))
import analyze_compare_methods as compare

MOBILE = SOURCE.parent / "experiment_output/evacuation_paper_run/closed_loop/icra2027_20260911_235806"
ONSET = 10.0


def compact_frame(row, origin):
    stamp = float(row["message_time_s"])
    return dict(source_stamp_s=stamp, elapsed_s=stamp - origin,
                accepted_frame_index=int(row["source_accepted_frame_index"]),
                reference_epoch=int(row["reference_epoch"]),
                streak=int(row["retained_birth_streak"]))


def native_births(rows, origin):
    """Reconstruct counted active frames, leaving missing histories unknown."""
    groups = collections.defaultdict(list)
    seen = {}
    for row in rows:
        key = (row["gdu_id"], row["reference_epoch"], row["source_accepted_frame_index"])
        signature = tuple(row[k] for k in (
            "message_time_s", "active_now", "retained_for_avoidance",
            "retained_birth_streak", "retained_birth_confirmations", "lifecycle_state"))
        if key in seen:
            assert seen[key] == signature, ("conflicting duplicate", key)
            continue
        seen[key] = signature
        groups[key[:2]].append(row)
    births = []
    for (gdu, epoch), history in sorted(groups.items()):
        history.sort(key=lambda r: float(r["message_time_s"]))
        counted, complete, previous_streak, previous_retained = [], True, 0, False
        for row in history:
            streak = int(row["retained_birth_streak"])
            active = compare.boolean(row["active_now"])
            retained = compare.boolean(row["retained_for_avoidance"])
            if not previous_retained:
                if streak == 0:
                    counted, complete = [], True
                elif active and streak == 1:
                    counted, complete = [compact_frame(row, origin)], True
                elif active and streak == previous_streak + 1:
                    counted.append(compact_frame(row, origin))
                    complete = complete and len(counted) == streak
                elif streak != previous_streak:
                    complete = False
            if retained and not previous_retained:
                required = int(row["retained_birth_confirmations"])
                complete = complete and len(counted) == streak == required
                first = counted[0]["source_stamp_s"] if complete else None
                stamp = float(row["message_time_s"])
                births.append(dict(
                    gdu_id=gdu, reference_epoch=int(epoch),
                    confirmation_source_stamp_s=stamp,
                    confirmation_elapsed_s=stamp - origin,
                    confirmation_accepted_frame_index=int(row["source_accepted_frame_index"]),
                    pending_first_active_source_stamp_s=first,
                    first_counted_source_stamp_s=first,
                    last_counted_source_stamp_s=counted[-1]["source_stamp_s"] if complete else None,
                    counted_source_frames=list(counted), required_confirmations=required,
                    evidence_complete=bool(complete),
                    post_onset_only=all(f["elapsed_s"] >= ONSET for f in counted) if complete else None,
                    pre_onset_counted_frame_count=sum(f["elapsed_s"] < ONSET for f in counted) if complete else None,
                    first_observation_already_retained=row is history[0]))
            previous_streak, previous_retained = streak, retained
    return sorted(births, key=lambda r: (r["confirmation_source_stamp_s"], r["gdu_id"]))


def audit_native_confirmations(directory, origin):
    """Public read-only helper for the paper confirmation analysis."""
    path = Path(directory) / "evolution/gdu_evolution.csv"
    births = native_births(compare.read_csv(path), origin)
    for row in births:
        row["source_file"] = str(path)
    return births


def pose_delta(a, b):
    distance = math.sqrt(sum((float(a["position_" + k + "_m"]) -
                              float(b["position_" + k + "_m"])) ** 2 for k in "xyz"))
    quaternions = [[float(row["orientation_" + k]) for k in "xyzw"] for row in (a, b)]
    quaternions = [[v / math.sqrt(sum(x*x for x in q)) for v in q] for q in quaternions]
    qa, qb = quaternions
    sign = 1 if sum(a*b for a, b in zip(qa, qb)) >= 0 else -1
    chord = math.sqrt(sum((a - sign*b)**2 for a, b in zip(qa, qb)))
    return distance, 4 * math.asin(min(1.0, chord / 2))


def source_span(stamps):
    stamps = sorted(set(stamps))
    return dict(frame_count=len(stamps), first_source_s=stamps[0] if stamps else None,
                last_source_s=stamps[-1] if stamps else None,
                observed_source_span_s=stamps[-1] - stamps[0] if len(stamps) > 1 else 0.0)


def motion_response_fields(stamps, origin):
    stamps = sorted(set(stamps))
    assert all(ONSET <= stamp - origin < 50 for stamp in stamps)
    first = stamps[0] if stamps else None
    return dict(motion_native_response_bool=bool(stamps),
                motion_native_response_count=len(stamps),
                motion_native_response_count_unit="DISTINCT_TARGET_SOURCE_FRAMES",
                motion_native_response_first_source_stamp_s=first,
                motion_native_response_first_elapsed_s=first - origin if first is not None else None,
                motion_native_response_latency_s=first - origin - ONSET if first is not None else None)


def read_json(path):
    compare.SOURCES.add(path)
    return json.loads(path.read_text())


def build_audit():
    compare.SOURCES.clear()
    events = compare.read_csv(PAPER / "confirmation_analysis/events.csv")
    assert len(events) == 960
    groups = collections.defaultdict(list)
    for row in events:
        groups[(row["method"], row["scenario_id"])].append(row)
    native, diagnostics, runs = [], [], []
    for (method, scenario), group in sorted(groups.items()):
        assert len(group) == 6
        sample = group[0]
        directory = Path(sample["source_file"]).parent.parent
        origin = float(sample["source_clock_origin_s"])
        info = dict(method=method, scenario_id=scenario,
                    speed_mm_s=float(sample["speed_mm_s"]), repeat=int(sample["repeat"]))
        if method == "alert":
            evolution_path = directory / "evolution/gdu_evolution.csv"
            evolution = compare.read_csv(evolution_path)
            inventory = {r["gdu_id"]: r for r in compare.read_csv(directory / "analysis/detection_track_inventory.csv")}
            births = native_births(evolution, origin)
            for birth in births:
                track = inventory.get(birth["gdu_id"], {})
                birth.update(info, source_file=str(evolution_path),
                             model_name=track.get("model_name", ""),
                             association_status=track.get("association_status", "MISSING_INVENTORY"))
            native.extend(births)
            baseline = [r for r in evolution if 0 <= float(r["message_time_s"]) - origin < ONSET]
            span = source_span(float(r["message_time_s"]) for r in baseline)
            truth = compare.read_csv(directory / "truth/m01_m06_motion_truth.csv")
            constant = {}
            for model in compare.MODELS:
                selected = [r for r in truth if r["model_name"] == model and 0 <= float(r["experiment_elapsed_s"]) < ONSET]
                constant[model] = dict(sample_count=len(selected),
                    maximum_position_change_m=max(pose_delta(selected[0], r)[0] for r in selected),
                    maximum_orientation_change_rad=max(pose_delta(selected[0], r)[1] for r in selected))
            for event in group:
                model = event["model_name"]
                gdus = {g for g, r in inventory.items() if r["association_status"] == "MATCHED" and r["model_name"] == model}
                candidates = [inventory[g] for g in gdus if compare.boolean(inventory[g]["confirmed"])
                              and ONSET <= float(inventory[g]["confirmation_experiment_elapsed_s"]) < 50]
                first = min(candidates, key=lambda r: (float(r["confirmation_experiment_elapsed_s"]), r["gdu_id"])) if candidates else None
                detected = first is not None
                confirmation_stamp = origin + float(first["confirmation_experiment_elapsed_s"]) if first else None
                matches = [b for b in births if detected and b["gdu_id"] == first["gdu_id"] and
                           abs(b["confirmation_source_stamp_s"] - confirmation_stamp) < 1e-6]
                assert len(matches) <= 1
                birth = matches[0] if matches else {}
                baseline_confirmations = [b for b in births if b["gdu_id"] in gdus and 0 <= b["confirmation_elapsed_s"] < ONSET]
                eligible = [b for b in births if b["gdu_id"] in gdus and ONSET <= b["confirmation_elapsed_s"] < 50
                            and b["evidence_complete"] and b["post_onset_only"] is True]
                diagnostics.append(dict(info, model_name=model, original_detected=detected,
                    original_confirmation_source_stamp_s=confirmation_stamp,
                    original_confirmation_gdu_id=first["gdu_id"] if first else None,
                    evidence_complete=birth.get("evidence_complete", False) if detected else None,
                    post_onset_only=birth.get("post_onset_only"),
                    pending_first_active_source_stamp_s=birth.get("pending_first_active_source_stamp_s"),
                    first_eligible_post_onset_birth_source_stamp_s=eligible[0]["confirmation_source_stamp_s"] if eligible else None,
                    eligible_post_onset_birth_count=len(eligible),
                    baseline_active_frame_count=sum(compare.boolean(r["active_now"]) for r in baseline if r["gdu_id"] in gdus),
                    baseline_confirmation_count=len(baseline_confirmations),
                    baseline_truth=constant[model],
                    **motion_response_fields((float(r["message_time_s"]) for r in evolution
                        if r["gdu_id"] in gdus and compare.boolean(r["active_now"])
                        and ONSET <= float(r["message_time_s"]) - origin < 50), origin)))
            runs.append(dict(info, **span,
                baseline_native_confirmed_track_count=len({b["gdu_id"] for b in births if 0 <= b["confirmation_elapsed_s"] < ONSET}),
                baseline_native_confirmations=[b for b in births if 0 <= b["confirmation_elapsed_s"] < ONSET]))
        else:
            frames = compare.load_frames(directory / "detections/detection_frames.csv.gz")
            baseline = [f for f in frames if 0 <= f["source_stamp_s"] - origin < ONSET]
            span = source_span(f["source_stamp_s"] for f in baseline)
            for event in group:
                model = event["model_name"]
                base = [dict(f, point_count=f["counts"][model]) for f in baseline]
                first, confirmed = compare.confirmation(base, 1, 3, .35)
                # This deliberately does not reset history at excitation onset.
                whole = [dict(f, point_count=f["counts"][model]) for f in frames
                         if 0 <= f["source_stamp_s"] - origin < 50]
                start_all, confirmation_all = compare.confirmation(whole, 1, 3, .35)
                _, motion_confirmation = compare.confirmation(
                    [f for f in whole if f["source_stamp_s"] - origin >= ONSET], 1, 3, .35)
                diagnostics.append(dict(info, model_name=model,
                    original_detected=motion_confirmation is not None,
                    first_eligible_post_onset_birth_source_stamp_s=motion_confirmation["source_stamp_s"] if motion_confirmation else None,
                    evidence_complete=True,
                    post_onset_only=None if not confirmation_all else start_all["source_stamp_s"] >= origin + ONSET,
                    baseline_active_frame_count=sum(f["point_count"] > 0 for f in base),
                    baseline_confirmation_count=int(confirmed is not None),
                    baseline_confirmation_source_s=None if not confirmed else confirmed["source_stamp_s"],
                    unreset_first_positive_source_s=None if not start_all else start_all["source_stamp_s"],
                    unreset_confirmation_source_s=None if not confirmation_all else confirmation_all["source_stamp_s"],
                    **motion_response_fields((f["source_stamp_s"] for f in whole
                        if f["source_stamp_s"] - origin >= ONSET and f["point_count"] > 0), origin)))
            other = [dict(f, point_count=f["counts"]["OTHER"]) for f in baseline]
            _, other_confirmation = compare.confirmation(other, 1, 3, .35)
            runs.append(dict(info, **span, baseline_other_three_frame_confirmation=other_confirmation is not None))
    for row in diagnostics:
        assert (row["first_eligible_post_onset_birth_source_stamp_s"] is None
                or row["motion_native_response_bool"]), ("confirmation without native response", row)
        assert not row["original_detected"] or row["motion_native_response_bool"], row
    summaries = []
    for method in compare.ALL_METHODS:
        for speed in compare.SPEEDS:
            selected = [r for r in diagnostics if r["method"] == method and r["speed_mm_s"] == speed]
            selected_runs = [r for r in runs if r["method"] == method and r["speed_mm_s"] == speed]
            assert len(selected) == 60 and len(selected_runs) == 10
            response_delays = [r["motion_native_response_latency_s"] for r in selected
                               if r["motion_native_response_bool"]]
            summaries.append(dict(method=method, speed_mm_s=speed, target_run_count=60,
                n_native_response_target_runs=len(response_delays),
                native_response_latency_median_s=statistics.median(response_delays) if response_delays else None,
                nominal_baseline_run_seconds=100.0,
                sampled_baseline_run_span_seconds=sum(r["observed_source_span_s"] for r in selected_runs),
                baseline_sampled_frame_count=sum(r["frame_count"] for r in selected_runs),
                baseline_confirmed_target_run_count=sum(r["baseline_confirmation_count"] > 0 for r in selected),
                baseline_active_target_run_count=sum(r["baseline_active_frame_count"] > 0 for r in selected),
                baseline_all_native_birth_count=sum(len(r.get("baseline_native_confirmations", [])) for r in selected_runs) if method == "alert" else None,
                baseline_other_three_frame_run_count=sum(r.get("baseline_other_three_frame_confirmation", False) for r in selected_runs) if method != "alert" else None,
                original_confirmed_count=sum(r["original_detected"] for r in selected),
                first_eligible_post_onset_target_run_count=sum(r["first_eligible_post_onset_birth_source_stamp_s"] is not None for r in selected),
                original_confirmed_post_onset_only_count=sum(r["original_detected"] and r["post_onset_only"] is True for r in selected),
                original_confirmed_baseline_assisted_count=sum(r["original_detected"] and r["post_onset_only"] is False for r in selected),
                original_confirmed_unknown_support_count=sum(r["original_detected"] and r["post_onset_only"] is None for r in selected)))

    mobile_truth = [r for r in compare.read_csv(MOBILE / "truth/m01_m06_motion_truth.csv") if r["model_name"] == "m05"]
    mobile_truth.sort(key=lambda r: float(r["truth_sim_time_s"]))
    change_indices = [i for i in range(1, len(mobile_truth)) if any(v > 1e-8 for v in pose_delta(mobile_truth[i-1], mobile_truth[i]))]
    last = change_indices[-1]
    stop = float(mobile_truth[last]["truth_sim_time_s"])
    decision = read_json(MOBILE / "mission_decision_summary.json")
    trigger = next(r["sim_time_s"] for r in decision["state_history"] if r["state"] == "DEFORMATION_DETECTED")
    mobile = dict(model_name="m05", pose_change_tolerance_m_and_rad=1e-8,
        actual_pose_last_change_source_s=stop,
        preceding_truth_source_s=float(mobile_truth[last-1]["truth_sim_time_s"]),
        stable_suffix_last_source_s=float(mobile_truth[-1]["truth_sim_time_s"]),
        stable_suffix_sample_count=len(mobile_truth)-last,
        trigger_decision_sim_time_s=trigger, stop_to_trigger_sim_seconds=trigger-stop,
        trigger_support_source_s=decision["first_detection_stamp_s"],
        stop_to_trigger_support_source_seconds=decision["first_detection_stamp_s"]-stop,
        retained_birth_source_s=decision["return_trigger_retained_since_stamp_s"],
        truth_source=str(MOBILE / "truth/m01_m06_motion_truth.csv"))
    runtime = read_json(PAPER / "runtime_audit.json")
    parameters = read_json(MOBILE / "ros_parameters.json")
    manifest = read_json(MOBILE / "run_manifest.json")
    versions = read_json(MOBILE / "provenance/execution_versions.json")
    merge = compare.read_csv(MOBILE / "sensors/dual_lidar_merge_diagnostics.csv")
    performance = compare.read_csv(MOBILE / "sensors/gazebo_performance_metrics.csv")
    thread_sources = []
    for relative in ("deform_monitor_v3/src/v1/node.cpp", "deform_monitor_v3/src/v1/uac_filter.cpp",
                     "deform_monitor_v3/src/v1/ros_bridge.cpp", "deform_monitor_v3/include/deform_monitor_v3/v1/uac_filter.hpp",
                     "deform_monitor_v3/include/deform_monitor_v3/v1/ros_bridge.hpp"):
        path = SOURCE / relative
        if not path.exists():
            continue
        compare.SOURCES.add(path)
        digest = compare.stationary.file_sha256(path)
        thread_sources.append(dict(path=str(path), current_sha256=digest,
            recorded_sha256=versions["source_sha256"].get(relative),
            matches_recorded_source=digest == versions["source_sha256"].get(relative),
            lines=[dict(line=i, text=line.strip()) for i, line in enumerate(path.read_text().splitlines(), 1)
                   if any(s in line for s in ("ros::spin()", "8U", "return PosteriorImpl(true)",
                       "return PredictIntervalImpl(stamp, enforce_update_gap, true, true", "nuisance_by_gdu, stamp, true",
                       "snapshot, filter, config, kMaximumDetectionTileWorkerCount"))]))
    metadata = dict(historical_cpu_model=None, historical_ram=None,
        historical_os_and_compiler=None, historical_actual_worker_count=None,
        metadata_limit="No hardware/runtime thread-count fields in retained manifest, execution_versions or ROS parameters; do not substitute the present host. Source worker bounds are not measured thread counts.",
        reviewed_manifest_keys=sorted(manifest),
        estimator_thread_policy="One synchronous ROS callback stream, with independent-GDU prediction/update/posterior and output serialization using bounded worker groups of up to eight including the caller. Historical source hashes match; actual worker creation counts are unrecorded.",
        recorded_thread_parameters={k:v for k,v in parameters.items() if any(s in k.lower() for s in ("thread", "worker", "parallel"))},
        thread_source_evidence=thread_sources,
        input_timing_last_diagnostic={k:merge[-1][k] for k in (
            "configured_sensor_update_rate_hz", "pairs_accepted", "accepted_measurement_interval_count",
            "minimum_accepted_measurement_interval_ns", "maximum_accepted_measurement_interval_ns",
            "mean_accepted_measurement_interval_ns", "realized_measurement_hz",
            "primary_queue_dropped", "secondary_queue_dropped")},
        recorded_gazebo_rtf_row_count=len(performance),
        published_cycle_statistics=runtime["mobile_published"])
    compare.SOURCES.update((Path(__file__).resolve(), Path(compare.__file__),
                            Path(compare.stationary.__file__),
                            SOURCE / "deform_evolution_map/src/deform_evolution_map/model.py"))
    sources = [dict(path=str(p), bytes=p.stat().st_size, sha256=compare.stationary.file_sha256(p))
               for p in sorted(compare.SOURCES)]
    return dict(policy={
        "motion_native_response": "In [10,50) source seconds, any target-associated dynamic-point frame for comparators, or any active_now update among all MATCHED inventory GDUs for Ours, without requiring later confirmation. Counts deduplicate target source timestamps; multiple active GDUs in one frame count once. Initial response is not calibrated detection or primary confirmation, and can coexist with baseline responses.",
        "diagnostic_baseline": "The 960 rows use events.csv only for run identity, source paths and truth clock origin. Native ALERT outcomes are rebuilt from unchanged raw inventory first MATCHED confirmation in [10,50); comparator outcomes are rebuilt from raw frames. Later primary eligibility changes do not erase original native outcomes.",
        "native_support": "Trace increments of the recorded retained_birth_streak on active source frames; zero resets, unchanged streak pauses, missing increments make support unknown. Confirmation does not reset at excitation onset.",
        "post_onset_only": "Every counted frame has source_stamp minus recorded truth origin >=10 s. False means baseline-assisted, not proof of a false detection. Unknown is never eligible.",
        "baseline": "0<=source elapsed<10 s; nominal exposure is 400 run-seconds per method, not a long-term false-alarm-rate trial. Sampled spans and counts are reported separately; six targets share each run.",
        "baseline_comparators": "The existing target-associated >=1 point, 3 consecutive frames, max source gap .35s rule, both on baseline and on the unreset 0..50 s stream.",
        "censoring": "An observed post-onset confirmation with pre-onset supporting frames is baseline-assisted; this is not automatically a baseline-confirmed or classically left-censored confirmation endpoint.",
        "stop": "Last recorded SE3 change above 1e-8 m/rad, followed by unchanged observed suffix; no command-duration substitution, extrapolation or claim of sub-sampling temporal resolution."},
        summaries=summaries, native_confirmations=native, event_diagnostics=diagnostics,
        baseline_runs=runs, mobile_response=mobile, runtime_metadata=metadata, sources=sources)


def self_check():
    def row(t, streak, active, retained=False):
        return dict(gdu_id="g", reference_epoch="1", source_accepted_frame_index=str(int(t*1000)),
            message_time_s=str(t), retained_birth_streak=str(streak), retained_birth_confirmations="3",
            active_now=str(active), retained_for_avoidance=str(retained), lifecycle_state="1" if retained else "5")
    assisted = native_births([row(6.212,1,True), row(8,1,False), row(9.612,2,True),
                             row(9.81,2,False), row(10.012,3,True,True)], 0)[0]
    assert assisted["evidence_complete"] and assisted["post_onset_only"] is False
    assert assisted["pre_onset_counted_frame_count"] == 2
    fresh = native_births([row(6,1,True), row(8,0,False), row(11,1,True),
                          row(12,2,True), row(13,3,True,True)], 0)[0]
    assert fresh["post_onset_only"] is True and fresh["first_counted_source_stamp_s"] == 11
    unknown = native_births([row(9,2,True), row(11,3,True,True)], 0)[0]
    assert not unknown["evidence_complete"] and unknown["post_onset_only"] is None
    response = motion_response_fields([11, 11, 12], 0)
    assert response["motion_native_response_count"] == 2 and response["motion_native_response_latency_s"] == 1
    assert motion_response_fields([], 0)["motion_native_response_first_source_stamp_s"] is None


if __name__ == "__main__":
    self_check()
    result = build_audit()
    (PAPER / "response_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PASS: native support, 960 event baseline diagnostics, and recorded M05 stop timing.")
    print(json.dumps(result["summaries"], indent=2))
    print(json.dumps(result["mobile_response"], indent=2))
