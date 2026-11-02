#!/usr/bin/env python3
"""Five-target manuscript statistics from retained outputs; never replay detectors."""
import argparse
import collections
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True
PAPER = Path(__file__).resolve().parent
DESTINATION = PAPER / "paper_analysis"
sys.path.insert(0, str(PAPER.parent / "analyze_script"))
import analyze_compare_methods as common

METHODS = ("m_detector", "spatiotemporal_normals", "dynablox", "m3c2", "alert")
TARGETS = ("m01", "m02", "m03", "m05", "m06")
SPEEDS = (1, 3, 5, 10)
PHASES = {"baseline": (0, 10), "motion": (10, 50), "hold": (50, 60)}
SOURCES = set()


def read_csv(path):
    SOURCES.add(Path(path).resolve())
    return common.read_csv(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def moments(values):
    values = [v for v in values if v is not None]
    return (statistics.mean(values) if values else None,
            statistics.stdev(values) if len(values) > 1 else None, len(values))


def bin_index(stamp, origin):
    # ROS time has nanosecond resolution; remove binary subtraction artifacts.
    elapsed = round(float(stamp) - origin, 9)
    assert math.isfinite(elapsed)
    return (math.floor(elapsed) if 0 <= elapsed < 60 else None), elapsed


def phase_at(index):
    return next(name for name, (start, end) in PHASES.items() if start <= index < end)


def native_frames(event):
    """One output timestamp and a set of positive retained target identities."""
    directory = Path(event["source_file"]).parent.parent
    if event["method"] != "alert":
        path = directory / "detections/detection_frames.csv.gz"
        SOURCES.add(path)
        return [(frame["source_stamp_s"],
                 {model for model in TARGETS if frame["counts"][model] > 0})
                for frame in common.load_frames(path)]
    inventory = read_csv(directory / "analysis/detection_track_inventory.csv")
    assert len({row["gdu_id"] for row in inventory}) == len(inventory)
    owners = {row["gdu_id"]: row["model_name"] for row in inventory
              if row["association_status"] == "MATCHED" and row["model_name"] in TARGETS}
    path = directory / "evolution/gdu_evolution.csv"
    SOURCES.add(path)
    frames = collections.defaultdict(set)
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            stamp = float(row["message_time_s"])
            active = frames[stamp]  # All published GDU timestamps establish availability.
            model = owners.get(row["gdu_id"])
            if model is not None and common.boolean(row["active_now"]):
                active.add(model)
    return sorted(frames.items())


def response_rows(events):
    groups = collections.defaultdict(list)
    for event in events:
        groups[(event["method"], event["scenario_id"])].append(event)
    assert len(groups) == 200
    bins, runs = [], []
    for method in METHODS:
        method_groups = [group for (name, _), group in groups.items() if name == method]
        assert {(int(float(g[0]["speed_mm_s"])), int(g[0]["repeat"])) for g in method_groups} == {
            (speed, repeat) for speed in SPEEDS for repeat in range(1, 11)}
        for group in sorted(method_groups, key=lambda g: (float(g[0]["speed_mm_s"]), int(g[0]["repeat"]))):
            assert len(group) == 5 and {r["model_name"] for r in group} == set(TARGETS)
            event = group[0]
            origin = float(event["source_clock_origin_s"])
            assert all(float(row["source_clock_origin_s"]) == origin for row in group)
            metadata = dict(method=method, speed_mm_s=int(float(event["speed_mm_s"])),
                            repeat=int(event["repeat"]), scenario_id=event["scenario_id"])
            published = collections.defaultdict(set)
            positives = {model: collections.defaultdict(set) for model in TARGETS}
            for stamp, active in native_frames(event):
                assert active <= set(TARGETS)
                index, elapsed = bin_index(stamp, origin)
                if index is None:
                    continue
                published[index].add(elapsed)
                for model in active:
                    positives[model][index].add(elapsed)
            for model in TARGETS:
                target_bins = []
                for index in range(60):
                    row = dict(metadata, model_name=model, phase=phase_at(index),
                               bin_start_elapsed_s=index, bin_end_elapsed_s=index + 1,
                               available=int(bool(published[index])), active=int(bool(positives[model][index])),
                               published_frame_count=len(published[index]),
                               positive_frame_count=len(positives[model][index]))
                    assert row["active"] <= row["available"]
                    assert positives[model][index] <= published[index]
                    target_bins.append(row)
                bins.extend(target_bins)
                for phase, (start, end) in PHASES.items():
                    selected = target_bins[start:end]
                    available = sum(r["available"] for r in selected)
                    active = sum(r["active"] for r in selected)
                    times = sorted(t for index in range(start, end) for t in positives[model][index])
                    runs.append(dict(metadata, model_name=model, phase=phase,
                        phase_start_elapsed_s=start, phase_end_elapsed_s=end,
                        possible_bins=end-start, available_bins=available, active_bins=active,
                        missing_bins=end-start-available,
                        activity_fraction=active/available if available else None,
                        coverage_fraction=available/(end-start),
                        published_frame_count=sum(r["published_frame_count"] for r in selected),
                        positive_frame_count=len(times), any_response=bool(times),
                        first_response_elapsed_s=times[0] if times else None,
                        first_response_latency_s=times[0]-start if times else None))
            print("READ", method, metadata["speed_mm_s"], metadata["repeat"], flush=True)
    assert len(bins) == 60000 and len(runs) == 3000
    return bins, runs


def response_summary(runs):
    output = []
    for method in METHODS:
        for speed in SPEEDS:
            for phase, (start, end) in PHASES.items():
                rows = [r for r in runs if (r["method"], r["speed_mm_s"], r["phase"]) == (method, speed, phase)]
                assert len(rows) == 50
                activity, coverage = [], []
                for repeat in range(1, 11):
                    group = [r for r in rows if r["repeat"] == repeat]
                    assert len(group) == 5 and len({r["available_bins"] for r in group}) == 1
                    activity.append(moments([r["activity_fraction"] for r in group])[0])
                    coverage.append(statistics.mean(r["coverage_fraction"] for r in group))
                am, ast, an = moments(activity)
                cm, cst, _ = moments(coverage)
                delays = [r["first_response_latency_s"] for r in rows if r["any_response"]]
                output.append(dict(method=method, speed_mm_s=speed, phase=phase, n_runs=10,
                    target_count=5, target_run_count=50, phase_duration_s=end-start,
                    activity_mean_percent=None if am is None else 100*am,
                    activity_sd_percent=None if ast is None else 100*ast, n_activity_runs=an,
                    coverage_mean_percent=100*cm, coverage_sd_percent=100*cst,
                    possible_bins=sum(r["possible_bins"] for r in rows),
                    available_bins=sum(r["available_bins"] for r in rows),
                    active_bins=sum(r["active_bins"] for r in rows),
                    missing_bins=sum(r["missing_bins"] for r in rows),
                    response_target_run_count=len(delays),
                    first_response_latency_median_s=statistics.median(delays) if delays else None))
    return output


def accuracy_rows():
    original = [r for r in read_csv(PAPER / "projected_analysis/run_metrics.csv") if r["model_name"] in TARGETS]
    coverage = {(float(r["speed_mm_s"]), int(r["repeat"]), r["model_name"], r["phase"]): r
                for r in read_csv(PAPER / "projected_analysis/independent_coverage_runs.csv")
                if r["model_name"] in TARGETS}
    runs = []
    for source in original:
        row = dict(method="alert", speed_mm_s=int(float(source["speed_mm_s"])),
                   repeat=int(source["repeat"]), scenario_id=source["run_name"], model_name=source["model_name"])
        for phase, source_name in (("full", "vector_rmse_mm"), ("micro", "micro_vector_rmse_mm"),
                                   ("hold", "hold_vector_rmse_mm")):
            row["conditional_%s_rmse_mm" % phase] = common.number(source[source_name])
        for field in ("n_samples", "micro_n_samples", "hold_n_samples", "n_source_frames", "micro_n_source_frames"):
            row[field] = int(source[field])
        for phase in ("motion", "hold"):
            c = coverage[(row["speed_mm_s"], row["repeat"], row["model_name"], phase)]
            row[phase + "_coverage_percent"] = float(c["coverage_percent"])
            row[phase + "_evaluable_bins"] = int(c["evaluable_bins"])
            row[phase + "_possible_bins"] = int(c["possible_bins"])
        assert (row["conditional_full_rmse_mm"] is None) == (row["n_samples"] == 0)
        runs.append(row)
    assert len(runs) == 200
    assert {(r["speed_mm_s"], r["repeat"], r["model_name"]) for r in runs} == {
        (s, repeat, model) for s in SPEEDS for repeat in range(1, 11) for model in TARGETS}
    summaries, aggregates = [], []
    for speed in SPEEDS:
        for model in TARGETS:
            group = [r for r in runs if r["speed_mm_s"] == speed and r["model_name"] == model]
            row = dict(method="alert", speed_mm_s=speed, model_name=model, n_runs=10)
            for phase in ("full", "micro", "hold"):
                mean, sd, n = moments([r["conditional_%s_rmse_mm" % phase] for r in group])
                row.update({"conditional_%s_rmse_mean_mm" % phase: mean,
                            "conditional_%s_rmse_sd_mm" % phase: sd,
                            "conditional_%s_rmse_n_runs" % phase: n})
            for phase in ("motion", "hold"):
                mean, sd, _ = moments([r[phase + "_coverage_percent"] for r in group])
                row[phase + "_coverage_mean_percent"] = mean
                row[phase + "_coverage_sd_percent"] = sd
            summaries.append(row)
        for repeat in range(1, 11):
            group = [r for r in runs if r["speed_mm_s"] == speed and r["repeat"] == repeat]
            values = [r["conditional_full_rmse_mm"] for r in group if r["conditional_full_rmse_mm"] is not None]
            aggregates.append(dict(method="alert", speed_mm_s=speed, repeat=repeat,
                target_count=5, n_targets_with_error=len(values),
                conditional_full_rmse_mm=math.sqrt(statistics.mean(v*v for v in values)) if values else None,
                motion_coverage_percent=statistics.mean(r["motion_coverage_percent"] for r in group),
                hold_coverage_percent=statistics.mean(r["hold_coverage_percent"] for r in group)))
    return runs, summaries, aggregates


def conditions(events, aggregates):
    output = []
    for speed in SPEEDS:
        group = [r for r in events if r["method"] == "alert" and float(r["speed_mm_s"]) == speed]
        assert len(group) == 50
        hits = [r for r in group if common.boolean(r["detected"])]
        assert all(common.boolean(r["post_onset_only"]) and common.boolean(r["evidence_complete"]) for r in hits)
        delays = [float(r["confirmation_latency_s"]) for r in hits]
        translations = [float(r["truth_translation_at_confirmation_mm"]) for r in hits]
        ag = [r for r in aggregates if r["speed_mm_s"] == speed]
        mean, sd, n = moments([r["conditional_full_rmse_mm"] for r in ag])
        row = dict(method="alert", speed_mm_s=speed, n_runs=10, target_count=5, event_count=50,
            confirmed_count=len(hits), unconfirmed_count=50-len(hits), confirmation_percent=2*len(hits),
            delay_mean_s=statistics.mean(delays) if delays else None,
            delay_min_s=min(delays) if delays else None,
            confirmation_latency_median_s=common.percentile(delays, 50),
            confirmation_latency_q25_s=common.percentile(delays, 25),
            confirmation_latency_q75_s=common.percentile(delays, 75),
            translation_median_mm=common.percentile(translations, 50),
            restricted_wait_s=(sum(delays)+40*(50-len(hits)))/50,
            conditional_full_rmse_mean_mm=mean, conditional_full_rmse_sd_mm=sd,
            n_runs_with_error=n, eligible_target_runs=sum(r["n_targets_with_error"] for r in ag),
            n_targets_with_error_min=min(r["n_targets_with_error"] for r in ag),
            n_targets_with_error_max=max(r["n_targets_with_error"] for r in ag))
        for phase in ("motion", "hold"):
            mean, sd, _ = moments([r[phase + "_coverage_percent"] for r in ag])
            row[phase + "_coverage_mean_percent"] = mean
            row[phase + "_coverage_sd_percent"] = sd
        output.append(row)
    return output


def self_check():
    assert [bin_index(t, 1000)[0] for t in (999, 1000, 1009.999999999, 1010, 1049.999999999, 1050, 1059.999999999, 1060)] == [None, 0, 9, 10, 49, 50, 59, None]
    assert bin_index(1268.359, 1258.359)[0] == 10
    assert [phase_at(i) for i in (0, 9, 10, 49, 50, 59)] == ["baseline", "baseline", "motion", "motion", "hold", "hold"]
    assert moments([None])[0] is None  # Missing output must not become a negative.
    assert math.sqrt(statistics.mean(v*v for v in (3, 4))) == math.sqrt(12.5)
    assert len(TARGETS) == 5 and "m04" not in TARGETS and "OTHER" not in TARGETS


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    self_check()
    if args.self_check:
        print("SELF-CHECK PASSED")
        return
    events = [r for filename in ("confirmation_analysis/events.csv", "m3c2_analysis/events.csv")
              for r in read_csv(PAPER / filename) if r["model_name"] in TARGETS]
    assert len(events) == 1000
    bins, response_runs = response_rows(events)
    response = response_summary(response_runs)
    accuracy_runs, accuracy, aggregates = accuracy_rows()
    condition = conditions(events, aggregates)
    tables = dict(response_bins=bins, response_runs=response_runs, response_summary=response,
                  accuracy_runs=accuracy_runs, accuracy_summary=accuracy,
                  accuracy_aggregate_runs=aggregates, condition_summary=condition)
    DESTINATION.mkdir(exist_ok=True)
    for name, rows in tables.items():
        assert all("model_name" not in r or r["model_name"] in TARGETS for r in rows)
        common.write_csv(DESTINATION / (name + ".csv"), rows)
    supplementary = [r for r in read_csv(PAPER / "m3c2_analysis/accuracy_summary.csv") if r["model_name"] in TARGETS]
    assert len(supplementary) == 20
    supplement_path = DESTINATION / "m3c2_geometry_supplement.json"
    supplement_path.write_text(json.dumps(dict(
        policy="Descriptive same-normal errors on M3C2's own valid cores; not a paired precision ranking with Ours.",
        targets=TARGETS, rows=supplementary), indent=2, allow_nan=False) + "\n")
    source_paths = SOURCES | common.SOURCES | {Path(__file__).resolve(), Path(common.__file__).resolve()}
    source_paths.update(PAPER / name for name in ("projected_analysis/manifest.json", "m3c2_analysis/manifest.json",
                                                "confirmation_analysis/manifest.json"))
    paths = {name: str(DESTINATION / (name + ".csv")) for name in tables}
    paths["m3c2_geometry_supplement"] = str(supplement_path)
    summary = dict(schema_version=1, methods=METHODS, targets=TARGETS, speeds_mm_s=SPEEDS,
        n_method_runs=200, targets_per_run=5, repeats_per_speed=10,
        policies=dict(
            population="All five specified targets and all ten recorded runs per speed. No failure, missing error, or prior baseline activity excludes a target-run.",
            time="Source timestamp minus the retained truth clock origin, rounded only to ROS nanosecond resolution; half-open [0,10), [10,50), [50,60) phases and [b,b+1) bins.",
            response="Native target-positive point frame, or any active_now update among all MATCHED target GDUs, regardless of later confirmation. OTHER is never reassigned to a target. No persistence rule is applied to the cross-method response statistic.",
            availability="At least one global published frame in the bin. Ours uses timestamps present in the full GDU evolution stream, including non-target GDUs; empty messages absent from this CSV cannot establish availability. Publication is not target observability.",
            missing="available=0 means unobserved; active=0 on such a bin is not a negative observation. Activity is active bins / available bins; a zero denominator is missing. Coverage uses all nominal bins.",
            summary="Within each run and phase take the equal-target mean of activity fractions, then mean and sample SD over runs with output. Coverage always includes all ten runs. Pooled counts remain diagnostic, not an independent-bin significance test.",
            confirmation="Only Ours: retained native confirmation with complete supporting evidence wholly after onset, using unchanged confirmation_audit events. No M3C2 three-frame comparison.",
            accuracy="Only Ours: unchanged native-confirmed, nonzero-rank, current field samples after native confirmation; own projected truth >=5 mm, through 60 s. Full means no upper displacement bound, not unconditional 0–60 s accuracy. Micro adds projected truth <=20 mm. Empty errors stay missing.",
            accuracy_aggregation="Per run sqrt(mean of available target-wise conditional full-range RMSE squared)); summary mean/sample SD over run scores. Available-target counts accompany every score. No zero substitution for missing targets.",
            estimation_coverage="Independent published geometry cohort without confirmation, association-survival, or magnitude gating; 0.5 s bins, motion 80 and hold 20 per target-run. This differs from the conditional error cohort and is temporal availability, not surface completeness.",
            supplement="M3C2 native normal-distance errors remain supplementary and do not enter the primary accuracy table."),
        schema={name: list(rows[0]) for name, rows in tables.items()},
        row_counts={name: len(rows) for name, rows in tables.items()}, paths=paths,
        condition_summary=condition, response_summary=response, accuracy_summary=accuracy,
        sources=[dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path)) for path in sorted(source_paths)],
        output_sha256={Path(path).name: digest(path) for path in paths.values()})
    (DESTINATION / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print("PAPER ANALYSIS COMPLETE", DESTINATION)
    print(json.dumps(condition, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
