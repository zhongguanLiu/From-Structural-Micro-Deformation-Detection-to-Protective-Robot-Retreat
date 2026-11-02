#!/usr/bin/env python3
"""Rebuild tables and verify five-target paper statistics, figures and bilingual text."""
import collections
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path
PAPER = Path(__file__).resolve().parent
OUTPUT = PAPER.parent.parent / "experiment_output"
CONFIRMATION = PAPER / "confirmation_analysis"
M3C2 = PAPER / "m3c2_analysis"
EVACUATION = OUTPUT / "evacuation_paper_run/closed_loop/icra2027_20260911_235806/analysis/evacuation"
RESULTS = PAPER / "paper_analysis"
SPEEDS = (1, 3, 5, 10)
METHODS = ("m_detector", "spatiotemporal_normals", "dynablox", "m3c2", "alert")
LABELS = ("M-detector", "ST normals", "Dynablox", "M3C2", "Ours")
# Original population for the immutable recording audits, not manuscript statistics.
MODELS = ("m01", "m02", "m03", "m04", "m05", "m06")
PAPER_MODELS = ("m01", "m02", "m03", "m05", "m06")


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def close(a, b):
    assert math.isclose(float(a), float(b), abs_tol=1e-9, rel_tol=1e-9), (a, b)


def runtime_summaries():
    audit = json.loads((PAPER / "runtime_audit.json").read_text())
    result = []
    for method in METHODS:
        runs = (audit["static_runs"] if method == "alert" else read_csv(M3C2 / "runtime_runs.csv")
                if method == "m3c2" else [r for r in audit["baseline_runs"] if r["method"] == method])
        assert len(runs) == 40 and len({r["runtime_source"] for r in runs}) == 40
        assert all(sum(float(r["speed_mm_s"]) == s for r in runs) == 10 for s in SPEEDS)
        values = [float(r["cycle_mean_ms"]) for r in runs]
        assert all(math.isfinite(v) and v > 0 for v in values)
        result.append(dict(method=method, n_runs=40, mean_ms=statistics.mean(values),
                           sd_ms=statistics.stdev(values)))
    return result


def paper_summary():
    summary = json.loads((RESULTS / "summary.json").read_text())
    assert summary["targets"] == list(PAPER_MODELS)
    assert summary["methods"] == list(METHODS) and summary["n_method_runs"] == 200
    return summary


def mean_sd(mean, sd):
    return "$" + r"{:.2f}\pm{:.2f}".format(mean, sd) + "$"


def table_rows():
    summary = paper_summary()
    result = {name: [] for name in ("performance", "runtime")}
    for row in summary["condition_summary"]:
        assert row["event_count"] == 50 and row["target_count"] == 5
        result["performance"].append(" & ".join((
            str(row["speed_mm_s"]), str(row["confirmed_count"]),
            f'{row["delay_mean_s"]:.2f}', f'{row["translation_median_mm"]:.2f}',
            mean_sd(row["conditional_full_rmse_mean_mm"], row["conditional_full_rmse_sd_mm"]))) + r" \\")
    for label, row in zip(LABELS, runtime_summaries()):
        result["runtime"].append(label + " & " + mean_sd(row["mean_ms"], row["sd_ms"]) + r" \\")
    return result


def table_block(text, label):
    matches = list(re.finditer(r"\\label\{tab:" + re.escape(label)
                              + r"\}.*?\\midrule\n(.*?)\\bottomrule", text, re.S))
    assert len(matches) == 1, (label, "Expected one literal table body")
    return matches[0]


def prose_values():
    summary = paper_summary()
    conditions = summary["condition_summary"]
    small = [r["translation_median_mm"] for r in conditions if r["speed_mm_s"] <= 5]
    runtime = {r["method"]: r["mean_ms"] for r in runtime_summaries()}
    low = next(r for r in conditions if r["speed_mm_s"] == 1)
    return (
        ", ".join(f'{r["confirmed_count"]}/{r["event_count"]}' for r in conditions),
        f"{min(small):.2f}--{max(small):.2f}",
        mean_sd(low["conditional_full_rmse_mean_mm"], low["conditional_full_rmse_sd_mm"]),
        f'{runtime["alert"]:.1f}', f'{runtime["m3c2"]:.1f}',
        f'{runtime["spatiotemporal_normals"]:.1f}--{runtime["dynablox"]:.1f}')


def check_paper_results(raw_hashes=False):
    from projected_accuracy import sha256
    from paper_results import self_check
    self_check()
    summary = paper_summary()
    for name, digest in summary["output_sha256"].items():
        assert sha256(RESULTS / name) == digest, name
    for source in summary["sources"]:
        if raw_hashes or Path(source["path"]).suffix == ".py":
            assert sha256(Path(source["path"])) == source["sha256"], source["path"]
    tables = {name: read_csv(path) for name, path in summary["paths"].items() if path.endswith(".csv")}
    for name, rows in tables.items():
        assert len(rows) == summary["row_counts"][name]
        assert all(r.get("model_name", PAPER_MODELS[0]) in PAPER_MODELS for r in rows)
    bins, runs = tables["response_bins"], tables["response_runs"]
    assert len(bins) == 60000 and len(runs) == 3000
    groups, unique = collections.defaultdict(list), set()
    for row in bins:
        key = (row["method"], row["scenario_id"], row["model_name"], row["phase"])
        index = int(row["bin_start_elapsed_s"])
        assert (*key, index) not in unique
        unique.add((*key, index))
        start, end = {"baseline": (0, 10), "motion": (10, 50), "hold": (50, 60)}[row["phase"]]
        assert start <= index < end and int(row["bin_end_elapsed_s"]) == index + 1
        assert int(row["active"]) == int(int(row["positive_frame_count"]) > 0)
        assert int(row["available"]) == int(int(row["published_frame_count"]) > 0)
        assert 0 <= int(row["active"]) <= int(row["available"]) <= 1
        assert int(row["positive_frame_count"]) <= int(row["published_frame_count"])
        groups[key].append(row)
    for row in runs:
        group = groups[(row["method"], row["scenario_id"], row["model_name"], row["phase"])]
        available, active = (sum(int(r[k]) for r in group) for k in ("available", "active"))
        assert len(group) == int(row["possible_bins"])
        assert available == int(row["available_bins"]) and active == int(row["active_bins"])
        assert len(group) - available == int(row["missing_bins"])
        close(row["coverage_fraction"], available / len(group))
        if available:
            close(row["activity_fraction"], active / available)
        else:
            assert row["activity_fraction"] == ""
    for row in summary["response_summary"]:
        selected = [r for r in runs if r["method"] == row["method"]
                    and float(r["speed_mm_s"]) == row["speed_mm_s"] and r["phase"] == row["phase"]]
        assert len(selected) == row["target_run_count"] == 50
        means, coverage = [], []
        for repeat in range(1, 11):
            group = [r for r in selected if int(r["repeat"]) == repeat]
            assert len(group) == 5 and {r["model_name"] for r in group} == set(PAPER_MODELS)
            values = [float(r["activity_fraction"]) for r in group if r["activity_fraction"]]
            if values:
                means.append(100 * statistics.mean(values))
            coverage.append(100 * statistics.mean(float(r["coverage_fraction"]) for r in group))
        assert len(means) == row["n_activity_runs"]
        if means:
            close(row["activity_mean_percent"], statistics.mean(means))
        if len(means) > 1:
            close(row["activity_sd_percent"], statistics.stdev(means))
        close(row["coverage_mean_percent"], statistics.mean(coverage))
    raw = {(r["speed_mm_s"], r["repeat"], r["model_name"]): r for r in read_csv(PAPER / "projected_analysis/run_metrics.csv")}
    raw_coverage = {(r["speed_mm_s"], r["repeat"], r["model_name"], r["phase"]): r
                    for r in read_csv(PAPER / "projected_analysis/independent_coverage_runs.csv")}
    accuracy = tables["accuracy_runs"]
    assert len(accuracy) == 200
    for row in accuracy:
        key = (row["speed_mm_s"], row["repeat"], row["model_name"])
        original = raw[key]
        for new, old in (("conditional_full_rmse_mm", "vector_rmse_mm"), ("conditional_micro_rmse_mm", "micro_vector_rmse_mm"),
                         ("conditional_hold_rmse_mm", "hold_vector_rmse_mm")):
            if original[old]:
                close(row[new], original[old])
            else:
                assert row[new] == ""
        for phase in ("motion", "hold"):
            close(row[phase + "_coverage_percent"], raw_coverage[(*key, phase)]["coverage_percent"])
    source_events = read_csv(CONFIRMATION / "events.csv")
    for row in summary["condition_summary"]:
        events = [r for r in source_events if r["method"] == "alert"
                  and float(r["speed_mm_s"]) == row["speed_mm_s"] and r["model_name"] in PAPER_MODELS]
        assert len(events) == row["event_count"] == 50
        hits = [r for r in events if r["detected"] == "True"]
        assert row["confirmed_count"] == len(hits)
        assert all(r["post_onset_only"] == r["evidence_complete"] == "True" for r in hits)
        delays = [float(r["confirmation_latency_s"]) for r in hits]
        close(row["delay_mean_s"], statistics.mean(delays))
        close(row["delay_min_s"], min(delays))
        close(row["translation_median_mm"], statistics.median(float(r["truth_translation_at_confirmation_mm"]) for r in hits))
        close(row["restricted_wait_s"], (sum(delays) + 40 * (50 - len(hits))) / 50)
        scores, motion, hold, count = [], [], [], 0
        for repeat in range(1, 11):
            selected = [r for r in accuracy if float(r["speed_mm_s"]) == row["speed_mm_s"] and int(r["repeat"]) == repeat]
            values = [float(r["conditional_full_rmse_mm"]) for r in selected if r["conditional_full_rmse_mm"]]
            count += len(values)
            if values:
                scores.append(math.sqrt(statistics.mean(v*v for v in values)))
            motion.append(statistics.mean(float(r["motion_coverage_percent"]) for r in selected))
            hold.append(statistics.mean(float(r["hold_coverage_percent"]) for r in selected))
        assert count == row["eligible_target_runs"]
        close(row["conditional_full_rmse_mean_mm"], statistics.mean(scores))
        close(row["conditional_full_rmse_sd_mm"], statistics.stdev(scores))
        close(row["motion_coverage_mean_percent"], statistics.mean(motion))
        close(row["hold_coverage_mean_percent"], statistics.mean(hold))
    print("PASS: five targets; 60,000 time bins; missingness, response, confirmation, conditional error and coverage.")

def check_projected_audit(raw_hashes=False):
    folder = PAPER / "projected_analysis"
    audit = json.loads((folder / "manifest.json").read_text())
    assert audit["run_count"] == len(audit["runs"]) == 40
    assert audit["script_sha256"] == hashlib.sha256((PAPER / "projected_accuracy.py").read_bytes()).hexdigest()
    for name, digest in audit["outputs_sha256"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
    assert audit["totals"]["n_samples"] == audit["totals"]["n_inconclusive_accuracy_samples"]
    metrics = read_csv(folder / "run_metrics.csv")
    assert len(metrics) == 240
    assert sum(int(r["n_samples"]) for r in metrics) == audit["totals"]["n_samples"]
    for row in metrics:
        assert (int(row["n_samples"]) > 0) == bool(row["vector_rmse_mm"])
        assert (int(row["hold_n_samples"]) > 0) == bool(row["hold_vector_rmse_mm"])
        assert (int(row["micro_n_samples"]) > 0) == bool(row["micro_vector_rmse_mm"])
        assert int(row["micro_n_samples"]) <= int(row["n_samples"])
        assert sum(int(v) for k, v in row.items() if k.startswith("excluded_") and k.endswith("_samples")) + int(row["n_samples"]) == int(row["confirmed_track_samples"])
    from projected_accuracy import self_check, sha256
    for run in audit["runs"]:
        assert run["duplicate_sample_keys"] == run["missing_reference_joins"] == 0
        if raw_hashes:
            for path, digest in run["inputs_sha256"].items():
                assert sha256(Path(path)) == digest, path
    self_check()
    independent = read_csv(folder / "independent_bins.csv")
    assert len(independent) == 240 * 120
    assert len({(r["run_id"], r["model_name"], r["bin_start_elapsed_s"]) for r in independent}) == len(independent)
    for row in independent:
        assert row["evaluable_available"] in ("True", "False")
        assert (row["evaluable_available"] == "True") == (int(row["n_evaluable_samples"]) > 0)
        assert sum(int(row[f"rank{k}_n_samples"]) for k in range(4)) == int(row["n_published_samples"])
    for row in read_csv(folder / "independent_coverage.csv"):
        start, end, denominator = (10, 50, 800) if row["phase"] == "motion" else (50, 60, 200)
        selected = [r for r in independent if r["speed_mm_s"] == row["speed_mm_s"]
                    and r["model_name"] == row["model_name"]
                    and start <= float(r["bin_start_elapsed_s"]) < end]
        assert len(selected) == denominator
        assert sum(r["evaluable_available"] == "True" for r in selected) == int(row["evaluable_bins"])
    print(f'PASS: projected-vector audit, {audit["totals"]["n_samples"]} eligible samples in 40 runs; raw hashes={raw_hashes}.')


def check_confirmation_audit(raw_hashes=False):
    from confirmation_audit import statistics_rows
    from projected_accuracy import sha256
    audit = json.loads((CONFIRMATION / "manifest.json").read_text())
    assert audit["event_count"] == 960 and audit["bootstrap_samples"] == 10000
    events = read_csv(CONFIRMATION / "events.csv")
    for row in events:
        row["repeat"] = int(row["repeat"])
        row["detected"] = row["detected"] == "True"
        for key in ("speed_mm_s", "confirmation_latency_s", "truth_translation_at_confirmation_mm",
                    "truth_translation_at_window_end_mm"):
            row[key] = float(row[key]) if row[key] else None
        for n in (1, 3, 5):
            row[f"detected_{n}_frame"] = row[f"detected_{n}_frame"] == "True"
        assert not row["detected"] or 0 <= row["confirmation_latency_s"] < 40
        if row["method"] == "alert" and row["detected"]:
            assert row["evidence_complete"] == row["post_onset_only"] == "True"
            frames = json.loads(row["counted_source_frames"])
            assert len(frames) == 3
            onset = float(row["source_clock_origin_s"]) + float(row["motion_onset_experiment_s"])
            assert all(float(frame["source_stamp_s"]) >= onset
                       for frame in frames)
    assert len({(r["method"], r["speed_mm_s"], r["repeat"], r["model_name"]) for r in events}) == 960
    # Recompute both point estimates and run-cluster intervals from events.
    for name, expected in zip(("summary", "curves"), statistics_rows(events)):
        actual = read_csv(CONFIRMATION / (name + ".csv"))
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual):
            for key, value in left.items():
                if value is None:
                    assert right[key] == ""
                elif isinstance(value, str):
                    assert right[key] == value
                else:
                    assert math.isclose(float(right[key]), value, abs_tol=1e-10, rel_tol=1e-12), (name, key)
    if raw_hashes:
        for source in audit["sources"]:
            assert sha256(Path(source["path"])) == source["sha256"], source["path"]
    print(f"PASS: 960 source-clock events and 10,000 run-cluster bootstrap resamples; raw hashes={raw_hashes}.")


def check_m3c2_audit(raw_hashes=False):
    from confirmation_audit import statistics_rows
    from projected_accuracy import sha256
    audit = json.loads((M3C2 / 'manifest.json').read_text())
    assert audit['run_count'] == 40 and audit['event_count'] == 240
    assert sha256(PAPER.parent / 'analyze_script/analyze_m3c2.py') == audit['script_sha256']
    batch = Path(audit['batch'])
    ledger = read_csv(batch / 'batch_status.csv')
    assert len(ledger) == 40 and all(r['status'] == 'SUCCESS' for r in ledger)
    for path, digest in audit['helper_sha256'].items():
        assert sha256(Path(path)) == digest, path
    for name, digest in audit['output_sha256'].items():
        assert sha256(batch / 'analysis' / name) == digest, name
        if (M3C2 / name).exists():
            assert sha256(M3C2 / name) == digest, name
    events = read_csv(M3C2 / 'events.csv')
    assert len(events) == 240
    assert {(float(r['speed_mm_s']), int(r['repeat']), r['model_name']) for r in events} == {
        (s, r, m) for s in SPEEDS for r in range(1,11) for m in MODELS}
    for r in events:
        r['repeat'] = int(r['repeat'])
        for key in ('speed_mm_s','confirmation_latency_s','truth_translation_at_confirmation_mm',
                    'truth_translation_at_window_end_mm'):
            r[key] = float(r[key]) if r[key] else None
        for key in ('detected','detected_1_frame','detected_3_frame','detected_5_frame'):
            r[key] = r[key] == 'True'
    for name, expected in zip(('summary','curves'), statistics_rows(events, methods=('m3c2',), labels={'m3c2':'M3C2'})):
        actual = read_csv(M3C2 / (name+'.csv'))
        assert len(actual) == len(expected)
        for a,b in zip(actual,expected):
            for key,value in b.items():
                if value is None: assert a[key] == ''
                elif isinstance(value,str): assert a[key] == value
                else: assert math.isclose(float(a[key]),value,abs_tol=1e-10)
    accuracy = read_csv(M3C2 / 'accuracy_runs.csv')
    for r in read_csv(M3C2 / 'accuracy_summary.csv'):
        group = [a for a in accuracy if (a['speed_mm_s'],a['model_name']) == (r['speed_mm_s'],r['model_name'])]
        assert len(group) == 10
        for key in ('micro_rmse_mm','confirmed_micro_rmse_mm','all_rmse_mm','hold_rmse_mm'):
            values = [float(a[key]) for a in group if a[key]]
            assert len(values) == int(r[key+'_n'])
            if values: assert math.isclose(statistics.mean(values),float(r[key+'_mean']))
            if len(values)>1: assert math.isclose(statistics.stdev(values),float(r[key+'_sd']),abs_tol=1e-10)
    if raw_hashes:
        for source in audit['sources']:
            assert sha256(Path(source['path'])) == source['sha256'], source['path']
        for r in ledger:
            run = batch / 'runs' / r['scenario_id']
            check = json.loads((run / 'm3c2/record_check.json').read_text())
            for name,digest in check['record_sha256'].items():
                assert sha256(run / name) == digest, name
    print('PASS: M3C2 40 runs, 240 events, normal-distance errors and traceable sources.')


def check_response_audit(raw_hashes=False):
    from response_audit import self_check
    from projected_accuracy import sha256
    self_check()
    audit = json.loads((PAPER / "response_audit.json").read_text())
    summaries = {(r["method"], float(r["speed_mm_s"])): r for r in audit["summaries"]}
    for row in read_csv(CONFIRMATION / "summary.csv"):
        original = summaries[row["method"], float(row["speed_mm_s"])]
        assert original["first_eligible_post_onset_target_run_count"] == int(row["confirmed_count"])
        selected = [r for r in audit["event_diagnostics"] if r["method"] == row["method"]
                    and r["speed_mm_s"] == float(row["speed_mm_s"])]
        assert len(selected) == 60
        assert sum(r["motion_native_response_bool"] for r in selected) == original["n_native_response_target_runs"]
        assert int(row["confirmed_count"]) <= original["n_native_response_target_runs"] <= 60
        for event in selected:
            responded = event["motion_native_response_bool"]
            assert responded == (event["motion_native_response_count"] > 0)
            assert event["motion_native_response_count_unit"] == "DISTINCT_TARGET_SOURCE_FRAMES"
            delay = event["motion_native_response_latency_s"]
            assert (delay is None) if not responded else (0 <= delay < 40)
    assert sum(r["baseline_all_native_birth_count"] or 0 for r in audit["summaries"]) == 3
    assert summaries["alert", 3]["original_confirmed_baseline_assisted_count"] == 1
    assert summaries["alert", 3]["original_confirmed_count"] == 51
    assert math.isclose(audit["mobile_response"]["stop_to_trigger_sim_seconds"], 2.041)
    for source in audit["sources"]:
        if raw_hashes or Path(source["path"]).suffix == ".py":
            assert sha256(Path(source["path"])) == source["sha256"], source["path"]
    print(f"PASS: independent native-history and short-baseline audit; recorded stop/trigger timing; raw hashes={raw_hashes}.")



def check_real_results(en, zh, raw_hashes=False):
    """Check the real figure against its recorded source and contributing queries."""
    folder = PAPER / "real_analysis"
    summary = json.loads((folder / "summary.json").read_text())
    source = Path(summary["source_run"])
    execution = json.loads((source / "metadata/execution.json").read_text())
    recording = json.loads((source / "metadata/manifest.json").read_text())
    assert execution["complete"] and recording["complete"] and recording["closed_cleanly"]
    assert execution["playback_start_s"] == 0 and execution["playback_duration_s"] == 60
    assert summary["playback_window_s"] == [0, 60]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for path in folder.iterdir():
        assert digest(path) == digest(source / "analysis" / path.name), path
    manifest = json.loads((PAPER / "figures/layout_manifest.json").read_text())
    figure = manifest["outputs"]["fig_real_response.pdf"]
    assert digest(PAPER / "figures/fig_real_response.pdf") == figure["sha256"]
    assert figure["statistic"] == "display_selected_population_not_confirmation"
    assert figure["count_axis"] == "linear" and figure["panels"] == 2
    assert figure["playback_window_s"] == summary["playback_window_s"]
    assert figure["font_family"] == "Times New Roman"
    if raw_hashes:
        for name, expected in summary["sha256"].items():
            path = Path(name) if Path(name).is_absolute() else PAPER.parent / name
            assert digest(path) == expected, path
    rows = read_csv(folder / "response_timeline.csv")
    close(figure["maximum_mm"], max(float(r["maximum_mm"]) for r in rows if r["maximum_mm"]))
    samples = collections.defaultdict(list)
    for sample in read_csv(folder / "response_samples.csv"):
        assert 0.95 <= float(sample["active_probability_lower"]) <= 1
        assert float(sample["displacement_mm"]) >= 1
        samples[sample["time_s"]].append(sample)
    assert len(rows) == summary["snapshots"] == recording["written"]["snapshots"] == 87
    assert sum(int(r["responding_anchors"]) > 0 for r in rows) == summary["snapshots_with_display_response"] == 78
    assert max(int(r["responding_anchors"]) for r in rows) == 62
    assert all(0 <= float(r["time_s"]) <= 60 for r in rows)
    for row in rows:
        group = samples[row["time_s"]]
        values = [float(s["displacement_mm"]) for s in group]
        assert len({s["surfel_id"] for s in group}) == len(group) == int(row["responding_anchors"])
        assert (int(row["reference_anchors"]), int(row["anchors_with_estimates"])) == (13177, 1345)
        assert int(row["formal_detected_queries"]) == 0
        if values:
            close(statistics.median(values), row["median_mm"])
            close(max(values), row["maximum_mm"])
        else:
            assert row["median_mm"] == row["maximum_mm"] == ""
    first = next(r for r in rows if int(r["responding_anchors"]))
    peak = max((r for r in rows if r["maximum_mm"]), key=lambda r: float(r["maximum_mm"]))
    assert int(first["responding_anchors"]) == 12 and int(rows[-1]["responding_anchors"]) == 56
    for value, printed in ((rows[0]["time_s"], "13.14"), (first["time_s"], "15.54"),
                           (first["median_mm"], "8.01"), (peak["time_s"], "44.54"),
                           (peak["maximum_mm"], "588.96"), (peak["median_mm"], "45.39"),
                           (rows[-1]["time_s"], "59.94"), (rows[-1]["median_mm"], "10.64")):
        assert f"{float(value):.{len(printed.split('.')[1])}f}" == printed
        assert printed in en and printed in zh
    last = read_csv(folder / "response_intervals.csv")[-1]
    assert int(last["samples"]) == 47 and int(last["units_present_at_every_sample"]) == 0
    assert f"{float(last['first_sample_s']):.2f}" == "35.24"
    if raw_hashes:
        import numpy as np
        sys.path.insert(0, str(PAPER.parent / "analyze_script"))
        from analyze_v1_real import anchor_counts, frame_summary, records, reference_index, stamp
        references = list(records(source, "references"))
        assert len(references) == 1
        lookup = reference_index(references[0]["reference_support"])
        snapshots = list(records(source, "snapshots"))
        assert len(snapshots) == len(rows)
        for raw, row in zip(snapshots, rows):
            close(stamp(raw["common"]["header"]["stamp"])-summary["time_zero_s"], row["time_s"])
            samples_raw = raw["physical_deformation"]
            actual = dict(frame_summary(samples_raw, summary["recorded_display_gates"]),
                          **anchor_counts(samples_raw, lookup, summary["recorded_display_gates"]))
            for key, value in actual.items():
                if np.isfinite(value): close(value, row[key])
                else: assert row[key] == ""
            assert not any(e["detection_allowed"] and e["detection_admission"] == 0 for e in raw["detection_evidence"])
    print("PASS: real-data figure/source identity, anchor counts, response gates and reported displacement statistics.")


def check():
    for checker in (check_paper_results, check_projected_audit, check_confirmation_audit,
                    check_response_audit, check_m3c2_audit):
        checker("--truth" in sys.argv)
    en, zh = ((PAPER / name).read_text() for name in ("main_en.tex", "main_zh.tex"))
    for text in (en, zh):
        assert r"\author{}" in text and r"\thanks{" not in text, "Anonymous submission author block"
        assert not re.search(r"Zhongguan[~ ]+Liu|Shuaidong[~ ]+Yuan|Lingwei[~ ]+Li|Zhexuan[~ ]+Zhang|Lu[~ ]+Pan|Corresponding authors?|通讯作者|通信作者|China University of Mining and Technology|中国矿业大学", text), "Author-identifying manuscript text"
    equations = lambda t: [re.sub(r"\s+", "", x) for x in re.findall(r"\\begin\{equation\}(.*?)\\end\{equation\}", t, re.S)]
    graphics = lambda t: re.findall(r"\\includegraphics\[(.*?)\]\{(.*?)\}", t)
    labels = lambda t: re.findall(r"\\label\{(.*?)\}", t)
    cites = lambda t: {k for group in re.findall(r"\\cite\{(.*?)\}", t) for k in group.split(",")}
    assert equations(en) == equations(zh) and len(equations(en)) == 7
    introduction_and_related_work = [
        "surmann24deploy", "fastlio2", "liu2023efficient", "kalenjuk2021processing",
        "winiwarter2021", "zahs22m3c2", "tuna2023x", "cwian2025madba",
        "lyu2025registration", "wei2022grid", "williams2021multidirectional",
        "wu24moving", "chen2025breaking", "zhao2025superloc", "canh2026bpda"]
    assert cites(en) == cites(zh) and len(cites(en)) == 23
    entries = re.findall(r"@\w+\{([^,]+),", (PAPER / "references.bib").read_text())
    assert len(entries) == len(set(entries)) and cites(en) == set(entries)
    assert not any("alert" in k.lower() for k in cites(en))
    for name, text in (("main_en", en), ("main_zh", zh)):
        order = list(dict.fromkeys(k for group in re.findall(r"\\cite\{(.*?)\}", text)
                                   for k in group.split(",")))
        assert order[:15] == introduction_and_related_work
        numbered = re.findall(r"\\bibitem\{([^}]+)\}", (PAPER / (name+".bbl")).read_text())
        assert numbered == order, (name, "Bibliography must follow first citation order")
    assert set(labels(en)) == set(labels(zh)) and len(set(labels(en))) == len(labels(en))
    assert len(graphics(en)) == len(graphics(zh)) == 8  # Figure 7 uses two adjacent PDFs.
    assert {p for _, p in graphics(en)} == {p for _, p in graphics(zh)}
    check_real_results(en, zh, "--truth" in sys.argv)
    for text in (en, zh):
        assert not re.search(r"\bM04\b|\bALERT\b|dual.?lidar|single.?lidar|双雷达|单雷达", text, re.I)
        assert set(re.findall(r"\\(?:eq)?ref\{(.*?)\}", text)) <= set(labels(text))
        assert len(re.findall(r"\\begin\{figure\*?\}", text)) == 7
        assert len(re.findall(r"\\begin\{table\*?\}", text)) == 2
        assert not re.search(r"\\figspace\{", text)
        assert not re.search(r"\\(?:clearpage|newpage|pagebreak|IEEEtriggeratref)\b", text), "No reserved pages or forced bibliography breaks"
        assert r"\input{simulation_tables}" not in text, "Keep numerical results directly in the manuscript"
        assert set(re.findall(r"\\newcommand\{\\(\w+)\}", text)) <= {"R", "Normal", "trans"}, "Only mathematical shorthands remain"
    assert not re.search(r"\bsix targets\b|\bsix-target\b|\bof 60\b|/60\b|three-frame criterion", en, re.I)
    title = re.search(r"\\title\{([^}]*)\}", en)[1]
    assert " ".join(title.replace(r"\\", " ").split()) == (
        "From Structural Micro-Deformation Detection to Protective Robot Retreat: "
        "Uncertainty-Aware Estimation via Repeated LiDAR Observations")
    for label, rows in table_rows().items():
        for text in (en, zh):
            assert table_block(text, label)[1] == "\n".join(rows) + "\n", label
        nr, nc = {"performance": (4, 5), "runtime": (5, 2)}[label]
        data = [r for r in rows if " & " in r]
        assert len(data) == nr and all(len(r.split(" & ")) == nc for r in data)
    for value in prose_values():
        # Independently edited abstracts may repeat a result a different number
        # of times; its source-derived value must still appear in both texts.
        assert value in en and value in zh, (value, "Literal result must match source statistics in both languages")
    from projected_accuracy import sha256
    manifest = json.loads((PAPER / "figures/layout_manifest.json").read_text())
    supplied = {"overview.pdf", "Experimentalscene .png"}
    assert set(manifest["outputs"]) | supplied == {Path(p).name for _, p in graphics(en)}
    assert manifest["targets"] == list(PAPER_MODELS) and manifest["event_denominator"] == 50
    for path, digest in manifest["inputs"].items():
        assert sha256(Path(path)) == digest, path
    for _, relative in graphics(en):
        assert (PAPER / relative).is_file(), relative
        if Path(relative).name in supplied:
            continue  # Supplied illustrations are not outputs of render_figures.py.
        meta = manifest["outputs"][Path(relative).name]
        assert meta["panels"] in (1, 2, 3, 4, 6) and meta["font_points"] >= 7
        assert meta["width_inches"] == 3.5
        assert sha256(PAPER / relative) == meta["sha256"], relative
    assert not re.search(r'(?i)"(?:model|model_name)"\s*:\s*"m04"', json.dumps(manifest))
    assert len(runtime_summaries()) == 5
    audit = json.loads((PAPER / "runtime_audit.json").read_text())
    for path, digest in audit["source_sha256"].items():
        if "--truth" in sys.argv or Path(path).suffix == ".py":
            assert sha256(Path(path)) == digest, path
    metrics = json.loads((EVACUATION / "metrics.json").read_text())
    indexed = {r["event"]: r for r in metrics["event_lineage"]}
    states = metrics["events"]
    origin, observe, monitor, trigger, completed = (states[k] for k in (
        "NAVIGATING_OUTBOUND", "STABILIZING_AT_TARGET", "MONITORING",
        "DEFORMATION_DETECTED", "COMPLETED"))
    maps = manifest["outputs"]["fig_evacuation_map.pdf"]
    assert maps["timestamp_background"] == "none"
    for name in ("fig_evacuation_map.pdf", "fig_evacuation_response.pdf"):
        assert manifest["outputs"][name]["font_family"] == "Times New Roman"
        fonts = subprocess.check_output(["pdffonts", str(PAPER / "figures" / name)], text=True).splitlines()[2:]
        assert fonts and all(re.fullmatch(r"\S*TimesNewRoman\S*\s+CID TrueType\s+Identity-H\s+yes\s+yes\s+yes\s+\d+\s+\d+", row.strip())
                             for row in fonts), (name, "Expected only embedded Times New Roman fonts")
    assert maps["panel_cutoff_source_s"] == [origin, observe, trigger, completed]
    for row, stamp in zip(maps["keyframes"], maps["panel_cutoff_source_s"]):
        assert row["event_sim_time_s"] == row["data_cutoff_sim_time_s"] == stamp
        assert row["time_origin_sim_s"] == origin
        assert all(row[k] <= stamp for k in ("map_source_time_s", "map_receive_time_s",
                                             "pose_source_time_s", "pose_receive_time_s"))
    assert all(row["return_point_count"] == 0 for row in maps["keyframes"][:3])
    assert maps["keyframes"][3]["return_point_count"] > 0
    chain = manifest["outputs"]["fig_evacuation_response.pdf"]
    assert chain["statistic"] == "observation_to_trigger" and chain["time_origin_source_s"] == origin
    assert chain["observation_source_s"] == observe and chain["monitoring_source_s"] == monitor
    assert chain["trigger_source_s"] == trigger
    assert chain["motion_onset_source_s"] == indexed["truth_running"]["source_time_s"]
    assert chain["confirmation_source_s"] == indexed["first_retained"]["source_time_s"]
    assert chain["retention_endpoint_source_s"] == indexed["persistence_source"]["source_time_s"]
    close(chain["preparation_duration_s"], monitor-observe)
    close(chain["retained_duration_s"], chain["retention_endpoint_source_s"]-chain["confirmation_source_s"])
    assert chain["retained_duration_s"] >= 10
    clocks = ("source_time_s", "evidence_publish_time_s", "evidence_receive_time_s", "evolution_receive_time_s")
    evidence = sorted((row for row in read_csv(EVACUATION / "trigger_evidence_chain.csv")
                       if float(row["source_time_s"]) >= monitor and all(float(row[k]) <= trigger for k in clocks)),
                      key=lambda row: float(row["source_time_s"]))
    for key, values in (
        ("evidence_source_s", [float(row["source_time_s"]) for row in evidence]),
        ("evidence_received_s", [max(float(row[k]) for k in clocks[1:]) for row in evidence]),
        ("evidence_displacement_mm", [float(row["displacement_rms_mm"]) for row in evidence])):
        assert len(values) == len(chain[key]) > 2
        for expected, actual in zip(values, chain[key]):
            close(expected, actual)
    assert chain["evidence_source_s"][-1] == chain["retention_endpoint_source_s"]
    assert all(row["retained_for_avoidance"] == "True" for row in evidence
               if float(row["source_time_s"]) >= chain["confirmation_source_s"])
    published = {
        "17.04": metrics["events"]["DEFORMATION_DETECTED"] - indexed["truth_running"]["source_time_s"],
        "63.70": metrics["events"]["COMPLETED"] - metrics["events"]["RETURNING_HOME"],
        "23.955": metrics["executed"]["length_m"],
        "0.738": metrics["arrival_check"]["gazebo_home_distance_m"],
        "26.45": observe-origin, "57.93": monitor-origin, "31.48": monitor-observe,
        "60.93": chain["motion_onset_source_s"]-origin,
        "66.68": chain["confirmation_source_s"]-origin,
        "75.93": chain["recorded_motion_stop_source_s"]-origin,
        "77.97": trigger-origin, "10.17": chain["retained_duration_s"],
        "2.041": trigger-chain["recorded_motion_stop_source_s"],
    }
    for cell, value in published.items():
        assert f"{value:.{len(cell.split('.')[1])}f}" == cell and cell in en and cell in zh
    assert metrics["executed"]["evaluated_distance_m"] == metrics["executed"]["length_m"]
    for path in (PAPER / "main_en.tex", PAPER / "main_zh.tex"):
        text = path.read_text()
        for label, nc in (("performance", 5), ("runtime", 2)):
            header = text.split(r"\label{tab:" + label + "}", 1)[1].split(r"\midrule", 1)[0]
            rows = [r for r in header.split(r"\\") if "&" in r]
            assert rows[-1].count("&") == nc - 1, (path, label)
        assert not re.search(r"Overfull|undefined|Missing character|Rerun to get cross-references",
                             path.with_suffix(".log").read_text()), path
        deps = [path, PAPER / "references.bib",
                *(PAPER / p for _, p in graphics(text))]
        assert path.with_suffix(".pdf").stat().st_mtime >= max(p.stat().st_mtime for p in deps)
    pdf = str(PAPER / "main_en.pdf")
    pages = int(re.search(r"Pages:\s+(\d+)", subprocess.check_output(["pdfinfo", pdf], text=True))[1])
    assert 2 <= pages <= 8, "Eight-page budget including references"
    text = subprocess.check_output(["pdftotext", pdf, "-"], text=True)
    conclusion = re.search(r"C\s*ONCLUSION", text)
    references = re.search(r"R\s*EFERENCES", text)
    assert conclusion and references and conclusion.start() < references.start()
    assert all(f"Fig. {i}." in text for i in range(1, 8))
    for path in (PAPER / "main_en.pdf", PAPER / "main_zh.pdf", *(PAPER / p for _, p in graphics(en))):
        if path.suffix.lower() != ".pdf":
            continue
        text = subprocess.check_output(["pdftotext", str(path), "-"], text=True)
        assert not re.search(r"\bM04\b|\bALERT\b", text, re.I), path
    print(f"PASS: two source-derived tables, seven numbered figures, bilingual text, five simulation targets and {pages}-page English layout.")


if __name__ == "__main__":
    if "--write-tables" in sys.argv:
        rows_by_table = table_rows()
        for name in ("main_en.tex", "main_zh.tex"):
            path = PAPER / name
            original = content = path.read_text()
            for label, rows in rows_by_table.items():
                match = table_block(content, label)
                content = content[:match.start(1)] + "\n".join(rows) + "\n" + content[match.end(1):]
            if content != original:
                path.write_text(content)
        print("Verified/wrote literal table rows in both manuscripts; no result macros generated.")
    elif "--data-only" in sys.argv:
        check_paper_results("--truth" in sys.argv)
    elif "--real-only" in sys.argv:
        check_real_results((PAPER / "main_en.tex").read_text(),
                           (PAPER / "main_zh.tex").read_text(), "--truth" in sys.argv)
    else:
        check()
