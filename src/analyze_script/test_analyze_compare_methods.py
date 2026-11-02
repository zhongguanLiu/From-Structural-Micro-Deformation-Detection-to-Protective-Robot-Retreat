#!/usr/bin/env python3
import importlib.util
import pathlib
import csv
import hashlib
import json
import math
import sys


SCRIPT = pathlib.Path(__file__).with_name("analyze_compare_methods.py")
SPEC = importlib.util.spec_from_file_location("comparison_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def verify_output(output):
    """Independent identities over a generated batch; run after full regeneration."""
    def read(name):
        with (output / name).open() as stream:
            return list(csv.DictReader(stream))
    events = read("confirmation_events.csv")
    assert len(events) == 960
    assert len({(r["method"],r["speed_mm_s"],r["repeat"],r["model_name"]) for r in events}) == 960
    physical = read("confirmation_displacement_summary.csv")
    curves = read("confirmation_displacement_curves.csv")
    common = read("integrated_method_speed_summary.csv")
    for s in physical:
        selected = [e for e in events if e["method"] == s["method"] and e["speed_mm_s"] == s["speed_mm_s"]
                    and e["included_in_primary_metrics"] == "True"]
        confirmed = [e for e in selected if e["detected"] == "True"]
        assert len(selected) == 50 and len(confirmed) == int(s["confirmed_count"])
        if confirmed:
            assert math.isclose(float(s["translation_median_mm"]), MODULE.percentile(
                [float(e["truth_translation_at_confirmation_mm"]) for e in confirmed],50),abs_tol=1e-10)
        else:
            assert s["translation_median_mm"] == ""
        for c in (c for c in curves if c["method"] == s["method"] and c["speed_mm_s"] == s["speed_mm_s"]):
            count = sum(float(e["truth_translation_at_confirmation_mm"]) <= float(c["translation_mm"]) for e in confirmed)
            assert math.isclose(float(c["confirmed_fraction"]), count/50, abs_tol=1e-12)
        r = next(r for r in common if r["method"] == s["method"] and r["speed_mm_s"] == s["speed_mm_s"])
        assert math.isclose(float(r["primary_recall"]),len(confirmed)/50,abs_tol=1e-12)
    bins = read("alert_tracking_bins.csv")
    assert len(bins) == 40*6*120 and len(read("alert_accuracy_runs.csv")) == 240
    assert all((r["estimate_mm"] == "") == (int(r["sample_count"]) == 0) for r in bins)
    for r in read("source_data_manifest.csv"):
        digest = hashlib.sha256()
        with pathlib.Path(r["path"]).open("rb") as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b""):
                digest.update(chunk)
        assert digest.hexdigest() == r["sha256"], r["path"]
    manifest = json.loads((output / "analysis_manifest.json").read_text())
    assert len(manifest["figures"]) == 11
    for check in manifest["figure_layout_checks"]:
        assert check["data_panels"] == check["rows"]*check["columns"]
        assert check["empty_panels"] == 0 and check["text_within_canvas"]
    for name, digest in manifest["generator_sha256"].items():
        assert hashlib.sha256(SCRIPT.with_name(name).read_bytes()).hexdigest() == digest
    print("full-output consistency and source hashes: PASS")


def frame(stamp, points):
    return {"source_stamp_s": stamp, "point_count": points,
            "elapsed_s": stamp, "motion_elapsed_s": max(0.0, stamp - 10.0)}


def main():
    start, end = MODULE.confirmation(
        [frame(10.0, 1), frame(10.1, 2), frame(10.2, 1)], 1, 3, 0.35)
    assert start["source_stamp_s"] == 10.0 and end["source_stamp_s"] == 10.2
    assert MODULE.confirmation(
        [frame(10.0, 1), frame(10.5, 1), frame(10.6, 1)], 1, 3, 0.35) == (None, None)
    assert MODULE.confirmation(
        [frame(10.0, 0), frame(10.1, 1), frame(10.2, 0)], 1, 1, 0.35)[0][
            "source_stamp_s"] == 10.1
    assert abs(MODULE.quaternion_angle((0, 0, 0, 1), (0, 0, 0, 1))) < 1e-12
    low, high = MODULE.wilson_interval(0, 10)
    assert low == 0.0 and 0.27 < high < 0.29
    events = [
        {"method": method, "speed_mm_s": speed, "model_name": model,
         "repeat": repeat, "detected": method == "alert",
         "included_in_primary_metrics": model in MODULE.PRIMARY_MODELS,
         "confirmation_latency_s": 2.0 if method == "alert" else None}
        for method in MODULE.ALL_METHODS for speed in MODULE.SPEEDS
        for model in MODULE.MODELS for repeat in range(1, 11)
    ]
    targets = MODULE.object_detection_rows(events)
    assert len(targets) == 96
    assert next(row for row in targets if row["method"] == "alert")[
        "detection_probability"] == 1.0
    runtime = [
        {"method": method, "speed_mm_s": speed, "repeat": repeat,
         "scenario_id": "{}_{}_{}".format(method, speed, repeat),
         "runtime_mean_ms": 100.0, "runtime_p95_ms": 180.0,
         "runtime_frame_count": 2, "runtime_samples_ms": [20.0, 180.0]}
        for method in MODULE.ALL_METHODS for speed in MODULE.SPEEDS
        for repeat in range(1, 11)
    ]
    summary, dynamics, runs = MODULE.shared_statistics(events, runtime, 500, 2027)
    assert len(runs) == 160
    alert = next(row for row in summary if row["method"] == "alert")
    assert alert["primary_recall"] == 1.0 and alert["restricted_mean_wait_s"] == 2.0
    failed = next(row for row in summary if row["method"] == "m_detector")
    assert failed["confirmation_latency_median_s"] is None
    assert failed["restricted_mean_wait_s"] == 40.0
    assert next(r for r in dynamics if r["method"] == "alert"
                and r["motion_elapsed_s"] == 2.0)["confirmed_count"] == 50
    # Five targets in one run must move together in the bootstrap.
    est, lo, hi = MODULE.cluster_mean_interval([0] * 5 + [1] * 5, 10000, 2027)
    assert est[0] == .5 and lo[0] <= .2 and hi[0] >= .8
    # Actual pooled durations, not a percentile of run percentiles.
    varied = [dict(r, runtime_samples_ms=[1.] * 99 + [1000.]) for r in runtime]
    checked, _, _ = MODULE.shared_statistics(events, varied, 20, 2027)
    assert checked[0]["runtime_p95_ms_pooled"] == 1.
    # Exact off-grid event time and right-censoring area (not a 0.5 s approximation).
    special = [dict(r, confirmation_latency_s=.123) if r["detected"] else dict(r)
               for r in events]
    s, curves, _ = MODULE.shared_statistics(special, runtime, 20, 2027)
    curve = [r for r in curves if r["method"] == "alert" and r["speed_mm_s"] == 1.]
    assert curve[1]["motion_elapsed_s"] == .123
    area = sum((b["motion_elapsed_s"] - a["motion_elapsed_s"])
               * (1 - a["cumulative_confirmation_fraction"])
               for a, b in zip(curve, curve[1:]))
    assert abs(area - .123) < 1e-12
    try:
        MODULE.confirmation_event_rows([], events + [events[0]])
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate target events must not pass validation")
    # Truth interpolation must not extrapolate or substitute commanded motion.
    assert MODULE.interpolate_truth([0, 1, 2], [0, 2, 5], .5) == 1
    for times, values, when in (([0,1],[0,1],2), ([0,0],[0,1],0)):
        try:
            MODULE.interpolate_truth(times, values, when)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid truth support accepted")
    import numpy as np
    q = MODULE.compare_figures.column_quantiles(np.array([[1,np.nan],[3,np.nan]]))
    assert q[1,0] == 2 and np.isnan(q[:,1]).all()
    # All six target axes contain truth even when no formal estimate exists.
    MODULE.compare_figures.configure()
    tracking = [{"speed_mm_s": 1., "model_name": model, "repeat": repeat,
                 "experiment_elapsed_s": .5*k+.25, "estimate_mm": None,
                 "truth_translation_mm": min(max(.5*k-10,0),40), "matched_truth_mm": None}
                for model in MODULE.MODELS for repeat in range(1,11) for k in range(120)]
    fig = MODULE.compare_figures.displacement_tracking(tracking, [], events, 1.)
    assert len(fig.axes) == 6 and all(ax.axison and ax.has_data() for ax in fig.axes)
    MODULE.plt.close(fig)
    print("comparison analysis unit checks: PASS")


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2:
        verify_output(pathlib.Path(sys.argv[1]).resolve())
