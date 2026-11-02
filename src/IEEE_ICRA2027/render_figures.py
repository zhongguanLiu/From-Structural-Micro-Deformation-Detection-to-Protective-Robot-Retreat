#!/usr/bin/env python3
"""Redraw paper figures from five-target analysis and the recorded mission."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.font_manager import FontProperties, findfont, fontManager
import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parent
ANALYSIS = PAPER / "paper_analysis"
OUTPUT = PAPER.parent.parent / "experiment_output"
EVACUATION = OUTPUT / "evacuation_paper_run/closed_loop/icra2027_20260911_235806/analysis/evacuation"
TARGETS = ("m01", "m02", "m03", "m05", "m06")
SPEEDS = (1, 3, 5, 10)
METHODS = ("m_detector", "spatiotemporal_normals", "dynablox", "m3c2", "alert")
LABELS = dict(zip(METHODS, ("M-detector", "ST normals", "Dynablox", "M3C2", "Ours")))
COLORS = dict(zip(METHODS, ("#2478A8", "#D68A24", "#14866D", "#98674D", "#762A83")))
OUTBOUND, RETURN, WARNING = "#2478A8", "#14866D", "#AB5360"
sys.path.insert(0, str(PAPER.parent / "analyze_script"))
import compare_figures as plots
import analyze_evacuation as evacuation


def hashes(paths):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in dict.fromkeys(paths)}


def geometry_digest(fig):
    """Ignore colour/layout; preserve curves, markers, polygons and map arrays."""
    digest = hashlib.sha256()
    for ax in fig.axes:
        arrays = [line.get_xydata() for line in ax.lines]
        arrays += [item.get_offsets() for item in ax.collections]
        arrays += [path.vertices for item in ax.collections for path in item.get_paths()]
        for item in ax.images:
            arrays += [np.ma.filled(item.get_array().astype(float), np.nan), item.get_extent()]
        for array in arrays:
            digest.update(np.asarray(array, dtype=np.float64).tobytes())
    return digest.hexdigest()


def save_paper_figure(fig, name, panels=None):
    for item in fig.findobj(plt.Text):
        item.set_fontsize(7.5)
        if name.startswith("fig_evacuation_"):
            assert item.get_fontproperties().get_name() == "Times New Roman"
    fig.canvas.draw()
    assert all(ax.axison and ax.has_data() for ax in fig.axes), "Empty panel"
    for item in fig.findobj(plt.Text):
        if item.get_visible() and item.get_text():
            box = item.get_window_extent(fig.canvas.get_renderer())
            assert (box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width+1
                    and box.y1 <= fig.bbox.height+1), item.get_text()
    path = PAPER / "figures" / name
    fig.savefig(path)
    result = dict(sha256=hashes([path])[str(path)], width_inches=float(fig.get_figwidth()),
                  height_inches=float(fig.get_figheight()), font_points=7.5,
                  panels=panels or len(fig.axes), geometry_sha256=geometry_digest(fig))
    plt.close(fig)
    return result


def phase_axes(ax):
    plots.phase_background(ax)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, pad=2)
    ax.set_axisbelow(True)


def phase_labels(ax):
    for left, right, label in ((0, 10, "Static"), (10, 50, "Motion"), (50, 60, "Hold")):
        ax.text((left+right)/2, 1.025, label, transform=ax.get_xaxis_transform(), ha="center", va="bottom")


def baseline_figure():
    """Distinct method rows show active fractions conditional on recorded output."""
    path = ANALYSIS / "response_bins.csv"
    bins = pd.read_csv(path)
    assert len(bins) == 60000 and set(bins.model_name) == set(TARGETS)
    assert bins.available.isin((0, 1)).all() and bins.active.le(bins.available).all()
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.65), sharex=True, sharey=True)
    fig.subplots_adjust(left=.245, right=.945, bottom=.13, top=.86, wspace=.20, hspace=.39)
    records = []
    for ax, speed, letter in zip(axes.flat, SPEEDS, "abcd"):
        phase_axes(ax)
        for index, method in enumerate(METHODS):
            group = bins[(bins.method == method) & (bins.speed_mm_s == speed)]
            assert len(group) == 3000
            grouped = group.groupby("bin_start_elapsed_s")
            assert grouped.size().eq(50).all()
            totals = grouped[["active", "available"]].sum().reindex(range(60))
            assert len(totals) == 60 and totals.notna().all().all()
            fraction = (totals.active / totals.available.where(totals.available.gt(0))).to_numpy()
            base = 4-index
            ax.hlines(base, 0, 60, color="#B8BDC1", lw=.45, zorder=1)
            ax.hlines(base+.70, 0, 60, color="#D9DDE1", lw=.35, ls=":", zorder=1)
            # The offset locates a method; its 0--100% amplitude remains linear.
            x, y = np.arange(61), np.r_[fraction, fraction[-1]]
            line, = ax.step(x, base+.70*y, where="post", color=COLORS[method], lw=1.0)
            ax.fill_between(x, base, base+.70*y, step="post", color=COLORS[method], alpha=.13, lw=0)
            assert np.allclose((line.get_ydata()-base)/.70, y, equal_nan=True)
            records.append(dict(method=method, speed_mm_s=speed,
                                active_counts=totals.active.astype(int).tolist(),
                                available_counts=totals.available.astype(int).tolist(),
                                fractions=[float(v) if np.isfinite(v) else None for v in fraction]))
        ax.set(ylim=(-.20, 4.98), yticks=np.arange(5)+.35,
               yticklabels=[LABELS[m] for m in reversed(METHODS)], xticks=(0, 30, 60))
        ax.tick_params(axis="y", length=0, pad=3)
        ax.set_title(f"({letter}) {speed} mm/s", loc="left", pad=15)
        if letter in "ab":
            phase_labels(ax)
        ax.plot([1.025, 1.025], [0, .70], transform=ax.get_yaxis_transform(),
                color="#444444", lw=.6, clip_on=False)
        for value, label in ((0, "0"), (.70, "1")):
            ax.text(1.025, value, label, transform=ax.get_yaxis_transform(),
                    ha="left", va="bottom" if value else "top")
    fig.text(.60, .977, "Fraction with a change indication (0–1)", ha="center", va="top")
    fig.supxlabel("Time since trial start (s)", y=.025)
    result = save_paper_figure(fig, "fig_baseline_comparison.pdf")
    result.update(inputs=hashes([path]), methods=list(METHODS), targets=list(TARGETS),
                  bin_seconds=1, target_runs_per_bin=50,
                  statistic="active_target_runs / available_target_runs", records=records)
    return result


def confirmation_figure():
    """First trial: projected measurement and recorded map retention."""
    folder = PAPER / "projected_analysis"
    retained_path = OUTPUT / "stationary_speed_sweep/batch_20260901_012253/runs/stationary_l1p0mmps_r01/analysis/object_motion_comparison.csv"
    event_path = PAPER / "confirmation_analysis/events.csv"
    paths = [folder / "tracking_bins.csv", retained_path, event_path]
    tracking, retained, events = [pd.read_csv(path) for path in paths]
    tracking = tracking[(tracking.speed_mm_s == 1) & (tracking.repeat == 1) & tracking.model_name.isin(TARGETS)]
    retained = retained[retained.model_name.isin(TARGETS) & retained.confirmed_detection_source & retained.retained_for_avoidance].copy()
    retained["elapsed_s"] = retained.experiment_elapsed_s_at_receive-retained.receive_time_s+retained.message_time_s
    retained = retained[retained.elapsed_s.between(0, 60, inclusive="left")]
    retained["bin"] = np.floor(retained.elapsed_s*2+1e-9).astype(int)
    events = events[(events.method == "alert") & (events.speed_mm_s == 1) & (events.repeat == 1) & events.model_name.isin(TARGETS)]
    example = tracking[tracking.model_name == "m01"]
    assert len(events) == 5 and len(example) == 120
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.20), sharex=True)
    fig.subplots_adjust(left=.20, right=.97, bottom=.14, top=.92, hspace=.52)
    for ax in axes:
        phase_axes(ax)
        ax.grid(axis="y", color="#D9DDE1", lw=.35)
    x = example.bin_start_elapsed_s.to_numpy()+.25
    axes[0].plot(x, example.estimate_projected_norm_mean_mm, color=COLORS["alert"], lw=1.15, label="Estimate")
    axes[0].plot(x, example.truth_projected_norm_mean_mm, color="#222222", ls="--", lw=.85, label="Truth")
    event = events[events.model_name == "m01"].iloc[0]
    confirm = float(event.confirmation_experiment_elapsed_s)
    axes[0].axvline(confirm, color="#444444", lw=.65, ls=":")
    axes[0].set(ylim=(-1, 44), yticks=(0, 20, 40), ylabel="Projected\ndisplacement (mm)")
    axes[0].legend(loc="upper left", frameon=False, handlelength=1.3, borderpad=.1)
    axes[0].text(.97, .10, "(a) M01", transform=axes[0].transAxes, ha="right")
    phase_labels(axes[0])
    records = []
    for index, model in enumerate(TARGETS):
        group = retained[retained.model_name == model]
        occupied = sorted(group.loc[np.isfinite(group.estimated_magnitude_m), "bin"].unique())
        axes[1].broken_barh([(int(b)/2, .5) for b in occupied], (index-.29, .58),
                            facecolor=COLORS["alert"], linewidth=0, alpha=.78)
        event = events[events.model_name == model].iloc[0]
        stamp = float(event.confirmation_experiment_elapsed_s) if event.detected else None
        if stamp is not None:
            axes[1].plot(stamp, index, marker="*", ms=5.5, color="#222222", ls="", zorder=4)
        records.append(dict(model=model, retained_bins=len(occupied), confirmation_s=stamp))
    axes[1].set(ylim=(4.55, -.55), yticks=range(5), yticklabels=[model.upper() for model in TARGETS],
                xlabel="Time since trial start (s)")
    axes[1].grid(False)
    axes[1].set_title("(b)", loc="left", pad=4)
    axes[1].legend(handles=[matplotlib.patches.Patch(color=COLORS["alert"], label="Retained"),
                            plt.Line2D([], [], ls="", marker="*", ms=5, color="#222222", label="Confirmed")],
                   loc="upper right", bbox_to_anchor=(1., 1.26), ncol=2, frameon=False,
                   handlelength=1., columnspacing=.9)
    result = save_paper_figure(fig, "fig_deformation_scale.pdf")
    result.update(inputs=hashes(paths), targets=list(TARGETS), speed=1, repeat=1,
                  population="fixed_first_run", bin_seconds=.5, retention=records)
    return result


def projected_figure():
    """Every eligible run remains visible, including large-excursion errors."""
    path = ANALYSIS / "accuracy_runs.csv"
    runs = pd.read_csv(path)
    assert len(runs) == 200 and set(runs.model_name) == set(TARGETS)
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.20), sharex=True, sharey=True)
    fig.subplots_adjust(left=.16, right=.97, bottom=.14, top=.89, wspace=.16, hspace=.38)
    records = []
    finite = runs.conditional_full_rmse_mm.dropna()
    assert len(finite) and finite.gt(0).all()
    high = float(np.ceil(finite.max()*1.12))
    for ax, speed, letter in zip(axes.flat, SPEEDS, "abcd"):
        for index, model in enumerate(TARGETS):
            group = runs[(runs.speed_mm_s == speed) & (runs.model_name == model)].sort_values("repeat")
            assert len(group) == 10 and group.repeat.nunique() == 10
            valid = group[group.conditional_full_rmse_mm.notna()]
            values = valid.conditional_full_rmse_mm.to_numpy()
            assert valid.n_samples.gt(0).all()
            # Fixed repeat offsets expose every observation without randomness.
            x = index + (valid.repeat.to_numpy()-5.5)*.045
            points = ax.scatter(x, values, s=7, color=COLORS["alert"], alpha=.72, linewidths=0, zorder=3)
            assert np.allclose(points.get_offsets()[:, 1], values)
            if len(values):
                q1, median, q3 = np.quantile(values, [.25, .5, .75])
                ax.vlines(index, q1, q3, color="#222222", lw=1.1, zorder=4)
                ax.hlines(median, index-.18, index+.18, color="#222222", lw=.9, zorder=4)
            if len(values) < 10:
                ax.text(index, .97, f"n={len(values)}", transform=ax.get_xaxis_transform(), ha="center", va="top")
            records.append(dict(speed_mm_s=speed, model=model, contributing_runs=len(values),
                                repeats=valid.repeat.astype(int).tolist(), rmse_mm=values.tolist(),
                                sample_counts=valid.n_samples.astype(int).tolist()))
        ax.set(ylim=(0, high), xlim=(-.5, 4.5), xticks=range(5),
               xticklabels=[model.upper() for model in TARGETS])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, pos: f"{value:g}"))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.tick_params(length=2, pad=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#D9DDE1", lw=.4)
        ax.set_title(f"({letter}) {speed} mm/s", loc="left", pad=4)
    fig.text(.58, .982, "Points: runs; bars: median / IQR", ha="center", va="top")
    fig.supylabel("Projected RMSE (mm)", x=.018)
    fig.supxlabel("Target", y=.018)
    result = save_paper_figure(fig, "fig_displacement_tracking_1mmps.pdf")
    result.update(inputs=hashes([path]), targets=list(TARGETS), records=records,
                  statistic="per_target_per_run_conditional_full_projected_rmse",
                  truth_lower_bound_mm=5, truth_upper_bound_mm=None,
                  population="after_native_confirmation_and_nonzero_observable_rank")
    return result


@plt.rc_context({"font.family": "Times New Roman"})
def real_figure():
    """Population statistics only; no object trajectory or confirmation inferred."""
    path = PAPER / "real_analysis/response_timeline.csv"
    rows = pd.read_csv(path)
    summary_path = PAPER / "real_analysis/summary.json"
    summary = json.loads(summary_path.read_text())
    assert len(rows) == summary["snapshots"]
    for font in (PAPER / ".local_fonts").glob("Times*.TTF"):
        fontManager.addfont(str(font))
    findfont(FontProperties(family="Times New Roman"), fallback_to_default=False)
    start, end = summary["playback_window_s"]
    x = rows.time_s.to_numpy()
    assert start <= x[0] < x[-1] <= end
    gap = np.r_[False, np.diff(x) > 2*np.median(np.diff(x))]
    gap |= rows.epoch.ne(rows.epoch.shift()).to_numpy() & (np.arange(len(rows)) > 0)
    # A separator preserves both recorded endpoints without inventing a continuous curve.
    indices = np.flatnonzero(gap)
    plot_x = np.insert(x, indices, np.nan)
    series = lambda column: np.insert(rows[column].to_numpy(dtype=float), indices, np.nan)
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.55), sharex=True,
                             gridspec_kw={"height_ratios": [1.55, 1]})
    fig.subplots_adjust(left=.19, right=.97, bottom=.12, top=.87, hspace=.35)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2, pad=2)
        ax.grid(axis="y", color="#D9DDE1", lw=.4)
        ax.axvspan(start, x[0], color="#DCE1E5", alpha=.5, lw=0)
    def displacement(ax):
        ax.plot(plot_x, series("maximum_mm"), color="#66737D", lw=.9, label="Maximum")
        ax.plot(plot_x, series("median_mm"), color=COLORS["alert"], lw=1.2, label="Median")
        ax.fill_between(plot_x, series("p25_mm"), series("p75_mm"), color=COLORS["alert"], alpha=.18,
                        label="Interquartile range", lw=0)
    displacement(axes[0])
    upper = max(50, np.ceil(rows.maximum_mm.max()*1.08/50)*50)
    axes[0].set(xlim=(start, end), xticks=np.arange(start, end+.01, 20), ylim=(0, upper), ylabel="Estimated displacement\n(mm)")
    axes[0].set_title("(a) Selected estimates", loc="left", pad=5)
    axes[0].text((start+x[0])/2, upper*.35, "Initialization", rotation=90,
                 ha="center", va="center", color="#737D86")
    inset = axes[0].inset_axes([.17, .55, .32, .4])
    displacement(inset)
    early = [float(np.floor(x[0])), min(end, float(np.floor(x[0])+7))]
    early_max = rows.loc[rows.time_s.between(*early), "maximum_mm"].max()
    inset.set(xlim=early, ylim=(0, max(25, np.ceil(early_max*1.1/25)*25)),
              xticks=[early[0]+1, early[1]-1])
    inset.tick_params(length=1.5, pad=1)
    inset.set_title("Early interval", pad=1)
    axes[1].step(plot_x, series("responding_anchors"), where="post", color=COLORS["alert"], lw=1.1)
    axes[1].fill_between(plot_x, series("responding_anchors"), step="post", color=COLORS["alert"], alpha=.15, lw=0)
    count_top = max(20, np.ceil(rows.responding_anchors.max()*1.1/20)*20)
    axes[1].set(ylim=(0, count_top), yticks=(0, count_top/2, count_top), xlabel="Time (s)",
                ylabel="Selected anchors")
    axes[1].set_title("(b) Number of selected anchors", loc="left", pad=5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(.57, 1.01), columnspacing=.8, handlelength=1.3)
    result = save_paper_figure(fig, "fig_real_response.pdf", panels=2)
    result.update(inputs=hashes([path, summary_path]), statistic="display_selected_population_not_confirmation",
                  early_interval_s=early, maximum_mm=float(rows.maximum_mm.max()),
                  playback_window_s=[start,end], font_family="Times New Roman",
                  count_axis="linear", reference_anchors=sorted(rows.reference_anchors.unique().tolist()),
                  query_anchors=sorted(rows.anchors_with_estimates.unique().tolist()))
    return result


@plt.rc_context({"font.family": "Times New Roman", "mathtext.fontset": "custom",
                 "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic",
                 "mathtext.bf": "Times New Roman:bold", "mathtext.cal": "Times New Roman:italic",
                 "mathtext.sf": "Times New Roman", "mathtext.tt": "Times New Roman",
                 "mathtext.fallback": None})
def evacuation_figures():
    """Four causal map snapshots and the observation-to-trigger evidence chain."""
    for font in (PAPER / ".local_fonts").glob("Times*.TTF"):
        fontManager.addfont(str(font))
    font_path = Path(findfont(FontProperties(family="Times New Roman"), fallback_to_default=False))
    assert FontProperties(fname=font_path).get_name() == "Times New Roman"
    run = EVACUATION.parent.parent
    metrics = json.loads((EVACUATION / "metrics.json").read_text())
    response_path = PAPER / "response_audit.json"
    response = json.loads(response_path.read_text())["mobile_response"]
    mission = json.loads((run / "mission_summary.json").read_text())
    states = metrics["events"]
    origin, observe, monitor, trigger, completed = (states[k] for k in (
        "NAVIGATING_OUTBOUND", "STABILIZING_AT_TARGET", "MONITORING",
        "DEFORMATION_DETECTED", "COMPLETED"))
    stages = [(name, states[key], states[key]) for name, key in (
        ("Start", "NAVIGATING_OUTBOUND"), ("Observe", "STABILIZING_AT_TARGET"),
        ("Retreat trigger", "DEFORMATION_DETECTED"), ("Returned", "COMPLETED"))]
    inputs = [run / "mission_summary.json", EVACUATION / "metrics.json", response_path,
              run / "navigation/global_costmaps.csv", run / "trajectories/navigation_base_odometry.csv",
              EVACUATION / "trigger_evidence_chain.csv"]
    captured = []
    with tempfile.TemporaryDirectory() as temporary:
        with patch.object(plt, "close", lambda fig: captured.append(fig)):
            source_panels, used = evacuation.plot_keyframes(run, Path(temporary), metrics["events"],
                metrics["event_lineage"], evacuation.rows(run / "trajectories/navigation_base_odometry.csv"),
                mission["home_map_xy"], stages=stages)
        assert [r["data_cutoff_sim_time_s"] for r in source_panels] == [origin, observe, trigger, completed]
        assert all(r["return_point_count"] == 0 for r in source_panels[:3])
        assert source_panels[3]["return_point_count"] > 0
        fig = captured[-1]
        before = geometry_digest(fig)
        inputs += [run / name for name in used]
        fig.set_layout_engine(None)
        fig.set_size_inches(3.5, 3.10)
        for i, ax in enumerate(fig.axes[:4]):
            ax.set_position([.12+(i % 2)*.445, .635-(i//2)*.355, .405, .30])
            ax.set_title(f"({'abcd'[i]}) " + stages[i][0], loc="left", pad=4)
            cutoff = source_panels[i]["data_cutoff_sim_time_s"]
            stamps = [item for item in ax.texts if item.get_text().startswith("$t=")]
            assert len(stamps) == 1
            assert stamps[0].get_bbox_patch() is None, "Time annotation must not hide map contents"
            stamps[0].set_text(f"$t={cutoff-metrics['events']['NAVIGATING_OUTBOUND']:.2f}$ s")
            assert cutoff == stages[i][1]
            ax.images[0].set_cmap(ListedColormap(["#DCE1E5", "#FAFAF8", "#35404A"]))
            ax.images[1].set_cmap(LinearSegmentedColormap.from_list("navigation_cost", ["#FFF0C7", "#D49B79", WARNING]))
            for line in ax.lines:
                line.set_linewidth(1.05)
                line.set_path_effects([])
                line.set_color(RETURN if "Return" in line.get_label() else OUTBOUND)
            ax.set_xticks([5, 10, 15, 20])
            ax.set_yticks([0, 5, 10])
            ax.tick_params(length=2, pad=2)
            ax.xaxis.labelpad = ax.yaxis.labelpad = 1
        colorbar = fig.axes[4]
        colorbar.set_box_aspect(None)
        colorbar.set_position([.24, .105, .66, .018])
        colorbar.set_xticks([0, 50, 99])
        colorbar.set_xlabel("")
        colorbar.tick_params(length=2, pad=2)
        cost_label = colorbar.set_title("Navigation cost", pad=1.5)
        legend = fig.legends[0]
        handles = legend.legend_handles
        legend.remove()
        handles[0].set_color(OUTBOUND)
        handles[1].set_color(RETURN)
        legend = fig.legend(handles, ["Outbound", "Return", "Home", "Observe"],
                            loc="lower center", ncol=4, frameon=False,
                            bbox_to_anchor=(.52, .005), borderpad=0, borderaxespad=0,
                            handlelength=1.2, handletextpad=.35, columnspacing=.8)
        for item in fig.findobj(plt.Text):
            item.set_fontsize(7.5)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bar_box = colorbar.get_tightbbox(renderer)
        label_box = cost_label.get_window_extent(renderer)
        legend_box = legend.get_window_extent(renderer)
        for ax in fig.axes[2:4]:
            assert ax.get_tightbbox(renderer).y0 > max(bar_box.y1, label_box.y1) + 2, "X axis overlaps cost scale"
        assert min(bar_box.y0, label_box.y0) > legend_box.y1 + 2, "Cost scale overlaps legend"
        assert geometry_digest(fig) == before, "Redrawing changed recorded map or trajectory geometry"
        output = save_paper_figure(fig, "fig_evacuation_map.pdf", panels=4)
        output.update(recorded_geometry_sha256=before, map_value="published_navigation_cost",
                      font_family="Times New Roman", timestamp_background="none",
                      completion_relative_to_entry_s=completed-origin,
                      panel_cutoff_source_s=[row["data_cutoff_sim_time_s"] for row in source_panels],
                      keyframes=source_panels)

    ledger = {row["event"]: row for row in metrics["event_lineage"]}
    onset = ledger["truth_running"]["source_time_s"]
    confirmed = ledger["first_retained"]["source_time_s"]
    retained_until = ledger["persistence_source"]["source_time_s"]
    stop = response["actual_pose_last_change_source_s"]
    assert origin < observe < monitor < onset < confirmed < stop < retained_until < trigger < completed
    assert np.isclose(retained_until-confirmed, 10.17) and retained_until-confirmed >= 10
    evidence = pd.read_csv(EVACUATION / "trigger_evidence_chain.csv")
    clock_columns = ["source_time_s", "evidence_publish_time_s", "evidence_receive_time_s", "evolution_receive_time_s"]
    assert np.isfinite(evidence[clock_columns].to_numpy()).all()
    # A pre-trigger source stamp is insufficient if its result arrived after c.
    evidence = evidence[(evidence.source_time_s >= monitor)
                        & evidence[clock_columns].le(trigger).all(axis=1)].sort_values("source_time_s")
    assert len(evidence) > 2 and evidence.gdu_id.nunique() == 1
    assert evidence.gdu_id.iloc[0] == ledger["first_retained"]["gdu_id"]
    assert evidence.evidence_joined.all() and np.isfinite(evidence.displacement_rms_mm).all()
    assert evidence.source_time_s.diff().dropna().gt(0).all()
    retained = evidence[evidence.retained_for_avoidance]
    assert retained.source_time_s.iloc[0] == confirmed
    assert evidence[evidence.source_time_s >= confirmed].retained_for_avoidance.all()
    assert retained.source_time_s.iloc[-1] == retained_until

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.10))
    a, b = axes
    a.set_position([.18, .57, .79, .32])
    b.set_position([.23, .18, .74, .24])
    x, y = evidence.source_time_s.to_numpy()-origin, evidence.displacement_rms_mm.to_numpy()
    # Keep larger observation gaps rather than implying interpolated measurements.
    gaps = np.flatnonzero(np.diff(x) > .5)+1
    a.axvspan(onset-origin, stop-origin, color="#DCE1E5", alpha=.45, lw=0)
    line, = a.plot(np.insert(x, gaps, np.nan), np.insert(y, gaps, np.nan), color=COLORS["alert"], lw=1.2)
    a.plot(confirmed-origin, retained.displacement_rms_mm.iloc[0], marker="*", color="#222222", ms=6)
    a.annotate("Confirmed", xy=(confirmed-origin, retained.displacement_rms_mm.iloc[0]),
               xytext=(confirmed-origin-1.5, 9.5), ha="center", fontsize=7.5,
               arrowprops=dict(arrowstyle="-", color="#66737D", lw=.6))
    a.text((onset+stop)/2-origin, 14.5, "Object motion", ha="center", color="#59636B")
    a.set(xlim=(monitor-origin-.5, trigger-origin+1), ylim=(0, 16), yticks=(0, 5, 10, 15),
          xticks=(60, 65, 70, 75), ylabel="Estimated displacement\nRMS (mm)")
    a.set_title("(e) Displacement before trigger", loc="left", pad=5)

    b.broken_barh([(observe-origin, monitor-observe)], (1.8, .4), facecolors="#DCE1E5")
    b.broken_barh([(monitor-origin, trigger-monitor)], (.8, .4), facecolors="#B5CDD9")
    b.broken_barh([(confirmed-origin, retained_until-confirmed)], (-.2, .4), facecolors=COLORS["alert"])
    b.text((observe+monitor)/2-origin, 2, f"{monitor-observe:.2f} s", ha="center", va="center")
    b.text((confirmed+retained_until)/2-origin, 0, f"{retained_until-confirmed:.2f} s",
           ha="center", va="center", color="white")
    b.plot(confirmed-origin, 0, marker="*", color="#222222", ms=5)
    b.plot([retained_until-origin, trigger-origin], [0, 0], color=WARNING, ls=":", lw=.9)
    b.set(xlim=(observe-origin-1, trigger-origin+1), ylim=(-.6, 2.6),
          yticks=(2, 1, 0), yticklabels=("Preparation", "Monitoring", "Retention"), xticks=(30, 40, 50, 60, 70))
    b.set_title("(f) Observation (b) to trigger (c)", loc="left", pad=5)
    for ax in axes:
        ax.axvline(trigger-origin, color=WARNING, ls="--", lw=.9)
        ax.text(trigger-origin-.3, .96, "(c)", transform=ax.get_xaxis_transform(),
                color=WARNING, va="top", ha="right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2, pad=2)
    a.grid(axis="y", color="#D9DDE1", lw=.35)
    b.spines[["left", "right", "top"]].set_visible(False)
    b.tick_params(axis="y", length=0)
    fig.supxlabel("Time since start (s)", y=.035)
    assert np.allclose(line.get_ydata()[np.isfinite(line.get_ydata())], y)
    result = save_paper_figure(fig, "fig_evacuation_response.pdf", panels=2)
    result.update(statistic="observation_to_trigger", font_family="Times New Roman", time_origin_source_s=origin,
                  observation_source_s=observe, monitoring_source_s=monitor,
                  motion_onset_source_s=onset, confirmation_source_s=confirmed,
                  recorded_motion_stop_source_s=stop, retention_endpoint_source_s=retained_until,
                  trigger_source_s=trigger, recorded_stop_relative_to_trigger_s=stop-trigger,
                  preparation_duration_s=monitor-observe, retained_duration_s=retained_until-confirmed,
                  evidence_source_s=evidence.source_time_s.tolist(),
                  evidence_received_s=evidence[clock_columns[1:]].max(axis=1).tolist(),
                  evidence_displacement_mm=y.tolist())
    return {"fig_evacuation_map.pdf": output, "fig_evacuation_response.pdf": result}, hashes(inputs)


def main():
    plots.configure()
    plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
                         "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
    if "--real-only" in sys.argv:
        manifest_path = PAPER / "figures/layout_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        old = manifest["outputs"]["fig_real_response.pdf"]
        for path in old.get("inputs", {}):
            manifest["inputs"].pop(path, None)
        result = real_figure()
        manifest["outputs"]["fig_real_response.pdf"] = result
        manifest["inputs"].update(result["inputs"])
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
        print("PASS: updated only the real-data figure from the executed playback window.")
        return
    if "--evacuation-only" in sys.argv:
        outputs = json.loads((PAPER / "figures/layout_manifest.json").read_text())["outputs"]
    else:
        outputs = {"fig_baseline_comparison.pdf": baseline_figure(),
                   "fig_deformation_scale.pdf": confirmation_figure(),
                   "fig_displacement_tracking_1mmps.pdf": projected_figure(),
                   "fig_real_response.pdf": real_figure()}
    recorded, inputs = evacuation_figures()
    outputs.update(recorded)
    for result in outputs.values():
        inputs.update(result.get("inputs", {}))
    manifest = dict(targets=list(TARGETS), methods=list(METHODS), event_denominator=50,
                    inputs=inputs, outputs=outputs)
    assert len(outputs) == 6
    (PAPER / "figures/layout_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("PASS: six figure files indexed; Figure 7 shows four mission stages and observation-to-trigger evidence.")


if __name__ == "__main__":
    main()
