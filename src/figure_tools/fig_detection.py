"""Figure: what displacement is present when a change is confirmed, and whether
the estimate degrades as that displacement accumulates.

Left, the empirical distribution of the ground-truth displacement at the moment
of confirmation, over all 50 target events of every speed. Right, the projected
displacement error against the ground-truth displacement already accumulated at
1 mm/s, pooled over its 50 target runs. Both panels summarize the population of
recorded trials rather than a single run.
"""

import numpy as np
import pandas as pd

import paper_style as ps

ERROR_SPEED = 1
ERROR_EDGES = np.array([2, 5, 8, 12, 16, 20, 25, 30, 35, 40], dtype=float)


def load():
    events = pd.read_csv(ps.CONFIRMATION / "events.csv")
    events = events[(events.method == "alert") & events.model_name.isin(ps.TARGETS)]
    tracking = pd.read_csv(ps.PROJECTED / "tracking_bins.csv")
    tracking = tracking[tracking.model_name.isin(ps.TARGETS)]
    assert len(events) == len(ps.TARGETS) * len(ps.SPEEDS) * 10
    return events, tracking


def detection_panel(ax, events):
    record = []
    for color, speed in zip(ps.SPEED_COLORS, ps.SPEEDS):
        block = events[events.speed_mm_s == speed]
        found = block[block.detected.astype(str) == "True"]
        values = np.sort(found.truth_translation_at_confirmation_mm.astype(float).to_numpy())
        share = np.arange(1, values.size + 1) / len(block)
        ax.step(np.concatenate(([0], values, [18])),
                np.concatenate(([0], share, [share[-1] if share.size else 0])),
                where="post", color=color, lw=1.1, zorder=3, label=f"{speed}")
        record.append(dict(speed=speed, events=int(len(block)), confirmed=int(values.size),
                           median_mm=float(np.median(values)), max_mm=float(values.max())))
    ax.axvline(5, color="#000000", lw=.6, ls=(0, (4, 2)), zorder=2)
    ax.text(5.5, .015, "5 mm task scale", ha="left", va="bottom", **ps.ANNOTATE)
    ax.set(xlim=(0, 18), xticks=(0, 5, 10, 15), ylim=(0, 1.04), yticks=(0, .5, 1),
           xlabel="Displacement at\nconfirmation (mm)", ylabel="Confirmed fraction")
    ax.legend(loc="center right", ncol=1, handlelength=.9, handletextpad=.4,
              borderaxespad=.2, labelspacing=.22, title="mm/s", fontsize=6.4,
              title_fontsize=6.4)
    return record


def error_panel(ax, tracking):
    block = tracking[(tracking.speed_mm_s == ERROR_SPEED)
                     & np.isfinite(tracking.vector_rmse_mm)
                     & np.isfinite(tracking.truth_projected_norm_mean_mm)]
    centres, median, low, high, counts = [], [], [], [], []
    for lo, hi in zip(ERROR_EDGES[:-1], ERROR_EDGES[1:]):
        window = block[(block.truth_projected_norm_mean_mm >= lo)
                       & (block.truth_projected_norm_mean_mm < hi)]
        if len(window) < 20:
            continue
        centres.append((lo + hi) / 2)
        median.append(float(window.vector_rmse_mm.median()))
        low.append(float(window.vector_rmse_mm.quantile(.25)))
        high.append(float(window.vector_rmse_mm.quantile(.75)))
        counts.append(int(len(window)))
    centres, median = np.array(centres), np.array(median)
    ax.fill_between(centres, low, high, color=ps.OURS, alpha=.22, lw=0, zorder=2,
                    label="Interquartile range")
    ax.plot(centres, median, color=ps.OURS, lw=1.1, marker="o", ms=2.6, zorder=3,
            label="Median")
    ax.set(xlim=(0, 41), xticks=(0, 10, 20, 30, 40), ylim=(0, 3.4), yticks=(0, 1, 2, 3),
           xlabel="Accumulated ground-truth\ndisplacement (mm)",
           ylabel="Displacement error (mm)")
    ax.legend(loc="upper right", handlelength=1.1, borderaxespad=.2, fontsize=6.4,
              labelspacing=.22)
    return [dict(centre_mm=c, median_mm=m, q25_mm=l, q75_mm=h, bins=n)
            for c, m, l, h, n in zip(centres, median, low, high, counts)]


def build():
    events, tracking = load()
    fig, axes = ps.plt.subplots(1, 2, figsize=(ps.COLUMN_WIDTH, 1.85))
    fig.subplots_adjust(left=.128, right=.985, bottom=.30, top=.965, wspace=.46)
    for ax in axes:
        ps.trim_axes(ax)
    detection = detection_panel(axes[0], events)
    error = error_panel(axes[1], tracking)
    return fig, dict(detection=detection, error=error, error_speed_mm_s=ERROR_SPEED)


def main():
    ps.use_style()
    fig, record = build()
    ps.save(fig, "fig_detection", dict(
        summary=record, targets=list(ps.TARGETS), speeds=list(ps.SPEEDS),
        inputs=ps.digest([ps.CONFIRMATION / "events.csv",
                          ps.PROJECTED / "tracking_bins.csv"])))
    print(record["detection"])


if __name__ == "__main__":
    main()
