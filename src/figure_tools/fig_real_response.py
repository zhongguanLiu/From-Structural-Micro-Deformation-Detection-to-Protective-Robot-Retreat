"""Figure: where the real sequence raises a warning, and how that region grows.

The recorded run stores, for every update, the ground-plane warning cost that
the retained support would raise from the responses of that update. The top row
keeps its left third free for an external image of the scene and then shows two
of those updates, the first responding one and a later one in which a second
surface has joined. The lower row follows the warned footprint and the estimated
magnitude over the whole sequence. Panel letters are left to the manuscript.
"""

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

import paper_style as ps

SPATIAL = (ps.OUTPUT / "real_60s/replay_20260915_014053/run_20260915_014056_04ff7f28"
           / "spatial_cost")
COST_MAP = LinearSegmentedColormap.from_list("navigation_cost",
                                             ["#FFF0C7", "#E0A878", "#D55E00"])
REFERENCE_MAP = ListedColormap(["#BFC4C9"])
VIEW = (-4.6, 6.4, -5.0, 4.4)   # metres, the part of the reference that carries responses
SECOND_TIME_S = 20.6            # a later update in which a second surface has joined


def load():
    grids = np.load(SPATIAL / "response_cost_maps.npz")
    summary = pd.read_csv(SPATIAL / "snapshot_summary.csv")
    assert len(summary) == grids["cost"].shape[0] == 87
    return grids, summary


def map_panel(ax, grids, reference, index):
    cost = grids["cost"][index].astype(float)
    resolution = float(grids["resolution_m"])
    origin = grids["origin_xy_m"]
    extent = (origin[0], origin[0] + cost.shape[1] * resolution,
              origin[1], origin[1] + cost.shape[0] * resolution)
    edges_x = np.arange(extent[0], extent[1] + resolution, resolution)
    edges_y = np.arange(extent[2], extent[3] + resolution, resolution)
    occupied, _, _ = np.histogram2d(reference[:, 0], reference[:, 1], bins=(edges_x, edges_y))
    ax.imshow(np.where(occupied.T > 0, 1.0, np.nan), origin="lower", extent=extent,
              cmap=REFERENCE_MAP, zorder=1, interpolation="nearest")
    ax.imshow(np.where(cost > 0, cost, np.nan), origin="lower", extent=extent,
              cmap=COST_MAP, vmin=1, vmax=99, zorder=2, interpolation="nearest")
    ax.set(xlim=VIEW[:2], ylim=VIEW[2:], xticks=(-4, 0, 4), yticks=(-4, 0, 4))
    ax.set_aspect("equal")
    ax.tick_params(length=2, pad=1.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("x (m)", labelpad=0.5)
    ax.text(VIEW[0] + .4, VIEW[3] - .3, f"$t$ = {float(grids['time_s'][index]):.1f} s",
            ha="left", va="top", **ps.ANNOTATE)
    return float(grids["time_s"][index])


def build():
    grids, summary = load()
    reference = grids["reference_xyz_m"]
    time = summary.time_s.to_numpy()
    area = summary.nonzero_cost_area_m2.to_numpy()
    median = summary.median_mm.to_numpy(dtype=float)
    responding = summary.responding_anchors.to_numpy()
    first = int(np.flatnonzero(responding > 0)[0])
    second = int(np.argmin(np.abs(time - SECOND_TIME_S)))

    fig = ps.plt.figure(figsize=(ps.COLUMN_WIDTH, 2.85))
    # The left third of the top row is left empty for an external scene image.
    middle = fig.add_axes([.375, .565, .27, .385])
    right = fig.add_axes([.665, .565, .27, .385])
    lower = fig.add_axes([.115, .145, .835, .285])

    t_first = map_panel(middle, grids, reference, first)
    t_second = map_panel(right, grids, reference, second)
    # The vertical scale is annotated on the far right so that the left third of
    # the row stays completely free for an external image.
    middle.tick_params(labelleft=False)
    right.tick_params(left=False, labelleft=False, right=True, labelright=True)
    right.yaxis.set_label_position("right")
    right.spines["right"].set_visible(True)
    right.spines["left"].set_visible(False)
    right.set_ylabel("y (m)", labelpad=1)

    ps.trim_axes(lower)
    lower.axvspan(0, float(time[0]), color=ps.PHASE_FILL[1], lw=0, zorder=0)
    lower.fill_between(time, 0, area, color=ps.OURS, alpha=.28, lw=0, zorder=2)
    lower.plot(time, area, color=ps.OURS, lw=1.0, zorder=3, label="Warned area")
    for stamp in (t_first, t_second):
        lower.axvline(stamp, color="#000000", lw=.6, ls=(0, (3, 2)), zorder=4)
    lower.set(xlim=(0, 60), xticks=(0, 10, 20, 30, 40, 50, 60), ylim=(0, 34),
              yticks=(0, 10, 20, 30), xlabel="Time (s)", ylabel="Warned area (m$^2$)")
    twin = lower.twinx()
    twin.plot(time, median, color="#4D4D4D", lw=.8, ls=(0, (3, 1.6)), zorder=3,
              label="Median magnitude")
    twin.set(ylim=(0, 90), yticks=(0, 40, 80), ylabel="Median (mm)")
    twin.spines["top"].set_visible(False)
    handles = lower.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    lower.legend(handles, ["Warned area", "Median magnitude"], loc="upper left",
                 ncol=2, columnspacing=.8, handlelength=1.2, borderaxespad=.2,
                 bbox_to_anchor=(-.015, 1.16), fontsize=6.4)

    record = dict(snapshots=int(len(summary)), responding=int((responding > 0).sum()),
                  first_response_s=t_first, first_area_m2=float(area[first]),
                  first_anchors=int(responding[first]), second_s=t_second,
                  second_area_m2=float(area[second]), second_anchors=int(responding[second]),
                  peak_area_m2=float(area.max()), peak_area_s=float(time[int(np.argmax(area))]),
                  resolution_m=float(grids["resolution_m"]), reference_points=int(len(reference)))
    return fig, record


def main():
    ps.use_style()
    fig, record = build()
    ps.save(fig, "fig_real_warning", dict(
        summary=record, reserved_panel="left third of the top row",
        inputs=ps.digest([SPATIAL / "response_cost_maps.npz",
                          SPATIAL / "snapshot_summary.csv"])), tight=False)
    print(record)


if __name__ == "__main__":
    main()
