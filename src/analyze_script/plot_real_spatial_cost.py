#!/usr/bin/env python3
"""Offline XY response-cost preview from recorded real LiDAR estimates, not navigation."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties, findfont, fontManager
import numpy as np
import yaml

from analyze_v1_real import (anchor_counts, anchor_key, csv_write, frame_summary,
                             records, recording_window, reference_index, sample_response, stamp)

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC / "deform_avoidance_map/src"))
from deform_avoidance_map.raster import GridSpec, rasterize_sources


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keyframes(times):
    times = np.asarray(times, dtype=float)
    if len(times) < 4 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Need four or more strictly ordered snapshot times")
    targets = np.linspace(times[0], times[-1], 4)
    indices = np.searchsorted(times, targets, side="right") - 1
    if len(set(indices)) != 4:
        raise ValueError("Too few distinct snapshots near the four time knots")
    return indices, targets


def project_frame(record, support, config):
    """Apply recorded display gates; union reference and displaced support in XY."""
    sources, selected = [], []
    for sample in record["physical_deformation"]:
        key = anchor_key(sample)
        if key not in support:
            raise ValueError("Query is not part of this frozen reference")
        amplitude, eligible = sample_response(sample, config)
        if not eligible:
            continue
        anchor = support[key]
        position = np.array(key[2:])
        displacement = np.array([sample["displacement_mean"][k] for k in ("x", "y", "z")])
        radius = float(anchor["support_radius_m"])
        if not np.isfinite(radius) or radius < 0:
            raise ValueError("Invalid recorded support radius")
        # A vertical displacement still warns its horizontal projection.
        sources.extend((float(p[0]), float(p[1]), radius) for p in (position, position + displacement))
        selected.append(dict(
            surfel_id="{high:016x}{low:016x}".format(**anchor["surfel_id"]),
            gdu_id="{high:016x}{low:016x}".format(**sample["gdu_id"]),
            x_m=position[0], y_m=position[1], z_m=position[2],
            dx_m=displacement[0], dy_m=displacement[1], dz_m=displacement[2],
            displacement_mm=amplitude, support_radius_m=radius,
            active_probability_lower=sample["active_probability_lower"],
            validity=sample["validity"], detection_admission=sample["detection_admission"]))
    return sources, selected


def draw(output, spec, reference_xyz, costs, summaries, indices):
    for path in (SRC / "IEEE_ICRA2027/.local_fonts").glob("Times*.TTF"):
        fontManager.addfont(str(path))
    font = findfont(FontProperties(family="Times New Roman"), fallback_to_default=False)
    if FontProperties(fname=font).get_name() != "Times New Roman":
        raise ValueError("Times New Roman is required; no font substitution")
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 10, "axes.linewidth": .6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.75), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.985, bottom=.205, top=.96, wspace=.08, hspace=.12)
    extent = (spec.origin_x, spec.origin_x + spec.width * spec.resolution,
              spec.origin_y, spec.origin_y + spec.height * spec.resolution)
    cmap = LinearSegmentedColormap.from_list("response_cost", ["#FFF0C7", "#D49B79", "#AB5360"])
    for panel, (ax, index) in enumerate(zip(axes.flat, indices)):
        ax.set_facecolor("#F7F8F9")
        im = ax.imshow(np.ma.masked_equal(costs[index], 0), extent=extent, origin="lower",
                       interpolation="nearest", cmap=cmap, vmin=0, vmax=99, zorder=1)
        ax.scatter(reference_xyz[:, 0], reference_xyz[:, 1], s=.6, c="#727C84",
                   alpha=.40, linewidths=0, rasterized=True, zorder=2)
        ax.set(xlim=extent[:2], ylim=extent[2:], aspect="equal")
        ax.set_xticks(np.arange(np.ceil(extent[0]/5)*5, extent[1], 5))
        ax.set_yticks(np.arange(np.ceil(extent[2]/5)*5, extent[3], 5))
        ax.set_title(f"({'abcd'[panel]})", loc="left", pad=4)
        label = ax.text(.97, .96, f"t = {summaries[index]['time_s']:.2f} s", transform=ax.transAxes,
                        ha="right", va="top", color="#20262D")
        assert label.get_bbox_patch() is None
        ax.tick_params(length=2, pad=2, bottom=panel >= 2, labelbottom=panel >= 2,
                       left=panel % 2 == 0, labelleft=panel % 2 == 0)
        if panel >= 2:
            ax.set_xlabel("x (m)", labelpad=2)
        if panel % 2 == 0:
            ax.set_ylabel("y (m)", labelpad=2)
    color_ax = fig.add_axes([.29, .081, .50, .018])
    bar = fig.colorbar(im, cax=color_ax, orientation="horizontal", ticks=[0, 50, 99])
    bar.ax.set_title("Deformation-response cost", fontsize=9, pad=3)
    bar.ax.tick_params(length=2, pad=2)
    reference = plt.Line2D([], [], marker=".", color="#727C84", linestyle="none", markersize=4)
    fig.legend([reference], ["Projected reference geometry"], frameon=False,
               loc="lower center", bbox_to_anchor=(.53, .006), borderaxespad=0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert min(ax.get_tightbbox(renderer).y0 for ax in axes[1]) > color_ax.get_tightbbox(renderer).y1
    for text in fig.findobj(plt.Text):
        if text.get_visible() and text.get_text():
            assert text.get_fontproperties().get_name() == "Times New Roman"
            box = text.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width+1 and box.y1 <= fig.bbox.height+1, (text.get_text(), box.bounds)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(output / ("fig_real_spatial_cost." + suffix), dpi=450)
    plt.close(fig)


def analyze(run, output, resolution=.1, core_radius=.75, halo_radius=1.25):
    if not all(np.isfinite([resolution, core_radius, halo_radius])) or not (resolution > 0 and 0 <= core_radius <= halo_radius):
        raise ValueError("Invalid raster resolution or inflation radii")
    manifest = json.loads((run / "metadata/manifest.json").read_text())
    if not manifest["complete"] or not manifest["closed_cleanly"] or manifest["error"] or manifest["queue_overflow"]:
        raise ValueError("Recording did not close completely")
    parameters = json.loads((run / "metadata/parameters.json").read_text())
    config = {k: float(parameters["/deform_monitor_v1_rviz"][k]) for k in (
        "minimum_arrow_active_probability", "displacement_display_scale", "minimum_display_arrow_m")}
    if (not np.isfinite(list(config.values())).all() or not 0 <= config["minimum_arrow_active_probability"] <= 1
            or config["displacement_display_scale"] <= 0 or config["minimum_display_arrow_m"] < 0):
        raise ValueError("Invalid recorded display gates")
    bag = yaml.safe_load((run / "metadata/bag_info.yaml").read_text())
    t0 = float(bag["start"])
    references = list(records(run, "references"))
    if len(references) != 1:
        raise ValueError("This preview requires one frozen reference epoch")
    reference = references[0]
    identity = lambda c: (c["session_id"], c["reference_epoch"], c["header"]["frame_id"])
    reference_id = identity(reference["common"])
    lookup = reference_index(reference["reference_support"])
    support = {anchor_key(p): p for p in reference["reference_support"]}
    reference_xyz = np.array([k[2:] for k in support])
    summaries, sources, query_xy, exports = [], [], [], []
    for index, record in enumerate(records(run, "snapshots")):
        common = record["common"]
        if identity(common) != reference_id or stamp(common["header"]["stamp"]) < stamp(reference["common"]["header"]["stamp"]):
            raise ValueError("Snapshot does not match the available reference")
        time = stamp(common["header"]["stamp"]) - t0
        current, selected = project_frame(record, support, config)
        samples = record["physical_deformation"]
        summary = dict(snapshot_index=index, time_s=time, publish_time_s=stamp(common["publish_stamp"])-t0,
                       receive_time_s=stamp(record["receive_ros"])-t0,
                       snapshot_id="{high:016x}{low:016x}".format(**common["snapshot_id"]),
                       accepted_frame_index=common["accepted_frame_index"],
                       **frame_summary(samples, config), **anchor_counts(samples, lookup, config))
        summary["formal_detected_queries"] = sum(e["detection_allowed"] and e["detection_admission"] == 0
                                                  for e in record["detection_evidence"])
        assert len(selected) == summary["responding_anchors"]
        summaries.append(summary); sources.append(current)
        query_xy.append(np.array([anchor_key(p)[2:4] for p in samples]).reshape(-1, 2))
        exports.extend(dict(snapshot_index=index, time_s=time, **row) for row in selected)
    assert len(summaries) == manifest["written"]["snapshots"]
    times = np.array([row["time_s"] for row in summaries])
    window = recording_window(run, bag, times)
    indices, targets = keyframes(times)
    all_sources = np.array([p for group in sources for p in group]).reshape(-1, 3)
    bounds = np.vstack([reference_xyz[:, :2], all_sources[:, :2]])
    padding = halo_radius + (float(all_sources[:, 2].max()) if len(all_sources) else 0) + resolution
    low = np.floor((bounds.min(axis=0) - padding) / resolution) * resolution
    high = np.ceil((bounds.max(axis=0) + padding) / resolution) * resolution
    spec = GridSpec.from_metric_bounds(resolution, *(high-low), *low)
    if spec.width * spec.height * len(summaries) > 20_000_000:
        raise ValueError("Preview grid exceeds 20 million frame-cells; increase resolution")
    costs, query_cells = [], []
    for row, group, xy in zip(summaries, sources, query_xy):
        costs.append(np.array(rasterize_sources(spec, group, core_radius, halo_radius), dtype=np.uint8).reshape(spec.height, spec.width))
        queried = np.zeros((spec.height, spec.width), dtype=bool)
        cells = np.floor((xy-low) / resolution).astype(int)
        queried[cells[:, 1], cells[:, 0]] = True
        query_cells.append(queried)
        row.update(core_area_m2=float(np.count_nonzero(costs[-1] == 99)*resolution**2),
                   nonzero_cost_area_m2=float(np.count_nonzero(costs[-1])*resolution**2))
    costs = np.stack(costs)
    output.mkdir(parents=True, exist_ok=True)
    csv_write(output / "snapshot_summary.csv", summaries)
    csv_write(output / "keyframes.csv", [dict(panel="abcd"[k], target_time_s=targets[k], **summaries[i]) for k, i in enumerate(indices)])
    if exports:
        import csv
        with gzip.open(output / "response_sources.csv.gz", "wt", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=exports[0].keys()); writer.writeheader(); writer.writerows(exports)
    np.savez_compressed(output / "response_cost_maps.npz", cost=costs, time_s=times,
                        queried_cells=np.stack(query_cells), reference_xyz_m=reference_xyz,
                        resolution_m=resolution, origin_xy_m=low, frame_id=reference_id[2], keyframe_indices=indices)
    draw(output, spec, reference_xyz, costs, summaries, indices)
    paths = [run / "metadata" / p for p in ("manifest.json", "parameters.json", "bag_info.yaml")]
    if (run / "metadata/execution.json").exists():
        paths.append(run / "metadata/execution.json")
    paths += [run / "measurements" / p for p in ("references.jsonl.gz", "snapshots.jsonl.gz")]
    paths += [Path(__file__), SRC / "analyze_script/analyze_v1_real.py", SRC / "deform_avoidance_map/src/deform_avoidance_map/raster.py"]
    metadata = dict(run=str(run), playback_window_s=window,
        time_origin="bag start; snapshot measurement timestamps", time_zero_s=t0,
        reference_identity=reference_id, resolution_m=resolution, core_radius_m=core_radius, halo_radius_m=halo_radius,
        cost_semantics="Offline current-response core/halo overlay; not recorded navigation cost or failure probability",
        support_policy="Maximum union of reference/current measured support; all heights projected to XY; uniform radii",
        temporal_policy="Independent recorded snapshots; no carry-forward, clearing claim, or prediction",
        zero_cost_semantics="No added response cost, not evidence of safe/free/unchanged space",
        selection="Latest snapshot at or before each of four equally spaced measurement-time knots",
        display_gates=config, source_sha256={str(p): sha256(p) for p in paths},
        output_sha256={p.name: sha256(p) for p in output.iterdir() if p.name.startswith(("fig_real_", "response_", "snapshot_", "keyframes."))})
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2)+"\n")
    timing = "、".join(f"{times[i]:.2f} s" for i in indices)
    table = "| 图 | 时间 s | 响应锚点数 | 位移中位数 mm | 非零代价面积 m² |\n|---|---:|---:|---:|---:|\n"
    for panel, i in zip("abcd", indices):
        row = summaries[i]
        median = f"{row['median_mm']:.2f}" if np.isfinite(row["median_mm"]) else "—"
        table += f"| {panel} | {row['time_s']:.2f} | {row['responding_anchors']} | {median} | {row['nonzero_cost_area_m2']:.2f} |\n"
    (output / "README.md").write_text(
        "# 真实实验二维形变响应代价：独立预览\n\n"
        f"四幅图使用同一坐标、比例尺和色标，时间为 {timing}，相对于 bag 开始；"
        "按观测时间等间隔取样，不按响应强度挑图。未插入或修改论文。\n\n"
        "灰点是冻结参考几何的 XY 投影，并非障碍物栅格。彩色区域由本次记录的 RViz 响应门槛选出锚点后离线生成；"
        f"网格 {resolution:g} m，核心缓冲 {core_radius:g} m，外围缓冲 {halo_radius:g} m，另加各锚点记录的支撑半径。"
        "核心代价 99，外围按距离线性降至零；重叠区域取最大值，不累加概率。参考位置及估计后位置均纳入，"
        "全部高度投影到 XY，无物体语义或手动选区；此预览不使用高处目标特例半径。\n\n"
        "这是当前响应派生的软代价层，不是本次实际发布的导航地图，也不是坍塌概率或未来预测。"
        "色值表示空间缓冲代价，不是位移幅值；区域面积增减不能直接解释成形变幅值增减。"
        "每帧独立使用当时的响应，没有虚构在线保留状态。响应消失不等于结构恢复，零附加代价不等于安全或可通行。"
        f"本次正式 DETECTED 条目共 {sum(r['formal_detected_queries'] for r in summaries)}；显示响应不当作正式确认。\n\n"
        "时间标签使用测量时间；snapshot_summary.csv 同时保留发布和接收时间，不将该图称为这些测量时刻已经发布的在线地图。"
        "未记录完整输入点云或 SLAM 占据地图，因此不凭空补墙、道路或自由空间。\n\n"
        "PDF/PNG 为展示图，SVG 保留可编辑文字；response_cost_maps.npz 保存全部快照代价、独立查询覆盖掩码、参考 XYZ 和坐标元数据；"
        "snapshot_summary.csv 保存逐帧响应数量、位移分位数与派生缓冲面积；keyframes.csv 保存四时刻及选择依据；"
        "response_sources.csv.gz 保存所有参与投影的原始位移、位置、ID 与有效性，manifest.json 保存输入及输出哈希。\n",
        encoding="utf-8")
    with (output / "README.md").open("a", encoding="utf-8") as stream:
        stream.write("\n## 四时刻统计\n\n" + table + "\n位移统计针对各时刻选中的锚点集合，不是同一个物体的运动曲线。\n")
    print(f"Saved {len(summaries)} maps; panels at {timing}\n{output / 'fig_real_spatial_cost.pdf'}")


def self_check():
    indices, knots = keyframes([1., 2., 4., 6., 8., 10.])
    assert indices.tolist() == [0, 2, 3, 5]
    assert np.all(np.array([1., 2., 4., 6., 8., 10.])[indices] <= knots)
    config = dict(minimum_arrow_active_probability=.95, displacement_display_scale=50., minimum_display_arrow_m=.05)
    sample = dict(gdu_id=dict(high=1, low=2), position=dict(x=0., y=0., z=3.),
                  displacement_mean=dict(x=0., y=0., z=.002), active_probability_lower=.99,
                  validity=1, detection_admission=4)
    reference = dict(sample, surfel_id=dict(high=3, low=4), support_radius_m=.05)
    support = {anchor_key(sample): reference}
    sources, selected = project_frame(dict(physical_deformation=[sample]), support, config)
    assert len(sources) == 2 and sources[0] == sources[1] and selected[0]["displacement_mm"] == 2
    spec = GridSpec(.1, 41, 41, -2.05, -2.05)
    overlay = np.array(rasterize_sources(spec, sources, .75, 1.25))
    assert overlay[spec.index(20, 20)] == 99 and overlay.max() == 99
    assert np.array_equal(overlay, rasterize_sources(spec, sources * 2, .75, 1.25))
    sample["active_probability_lower"] = .5
    empty, selected = project_frame(dict(physical_deformation=[sample]), support, config)
    assert not empty and not selected and not any(rasterize_sources(spec, empty, .75, 1.25))
    print("PASS: temporal selection, vertical projection, source gates, max fusion and empty responses")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        self_check()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("run", type=Path)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--resolution", type=float, default=.1)
        parser.add_argument("--core-radius", type=float, default=.75)
        parser.add_argument("--halo-radius", type=float, default=1.25)
        args = parser.parse_args()
        analyze(args.run.resolve(), args.output.resolve(), args.resolution, args.core_radius, args.halo_radius)
