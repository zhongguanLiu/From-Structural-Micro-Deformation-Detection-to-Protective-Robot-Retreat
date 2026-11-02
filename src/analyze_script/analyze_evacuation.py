#!/usr/bin/env python3
"""One recorded evacuation; immutable inputs, no ROS, no simulated trajectories."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.patheffects as path_effects
from evacuation_timeline import (event_chain, plot_chain, path_following,
                                 OUTBOUND_COLOR, RETURN_COLOR, DEFORMATION_COLOR)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       "deform_monitor_bringup/scripts"))
from planning_core import GridGeometry, astar, inflate_obstacles, fuse_planning_layers
from closed_loop_core import rectangular_footprint_is_traversable


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save_csv(path, records):
    if not records:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def paper_metrics(result):
    """A compact numerical index, retaining missing events and explicit clocks."""
    records = []
    def add(name, value, unit, definition, source):
        if value is not None and not math.isfinite(value):
            raise ValueError('Non-finite paper metric: '+name)
        records.append(dict(metric=name, value=value, unit=unit,
                            definition=definition, source=source))
    origin = result['events']['NAVIGATING_OUTBOUND']
    for row in result['event_lineage']:
        stamp = row['source_time_s']
        add(row['event']+'_relative_to_outbound', None if stamp is None else stamp-origin,
            'simulation s', 'Source timestamp minus NAVIGATING_OUTBOUND; blank means unobserved',
            row['source_file']+'; event_lineage.csv:'+row['event'])
    for row in result['response_intervals']:
        add(row['start_event']+'_to_'+row['end_event'], row['duration_sim_s'],
            'simulation s', row['interpretation']+'; '+row['clock'], 'response_intervals.csv')
    for key, definition in (
            ('length_m', 'Executed return polyline length; SLAM-derived navigation pose'),
            ('exposure_m', 'Distance-weighted mean endpoint footprint cost / 99 on frozen deformation layer'),
            ('inside_cost_distance_m', 'Arc-length integral of endpoint-averaged nonzero footprint-cost indicator')):
        add('executed_'+key, result['executed'][key], 'm', definition,
            'trajectories/navigation_base_odometry.csv; navigation/maps/decision_map.npz')
    for key in ('slam_home_distance_m', 'gazebo_home_distance_m'):
        add(key, result['arrival_check'][key], 'm', 'Distance to HOME at mission completion',
            'mission_summary.json; trajectories/gazebo_robot_truth.csv')
    add('return_nonempty_path_publications', result['return_published_plan_count'], 'count',
        'Nonempty Path publications, NOT independently measured planner invocations',
        'navigation/global_plans.csv; path_publications.csv')
    return records


def available_at(records, stamp):
    """Only messages both generated and received by the panel's event time."""
    return [r for r in records if float(r['message_time_s']) <= stamp
            and float(r.get('receive_time_s', r['ros_time_s'])) <= stamp]


def phase_trajectories(odometry, stamp, states):
    available = available_at(odometry, stamp)
    outbound = sorted([r for r in available if states['NAVIGATING_OUTBOUND'] <= float(r['message_time_s'])
                       <= states['STABILIZING_AT_TARGET']], key=lambda r: float(r['message_time_s']))
    returning = sorted([r for r in available if float(r['message_time_s']) >= states['RETURNING_HOME']],
                       key=lambda r: float(r['message_time_s']))
    return outbound, returning


def plot_keyframes(run, out, stamps, ledger, odometry, home, *, stages=None):
    events = {r['event']: r for r in ledger}
    if 'truth_running' not in events:
        raise ValueError('No recorded object-motion onset for the process figure')
    origin = stamps['NAVIGATING_OUTBOUND']
    # Panel d is a retrospective trajectory overview labelled by its trigger,
    # not an instantaneous snapshot claiming that the return already happened.
    motion_start = events['truth_running']['source_time_s']
    if stages is None:
        stages = [('Start', origin, origin),
                  ('Observation pose', stamps['STABILIZING_AT_TARGET'], stamps['STABILIZING_AT_TARGET']),
                  ('Object motion begins', motion_start, motion_start),
                  ('Deformation-triggered return', stamps['DEFORMATION_DETECTED'], stamps['COMPLETED'])]
    if (len(stages) != 4 or any(not math.isfinite(t) or not math.isfinite(cutoff)
                               or t > cutoff for _, t, cutoff in stages)
            or any(a[2] > b[2] for a, b in zip(stages, stages[1:]))):
        raise ValueError('Four ordered, finite keyframe stages are required')
    map_records = rows(run / 'navigation/global_costmaps.csv')
    frame = None
    panels, manifest, used_files, pose_exports = [], [], [], []
    for index, (name, event_stamp, stamp) in enumerate(stages):
        available_maps, available_poses = available_at(map_records, stamp), available_at(odometry, stamp)
        if not available_maps or not available_poses:
            raise ValueError('No past map/pose for '+name)
        selected_map = max(available_maps, key=lambda r: float(r['ros_time_s']))
        path = (run / selected_map['relative_path']).resolve()
        if run not in path.parents:
            raise ValueError('Keyframe map path escapes run directory')
        with np.load(path, allow_pickle=False) as data:
            grid, metadata = data['data'].copy(), json.loads(str(data['metadata']))
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()+grid.tobytes()).hexdigest()
        if digest != selected_map['map_id']:
            raise ValueError('Keyframe map hash mismatch')
        if not np.allclose(metadata['orientation'], [0, 0, 0, 1], atol=1.e-6):
            raise ValueError('Rotated keyframe grids are unsupported')
        if frame is not None and frame != metadata['frame_id']:
            raise ValueError('Keyframe coordinate frames differ')
        frame = metadata['frame_id']
        pose_row = max(available_poses, key=lambda r: float(r['message_time_s']))
        if pose_row['frame_id'] != frame:
            raise ValueError('Keyframe pose frame differs from map')
        pose = np.array([float(pose_row[k]) for k in ('x_m', 'y_m', 'yaw_rad')])
        outbound, returning = phase_trajectories(odometry, stamp, stamps)
        if any(r['frame_id'] != frame for r in outbound+returning):
            raise ValueError('Executed trajectory frame mismatch')
        xy = []
        for phase, group in [('outbound', outbound), ('return', returning)]:
            xy.append(np.array([[float(r['x_m']), float(r['y_m'])] for r in group]).reshape(-1, 2))
            for r in group:
                pose_exports.append(dict(panel=index+1, plotted_phase=phase, **r))
        record = dict(panel=index+1, event=name, event_sim_time_s=event_stamp,
                      data_cutoff_sim_time_s=stamp,
                      time_origin_sim_s=origin, relative_to_outbound_start_s=event_stamp-origin,
                      relative_to_monitoring_s=event_stamp-stamps['MONITORING'],
                      map_file=selected_map['relative_path'], map_id=digest,
                      map_source_time_s=float(selected_map['message_time_s']),
                      map_receive_time_s=float(selected_map['ros_time_s']),
                      map_age_s=stamp-float(selected_map['message_time_s']),
                      pose_source_time_s=float(pose_row['message_time_s']),
                      pose_receive_time_s=float(pose_row['receive_time_s']),
                      pose_age_s=stamp-float(pose_row['message_time_s']),
                      robot_x_m=pose[0], robot_y_m=pose[1], robot_yaw_rad=pose[2],
                      decision_trigger_relative_s=stamps['DEFORMATION_DETECTED']-origin,
                      outbound_point_count=len(outbound), return_point_count=len(returning))
        manifest.append(record)
        panels.append((grid, metadata, pose, xy[0], xy[1]))
        used_files.append(selected_map['relative_path'])
    save_csv(out / 'keyframe_sources.csv', manifest)
    save_csv(out / 'keyframe_executed_poses.csv', pose_exports)
    # This obsolete derived export described the former 3-panel planned-path figure.
    obsolete = out / 'keyframe_published_paths.csv'
    if obsolete.is_file():
        obsolete.unlink()
    # Framing can be common; each panel still draws only its own past data.
    context = np.vstack([np.asarray(home).reshape(1, 2), panels[1][2][:2].reshape(1, 2)]
                        + [xy for panel in panels for xy in panel[3:] if len(xy)])
    low, high = context.min(axis=0)-2.5, context.max(axis=0)+2.5
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.35), sharex=True, sharey=True, layout='constrained')
    for i, (ax, (grid, metadata, pose, outbound, returning), r) in enumerate(zip(axes.flat, panels, manifest)):
        res = metadata['resolution']
        x, y = metadata['origin']
        extent = [x, x+grid.shape[1]*res, y, y+grid.shape[0]*res]
        categories = np.where(grid < 0, 0, np.where(grid >= 100, 2, 1))
        ax.set_facecolor('#d9dde1')  # Unmapped extent must not appear free.
        ax.imshow(categories, origin='lower', extent=extent, interpolation='nearest',
                  cmap=ListedColormap(['#d9dde1', '#fafafa', '#303842']), vmin=0, vmax=2)
        im = ax.imshow(np.ma.masked_where((grid <= 0) | (grid >= 100), grid), origin='lower',
                       extent=extent, interpolation='nearest',
                       cmap=LinearSegmentedColormap.from_list('deformation', ['#fff4bd', '#ef9b62', DEFORMATION_COLOR]),
                       vmin=0, vmax=99, alpha=.8)
        for xy, color, style, label in [(outbound, OUTBOUND_COLOR, '-', 'Outbound trajectory'),
                                         (returning, RETURN_COLOR, '--', 'Return trajectory')]:
            if len(xy):
                line, = ax.plot(*xy.T, color=color, ls=style, lw=1.5, label=label, zorder=4)
                line.set_path_effects([path_effects.Stroke(linewidth=2.7, foreground='white'), path_effects.Normal()])
                # Direction is indicated on the recorded polyline, with no smoothing.
                if len(xy) > 5:
                    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
                    j0, j1 = np.searchsorted(arc, [max(0., arc[-1]/2-.2), min(arc[-1], arc[-1]/2+.2)])
                    ax.annotate('', xy=xy[j1], xytext=xy[j0],
                                arrowprops=dict(arrowstyle='->', color=color, lw=1.2), zorder=5)
        if i == 3:
            ax.scatter(*pose[:2], s=14, color='#202a35', edgecolors='white', linewidths=.5, zorder=7)
        ax.scatter(*home, marker='*', color='#163d69', edgecolors='white', linewidths=.5,
                   s=55, zorder=8, label='Start / home')
        if i > 0:
            ax.scatter(*panels[1][2][:2], marker='D', facecolors='white', edgecolors='#202a35',
                       linewidths=.8, s=24, zorder=6, label='Observation pose')
        ax.set_title('({}) {}'.format('abcd'[i], r['event']), loc='left', fontsize=8)
        label = '$t={:.2f}$ s'.format(r['relative_to_outbound_start_s'])
        ax.text(.97, .96, label, transform=ax.transAxes, ha='right', va='top', fontsize=8, zorder=10)
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), aspect='equal')
        ax.tick_params(axis='both', direction='out', length=3)
        ax.tick_params(axis='x', bottom=i >= 2, labelbottom=i >= 2)
        ax.tick_params(axis='y', left=i % 2 == 0, labelleft=i % 2 == 0)
        if i >= 2:
            ax.set_xlabel('x (m)')
        if i % 2 == 0:
            ax.set_ylabel('y (m)')
    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False, fontsize=7)
    fig.colorbar(im, ax=list(axes.flat), orientation='horizontal', shrink=.65, fraction=.035, pad=.025,
                 label='Published navigation cost (0–99)')
    for ext in ('pdf', 'png'):
        fig.savefig(out / ('fig_evacuation_map.'+ext))
    plt.close(fig)
    return manifest, used_files


def footprint_cost(grid, geometry, pose, half_length, half_width):
    """Max overlay cost over exact oriented rectangle/cell intersections.

    Reuse the navigation SAT predicate; out-of-map is missing, not zero cost.
    """
    args = (grid.ravel(), geometry.width, geometry.height, geometry.resolution,
            (geometry.origin_x, geometry.origin_y), pose, half_length, half_width)
    if not rectangular_footprint_is_traversable(*args, lethal_cost=101):
        return float("nan")
    low, high = 0, 100
    while low < high:
        middle = (low + high) // 2
        if rectangular_footprint_is_traversable(*args, lethal_cost=middle + 1):
            high = middle
        else:
            low = middle + 1
    return float(low)


def sustained_exit(times, costs, threshold, duration=1.0, maximum_gap=0.5):
    """First observed exit sustained for 1 simulation second; never bridge gaps."""
    start = None
    for index, (stamp, cost) in enumerate(zip(times, costs)):
        if (not np.isfinite(cost) or cost >= threshold
                or (index and stamp - times[index - 1] > maximum_gap)):
            start = None
        if np.isfinite(cost) and cost < threshold:
            if start is None:
                start = float(stamp)
            if stamp - start >= duration:
                return start
    return None


def route_metrics(xy, costs):
    distance = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    valid = np.isfinite(costs[:-1]) & np.isfinite(costs[1:])
    return dict(length_m=float(distance.sum()),
                evaluated_distance_m=float(distance[valid].sum()),
                exposure_m=float(np.sum(distance[valid] *
                    (costs[:-1][valid] + costs[1:][valid]) / 198.0)),
                inside_cost_distance_m=float(np.sum(distance[valid] *
                    ((costs[:-1][valid] > 0).astype(float) +
                     (costs[1:][valid] > 0).astype(float)) / 2.0)))


def require_completed_run(summary, manifest, intervened):
    if (summary.get("outcome") != "COMPLETED"
            or manifest.get("final_output_contract_status") != "COMPLETED"
            or intervened):
        raise ValueError("Refusing publication figures: return or recording is incomplete, or the trial was intervened")


def navigation_background(published, metadata, geometry, frame_id):
    """Exact common-domain crop after live SLAM expands; never fill missing data."""
    if (metadata["frame_id"] != frame_id
            or abs(metadata["resolution"]-geometry.resolution) > 1.e-6):
        raise ValueError("Navigation and deformation frame/resolution mismatch")
    offset = (np.array([geometry.origin_x, geometry.origin_y])
              - metadata["origin"]) / geometry.resolution
    rounded = np.rint(offset)
    if not np.isfinite(offset).all() or np.max(np.abs(offset-rounded)) > 1.e-4:
        raise ValueError("Navigation and deformation grids are not aligned")
    x, y = rounded.astype(int)
    if (x < 0 or y < 0 or x+geometry.width > published.shape[1]
            or y+geometry.height > published.shape[0]):
        raise ValueError("Navigation map does not cover the evaluation domain")
    return published[y:y+geometry.height, x:x+geometry.width].copy()


def imu_yaw_check(imu, truth, start, end):
    """Offline check for this planar simulator: gyro intervals versus pose yaw."""
    it = np.array([float(r["message_time_s"]) for r in imu])
    tt = np.array([float(r["message_time_s"]) for r in truth])
    wz = np.array([float(r["wz_rad_s"]) for r in imu])
    yaw = np.unwrap([float(r["model_yaw_rad"]) for r in truth])
    if (len(it) < 2 or len(tt) < 2 or np.any(np.diff(it) <= 0)
            or np.any(np.diff(tt) <= 0) or not np.isfinite(it).all()
            or not np.isfinite(tt).all() or not np.isfinite(wz).all()
            or not np.isfinite(yaw).all() or not start < end
            or max(it[0], tt[0]) > start or min(it[-1], tt[-1]) < end):
        raise ValueError("IMU/truth timestamps do not cover a valid return interval")
    # Each simulated gyro value averages the PRECEDING publication interval.
    integrated = np.r_[0., np.cumsum(wz[1:] * np.diff(it))]
    samples = tt[(tt >= start) & (tt <= end)]
    if not len(samples):
        raise ValueError("No truth samples inside return interval")
    delta_imu = np.interp(samples, it, integrated) - np.interp(start, it, integrated)
    delta_truth = np.interp(samples, tt, yaw) - np.interp(start, tt, yaw)
    error = delta_imu - delta_truth
    return dict(sample_count=len(imu), median_interval_s=float(np.median(np.diff(it))),
                yaw_change_rmse_rad=float(np.sqrt(np.mean(error**2))),
                yaw_change_max_error_rad=float(np.max(np.abs(error))))


def main(run):
    run = run.resolve()
    out = run / "analysis/evacuation"
    inputs = ["navigation/maps/decision_map.npz", "navigation/global_plans.csv",
              "navigation/action_events.csv", "trajectories/navigation_base_odometry.csv",
              "events/mission_states.csv", "mission_summary.json", "ros_parameters.json",
              "evolution/gdu_evolution.csv", "maps/avoidance_frames.csv",
              "trajectories/gazebo_robot_truth.csv", "truth/m01_m06_motion_truth.csv"]
    inputs.append("navigation/global_costmaps.csv")
    inputs.extend(["deformation/detection_evidence.csv", "trajectories/cmd_vel.csv"])
    for relative in inputs:
        if not (run / relative).is_file():
            raise ValueError("Incomplete recording: " + relative)
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if manifest_path.is_file():
        inputs.append("run_manifest.json")
    summary = json.loads((run / "mission_summary.json").read_text())
    require_completed_run(summary, manifest,
                          (run / "navigation/diagnostic_intervention.json").exists())
    out.mkdir(parents=True, exist_ok=True)
    params = json.loads((run / "ros_parameters.json").read_text())
    with np.load(run / inputs[0], allow_pickle=False) as stored:
        base, overlay, fused = (stored[name].copy() for name in ("base", "overlay", "fused"))
        meta = json.loads(str(stored["metadata"]))
    if base.shape != overlay.shape or not np.isfinite(overlay).all():
        raise ValueError("Invalid frozen layers")
    unsigned_metadata = dict(meta)
    snapshot_id = unsigned_metadata.pop("snapshot_id")
    digest = hashlib.sha256(json.dumps(unsigned_metadata, sort_keys=True).encode()
                            + base.tobytes() + overlay.tobytes()).hexdigest()
    if digest != snapshot_id or not np.array_equal(fused, fuse_planning_layers(base, overlay)):
        raise ValueError("Frozen decision map hash/fusion mismatch")
    geometry = GridGeometry(meta["resolution"], base.shape[1], base.shape[0], *meta["origin"])
    footprint = np.asarray(params["/move_base/global_costmap/footprint"], dtype=float)
    half_length, half_width = np.max(np.abs(footprint), axis=0)
    events = rows(run / "events/mission_states.csv")
    stamps = {item["state"]: float(item["message_time_s"] or item["ros_time_s"])
              for item in events}
    decision = stamps["DEFORMATION_DETECTED"]
    return_start = stamps["RETURNING_HOME"]
    completion = stamps.get("COMPLETED")
    # One reproducible evaluation background: the last published navigation
    # costmap before RETURNING_HOME, not a retrospectively selected clear map.
    available_maps = [item for item in rows(run / "navigation/global_costmaps.csv")
                      if float(item["message_time_s"]) <= return_start]
    if not available_maps:
        raise ValueError("No recorded navigation costmap before return")
    selected_map = max(available_maps, key=lambda item: float(item["message_time_s"]))
    selected_path = (run / selected_map["relative_path"]).resolve()
    if run not in selected_path.parents:
        raise ValueError("Navigation snapshot path escapes run directory")
    with np.load(selected_path, allow_pickle=False) as saved:
        published = saved["data"].copy()
        published_metadata = json.loads(str(saved["metadata"]))
    published_hash = hashlib.sha256(json.dumps(published_metadata, sort_keys=True).encode()
                                     + published.tobytes()).hexdigest()
    if published_hash != selected_map["map_id"]:
        raise ValueError("Recorded navigation costmap hash mismatch")
    inputs.append(selected_map["relative_path"])
    # Hash the full original snapshot first, then evaluate the unchanged
    # decision domain. Map growth must neither shift costs nor invent free cells.
    published = navigation_background(published, published_metadata, geometry, meta["frame_id"])
    # Remove costmap inflation/soft values before applying the identical
    # footprint inflation used by all three offline A* policies.
    base = np.where(published < 0, -1, np.where(published >= 100, 100, 0)).astype(np.int16)
    odometry = rows(run / "trajectories/navigation_base_odometry.csv")
    selected = [item for item in odometry
                if float(item["message_time_s"]) >= return_start
                and (completion is None or float(item["message_time_s"]) <= completion)]
    if len(selected) < 2 or any(item["frame_id"] != meta["frame_id"] for item in selected):
        raise ValueError("Missing return trajectory or inconsistent coordinate frame")
    selected.sort(key=lambda item: float(item["message_time_s"]))
    times = np.array([float(item["message_time_s"]) for item in selected])
    poses = np.array([[float(item[key]) for key in ("x_m", "y_m", "yaw_rad")]
                      for item in selected])
    if np.any(np.diff(times) <= 0):
        raise ValueError("Return odometry timestamps must be strictly increasing")
    costs = np.array([footprint_cost(overlay, geometry, pose, half_length, half_width)
                      for pose in poses])
    moved = np.flatnonzero(np.linalg.norm(poses[:, :2] - poses[0, :2], axis=1) >= 0.05)
    motion = float(times[moved[0]]) if len(moved) else None
    exit_core = sustained_exit(times, costs, 99) if costs[0] >= 99 else None
    exit_all = sustained_exit(times, costs, 1) if costs[0] > 0 else None
    actual = route_metrics(poses[:, :2], costs)
    actual.update(method="Executed (SLAM pose)", trajectory_observed=True,
                  return_completed=summary.get("outcome") == "COMPLETED",
                  initial_footprint_cost=float(costs[0]) if np.isfinite(costs[0]) else None,
                  initially_inside_core=bool(costs[0] >= 99),
                  motion_time_s=motion, exit_core_time_s=exit_core,
                  exit_cost_region_time_s=exit_all,
                  motion_to_exit_s=(max(0.0, exit_all - motion)
                                    if exit_all is not None and motion is not None else None),
                  max_sample_gap_s=float(np.diff(times).max()))
    save_csv(out / "executed_exposure.csv", [
        dict(time_s=float(t), relative_to_decision_s=float(t - decision),
             x_m=float(p[0]), y_m=float(p[1]), yaw_rad=float(p[2]),
             footprint_cost=float(c)) for t, p, c in zip(times, poses, costs)])

    # Same A*, exact same start/goal and geometry for all OFFLINE policies.
    # These are not the live Navfn planner, and are never called executed routes.
    grid = inflate_obstacles(base, math.hypot(half_length, half_width), geometry.resolution)
    start = geometry.world_to_cell(*poses[0, :2])
    goal = geometry.world_to_cell(*summary["home_map_xy"])
    comparisons = []
    for name, weight, hard in (("SLAM-only A*", 0.0, False),
                               ("Hard-core A*", 4.0, True),
                               ("Soft-cost A*", 4.0, False)):
        current = grid.copy()
        if hard:
            current = base.copy()
            current[overlay >= 99] = 100
            current = inflate_obstacles(current, math.hypot(half_length, half_width), geometry.resolution)
        begin = time.perf_counter()
        path, _ = astar(current, start, goal, overlay=overlay, warning_weight=weight,
                        unknown_is_traversable=False)
        duration = 1000.0 * (time.perf_counter() - begin)
        record = dict(method=name, feasible=path is not None, planning_ms=duration,
                      length_m=None, evaluated_distance_m=None, exposure_m=None,
                      inside_cost_distance_m=None)
        if path:
            xy = np.array([geometry.cell_to_world(cell) for cell in path])
            directions = np.diff(xy, axis=0)
            yaw = np.arctan2(directions[:, 1], directions[:, 0])
            yaw = np.r_[yaw, yaw[-1]] if len(yaw) else np.array([poses[0, 2]])
            values = np.array([footprint_cost(overlay, geometry, (*point, angle),
                                             half_length, half_width)
                               for point, angle in zip(xy, yaw)])
            record.update(route_metrics(xy, values))
            save_csv(out / (name.split()[0].lower() + "_counterfactual.csv"),
                     [dict(x_m=p[0], y_m=p[1], footprint_cost=v) for p, v in zip(xy, values)])
        comparisons.append(record)
    save_csv(out / "offline_policy_comparison.csv", comparisons)
    plans = rows(run / "navigation/global_plans.csv")
    return_plans = [item for item in plans if float(item["ros_time_s"]) >= return_start
                    and int(item["point_index"]) >= 0
                    and item["frame_id"] == meta["frame_id"]]
    result = dict(run=str(run), map=meta, executed=actual, offline_comparisons=comparisons,
                  evaluation_navigation_map=selected_map,
                  events=stamps, mission_status=summary.get("outcome"),
                  recording_contract=manifest.get("final_output_contract_status"),
                  return_published_plan_count=len({item["plan_id"] for item in return_plans}),
                  limitations=["Single run, not a success-rate estimate.",
                               "Exposure is a deformation-navigation cost, not collapse probability.",
                               "Deformation costs are frozen; both planners keep live LiDAR obstacle updates.",
                               "Offline policies use the last published pre-return navigation geometry; not exact live planner inputs.",
                               "A* counterfactuals are offline, not Navfn executions.",
                               "Executed exposure uses recorded navigation-base odometry derived from FAST-LIO, not Gazebo truth.",
                               "Footprint exposure uses sampled poses and trapezoidal quadrature; not continuous swept-volume certification."])
    source_candidates = [item for item in rows(run / "maps/avoidance_frames.csv")
                         if abs(float(item["message_time_s"]) - meta["overlay_stamp_s"]) < 1e-6]
    result["frozen_overlay_source_candidates"] = [
        {key: item[key] for key in ("source_snapshot_id", "source_accepted_frame_index", "map_revision",
                                   "all_current_sources_within_map")}
        for item in source_candidates]
    action_events = rows(run / "navigation/action_events.csv")
    result["aborted_outbound_action_count"] = sum(
        item["event"] == "result" and item["status"] == "4"
        and stamps['NAVIGATING_OUTBOUND'] <= float(item['ros_time_s']) <= stamps['STABILIZING_AT_TARGET']
        for item in action_events)
    result["return_action_events"] = [item for item in action_events
                                      if float(item["ros_time_s"]) >= return_start]
    result["aborted_return_action_count"] = sum(
        item["event"] == "result" and item["status"] == "4"
        for item in result["return_action_events"])
    recoveries = []
    for item in result['return_action_events']:
        if item['event'] != 'result' or item['status'] != '4':
            continue
        start_stamp = float(item['ros_time_s'])
        later_goals = [float(g['ros_time_s']) for g in result['return_action_events']
                       if g['event'] == 'goal' and float(g['ros_time_s']) > start_stamp]
        stop_stamp = min(later_goals) if later_goals else None
        recoveries.append(dict(abort_receive_time_s=start_stamp, next_goal_receive_time_s=stop_stamp,
                               duration_sim_s=None if stop_stamp is None else stop_stamp-start_stamp,
                               aborted_goal_id=item['goal_id'], reason=item['text'],
                               source='navigation/action_events.csv'))
    result['recorded_action_retries'] = recoveries
    save_csv(out / 'action_retries.csv', recoveries)
    truth = rows(run / "trajectories/gazebo_robot_truth.csv")
    truth_times = np.array([float(item["message_time_s"]) for item in truth])
    truth_xy = np.array([[float(item["model_x_m"]), float(item["model_y_m"])] for item in truth])
    if not truth_times[0] <= completion <= truth_times[-1]:
        raise ValueError("Robot truth does not cover arrival")
    truth_arrival = np.array([np.interp(completion, truth_times, truth_xy[:, i]) for i in range(2)])
    result["arrival_check"] = dict(
        return_duration_sim_s=completion-return_start,
        slam_home_distance_m=summary["final_home_distance_m"],
        gazebo_home_distance_m=float(np.linalg.norm(truth_arrival-truth_xy[0])))
    if (run / "sensors/imu.csv").is_file():
        inputs.append("sensors/imu.csv")
        result["imu_yaw_check"] = imu_yaw_check(rows(run / "sensors/imu.csv"), truth, return_start, completion)
    evolutions = rows(run / "evolution/gdu_evolution.csv")
    evidence = rows(run / "deformation/detection_evidence.csv")
    map_frames = rows(run / "maps/avoidance_frames.csv")
    commands = rows(run / "trajectories/cmd_vel.csv")
    ledger, intervals, publications, trigger_series = event_chain(
        summary, stamps, meta, evolutions, evidence, map_frames,
        rows(run / "navigation/action_events.csv"), plans, commands,
        rows(run / "truth/m01_m06_motion_truth.csv"), motion, exit_all, state_events=events)
    save_csv(out / "event_lineage.csv", ledger)
    save_csv(out / "response_intervals.csv", intervals)
    save_csv(out / "path_publications.csv", publications)
    valid_publications = [p for p in publications if p['point_count']]
    result['return_published_plan_count'] = len(valid_publications)
    result["event_lineage"] = ledger
    result["response_intervals"] = intervals
    result["trigger_activation_mode"] = params.get(
        "/deformation_evolution/deform_evolution/activation_mode", "UNRECORDED")
    # The public evidence and evolution layers share exact frame/epoch/GDU keys.
    indexed_evidence = {(r['reference_epoch'], r['accepted_frame_index'], r['gdu_id']): r
                        for r in evidence}
    linked_series = []
    for r in trigger_series:
        e = indexed_evidence.get((r['reference_epoch'], r['source_accepted_frame_index'], r['gdu_id']))
        linked_series.append(dict(source_time_s=float(r['message_time_s']),
                                  relative_to_monitoring_s=float(r['message_time_s'])-stamps['MONITORING'],
                                  snapshot_id=r['source_snapshot_id'], frame_index=r['source_accepted_frame_index'],
                                  gdu_id=r['gdu_id'], displacement_rms_mm=1000*float(r['displacement_rms_m']),
                                  active_now=r['active_now'], retained_for_avoidance=r['retained_for_avoidance'],
                                  evidence_joined=e is not None,
                                  detection_admission=e['detection_admission'] if e else '',
                                  detection_allowed=e['detection_allowed'] if e else '',
                                  evidence_publish_time_s=e['publish_time_s'] if e else '',
                                  evidence_receive_time_s=e['receive_time_s'] if e else '',
                                  evolution_receive_time_s=r['receive_time_s']))
    save_csv(out / "trigger_evidence_chain.csv", linked_series)
    result['unmatched_trigger_evidence_rows'] = sum(not r['evidence_joined'] for r in linked_series)
    following = path_following(plans, publications, times, poses, stamps['MONITORING'])
    save_csv(out / 'path_following.csv', following)
    deviations = [r['distance_to_last_published_path_m'] for r in following
                  if r['distance_to_last_published_path_m'] is not None]
    unique_stamps = sorted({r['source_time_s'] for r in publications})
    result['path_publication_audit'] = dict(
        publication_count=len(publications), distinct_source_timestamp_count=len(unique_stamps),
        median_distinct_source_interval_sim_s=float(np.median(np.diff(unique_stamps))) if len(unique_stamps)>1 else None,
        evaluated_pose_count=len(deviations), missing_path_pose_count=len(following)-len(deviations),
        point_to_latest_path_median_m=float(np.median(deviations)) if deviations else None,
        point_to_latest_path_p95_m=float(np.percentile(deviations, 95)) if deviations else None)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                         "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "pdf.fonttype": 42, "savefig.dpi": 350})
    keyframes, keyframe_files = plot_keyframes(run, out, stamps, ledger, odometry, summary['home_map_xy'])
    inputs.extend(f for f in keyframe_files if f not in inputs)
    result['keyframe_sources'] = keyframes
    (out / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    save_csv(out / 'paper_metrics.csv', paper_metrics(result))
    plot_chain(out, ledger, trigger_series, times, costs, stamps)
    (out / 'FIGURE_CAPTION.md').write_text('\n'.join([
        '# 形变证据与实际撤离：双语图注草案（未写入论文）', '',
        '## 中文', '',
        '微小形变证据与实际撤离响应。两个子图均以返航触发时刻为零点，横轴为仿真时间。'
        '（a）触发此次撤离的局部监测单元的位移估计均方根幅度；横括号表示同一变化来源持续保留的证据区间，虚线表示撤离触发。'
        '（b）沿实际记录的机器人位姿计算的车体矩形足迹最大形变代价，采用触发时冻结的形变代价层。'
        '只展示实际记录时段，不将形变估计外推至返航；运动中的停滞与外围区域再接触均保留。'
        '代价为规划回避指标，不是结构失效概率；零代价仅表示车体未覆盖该层的非零区域。', '',
        '## English', '',
        'Deformation evidence and executed withdrawal response. Both panels use the return trigger as '
        'the origin of simulation time. (a) RMS displacement estimate of the local monitoring unit '
        'that triggered withdrawal. The bracket denotes the interval of continuously retained evidence '
        'from the same source; dashed lines mark the trigger. (b) Maximum deformation cost under the '
        'rectangular robot footprint at recorded poses, evaluated on the deformation layer frozen at '
        'the decision. Only recorded intervals are shown, without extrapolating deformation estimates '
        'into the return. Stalls and re-entry into peripheral cost regions are retained. The cost is a '
        'planning penalty, not a structural-failure probability; zero indicates no footprint overlap '
        'with nonzero cells of this layer.', '',
        '## 本轮解释与来源', '',
        '- 触发源 GDU：`'+str(summary['return_trigger_algorithm_source_id'])+'`；选取依据是实际任务触发源，不是事后选择曲线最好看的目标。',
        '- 触发源时刻：{:.3f} s；空间地图中对应 t={:.3f} s。本图横轴减去前者，空间地图仍减去进场时刻。'.format(decision, decision-stamps['NAVIGATING_OUTBOUND']),
        '- 蓝线不是 Gazebo 真值，也不代表全部物体的平均位移；不是精度对照或完整物体运动时序。持续保留也不等于物体持续运动。',
        '- 激活口径为 `'+result['trigger_activation_mode']+'`，不能将演化层的激活/保留改写为严格概率检测准入。',
        '- 返程有 {} 次 ABORTED 动作；恢复区间见 action_retries.csv，未从代价曲线中删去对应样本或压缩时间。'.format(result['aborted_return_action_count']),
        '- 上图来源：trigger_evidence_chain.csv、evolution/gdu_evolution.csv、event_lineage.csv；下图来源：executed_exposure.csv、原始导航里程计与 navigation/maps/decision_map.npz。',
        '- 单次案例展示感知驱动撤离及已感知形变区域的接触变化，不能据此宣称绝对安全、六目标全部检出或重复试验的统计优势。',
    ])+'\n')
    # Retire only named, derived plots; all recorded data and numerical audits remain.
    for name in ('fig_evacuation_timeline', 'fig_navigation_response', 'fig_path_following'):
        for ext in ('pdf', 'png'):
            obsolete = out / (name+'.'+ext)
            if obsolete.is_file():
                obsolete.unlink()
    for relative in ('scene/scene_startup.json', 'scene/loaded_scene.world', 'scene/motion_config.yaml',
                     'scene/scene_before_motion.json', 'scene/scene_after_return.json',
                     'provenance/execution_versions.json', 'provenance/execution_sources.zip',
                     'runtime/per_cycle.csv', 'deformation/algorithm_displacement.csv',
                     'analysis/model_detection_event_metrics.csv',
                     'analysis/detection_track_inventory.csv', 'analysis/detector_runtime_summary.json'):
        if (run / relative).is_file():
            inputs.append(relative)
    provenance = {relative: hashlib.sha256((run / relative).read_bytes()).hexdigest()
                  for relative in inputs}
    provenance["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    provenance["timeline_source_sha256"] = hashlib.sha256(
        Path(__file__).with_name('evacuation_timeline.py').read_bytes()).hexdigest()
    (out / "SOURCE_HASHES.json").write_text(json.dumps(provenance, indent=2) + "\n")
    report = ["# 单次撤离：形变证据—地图—路径更新—实际行驶", "", "原始运行：`" + str(run) + "`", "",
              "## 先看什么", "",
              "1. [完整过程地图](fig_evacuation_map.pdf)：2×2 空间过程，d 仅标注触发时间，事后叠加完整实际返程。",
              "2. [证据与返航响应](fig_evacuation_response.pdf)：单栏上下两个子图，a 撤离前的触发源位移证据，b 撤离后的实际车体形变代价；两面板均以返航触发为零点，上图时间非正、下图时间非负。不重复地图、轨迹或里程碑清单。双语图注见 FIGURE_CAPTION.md。", "",
              "## 本次记录能说明什么", "",
              "任务完成返航，但触发来自研究模式演化证据，不等于严格检测准入通过。",
              "决策与返航之间实际存在 {:.3f} s 流程等待，不能把它隐藏或解释为规划计算耗时。".format(return_start-decision),
              "返航 {:.3f} s，实际导航里程计轨迹长 {:.3f} m；真值起点距离 {:.3f} m。".format(
                  completion-return_start, actual['length_m'], result['arrival_check']['gazebo_home_distance_m']), "",
              "形变代价使用冻结决策层；实时激光障碍仍更新，99 为有限软代价，不代表坍塌概率。",
              "决策层完整覆盖标记：{}。False 表示部分源的膨胀区域被地图边界裁剪；只评价已记录的地图域，不将域外视为安全或完整覆盖。".format(
                  ', '.join(r['all_current_sources_within_map'] for r in source_candidates) or '无法关联'),
              "离线 A* 对照使用返航命令前最后发布的导航地图；关键时刻图另用各时刻之前实际收到的地图。实际控制过程不冻结真实障碍。",
              "实际路线来自 navigation_base 里程计；离线 A* 对照不是实际 Navfn 执行。", "",
              "进场/返程动作 ABORTED 次数：{}/{}。重试与转向耗时、真实轨迹均保留；完成任务不代表过程中从未中止局部动作。".format(
                  result['aborted_outbound_action_count'], result['aborted_return_action_count']), "",
              "完整数值及单位见 `metrics.json`，不将缺测值转换为零。", "",
              "论文数值索引见 `paper_metrics.csv`：每行包含指标、数值、单位、定义与原始来源。空值是未观测，不是零。",
              "形变精度与误检测请联查上一级 `model_detection_event_metrics.csv`、`detection_track_inventory.csv` 和原始 `deformation/algorithm_displacement.csv`；不能用触发 GDU 的位移曲线替代六个物体的精度比较。",
              "检测耗时见上一级 `detector_runtime_summary.json`，原始逐次值在 `runtime/per_cycle.csv`；它是检测回调墙钟总耗时，未记录各内部阶段耗时，不能和仿真时间间隔混用。",
              "本次运行代码/配置存档及已编译二进制哈希见 `provenance/`（若存在）；实际加载场景见 `scene/`，有效参数见 `ros_parameters.json`。系统依赖与网格资产未整体打包。", "",
              "离线对照见 offline_policy_comparison.csv；全部输入哈希见 SOURCE_HASHES.json。",
              "首次出区要求完整矩形足迹连续 1 s 不接触代价区域，超过 0.5 s 的采样缺口打断确认；不表示后续从未再次进入外围软代价带。",
              "未观察到出区时保留 null，不以零秒或到达目标冒充出区。",
              "空间地图以 NAVIGATING_OUTBOUND 为零点；定量响应图以 DEFORMATION_DETECTED 的源时间为零点。CSV 保留原始仿真时间及明确命名的相对时间，不能混用。",
              "本次初始足迹最大代价为 " + str(actual["initial_footprint_cost"]) +
              "；若小于 99，则没有验证初始足迹位于核心区内部的困难情形。",
              "一轮实验只能展示案例，不能计算多次实验成功率或显著性。"]
    report.extend(["", "## 到达与仿真 IMU 校验", "",
                   "```json", json.dumps({k: result[k] for k in ("arrival_check", "imu_yaw_check") if k in result}, indent=2), "```",
                   "Gazebo 距离由模型原点在世界坐标中的起始/到达位置计算；IMU 校验仅用于平面运动仿真的离线诊断。"])
    report.extend(['', '## 感知证据—撤离效果与事件来源', '',
                   '`fig_evacuation_response.pdf` 为 3.5 英寸单栏双子图：a 画触发 GDU 的估计位移均方根和持续证据区间，b 画实际位姿在冻结形变层上的矩形足迹最大代价。图内只保留曲线、轴标题、持续时间括号及零时刻虚线。',
                   '两子图采用同一返航触发时间零点，但显示前后不同区间，不将尚未记录的触发瞬间位移或返航起始代价补齐。位移止于实际观测，代价曲线保留所有实际采样值，超过 0.5 s 的位姿采样缺口断线，不作空间或时间平滑。',
                   '图内不再绘制消息源/接收时间对照、路径长度、HOME 距离或恢复灰带；这些数值仍保存在事件、路径和动作审计 CSV 中。原地调整、自动恢复和外围再接触没有从实际代价曲线中删去。双语图注与本轮口径见 FIGURE_CAPTION.md。',
                   '距离加权暴露为实测相邻位姿间弧长乘以端点足迹代价平均值/99，单位为 m，不是坍塌概率。单次实验不绘制伪置信区间。点到最新已接收路径的距离仍保留在 path_following.csv，不再单独出图。',
                   '所有数值可从 `event_lineage.csv`、`response_intervals.csv`、`trigger_evidence_chain.csv` 和 `path_publications.csv` 回溯。', '',
                   '| 事件 | 相对监测开始 / s | 仿真源时间 / s | 记录器接收时间 / s |',
                   '|---|---:|---:|---:|'])
    def fmt(value):
        return '未记录/不适用' if value is None else '{:.3f}'.format(value)
    event_names = dict(monitoring='开始监测', truth_running='真值激励开始', truth_holding='真值停止运动并保持',
                       first_active='触发 GDU 首次 active（研究模式）', first_retained='同一 GDU 首次保留',
                       first_trigger_map='首次进入避让地图（按精确 ID 关联）', persistence_source='持续时间满足的源数据帧',
                       persistence_decision='任务触发撤离 / 检测记录截止', map_freeze='决策地图冻结',
                       return_state='进入返航阶段', return_goal='下发返航目标', first_return_path='随后首条非空路径发布',
                       first_return_control='返航阶段首次非零控制（包括目标下发前的转向）',
                       first_motion_command='目标下发后首条非零控制命令（仅接收时间）', measured_motion='实测平移达到 5 cm',
                       first_sustained_exit='首次持续出区的区间起点', arrival='任务抵达起点',
                       strict_detection='触发 GDU 严格检测准入')
    for r in ledger:
        report.append('| {} | {} | {} | {} |'.format(event_names.get(r['event'], r['event']), fmt(r['relative_to_monitoring_s']),
                      fmt(r['source_time_s']), fmt(r['receive_time_s'])))
    report.extend(['', '### 必须区分的含义', '',
                   '- 当前激活策略：`'+result['trigger_activation_mode']+'`。active/retained 是演化层状态，不可改写为严格 DETECTED_AT_SCALE。严格检测事件未出现时记录为空，不能当作零延迟。',
                   '- 真值只说明激励开始/停止的实际时刻；未进行目标表面精确关联前，不能把激励至 GDU 激活间隔当作某个物体的检测精度指标。',
                   '- GDU—地图通过 snapshot ID 和 accepted frame 联结；相同源时间不表示地图计算耗时为零。不同订阅回调接收顺序也不等于处理先后顺序。',
                   '- 决策至返航任务阶段间隔为 {:.3f} s，是任务准备与停止激励等流程耗时，不能计作检测或规划计算耗时。旧版等待导出全部完成；修正版只等待输入捕获和决策地图就绪，导出异步继续。具体运行版本以该次源代码记录、配置和日志为准；不能仅凭间隔长度推断。'.format(return_start-decision),
                   '- RETURNING_HOME 包含必要的车体朝向对准；HOME 目标可在转向完成后才下发。首次返航控制、目标下发后控制、实际平移是不同事件，均保存在事件表，不把原地转向或导出耗时计作规划器求解时间。',
                   '- 路径记录包含 {} 次非空发布，不等于 {} 次独立重规划。Path 不含 goal ID 或求解起止事件；目标至后续首条路径发布间隔仅为观测响应，不能称为 Navfn 求解时间。'.format(len([p for p in publications if p['point_count']]), len([p for p in publications if p['point_count']])),
                   '- 检测记录截止后不续画形变测量；冻结的形变层仅用于评价后续行驶的代价，不冒充新的测量或预测。路径与实际运动持续记录至返航完成。',
                   '- 首次持续出区是事后确认的区间起点，不是在线控制器事件；随后再次进入外围软代价带的曲线完整保留。',
                   '- 本次到达容差后取消动作，PREEMPTED 不等于返航失败；以任务 COMPLETED 和实际到达距离联合核对。',
                   '- 本组图展示一条有时间证据的闭环执行链，不能单独证明严格检测成功、预测精度或导航算法的统计优越性。', '',
                   '证据精确联结缺失行数：{}。'.format(result['unmatched_trigger_evidence_rows'])])
    report.extend(['', '### 路径发布与跟随核对', '',
                   '```json', json.dumps(result['path_publication_audit'], indent=2), '```',
                   '同一源时间可能发布多条路径，distinct_source_timestamp_count 也不是已测量的求解次数。点到路径距离是几何跟随诊断，不是动态控制器误差真值；它受 SLAM 位姿误差、路径修订和发布/接收延迟影响。路径年龄和实际选用 ID 见 path_following.csv。'])
    report.extend(['', '## 完整过程 2×2 地图：读图边界', '',
                   '四面板共用坐标范围、比例尺与色标；上排不画 x 轴刻度线和数字，右列不重复 y 轴刻度。',
                   'a 起点准备进场；b 到达观测位置；c Gazebo 真值记录的物体运动开始；d 形变触发的返航过程总览。机器人进场轨迹保留于 b/c/d，完整实际返程仅在 d 绘制。',
                   'a/b/c 的数据截止于标注事件，d 的数据截止于返航完成；地图和位姿的源时间与记录器接收时间均不晚于各自数据截止时间。不进行空间平滑，不以规划路线代替执行轨迹。',
                   '空间图右上角 t 以 NAVIGATING_OUTBOUND 为零点；定量响应图改以返航触发为零点。监测开始位于空间图 t={:.3f} s，原始仿真时间均保留在来源表。'.format(stamps['MONITORING']-stamps['NAVIGATING_OUTBOUND']),
                   '颜色是发布的导航总代价（包含障碍膨胀和形变影响），不是单独形变层或坍塌概率；深灰为 >=100，浅灰为 UNKNOWN。不从融合地图反推独立形变层。',
                   'd 右上角仅标注返航触发时间，不在图内标注到达 HOME 的时间。d 是事后叠加完整返航轨迹的过程总览，不是触发瞬间的地图快照；返航完成时间保留在原始记录和来源表的数据截止时间中，供后续图注使用。此处触发仍为研究模式持续保留证据，不等于严格检测准入。',
                   '星号为起点/返航目标，菱形为 b 时刻已知的观测位置，d 中黑点为返航完成时最近已收到的机器人位姿。图中没有使用 m01–m06 的先验框选。',
                   'keyframe_sources.csv 分别保存标注事件时间 event_sim_time_s 和数据截止时间 data_cutoff_sim_time_s，地图及位姿年龄相对于数据截止时间计算；同时保留地图文件/哈希、源时间/接收时间和轨迹点数。keyframe_executed_poses.csv 按 panel 和 plotted_phase 保存实际绘制的轨迹。旧三面板图的 keyframe_published_paths.csv 已移除（非原始记录）。', '',
                   '| 面板 | 标注事件仿真时间 / s | 图中 t / s | 数据截止仿真时间 / s | 地图年龄 / s | 位姿年龄 / s | 进场/返程点数 |',
                   '|---|---:|---:|---:|---:|---:|---|'])
    for r in keyframes:
        report.append('| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {}/{} |'.format(
            r['panel'], r['event_sim_time_s'], r['relative_to_outbound_start_s'],
            r['data_cutoff_sim_time_s'], r['map_age_s'], r['pose_age_s'],
            r['outbound_point_count'], r['return_point_count']))
    if (run / 'scene/scene_startup.json').is_file():
        scene = json.loads((run / 'scene/scene_startup.json').read_text())
        report.extend(['', '## 本次场景版本与静止核对', '',
                       '加载世界 SHA-256：`'+scene['world_sha256']+'`。原样副本见 `scene/loaded_scene.world`，运动配置见 `scene/motion_config.yaml`。',
                       '以下核对仅读取 Gazebo，不向检测器提供模型先验；同一保存场景/参数保证初始配置可核对，并不意味着异步 SLAM 每次输出逐点相同。'])
        for name in ('scene_startup.json', 'scene_before_motion.json', 'scene_after_return.json'):
            if (run / 'scene' / name).is_file():
                audit = json.loads((run / 'scene' / name).read_text())
                report.append('- `{}`：{}，{} 个模型。'.format(name, audit['status'], len(audit['models'])))
    (out / "REPORT.md").write_text("\n".join(report) + "\n")
    onset = next(r['source_time_s'] for r in ledger if r['event'] == 'truth_running')
    origin = stamps['NAVIGATING_OUTBOUND']
    notes = [
        '# 本轮论文分析速查（未写入论文）', '',
        '一次独立闭环案例；以下时间均为仿真时间，以机器人开始进场为零点。', '',
        '- 到达观测位置：{:.2f} s；开始监测：{:.2f} s。'.format(stamps['STABILIZING_AT_TARGET']-origin, stamps['MONITORING']-origin),
        '- 物体开始运动：{:.2f} s；触发返航：{:.2f} s；两者间隔：{:.2f} s。'.format(onset-origin, decision-origin, decision-onset),
        '- 到达 HOME：{:.2f} s（供图 d 图注使用，不写入图内）。'.format(completion-origin),
        '- 实际返航轨迹长 {:.3f} m；冻结形变层上的距离加权暴露 {:.3f} m；Gazebo 真值终点距离 HOME {:.3f} m。'.format(actual['length_m'], actual['exposure_m'], result['arrival_check']['gazebo_home_distance_m']),
        '- 进场/返程 ABORTED 动作次数：{}/{}。'.format(result['aborted_outbound_action_count'], result['aborted_return_action_count']), '',
        '这些数值展示“形变证据—地图—任务触发—真实返航”的执行链，不是结构失效概率、单物体检测延迟或多次实验统计结论。当前激活方式为 `'+result['trigger_activation_mode']+'`；严格检测事件请以 event_lineage.csv 为准。', '',
        '图表入口：[2×2 过程地图](fig_evacuation_map.pdf)、[定量响应图](fig_evacuation_response.pdf)。',
        '定量响应图为单栏双子图，使用返航触发为时间零点；不重复空间地图或事件清单。双语图注见 FIGURE_CAPTION.md；内部状态、恢复与路径诊断只在记录和报告中解释。',
        '数值与来源：[paper_metrics.csv](paper_metrics.csv)、[完整审计报告](REPORT.md)、[输入哈希](SOURCE_HASHES.json)。',
        '单物体事件指标：上一级 model_detection_event_metrics.csv；全部检测轨迹（含未关联/误检测候选）：detection_track_inventory.csv。真值、测量、协方差、有效性保留在原始 CSV，不能只筛选成功样本。',
        '检测回调墙钟总耗时：上一级 detector_runtime_summary.json；内部阶段计时没有被记录。返航后的里程计与检测截止前的精度评估分开处理。',
    ]
    object_file = run / 'analysis/model_detection_event_metrics.csv'
    if object_file.is_file():
        objects = rows(object_file)
        notes.extend(['', '## 本轮逐物体结果（含未检出）', '',
                      '| 物体 | 确认检出 | 确认延迟 / 仿真 s | 最大真值位移 / mm |',
                      '|---|---|---:|---:|'])
        for item in objects:
            delay = item['confirmation_latency_s']
            notes.append('| {} | {} | {} | {:.3f} |'.format(item['model_name'],
                '是' if item['detected'] == 'True' else '否',
                '{:.3f}'.format(float(delay)) if delay else '未确认',
                1000*float(item['maximum_truth_displacement_m'])))
        notes.extend(['', '该表是现有离线真值关联/轨迹确认口径，不等于严格概率检测准入。未确认对象不能删除或记为零延迟；m04 仍展示，但按既定口径不纳入主要指标。'])
    (out / 'PAPER_NOTES_ZH.md').write_text('\n'.join(notes)+'\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    main(args.run_directory)
