#!/usr/bin/env python3
"""Export data-derived maps and static-robot path comparisons without bags."""

import csv
from collections import deque
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/deform_monitor_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry, Path
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

from deform_avoidance_map.msg import DeformationAvoidanceMap
from deform_evolution_map.msg import DeformationEvolutionFrame
# catkin's devel-space relay executes this source while keeping the relay
# directory in sys.path.  Add the real source directory so the pure planning
# module is imported as code, not as another relay wrapper.
_SCRIPT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
if _SCRIPT_DIRECTORY not in sys.path:
    sys.path.insert(0, _SCRIPT_DIRECTORY)
from planning_core import (
    GridGeometry,
    astar,
    choose_comparison_goal,
    complete_forecast_horizons,
    crop_occupancy_grid,
    deformation_changes_route,
    estimate_floor_z,
    fuse_planning_layers,
    first_rotation_clearance_index,
    inflate_obstacles,
    lidar_observation_grid,
    nearest_traversable,
    path_length_m,
    prediction_window_covers_horizons,
    select_latest_forecast_candidate,
    snapshot_identity_matches,
)


def _finite_point(point):
    return all(math.isfinite(value) for value in (point.x, point.y, point.z))


class StaticPlanningEvaluation:
    def __init__(self):
        prefix = "~planning_evaluation/"
        self.world_frame = rospy.get_param(prefix + "world_frame", "camera_init")
        self.extent_m = float(rospy.get_param(prefix + "local_extent_m", 12.0))
        self.resolution = float(rospy.get_param(prefix + "resolution_m", 0.10))
        self.robot_radius = float(rospy.get_param(prefix + "robot_radius_m", 0.35))
        self.rotation_clearance = float(rospy.get_param(
            prefix + "rotation_clearance_m", 1.00))
        self.rotation_acceptance_tolerance = float(rospy.get_param(
            prefix + "rotation_acceptance_tolerance_m", 0.15))
        self.minimum_reverse_distance = float(rospy.get_param(
            prefix + "minimum_reverse_distance_m", 5.0))
        self.maximum_range = float(rospy.get_param(prefix + "maximum_range_m", 8.0))
        self.minimum_goal_radius = float(rospy.get_param(
            prefix + "minimum_goal_radius_m", 1.5))
        self.maximum_goal_radius = float(rospy.get_param(
            prefix + "maximum_goal_radius_m", 5.5))
        self.expected_forecast_horizons = [float(value) for value in rospy.get_param(
            prefix + "expected_forecast_horizons_s", [0.5, 1.0, 2.0])]
        self.minimum_cloud_frames = int(rospy.get_param(
            prefix + "minimum_cloud_frames", 20))
        self.cloud_window_frames = int(rospy.get_param(
            prefix + "cloud_window_frames", 40))
        self.cloud_window_duration = float(rospy.get_param(
            prefix + "cloud_window_duration_s", 4.0))
        self.snapshot_cloud_tolerance = float(rospy.get_param(
            prefix + "snapshot_cloud_tolerance_s", 1.0))
        self.maximum_points = int(rospy.get_param(prefix + "maximum_points", 300000))
        self.cloud_stride = int(rospy.get_param(prefix + "cloud_stride", 4))
        self.require_active_source = bool(rospy.get_param(
            prefix + "require_active_source", True))
        self.export_mission_evacuation = bool(rospy.get_param(
            prefix + "export_mission_evacuation", False))
        self.require_detection_state = bool(rospy.get_param(
            prefix + "require_detection_state", False))
        self.require_local_path_difference = bool(rospy.get_param(
            prefix + "require_local_path_difference", True))
        self.use_latest_complete_forecast_before_cutoff = bool(rospy.get_param(
            prefix + "use_latest_complete_forecast_before_cutoff", False))
        self.forecast_candidate_history_size = int(rospy.get_param(
            prefix + "forecast_candidate_history_size", 8))
        self.output_root = rospy.get_param(
            prefix + "output_root",
            str(Path(__file__).resolve().parents[3] / "experiment_output" / "analysis"))
        self.run_name = rospy.get_param(prefix + "run_name", "")
        if (self.minimum_cloud_frames <= 0
                or self.cloud_window_frames < self.minimum_cloud_frames
                or self.cloud_window_duration <= 0.0
                or self.snapshot_cloud_tolerance < 0.0
                or self.maximum_points < 1000
                or not math.isfinite(self.rotation_clearance)
                or self.rotation_clearance <= 0.0
                or not math.isfinite(self.rotation_acceptance_tolerance)
                or self.rotation_acceptance_tolerance < 0.0
                or not math.isfinite(self.minimum_reverse_distance)
                or self.minimum_reverse_distance < 0.0
                or self.forecast_candidate_history_size < 2):
            raise ValueError("invalid point-cloud observation window")
        self.cloud_frames = 0
        self.cloud_chunks = deque(maxlen=self.cloud_window_frames)
        self.point_count = 0
        self.odom = None
        self.avoidance = None
        self.evolution = None
        self.base_map = None
        self.mission_home = None
        self.mission_state = ""
        self.capture_cutoff_frame_index = None
        self.pending_forecast_avoidance = {}
        self.pending_forecast_evolution = {}
        self.forecast_candidates = deque(
            maxlen=self.forecast_candidate_history_size)
        self.finished = False

        rospy.Subscriber(rospy.get_param(prefix + "cloud_topic", "/cloud_registered"),
                         PointCloud2, self._cloud_callback, queue_size=1)
        rospy.Subscriber(rospy.get_param(prefix + "odometry_topic", "/Odometry"),
                         Odometry, self._odom_callback, queue_size=2)
        rospy.Subscriber(rospy.get_param(
            prefix + "avoidance_topic", "/deform/avoidance/frame"),
            DeformationAvoidanceMap, self._avoidance_callback, queue_size=2)
        rospy.Subscriber(rospy.get_param(
            prefix + "evolution_topic", "/deform/evolution/frame"),
            DeformationEvolutionFrame, self._evolution_callback, queue_size=2)
        rospy.Subscriber(rospy.get_param(
            prefix + "base_map_topic", "/map"),
            OccupancyGrid, self._base_map_callback, queue_size=1)
        rospy.Subscriber(rospy.get_param(
            prefix + "mission_status_topic", "/deform/mission/status"),
            String, self._mission_status_callback, queue_size=2)
        self.base_map_publisher = rospy.Publisher(
            "/deform/evaluation/slam_local_map", OccupancyGrid,
            queue_size=1, latch=True)
        self.fused_map_publisher = rospy.Publisher(
            "/deform/evaluation/fused_cost_map", OccupancyGrid,
            queue_size=1, latch=True)
        self.baseline_path_publisher = rospy.Publisher(
            "/deform/evaluation/baseline_path", Path, queue_size=1, latch=True)
        self.avoidance_path_publisher = rospy.Publisher(
            "/deform/evaluation/avoidance_path", Path, queue_size=1, latch=True)
        self.completion_publisher = rospy.Publisher(
            "/deform/evaluation/status", String, queue_size=1, latch=True)
        self.timer = rospy.Timer(rospy.Duration(0.5), self._try_export)
        rospy.loginfo("[planning_evaluation] waits for point cloud and a complete avoidance snapshot")

    def _cloud_callback(self, message):
        if self.finished or message.header.frame_id != self.world_frame:
            return
        values = []
        for index, point in enumerate(point_cloud2.read_points(
                message, field_names=("x", "y", "z"), skip_nans=True)):
            if index % max(1, self.cloud_stride) != 0:
                continue
            if all(math.isfinite(value) for value in point):
                values.append(point)
        if values:
            array = np.asarray(values, dtype=np.float32)
            per_frame_limit = max(
                1, self.maximum_points // self.cloud_window_frames)
            if len(array) > per_frame_limit:
                stride = int(math.ceil(len(array) / float(per_frame_limit)))
                array = array[::stride][:per_frame_limit]
            self.cloud_chunks.append((message.header.stamp.to_sec(), array))
            self.point_count = sum(len(entry[1]) for entry in self.cloud_chunks)
        self.cloud_frames += 1

    def _odom_callback(self, message):
        if not self.finished and message.header.frame_id == self.world_frame:
            self.odom = message

    def _avoidance_callback(self, message):
        if not self.finished and message.header.frame_id == self.world_frame:
            self.avoidance = message
            self._cache_forecast_message("avoidance", message)

    def _evolution_callback(self, message):
        if (not self.finished and message.header.frame_id == self.world_frame
                and message.complete_snapshot and message.input_contract_valid):
            self.evolution = message
            self._cache_forecast_message("evolution", message)

    def _base_map_callback(self, message):
        if (not self.finished and message.header.frame_id == self.world_frame
                and len(message.data) == message.info.width * message.info.height):
            self.base_map = message

    def _mission_status_callback(self, message):
        if self.finished:
            return
        try:
            document = json.loads(message.data)
            self.mission_state = str(document.get("state", ""))
            home = document.get("home_map_xy")
            if (isinstance(home, list) and len(home) == 2
                    and all(math.isfinite(float(value)) for value in home)):
                self.mission_home = (float(home[0]), float(home[1]))
            cutoff = document.get("capture_cutoff", {})
            if isinstance(cutoff, dict):
                frame_index = cutoff.get("accepted_frame_index")
                if frame_index is not None:
                    frame_index = int(frame_index)
                    if frame_index >= 0:
                        self.capture_cutoff_frame_index = frame_index
        except (TypeError, ValueError):
            rospy.logwarn_throttle(
                2.0, "[planning_evaluation] invalid mission status JSON")

    @staticmethod
    def _grid_contract_valid(grid, source_stamp_s=None):
        try:
            width = int(grid.info.width)
            height = int(grid.info.height)
            resolution = float(grid.info.resolution)
            origin = grid.info.origin.position
            stamp_s = float(grid.header.stamp.to_sec())
            return bool(
                width > 0 and height > 0
                and len(grid.data) == width * height
                and math.isfinite(resolution) and resolution > 0.0
                and all(math.isfinite(float(value))
                        for value in (origin.x, origin.y, origin.z, stamp_s))
                and (source_stamp_s is None
                     or abs(stamp_s - float(source_stamp_s)) <= 1.0e-9))
        except (AttributeError, TypeError, ValueError, OverflowError):
            return False

    def _cache_forecast_message(self, kind, message):
        snapshot_id = str(getattr(message, "source_snapshot_id", ""))
        if not snapshot_id:
            return
        target = (self.pending_forecast_avoidance if kind == "avoidance"
                  else self.pending_forecast_evolution)
        target[snapshot_id] = message
        while len(target) > 2 * self.forecast_candidate_history_size:
            del target[next(iter(target))]
        avoidance = self.pending_forecast_avoidance.get(snapshot_id)
        evolution = self.pending_forecast_evolution.get(snapshot_id)
        if avoidance is None or evolution is None:
            return
        if not snapshot_identity_matches(
                avoidance.source_snapshot_id,
                avoidance.source_accepted_frame_index,
                avoidance.header.stamp.to_sec(),
                evolution.source_snapshot_id,
                evolution.source_accepted_frame_index,
                evolution.header.stamp.to_sec()):
            return
        if not prediction_window_covers_horizons(
                evolution.header.stamp.to_sec(),
                avoidance.prediction_valid_until.to_sec(),
                evolution.prediction_valid_until.to_sec(),
                self.expected_forecast_horizons):
            return
        if (not evolution.complete_snapshot
                or not evolution.input_contract_valid
                or not avoidance.all_current_sources_within_map
                or not avoidance.all_forecast_sources_within_map
                or not self._grid_contract_valid(
                    avoidance.current_overlay,
                    avoidance.header.stamp.to_sec())
                or not complete_forecast_horizons(
                    avoidance.forecast_horizons_s,
                    len(avoidance.forecast_overlays),
                    self.expected_forecast_horizons)
                or any(not self._grid_contract_valid(
                    grid, avoidance.header.stamp.to_sec())
                       for grid in avoidance.forecast_overlays)):
            return
        valid_evolution_horizons = []
        for gdu in evolution.gdus:
            if not (gdu.active_now or gdu.retained_for_avoidance):
                continue
            valid_evolution_horizons.extend(
                forecast.horizon_s for forecast in gdu.forecasts
                if forecast.valid and forecast.predicted_support)
        if not all(any(abs(float(actual) - float(expected)) <= 1.0e-9
                           for actual in valid_evolution_horizons)
                   for expected in self.expected_forecast_horizons):
            return
        candidate = {
            "source_snapshot_id": snapshot_id,
            "source_accepted_frame_index": int(
                evolution.source_accepted_frame_index),
            "source_stamp_s": float(evolution.header.stamp.to_sec()),
            "avoidance": avoidance,
            "evolution": evolution,
        }
        self.forecast_candidates = deque(
            (entry for entry in self.forecast_candidates
             if entry["source_snapshot_id"] != snapshot_id),
            maxlen=self.forecast_candidate_history_size)
        self.forecast_candidates.append(candidate)
        self.pending_forecast_avoidance.pop(snapshot_id, None)
        self.pending_forecast_evolution.pop(snapshot_id, None)

    def _selected_forecast_candidate(self):
        current_index = int(self.evolution.source_accepted_frame_index)
        if self.use_latest_complete_forecast_before_cutoff:
            if self.capture_cutoff_frame_index is None:
                return None
            cutoff = min(current_index, self.capture_cutoff_frame_index)
        else:
            cutoff = current_index
        candidate = select_latest_forecast_candidate(
            self.forecast_candidates, cutoff)
        if (candidate is None
                or (not self.use_latest_complete_forecast_before_cutoff
                    and int(candidate["source_accepted_frame_index"])
                    != current_index)):
            return None
        return candidate

    def _snapshot_cloud_entries(self, source_stamp_s=None):
        if self.evolution is None and source_stamp_s is None:
            return []
        source_stamp = (self.evolution.header.stamp.to_sec()
                        if source_stamp_s is None
                        else float(source_stamp_s))
        earliest = source_stamp - self.cloud_window_duration
        latest = source_stamp + self.snapshot_cloud_tolerance
        entries = [entry for entry in self.cloud_chunks
                   if earliest <= entry[0] <= latest]
        if (not entries or min(abs(entry[0] - source_stamp)
                               for entry in entries)
                > self.snapshot_cloud_tolerance):
            return []
        return entries

    def _ready(self):
        if (self.finished or self.odom is None or self.avoidance is None
                or self.evolution is None
                or len(self.cloud_chunks) < self.minimum_cloud_frames):
            return False
        if (self.export_mission_evacuation
                and (self.base_map is None or self.mission_home is None)):
            return False
        if (self.require_detection_state
                and self.mission_state not in (
                    "DEFORMATION_DETECTED", "MONITORING_COMPLETE",
                    "RETURNING_HOME", "COMPLETED")):
            return False
        if (self.use_latest_complete_forecast_before_cutoff
                and self.capture_cutoff_frame_index is None):
            return False
        if (self.capture_cutoff_frame_index is not None
                and int(self.evolution.source_accepted_frame_index)
                < self.capture_cutoff_frame_index):
            # MONITORING_COMPLETE and the map topics are asynchronous.  Do not
            # mislabel a lagging pre-cutoff frame as the final current map.
            return False
        cloud_entries = self._snapshot_cloud_entries()
        if (len(cloud_entries) < self.minimum_cloud_frames
                or sum(len(entry[1]) for entry in cloud_entries) < 1000):
            return False
        if not self.avoidance.all_current_sources_within_map:
            rospy.logwarn_throttle(
                2.0, "[planning_evaluation] exporting bounded map with clipped "
                "source halos; coverage=false and source IDs are recorded")
        if self.require_active_source and not self.avoidance.active_source_ids:
            return False
        now = rospy.Time.now()
        if (not self.avoidance.valid_until.is_zero()
                and now > self.avoidance.valid_until):
            return False
        return snapshot_identity_matches(
            self.avoidance.source_snapshot_id,
            self.avoidance.source_accepted_frame_index,
            self.avoidance.header.stamp.to_sec(),
            self.evolution.source_snapshot_id,
            self.evolution.source_accepted_frame_index,
            self.evolution.header.stamp.to_sec())

    def _try_export(self, _event):
        if not self._ready():
            return
        self.finished = True
        try:
            # Callbacks stop accepting new inputs once finished is set. Let
            # navigation depart after capture, not after plotting and disk I/O.
            capture_time_s = rospy.Time.now().to_sec()
            self.completion_publisher.publish(String(data=json.dumps({
                "capture_ready": True, "success": False,
                "capture_time_s": capture_time_s,
                "source_snapshot_id": self.evolution.source_snapshot_id,
                "source_accepted_frame_index": int(self.evolution.source_accepted_frame_index),
            }, sort_keys=True)))
            output = self._export()
            rospy.logwarn("[planning_evaluation] export completed: %s", output)
            with open(os.path.join(output, "metrics.json"), "r",
                      encoding="utf-8") as stream:
                exported_metrics = json.load(stream)
            status = {
                "capture_ready": True,
                "capture_time_s": capture_time_s,
                "export_complete_time_s": rospy.Time.now().to_sec(),
                "success": True,
                "output_directory": output,
                "source_snapshot_id": self.evolution.source_snapshot_id,
                "source_accepted_frame_index": int(
                    self.evolution.source_accepted_frame_index),
            }
            if "mission_evacuation" in exported_metrics:
                status["mission_evacuation"] = exported_metrics[
                    "mission_evacuation"]
            self.completion_publisher.publish(String(data=json.dumps(
                status, sort_keys=True)))
        except Exception as error:  # keep the main experiment alive on exporter failure
            self.finished = False
            rospy.logerr("[planning_evaluation] export failed: %s", error)

    def _grid_message(self, header, geometry, data):
        message = OccupancyGrid()
        message.header = header
        message.header.frame_id = self.world_frame
        message.info = MapMetaData()
        message.info.map_load_time = header.stamp
        message.info.resolution = geometry.resolution
        message.info.width = geometry.width
        message.info.height = geometry.height
        message.info.origin.position.x = geometry.origin_x
        message.info.origin.position.y = geometry.origin_y
        message.info.origin.orientation.w = 1.0
        message.data = np.asarray(data, dtype=np.int16).reshape(-1).tolist()
        return message

    def _path_message(self, header, geometry, cells):
        message = Path()
        message.header = header
        message.header.frame_id = self.world_frame
        for cell in cells:
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x, pose.pose.position.y = geometry.cell_to_world(cell)
            pose.pose.orientation.w = 1.0
            message.poses.append(pose)
        return message

    @staticmethod
    def _extent(geometry):
        return [geometry.origin_x,
                geometry.origin_x + geometry.width * geometry.resolution,
                geometry.origin_y,
                geometry.origin_y + geometry.height * geometry.resolution]

    @staticmethod
    def _path_xy(geometry, path):
        values = [geometry.cell_to_world(cell) for cell in path]
        return ([value[0] for value in values], [value[1] for value in values])

    def _plot_base(self, axis, occupancy, geometry, title):
        encoded = np.where(occupancy < 0, 1, np.where(occupancy >= 100, 2, 0))
        axis.imshow(encoded, origin="lower", extent=self._extent(geometry),
                    interpolation="nearest",
                    cmap=ListedColormap(["white", "#b8bec9", "#171717"]),
                    vmin=0, vmax=2)
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")

    def _plot_overlay(self, axis, overlay, geometry, title):
        image = axis.imshow(overlay, origin="lower", extent=self._extent(geometry),
                            interpolation="nearest", cmap="inferno",
                            vmin=0, vmax=99)
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        return image

    def _plot_fused(self, axis, occupancy, overlay, geometry, title):
        """Render UNKNOWN, SLAM obstacles and deformation as distinct data."""
        occupancy = np.asarray(occupancy)
        overlay = np.asarray(overlay)
        rgba = np.ones(occupancy.shape + (4,), dtype=float)
        rgba[occupancy < 0] = (0.72, 0.75, 0.79, 1.0)
        rgba[occupancy >= 100] = (0.09, 0.09, 0.09, 1.0)
        deformation_cells = ((occupancy >= 0) & (occupancy < 100)
                             & (overlay > 0))
        colors = plt.get_cmap("inferno")(
            np.clip(overlay.astype(float) / 100.0, 0.0, 1.0))
        rgba[deformation_cells] = colors[deformation_cells]
        axis.imshow(rgba, origin="lower", extent=self._extent(geometry),
                    interpolation="nearest")
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")

    @staticmethod
    def _save_figure(figure, path):
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)

    @staticmethod
    def _write_pgm_yaml(directory, name, data, geometry):
        values = np.asarray(data)
        pixels = np.where(values < 0, 205,
                          np.where(values >= 100, 0,
                                   np.clip(254 - values * 2.54, 1, 254)))
        pixels = np.flipud(pixels.astype(np.uint8))
        pgm_path = os.path.join(directory, name + ".pgm")
        with open(pgm_path, "wb") as stream:
            stream.write(("P5\n{} {}\n255\n".format(
                geometry.width, geometry.height)).encode("ascii"))
            stream.write(pixels.tobytes())
        yaml_path = os.path.join(directory, name + ".yaml")
        with open(yaml_path, "w", encoding="utf-8") as stream:
            stream.write("image: {}.pgm\n".format(name))
            stream.write("resolution: {:.12g}\n".format(geometry.resolution))
            stream.write("origin: [{:.12g}, {:.12g}, 0.0]\n".format(
                geometry.origin_x, geometry.origin_y))
            stream.write("negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")

    @staticmethod
    def _binary_iou(first, second, threshold):
        first_mask = np.asarray(first) >= threshold
        second_mask = np.asarray(second) >= threshold
        union = np.logical_or(first_mask, second_mask).sum()
        if union == 0:
            return 1.0
        return float(np.logical_and(first_mask, second_mask).sum()) / float(union)

    def _export(self):
        stamp = self.evolution.header.stamp
        forecast_candidate = self._selected_forecast_candidate()
        forecast_evolution = (forecast_candidate["evolution"]
                              if forecast_candidate is not None else None)
        forecast_avoidance = (forecast_candidate["avoidance"]
                              if forecast_candidate is not None else None)
        start_x = float(self.odom.pose.pose.position.x)
        start_y = float(self.odom.pose.pose.position.y)
        geometry = GridGeometry.centered(
            start_x, start_y, self.extent_m, self.resolution)
        cloud_entries = self._snapshot_cloud_entries()
        if len(cloud_entries) < self.minimum_cloud_frames:
            raise RuntimeError("point-cloud window no longer matches source snapshot")
        cloud_stamps = [entry[0] for entry in cloud_entries]
        points = np.concatenate(
            [entry[1] for entry in cloud_entries], axis=0).astype(float)
        forecast_cloud_entries = []
        if forecast_candidate is not None:
            forecast_cloud_entries = self._snapshot_cloud_entries(
                forecast_candidate["source_stamp_s"])
        if forecast_candidate is None:
            forecast_context_points = points
        elif (len(forecast_cloud_entries) >= self.minimum_cloud_frames
              and sum(len(entry[1]) for entry in forecast_cloud_entries)
              >= 1000):
            forecast_context_points = np.concatenate(
                [entry[1] for entry in forecast_cloud_entries], axis=0).astype(float)
        else:
            # Never relabel the final-current point window as the historical
            # forecast source.  The structural supports remain fully usable.
            forecast_context_points = np.empty((0, 3), dtype=float)
        floor_z = estimate_floor_z(points, (start_x, start_y))
        occupancy, free_count, obstacle_count = lidar_observation_grid(
            points, (start_x, start_y), geometry, floor_z,
            maximum_range=self.maximum_range)
        occupancy = inflate_obstacles(
            occupancy, self.robot_radius, geometry.resolution)
        grid = self.avoidance.current_overlay
        current_overlay = crop_occupancy_grid(
            grid.data, grid.info.resolution,
            (grid.info.origin.position.x, grid.info.origin.position.y),
            grid.info.width, grid.info.height, geometry)
        if int(np.max(current_overlay)) <= 0:
            raise RuntimeError(
                "complete deformation snapshot has no support in the local planning grid")
        forecast_anchor_overlay = None
        forecast_overlays = {}
        if forecast_avoidance is not None:
            anchor = forecast_avoidance.current_overlay
            forecast_anchor_overlay = crop_occupancy_grid(
                anchor.data, anchor.info.resolution,
                (anchor.info.origin.position.x,
                 anchor.info.origin.position.y),
                anchor.info.width, anchor.info.height, geometry)
            for expected in self.expected_forecast_horizons:
                match = next(
                    (forecast for horizon, forecast in zip(
                        forecast_avoidance.forecast_horizons_s,
                        forecast_avoidance.forecast_overlays)
                     if abs(float(horizon) - float(expected)) <= 1.0e-9),
                    None)
                if match is not None:
                    forecast_overlays[float(expected)] = crop_occupancy_grid(
                        match.data, match.info.resolution,
                        (match.info.origin.position.x,
                         match.info.origin.position.y),
                        match.info.width, match.info.height, geometry)

        requested_start = geometry.world_to_cell(start_x, start_y)
        start = nearest_traversable(occupancy, requested_start)
        planning_overlay = np.minimum(current_overlay, 99)
        local_start_clearance = False  # Legacy metric: soft egress never clears data.
        comparison = choose_comparison_goal(
            occupancy, planning_overlay, start, geometry,
            self.minimum_goal_radius, self.maximum_goal_radius)
        if comparison is None:
            raise RuntimeError("no pair of nearby paths exists in the observed local map")
        local_paths_differ = (
            comparison["baseline_path"] != comparison["avoidance_path"])
        local_baseline_exposed = comparison["baseline_overlay_max"] > 0.0
        local_route_affected = deformation_changes_route(
            local_paths_differ, comparison["baseline_overlay_max"])
        if self.require_local_path_difference and not local_route_affected:
            raise RuntimeError(
                "no nearby goal exposes a deformation-induced path difference yet")
        goal = comparison["goal"]
        baseline_path = comparison["baseline_path"]
        avoidance_path = comparison["avoidance_path"]
        comparison["baseline_overlay_max"] = max(
            float(current_overlay[row, column])
            for column, row in baseline_path)
        fused_cost = fuse_planning_layers(occupancy, current_overlay)

        mission_comparison = None
        if self.export_mission_evacuation:
            base = self.base_map
            global_geometry = GridGeometry(
                base.info.resolution, base.info.width, base.info.height,
                base.info.origin.position.x, base.info.origin.position.y)
            global_occupancy = np.asarray(base.data, dtype=np.int16).reshape(
                base.info.height, base.info.width)
            global_occupancy = inflate_obstacles(
                global_occupancy, self.robot_radius, global_geometry.resolution)
            global_overlay = crop_occupancy_grid(
                grid.data, grid.info.resolution,
                (grid.info.origin.position.x, grid.info.origin.position.y),
                grid.info.width, grid.info.height, global_geometry)
            requested_global_start = global_geometry.world_to_cell(
                start_x, start_y)
            requested_global_goal = global_geometry.world_to_cell(
                self.mission_home[0], self.mission_home[1])
            if requested_global_start is None or requested_global_goal is None:
                raise RuntimeError(
                    "mission start or recorded home lies outside the SLAM map")
            global_start = nearest_traversable(
                global_occupancy, requested_global_start,
                unknown_is_traversable=False)
            global_goal = nearest_traversable(
                global_occupancy, requested_global_goal,
                unknown_is_traversable=False)
            planning_global_overlay = np.minimum(global_overlay, 99)
            global_start_clearance = False
            planning_global_fused = fuse_planning_layers(
                global_occupancy, planning_global_overlay)
            baseline_global, baseline_global_cost = astar(
                global_occupancy, global_start, global_goal,
                unknown_is_traversable=False)
            avoidance_global, avoidance_global_cost = astar(
                global_occupancy, global_start, global_goal,
                overlay=planning_global_overlay,
                unknown_is_traversable=False)
            if not baseline_global or not avoidance_global:
                raise RuntimeError(
                    "no known-free baseline/avoidance route connects target to home")
            rotation_transition = first_rotation_clearance_index(
                planning_global_fused, avoidance_global,
                global_geometry.resolution, self.minimum_reverse_distance,
                self.rotation_clearance
                + self.rotation_acceptance_tolerance)
            # Diagnostic for the optional reverse-turn controller, not a
            # prerequisite for normal forward move_base evacuation.
            rotation_index, rotation_distance = (
                rotation_transition if rotation_transition is not None else (None, None))
            global_baseline_max = max(
                int(global_overlay[row, column])
                for column, row in baseline_global)
            mission_comparison = {
                "geometry": global_geometry,
                "occupancy": global_occupancy,
                "overlay": global_overlay,
                "fused": fuse_planning_layers(
                    global_occupancy, global_overlay),
                "start": global_start,
                "goal": global_goal,
                "baseline_path": baseline_global,
                "avoidance_path": avoidance_global,
                "baseline_cost": baseline_global_cost,
                "avoidance_cost": avoidance_global_cost,
                "baseline_max_deformation_cost": global_baseline_max,
                "initial_start_footprint_clearance_applied":
                    global_start_clearance,
                "rotation_transition_index": rotation_index,
                "rotation_transition_distance_m": rotation_distance,
                "rotation_clearance_m": self.rotation_clearance,
                "rotation_selection_clearance_m": (
                    self.rotation_clearance
                    + self.rotation_acceptance_tolerance),
                "rotation_acceptance_tolerance_m": (
                    self.rotation_acceptance_tolerance),
            }

        run_name = self.run_name or time.strftime("static_planning_%Y%m%d_%H%M%S")
        directory = os.path.join(self.output_root, run_name)
        os.makedirs(directory, exist_ok=False)
        extent = self._extent(geometry)

        # 1. Registered LiDAR top view, height relative to the estimated floor.
        figure, axis = plt.subplots(figsize=(7, 6))
        display_points = points[::max(1, len(points) // 70000)]
        scatter = axis.scatter(display_points[:, 0], display_points[:, 1],
                               c=display_points[:, 2] - floor_z,
                               s=0.35, cmap="viridis", vmin=0.0, vmax=2.0)
        axis.plot(start_x, start_y, "b^", markersize=8, label="stationary robot")
        axis.set(xlim=extent[:2], ylim=extent[2:], aspect="equal",
                 title="Registered LiDAR data (top view)", xlabel="x [m]", ylabel="y [m]")
        axis.legend(loc="upper right")
        figure.colorbar(scatter, ax=axis, label="height above floor [m]")
        self._save_figure(figure, os.path.join(directory, "01_lidar_top_view.png"))

        figure, axis = plt.subplots(figsize=(7, 6))
        self._plot_base(axis, occupancy, geometry,
                        "LiDAR-SLAM derived local occupancy")
        self._save_figure(figure, os.path.join(directory, "02_slam_local_occupancy.png"))

        figure, axis = plt.subplots(figsize=(7, 6))
        image = self._plot_overlay(axis, current_overlay, geometry,
                                   "Current deformation avoidance layer")
        figure.colorbar(image, ax=axis, label="soft planning cost [0,99]")
        self._save_figure(figure, os.path.join(directory, "03_deformation_current_overlay.png"))

        if forecast_anchor_overlay is not None:
            figure, axis = plt.subplots(figsize=(7, 6))
            image = self._plot_overlay(
                axis, forecast_anchor_overlay, geometry,
                "Runtime forecast anchor (frame {})".format(
                    forecast_candidate["source_accepted_frame_index"]))
            figure.colorbar(image, ax=axis, label="soft planning cost [0,99]")
            self._save_figure(
                figure, os.path.join(
                    directory, "03_forecast_anchor_current_overlay.png"))

        for index, horizon in enumerate(self.expected_forecast_horizons, 4):
            figure, axis = plt.subplots(figsize=(7, 6))
            forecast = forecast_overlays.get(float(horizon))
            if forecast is not None:
                image = self._plot_overlay(
                    axis, forecast, geometry,
                    "Kinematic forecast overlay t+{:.1f} s".format(horizon))
                figure.colorbar(image, ax=axis, label="soft visualization cost [0,99]")
                suffix = ""
            else:
                self._plot_overlay(
                    axis, (forecast_anchor_overlay
                           if forecast_anchor_overlay is not None
                           else current_overlay), geometry,
                    "Kinematic forecast t+{:.1f} s: unavailable".format(horizon))
                axis.text(
                    0.5, 0.5,
                    "NO VALID KINEMATIC FORECAST\nvelocity observability/capability gate not satisfied",
                    transform=axis.transAxes, ha="center", va="center",
                    color="white", fontsize=10,
                    bbox={"facecolor": "#252525", "alpha": 0.85, "pad": 8})
                suffix = "_unavailable"
            self._save_figure(
                figure, os.path.join(directory,
                                     "{:02d}_forecast_{:.1f}s{}.png".format(
                                         index, horizon, suffix)))

        # 2-D evolution vectors preserve the fixed-reference/current distinction.
        figure, axis = plt.subplots(figsize=(7, 6))
        self._plot_base(axis, occupancy, geometry,
                        "Detected deformation vectors (display x50)")
        for gdu in self.evolution.gdus:
            if not (gdu.active_now or gdu.retained_for_avoidance):
                continue
            for reference, displacement in zip(
                    gdu.reference_support, gdu.displacement_samples):
                if not _finite_point(reference):
                    continue
                axis.arrow(reference.x, reference.y,
                           50.0 * displacement.x, 50.0 * displacement.y,
                           color="#e31a1c", width=0.008,
                           head_width=0.07, length_includes_head=True, alpha=0.8)
        self._save_figure(figure, os.path.join(directory, "07_deformation_vector_map.png"))

        def path_figure(path, title, filename, color):
            figure, axis = plt.subplots(figsize=(7, 6))
            self._plot_base(axis, occupancy, geometry, title)
            x_values, y_values = self._path_xy(geometry, path)
            axis.plot(x_values, y_values, color=color, linewidth=2.2)
            sx, sy = geometry.cell_to_world(start)
            gx, gy = geometry.cell_to_world(goal)
            axis.plot(sx, sy, "b^", markersize=8)
            axis.plot(gx, gy, "g*", markersize=12)
            self._save_figure(figure, os.path.join(directory, filename))

        path_figure(baseline_path, "Baseline path (SLAM geometry only)",
                    "08_path_baseline.png", "#2171b5")
        path_figure(avoidance_path, "Path with deformation avoidance layer",
                    "09_path_avoidance.png", "#31a354")

        figure, axis = plt.subplots(figsize=(8, 7))
        self._plot_overlay(axis, current_overlay, geometry,
                           "Path comparison on deformation cost")
        bx, by = self._path_xy(geometry, baseline_path)
        ax, ay = self._path_xy(geometry, avoidance_path)
        axis.plot(bx, by, color="#2b8cbe", linewidth=2.0,
                  label="SLAM-only baseline")
        axis.plot(ax, ay, color="#41ab5d", linewidth=2.5,
                  label="deformation-aware")
        sx, sy = geometry.cell_to_world(start)
        gx, gy = geometry.cell_to_world(goal)
        axis.plot(sx, sy, "b^", markersize=9, label="stationary start")
        axis.plot(gx, gy, "g*", markersize=13, label="selected nearby goal")
        axis.legend(loc="upper right")
        self._save_figure(figure, os.path.join(directory, "10_path_comparison.png"))

        # Keep reference/current/forecast geometry on one atomic runtime
        # forecast snapshot.  The final current frame remains in map 03 and 07.
        evolution_for_forecast_display = (
            forecast_evolution if forecast_evolution is not None
            else self.evolution)
        figure = plt.figure(figsize=(9, 7))
        axis = figure.add_subplot(111, projection="3d")
        if len(forecast_context_points):
            cloud3d = forecast_context_points[
                ::max(1, len(forecast_context_points) // 30000)]
            axis.scatter(cloud3d[:, 0], cloud3d[:, 1], cloud3d[:, 2],
                         s=0.25, c="#a9a9a9", alpha=0.20)
        for gdu in evolution_for_forecast_display.gdus:
            if not (gdu.active_now or gdu.retained_for_avoidance):
                continue
            for reference, current in zip(gdu.reference_support,
                                          gdu.current_support):
                axis.plot([reference.x, current.x], [reference.y, current.y],
                          [reference.z, current.z], color="#e31a1c", linewidth=1.0)
                axis.scatter([reference.x], [reference.y], [reference.z],
                             c="#1f78b4", s=5)
                axis.scatter([current.x], [current.y], [current.z],
                             c="#e31a1c", s=7)
            for forecast in gdu.forecasts:
                if forecast.valid:
                    axis.scatter([point.x for point in forecast.predicted_support],
                                 [point.y for point in forecast.predicted_support],
                                 [point.z for point in forecast.predicted_support],
                                 c="#984ea3", s=4, alpha=0.35)
        axis.set(xlim=extent[:2], ylim=extent[2:],
                 xlabel="x [m]", ylabel="y [m]", zlabel="z [m]",
                 title="3-D runtime forecast bundle (single source snapshot)")
        self._save_figure(figure, os.path.join(directory, "11_deformation_3d.png"))

        # Compact paper-style dashboard.
        figure, axes = plt.subplots(3, 3, figsize=(18, 16))
        self._plot_base(axes[0, 0], occupancy, geometry, "(a) SLAM local occupancy")
        self._plot_overlay(axes[0, 1], current_overlay, geometry,
                           "(b) current deformation cost")
        self._plot_fused(axes[0, 2], occupancy, current_overlay, geometry,
                         "(c) fused planning map")
        ordered_forecasts = [(horizon, forecast_overlays.get(float(horizon)))
                             for horizon in self.expected_forecast_horizons]
        for column in range(3):
            if column < len(ordered_forecasts):
                horizon, values = ordered_forecasts[column]
                if values is not None:
                    self._plot_overlay(axes[1, column], values, geometry,
                                       "({}) runtime forecast +{:.1f}s (frame {})".format(
                                           chr(ord('d') + column), horizon,
                                           forecast_candidate[
                                               "source_accepted_frame_index"]))
                else:
                    self._plot_overlay(
                        axes[1, column],
                        (forecast_anchor_overlay
                         if forecast_anchor_overlay is not None
                         else current_overlay), geometry,
                        "({}) forecast +{:.1f}s unavailable".format(
                            chr(ord('d') + column), horizon))
                    axes[1, column].text(
                        0.5, 0.5, "velocity capability gate not satisfied",
                        transform=axes[1, column].transAxes,
                        ha="center", va="center", color="white",
                        bbox={"facecolor": "#252525", "alpha": 0.85})
            else:
                axes[1, column].axis("off")
        self._plot_base(axes[2, 0], occupancy, geometry, "(g) SLAM-only path")
        axes[2, 0].plot(bx, by, color="#2b8cbe", linewidth=2.0)
        self._plot_overlay(axes[2, 1], current_overlay, geometry,
                           "(h) deformation-aware path")
        axes[2, 1].plot(ax, ay, color="#41ab5d", linewidth=2.2)
        self._plot_overlay(axes[2, 2], current_overlay, geometry,
                           "(i) path comparison")
        axes[2, 2].plot(bx, by, color="#2b8cbe", linewidth=1.8,
                        label="baseline")
        axes[2, 2].plot(ax, ay, color="#41ab5d", linewidth=2.2,
                        label="avoidance")
        axes[2, 2].legend(loc="upper right")
        self._save_figure(figure, os.path.join(directory, "12_complete_dashboard.png"))

        figure, axis = plt.subplots(figsize=(7, 6))
        self._plot_fused(axis, occupancy, current_overlay, geometry,
                         "Fused SLAM and deformation planning map")
        self._save_figure(
            figure, os.path.join(directory, "13_fused_planning_map.png"))

        forecast_self_consistency = None
        if forecast_candidate is not None and forecast_overlays:
            current_stamp_s = float(stamp.to_sec())
            forecast_source_stamp_s = float(
                forecast_candidate["source_stamp_s"])
            comparison_horizon = min(
                forecast_overlays,
                key=lambda value: abs(
                    forecast_source_stamp_s + value - current_stamp_s))
            predicted = forecast_overlays[comparison_horizon]
            alignment_error_s = abs(
                forecast_source_stamp_s + comparison_horizon
                - current_stamp_s)
            forecast_self_consistency = {
                "comparison_horizon_s": float(comparison_horizon),
                "target_current_snapshot_stamp_s": current_stamp_s,
                "prediction_target_stamp_s": (
                    forecast_source_stamp_s + comparison_horizon),
                "temporal_alignment_error_s": alignment_error_s,
                "positive_cost_iou": self._binary_iou(
                    predicted, current_overlay, 1),
                "high_cost_core_iou": self._binary_iou(
                    predicted, current_overlay, 99),
                "high_cost_core_threshold": 99,
                "deformation_cost_policy": "FINITE_SOFT_COST",
                "interpretation": (
                    "runtime forecast versus a later observed avoidance layer; "
                    "this is temporal self-consistency, not collapse probability"),
            }
            figure, axes = plt.subplots(1, 3, figsize=(18, 6))
            self._plot_overlay(
                axes[0], forecast_anchor_overlay, geometry,
                "Forecast anchor frame {}".format(
                    forecast_candidate["source_accepted_frame_index"]))
            self._plot_overlay(
                axes[1], predicted, geometry,
                "Runtime prediction +{:.1f}s".format(comparison_horizon))
            self._plot_overlay(
                axes[2], current_overlay, geometry,
                "Later current frame {} (|dt|={:.3f}s)".format(
                    self.evolution.source_accepted_frame_index,
                    alignment_error_s))
            self._save_figure(
                figure, os.path.join(
                    directory, "14_runtime_forecast_self_consistency.png"))

        if mission_comparison is not None:
            global_geometry = mission_comparison["geometry"]
            global_occupancy = mission_comparison["occupancy"]
            global_overlay = mission_comparison["overlay"]
            global_baseline = mission_comparison["baseline_path"]
            global_avoidance = mission_comparison["avoidance_path"]
            global_start = mission_comparison["start"]
            global_goal = mission_comparison["goal"]

            figure, axis = plt.subplots(figsize=(9, 8))
            self._plot_base(axis, global_occupancy, global_geometry,
                            "Full SLAM map: target-to-home evacuation")
            bx_global, by_global = self._path_xy(
                global_geometry, global_baseline)
            ax_global, ay_global = self._path_xy(
                global_geometry, global_avoidance)
            axis.plot(bx_global, by_global, color="#2b8cbe", linewidth=2.0,
                      label="SLAM-only baseline")
            axis.plot(ax_global, ay_global, color="#31a354", linewidth=2.5,
                      label="deformation-aware evacuation")
            start_global_xy = global_geometry.cell_to_world(global_start)
            goal_global_xy = global_geometry.cell_to_world(global_goal)
            axis.plot(*start_global_xy, "r^", markersize=9,
                      label="detection position")
            axis.plot(*goal_global_xy, "g*", markersize=13,
                      label="recorded home")
            axis.legend(loc="best")
            self._save_figure(
                figure, os.path.join(directory,
                                     "14_global_evacuation_comparison.png"))

            figure, axis = plt.subplots(figsize=(9, 8))
            self._plot_fused(
                axis, global_occupancy, global_overlay, global_geometry,
                "Full SLAM + deformation evacuation map")
            axis.plot(ax_global, ay_global, color="#31a354", linewidth=2.5)
            axis.plot(*start_global_xy, "r^", markersize=9)
            axis.plot(*goal_global_xy, "g*", markersize=13)
            self._save_figure(
                figure, os.path.join(directory,
                                     "15_global_fused_evacuation_map.png"))

            self._write_pgm_yaml(
                directory, "global_slam_map", global_occupancy,
                global_geometry)
            self._write_pgm_yaml(
                directory, "global_deformation_overlay", global_overlay,
                global_geometry)
            self._write_pgm_yaml(
                directory, "global_fused_evacuation_map",
                mission_comparison["fused"], global_geometry)
            for filename, path in (
                    ("global_baseline_path.csv", global_baseline),
                    ("global_avoidance_path.csv", global_avoidance)):
                with open(os.path.join(directory, filename), "w", newline="",
                          encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["index", "x_m", "y_m"])
                    for index, cell in enumerate(path):
                        x, y = global_geometry.cell_to_world(cell)
                        writer.writerow([index, "{:.9f}".format(x),
                                         "{:.9f}".format(y)])

        self._write_pgm_yaml(directory, "slam_local_map", occupancy, geometry)
        self._write_pgm_yaml(directory, "deformation_current_overlay",
                             current_overlay, geometry)
        self._write_pgm_yaml(directory, "fused_planning_map", fused_cost, geometry)
        if forecast_anchor_overlay is not None:
            self._write_pgm_yaml(
                directory, "forecast_anchor_current_overlay",
                forecast_anchor_overlay, geometry)
        for horizon, values in sorted(forecast_overlays.items()):
            self._write_pgm_yaml(directory,
                                 "forecast_{:.1f}s".format(horizon),
                                 values, geometry)
        map_arrays = {
            "occupancy": occupancy,
            "current_overlay": current_overlay,
            "fused_cost": fused_cost,
            "free_count": free_count,
            "obstacle_count": obstacle_count,
        }
        if forecast_anchor_overlay is not None:
            map_arrays["forecast_anchor_current_overlay"] = (
                forecast_anchor_overlay)
        map_arrays.update({
            "forecast_{:.1f}s".format(key): value
            for key, value in forecast_overlays.items()})
        np.savez_compressed(
            os.path.join(directory, "all_map_arrays.npz"), **map_arrays)

        for filename, path in (("baseline_path.csv", baseline_path),
                               ("avoidance_path.csv", avoidance_path)):
            with open(os.path.join(directory, filename), "w", newline="",
                      encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["index", "x_m", "y_m"])
                for index, cell in enumerate(path):
                    x, y = geometry.cell_to_world(cell)
                    writer.writerow([index, "{:.9f}".format(x),
                                     "{:.9f}".format(y)])

        baseline_length = path_length_m(baseline_path, geometry.resolution)
        avoidance_length = path_length_m(avoidance_path, geometry.resolution)
        baseline_max_cost = max(int(current_overlay[row, column])
                                for column, row in baseline_path)
        forecast_prediction_valid_until_s = None
        if (forecast_avoidance is not None
                and not forecast_avoidance.prediction_valid_until.is_zero()):
            forecast_prediction_valid_until_s = float(
                forecast_avoidance.prediction_valid_until.to_sec())
        forecast_source_stamp_s = (
            float(forecast_candidate["source_stamp_s"])
            if forecast_candidate is not None else None)
        metrics = {
            "source_snapshot_id": self.evolution.source_snapshot_id,
            "source_accepted_frame_index": int(
                self.evolution.source_accepted_frame_index),
            "world_frame": self.world_frame,
            "robot_moved_by_evaluator": False,
            "semantic_recognition_used": False,
            "rosbag_recorded": False,
            "floor_z_m": floor_z,
            "point_count": int(len(points)),
            "cloud_frames_received_total": int(self.cloud_frames),
            "cloud_frames_in_snapshot_window": int(len(cloud_entries)),
            "cloud_window_start_stamp_s": min(cloud_stamps),
            "cloud_window_end_stamp_s": max(cloud_stamps),
            "deformation_snapshot_stamp_s": stamp.to_sec(),
            "active_source_ids": list(self.avoidance.active_source_ids),
            "current_snapshot_id": self.evolution.source_snapshot_id,
            "current_snapshot_accepted_frame_index": int(
                self.evolution.source_accepted_frame_index),
            "current_snapshot_stamp_s": float(stamp.to_sec()),
            "forecast_selection_policy": (
                "LATEST_COMPLETE_PAIRED_SNAPSHOT_AT_OR_BEFORE_CAPTURE_CUTOFF"
                if self.use_latest_complete_forecast_before_cutoff
                else "CURRENT_SNAPSHOT_ONLY"),
            "forecast_capture_cutoff_frame_index": (
                self.capture_cutoff_frame_index),
            "forecast_source_snapshot_id": (
                forecast_candidate["source_snapshot_id"]
                if forecast_candidate is not None else None),
            "forecast_source_accepted_frame_index": (
                int(forecast_candidate["source_accepted_frame_index"])
                if forecast_candidate is not None else None),
            "forecast_source_stamp_s": forecast_source_stamp_s,
            "forecast_snapshot_age_to_current_s": (
                float(stamp.to_sec()) - forecast_source_stamp_s
                if forecast_source_stamp_s is not None else None),
            "forecast_prediction_valid_until_s": (
                forecast_prediction_valid_until_s),
            "forecast_prediction_reaches_current_snapshot": bool(
                forecast_prediction_valid_until_s is not None
                and forecast_prediction_valid_until_s
                >= float(stamp.to_sec())),
            "forecast_source_active_ids": (
                list(forecast_avoidance.active_source_ids)
                if forecast_avoidance is not None else []),
            "forecast_3d_source_aligned_cloud_available": bool(
                len(forecast_context_points)),
            "forecast_3d_source_aligned_cloud_frame_count": int(
                len(forecast_cloud_entries)
                if forecast_candidate is not None else len(cloud_entries)),
            "start_world_m": list(geometry.cell_to_world(start)),
            "goal_world_m": list(geometry.cell_to_world(goal)),
            "baseline_length_m": baseline_length,
            "avoidance_length_m": avoidance_length,
            "length_increase_m": avoidance_length - baseline_length,
            "baseline_max_deformation_cost": baseline_max_cost,
            "baseline_overlay_max_from_selection": comparison[
                "baseline_overlay_max"],
            "local_paths_differ": local_paths_differ,
            "local_baseline_exposed_to_deformation": local_baseline_exposed,
            "local_route_affected_by_deformation": local_route_affected,
            "local_path_difference_required": bool(
                self.require_local_path_difference),
            "initial_start_footprint_clearance_applied":
                local_start_clearance,
            "baseline_weighted_cost": comparison["baseline_cost"],
            "avoidance_weighted_cost": comparison["avoidance_cost"],
            "goal_selection_score": comparison["score"],
            "current_map_complete": bool(
                self.avoidance.all_current_sources_within_map),
            "out_of_bounds_current_source_ids": list(
                self.avoidance.out_of_bounds_current_source_ids),
            "forecast_map_complete": bool(
                forecast_avoidance is not None
                and forecast_avoidance.all_forecast_sources_within_map),
            "current_snapshot_forecast_map_complete": bool(
                self.avoidance.all_forecast_sources_within_map),
            "forecast_is_visualization_only": bool(
                forecast_avoidance.forecast_visualization_only
                if forecast_avoidance is not None
                else self.avoidance.forecast_visualization_only),
            "expected_forecast_horizons_s": self.expected_forecast_horizons,
            "available_forecast_horizons_s": sorted(forecast_overlays),
            "unavailable_forecast_horizons_s": [
                value for value in self.expected_forecast_horizons
                if float(value) not in forecast_overlays],
            "forecast_self_consistency": forecast_self_consistency,
        }
        if mission_comparison is not None:
            global_geometry = mission_comparison["geometry"]
            global_baseline = mission_comparison["baseline_path"]
            global_avoidance = mission_comparison["avoidance_path"]
            metrics["mission_evacuation"] = {
                "deformation_cost_policy": "FINITE_SOFT_COST",
                "route_difference_required": False,
                "goal_is_recorded_home": True,
                "unknown_is_traversable": False,
                "start_world_m": list(global_geometry.cell_to_world(
                    mission_comparison["start"])),
                "home_world_m": list(global_geometry.cell_to_world(
                    mission_comparison["goal"])),
                "baseline_length_m": path_length_m(
                    global_baseline, global_geometry.resolution),
                "avoidance_length_m": path_length_m(
                    global_avoidance, global_geometry.resolution),
                "baseline_weighted_cost": mission_comparison["baseline_cost"],
                "avoidance_weighted_cost": mission_comparison["avoidance_cost"],
                "baseline_max_deformation_cost": mission_comparison[
                    "baseline_max_deformation_cost"],
                "paths_differ": global_baseline != global_avoidance,
                "initial_start_footprint_clearance_applied":
                    mission_comparison[
                        "initial_start_footprint_clearance_applied"],
                "rotation_transition_path_index": mission_comparison[
                    "rotation_transition_index"],
                "rotation_transition_distance_m": mission_comparison[
                    "rotation_transition_distance_m"],
                "rotation_transition_world_m": list(
                    global_geometry.cell_to_world(global_avoidance[
                        mission_comparison["rotation_transition_index"]]))
                    if mission_comparison["rotation_transition_index"] is not None else None,
                "required_rotation_clearance_m": mission_comparison[
                    "rotation_clearance_m"],
                "selection_rotation_clearance_m": mission_comparison[
                    "rotation_selection_clearance_m"],
                "rotation_acceptance_tolerance_m": mission_comparison[
                    "rotation_acceptance_tolerance_m"],
            }
        with open(os.path.join(directory, "metrics.json"), "w",
                  encoding="utf-8") as stream:
            json.dump(metrics, stream, indent=2, sort_keys=True)

        header = self.evolution.header
        self.base_map_publisher.publish(
            self._grid_message(header, geometry, occupancy))
        self.fused_map_publisher.publish(
            self._grid_message(header, geometry, fused_cost))
        if mission_comparison is None:
            published_geometry = geometry
            published_baseline = baseline_path
            published_avoidance = avoidance_path
        else:
            published_geometry = mission_comparison["geometry"]
            published_baseline = mission_comparison["baseline_path"]
            published_avoidance = mission_comparison["avoidance_path"]
        self.baseline_path_publisher.publish(
            self._path_message(header, published_geometry, published_baseline))
        self.avoidance_path_publisher.publish(
            self._path_message(header, published_geometry, published_avoidance))
        return directory


if __name__ == "__main__":
    rospy.init_node("static_planning_evaluation")
    StaticPlanningEvaluation()
    rospy.spin()
