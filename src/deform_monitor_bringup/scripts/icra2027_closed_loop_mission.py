#!/usr/bin/env python3
"""ICRA2027 map-building, deformation-triggered autonomous return mission."""

import copy
import csv
import hashlib
import json
import math
import os
import sys
import termios
import threading
import time
import uuid

import actionlib
import dynamic_reconfigure.client
import numpy as np
import rospy
import rospkg
import tf.transformations as transformations
from geometry_msgs.msg import PoseStamped, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import OccupancyGrid, Odometry
from nav_msgs.srv import GetPlan
from std_msgs.msg import String
from std_srvs.srv import Empty, Trigger, TriggerResponse

from deform_evolution_map.msg import DeformationEvolutionFrame
from deform_avoidance_map.msg import DeformationAvoidanceMap
from deform_monitor_v3.msg import (
    V1DetectionStatus,
    V1DetectionTile,
    V1StableId128,
)
from deform_monitor_v3.srv import SetV1MonitoringPhase, SetV1MonitoringPhaseRequest

# catkin_install_python creates an executable wrapper in devel/lib.  That
# wrapper is not an importable library, so force this pure helper import to the
# package source/share scripts directory returned by rospack.
_PACKAGE_SCRIPTS = os.path.join(
    rospkg.RosPack().get_path("deform_monitor_bringup"), "scripts")
if _PACKAGE_SCRIPTS in sys.path:
    sys.path.remove(_PACKAGE_SCRIPTS)
sys.path.insert(0, _PACKAGE_SCRIPTS)

from closed_loop_core import (  # noqa: E402
    bounded_disk_cells,
    circular_footprint_is_traversable,
    forward_path_tracking_command,
    goal_or_frontier,
    integrate_unicycle,
    normalize_angle,
    retained_source_persistence_update,
    pose_within_tolerance,
    rectangular_footprint_is_traversable,
    recorder_finalization_result,
    reverse_path_waypoints,
    reverse_path_tracking_command,
    select_safe_forward_command,
    select_safe_reverse_command,
    swept_rectangular_command_is_traversable,
    waypoint_passed,
)
from planning_core import (  # noqa: E402
    GridGeometry,
    astar,
    inflate_obstacles,
)


def yaw_from_pose(pose):
    quaternion = pose.orientation
    return transformations.euler_from_quaternion(
        (quaternion.x, quaternion.y, quaternion.z, quaternion.w))[2]


def pose2d(message):
    pose = message.pose.pose
    return (float(pose.position.x), float(pose.position.y),
            yaw_from_pose(pose))


def stable_id_string(stable_id):
    return "{:016x}{:016x}".format(stable_id.high, stable_id.low)


def world_to_cell(message, point):
    column = int(math.floor((point[0] - message.info.origin.position.x)
                            / message.info.resolution))
    row = int(math.floor((point[1] - message.info.origin.position.y)
                         / message.info.resolution))
    if not (0 <= column < message.info.width
            and 0 <= row < message.info.height):
        return None
    return column, row


def world_to_cell_unbounded(message, point):
    return (int(math.floor((point[0] - message.info.origin.position.x)
                           / message.info.resolution)),
            int(math.floor((point[1] - message.info.origin.position.y)
                           / message.info.resolution)))


def cell_to_world(message, cell):
    return (message.info.origin.position.x
            + (cell[0] + 0.5) * message.info.resolution,
            message.info.origin.position.y
            + (cell[1] + 0.5) * message.info.resolution)


class MissionFailure(RuntimeError):
    pass


class HeadingClearanceLost(MissionFailure):
    """The latest fused map no longer supports an in-place turn."""


def confirm_video_start(reader, terminal):
    terminal.write("\n[READY_TO_RECORD] Gazebo / RViz are ready; simulation is paused.\n"
                   "Arrange the windows, start your video recorder, then return to this terminal.\n")
    while True:
        terminal.write("Enter 1 or press Enter to start the experiment; Ctrl+C to cancel: ")
        terminal.flush()
        answer = reader.readline()
        if not answer:
            terminal.write("\nNo start confirmation received; the experiment remains waiting.\n")
            terminal.flush()
            return False
        if answer.strip() in ("", "1"):
            return True
        terminal.write("Not started: enter 1 or press Enter.\n")


class ClosedLoopMission:
    def __init__(self):
        self.lock = threading.RLock()
        self.lio_odom = None
        self.navigation_map = None
        self.monitor_status = None
        self.evolution = None
        self.avoidance = None
        self.evaluation_status = None
        self.motion_state = ""
        self.motion_experiment_elapsed_s = None
        self.recorder_timeline_status = None
        self.distance_travelled = 0.0
        self.last_distance_pose = None
        self.state = "WAITING_FOR_INPUTS"
        self.state_history = []
        self.start_wall_time = time.monotonic()
        self.run_session_id = uuid.uuid4().hex
        self.recording_cutoff_accepted_frame_index = None
        self.recording_cutoff_experiment_elapsed_s = None
        self.completion_request = None
        self.home_map = None
        self.arrival_map_pose = None
        self.target_map = None
        self.detected_source_ids = []
        self.observed_algorithm_source_ids = []
        self.return_trigger_retained_since_stamp_s = None
        self.return_trigger_retained_duration_s = 0.0
        self.return_trigger_algorithm_source_id = None
        self.persistent_trigger_observed = False
        self.first_detection_stamp_s = None
        self.first_detection_accepted_frame_index = None
        self.full_sequence_final_accepted_frame_index = None
        self.full_sequence_motion_completed = False
        self.return_trigger_contributing_source_ids = []
        self.maximum_commanded_linear_speed_mps = 0.0
        self.return_path_map = []
        self.return_reverse_profile_applied = False
        self.return_live_replan_count = 0
        self.return_forward_profile_restored = False
        self.return_navigation_map = None
        self.return_command_pose = None
        self.return_command_stamp_s = None
        self.return_command_linear_mps = 0.0
        self.return_command_angular_rps = 0.0
        self.return_waypoint_count = 0
        self.frontier_goal_count = 0
        self.move_base_goal_count = 0
        self.failure = ""
        self.detector_diagnostics = {
            "lod2_frame_count": 0,
            "maximum_gdu_count": 0,
            "maximum_physical_sample_count": 0,
            "maximum_evidence_count": 0,
            "maximum_task_scale_probability": 0.0,
            "maximum_displacement_magnitude_m": 0.0,
            "latest_status_gdu_count": 0,
            "latest_status_phase": 0,
            "latest_status_reference_frozen": False,
            "latest_deadline_failure_count": 0,
            "maximum_deadline_failure_count": 0,
            "latest_resource_failure_count": 0,
            "maximum_resource_failure_count": 0,
            "latest_lod2_stamp_s": None,
            "latest_lod2_interval_s": None,
            "minimum_lod2_interval_s": None,
            "maximum_lod2_interval_s": 0.0,
            "evolution_frame_count": 0,
            "latest_evolution_accepted_frame_index": None,
            "latest_active_source_count": 0,
            "maximum_active_source_count": 0,
            "latest_retained_source_count": 0,
            "maximum_retained_source_count": 0,
            "avoidance_frame_count": 0,
            "latest_avoidance_accepted_frame_index": None,
            "latest_avoidance_source_count": 0,
            "maximum_avoidance_source_count": 0,
            "latest_all_current_sources_within_map": None,
            "latest_all_forecast_sources_within_map": None,
            "latest_out_of_bounds_current_source_count": 0,
            "latest_out_of_bounds_forecast_source_count": 0,
            "gdu_maxima": {},
        }

        prefix = "~mission/"
        self.skip_outbound_navigation = bool(rospy.get_param(
            prefix + "skip_outbound_navigation", False))
        self.return_after_detection = bool(rospy.get_param(
            prefix + "return_after_detection", True))
        self.full_process_observation_only = bool(rospy.get_param(
            prefix + "full_process_observation_only", False))
        self.full_process_expected_duration = float(rospy.get_param(
            prefix + "full_process_expected_experiment_duration_s", 0.0))
        self.require_recorder_finalization = bool(rospy.get_param(
            prefix + "require_recorder_finalization", False))
        self.recorder_finalization_timeout = float(rospy.get_param(
            prefix + "recorder_finalization_timeout_s", 180.0))
        if (not math.isfinite(self.recorder_finalization_timeout)
                or self.recorder_finalization_timeout <= 0.0):
            raise ValueError("recorder_finalization_timeout_s must be positive")
        if (self.full_process_observation_only
                and (not self.skip_outbound_navigation
                     or self.return_after_detection)):
            raise ValueError(
                "full_process_observation_only requires direct stationary "
                "monitoring without return navigation")
        if (self.full_process_observation_only
                and (not math.isfinite(self.full_process_expected_duration)
                     or self.full_process_expected_duration <= 0.0)):
            raise ValueError(
                "full-process observation requires a positive expected "
                "experiment duration")
        self.return_trigger_persistence = float(rospy.get_param(
            prefix + "return_trigger_persistence_s", 10.0))
        if (not math.isfinite(self.return_trigger_persistence)
                or self.return_trigger_persistence <= 0.0):
            raise ValueError("invalid retained-cause return trigger")
        # Online triggering is deliberately semantic-free.  This fixed label
        # exists only to keep legacy JSON consumers type-compatible; it is not
        # read from configuration and never names a Gazebo model or region.
        self.return_trigger_scope = "GENERIC_ANY_ALGORITHM_GDU"
        approach_map = rospy.get_param(prefix + "approach_map_xy", [])
        self.approach_map = tuple(float(value) for value in approach_map)
        if len(self.approach_map) != 2:
            raise ValueError(
                "approach_map_xy must contain exactly two camera_init values")
        if not all(math.isfinite(value) for value in self.approach_map):
            raise ValueError("approach_map_xy must be finite")
        self.approach_yaw_map = math.radians(float(rospy.get_param(
            prefix + "approach_yaw_map_deg", 0.0)))
        self.arrival_tolerance = float(rospy.get_param(
            prefix + "arrival_tolerance_m", 0.8))
        self.arrival_yaw_tolerance = math.radians(float(rospy.get_param(
            prefix + "arrival_yaw_tolerance_deg", 45.0)))
        self.home_tolerance = float(rospy.get_param(
            prefix + "home_tolerance_m", 0.8))
        self.heading_alignment_timeout = float(rospy.get_param(
            prefix + "heading_alignment_timeout_s", 45.0))
        if self.heading_alignment_timeout <= 0.0:
            raise ValueError("heading alignment timeout must be positive")
        self.robot_radius = float(rospy.get_param(
            prefix + "robot_radius_m", 0.30))
        self.footprint_half_length = float(rospy.get_param(
            prefix + "footprint_half_length_m", 0.36))
        self.footprint_half_width = float(rospy.get_param(
            prefix + "footprint_half_width_m", 0.30))
        if (self.footprint_half_length <= 0.0
                or self.footprint_half_width <= 0.0):
            raise ValueError("robot footprint half extents must be positive")
        self.frontier_backoff = float(rospy.get_param(
            prefix + "frontier_backoff_m", 0.45))
        self.navigation_timeout = float(rospy.get_param(
            prefix + "navigation_timeout_s", 360.0))
        self.goal_timeout = float(rospy.get_param(
            prefix + "single_goal_timeout_s", 75.0))
        self.goal_progress_timeout = float(rospy.get_param(
            prefix + "goal_progress_timeout_s", 35.0))
        self.return_goal_progress_timeout = float(rospy.get_param(
            prefix + "return_goal_progress_timeout_s", 120.0))
        self.goal_progress_distance = float(rospy.get_param(
            prefix + "goal_progress_distance_m", 0.15))
        self.frontier_blacklist_radius = float(rospy.get_param(
            prefix + "frontier_blacklist_radius_m", 0.90))
        self.return_waypoint_spacing = float(rospy.get_param(
            prefix + "return_waypoint_spacing_m", 2.0))
        self.return_reverse_escape_distance = float(rospy.get_param(
            prefix + "return_reverse_escape_distance_m", 0.75))
        self.return_reverse_escape_tolerance = float(rospy.get_param(
            prefix + "return_reverse_escape_tolerance_m", 0.25))
        self.return_forward_max_speed = float(rospy.get_param(
            prefix + "return_forward_max_speed_mps", 1.10))
        self.return_unsafe_hold_timeout = float(rospy.get_param(
            prefix + "return_unsafe_hold_timeout_s", 10.0))
        if (self.goal_progress_timeout <= 0.0
                or self.return_goal_progress_timeout <= 0.0
                or self.goal_progress_distance <= 0.0
                or self.frontier_blacklist_radius < 0.0
                or not math.isfinite(self.return_waypoint_spacing)
                or self.return_waypoint_spacing <= 0.0
                or not math.isfinite(self.return_reverse_escape_distance)
                or self.return_reverse_escape_distance <= 0.0
                or self.return_reverse_escape_distance
                > self.return_waypoint_spacing
                or not math.isfinite(self.return_reverse_escape_tolerance)
                or self.return_reverse_escape_tolerance <= 0.0
                or self.return_reverse_escape_tolerance
                >= self.return_reverse_escape_distance
                or not math.isfinite(self.return_forward_max_speed)
                or not 0.20 <= self.return_forward_max_speed <= 1.25
                or not math.isfinite(self.return_unsafe_hold_timeout)
                or self.return_unsafe_hold_timeout <= 0.0):
            raise ValueError("invalid navigation progress watchdog")
        self.reference_timeout = float(rospy.get_param(
            prefix + "reference_timeout_s", 60.0))
        self.motion_start_timeout = float(rospy.get_param(
            prefix + "motion_start_timeout_s", 180.0))
        if (not math.isfinite(self.motion_start_timeout)
                or self.motion_start_timeout <= 0.0):
            raise ValueError("motion_start_timeout_s must be positive")
        self.detection_timeout = float(rospy.get_param(
            prefix + "detection_timeout_s", 90.0))
        self.evaluation_timeout = float(rospy.get_param(
            prefix + "evaluation_timeout_s", 120.0))
        if self.evaluation_timeout <= 0.0:
            raise ValueError("evaluation timeout must be positive")
        self.stable_dwell = float(rospy.get_param(
            prefix + "stable_dwell_s", 4.0))
        self.reference_preparation_dwell = float(rospy.get_param(
            prefix + "reference_preparation_dwell_s", 3.5))
        if self.reference_preparation_dwell < 0.5:
            raise ValueError("reference preparation dwell must be at least 0.5 s")
        self.roi_half_extent = float(rospy.get_param(
            prefix + "monitor_roi_half_extent_m", 10.0))
        self.roi_below = float(rospy.get_param(
            prefix + "monitor_roi_below_sensor_m", 1.0))
        self.roi_above = float(rospy.get_param(
            prefix + "monitor_roi_above_sensor_m", 6.5))
        if self.roi_half_extent <= 0.0 or self.roi_below < 0.0 or self.roi_above <= 0.0:
            raise ValueError("monitor ROI extents must be positive")
        # Latch one generic monitoring volume from the current LIO sensor pose
        # at reference start.  No configured world coordinate, Gazebo model
        # box, semantic label or hand-selected stable-support region is read by
        # the online detector/trigger path.
        self.monitor_roi_anchor_map_xyz = None
        self.monitor_roi_bounds_map = None
        self.output_directory = os.path.abspath(os.path.expanduser(
            rospy.get_param(prefix + "output_directory",
                            "/tmp/icra2027_closed_loop")))
        self.session_id = rospy.get_param(
            prefix + "session_id", "icra2027-closed-loop-v1")
        self.motion_namespace = "/" + str(rospy.get_param(
            prefix + "motion_namespace", "/icra2027_motion")).strip("/")
        self.stop_motion_on_return = bool(rospy.get_param(
            prefix + "stop_motion_on_return", True))

        self.status_publisher = rospy.Publisher(
            "/deform/mission/status", String, queue_size=1, latch=True)
        self.clock_text_publisher = rospy.Publisher(
            "/deform/mission/clock_text", String, queue_size=1, latch=True)
        self.goal_publisher = rospy.Publisher(
            "/deform/mission/goal", PoseStamped, queue_size=1, latch=True)
        self.cmd_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("/navigation/odometry", Odometry, self._lio_callback,
                         queue_size=5)
        rospy.Subscriber("/cmd_vel", Twist, self._command_callback,
                         queue_size=20)
        rospy.Subscriber("/mission/navigation_map", OccupancyGrid,
                         self._map_callback, queue_size=1)
        rospy.Subscriber("/deform/v1/status", V1DetectionStatus,
                         self._status_callback, queue_size=5)
        rospy.Subscriber("/deform/v1/detection_tiles", V1DetectionTile,
                         self._tile_callback, queue_size=5)
        rospy.Subscriber("/deform/evolution/frame", DeformationEvolutionFrame,
                         self._evolution_callback, queue_size=2)
        rospy.Subscriber("/deform/avoidance/frame", DeformationAvoidanceMap,
                         self._avoidance_callback, queue_size=2)
        rospy.Subscriber(self.motion_namespace + "/status", String,
                         self._motion_status_callback, queue_size=5)
        rospy.Subscriber("/deform/experiment_recorder/status", String,
                         self._recorder_status_callback, queue_size=5)
        rospy.Subscriber("/deform/evaluation/status", String,
                         self._evaluation_status_callback, queue_size=2)
        self.move_base = actionlib.SimpleActionClient(
            "/move_base", MoveBaseAction)
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _restore_forward_retreat(self):
        """Restore cruise speed with bounded collision-checked backing."""
        client = dynamic_reconfigure.client.Client(
            "/move_base/DWAPlannerROS", timeout=8.0)
        updated = client.update_configuration({
            "min_vel_x": -0.30,
            "max_vel_x": self.return_forward_max_speed,
            "min_vel_trans": 0.05,
            "max_vel_theta": 0.90,
            "forward_point_distance": 0.25,
            "path_distance_bias": 40.0,
            "goal_distance_bias": 24.0,
            "yaw_goal_tolerance": math.radians(30.0),
        })
        if (abs(float(updated.get("min_vel_x", 1.0)) + 0.30) > 1.0e-6
                or abs(float(updated.get("max_vel_x", 0.0))
                       - self.return_forward_max_speed)
                > 1.0e-6
                or abs(float(updated.get("min_vel_trans", 0.0))
                       - 0.05) > 1.0e-6
                or abs(float(updated.get("max_vel_theta", 0.0))
                       - 0.90) > 1.0e-6
                or abs(float(updated.get("forward_point_distance", 0.0))
                       - 0.25) > 1.0e-6
                or abs(float(updated.get("path_distance_bias", 0.0))
                       - 40.0) > 1.0e-6
                or abs(float(updated.get("goal_distance_bias", 0.0))
                       - 24.0) > 1.0e-6):
            raise MissionFailure("DWA forward-retreat profile was not restored")
        self.return_forward_profile_restored = True

    def _load_reverse_evacuation_waypoints(self, evaluation):
        output_directory = str(evaluation.get("output_directory", ""))
        path_file = os.path.join(output_directory, "global_avoidance_path.csv")
        metrics_file = os.path.join(output_directory, "metrics.json")
        if (not output_directory or not os.path.isfile(path_file)
                or not os.path.isfile(metrics_file)):
            raise MissionFailure(
                "exported global avoidance path or metrics are missing")
        points = []
        try:
            with open(path_file, "r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames is None or not {"x_m", "y_m"}.issubset(
                        set(reader.fieldnames)):
                    raise ValueError("invalid global avoidance path schema")
                for row in reader:
                    points.append((float(row["x_m"]), float(row["y_m"])))
            with open(metrics_file, "r", encoding="utf-8") as stream:
                metrics = json.load(stream)
            evacuation = metrics["mission_evacuation"]
            transition_index = int(
                evacuation["rotation_transition_path_index"])
            required_clearance = float(
                evacuation["required_rotation_clearance_m"])
            transition_distance = float(
                evacuation["rotation_transition_distance_m"])
            if (transition_index <= 0 or transition_index + 1 >= len(points)
                    or not math.isfinite(required_clearance)
                    or required_clearance <= 0.0
                    or not math.isfinite(transition_distance)
                    or transition_distance < self.return_reverse_escape_distance):
                raise ValueError("invalid rotation transition in metrics")
            current = pose2d(self._snapshot("lio_odom"))
            reverse_points = points[:transition_index + 1]
            waypoints = reverse_path_waypoints(
                reverse_points, current[:2], reverse_points[-1],
                spacing_m=self.return_waypoint_spacing,
                first_spacing_m=self.return_reverse_escape_distance,
                endpoint_tolerance_m=max(self.home_tolerance, 0.75),
                maximum_step_m=0.35)
            forward_heading = math.atan2(
                points[transition_index + 1][1] - points[transition_index][1],
                points[transition_index + 1][0] - points[transition_index][0])
            return (waypoints, tuple(points), transition_index,
                    normalize_angle(forward_heading), transition_distance,
                    required_clearance)
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise MissionFailure(
                "exported global avoidance path rejected: {}".format(error))

    def _return_home(self):
        """Use live SLAM geometry with the retained deformation soft costs."""
        # Keep the same configured DWA horizon/critics as outbound. Shortening
        # the horizon at return caused repeated stalls on the new scene's bends.
        # Only real geometry and UNKNOWN can invalidate the footprint.
        # Finite deformation costs need no start clearing or egress refresh.
        self._wait(
            self._current_footprint_is_traversable,
            self.return_unsafe_hold_timeout,
            "RETURNING_HOME geometric footprint check")
        current = self._return_pose()
        home_heading = math.atan2(self.home_map[1] - current[1],
                                  self.home_map[0] - current[0])
        try:
            # Avoid a long reverse leg from the outward-facing observation
            # pose. This changes orientation only; Navfn still chooses the
            # return route, including finite deformation costs.
            self._align_heading_in_place(home_heading, math.radians(20.0))
        except HeadingClearanceLost:
            rospy.logwarn("[closed_loop] HOME turn clearance unavailable; DWA must create clearance")
        self.return_waypoint_count = 1
        rospy.logwarn(
            "[closed_loop] MOVE_BASE_FUSED_RETURN: planning HOME on live "
            "SLAM + retained deformation costs; UNKNOWN remains forbidden")
        self._navigate(self.home_map[:2], self.home_tolerance, False,
                       "deformation-aware return")

    def _rotation_clearance_at_pose(self, pose, required_clearance):
        navigation_map = self._return_map_snapshot()
        return (navigation_map is not None
                and circular_footprint_is_traversable(
                    navigation_map.data, navigation_map.info.width,
                    navigation_map.info.height,
                    navigation_map.info.resolution,
                    (navigation_map.info.origin.position.x,
                     navigation_map.info.origin.position.y),
                    pose[:2], required_clearance))

    def _current_footprint_is_traversable(self):
        navigation_map = self._snapshot("navigation_map")
        odometry = self._snapshot("lio_odom")
        if navigation_map is None or odometry is None:
            return False
        return rectangular_footprint_is_traversable(
            navigation_map.data, navigation_map.info.width,
            navigation_map.info.height, navigation_map.info.resolution,
            (navigation_map.info.origin.position.x,
             navigation_map.info.origin.position.y),
            pose2d(odometry), self.footprint_half_length,
            self.footprint_half_width)

    def _return_map_snapshot(self):
        if self.return_navigation_map is not None:
            return self.return_navigation_map
        return self._snapshot("navigation_map")

    def _return_pose(self):
        if self.return_command_pose is None:
            return pose2d(self._snapshot("lio_odom"))
        now = rospy.Time.now().to_sec()
        elapsed = max(0.0, now - self.return_command_stamp_s)
        self.return_command_pose = integrate_unicycle(
            self.return_command_pose, self.return_command_linear_mps,
            self.return_command_angular_rps, elapsed)
        self.return_command_stamp_s = now
        return self.return_command_pose

    def _publish_control_command(self, command):
        if self.return_command_pose is not None:
            self._return_pose()
            self.return_command_linear_mps = float(command.linear.x)
            self.return_command_angular_rps = float(command.angular.z)
        self.cmd_publisher.publish(command)

    def _next_live_rotation_transition(self, path, current_index,
                                       required_clearance):
        """Find a farther turn point whose tolerance disk remains safe."""
        navigation_map = self._return_map_snapshot()
        if navigation_map is None:
            return None
        # If the tracker accepts any pose in the target disk, a clearance of
        # R+tolerance at its centre guarantees at least R at the accepted pose.
        protected_radius = (float(required_clearance)
                            + self.return_reverse_escape_tolerance)
        accumulated = 0.0
        for index in range(current_index + 1, len(path) - 1):
            accumulated += math.hypot(
                path[index][0] - path[index - 1][0],
                path[index][1] - path[index - 1][1])
            if accumulated < 0.50:
                continue
            if circular_footprint_is_traversable(
                    navigation_map.data, navigation_map.info.width,
                    navigation_map.info.height,
                    navigation_map.info.resolution,
                    (navigation_map.info.origin.position.x,
                     navigation_map.info.origin.position.y),
                    path[index], protected_radius):
                return index
        return None

    def _track_reverse_fused_path(self, path, transition):
        """Track a validated fused-map path with reverse unicycle control."""
        self.move_base.cancel_all_goals()
        self._stop()
        tracking_path = list(path)
        tracking_lookahead = 1.00
        index = 0
        unsafe_hold_started = None
        best_distance = self._distance_to(transition)
        deadline = time.monotonic() + self.navigation_timeout
        progress_deadline = time.monotonic() + self.return_goal_progress_timeout
        next_progress_log = time.monotonic()
        rate = rospy.Rate(15.0)
        self.return_reverse_profile_applied = True
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            current = self._return_pose()
            remaining = math.hypot(current[0] - transition[0],
                                   current[1] - transition[1])
            if time.monotonic() >= next_progress_log:
                rospy.logwarn(
                    "[closed_loop] RETURN_PROGRESS reverse: %.2f m remaining",
                    remaining)
                next_progress_log = time.monotonic() + 5.0
            if remaining <= self.return_reverse_escape_tolerance:
                self._stop()
                return
            index, linear, angular, _ = reverse_path_tracking_command(
                tracking_path, current, index,
                lookahead_m=tracking_lookahead,
                speed_mps=0.20, heading_gain=0.8,
                # The current dual-LiDAR/IMU FAST-LIO chain was observed to
                # follow 0.15 rad/s yaw consistently.  At 0.60 rad/s the
                # truth yaw outran the estimate, creating positive feedback.
                maximum_angular_speed=0.15, search_window=30)
            navigation_map = self._return_map_snapshot()
            command_is_safe = (navigation_map is not None
                and swept_rectangular_command_is_traversable(
                    navigation_map.data, navigation_map.info.width,
                    navigation_map.info.height, navigation_map.info.resolution,
                    (navigation_map.info.origin.position.x,
                     navigation_map.info.origin.position.y),
                    current, self.footprint_half_length,
                    self.footprint_half_width, linear, angular,
                    horizon_s=0.8, step_s=0.1))
            if not command_is_safe:
                if navigation_map is None:
                    self._stop()
                    raise MissionFailure("reverse tracker lost navigation map")
                try:
                    tracking_path = self._replan_on_return_map(
                        navigation_map, current, transition)
                except MissionFailure:
                    tracking_path = []
                tracking_lookahead = 0.60
                if not tracking_path:
                    if unsafe_hold_started is None:
                        unsafe_hold_started = time.monotonic()
                    self._stop()
                    if (time.monotonic() - unsafe_hold_started
                            >= self.return_unsafe_hold_timeout):
                        raise MissionFailure(
                            "live reverse map persistently has no safe path")
                    rate.sleep()
                    continue
                index, linear, angular, _ = reverse_path_tracking_command(
                    tracking_path, current, 0, lookahead_m=0.60,
                    speed_mps=0.20, heading_gain=0.8,
                    maximum_angular_speed=0.15, search_window=30)
                if not swept_rectangular_command_is_traversable(
                        navigation_map.data, navigation_map.info.width,
                        navigation_map.info.height,
                        navigation_map.info.resolution,
                        (navigation_map.info.origin.position.x,
                         navigation_map.info.origin.position.y),
                        current, self.footprint_half_length,
                        self.footprint_half_width, linear, angular,
                        horizon_s=0.8, step_s=0.1):
                    lattice_command = select_safe_reverse_command(
                        navigation_map.data, navigation_map.info.width,
                        navigation_map.info.height,
                        navigation_map.info.resolution,
                        (navigation_map.info.origin.position.x,
                         navigation_map.info.origin.position.y),
                        current, tracking_path, transition,
                        self.footprint_half_length,
                        self.footprint_half_width,
                        horizon_s=0.8, step_s=0.1)
                    if lattice_command is None:
                        if unsafe_hold_started is None:
                            unsafe_hold_started = time.monotonic()
                        self._stop()
                        if (time.monotonic() - unsafe_hold_started
                                >= self.return_unsafe_hold_timeout):
                            raise MissionFailure(
                                "live reverse map persistently has no safe command")
                        rate.sleep()
                        continue
                    linear, angular = lattice_command
                self.return_live_replan_count += 1
                rospy.logwarn(
                    "[closed_loop] live fused-map reverse replan %d: %d points",
                    self.return_live_replan_count, len(tracking_path))
            unsafe_hold_started = None
            command = Twist()
            command.linear.x = linear
            command.angular.z = angular
            self._publish_control_command(command)
            if remaining <= best_distance - self.goal_progress_distance:
                best_distance = remaining
                progress_deadline = (time.monotonic()
                                     + self.return_goal_progress_timeout)
            if time.monotonic() >= progress_deadline:
                self._stop()
                raise MissionFailure(
                    "reverse fused-map tracker made no distance progress")
            rate.sleep()
        self._stop()
        raise MissionFailure("reverse fused-map tracker timed out")

    def _track_forward_fused_path(self, path, goal):
        """Track the frozen fused-map A* path forward to recorded HOME."""
        self.move_base.cancel_all_goals()
        self._stop()
        tracking_path = list(path)
        index = 0
        unsafe_hold_started = None
        best_distance = self._distance_to(goal)
        deadline = time.monotonic() + self.navigation_timeout
        progress_deadline = time.monotonic() + self.return_goal_progress_timeout
        next_progress_log = time.monotonic()
        rate = rospy.Rate(15.0)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            current = self._return_pose()
            remaining = math.hypot(current[0] - goal[0],
                                   current[1] - goal[1])
            if time.monotonic() >= next_progress_log:
                rospy.logwarn(
                    "[closed_loop] RETURN_PROGRESS forward: %.2f m remaining",
                    remaining)
                next_progress_log = time.monotonic() + 5.0
            if remaining <= self.home_tolerance:
                self._stop()
                return
            index, linear, angular, _ = forward_path_tracking_command(
                tracking_path, current, index, lookahead_m=1.00,
                speed_mps=self.return_forward_max_speed,
                heading_gain=0.8, maximum_angular_speed=0.30,
                search_window=30)
            navigation_map = self._return_map_snapshot()
            command_is_safe = (navigation_map is not None
                and swept_rectangular_command_is_traversable(
                    navigation_map.data, navigation_map.info.width,
                    navigation_map.info.height, navigation_map.info.resolution,
                    (navigation_map.info.origin.position.x,
                     navigation_map.info.origin.position.y),
                    current, self.footprint_half_length,
                    self.footprint_half_width, linear, angular,
                    horizon_s=0.8, step_s=0.1))
            if not command_is_safe:
                if navigation_map is None:
                    self._stop()
                    raise MissionFailure("forward tracker lost navigation map")
                try:
                    tracking_path = self._replan_on_return_map(
                        navigation_map, current, goal)
                except MissionFailure:
                    tracking_path = []
                index = 0
                lattice_command = (select_safe_forward_command(
                    navigation_map.data, navigation_map.info.width,
                    navigation_map.info.height,
                    navigation_map.info.resolution,
                    (navigation_map.info.origin.position.x,
                     navigation_map.info.origin.position.y),
                    current, tracking_path, goal,
                    self.footprint_half_length,
                    self.footprint_half_width,
                    horizon_s=0.8, step_s=0.1)
                    if tracking_path else None)
                if lattice_command is None:
                    if unsafe_hold_started is None:
                        unsafe_hold_started = time.monotonic()
                    self._stop()
                    if (time.monotonic() - unsafe_hold_started
                            >= self.return_unsafe_hold_timeout):
                        raise MissionFailure(
                            "frozen forward map persistently has no safe command")
                    rate.sleep()
                    continue
                linear, angular = lattice_command
                self.return_live_replan_count += 1
            unsafe_hold_started = None
            command = Twist()
            command.linear.x = linear
            command.angular.z = angular
            self._publish_control_command(command)
            if remaining <= best_distance - self.goal_progress_distance:
                best_distance = remaining
                progress_deadline = (time.monotonic()
                                     + self.return_goal_progress_timeout)
            if time.monotonic() >= progress_deadline:
                self._stop()
                raise MissionFailure(
                    "forward fused-map tracker made no distance progress")
            rate.sleep()
        self._stop()
        raise MissionFailure("forward fused-map tracker timed out")

    def _replan_on_return_map(self, navigation_map, current, transition):
        """Plan a strict known-free centre path on the frozen return grid."""
        geometry = GridGeometry(
            navigation_map.info.resolution, navigation_map.info.width,
            navigation_map.info.height,
            navigation_map.info.origin.position.x,
            navigation_map.info.origin.position.y)
        occupancy = np.asarray(navigation_map.data, dtype=np.int16).reshape(
            geometry.height, geometry.width)
        inflated = inflate_obstacles(
            occupancy, self.robot_radius, geometry.resolution)
        start = geometry.world_to_cell(current[0], current[1])
        goal = geometry.world_to_cell(transition[0], transition[1])
        if start is None or goal is None:
            raise MissionFailure("live reverse replan endpoint left map bounds")
        if (not 0 <= occupancy[start[1], start[0]] < 100
                or not rectangular_footprint_is_traversable(
                    navigation_map.data, geometry.width, geometry.height,
                    geometry.resolution, (geometry.origin_x, geometry.origin_y),
                    current, self.footprint_half_length, self.footprint_half_width)):
            raise MissionFailure("live replan starts in blocked or UNKNOWN geometry")
        # The exact oriented-footprint test above proved the current pose safe.
        # Circular inflation can still mark its centre at a diagonal corner;
        # exempt that one known centre, never UNKNOWN or subsequent cells.
        if inflated[start[1], start[0]] >= 100:
            inflated = inflated.copy()
            inflated[start[1], start[0]] = 0
        cells, _ = astar(
            inflated, start, goal, unknown_is_traversable=False)
        if not cells or len(cells) < 2:
            raise MissionFailure(
                "live fused map has no known-free reverse path")
        points = [geometry.cell_to_world(cell) for cell in cells]
        points[0] = (float(current[0]), float(current[1]))
        return points

    def _lio_callback(self, message):
        with self.lock:
            self.lio_odom = copy.deepcopy(message)
            current = (message.pose.pose.position.x,
                       message.pose.pose.position.y)
            if self.last_distance_pose is not None:
                step = math.hypot(current[0] - self.last_distance_pose[0],
                                  current[1] - self.last_distance_pose[1])
                if math.isfinite(step) and step < 1.0:
                    self.distance_travelled += step
            self.last_distance_pose = current
            if self.state == "RETURNING_HOME":
                if (not self.return_path_map
                        or math.hypot(current[0] - self.return_path_map[-1][0],
                                      current[1] - self.return_path_map[-1][1])
                        >= 0.10):
                    self.return_path_map.append([float(current[0]),
                                                 float(current[1])])

    def _command_callback(self, message):
        speed = math.hypot(float(message.linear.x), float(message.linear.y))
        if math.isfinite(speed):
            with self.lock:
                self.maximum_commanded_linear_speed_mps = max(
                    self.maximum_commanded_linear_speed_mps, speed)

    def _recorder_status_callback(self, message):
        try:
            document = json.loads(message.data)
            if not isinstance(document, dict):
                raise TypeError("recorder status must be an object")
            # The mission consumes only the terminal recorder acknowledgement.
            # Truth row counts, truth stamps, checkpoints and high-water
            # diagnostics remain recorder/offline data and are intentionally
            # not copied into online mission state or mission_summary.json.
            acknowledgement = {
                "finalization_started": bool(document.get(
                    "finalization_started", False)),
                "finalized": bool(document.get("finalized", False)),
                "finalization_error": str(document.get(
                    "finalization_error", "") or ""),
                "required_artifact_errors": list(document.get(
                    "required_artifact_errors", [])),
                "finalization_request_id": str(document.get(
                    "finalization_request_id", "") or ""),
                "run_manifest": str(document.get("run_manifest", "") or ""),
            }
        except (TypeError, ValueError):
            rospy.logerr_throttle(
                2.0, "invalid experiment-recorder status JSON")
            return
        with self.lock:
            self.recorder_timeline_status = acknowledgement

    def _map_callback(self, message):
        if (message.header.frame_id == "camera_init"
                and len(message.data) == message.info.width * message.info.height):
            with self.lock:
                self.navigation_map = copy.deepcopy(message)

    def _status_callback(self, message):
        with self.lock:
            self.monitor_status = copy.deepcopy(message)
            self.detector_diagnostics["latest_status_gdu_count"] = int(
                message.gdu_count)
            self.detector_diagnostics["maximum_gdu_count"] = max(
                self.detector_diagnostics["maximum_gdu_count"],
                int(message.gdu_count))
            self.detector_diagnostics["latest_status_phase"] = int(
                message.phase)
            self.detector_diagnostics["latest_status_reference_frozen"] = bool(
                message.reference_frozen)
            deadline_failures = int(message.deadline_failure_count)
            resource_failures = int(message.resource_failure_count)
            self.detector_diagnostics["latest_deadline_failure_count"] = (
                deadline_failures)
            self.detector_diagnostics["maximum_deadline_failure_count"] = max(
                self.detector_diagnostics["maximum_deadline_failure_count"],
                deadline_failures)
            self.detector_diagnostics["latest_resource_failure_count"] = (
                resource_failures)
            self.detector_diagnostics["maximum_resource_failure_count"] = max(
                self.detector_diagnostics["maximum_resource_failure_count"],
                resource_failures)

    def _tile_callback(self, message):
        if int(message.lod) != 2:
            return
        stamp_s = float(message.common.header.stamp.to_sec())
        probabilities = []
        evidence_updates = {}
        for evidence in message.detection_evidence:
            probability = (evidence.task_scale_probability_mean
                           if evidence.task_scale_probability_mean_identified
                           else evidence.task_scale_probability_upper)
            if math.isfinite(probability):
                probabilities.append(float(probability))
                evidence_updates[stable_id_string(evidence.gdu_id)] = {
                    "probability": float(probability),
                    "observable_rank": int(evidence.observable_rank),
                    "effective_spatial_support": (
                        float(evidence.effective_spatial_support)
                        if math.isfinite(evidence.effective_spatial_support)
                        else 0.0),
                    "effective_temporal_support": (
                        float(evidence.effective_temporal_support)
                        if math.isfinite(evidence.effective_temporal_support)
                        else 0.0),
                    "detection_admission": int(evidence.detection_admission),
                    "validity": int(evidence.validity),
                }
        magnitudes = []
        magnitudes_by_gdu = {}
        for sample in message.physical_deformation:
            vector = sample.displacement_mean
            magnitude = math.sqrt(vector.x * vector.x + vector.y * vector.y
                                  + vector.z * vector.z)
            if math.isfinite(magnitude):
                magnitudes.append(magnitude)
                key = stable_id_string(sample.gdu_id)
                magnitudes_by_gdu[key] = max(
                    magnitudes_by_gdu.get(key, 0.0), magnitude)
        with self.lock:
            diagnostics = self.detector_diagnostics
            previous_stamp_s = diagnostics["latest_lod2_stamp_s"]
            if (previous_stamp_s is not None and math.isfinite(stamp_s)
                    and stamp_s > previous_stamp_s):
                interval_s = stamp_s - previous_stamp_s
                diagnostics["latest_lod2_interval_s"] = interval_s
                current_minimum = diagnostics["minimum_lod2_interval_s"]
                diagnostics["minimum_lod2_interval_s"] = (
                    interval_s if current_minimum is None
                    else min(current_minimum, interval_s))
                diagnostics["maximum_lod2_interval_s"] = max(
                    diagnostics["maximum_lod2_interval_s"], interval_s)
            if math.isfinite(stamp_s):
                diagnostics["latest_lod2_stamp_s"] = stamp_s
            diagnostics["lod2_frame_count"] += 1
            diagnostics["maximum_physical_sample_count"] = max(
                diagnostics["maximum_physical_sample_count"],
                len(message.physical_deformation))
            diagnostics["maximum_evidence_count"] = max(
                diagnostics["maximum_evidence_count"],
                len(message.detection_evidence))
            if probabilities:
                diagnostics["maximum_task_scale_probability"] = max(
                    diagnostics["maximum_task_scale_probability"],
                    max(probabilities))
            if magnitudes:
                diagnostics["maximum_displacement_magnitude_m"] = max(
                    diagnostics["maximum_displacement_magnitude_m"],
                    max(magnitudes))
            for key in sorted(set(evidence_updates) | set(magnitudes_by_gdu)):
                update = evidence_updates.get(key, {})
                entry = diagnostics["gdu_maxima"].setdefault(key, {
                    "maximum_task_scale_probability": 0.0,
                    "maximum_displacement_magnitude_m": 0.0,
                    "latest_observable_rank": 0,
                    "maximum_effective_spatial_support": 0.0,
                    "maximum_effective_temporal_support": 0.0,
                    "latest_detection_admission": 0,
                    "latest_validity": 2,
                })
                entry["maximum_task_scale_probability"] = max(
                    entry["maximum_task_scale_probability"],
                    update.get("probability", 0.0))
                entry["maximum_displacement_magnitude_m"] = max(
                    entry["maximum_displacement_magnitude_m"],
                    magnitudes_by_gdu.get(key, 0.0))
                if update:
                    entry["latest_observable_rank"] = update["observable_rank"]
                    entry["maximum_effective_spatial_support"] = max(
                        entry["maximum_effective_spatial_support"],
                        update["effective_spatial_support"])
                    entry["maximum_effective_temporal_support"] = max(
                        entry["maximum_effective_temporal_support"],
                        update["effective_temporal_support"])
                    entry["latest_detection_admission"] = update[
                        "detection_admission"]
                    entry["latest_validity"] = update["validity"]

    def _evolution_callback(self, message):
        with self.lock:
            self.evolution = copy.deepcopy(message)
            active_count = sum(bool(gdu.active_now) for gdu in message.gdus)
            retained_count = sum(
                bool(gdu.retained_for_avoidance) for gdu in message.gdus)
            diagnostics = self.detector_diagnostics
            diagnostics["evolution_frame_count"] += 1
            diagnostics["latest_evolution_accepted_frame_index"] = int(
                message.source_accepted_frame_index)
            diagnostics["latest_active_source_count"] = active_count
            diagnostics["maximum_active_source_count"] = max(
                diagnostics["maximum_active_source_count"], active_count)
            diagnostics["latest_retained_source_count"] = retained_count
            diagnostics["maximum_retained_source_count"] = max(
                diagnostics["maximum_retained_source_count"], retained_count)

    def _avoidance_callback(self, message):
        with self.lock:
            self.avoidance = copy.deepcopy(message)
            source_count = len(message.active_source_ids)
            diagnostics = self.detector_diagnostics
            diagnostics["avoidance_frame_count"] += 1
            diagnostics["latest_avoidance_accepted_frame_index"] = int(
                message.source_accepted_frame_index)
            diagnostics["latest_avoidance_source_count"] = source_count
            diagnostics["maximum_avoidance_source_count"] = max(
                diagnostics["maximum_avoidance_source_count"], source_count)
            diagnostics["latest_all_current_sources_within_map"] = bool(
                message.all_current_sources_within_map)
            diagnostics["latest_all_forecast_sources_within_map"] = bool(
                message.all_forecast_sources_within_map)
            diagnostics["latest_out_of_bounds_current_source_count"] = len(
                message.out_of_bounds_current_source_ids)
            diagnostics["latest_out_of_bounds_forecast_source_count"] = len(
                message.out_of_bounds_forecast_source_ids)

    def _motion_status_callback(self, message):
        try:
            document = json.loads(message.data)
            if not isinstance(document, dict):
                raise TypeError("motion status must be an object")
            state = str(document.get("state", ""))
            experiment_elapsed = float(document.get(
                "experiment_elapsed", float("nan")))
            motion_elapsed = float(document.get(
                "motion_elapsed", float("nan")))
            if not state or not math.isfinite(experiment_elapsed):
                raise ValueError("motion status lacks finite progress")
        except (TypeError, ValueError):
            rospy.logerr_throttle(2.0, "invalid ICRA2027 motion status JSON")
            return
        with self.lock:
            previous_state = self.motion_state
            self.motion_state = state
            self.motion_experiment_elapsed_s = experiment_elapsed
            mission_state = self.state
            trigger_elapsed = self.return_trigger_retained_duration_s
        clock_text = (
            "MISSION={} | EXPERIMENT={:.1f}s | MOTION={} {} | "
            "TRIGGER={:.1f}/{:.1f}s".format(
                mission_state, experiment_elapsed, state,
                ("{:.1f}s".format(motion_elapsed)
                 if math.isfinite(motion_elapsed) else "n/a"),
                trigger_elapsed, self.return_trigger_persistence))
        self.clock_text_publisher.publish(String(data=clock_text))
        rospy.loginfo_throttle(
            2.0, "[closed_loop] EXPERIMENT_CLOCK %s", clock_text)
        if state != previous_state:
            rospy.logwarn(
                "[closed_loop] ===== OBJECT_MOTION_%s ===== "
                "experiment=%.2f s motion=%s",
                state, experiment_elapsed,
                ("{:.2f} s".format(motion_elapsed)
                 if math.isfinite(motion_elapsed) else "n/a"))

    def _evaluation_status_callback(self, message):
        try:
            document = json.loads(message.data)
            if not isinstance(document, dict):
                return
        except (TypeError, ValueError):
            rospy.logerr_throttle(2.0, "invalid planning evaluation status JSON")
            return
        with self.lock:
            self.evaluation_status = document

    def _snapshot(self, attribute):
        with self.lock:
            return copy.deepcopy(getattr(self, attribute))

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _completion_high_water(self, decision_summary_path):
        lio = self._snapshot("lio_odom")
        evolution = self._snapshot("evolution")
        cutoff_index = self.recording_cutoff_accepted_frame_index
        if cutoff_index is None and evolution is not None:
            cutoff_index = int(evolution.source_accepted_frame_index)
        cutoff_elapsed = self.recording_cutoff_experiment_elapsed_s
        if cutoff_elapsed is None:
            cutoff_elapsed = self.motion_experiment_elapsed_s
        request = {
            "schema_version": 1,
            "request_id": self.run_session_id,
            "run_session_id": self.run_session_id,
            "mission_outcome": "COMPLETED",
            "output_directory": self.output_directory,
            "decision_summary_relative_path": "mission_decision_summary.json",
            "decision_summary_sha256": self._sha256(decision_summary_path),
            "required_motion_experiment_elapsed_s": cutoff_elapsed,
            "required_detector_accepted_frame_index": cutoff_index,
            "required_evolution_accepted_frame_index": cutoff_index,
            "required_avoidance_accepted_frame_index": cutoff_index,
            "required_navigation_stamp_s": (
                lio.header.stamp.to_sec() if lio is not None else None),
        }
        return request

    def _write_decision_summary(self):
        summary_path = self._summary("COMPLETED")
        decision_path = os.path.join(
            self.output_directory, "mission_decision_summary.json")
        temporary = decision_path + ".tmp"
        with open(summary_path, "rb") as source, open(temporary, "wb") as target:
            target.write(source.read())
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, decision_path)
        return decision_path

    def _set_state(self, state, detail=""):
        self.state = state
        entry = {
            "state": state,
            "wall_elapsed_s": time.monotonic() - self.start_wall_time,
            "sim_time_s": rospy.Time.now().to_sec(),
            "detail": detail,
        }
        self.state_history.append(entry)
        cutoff_payload = None
        if state in ("DEFORMATION_DETECTED", "MONITORING_COMPLETE"):
            evolution = self._snapshot("evolution")
            if evolution is not None:
                self.recording_cutoff_accepted_frame_index = int(
                    evolution.source_accepted_frame_index)
            self.recording_cutoff_experiment_elapsed_s = (
                self.motion_experiment_elapsed_s)
            cutoff_payload = {
                "accepted_frame_index": (
                    self.recording_cutoff_accepted_frame_index),
                "motion_experiment_elapsed_s": (
                    self.recording_cutoff_experiment_elapsed_s),
            }
        completion_request = None
        if state == "COMPLETED" and self.require_recorder_finalization:
            # The recorder's analyses consume mission_summary.json.  Commit a
            # stable decision copy before publishing the cross-topic finalize
            # request, then rewrite mission_summary after terminal handshake.
            decision_path = self._write_decision_summary()
            completion_request = self._completion_high_water(decision_path)
            self.completion_request = copy.deepcopy(completion_request)
        payload = dict(entry)
        payload["home_map_xy"] = (list(self.home_map[:2])
                                  if self.home_map else None)
        payload["monitor_roi_policy"] = "CURRENT_LIO_POSE_GENERIC_LOCAL_VOLUME"
        payload["monitor_roi_anchor_map_xyz"] = (
            list(self.monitor_roi_anchor_map_xyz)
            if self.monitor_roi_anchor_map_xyz is not None else None)
        payload["monitor_roi_bounds_map"] = copy.deepcopy(
            self.monitor_roi_bounds_map)
        if cutoff_payload is not None:
            payload["capture_cutoff"] = cutoff_payload
        if completion_request is not None:
            payload["finalization_request"] = completion_request
        self.status_publisher.publish(String(data=json.dumps(
            payload, sort_keys=True, ensure_ascii=False)))
        if state in ("BUILDING_REFERENCE", "MONITORING",
                     "DEFORMATION_DETECTED", "RETURNING_HOME"):
            rospy.logwarn("[closed_loop] ===== %s ===== %s", state, detail)
        else:
            rospy.loginfo("[closed_loop] %s: %s", state, detail)

    def _wait(self, predicate, timeout, description):
        deadline = time.monotonic() + timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.05)
        raise MissionFailure("timeout waiting for " + description)

    def _wait_for_move_base(self, timeout):
        deadline = time.monotonic() + timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            # A short ROS-duration probe may complete immediately when the
            # saved Gazebo world advances /clock by a large first step; the
            # surrounding deadline is deliberately a wall-clock contract.
            if self.move_base.wait_for_server(rospy.Duration(0.2)):
                return True
            time.sleep(0.05)
        return False

    def _stop(self):
        self.move_base.cancel_all_goals()
        command = Twist()
        for _ in range(5):
            self._publish_control_command(command)
            time.sleep(0.03)

    def _goal_message(self, point, yaw):
        goal = MoveBaseGoal()
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = "camera_init"
        goal.target_pose.pose.position.x = point[0]
        goal.target_pose.pose.position.y = point[1]
        quaternion = transformations.quaternion_from_euler(0.0, 0.0, yaw)
        goal.target_pose.pose.orientation.x = quaternion[0]
        goal.target_pose.pose.orientation.y = quaternion[1]
        goal.target_pose.pose.orientation.z = quaternion[2]
        goal.target_pose.pose.orientation.w = quaternion[3]
        self.goal_publisher.publish(goal.target_pose)
        return goal

    def _distance_to(self, point):
        if self.return_command_pose is not None:
            current = self._return_pose()
            return math.hypot(current[0] - point[0], current[1] - point[1])
        odometry = self._snapshot("lio_odom")
        if odometry is None:
            return float("inf")
        return math.hypot(odometry.pose.pose.position.x - point[0],
                          odometry.pose.pose.position.y - point[1])

    def _pose_reached(self, point, xy_tolerance, desired_yaw=None,
                      yaw_tolerance=None):
        if self.return_command_pose is not None:
            return pose_within_tolerance(
                self._return_pose(), point, xy_tolerance,
                desired_yaw, yaw_tolerance)
        odometry = self._snapshot("lio_odom")
        return (odometry is not None and pose_within_tolerance(
            pose2d(odometry), point, xy_tolerance,
            desired_yaw, yaw_tolerance))

    def _send_goal(self, point, final_target, timeout, desired_yaw=None,
                   xy_tolerance=0.35, yaw_tolerance=None,
                   acceptance_target=None, acceptance_xy_tolerance=None,
                   progress_timeout=None,
                   pass_through_cross_track_tolerance=None):
        odometry = self._snapshot("lio_odom")
        current = pose2d(odometry)
        segment_start = current[:2]
        yaw = (float(desired_yaw) if desired_yaw is not None else
               math.atan2(final_target[1] - current[1],
                          final_target[0] - current[0]))
        self.move_base_goal_count += 1
        self.move_base.send_goal(self._goal_message(point, yaw))
        deadline = time.monotonic() + timeout
        best_distance = self._distance_to(point)
        progress_limit = (self.goal_progress_timeout
                          if progress_timeout is None else
                          float(progress_timeout))
        if not math.isfinite(progress_limit) or progress_limit <= 0.0:
            raise MissionFailure("goal progress timeout must be positive")
        progress_deadline = time.monotonic() + progress_limit
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            current_pose = pose2d(self._snapshot("lio_odom"))
            if (pass_through_cross_track_tolerance is not None
                    and waypoint_passed(
                        segment_start, point, current_pose,
                        pass_through_cross_track_tolerance)):
                self.move_base.cancel_goal()
                self._stop()
                return True
            if (acceptance_target is not None
                    and acceptance_xy_tolerance is not None
                    and self._pose_reached(
                        acceptance_target, acceptance_xy_tolerance,
                        desired_yaw, yaw_tolerance)):
                self.move_base.cancel_goal()
                self._stop()
                return True
            if self._pose_reached(point, xy_tolerance, desired_yaw,
                                  yaw_tolerance):
                self.move_base.cancel_goal()
                self._stop()
                return True
            state = self.move_base.get_state()
            if state == 3:  # actionlib_msgs/GoalStatus.SUCCEEDED
                return True
            if state in (4, 5, 8, 9):
                return False
            current_distance = self._distance_to(point)
            if current_distance <= best_distance - self.goal_progress_distance:
                best_distance = current_distance
                progress_deadline = time.monotonic() + progress_limit
            if (current_distance > xy_tolerance
                    and time.monotonic() >= progress_deadline):
                rospy.logwarn(
                    "[closed_loop] cancelling stalled goal at (%.2f, %.2f); "
                    "best remaining distance %.2f m", point[0], point[1],
                    best_distance)
                self.move_base.cancel_goal()
                self._stop()
                return False
            time.sleep(0.1)
        self.move_base.cancel_goal()
        self._stop()
        return False

    def _align_heading_inside_target_disk(self, target, tolerance,
                                          desired_yaw, yaw_tolerance):
        """Align heading after position acceptance, with bounded hysteresis.

        The final acceptance remains the exact task disk.  A kinematic Gazebo
        base can drift a few centimetres during a commanded pure rotation, so
        a small guard band prevents a transient boundary crossing from being
        treated as a mission failure.  If yaw becomes valid outside the exact
        disk, the caller restores translation and approaches again.
        """
        if self._pose_reached(target, tolerance, desired_yaw, yaw_tolerance):
            return True
        alignment_guard = tolerance + 0.10
        client = dynamic_reconfigure.client.Client(
            "/move_base/DWAPlannerROS", timeout=8.0)
        rotation_profile = client.update_configuration({
            "min_vel_x": 0.0,
            "max_vel_x": 0.0,
            "xy_goal_tolerance": float(tolerance),
        })
        if (abs(float(rotation_profile.get("min_vel_x", 1.0))) > 1.0e-9
                or abs(float(rotation_profile.get("max_vel_x", 1.0))) > 1.0e-9):
            raise MissionFailure("DWA pure-rotation profile was not applied")
        try:
            current = pose2d(self._snapshot("lio_odom"))
            self.move_base.send_goal(
                self._goal_message(current[:2], desired_yaw))
            deadline = time.monotonic() + self.heading_alignment_timeout
            while not rospy.is_shutdown() and time.monotonic() < deadline:
                if self._pose_reached(target, tolerance,
                                      desired_yaw, yaw_tolerance):
                    self.move_base.cancel_goal()
                    self._stop()
                    return True
                current_pose = pose2d(self._snapshot("lio_odom"))
                yaw_is_valid = (abs(normalize_angle(
                    current_pose[2] - desired_yaw)) <= yaw_tolerance)
                if yaw_is_valid:
                    self.move_base.cancel_goal()
                    self._stop()
                    return False
                if not self._pose_reached(target, alignment_guard):
                    self.move_base.cancel_goal()
                    self._stop()
                    raise MissionFailure(
                        "heading alignment exceeded the bounded target guard")
                if self.move_base.get_state() in (4, 5, 8, 9):
                    break
                time.sleep(0.1)
            self.move_base.cancel_goal()
            self._stop()
            raise MissionFailure("target heading did not reach its tolerance")
        finally:
            restored = client.update_configuration({
                "min_vel_x": -0.30,
                "max_vel_x": 1.25,
                "xy_goal_tolerance": 0.15,
            })
            if (abs(float(restored.get("min_vel_x", 1.0)) + 0.30) > 1.0e-6
                    or abs(float(restored.get("max_vel_x", 0.0))
                           - 1.25) > 1.0e-6):
                rospy.logerr("DWA outbound profile restoration failed")

    def _align_heading_in_place(self, desired_yaw, yaw_tolerance):
        """Rotate at a verified pose with direct swept-footprint control."""
        initial = self._return_pose()
        if abs(normalize_angle(initial[2] - desired_yaw)) <= yaw_tolerance:
            return
        self.move_base.cancel_all_goals()
        self._stop()
        deadline = time.monotonic() + self.heading_alignment_timeout
        rate = rospy.Rate(15.0)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            current = self._return_pose()
            error = normalize_angle(desired_yaw - current[2])
            if abs(error) <= yaw_tolerance:
                self._stop()
                return
            angular = max(-0.15, min(0.15, 0.8 * error))
            navigation_map = self._return_map_snapshot()
            if (navigation_map is None
                    or not swept_rectangular_command_is_traversable(
                        navigation_map.data, navigation_map.info.width,
                        navigation_map.info.height,
                        navigation_map.info.resolution,
                        (navigation_map.info.origin.position.x,
                         navigation_map.info.origin.position.y),
                        current, self.footprint_half_length,
                        self.footprint_half_width, 0.0, angular,
                        horizon_s=0.8, step_s=0.1)):
                self._stop()
                raise HeadingClearanceLost(
                    "in-place heading alignment lost swept-footprint clearance")
            command = Twist()
            command.angular.z = angular
            self._publish_control_command(command)
            rate.sleep()
        self._stop()
        raise MissionFailure("retreat heading did not reach its tolerance")

    def _align_path_departure(self, path):
        """Resolve a large departure turn before asking DWA to translate.

        The known-free selection path supplies only a local bearing. Actual
        translation still follows Navfn/DWA; rotation reuses the live-map
        swept rectangular-footprint check, with no ground truth or teleport.
        """
        current = self._return_pose()
        previous, distance = current[:2], 0.0
        point = previous
        for point in path:
            distance += math.hypot(point[0] - previous[0], point[1] - previous[1])
            if distance >= 1.5:
                break
            previous = point
        if distance < 0.5:
            return
        heading = math.atan2(point[1] - current[1], point[0] - current[0])
        error = normalize_angle(heading - current[2])
        if abs(error) <= math.radians(20.0):
            return
        rospy.logwarn("[closed_loop] ALIGN_PATH: heading error %.1f deg",
                      math.degrees(error))
        try:
            self._align_heading_in_place(heading, math.radians(10.0))
        except HeadingClearanceLost:
            # Do not force a rotation where only a translating arc can fit.
            rospy.logwarn("[closed_loop] turn clearance unavailable; using collision-checked DWA arcs")

    def _recover_return_heading(self, target, tolerance):
        """Re-align to a fresh native plan, never clear costs or explore."""
        current = self._return_pose()
        start = self._goal_message(current[:2], current[2]).target_pose
        goal = self._goal_message(target, current[2]).target_pose
        try:
            rospy.wait_for_service("/move_base/make_plan", timeout=8.0)
            plan = rospy.ServiceProxy("/move_base/make_plan", GetPlan)(
                start, goal, tolerance).plan
        except (rospy.ROSException, rospy.ServiceException) as error:
            raise MissionFailure("return recovery cannot obtain a native plan: " + str(error))
        points = [(pose.pose.position.x, pose.pose.position.y) for pose in plan.poses]
        # Noetic move_base leaves Path.header empty; each PoseStamped carries its frame.
        if (not points or any(pose.header.frame_id != start.header.frame_id for pose in plan.poses)
                or not all(math.isfinite(v) for point in points for v in point)
                or math.hypot(points[0][0] - current[0], points[0][1] - current[1]) > 0.5
                or math.hypot(points[-1][0] - target[0], points[-1][1] - target[1]) > tolerance):
            raise MissionFailure("return recovery has no valid native route to HOME")
        self._align_path_departure(points)

    def _navigate(self, target, tolerance, permit_frontiers, label,
                  desired_yaw=None, yaw_tolerance=None):
        if not permit_frontiers:
            # Navfn owns return reachability and its actual path. Do not run
            # a second circular-BFS planner on a different map, or substitute
            # exploratory spins for HOME after a local-planner failure.
            deadline = time.monotonic() + self.navigation_timeout
            for attempt in range(3):  # Initial goal plus at most two recovery attempts.
                remaining = deadline - time.monotonic()
                if rospy.is_shutdown() or remaining <= 0.0:
                    break
                success = self._send_goal(
                    target, target, remaining, acceptance_target=target,
                    acceptance_xy_tolerance=tolerance,
                    progress_timeout=self.return_goal_progress_timeout)
                self._stop()
                if success:
                    return
                if attempt < 2 and time.monotonic() < deadline:
                    rospy.logwarn("[closed_loop] RETURN_RECOVERY %d/2: re-aligning to native HOME plan",
                                  attempt + 1)
                    self._recover_return_heading(target, tolerance)
            self._stop()
            raise MissionFailure("move_base could not complete " + label)
        deadline = time.monotonic() + self.navigation_timeout
        blacklisted = set()
        no_progress_count = 0
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if self._pose_reached(target, tolerance, desired_yaw,
                                  yaw_tolerance):
                self._stop()
                return
            navigation_map = self._snapshot("navigation_map")
            odometry = self._snapshot("lio_odom")
            if navigation_map is None or odometry is None:
                time.sleep(0.1)
                continue
            current_pose = pose2d(odometry)
            start_cell = world_to_cell(navigation_map, current_pose[:2])
            target_cell = world_to_cell_unbounded(navigation_map, target)
            if start_cell is None:
                raise MissionFailure("robot lies outside the current map")
            array = np.asarray(navigation_map.data, dtype=np.int16).reshape(
                navigation_map.info.height, navigation_map.info.width)
            plan = goal_or_frontier(
                array, start_cell, target_cell, navigation_map.info.resolution,
                arrival_tolerance_m=tolerance,
                robot_radius_m=self.robot_radius,
                frontier_backoff_m=self.frontier_backoff,
                blacklisted=blacklisted)
            if plan is None:
                no_progress_count += 1
                if no_progress_count > 8:
                    raise MissionFailure(
                        "no reachable known-free goal or frontier for " + label)
                # Request an in-place scan through move_base.  This uses no
                # unknown translational cell and can expose a new frontier.
                spin_goal = self._goal_message(
                    current_pose[:2], normalize_angle(current_pose[2] + math.pi / 2.0))
                self.move_base.send_goal(spin_goal)
                self.move_base.wait_for_result(rospy.Duration(8.0))
                self.move_base.cancel_goal()
                time.sleep(0.5)
                continue
            mode, endpoint_cell, selection_path = plan
            if mode == "FRONTIER" and not permit_frontiers:
                raise MissionFailure(
                    label + " has no route entirely inside the built map")
            endpoint = cell_to_world(navigation_map, endpoint_cell)
            if mode == "FRONTIER":
                self.frontier_goal_count += 1
            before = self._distance_to(target)
            # Latch the requested task position first.  Combining exact grid
            # center convergence and final heading in one move_base goal can
            # make a vehicle that is already inside the requested 0.80 m
            # disk drive out again merely to turn.  Heading is aligned only
            # after the translational goal has been cancelled and stopped.
            final_yaw = None
            final_yaw_tolerance = None
            # This tolerance is relative to the selected reachable grid-cell
            # center, not the requested task point.  It must stay smaller than
            # the outer task tolerance so the two errors cannot add up.
            endpoint_tolerance = 0.18 if mode == "GOAL" else 0.35
            self._align_path_departure([
                cell_to_world(navigation_map, cell) for cell in selection_path])
            succeeded = self._send_goal(
                endpoint, target, self.goal_timeout,
                desired_yaw=final_yaw,
                xy_tolerance=endpoint_tolerance,
                yaw_tolerance=final_yaw_tolerance,
                acceptance_target=(target if mode == "GOAL" else None),
                acceptance_xy_tolerance=(tolerance
                                         if mode == "GOAL" else None),
                progress_timeout=(self.return_goal_progress_timeout
                                  if self.state == "RETURNING_HOME" else
                                  self.goal_progress_timeout))
            after = self._distance_to(target)
            if not succeeded or after > before - 0.10:
                # A failed exploratory frontier is replaceable.  The final
                # task goal is not: blacklisting HOME/GOAL deletes the mission
                # objective and turns a recoverable local stall into a false
                # global "no route" result.
                if mode == "FRONTIER":
                    blacklist_radius_cells = int(math.ceil(
                        self.frontier_blacklist_radius
                        / navigation_map.info.resolution))
                    blacklisted.update(bounded_disk_cells(
                        endpoint_cell, blacklist_radius_cells,
                        navigation_map.info.width, navigation_map.info.height))
                no_progress_count += 1
            else:
                no_progress_count = 0
            if (mode == "GOAL" and succeeded
                    and self._pose_reached(target, tolerance)):
                self._stop()
                if desired_yaw is not None:
                    if not self._align_heading_inside_target_disk(
                            target, tolerance, desired_yaw, yaw_tolerance):
                        continue
                return
        raise MissionFailure("navigation timeout for " + label)

    def _monitor_center_map(self):
        if self.monitor_roi_anchor_map_xyz is not None:
            return self.monitor_roi_anchor_map_xyz[:2]
        odometry = self._snapshot("lio_odom")
        if odometry is None:
            raise MissionFailure("LIO pose unavailable for generic monitor ROI")
        center = odometry.pose.pose.position
        values = (float(center.x), float(center.y), float(center.z))
        if not all(math.isfinite(value) for value in values):
            raise MissionFailure("LIO pose is invalid for generic monitor ROI")
        self.monitor_roi_anchor_map_xyz = values
        return values[:2]

    def _phase_request(self, phase, effective_stamp, selection_cutoff):
        request = SetV1MonitoringPhaseRequest()
        request.session_id = self.session_id
        request.target_phase = phase
        roi_center = self._monitor_center_map()
        center_z = self.monitor_roi_anchor_map_xyz[2]
        bounds = [
            roi_center[0] - self.roi_half_extent,
            roi_center[1] - self.roi_half_extent,
            center_z - self.roi_below,
            roi_center[0] + self.roi_half_extent,
            roi_center[1] + self.roi_half_extent,
            center_z + self.roi_above,
        ]
        self.monitor_roi_bounds_map = bounds
        request.roi_min.x, request.roi_min.y, request.roi_min.z = bounds[:3]
        request.roi_max.x, request.roi_max.y, request.roi_max.z = bounds[3:]
        request.effective_stamp = effective_stamp
        request.selection_support_cutoff = selection_cutoff
        # Build this mission's baseline only from the stationary generic local
        # volume after arrival; pre-arrival mapping context is not deformation-
        # reference evidence for this task.
        request.trigger_source = 2
        request.expected_previous_snapshot = V1StableId128()
        return request

    def _start_monitoring(self):
        rospy.wait_for_service("/deform/v1/set_monitoring_phase",
                               timeout=self.reference_timeout)
        service = rospy.ServiceProxy(
            "/deform/v1/set_monitoring_phase", SetV1MonitoringPhase)
        status = self._wait(
            lambda: (self._snapshot("monitor_status")
                     if self._snapshot("monitor_status") is not None
                     and self._snapshot("monitor_status").phase == 0
                     and self._snapshot("monitor_status").last_attempt_stamp.to_sec() > 0.0
                     else None),
            self.reference_timeout, "fresh MAPPING_CONTEXT status")
        cutoff = status.last_attempt_stamp
        response = service(self._phase_request(1, cutoff, cutoff))
        if not response.accepted:
            raise MissionFailure("MONITOR_REFERENCE rejected: {}".format(
                list(response.reason_codes)))
        frozen = self._wait(
            lambda: (self._snapshot("monitor_status")
                     if self._snapshot("monitor_status") is not None
                     and self._snapshot("monitor_status").phase == 1
                     and self._snapshot("monitor_status").reference_frozen
                     else None),
            self.reference_timeout, "frozen local reference")
        response = service(self._phase_request(
            2, frozen.last_attempt_stamp, cutoff))
        if not response.accepted:
            raise MissionFailure("MONITOR_UPDATE rejected: {}".format(
                list(response.reason_codes)))
        self._wait(
            lambda: (self._snapshot("monitor_status")
                     if self._snapshot("monitor_status") is not None
                     and self._snapshot("monitor_status").phase == 2
                     else None),
            self.reference_timeout, "MONITOR_UPDATE")

    def _motion_call(self, service_name):
        full_name = self.motion_namespace + "/" + service_name
        rospy.wait_for_service(full_name, timeout=15.0)
        response = rospy.ServiceProxy(full_name, Trigger)()
        if not response.success:
            raise MissionFailure(full_name + " failed: " + response.message)
        return response.message

    def _prepare_motion_scene(self):
        """Reset and arm targets before any deformation reference is built."""
        self._motion_call("reset")
        self._wait(lambda: (self._snapshot("motion_state")
                            if self._snapshot("motion_state") in
                            ("UNCONFIGURED", "ARMED")
                            else None), 15.0, "motion plugin reset/ARMED state")
        self._motion_call("reload")
        self._wait(lambda: (self._snapshot("motion_state")
                            if self._snapshot("motion_state") == "ARMED"
                            else None), 15.0, "motion plugin reload/ARMED state")

    def _trigger_motion(self):
        # Reference geometry is immutable from this point onward.  Resetting
        # here would create a synthetic jump between the frozen reference and
        # the first monitoring frame whenever a dynamic Gazebo body settled
        # between world load and target arrival.
        rospy.logwarn(
            "[closed_loop] ===== OBJECT_MOTION_COMMAND_SENT ===== "
            "micro-deformation detection is already active")
        self._motion_call("start")
        self._wait(lambda: (self._snapshot("motion_state")
                            if self._snapshot("motion_state") == "RUNNING"
                            else None), self.motion_start_timeout,
                   "motion plugin RUNNING state")
        rospy.logwarn(
            "[closed_loop] ===== OBJECT_MOTION_CONFIRMED_RUNNING =====")

    @staticmethod
    def _retained_algorithm_sources(frame):
        """Return causal GDU IDs currently retained by the risk-map layer.

        ``retained_for_avoidance`` is the authoritative causal-state bit.  It
        is born only after the evolution layer confirms three independent,
        fresh observable active LOD2 measurements inside its 15 s source-time
        window, and is removed only after three fresh observable clear
        measurements.  Current
        ``active_now`` or observability may drop temporarily without erasing
        that retained hazard, so neither is a second trigger gate here.
        """
        return tuple(sorted(set(
            str(gdu.gdu_id) for gdu in frame.gdus
            if gdu.retained_for_avoidance and str(gdu.gdu_id))))

    def _wait_for_avoidance_source(self, accepted_index, source_id):
        source_id = str(source_id)
        return self._wait(
            lambda: (self._snapshot("avoidance")
                     if self._snapshot("avoidance") is not None
                     and int(self._snapshot(
                         "avoidance").source_accepted_frame_index)
                     >= int(accepted_index)
                     and source_id in {
                         str(value) for value in self._snapshot(
                             "avoidance").active_source_ids}
                     else None),
            15.0, "retained trigger GDU committed to the navigation map")

    def _record_detection_snapshot(self, observed_source_ids,
                                   trigger_source_id):
        observed = sorted(set(str(value) for value in observed_source_ids))
        trigger = str(trigger_source_id)
        self.detected_source_ids = observed
        self.observed_algorithm_source_ids = observed
        self.return_trigger_algorithm_source_id = trigger
        self.return_trigger_contributing_source_ids = [trigger]
        self.persistent_trigger_observed = True

    def _full_process_completion_ready(self):
        """Require only the model-agnostic motion progress contract."""
        with self.lock:
            motion_state = self.motion_state
            motion_elapsed = self.motion_experiment_elapsed_s
        expected = self.full_process_expected_duration
        return (motion_state == "COMPLETED"
                and motion_elapsed is not None
                and motion_elapsed + 1.0e-6 >= expected)

    def _wait_for_recorder_finalization(self):
        """Keep the process tree alive until every final product is committed."""
        if not self.require_recorder_finalization:
            return None

        def ready():
            document = self._snapshot("recorder_timeline_status")
            finalized, error = recorder_finalization_result(document)
            if error:
                raise MissionFailure(
                    "experiment recorder finalization failed: " + error)
            return document if finalized else None

        status = self._wait(
            ready, self.recorder_finalization_timeout,
            "experiment recorder run_manifest finalization")
        if status.get("finalization_request_id") != self.run_session_id:
            raise MissionFailure(
                "recorder finalized a different run session")
        if status.get("required_artifact_errors"):
            raise MissionFailure(
                "recorder finalized with required artifact errors")
        if status.get("run_manifest") != "run_manifest.json":
            raise MissionFailure(
                "recorder finalization acknowledgement lacks run_manifest")
        manifest = os.path.join(self.output_directory, "run_manifest.json")
        if not os.path.isfile(manifest):
            raise MissionFailure(
                "recorder reported finalized without run_manifest.json")
        return status

    def _observe_full_motion_process(self):
        baseline = self._snapshot("evolution")
        baseline_index = (int(baseline.source_accepted_frame_index)
                          if baseline is not None else -1)
        source_intervals = {}
        observed_sources = set()
        trigger_recorded = False
        last_processed_index = baseline_index
        deadline = time.monotonic() + self.detection_timeout
        rate = rospy.Rate(20.0)
        while not rospy.is_shutdown():
            self._stop()
            frame = self._snapshot("evolution")
            if frame is not None:
                accepted_index = int(frame.source_accepted_frame_index)
                if accepted_index > last_processed_index:
                    last_processed_index = accepted_index
                    if (not frame.complete_snapshot
                            or not frame.input_contract_valid
                            or int(frame.validity) >= 2):
                        source_intervals = {}
                        if not trigger_recorded:
                            with self.lock:
                                self.return_trigger_retained_since_stamp_s = None
                                self.return_trigger_retained_duration_s = 0.0
                        continue
                    stamp_s = float(frame.header.stamp.to_sec())
                    retained_sources = self._retained_algorithm_sources(frame)
                    observed_sources.update(retained_sources)
                    source_intervals, durations, ready_sources = (
                        retained_source_persistence_update(
                            source_intervals, stamp_s, retained_sources,
                            self.return_trigger_persistence))
                    leading_source = (min(
                        durations, key=lambda source: (-durations[source],
                                                       source))
                        if durations else None)
                    if not trigger_recorded:
                        with self.lock:
                            self.return_trigger_retained_since_stamp_s = (
                                source_intervals[leading_source][0]
                                if leading_source is not None else None)
                            self.return_trigger_retained_duration_s = (
                                durations[leading_source]
                                if leading_source is not None else 0.0)
                    if ready_sources and not trigger_recorded:
                        trigger_source = min(
                            ready_sources,
                            key=lambda source: (-durations[source], source))
                        with self.lock:
                            self.return_trigger_retained_since_stamp_s = (
                                source_intervals[trigger_source][0])
                            self.return_trigger_retained_duration_s = (
                                durations[trigger_source])
                        self._wait_for_avoidance_source(
                            accepted_index, trigger_source)
                        self._record_detection_snapshot(
                            observed_sources, trigger_source)
                        trigger_recorded = True
                        self.first_detection_stamp_s = stamp_s
                        self.first_detection_accepted_frame_index = (
                            accepted_index)
                        rospy.loginfo(
                            "[closed_loop] generic retained deformation "
                            "cause=%s persisted %.2f s during full-process "
                            "observation", trigger_source,
                            durations[trigger_source])
            motion_state = self._snapshot("motion_state")
            if self._full_process_completion_ready():
                break
            if motion_state in ("ERROR", "STOPPED", "UNCONFIGURED"):
                raise MissionFailure(
                    "full-process motion entered terminal state {}".format(
                        motion_state))
            if time.monotonic() >= deadline:
                raise MissionFailure(
                    "full-process observation timed out before motion completion")
            rate.sleep()
        self.full_sequence_motion_completed = True
        self.full_sequence_final_accepted_frame_index = last_processed_index
        self.observed_algorithm_source_ids = sorted(observed_sources)
        self.detected_source_ids = sorted(observed_sources)
        if last_processed_index > baseline_index:
            self._wait(
                lambda: (self._snapshot("avoidance")
                         if self._snapshot("avoidance") is not None
                         and int(self._snapshot("avoidance").source_accepted_frame_index)
                         >= last_processed_index
                         else None),
                15.0, "final avoidance snapshot after full-process observation")
        return {
            "trigger_recorded": trigger_recorded,
            "last_processed_index": last_processed_index,
            "algorithm_source_ids": sorted(observed_sources),
        }

    def _wait_detection(self):
        baseline = self._snapshot("evolution")
        baseline_index = (int(baseline.source_accepted_frame_index)
                          if baseline is not None else -1)

        source_intervals = {}
        observed_sources = set()

        def detected():
            nonlocal source_intervals
            frame = self._snapshot("evolution")
            if frame is None:
                return None
            if int(frame.source_accepted_frame_index) <= baseline_index:
                return None
            if (not frame.complete_snapshot or not frame.input_contract_valid
                    or int(frame.validity) >= 2):
                source_intervals = {}
                with self.lock:
                    self.return_trigger_retained_since_stamp_s = None
                    self.return_trigger_retained_duration_s = 0.0
                return None
            stamp_s = float(frame.header.stamp.to_sec())
            retained_sources = self._retained_algorithm_sources(frame)
            observed_sources.update(retained_sources)
            source_intervals, durations, ready_sources = (
                retained_source_persistence_update(
                    source_intervals, stamp_s, retained_sources,
                    self.return_trigger_persistence))
            leading_source = (min(
                durations,
                key=lambda source: (-durations[source], source))
                if durations else None)
            with self.lock:
                self.return_trigger_retained_since_stamp_s = (
                    source_intervals[leading_source][0]
                    if leading_source is not None else None)
                self.return_trigger_retained_duration_s = (
                    durations[leading_source]
                    if leading_source is not None else 0.0)
            if not retained_sources:
                rospy.loginfo_throttle(
                    2.0, "[closed_loop] waiting for any confirmed retained "
                    "deformation cause")
                return None
            if not ready_sources:
                rospy.loginfo_throttle(
                    2.0, "[closed_loop] generic retained deformation leading "
                    "cause=%s %.2f/%.2f s (%d retained cause(s))",
                    leading_source, durations[leading_source],
                    self.return_trigger_persistence, len(retained_sources))
                return None
            trigger_source = min(
                ready_sources,
                key=lambda source: (-durations[source], source))
            with self.lock:
                self.return_trigger_retained_since_stamp_s = (
                    source_intervals[trigger_source][0])
                self.return_trigger_retained_duration_s = (
                    durations[trigger_source])
            return frame, trigger_source, sorted(observed_sources)

        frame, trigger_source, observed = self._wait(
            detected, self.detection_timeout,
            "persistent generic single-GDU retained-cause return trigger")
        accepted_index = int(frame.source_accepted_frame_index)
        self._wait_for_avoidance_source(accepted_index, trigger_source)
        self._record_detection_snapshot(observed, trigger_source)
        self.first_detection_stamp_s = float(frame.header.stamp.to_sec())
        self.first_detection_accepted_frame_index = accepted_index

    def _summary(self, outcome):
        current = self._snapshot("lio_odom")
        navigation_map = self._snapshot("navigation_map")
        current_pose = pose2d(current) if current is not None else None
        summary = {
            "run_session_id": self.run_session_id,
            "outcome": outcome,
            "failure": self.failure,
            "monitor_roi_policy": "CURRENT_LIO_POSE_GENERIC_LOCAL_VOLUME",
            "monitor_roi_anchor_map_xyz": (
                list(self.monitor_roi_anchor_map_xyz)
                if self.monitor_roi_anchor_map_xyz is not None else None),
            "monitor_roi_bounds_map": copy.deepcopy(
                self.monitor_roi_bounds_map),
            "monitor_roi_half_extent_m": self.roi_half_extent,
            "monitor_roi_below_sensor_m": self.roi_below,
            "monitor_roi_above_sensor_m": self.roi_above,
            "online_registered_regions_disabled": True,
            "approach_coordinate_frame": "camera_init",
            "configured_approach_map_xy": list(self.approach_map),
            "approach_yaw_map_deg": math.degrees(self.approach_yaw_map),
            "arrival_tolerance_m": self.arrival_tolerance,
            "arrival_yaw_tolerance_deg": math.degrees(
                self.arrival_yaw_tolerance),
            "target_map_xy": list(self.target_map) if self.target_map else None,
            "home_map_xy": list(self.home_map[:2]) if self.home_map else None,
            "final_map_pose": list(current_pose) if current_pose else None,
            "final_home_distance_m": (self._distance_to(self.home_map[:2])
                                      if self.home_map else None),
            "arrival_map_pose": (list(self.arrival_map_pose)
                                 if self.arrival_map_pose else None),
            "arrival_map_distance_m": (
                math.hypot(self.arrival_map_pose[0] - self.target_map[0],
                           self.arrival_map_pose[1] - self.target_map[1])
                if self.arrival_map_pose and self.target_map else None),
            "arrival_map_yaw_error_deg": (
                math.degrees(abs(normalize_angle(
                    self.arrival_map_pose[2] - self.approach_yaw_map)))
                if self.arrival_map_pose else None),
            "final_navigation_map": ({
                "width": int(navigation_map.info.width),
                "height": int(navigation_map.info.height),
                "resolution_m": float(navigation_map.info.resolution),
            } if navigation_map is not None else None),
            "distance_travelled_m": self.distance_travelled,
            "frontier_goal_count": self.frontier_goal_count,
            "move_base_goal_count": self.move_base_goal_count,
            "detected_source_ids": self.detected_source_ids,
            "observed_algorithm_source_ids": list(
                self.observed_algorithm_source_ids),
            "return_reverse_profile_applied": bool(
                self.return_reverse_profile_applied),
            "return_live_replan_count": int(self.return_live_replan_count),
            "return_forward_profile_restored": bool(
                self.return_forward_profile_restored),
            "return_waypoint_count": int(self.return_waypoint_count),
            "skip_outbound_navigation": bool(
                self.skip_outbound_navigation),
            "return_after_detection": bool(self.return_after_detection),
            "full_process_observation_only": bool(
                self.full_process_observation_only),
            "full_process_expected_experiment_duration_s": (
                self.full_process_expected_duration),
            "recorder_finalization_required": bool(
                self.require_recorder_finalization),
            "recorder_finalization_status": copy.deepcopy(
                self.recorder_timeline_status),
            "completion_request": copy.deepcopy(self.completion_request),
            "motion_status_state_at_end": self.motion_state,
            "motion_experiment_elapsed_s_at_end": (
                self.motion_experiment_elapsed_s),
            "experiment_mode": (
                "stationary_full_process"
                if self.full_process_observation_only
                else ("direct_detection"
                      if self.skip_outbound_navigation
                      and not self.return_after_detection
                      else "closed_loop")),
            "persistent_trigger_observed": bool(
                self.persistent_trigger_observed),
            "first_detection_stamp_s": self.first_detection_stamp_s,
            "first_detection_accepted_frame_index": (
                self.first_detection_accepted_frame_index),
            "full_sequence_motion_completed": bool(
                self.full_sequence_motion_completed),
            "full_sequence_final_accepted_frame_index": (
                self.full_sequence_final_accepted_frame_index),
            "return_trigger_scope": self.return_trigger_scope,
            "return_trigger_algorithm_source_id": (
                self.return_trigger_algorithm_source_id),
            "return_trigger_evidence_policy": (
                "CONTINUOUS_RETAINED_CAUSE_SAME_GDU_ID;"
                "BIRTH_AT_FIRST_RETAINED_EVOLUTION_SNAPSHOT;"
                "ACTIVE_NOW_OR_CURRENT_OBSERVABILITY_NOT_REQUIRED_AFTER_BIRTH;"
                "ABSENCE_OR_INVALID_INCOMPLETE_FRAME_RESETS_TIMER;"
                "NON_MONOTONE_SOURCE_TIME_RESTARTS_SOURCE_TIMER;"
                "IRREGULAR_SAMPLING_GAP_DOES_NOT_CLEAR_RETAINED_CAUSE;"
                "NO_CROSS_GDU_OR_SPATIAL_EVENT_STITCHING;"
                "EXACT_TRIGGER_SOURCE_REQUIRED_IN_AVOIDANCE_ACTIVE_IDS;"
                "NO_SEMANTIC_REGION_GAZEBO_TRUTH_MOTION_OR_ERROR_GATE"),
            "return_trigger_persistence_s": self.return_trigger_persistence,
            "return_trigger_retained_since_stamp_s": (
                self.return_trigger_retained_since_stamp_s),
            "return_trigger_retained_duration_s": (
                self.return_trigger_retained_duration_s),
            "return_trigger_contributing_source_ids": list(
                self.return_trigger_contributing_source_ids),
            "maximum_commanded_linear_speed_mps": (
                self.maximum_commanded_linear_speed_mps),
            "return_path_map": copy.deepcopy(self.return_path_map),
            "state_history": self.state_history,
            "rosbag_recorded": False,
            "semantic_recognition_used": False,
            "planner_unknown_is_traversable": False,
            "detector_diagnostics": copy.deepcopy(self.detector_diagnostics),
            "planning_evaluation": copy.deepcopy(
                self.evaluation_status),
        }
        os.makedirs(self.output_directory, exist_ok=True)
        temporary = os.path.join(self.output_directory, "mission_summary.json.tmp")
        final = os.path.join(self.output_directory, "mission_summary.json")
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True,
                      ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, final)
        return final

    def _wait_for_operator_start(self):
        self._stop()
        rospy.wait_for_service("/gazebo/pause_physics", timeout=15.0)
        rospy.wait_for_service("/gazebo/unpause_physics", timeout=15.0)
        rospy.ServiceProxy("/gazebo/pause_physics", Empty)()
        self._set_state("WAITING_FOR_OPERATOR",
                        "READY_TO_RECORD: simulation paused; press Enter or 1 in the launch terminal")
        requested = threading.Event()

        def start(_request):
            requested.set()
            return TriggerResponse(success=True, message="Operator start accepted")

        service = rospy.Service("/deform/mission/start", Trigger, start)
        try:
            # Wall-clock wait: Gazebo /clock is deliberately stopped here.
            while not requested.wait(0.1):
                if rospy.is_shutdown():
                    raise MissionFailure("cancelled while waiting for operator start")
        finally:
            service.shutdown()
        if rospy.is_shutdown():
            raise MissionFailure("cancelled before operator start")
        rospy.ServiceProxy("/gazebo/unpause_physics", Empty)()
        self._set_state("OPERATOR_STARTED", "recording preparation complete; experiment starting")

    def _run(self):
        try:
            self._set_state("WAITING_FOR_INPUTS",
                            "waiting for FAST-LIO odometry and map")
            if ((not self.skip_outbound_navigation
                 or self.return_after_detection)
                    and not self._wait_for_move_base(90.0)):
                raise MissionFailure("move_base action server unavailable")
            self._wait(lambda: self._snapshot("lio_odom"), 90.0,
                       "FAST-LIO odometry")
            self._wait(lambda: self._snapshot("navigation_map"), 90.0,
                       "online occupancy map")
            if rospy.get_param("~mission/wait_for_operator", False):
                self._wait_for_operator_start()
            map_robot = pose2d(self._snapshot("lio_odom"))
            self.home_map = map_robot
            if self.skip_outbound_navigation:
                # Detection benchmark mode intentionally starts at the
                # observation pose.  Its local camera_init origin is the
                # target; no inaccessible pre-existing map coordinate is
                # fabricated after teleport/spawn.
                self.target_map = map_robot[:2]
            else:
                self.target_map = self.approach_map

            if not self.skip_outbound_navigation:
                self._set_state("NAVIGATING_OUTBOUND",
                                "known-free planning to camera_init ({:.2f}, {:.2f})".format(
                                    self.target_map[0], self.target_map[1]))
                self._navigate(self.target_map, self.arrival_tolerance, True,
                               "target approach",
                               desired_yaw=self.approach_yaw_map,
                               yaw_tolerance=self.arrival_yaw_tolerance)
            else:
                self._set_state(
                    "DIRECT_MONITORING_START",
                    "robot spawned at the observation pose; outbound navigation omitted")
            self.arrival_map_pose = pose2d(self._snapshot("lio_odom"))
            self._set_state("STABILIZING_AT_TARGET",
                            "robot stopped before reference construction")
            self._stop()
            self._prepare_motion_scene()
            time.sleep(self.stable_dwell)

            self._set_state("BUILDING_REFERENCE",
                            "freezing a generic sensor-local deformation reference")
            time.sleep(self.reference_preparation_dwell)
            self._start_monitoring()
            self._set_state(
                "MONITORING",
                "micro-deformation detection active; reference frozen; "
                "starting configured Gazebo excitation")
            self._trigger_motion()
            if self.full_process_observation_only:
                observation = self._observe_full_motion_process()
                self._set_state(
                    "MONITORING_COMPLETE",
                    ("full-process observation completed; "
                     "persistent_trigger={} algorithm_sources={}").format(
                         observation["trigger_recorded"],
                         ",".join(observation["algorithm_source_ids"])))
                evaluation = self._wait(
                    lambda: (self._snapshot("evaluation_status")
                             if self._snapshot("evaluation_status") is not None
                             and self._snapshot("evaluation_status").get(
                                 "success", False)
                             and int(self._snapshot(
                                 "evaluation_status").get(
                                     "source_accepted_frame_index", -1))
                             >= observation["last_processed_index"]
                             else None),
                    self.evaluation_timeout,
                    "stationary deformation-map export after full-process observation")
                rospy.loginfo("[closed_loop] stationary full-process export ready: %s",
                              evaluation.get("output_directory", ""))
                self._set_state(
                    "COMPLETED",
                    "stationary full-process deformation observation completed")
                self._wait_for_recorder_finalization()
                output = self._summary("COMPLETED")
                rospy.loginfo("[closed_loop] experiment summary: %s", output)
                rospy.signal_shutdown("stationary full-process benchmark completed")
                return
            self._wait_detection()

            self._set_state("DEFORMATION_DETECTED",
                            "generic single-GDU retained cause >= {:.1f}s; "
                            "trigger_source={} observed_sources={}".format(
                                self.return_trigger_persistence,
                                self.return_trigger_algorithm_source_id,
                                ",".join(self.detected_source_ids)))
            # Use the cutoff just published to all consumers, not a newer
            # asynchronous evolution frame that can arrive after the event.
            detected_index = int(self.recording_cutoff_accepted_frame_index)
            # Capture the stationary analysis inputs and owner's atomic map
            # snapshot, but never hold the robot for plots/export completion.
            # Stationary-only runs still wait for their completed export.
            def departure_inputs_ready():
                status = self._snapshot("evaluation_status")
                if status is None or int(status.get("source_accepted_frame_index", -1)) < detected_index:
                    return None
                if not self.return_after_detection:
                    return status if status.get("success", False) else None
                captured = status.get("capture_ready", False) or status.get("success", False)
                map_path = os.path.join(self.output_directory, "navigation", "maps", "decision_map.npz")
                return status if captured and os.path.isfile(map_path) else None
            evaluation = self._wait(
                departure_inputs_ready,
                self.evaluation_timeout,
                "stationary input capture and retained decision map")
            rospy.loginfo("[closed_loop] departure inputs captured; export_complete=%s; source_frame=%s",
                          evaluation.get("success", False), evaluation.get("source_accepted_frame_index"))
            # Route differences are an analysis result, never permission to
            # evacuate. The same corridor may be the only geometric route.
            if not self.return_after_detection:
                if self.stop_motion_on_return:
                    try:
                        self._motion_call("stop")
                    except Exception as error:
                        rospy.logwarn("failed to stop target motion: %s", error)
                self._set_state(
                    "COMPLETED",
                    "direct deformation-detection benchmark completed")
                self._wait_for_recorder_finalization()
                output = self._summary("COMPLETED")
                rospy.loginfo("[closed_loop] experiment summary: %s", output)
                rospy.signal_shutdown("direct detection benchmark completed")
                return

            # The controlled Gazebo excitation has already served its purpose
            # once the detection snapshot and route comparison are frozen.
            # Stop it before navigation so a later scheduled excitation does
            # not become moving foreground for FAST-LIO during the return.
            # The latched algorithm causes and avoidance map are retained.
            if self.stop_motion_on_return:
                try:
                    self._motion_call("stop")
                except Exception as error:
                    raise MissionFailure(
                        "failed to stop target excitation before return: {}"
                        .format(error))
            self._set_state("RETURNING_HOME",
                            "move_base planning on live SLAM + retained deformation costs; "
                            "do not stop until RUN_COMPLETE")
            self._return_home()
            self._set_state(
                "COMPLETED",
                "robot returned; finalizing files, do not stop yet")
            self._wait_for_recorder_finalization()
            output = self._summary("COMPLETED")
            rospy.logwarn(
                "[closed_loop] ===== RUN_COMPLETE ===== robot stopped at HOME; "
                "press Ctrl+C when inspection is finished; summary=%s",
                output)
        except Exception as error:
            self.failure = str(error)
            self._stop()
            self._set_state("FAILED", self.failure)
            output = self._summary("FAILED")
            rospy.logerr("[closed_loop] mission failed; summary: %s", output)
            rospy.signal_shutdown("closed-loop mission failed")


def main():
    if "--operator-start" in rospy.myargv():
        rospy.init_node("icra2027_operator_start")
        rospy.wait_for_service("/deform/mission/start")
        # roslaunch owns the wrapper's stdout pipeline; read the controlling
        # terminal explicitly so the background helper cannot consume logs.
        with open("/dev/tty", encoding="utf-8") as reader, open(
                "/dev/tty", "w", encoding="utf-8", buffering=1) as terminal:
            termios.tcflush(reader, termios.TCIFLUSH)
            if confirm_video_start(reader, terminal):
                response = rospy.ServiceProxy("/deform/mission/start", Trigger)()
                terminal.write("\n{}\n".format(response.message))
                if not response.success:
                    raise RuntimeError(response.message)
        return
    rospy.init_node("icra2027_closed_loop_mission")
    ClosedLoopMission()
    rospy.spin()


if __name__ == "__main__":
    main()
