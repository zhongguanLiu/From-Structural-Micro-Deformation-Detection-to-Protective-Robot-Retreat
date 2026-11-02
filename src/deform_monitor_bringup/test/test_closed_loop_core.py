#!/usr/bin/env python3

import ast
import math
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import numpy as np
import yaml

SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
SOURCE_ROOT = os.path.abspath(os.path.join(SCRIPTS, "..", ".."))
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from closed_loop_core import (  # noqa: E402
    bounded_disk_cells,
    compose_planar_body_pose,
    compose_se2,
    forward_path_tracking_command,
    goal_or_frontier,
    map_from_world_anchor,
    pose_within_tolerance,
    persistent_evidence_update,
    retained_source_persistence_update,
    reverse_path_waypoints,
    reverse_path_tracking_command,
    rectangular_footprint_is_traversable,
    recorder_finalization_result,
    select_safe_forward_command,
    select_safe_reverse_command,
    swept_reverse_command_is_traversable,
    swept_rectangular_command_is_traversable,
    transform_point,
    waypoint_passed,
)


class ClosedLoopCoreTest(unittest.TestCase):
    def test_departure_alignment_uses_local_path_bearing_and_keeps_clearance_gate(self):
        with open(os.path.join(SCRIPTS, 'icra2027_closed_loop_mission.py')) as stream:
            tree = ast.parse(stream.read())
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                        and n.name == '_align_path_departure')
        owner = SimpleNamespace(_return_pose=lambda: (0., 0., 0.), _align_heading_in_place=Mock())
        context = dict(math=math, normalize_angle=lambda a: math.atan2(math.sin(a), math.cos(a)),
                       HeadingClearanceLost=RuntimeError, rospy=SimpleNamespace(logwarn=Mock()))
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<path-alignment>', 'exec'), context)
        align = context['_align_path_departure']
        align(owner, [(0., 0.), (1., 0.), (2., 0.)])
        owner._align_heading_in_place.assert_not_called()
        align(owner, [(0., 0.), (0., 1.), (0., 2.), (5., 2.)])
        owner._align_heading_in_place.assert_called_once_with(math.pi/2, math.radians(10))
        owner._align_heading_in_place.side_effect = RuntimeError('no swept clearance')
        align(owner, [(0., 0.), (0., 1.), (0., 2.)])  # No forced unsafe turn.

    def test_return_keeps_configured_controller_before_sending_home(self):
        with open(os.path.join(SCRIPTS, 'icra2027_closed_loop_mission.py')) as stream:
            tree = ast.parse(stream.read())
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                        and n.name == '_return_home')
        client = Mock()
        owner = SimpleNamespace(_wait=Mock(), _current_footprint_is_traversable=Mock(),
                                return_unsafe_hold_timeout=10., home_map=[0., 0.],
                                home_tolerance=.8, _navigate=Mock(),
                                _return_pose=lambda: (2., 2., 0.), _align_heading_in_place=Mock())
        context = dict(math=math, MissionFailure=RuntimeError,
                       HeadingClearanceLost=ValueError,
                       dynamic_reconfigure=SimpleNamespace(client=SimpleNamespace(Client=Mock(return_value=client))),
                       rospy=SimpleNamespace(logwarn=Mock()))
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<return-profile>', 'exec'), context)
        context['_return_home'](owner)
        client.update_configuration.assert_not_called()
        owner._navigate.assert_called_once_with([0., 0.], .8, False, 'deformation-aware return')
        owner._align_heading_in_place.assert_called_once_with(-3*math.pi/4, math.radians(20))

    def test_bounded_export_reports_clipping_without_blocking_departure(self):
        with open(os.path.join(SCRIPTS, 'static_planning_evaluation.py')) as stream:
            tree = ast.parse(stream.read())
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                        and n.name == '_ready')
        stamp = SimpleNamespace(to_sec=lambda: 1.)
        frame = dict(source_snapshot_id='S', source_accepted_frame_index=10,
                     header=SimpleNamespace(stamp=stamp))
        owner = SimpleNamespace(
            finished=False, odom=object(), cloud_chunks=[1, 2], minimum_cloud_frames=2,
            export_mission_evacuation=False, require_detection_state=True,
            mission_state='DEFORMATION_DETECTED', capture_cutoff_frame_index=10,
            use_latest_complete_forecast_before_cutoff=False, require_active_source=True,
            evolution=SimpleNamespace(**frame),
            avoidance=SimpleNamespace(**frame, all_current_sources_within_map=False,
                                      active_source_ids=['G'],
                                      valid_until=SimpleNamespace(is_zero=lambda: True)),
            _snapshot_cloud_entries=lambda: [(1., [0]*600), (1.1, [0]*600)])
        warning = Mock()
        context = dict(rospy=SimpleNamespace(logwarn_throttle=warning,
                       Time=SimpleNamespace(now=lambda: 1.)),
                       snapshot_identity_matches=lambda *args: True)
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<export-ready>', 'exec'), context)
        self.assertTrue(context['_ready'](owner))
        warning.assert_called_once()
        owner.evolution.source_accepted_frame_index = 9
        self.assertFalse(context['_ready'](owner))  # Never accept a pre-cutoff frame.

    def test_departure_waits_for_capture_not_plot_completion(self):
        with open(os.path.join(SCRIPTS, 'icra2027_closed_loop_mission.py')) as stream:
            tree = ast.parse(stream.read())
        function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                        and node.name == 'departure_inputs_ready')
        status = dict(source_accepted_frame_index=10, capture_ready=True, success=False)
        owner = SimpleNamespace(return_after_detection=True, output_directory='/test',
                                _snapshot=lambda name: status)
        fake_path = SimpleNamespace(join=os.path.join, isfile=Mock(return_value=True))
        context = dict(self=owner, detected_index=10, os=SimpleNamespace(path=fake_path))
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<departure-gate>', 'exec'), context)
        ready = context['departure_inputs_ready']
        self.assertIs(ready(), status)  # Export is still running.
        fake_path.isfile.return_value = False
        self.assertIsNone(ready())  # No owner's retained map yet.
        fake_path.isfile.return_value = True
        status['source_accepted_frame_index'] = 9
        self.assertIsNone(ready())
        status['source_accepted_frame_index'] = 10
        status['capture_ready'] = False
        self.assertIsNone(ready())
        status['capture_ready'] = True
        owner.return_after_detection = False
        self.assertIsNone(ready())  # Stationary benchmark still needs export.
        status['success'] = True
        self.assertIs(ready(), status)

    def test_recorder_finalization_handshake_distinguishes_wait_ready_error(self):
        self.assertEqual(recorder_finalization_result(None), (False, ""))
        self.assertEqual(recorder_finalization_result(
            {"finalization_started": True, "finalized": False}),
            (False, ""))
        self.assertEqual(recorder_finalization_result(
            {"finalized": True, "finalization_error": ""}),
            (True, ""))
        self.assertEqual(recorder_finalization_result(
            {"finalized": False, "finalization_error": "disk failure"}),
            (False, "disk failure"))

    def test_persistent_deformation_timer_requires_continuity(self):
        first, duration, ready = persistent_evidence_update(
            None, 12.0, True, 10.0)
        self.assertEqual(first, 12.0)
        self.assertEqual(duration, 0.0)
        self.assertFalse(ready)
        first, duration, ready = persistent_evidence_update(
            first, 22.0, True, 10.0)
        self.assertEqual(duration, 10.0)
        self.assertTrue(ready)
        first, duration, ready = persistent_evidence_update(
            first, 22.1, False, 10.0)
        self.assertIsNone(first)
        self.assertEqual(duration, 0.0)
        self.assertFalse(ready)

    def test_retained_source_timer_never_stitches_different_gdus(self):
        intervals = {}
        for stamp in range(0, 6):
            intervals, durations, ready = retained_source_persistence_update(
                intervals, float(stamp), ("gdu-a",), 10.0)
        self.assertAlmostEqual(durations["gdu-a"], 5.0)
        self.assertEqual(ready, ())
        for stamp in range(6, 12):
            intervals, durations, ready = retained_source_persistence_update(
                intervals, float(stamp), ("gdu-b",), 10.0)
        self.assertEqual(set(intervals), {"gdu-b"})
        self.assertAlmostEqual(durations["gdu-b"], 5.0)
        self.assertEqual(ready, ())

    def test_retained_source_timer_requires_one_same_gdu_for_full_duration(self):
        intervals = {}
        for stamp in range(0, 11):
            intervals, durations, ready = retained_source_persistence_update(
                intervals, float(stamp), ("gdu-a", "gdu-short"), 10.0)
            if stamp == 5:
                intervals, _, _ = retained_source_persistence_update(
                    intervals, stamp + 0.1, ("gdu-a",), 10.0)
        self.assertAlmostEqual(durations["gdu-a"], 10.0)
        self.assertEqual(ready, ("gdu-a",))

    def test_retained_source_timer_resets_only_after_source_removal(self):
        intervals, _, _ = retained_source_persistence_update(
            {}, 1.0, ("gdu-a",), 10.0)
        intervals, durations, ready = retained_source_persistence_update(
            intervals, 1.5, (), 10.0)
        self.assertEqual(intervals, {})
        self.assertEqual(durations, {})
        self.assertEqual(ready, ())

    def test_retained_source_timer_survives_irregular_sampling_gap(self):
        intervals, _, _ = retained_source_persistence_update(
            {}, 1.0, ("gdu-a",), 10.0)
        intervals, durations, ready = retained_source_persistence_update(
            intervals, 12.0, ("gdu-a",), 10.0)
        self.assertEqual(intervals["gdu-a"], (1.0, 12.0))
        self.assertEqual(durations["gdu-a"], 11.0)
        self.assertEqual(ready, ("gdu-a",))

    def test_retained_source_timer_restarts_on_non_monotone_source_clock(self):
        intervals, durations, ready = retained_source_persistence_update(
            {"gdu-a": (2.0, 5.0)}, 4.0, ("gdu-a",), 10.0)
        self.assertEqual(intervals["gdu-a"], (4.0, 4.0))
        self.assertEqual(durations["gdu-a"], 0.0)
        self.assertEqual(ready, ())

    def test_retained_source_timer_rejects_invalid_clock_contract(self):
        with self.assertRaises(ValueError):
            retained_source_persistence_update(
                {}, float("nan"), ("gdu-a",), 10.0)
        with self.assertRaises(ValueError):
            retained_source_persistence_update(
                {}, 1.0, ("gdu-a",), 0.0)

    def test_pass_through_waypoint_requires_crossing_inside_corridor(self):
        self.assertTrue(waypoint_passed(
            (0.0, 0.0), (1.0, 0.0), (1.2, 0.05), 0.10))
        self.assertFalse(waypoint_passed(
            (0.0, 0.0), (1.0, 0.0), (0.9, 0.0), 0.10))
        self.assertFalse(waypoint_passed(
            (0.0, 0.0), (1.0, 0.0), (1.2, 0.20), 0.10))

    def test_soft_start_and_complete_soft_band_remain_reachable(self):
        grid = np.zeros((11, 15), dtype=np.int16)
        grid[:, :8] = 99
        original = grid.copy()
        result = goal_or_frontier(
            grid, (2, 5), (12, 5), 0.1,
            arrival_tolerance_m=0.0, robot_radius_m=0.0)
        self.assertEqual(result[0], "GOAL")
        self.assertEqual(result[2][0], (2, 5))
        np.testing.assert_array_equal(grid, original)
        self.assertTrue(rectangular_footprint_is_traversable(
            grid.ravel(), 15, 11, .1, (0., 0.), (.25, .55, 0.), .15, .15))

    def test_reverse_path_waypoints_validate_endpoints_and_face_backward(self):
        path = [(0.0, 0.0), (0.2, 0.0), (0.4, 0.0),
                (0.6, 0.0), (0.8, 0.0), (1.0, 0.0)]
        waypoints = reverse_path_waypoints(
            path, (0.0, 0.0), (1.0, 0.0), spacing_m=0.5,
            endpoint_tolerance_m=0.1, maximum_step_m=0.25)
        self.assertEqual(waypoints[-1][:2], (1.0, 0.0))
        for waypoint in waypoints:
            self.assertAlmostEqual(abs(waypoint[2]), math.pi, places=12)

    def test_reverse_path_uses_short_first_escape_then_nominal_spacing(self):
        path = [(0.1 * index, 0.0) for index in range(41)]
        waypoints = reverse_path_waypoints(
            path, (0.0, 0.0), (4.0, 0.0), spacing_m=2.0,
            first_spacing_m=0.75, endpoint_tolerance_m=0.1,
            maximum_step_m=0.11)
        self.assertAlmostEqual(waypoints[0][0], 0.8, places=12)
        self.assertAlmostEqual(waypoints[1][0], 2.8, places=12)
        self.assertEqual(waypoints[-1][:2], (4.0, 0.0))

    def test_reverse_path_waypoints_reject_discontinuity(self):
        with self.assertRaises(ValueError):
            reverse_path_waypoints(
                [(0.0, 0.0), (1.0, 0.0)], (0.0, 0.0), (1.0, 0.0),
                maximum_step_m=0.35)

    def test_reverse_path_waypoints_reject_wrong_home(self):
        with self.assertRaises(ValueError):
            reverse_path_waypoints(
                [(0.0, 0.0), (0.2, 0.0)], (0.0, 0.0), (2.0, 0.0),
                endpoint_tolerance_m=0.1, maximum_step_m=0.35)

    def test_target_reset_precedes_reference_freeze(self):
        mission_path = os.path.join(SCRIPTS, "icra2027_closed_loop_mission.py")
        with open(mission_path, "r", encoding="utf-8") as stream:
            source = stream.read()
        run = source[source.index("    def _run(self):"):]
        prepare = run.index("self._prepare_motion_scene()")
        build = run.index('self._set_state("BUILDING_REFERENCE"')
        freeze = run.index("self._start_monitoring()")
        trigger = run.index("self._trigger_motion()")
        self.assertLess(prepare, build)
        self.assertLess(build, freeze)
        self.assertLess(freeze, trigger)

        trigger_body = source[
            source.index("    def _trigger_motion(self):"):
            source.index("    @staticmethod\n    def _retained_algorithm_sources")]
        self.assertNotIn('self._motion_call("reset")', trigger_body)
        self.assertNotIn('self._motion_call("reload")', trigger_body)

    def test_lio_only_arrival_precedes_reference_freeze(self):
        mission_path = os.path.join(SCRIPTS, "icra2027_closed_loop_mission.py")
        with open(mission_path, "r", encoding="utf-8") as stream:
            source = stream.read()
        run = source[source.index("    def _run(self):"):]
        arrival = run.index("self.arrival_map_pose =")
        prepare = run.index("self._prepare_motion_scene()")
        freeze = run.index("self._start_monitoring()")
        self.assertLess(arrival, prepare)
        self.assertLess(prepare, freeze)
        self.assertNotIn("truth_odom", run)
        self.assertNotIn("map_from_world_anchor", run)
        self.assertIn("self.target_map = self.approach_map", run)

    def test_anchor_maps_world_robot_to_map_robot(self):
        world_robot = (2.0, -3.0, math.pi / 2.0)
        map_robot = (0.3, -0.2, -0.1)
        transform = map_from_world_anchor(map_robot, world_robot)
        mapped = compose_se2(transform, world_robot)
        for actual, expected in zip(mapped, map_robot):
            self.assertAlmostEqual(actual, expected, places=9)

    def test_transform_target_uses_anchor_not_semantics(self):
        transform = map_from_world_anchor(
            (0.0, 0.0, 0.0), (2.0, -3.0, math.pi / 2.0))
        target = transform_point(transform, (2.0, 2.0))
        self.assertAlmostEqual(target[0], 5.0, places=9)
        self.assertAlmostEqual(target[1], 0.0, places=9)

    def test_truth_anchor_uses_calibrated_fastlio_body_not_model_origin(self):
        body = compose_planar_body_pose(
            (-4.25, 12.54, 1.505898, math.pi / 2.0),
            (0.2801, 0.0, 0.616928878, 0.0))
        self.assertAlmostEqual(body[0], -4.25, places=9)
        self.assertAlmostEqual(body[1], 12.8201, places=9)
        self.assertAlmostEqual(body[2], 2.122826878, places=9)
        self.assertAlmostEqual(body[3], math.pi / 2.0, places=9)

    def test_goal_is_used_when_reachable_in_known_free_space(self):
        grid = np.zeros((15, 15), dtype=np.int16)
        result = goal_or_frontier(
            grid, (2, 2), (12, 12), 0.1,
            arrival_tolerance_m=0.1, robot_radius_m=0.0)
        self.assertEqual(result[0], "GOAL")
        self.assertEqual(result[1], (12, 12))

    def test_unknown_is_never_in_frontier_path(self):
        grid = np.full((15, 15), -1, dtype=np.int16)
        grid[4:11, 2:8] = 0
        result = goal_or_frontier(
            grid, (3, 7), (13, 7), 0.1,
            arrival_tolerance_m=0.1, robot_radius_m=0.0,
            frontier_backoff_m=0.1)
        self.assertEqual(result[0], "FRONTIER")
        for column, row in result[2]:
            self.assertEqual(grid[row, column], 0)
        self.assertLessEqual(result[1][0], 7)

    def test_goal_outside_current_map_selects_goal_directed_frontier(self):
        grid = np.full((11, 11), -1, dtype=np.int16)
        grid[3:8, 2:9] = 0
        result = goal_or_frontier(
            grid, (3, 5), (30, 5), 0.1,
            arrival_tolerance_m=0.1, robot_radius_m=0.0,
            frontier_backoff_m=0.0)
        self.assertEqual(result[0], "FRONTIER")
        self.assertEqual(result[1][0], 8)

    def test_actual_obstacle_band_is_not_crossed(self):
        grid = np.zeros((11, 11), dtype=np.int16)
        grid[:, 5] = 100
        result = goal_or_frontier(
            grid, (2, 5), (8, 5), 0.1,
            arrival_tolerance_m=0.1, robot_radius_m=0.0)
        self.assertIsNone(result)
        for blocked in (-1, 100):
            grid[5, 2] = blocked
            self.assertIsNone(goal_or_frontier(
                grid, (2, 5), (3, 5), .1, robot_radius_m=0.0))

    def test_blacklist_changes_selected_frontier(self):
        grid = np.full((9, 9), -1, dtype=np.int16)
        grid[2:7, 2:7] = 0
        first = goal_or_frontier(
            grid, (4, 4), (8, 4), 0.1,
            arrival_tolerance_m=0.1, robot_radius_m=0.0,
            frontier_backoff_m=0.0)
        second = goal_or_frontier(
            grid, (4, 4), (8, 4), 0.1,
            arrival_tolerance_m=0.1, robot_radius_m=0.0,
            frontier_backoff_m=0.0, blacklisted={first[1]})
        self.assertNotEqual(first[1], second[1])

    def test_failed_frontier_blacklist_is_bounded_disk(self):
        cells = set(bounded_disk_cells((1, 1), 2, 5, 4))
        self.assertIn((1, 1), cells)
        self.assertIn((3, 1), cells)
        self.assertNotIn((3, 3), cells)
        self.assertNotIn((-1, 1), cells)

    def test_navigation_has_distance_progress_watchdog(self):
        mission_path = os.path.join(SCRIPTS, "icra2027_closed_loop_mission.py")
        with open(mission_path, "r", encoding="utf-8") as stream:
            source = stream.read()
        send_goal = source[
            source.index("    def _send_goal(self,"):
            source.index("    def _navigate(self,")]
        self.assertIn("progress_deadline", send_goal)
        self.assertIn("current_distance > xy_tolerance", send_goal)
        self.assertIn("waypoint_passed(", send_goal)
        self.assertIn("pass_through_cross_track_tolerance", send_goal)
        self.assertIn("self.move_base.cancel_goal()", send_goal)

    def test_return_reuses_move_base_on_frozen_fused_map(self):
        config_path = os.path.abspath(os.path.join(
            SCRIPTS, "..", "config", "icra2027_move_base.yaml"))
        with open(config_path, "r", encoding="utf-8") as stream:
            config = stream.read()
        self.assertIn("max_vel_x: 1.25", config)
        self.assertIn("min_vel_x: -0.30", config)
        self.assertIn("forward_point_distance: 0.25", config)
        self.assertIn("recovery_behavior_enabled: false", config)
        self.assertIn("clearing_rotation_allowed: false", config)
        self.assertNotIn("rotate_recovery:", config)

        mission_path = os.path.join(SCRIPTS, "icra2027_closed_loop_mission.py")
        with open(mission_path, "r", encoding="utf-8") as stream:
            source = stream.read()
        profile = source[
            source.index("    def _return_home(self):"):
            source.index("    def _rotation_clearance_at_pose(")]
        self.assertIn("self._current_footprint_is_traversable", profile)
        self.assertIn('"RETURNING_HOME geometric footprint check"', profile)
        self.assertIn("MOVE_BASE_FUSED_RETURN", profile)
        self.assertIn("self._navigate(self.home_map[:2]", profile)
        self.assertIn("self.home_tolerance, False", profile)
        self.assertNotIn("_track_reverse_fused_path", profile)
        self.assertNotIn("_track_forward_fused_path", profile)
        run = source[source.index("    def _run(self):"):]
        detected = run.index('self._set_state("DEFORMATION_DETECTED"')
        home = run.index("self._return_home()")
        self.assertLess(detected, home)
        self.assertLess(run.index('self._motion_call("stop")', detected), home)
        self.assertNotIn('signal_shutdown("closed-loop mission completed")',
                         run)
        self.assertIn("press Ctrl+C", run)

    def test_navigation_uses_lio_calibrated_robot_centre_without_truth(self):
        root = os.path.abspath(os.path.join(SCRIPTS, ".."))
        for relative in ("config/icra2027_move_base.yaml",
                         "config/icra2027_global_costmap.yaml",
                         "config/icra2027_local_costmap.yaml"):
            with open(os.path.join(root, relative), "r", encoding="utf-8") as stream:
                self.assertIn("robot_base_frame: navigation_base", stream.read())
        with open(os.path.join(SCRIPTS, "icra2027_closed_loop_mission.py"),
                  "r", encoding="utf-8") as stream:
            mission = stream.read()
        self.assertIn('rospy.Subscriber("/navigation/odometry"', mission)
        self.assertNotIn('rospy.Subscriber("/ground_truth/odom"', mission)
        self.assertNotIn("truth_model_to_navigation_base_xyzyaw", mission)
        self.assertNotIn("required_robot_truth_stamp_s", mission)
        for filename, key in [('icra2027_global_costmap.yaml', 'global_costmap'),
                              ('icra2027_local_costmap.yaml', 'local_costmap')]:
            with open(os.path.join(root, 'config', filename)) as stream:
                layer = yaml.safe_load(stream)[key]
            self.assertEqual(layer['inflation_layer'], dict(inflation_radius=1., cost_scaling_factor=2.))
            self.assertEqual(layer['footprint'], [[.36, .30], [.36, -.30], [-.36, -.30], [-.36, .30]])

    def test_reverse_tracker_faces_opposite_path_and_never_commands_forward(self):
        index, linear, angular, error = reverse_path_tracking_command(
            [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
            (0.0, 0.0, math.pi), 0, lookahead_m=0.5)
        self.assertEqual(index, 0)
        self.assertLess(linear, 0.0)
        self.assertAlmostEqual(angular, 0.0, places=7)
        self.assertAlmostEqual(error, 0.0, places=7)

    def test_forward_tracker_faces_path_and_never_commands_reverse(self):
        index, linear, angular, error = forward_path_tracking_command(
            [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
            (0.0, 0.0, 0.0), 0, lookahead_m=0.5)
        self.assertEqual(index, 0)
        self.assertGreater(linear, 0.0)
        self.assertAlmostEqual(angular, 0.0, places=7)
        self.assertAlmostEqual(error, 0.0, places=7)

    def test_forward_lattice_selects_only_swept_safe_command(self):
        grid = np.zeros((41, 41), dtype=np.int16)
        command = select_safe_forward_command(
            grid.reshape(-1), 41, 41, 0.1, (-2.05, -2.05),
            (0.0, 0.0, 0.0), [(0.0, 0.0), (1.0, 0.5)],
            (1.0, 0.5), 0.36, 0.30)
        self.assertIsNotNone(command)
        self.assertGreater(command[0], 0.0)
        self.assertLessEqual(abs(command[1]), 0.30)
        self.assertTrue(swept_rectangular_command_is_traversable(
            grid.reshape(-1), 41, 41, 0.1, (-2.05, -2.05),
            (0.0, 0.0, 0.0), 0.36, 0.30, command[0], command[1]))

    def test_reverse_swept_footprint_rejects_unknown_and_lethal(self):
        grid = np.zeros((21, 21), dtype=np.int16)
        self.assertTrue(swept_reverse_command_is_traversable(
            grid.reshape(-1), 21, 21, 0.1, (-1.05, -1.05),
            (0.0, 0.0, 0.0), 0.2, -0.2, 0.0, 0.5, 0.1))
        grid[10, 8] = -1
        self.assertFalse(swept_reverse_command_is_traversable(
            grid.reshape(-1), 21, 21, 0.1, (-1.05, -1.05),
            (0.0, 0.0, 0.0), 0.2, -0.2, 0.0, 0.5, 0.1))
        grid[10, 8] = 100
        self.assertFalse(swept_reverse_command_is_traversable(
            grid.reshape(-1), 21, 21, 0.1, (-1.05, -1.05),
            (0.0, 0.0, 0.0), 0.2, -0.2, 0.0, 0.5, 0.1))

    def test_reverse_lattice_selects_only_swept_safe_command(self):
        grid = np.zeros((41, 41), dtype=np.int16)
        command = select_safe_reverse_command(
            grid.reshape(-1), 41, 41, 0.1, (-2.05, -2.05),
            (0.0, 0.0, 0.0), [(0.0, 0.0), (-1.0, 0.5)],
            (-1.0, 0.5), 0.36, 0.30)
        self.assertIsNotNone(command)
        self.assertLess(command[0], 0.0)
        self.assertLessEqual(abs(command[1]), 0.15)
        self.assertTrue(swept_rectangular_command_is_traversable(
            grid.reshape(-1), 41, 41, 0.1, (-2.05, -2.05),
            (0.0, 0.0, 0.0), 0.36, 0.30, command[0], command[1]))
        grid[20, 20] = -1
        self.assertIsNone(select_safe_reverse_command(
            grid.reshape(-1), 41, 41, 0.1, (-2.05, -2.05),
            (0.0, 0.0, 0.0), [(0.0, 0.0), (-1.0, 0.5)],
            (-1.0, 0.5), 0.36, 0.30))
    def test_circular_footprint_does_not_add_half_diagonal_to_radius(self):
        # The lethal cell is diagonally separated from the 0.20 m circle.
        # A centre-distance + half-diagonal approximation rejects it, while
        # the exact circle-square distance correctly keeps it outside.
        from closed_loop_core import circular_footprint_is_traversable
        grid = np.zeros((21, 21), dtype=np.int16)
        grid[15, 15] = 100
        self.assertTrue(circular_footprint_is_traversable(
            grid.reshape(-1), 21, 21, 0.1, (-1.05, -1.05),
            (0.23, 0.50), 0.20))

    def test_oriented_rectangle_checks_front_corners_and_unknown(self):
        grid = np.zeros((31, 31), dtype=np.int16)
        origin = (-1.55, -1.55)
        # Cell centred at (0.35, 0.25) overlaps a 0.36 x 0.30 half-extent
        # footprint at yaw zero, even though it is outside the 0.30 m
        # inscribed circle used only for A* connectivity.
        grid[18, 19] = 100
        self.assertFalse(rectangular_footprint_is_traversable(
            grid.reshape(-1), 31, 31, 0.1, origin,
            (0.0, 0.0, 0.0), 0.36, 0.30))
        grid[18, 19] = -1
        self.assertFalse(rectangular_footprint_is_traversable(
            grid.reshape(-1), 31, 31, 0.1, origin,
            (0.0, 0.0, 0.0), 0.36, 0.30))
        grid[18, 19] = 0
        self.assertTrue(rectangular_footprint_is_traversable(
            grid.reshape(-1), 31, 31, 0.1, origin,
            (0.0, 0.0, 0.0), 0.36, 0.30))
    def test_stationary_full_process_skips_navigation_and_keeps_full_observation(self):
        launch_path = os.path.abspath(os.path.join(
            SCRIPTS, "..", "launch", "icra2027_stationary_full_process.launch"))
        with open(launch_path, "r", encoding="utf-8") as stream:
            launch = stream.read()
        self.assertIn('<arg name="skip_outbound_navigation" value="true" />',
                      launch)
        self.assertIn('<arg name="return_after_detection" value="false" />',
                      launch)
        self.assertIn(
            '<arg name="mission_full_process_observation_only" value="true" />',
            launch)
        self.assertIn(
            '<arg name="recording_freeze_on_detection_state" value="false" />',
            launch)
        self.assertIn(
            '<arg name="recording_freeze_on_monitoring_complete" value="true" />',
            launch)
        self.assertIn(
            '<arg name="primary_lidar_downsample" default="1" />', launch)
        self.assertIn(
            '<arg name="secondary_lidar_downsample" default="2" />', launch)
        self.assertIn(
            '<arg name="lidar_update_rate_hz" default="10.0" />', launch)
        self.assertIn(
            '<arg name="lidar_update_rate_hz" '
            'value="$(arg lidar_update_rate_hz)" />', launch)
        self.assertIn(
            '<arg name="mission_full_process_expected_duration_s" value="60.0" />',
            launch)
        self.assertNotIn("mission_require_recorder_timeline_completion", launch)
        self.assertIn("mission_require_recorder_finalization", launch)
        self.assertNotIn("rosbag", launch.lower())

        mission_path = os.path.join(SCRIPTS, "icra2027_closed_loop_mission.py")
        with open(mission_path, "r", encoding="utf-8") as stream:
            source = stream.read()
        run = source[source.index("    def _run(self):"):]
        self.assertIn("if not self.skip_outbound_navigation:", run)
        self.assertIn("self._prepare_motion_scene()", run)
        self.assertIn("self._start_monitoring()", run)
        self.assertIn("self._trigger_motion()", run)
        self.assertIn("self._wait_detection()", run)
        self.assertIn("self._observe_full_motion_process()", run)
        self.assertIn('self._set_state(\n                    "MONITORING_COMPLETE"', run)
        self.assertIn("self._wait_for_recorder_finalization()", run)
        set_state = source[
            source.index("    def _set_state(self,"):
            source.index("    def _wait(self,")]
        self.assertLess(set_state.index("self._write_decision_summary()"),
                        set_state.index("self.status_publisher.publish"))
        self.assertIn('payload["finalization_request"]', set_state)

        recorder_path = os.path.join(
            SCRIPTS, "icra2027_experiment_recorder.py")
        with open(recorder_path, "r", encoding="utf-8") as stream:
            recorder = stream.read()
        self.assertIn("_finalization_high_water", recorder)
        self.assertIn("mission_decision_summary.json", recorder)
        self.assertIn("deformation_capture_freeze_committed", recorder)
        self.assertIn("required_artifact_errors", recorder)

    def test_only_two_primary_icra2027_entry_scripts_are_installed(self):
        package = os.path.abspath(os.path.join(SCRIPTS, ".."))
        entry_scripts = sorted(
            name for name in os.listdir(SCRIPTS)
            if name.startswith("run_icra2027_") and name.endswith(".sh"))
        self.assertEqual(entry_scripts, [
            "run_icra2027_closed_loop.sh",
            "run_icra2027_stationary_full_process.sh",
        ])

        with open(os.path.join(package, "CMakeLists.txt"), "r",
                  encoding="utf-8") as stream:
            cmake = stream.read()
        self.assertIn("scripts/run_icra2027_closed_loop.sh", cmake)
        self.assertIn("scripts/run_icra2027_stationary_full_process.sh", cmake)
        self.assertNotIn("run_icra2027_detection_benchmark", cmake)
        self.assertFalse(os.path.exists(os.path.join(
            package, "launch", "icra2027_detection_benchmark.launch")))

    def test_current_endpoint_tolerances_match_experiment_contract(self):
        config_path = os.path.abspath(os.path.join(
            SCRIPTS, "..", "config", "icra2027_closed_loop.yaml"))
        with open(config_path, "r", encoding="utf-8") as stream:
            config = stream.read()
        self.assertIn("arrival_tolerance_m: 0.80", config)
        self.assertIn("arrival_yaw_tolerance_deg: 45.0", config)
        self.assertIn("home_tolerance_m: 0.80", config)
        self.assertIn("return_forward_max_speed_mps: 1.10", config)

    def test_closed_loop_defaults_to_calibrated_complementary_dual_lidar(self):
        launch_path = os.path.abspath(os.path.join(
            SCRIPTS, "..", "launch", "icra2027_closed_loop.launch"))
        with open(launch_path, "r", encoding="utf-8") as stream:
            source = stream.read()
        self.assertIn(
            '<arg name="enable_rear_lidar" default="true" />', source)
        self.assertIn(
            '<arg name="enable_rear_lidar" value="$(arg enable_rear_lidar)" />',
            source)
        self.assertIn(
            '<arg name="lidar_update_rate_hz" default="10.0" />', source)
        self.assertIn(
            '<arg name="lidar_min_range_m" default="0.5" />', source)
        self.assertIn(
            '<arg name="lidar_max_range_m" default="20.0" />', source)
        self.assertIn(
            '<arg name="lidar_update_rate_hz" '
            'value="$(arg lidar_update_rate_hz)" />', source)
        self.assertIn(
            '<arg name="primary_lidar_downsample" default="1" />', source)
        self.assertIn(
            '<arg name="secondary_lidar_downsample" default="2" />', source)
        self.assertIn(
            '<arg name="primary_lidar_downsample"\n'
            '         value="$(arg primary_lidar_downsample)" />', source)
        self.assertIn(
            '<arg name="secondary_lidar_downsample"\n'
            '         value="$(arg secondary_lidar_downsample)" />', source)
        for parameter, argument in (
                ("configured_sensor_update_rate_hz", "lidar_update_rate_hz"),
                ("configured_min_range_m", "lidar_min_range_m"),
                ("configured_max_range_m", "lidar_max_range_m"),
                ("primary_downsample", "primary_lidar_downsample"),
                ("secondary_downsample", "secondary_lidar_downsample")):
            self.assertIn(
                'name="dual_lidar_contract/{}"'.format(parameter), source)
            self.assertIn('value="$(arg {})"'.format(argument), source)

    def test_icra2027_uses_task_specific_avoidance_clearance(self):
        package = os.path.abspath(os.path.join(SCRIPTS, ".."))
        launch_path = os.path.join(package, "launch", "icra2027_closed_loop.launch")
        with open(launch_path, "r", encoding="utf-8") as stream:
            launch = stream.read()
        self.assertIn("config/icra2027_avoidance.yaml", launch)
        self.assertIn('<arg name="config" value="$(arg avoidance_config)" />',
                      launch)

        config_path = os.path.join(package, "config", "icra2027_avoidance.yaml")
        with open(config_path, "r", encoding="utf-8") as stream:
            config = stream.read()
        self.assertIn("hard_avoidance_radius_m: 2.20", config)
        self.assertIn("warning_radius_m: 3.00", config)
        self.assertIn("elevated_source_min_z_m: 2.70", config)
        self.assertIn("elevated_hard_projection_radius_m: 1.20", config)
        self.assertIn("elevated_warning_projection_radius_m: 1.80", config)
        self.assertIn("vertical_projection_policy: ALL_ACTIVE_SUPPORT_TO_GROUND",
                      config)
        for name in (
                "hard_avoidance_radius_m", "warning_radius_m",
                "elevated_hard_projection_radius_m",
                "elevated_warning_projection_radius_m"):
            self.assertIn(
                '<arg name="{}" default="" />'.format(name), launch)
            self.assertIn('value="$(arg {})"'.format(name), launch)
        avoidance_launch_path = os.path.abspath(os.path.join(
            package, "..", "deform_avoidance_map", "launch", "avoidance.launch"))
        with open(avoidance_launch_path, "r", encoding="utf-8") as stream:
            avoidance_launch = stream.read()
        for name in (
                "hard_avoidance_radius_m", "warning_radius_m",
                "elevated_hard_projection_radius_m",
                "elevated_warning_projection_radius_m"):
            self.assertIn(
                'name="deform_avoidance/{}"'.format(name),
                avoidance_launch)
            self.assertIn(
                "if=\"$(eval arg('{}') != '')\"".format(name),
                avoidance_launch)
            self.assertIn('value="$(arg {})"'.format(name), avoidance_launch)

        mission_config_path = os.path.join(
            package, "config", "icra2027_closed_loop.yaml")
        with open(mission_config_path, "r", encoding="utf-8") as stream:
            mission_config = stream.read()
        self.assertIn("approach_map_xy: [17.84, 6.70]", mission_config)
        self.assertIn('<arg name="robot_y" default="-5.30000" />', launch)

        proxy_path = os.path.join(package, "scripts", "mission_map_proxy.py")
        with open(proxy_path, "r", encoding="utf-8") as stream:
            proxy = stream.read()
        self.assertNotIn("clear_initial_footprint", proxy)
        self.assertIn("fuse_planning_layers", proxy)
        self.assertIn('state in ("DEFORMATION_DETECTED", "MONITORING_COMPLETE")',
                      proxy)
        self.assertIn("self.snapshot_locked = True", proxy)
        self.assertIn("selected_base = self.base", proxy)
        self.assertIn("overlay_on_base(selected_base, selected_overlay)", proxy)
        for layer in ("global", "local"):
            with open(os.path.join(package, "config", "icra2027_" + layer + "_costmap.yaml")) as stream:
                costmap = yaml.safe_load(stream)[layer + "_costmap"]
            self.assertEqual(costmap["static_layer"]["lethal_cost_threshold"], 100)
            self.assertFalse(costmap["static_layer"]["trinary_costmap"])
            self.assertEqual(costmap["static_layer"]["map_topic"], "/mission/navigation_map")

        evaluator_path = os.path.join(
            package, "scripts", "static_planning_evaluation.py")
        with open(evaluator_path, "r", encoding="utf-8") as stream:
            evaluator = stream.read()
        self.assertIn("self.require_local_path_difference", evaluator)
        self.assertIn("local_route_affected_by_deformation", evaluator)
        self.assertIn("selection_rotation_clearance_m", evaluator)
        self.assertIn("rotation_acceptance_tolerance_m", evaluator)
        self.assertIn("planning_global_fused, avoidance_global", evaluator)

        mission_path = os.path.join(
            package, "scripts", "icra2027_closed_loop_mission.py")
        with open(mission_path, "r", encoding="utf-8") as stream:
            mission = stream.read()
        self.assertNotIn("deformation map did not alter the target-to-home route", mission)
        self.assertNotIn("deformation_changes_route(", mission)
        self.assertNotIn("avoidance route has no data-supported rotation-clearance point", evaluator)

        stationary_launch = os.path.join(
            package, "launch", "icra2027_stationary_full_process.launch")
        with open(stationary_launch, "r", encoding="utf-8") as stream:
            stationary = stream.read()
        self.assertIn(
            'name="planning_require_local_path_difference" value="false"',
            stationary)
        self.assertIn(
            'name="planning_use_latest_complete_forecast_before_cutoff"\n'
            '         value="true"', stationary)
        self.assertIn(
            'name="planning_use_latest_complete_forecast_before_cutoff"\n'
            '       default="false"', launch)
        self.assertIn("LATEST_COMPLETE_PAIRED_SNAPSHOT_AT_OR_BEFORE_CAPTURE_CUTOFF",
                      evaluator)
        self.assertIn("14_runtime_forecast_self_consistency.png", evaluator)

    def test_gazebo_truth_configuration_is_recorder_only(self):
        package = os.path.abspath(os.path.join(SCRIPTS, ".."))
        with open(os.path.join(package, "config", "icra2027_closed_loop.yaml"),
                  encoding="utf-8") as stream:
            mission = yaml.safe_load(stream)["mission"]
        with open(os.path.join(package, "config", "icra2027_evaluation.yaml"),
                  encoding="utf-8") as stream:
            evaluation = yaml.safe_load(stream)["mission"]

        forbidden_online_keys = {
            "return_trigger_model",
            "monitor_regions_world",
            "monitor_region_labels",
            "stable_support_regions_world",
            "required_detection_models",
            "model_association_tolerance_m",
            "truth_model_to_fastlio_body_xyzyaw",
            "truth_model_to_navigation_base_xyzyaw",
            "motion_truth_topic",
            "approach_world_xy",
        }
        self.assertFalse(forbidden_online_keys.intersection(mission))
        self.assertEqual(mission["return_trigger_persistence_s"], 10.0)
        self.assertNotIn(
            "return_trigger_maximum_observation_gap_s", mission)
        self.assertEqual(set(evaluation), {
            "required_detection_models",
            "model_association_tolerance_m",
            "truth_model_to_fastlio_body_xyzyaw",
        })
        self.assertEqual(evaluation["required_detection_models"],
                         ["m01", "m02", "m03", "m05", "m06"])
        self.assertEqual(set(evaluation["model_association_tolerance_m"]),
                         {"m01", "m02", "m03", "m04", "m05", "m06"})

        launch_path = os.path.join(
            package, "launch", "icra2027_closed_loop.launch")
        launch = ET.parse(launch_path).getroot()
        launch_args = {item.get("name"): item.get("default")
                       for item in launch.findall("arg")}
        # The canonical structured recorder consumes the truth topic.  The
        # Gazebo package's legacy duplicate CSV writer stays disabled.
        self.assertEqual(launch_args["record_truth"], "false")
        mission_node = next(node for node in launch.findall("node")
                            if node.get("name") ==
                            "icra2027_closed_loop_mission")
        recorder_node = next(node for node in launch.findall("node")
                             if node.get("name") ==
                             "icra2027_experiment_recorder")
        mission_files = [item.get("file") for item in
                         mission_node.findall("rosparam")]
        recorder_files = [item.get("file") for item in
                          recorder_node.findall("rosparam")]
        self.assertEqual(mission_files, ["$(arg mission_config)"])
        self.assertEqual(recorder_files, [
            "$(arg mission_config)", "$(arg evaluation_config)"])
        for node in launch.findall("node"):
            if node is recorder_node:
                continue
            self.assertNotIn("$(arg evaluation_config)", [
                item.get("file") for item in node.findall("rosparam")])

        with open(os.path.join(
                package, "scripts", "icra2027_closed_loop_mission.py"),
                encoding="utf-8") as stream:
            online_source = stream.read()
        for forbidden in (
                "self.motion_truth_topic",
                "self.truth_models",
                "self.return_min_model_distance_m",
                "def _motion_truth_callback",
                'Subscriber("/ground_truth/odom"',
                "self.truth_odom",
                "required_robot_truth_stamp_s"):
            self.assertNotIn(forbidden, online_source)

        summary = online_source[
            online_source.index("    def _summary(self, outcome):"):
            online_source.index("    def _run(self):")]
        for forbidden_field in (
                '"final_world_pose"', '"arrival_world_pose"',
                '"map_from_world_se2"',
                '"truth_model_to_navigation_base_xyzyaw"'):
            self.assertNotIn(forbidden_field, summary)

    def test_detector_failure_counters_are_reported_but_not_trigger_inputs(self):
        mission_path = os.path.join(SCRIPTS, "icra2027_closed_loop_mission.py")
        with open(mission_path, "r", encoding="utf-8") as stream:
            source = stream.read()
        status = source[
            source.index("    def _status_callback(self, message):"):
            source.index("    def _tile_callback(self, message):")]
        for field in (
                "latest_deadline_failure_count",
                "maximum_deadline_failure_count",
                "latest_resource_failure_count",
                "maximum_resource_failure_count"):
            self.assertIn(field, status)
        wait_detection = source[
            source.index("    def _wait_detection(self):"):
            source.index("    def _summary(self, outcome):")]
        self.assertNotIn("deadline_failure_count", wait_detection)
        self.assertNotIn("resource_failure_count", wait_detection)

    def test_arrival_requires_position_and_heading(self):
        tolerance = math.radians(45.0)
        self.assertTrue(pose_within_tolerance(
            (22.75, 6.35, math.radians(40.0)), (22.2, 6.7), 0.8,
            0.0, tolerance))
        self.assertFalse(pose_within_tolerance(
            (22.75, 6.35, math.radians(46.0)), (22.2, 6.7), 0.8,
            0.0, tolerance))
        self.assertFalse(pose_within_tolerance(
            (23.10, 6.35, 0.0), (22.2, 6.7), 0.8, 0.0, tolerance))

    def test_arrival_latches_position_before_heading_alignment(self):
        mission_path = os.path.join(SCRIPTS, "icra2027_closed_loop_mission.py")
        with open(mission_path, "r", encoding="utf-8") as stream:
            source = stream.read()
        navigate = source[
            source.index("    def _navigate(self,"):
            source.index("    def _monitor_center_map(self):")]
        self.assertIn("acceptance_target=(target if mode == \"GOAL\"", navigate)
        self.assertIn("self._align_heading_inside_target_disk(", navigate)
        self.assertIn("and self._pose_reached(target, tolerance))", navigate)

        align = source[
            source.index("    def _align_heading_inside_target_disk("):
            source.index("    def _navigate(self,")]
        self.assertIn("alignment_guard = tolerance + 0.10", align)
        self.assertIn("return False", align)
        self.assertIn("heading alignment exceeded the bounded target guard", align)
        self.assertIn('"min_vel_x": 0.0', align)
        self.assertIn('"max_vel_x": 0.0', align)
        self.assertIn('"xy_goal_tolerance": float(tolerance)', align)
        self.assertIn('"max_vel_x": 1.25', align)

    def test_blocked_return_never_substitutes_exploration_spin(self):
        with open(os.path.join(SCRIPTS, "icra2027_closed_loop_mission.py")) as stream:
            tree = ast.parse(stream.read())
        method = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef) and node.name == "_navigate")
        mission = SimpleNamespace(
            navigation_timeout=10.0, return_goal_progress_timeout=5.0,
            _send_goal=Mock(return_value=False), _stop=Mock(), move_base=Mock(),
            _recover_return_heading=Mock())
        namespace = dict(MissionFailure=RuntimeError,
                         time=SimpleNamespace(monotonic=lambda: 0.),
                         rospy=SimpleNamespace(is_shutdown=lambda: False, logwarn=Mock()))
        exec(compile(ast.Module(body=[method], type_ignores=[]), "mission", "exec"), namespace)
        with self.assertRaisesRegex(RuntimeError, "move_base could not complete HOME"):
            namespace["_navigate"](mission, (0, 0), 0.8, False, "HOME")
        self.assertEqual(mission._send_goal.call_count, 3)
        self.assertEqual(mission._recover_return_heading.call_count, 2)
        mission.move_base.send_goal.assert_not_called()
        mission._send_goal.assert_called_with(
            (0, 0), (0, 0), 10.0, acceptance_target=(0, 0),
            acceptance_xy_tolerance=0.8, progress_timeout=5.0)
        mission._send_goal.return_value = True
        mission._recover_return_heading.reset_mock()
        namespace["_navigate"](mission, (0, 0), 0.8, False, "HOME")
        mission._recover_return_heading.assert_not_called()
        mission._send_goal.return_value = False
        mission._recover_return_heading.side_effect = RuntimeError('no native route')
        with self.assertRaisesRegex(RuntimeError, 'no native route'):
            namespace["_navigate"](mission, (0, 0), .8, False, "HOME")

    def test_return_recovery_only_aligns_to_a_valid_native_plan(self):
        with open(os.path.join(SCRIPTS, 'icra2027_closed_loop_mission.py')) as stream:
            tree = ast.parse(stream.read())
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                        and n.name == '_recover_return_heading')
        pose = lambda x, y: SimpleNamespace(header=SimpleNamespace(frame_id='camera_init'),
                                           pose=SimpleNamespace(position=SimpleNamespace(x=x, y=y)))
        plan = SimpleNamespace(header=SimpleNamespace(frame_id=''),
                               poses=[pose(0., 0.), pose(0., 2.)])
        service = Mock(return_value=SimpleNamespace(plan=plan))
        owner = SimpleNamespace(_return_pose=lambda: (0., 0., 0.),
            _goal_message=lambda *args: SimpleNamespace(target_pose=SimpleNamespace(
                header=SimpleNamespace(frame_id='camera_init'))), _align_path_departure=Mock())
        context = dict(math=math, MissionFailure=RuntimeError, GetPlan=object,
            rospy=SimpleNamespace(wait_for_service=Mock(), ServiceProxy=Mock(return_value=service),
                                  ROSException=TimeoutError, ServiceException=OSError))
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<return-recovery>', 'exec'), context)
        recover = context['_recover_return_heading']
        recover(owner, (0., 2.), .8)
        owner._align_path_departure.assert_called_once_with([(0., 0.), (0., 2.)])
        for poses, frame in (([], 'camera_init'), ([pose(0., 0.)], 'camera_init'),
                             ([pose(float('nan'), 0.)], 'camera_init'),
                             ([pose(1., 0.), pose(0., 2.)], 'camera_init'),
                             ([pose(0., 0.), pose(0., 2.)], 'world')):
            plan.poses = poses
            for p in poses:
                p.header.frame_id = frame
            with self.assertRaisesRegex(RuntimeError, 'no valid native route'):
                recover(owner, (0., 2.), .8)
        with open(os.path.join(SCRIPTS, '..', 'config', 'icra2027_move_base.yaml')) as stream:
            config = stream.read()
        self.assertIn('make_plan_clear_costmap: false', config)
        self.assertIn('make_plan_add_unreachable_goal: false', config)

    def test_runtime_trigger_is_generic_and_same_gdu_only(self):
        mission_path = os.path.join(SCRIPTS, "icra2027_closed_loop_mission.py")
        with open(mission_path, "r", encoding="utf-8") as stream:
            source = stream.read()
        self.assertNotIn(
            'rospy.get_param(prefix + "return_trigger_model"', source)
        self.assertNotIn(
            'rospy.get_param(prefix + "monitor_regions_world"', source)
        self.assertNotIn(
            'rospy.get_param(prefix + "monitor_roi_center_world_xy"', source)
        self.assertNotIn(
            'rospy.get_param(prefix + "stable_support_regions_world"', source)
        self.assertNotIn("_registered_region_coverage", source)
        self.assertIn("retained_source_persistence_update(", source)
        self.assertIn("_retained_algorithm_sources(frame)", source)
        retained_extractor = source[
            source.index("    def _retained_algorithm_sources(frame):"):
            source.index("    def _wait_for_avoidance_source")]
        self.assertIn("gdu.retained_for_avoidance", retained_extractor)
        self.assertNotIn("gdu.active_now", retained_extractor)
        self.assertNotIn("gdu.validity", retained_extractor)
        self.assertIn(
            '"CONTINUOUS_RETAINED_CAUSE_SAME_GDU_ID;"', source)
        self.assertIn(
            '"ACTIVE_NOW_OR_CURRENT_OBSERVABILITY_NOT_REQUIRED_AFTER_BIRTH;"',
            source)
        self.assertNotIn("return_trigger_maximum_observation_gap_s", source)
        wait_detection = source[
            source.index("    def _wait_detection(self):"):
            source.index("    def _summary(self, outcome):")]
        self.assertIn("not frame.complete_snapshot", wait_detection)
        self.assertIn("not frame.input_contract_valid", wait_detection)
        self.assertIn("int(frame.validity) >= 2", wait_detection)
        self.assertIn("source_intervals = {}", wait_detection)
        self.assertIn("_wait_for_avoidance_source(", source)
        self.assertIn(
            '"CURRENT_LIO_POSE_GENERIC_LOCAL_VOLUME"', source)
        self.assertIn(
            '"NO_CROSS_GDU_OR_SPATIAL_EVENT_STITCHING;"', source)

    def test_operator_sees_motion_detection_return_and_completion_milestones(self):
        mission_path = os.path.join(SCRIPTS, "icra2027_closed_loop_mission.py")
        with open(mission_path, "r", encoding="utf-8") as stream:
            source = stream.read()
        for marker in (
                "OBJECT_MOTION_COMMAND_SENT",
                "OBJECT_MOTION_CONFIRMED_RUNNING",
                "EXPERIMENT_CLOCK",
                "MOVE_BASE_FUSED_RETURN",
                "RUN_COMPLETE"):
            self.assertIn(marker, source)
        self.assertIn('"/deform/mission/clock_text"', source)

    def test_generic_monitor_roi_fallback_matches_frozen_yaml(self):
        mission_path = os.path.join(SCRIPTS, "icra2027_closed_loop_mission.py")
        with open(mission_path, "r", encoding="utf-8") as stream:
            source = stream.read()
        self.assertIn(
            'prefix + "monitor_roi_half_extent_m", 10.0', source)
        self.assertIn(
            'prefix + "monitor_roi_below_sensor_m", 1.0', source)
        self.assertIn(
            'prefix + "monitor_roi_above_sensor_m", 6.5', source)

        config_path = os.path.join(
            SOURCE_ROOT, "deform_monitor_bringup", "config",
            "icra2027_closed_loop.yaml")
        with open(config_path, "r", encoding="utf-8") as stream:
            mission = yaml.safe_load(stream)["mission"]
        self.assertEqual(mission["monitor_roi_half_extent_m"], 10.0)
        self.assertEqual(mission["monitor_roi_below_sensor_m"], 1.0)
        self.assertEqual(mission["monitor_roi_above_sensor_m"], 6.5)


if __name__ == "__main__":
    unittest.main()
