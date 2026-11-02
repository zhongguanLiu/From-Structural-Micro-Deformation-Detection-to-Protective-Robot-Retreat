#!/usr/bin/env python3
import pathlib
import math
import sys
import unittest

import numpy as np

SCRIPT_ROOT = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))

from planning_core import (  # noqa: E402
    GridGeometry,
    astar,
    choose_comparison_goal,
    complete_forecast_horizons,
    crop_occupancy_grid,
    deformation_changes_route,
    first_rotation_clearance_index,
    fuse_planning_layers,
    inflate_obstacles,
    lidar_observation_grid,
    nearest_traversable,
    path_length_m,
    prediction_window_covers_horizons,
    select_latest_forecast_candidate,
    snapshot_identity_matches,
)


class PlanningCoreTest(unittest.TestCase):
    def test_snapshot_pair_requires_id_index_and_stamp(self):
        self.assertTrue(snapshot_identity_matches(
            "snapshot-a", 17, 12.5, "snapshot-a", 17, 12.5))
        self.assertFalse(snapshot_identity_matches(
            "snapshot-a", 17, 12.5, "snapshot-b", 17, 12.5))
        self.assertFalse(snapshot_identity_matches(
            "snapshot-a", 17, 12.5, "snapshot-a", 18, 12.5))
        self.assertFalse(snapshot_identity_matches(
            "snapshot-a", 17, 12.5, "snapshot-a", 17, 12.6))

    def test_complete_forecast_bundle_rejects_missing_duplicate_and_extra(self):
        expected = [0.5, 1.0, 2.0]
        self.assertTrue(complete_forecast_horizons(
            [2.0, 0.5, 1.0], 3, expected))
        self.assertFalse(complete_forecast_horizons(
            [0.5, 1.0], 2, expected))
        self.assertFalse(complete_forecast_horizons(
            [0.5, 1.0, 1.0], 3, expected))
        self.assertFalse(complete_forecast_horizons(
            [0.5, 1.0, 2.0, 3.0], 4, expected))
        self.assertFalse(complete_forecast_horizons(
            expected, 2, expected))

    def test_latest_forecast_candidate_obeys_capture_cutoff(self):
        candidates = [
            {"source_snapshot_id": "s100",
             "source_accepted_frame_index": 100,
             "source_stamp_s": 10.0},
            {"source_snapshot_id": "s103",
             "source_accepted_frame_index": 103,
             "source_stamp_s": 10.3},
            {"source_snapshot_id": "s101",
             "source_accepted_frame_index": 101,
             "source_stamp_s": 10.1},
        ]
        selected = select_latest_forecast_candidate(candidates, 102)
        self.assertEqual(selected["source_snapshot_id"], "s101")
        self.assertIsNone(select_latest_forecast_candidate(candidates, 99))

    def test_prediction_ttl_must_match_and_cover_farthest_horizon(self):
        expected = [0.5, 1.0, 2.0]
        self.assertTrue(prediction_window_covers_horizons(
            10.0, 12.0, 12.0, expected))
        self.assertTrue(prediction_window_covers_horizons(
            0.0, 2.0, 2.0, expected))
        self.assertFalse(prediction_window_covers_horizons(
            10.0, 11.999, 11.999, expected))
        self.assertFalse(prediction_window_covers_horizons(
            10.0, 12.0, 12.1, expected))
        self.assertFalse(prediction_window_covers_horizons(
            10.0, 0.0, 0.0, expected))

    def test_deformation_route_contract_requires_change_and_exposure(self):
        self.assertTrue(deformation_changes_route(True, 1))
        self.assertFalse(deformation_changes_route(False, 1))
        self.assertFalse(deformation_changes_route(True, 0))
        self.assertFalse(deformation_changes_route(False, 0))
        self.assertFalse(deformation_changes_route(True, float("nan")))
        self.assertFalse(deformation_changes_route(True, "invalid"))

    def test_soft_start_not_moved_or_erased_and_only_exit_is_allowed(self):
        occupancy = np.zeros((1, 12), dtype=np.int16)
        overlay = np.full_like(occupancy, 99)
        overlay[:, 7:] = 0
        original = overlay.copy()
        self.assertEqual(nearest_traversable(occupancy, (1, 0), overlay), (1, 0))
        path, cost = astar(occupancy, (1, 0), (10, 0), overlay,
                           unknown_is_traversable=False)
        self.assertEqual(path, [(x, 0) for x in range(1, 11)])
        self.assertAlmostEqual(cost, 9 + 4 * (5 + .5))
        np.testing.assert_array_equal(overlay, original)

    def test_full_soft_map_and_soft_goal_remain_reachable(self):
        occupancy = np.zeros((8, 8), dtype=np.int16)
        # Even a legacy deformation-only 100 is capped, not treated as geometry.
        overlay = np.full_like(occupancy, 100)
        fused = fuse_planning_layers(occupancy, overlay)
        self.assertTrue(np.all(fused == 99))
        first, cost = astar(occupancy, (1, 1), (6, 6), overlay)
        second, fused_cost = astar(fused, (1, 1), (6, 6))
        self.assertEqual(first, second)
        self.assertAlmostEqual(cost, fused_cost)
        self.assertAlmostEqual(cost, 5 * math.sqrt(2) * 5)

    def test_real_walls_unknown_and_corners_stay_blocked(self):
        occupancy = np.zeros((8, 8), dtype=np.int16)
        overlay = np.full_like(occupancy, 99)
        for barrier in (100, -1):
            occupancy[:, 4] = barrier
            self.assertIsNone(astar(occupancy, (1, 3), (6, 3), overlay,
                                     unknown_is_traversable=False)[0])
        self.assertIsNone(astar(np.array([[0, 100], [100, 0]]),
                                 (0, 0), (1, 1))[0])
        occupancy[3, 1] = 100
        self.assertIsNone(astar(occupancy, (1, 3), (2, 3), overlay)[0])

    def test_invalid_costs_and_outside_endpoints(self):
        occupancy = np.zeros((2, 2))
        for weight in (-1, float("nan")):
            with self.assertRaises(ValueError):
                astar(occupancy, (0, 0), (1, 1), warning_weight=weight)
        self.assertIsNone(astar(occupancy, (-1, 0), (1, 1))[0])
        with self.assertRaises(ValueError):
            astar(occupancy, (0, 0), (1, 1), np.full((2, 2), float("nan")))

    def test_inflation_uses_circle_to_occupied_cell_distance(self):
        occupancy = np.zeros((9, 9), dtype=np.int16)
        occupancy[4, 4] = 100
        inflated = inflate_obstacles(occupancy, 0.42, 0.15)
        # Three cells away: centre distance is 0.45 m, but the nearest edge
        # of the occupied 0.15 m square is only 0.375 m away.
        self.assertEqual(inflated[4, 7], 100)
        self.assertEqual(inflated[4, 8], 0)

    def test_lidar_rays_create_observed_free_and_obstacle_cells(self):
        geometry = GridGeometry(1.0, 10, 10, 0.0, 0.0)
        points = np.array([[8.5, 5.5, 0.0], [7.5, 5.5, 1.0]])
        occupancy, free_count, obstacle_count = lidar_observation_grid(
            points, (1.5, 5.5), geometry, floor_z=0.0,
            maximum_range=9.0)
        self.assertGreater(free_count[5, 4], 0)
        self.assertEqual(occupancy[5, 7], 100)
        self.assertGreater(obstacle_count[5, 7], 0)

    def test_finite_deformation_prefers_a_longer_detour(self):
        geometry = GridGeometry(1.0, 20, 11, 0.0, 0.0)
        occupancy = np.zeros((11, 20), dtype=np.int16)
        overlay = np.zeros_like(occupancy)
        overlay[3:8, 9:12] = 100
        baseline, _ = astar(occupancy, (1, 5), (18, 5))
        avoided, _ = astar(occupancy, (1, 5), (18, 5), overlay)
        self.assertIsNotNone(avoided)
        self.assertGreater(path_length_m(avoided, 1.0),
                           path_length_m(baseline, 1.0))
        self.assertTrue(all(overlay[y, x] == 0 for x, y in avoided))
        fused_path, _ = astar(fuse_planning_layers(occupancy, overlay), (1, 5), (18, 5))
        self.assertEqual(avoided, fused_path)

    def test_goal_selection_produces_two_paths_without_moving_start(self):
        geometry = GridGeometry(0.25, 48, 48, -6.0, -6.0)
        occupancy = np.zeros((48, 48), dtype=np.int16)
        overlay = np.zeros_like(occupancy)
        overlay[20:29, 27:31] = 100
        result = choose_comparison_goal(
            occupancy, overlay, (24, 24), geometry, 2.0, 4.0)
        self.assertIsNotNone(result)
        self.assertEqual(result["baseline_path"][0], (24, 24))
        self.assertEqual(result["avoidance_path"][0], (24, 24))
        self.assertGreater(result["baseline_overlay_max"], 0.0)

    def test_crop_uses_world_coordinates_not_array_indices(self):
        target = GridGeometry(1.0, 2, 2, 1.0, 1.0)
        source = np.arange(16, dtype=np.int16).reshape(4, 4)
        cropped = crop_occupancy_grid(
            source.reshape(-1), 1.0, (0.0, 0.0), 4, 4, target)
        np.testing.assert_array_equal(cropped, np.array([[5, 6], [9, 10]]))

    def test_fused_export_preserves_slam_unknown(self):
        occupancy = np.array([[-1, 0, 100, 20]], dtype=np.int16)
        deformation = np.array([[100, 80, 40, 70]], dtype=np.int16)
        fused = fuse_planning_layers(occupancy, deformation)
        np.testing.assert_array_equal(
            fused, np.array([[-1, 80, 100, 70]], dtype=np.int16))

    def test_global_evacuation_cannot_cross_unknown(self):
        occupancy = np.zeros((5, 7), dtype=np.int16)
        occupancy[:, 3] = -1
        permissive, _ = astar(
            occupancy, (1, 2), (5, 2), unknown_is_traversable=True)
        strict, _ = astar(
            occupancy, (1, 2), (5, 2), unknown_is_traversable=False)
        self.assertIsNotNone(permissive)
        self.assertIsNone(strict)

    def test_rotation_transition_requires_known_circular_clearance(self):
        occupancy = np.zeros((15, 20), dtype=np.int16)
        occupancy[:, 7] = 100
        path = [(column, 7) for column in range(1, 19)]
        # The path passes a wall at column 7; only a later point has the full
        # two-cell clearance neighborhood required for an arbitrary-yaw turn.
        transition = first_rotation_clearance_index(
            occupancy, path, 1.0, minimum_path_distance_m=7.0,
            clearance_radius_m=2.0)
        self.assertIsNotNone(transition)
        self.assertGreaterEqual(path[transition[0]][0], 10)

    def test_rotation_transition_rejects_unknown_clearance(self):
        occupancy = np.zeros((9, 12), dtype=np.int16)
        occupancy[:, 6:] = -1
        path = [(column, 4) for column in range(1, 11)]
        transition = first_rotation_clearance_index(
            occupancy, path, 1.0, minimum_path_distance_m=5.0,
            clearance_radius_m=1.5)
        self.assertIsNone(transition)


if __name__ == "__main__":
    unittest.main()
