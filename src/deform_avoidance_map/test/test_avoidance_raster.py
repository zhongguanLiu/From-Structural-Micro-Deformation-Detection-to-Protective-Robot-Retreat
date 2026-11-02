#!/usr/bin/env python3
from collections import deque
import random
import time
import unittest

from deform_avoidance_map.raster import (
    GridSpec,
    _fuse_overlay_scalar,
    _rasterize_policy_sources_scalar,
    fuse_overlay,
    rasterize_policy_sources,
    rasterize_sources,
)


def shortest_free_path(spec, data, start, goal):
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        (column, row), distance = queue.popleft()
        if (column, row) == goal:
            return distance
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = (column + dc, row + dr)
            if neighbor in visited:
                continue
            nc, nr = neighbor
            if not (0 <= nc < spec.width and 0 <= nr < spec.height):
                continue
            if data[spec.index(nc, nr)] >= 100:
                continue
            visited.add(neighbor)
            queue.append((neighbor, distance + 1))
    return None


class AvoidanceRasterTest(unittest.TestCase):
    def test_vectorized_raster_is_cell_exact_against_scalar_reference(self):
        generator = random.Random(98431)
        spec = GridSpec(0.1, 87, 63, -2.13, -1.77)
        sources = []
        for _ in range(300):
            hard = generator.choice([0.0, 0.06, 0.25, 0.75])
            warning = hard + generator.choice([0.0, 0.01, 0.35, 0.50])
            sources.append((
                generator.uniform(-3.0, 8.0),
                generator.uniform(-3.0, 6.0),
                generator.choice([-0.1, 0.0, 0.03, 0.1]),
                hard, warning))
        expected = _rasterize_policy_sources_scalar(spec, sources, 99, 98)
        actual = rasterize_policy_sources(spec, sources, 99, 98)
        self.assertEqual(actual, expected)
        self.assertEqual(actual,
                         rasterize_policy_sources(spec, sources, 99, 98))

    def test_nearest_distance_preserves_policy_transition_boundaries(self):
        spec = GridSpec(0.1, 25, 5, 0.0, 0.0)
        center_x = 0.05
        center_y = 0.05
        # Include both tolerance boundaries and half-integer warning costs,
        # where a different rounding or distance convention would alter a
        # cell by one cost unit.
        distances = [
            0.25 - 1.0e-12,
            0.25,
            0.25 + 1.0e-12,
            1.25 - 1.0e-12,
            1.25,
        ]
        distances.extend(
            1.25 - (value + 0.5) * (1.25 - 0.25) / 99.0
            for value in (1, 17, 48, 81, 97))
        for distance in distances:
            sources = [(center_x + distance, center_y,
                        0.0, 0.25, 1.25)]
            self.assertEqual(
                rasterize_policy_sources(spec, sources),
                _rasterize_policy_sources_scalar(spec, sources))

    def test_vectorized_8k_raster_has_bounded_runtime(self):
        spec = GridSpec(0.1, 400, 400, -20.0, -20.0)
        sources = [
            (-1.5 + 0.03 * (index % 100),
             -0.1 + 0.003 * (index // 100),
             0.05, 1.5, 2.0)
            for index in range(8000)
        ]
        start = time.monotonic()
        result = rasterize_policy_sources(spec, sources)
        elapsed = time.monotonic() - start
        self.assertEqual(len(result), 160000)
        # The scalar implementation requires several seconds for this exact
        # workload.  Four seconds leaves generous margin on a loaded test host
        # while still preventing accidental restoration of the Python cell
        # loop in the 8k-support operating envelope.
        self.assertLess(elapsed, 4.0)

    def test_finite_core_and_warning_halo(self):
        spec = GridSpec(1.0, 11, 11, 0.0, 0.0)
        data = rasterize_sources(spec, [(5.5, 5.5, 0.0)], 1.0, 3.0)
        self.assertEqual(data[spec.index(5, 5)], 99)
        self.assertEqual(data[spec.index(6, 5)], 99)
        self.assertGreater(data[spec.index(7, 5)], 0)
        self.assertLess(data[spec.index(7, 5)], 100)
        self.assertEqual(data[spec.index(0, 0)], 0)

    def test_subcell_source_is_not_quantized_to_cell_center(self):
        spec = GridSpec(0.1, 4, 3, 0.0, 0.0)
        # The source lies close to the right edge of cell (0, 1).  Both cell
        # centers x=0.05 and x=0.15 are within the exact 0.06 m hard radius.
        # Center-quantized rasterization would incorrectly mark only cell 0.
        data = rasterize_sources(
            spec, [(0.099, 0.15, 0.0)], 0.06, 0.06)
        self.assertEqual(data[spec.index(0, 1)], 99)
        self.assertEqual(data[spec.index(1, 1)], 99)

    def test_source_just_outside_grid_still_rasterizes_overlap(self):
        spec = GridSpec(0.1, 4, 3, 0.0, 0.0)
        data = rasterize_sources(
            spec, [(-0.01, 0.15, 0.0)], 0.07, 0.07)
        self.assertEqual(data[spec.index(0, 1)], 99)
        self.assertEqual(data[spec.index(1, 1)], 0)

    def test_reference_and_current_positions_can_both_be_avoided(self):
        spec = GridSpec(0.5, 20, 10, 0.0, 0.0)
        data = rasterize_sources(
            spec, [(2.0, 2.0, 0.0), (4.0, 2.0, 0.0)], 0.5, 0.5)
        self.assertEqual(data[spec.index(4, 4)], 99)
        self.assertEqual(data[spec.index(8, 4)], 99)

    def test_amplitude_change_does_not_invent_spatial_expansion(self):
        spec = GridSpec(0.5, 20, 10, 0.0, 0.0)
        # Rasterization deliberately accepts geometry, not displacement
        # magnitude.  A fixed GDU whose amplitude rises keeps the same map
        # footprint; magnitude remains an evolution attribute.
        t0 = rasterize_sources(spec, [(4.0, 2.0, 0.1)], 0.5, 1.0)
        t1 = rasterize_sources(spec, [(4.0, 2.0, 0.1)], 0.5, 1.0)
        self.assertEqual(t0, t1)

    def test_support_expands_only_when_new_geometry_is_added(self):
        spec = GridSpec(0.5, 20, 10, 0.0, 0.0)
        t0 = rasterize_sources(spec, [(3.0, 2.0, 0.0)], 0.5, 0.5)
        t1 = rasterize_sources(
            spec, [(3.0, 2.0, 0.0), (7.0, 2.0, 0.0)], 0.5, 0.5)
        self.assertEqual(t0[spec.index(14, 4)], 0)
        self.assertEqual(t1[spec.index(14, 4)], 99)

    def test_deformation_does_not_remove_geometric_connectivity(self):
        spec = GridSpec(1.0, 20, 11, 0.0, 0.0)
        empty = [0] * (spec.width * spec.height)
        direct = shortest_free_path(spec, empty, (1, 5), (18, 5))
        avoided = rasterize_sources(
            spec, [(10.5, 5.5, 0.0)], 2.0, 2.0)
        detour = shortest_free_path(spec, avoided, (1, 5), (18, 5))
        self.assertEqual(direct, 17)
        self.assertIsNotNone(detour)
        self.assertEqual(detour, direct)  # Unweighted reachability, not cost minimization.
        self.assertEqual(avoided[spec.index(10, 5)], 99)

    def test_lethal_deformation_configuration_is_rejected(self):
        spec = GridSpec(1., 3, 3, 0., 0.)
        with self.assertRaises(ValueError):
            rasterize_sources(spec, [(1., 1., 0.)], 1., 2., 100, 99)
        with self.assertRaises(ValueError):
            rasterize_policy_sources(spec, [], 100, 99)

    def test_slam_unknown_is_not_owned_by_deformation_layer(self):
        base = [0, -1, 80, 100]
        overlay = [100, 100, 90, 0]
        self.assertEqual(fuse_overlay(base, overlay), [99, -1, 90, 100])

    def test_vectorized_fusion_matches_scalar_for_non_integral_inputs(self):
        base = [0.9, -0.1, 80.8, 0.0, -1.0]
        overlay = [100.0, 100.0, 90.9, 0.7, -5.0]
        self.assertEqual(fuse_overlay(base, overlay),
                         _fuse_overlay_scalar(base, overlay))

    def test_out_of_bounds_source_is_ignored_transactionally(self):
        spec = GridSpec(1.0, 4, 4, 0.0, 0.0)
        data = rasterize_sources(spec, [(100.0, 100.0, 0.0)], 1.0, 2.0)
        self.assertEqual(data, [0] * 16)

    def test_map_coverage_includes_warning_radius(self):
        spec = GridSpec(1.0, 10, 10, 0.0, 0.0)
        self.assertTrue(spec.circle_within_grid(5.0, 5.0, 2.0))
        self.assertFalse(spec.circle_within_grid(0.5, 5.0, 1.0))
        self.assertFalse(spec.circle_within_grid(9.5, 5.0, 1.0))

    def test_elevated_footprint_penalizes_below_but_leaves_side_lane(self):
        spec = GridSpec(0.1, 120, 80, 0.0, 0.0)
        # A 2 m suspended plate is represented by observed support samples.
        # The task policy joins its footprint with a 0.75 m hard projection;
        # a side lane 1.4 m away remains outside the 1.25 m warning band.
        sources = [(4.0 + 0.1 * index, 4.0, 0.0, 0.75, 1.25)
                   for index in range(21)]
        data = rasterize_policy_sources(spec, sources)
        below = spec.world_to_cell(5.0, 4.0)
        near_side = spec.world_to_cell(5.0, 4.6)
        free_side = spec.world_to_cell(5.0, 5.4)
        self.assertEqual(data[spec.index(*below)], 99)
        self.assertEqual(data[spec.index(*near_side)], 99)
        self.assertEqual(data[spec.index(*free_side)], 0)

    def test_policy_sources_can_mix_ordinary_and_elevated_clearances(self):
        spec = GridSpec(0.1, 120, 80, 0.0, 0.0)
        data = rasterize_policy_sources(spec, [
            (2.0, 2.0, 0.5, 1.5, 2.0),
            (8.0, 2.0, 0.0, 0.25, 0.60),
        ])
        self.assertEqual(data[spec.index(*spec.world_to_cell(3.9, 2.0))], 99)
        self.assertEqual(data[spec.index(*spec.world_to_cell(8.1, 2.0))], 99)
        self.assertEqual(data[spec.index(*spec.world_to_cell(8.8, 2.0))], 0)

    def test_current_and_forecast_rasters_max_fuse_exactly(self):
        spec = GridSpec(0.1, 80, 60, -1.0, -2.0)
        current = [
            (0.099, 0.15, 0.03, 0.25, 0.60),
            (2.35, -0.77, 0.10, 0.75, 1.25),
        ]
        forecast = [
            (0.55, 0.15, 0.12, 0.25, 0.60),
            (3.01, -0.41, 0.05, 0.75, 1.25),
        ]
        concatenated = rasterize_policy_sources(
            spec, current + forecast)
        reused = fuse_overlay(
            rasterize_policy_sources(spec, current),
            rasterize_policy_sources(spec, forecast))
        self.assertEqual(reused, concatenated)


if __name__ == "__main__":
    unittest.main()
