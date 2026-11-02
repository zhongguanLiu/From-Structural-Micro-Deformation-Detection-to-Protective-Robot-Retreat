#!/usr/bin/env python3
"""Deterministic 2-D mapping and path comparison for the research exporter.

This module is an evaluation consumer.  It never feeds a pose, label, path or
cost back into the deformation detector.
"""

import heapq
import math
from collections import deque

import numpy as np


class GridGeometry:
    def __init__(self, resolution, width, height, origin_x, origin_y):
        self.resolution = float(resolution)
        self.width = int(width)
        self.height = int(height)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        if (not math.isfinite(self.resolution) or self.resolution <= 0.0
                or self.width <= 0 or self.height <= 0):
            raise ValueError("invalid planning grid")

    @classmethod
    def centered(cls, center_x, center_y, extent_m, resolution):
        cells = int(math.ceil(float(extent_m) / float(resolution)))
        return cls(resolution, cells, cells,
                   center_x - 0.5 * cells * resolution,
                   center_y - 0.5 * cells * resolution)

    def world_to_cell(self, x, y):
        column = int(math.floor((x - self.origin_x) / self.resolution))
        row = int(math.floor((y - self.origin_y) / self.resolution))
        if not (0 <= column < self.width and 0 <= row < self.height):
            return None
        return column, row

    def cell_to_world(self, cell):
        column, row = cell
        return (self.origin_x + (column + 0.5) * self.resolution,
                self.origin_y + (row + 0.5) * self.resolution)


def bresenham(first, second):
    x0, y0 = first
    x1, y1 = second
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    output = []
    while True:
        output.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += sx
        if twice <= dx:
            error += dx
            y0 += sy
    return output


def estimate_floor_z(points, start_xy, radius_m=4.0):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be a finite Nx3 array")
    squared = ((points[:, 0] - start_xy[0]) ** 2
               + (points[:, 1] - start_xy[1]) ** 2)
    local = points[squared <= radius_m * radius_m, 2]
    if local.size < 20:
        local = points[:, 2]
    if local.size == 0:
        raise ValueError("cannot estimate floor from an empty cloud")
    # A low quantile is stable for a stationary ground robot and does not need
    # semantic ground classification.  The result is only used for a vertical
    # collision band in this evaluation map.
    return float(np.percentile(local, 5.0))


def lidar_observation_grid(points, sensor_xy, geometry, floor_z,
                           obstacle_min_height=0.15,
                           obstacle_max_height=1.60,
                           maximum_range=8.0):
    """Ray-carve observed free cells and low-height obstacle endpoints."""
    points = np.asarray(points, dtype=float)
    free_count = np.zeros((geometry.height, geometry.width), dtype=np.uint16)
    obstacle_count = np.zeros_like(free_count)
    start = geometry.world_to_cell(sensor_xy[0], sensor_xy[1])
    if start is None:
        raise ValueError("sensor is outside the local evaluation grid")
    for x, y, z in points:
        distance = math.hypot(x - sensor_xy[0], y - sensor_xy[1])
        if not math.isfinite(distance) or distance <= 0.05 or distance > maximum_range:
            continue
        height = z - floor_z
        is_ground = -0.20 <= height < obstacle_min_height
        is_obstacle = obstacle_min_height <= height <= obstacle_max_height
        if not (is_ground or is_obstacle):
            continue
        endpoint = geometry.world_to_cell(x, y)
        if endpoint is None:
            continue
        ray = bresenham(start, endpoint)
        free_ray = ray[:-1] if is_obstacle else ray
        for column, row in free_ray:
            if free_count[row, column] < np.iinfo(np.uint16).max:
                free_count[row, column] += 1
        if is_obstacle:
            column, row = endpoint
            if obstacle_count[row, column] < np.iinfo(np.uint16).max:
                obstacle_count[row, column] += 1
    occupancy = np.full((geometry.height, geometry.width), -1, dtype=np.int16)
    occupancy[free_count > 0] = 0
    occupancy[obstacle_count >= 1] = 100
    return occupancy, free_count, obstacle_count


def inflate_obstacles(occupancy, radius_m, resolution):
    output = np.array(occupancy, dtype=np.int16, copy=True)
    occupied = np.argwhere(np.asarray(occupancy) >= 100)
    radius_cells = int(math.ceil(max(0.0, radius_m) / resolution))
    offsets = []
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            # Candidate robot centres lie at grid-cell centres, while an
            # occupied cell denotes its full square area.  Inflate when the
            # configured circular footprint intersects that square; centre
            # distance alone under-inflates by half a cell.
            delta_x = max(abs(dx) * resolution - 0.5 * resolution, 0.0)
            delta_y = max(abs(dy) * resolution - 0.5 * resolution, 0.0)
            if math.hypot(delta_x, delta_y) <= radius_m + 1.0e-12:
                offsets.append((dx, dy))
    height, width = output.shape
    for row, column in occupied:
        for dx, dy in offsets:
            x = column + dx
            y = row + dy
            if 0 <= x < width and 0 <= y < height:
                output[y, x] = 100
    return output


def fuse_planning_layers(occupancy, deformation):
    """Fuse known SLAM cells with deformation cost and preserve UNKNOWN.

    This is only a serialization/view of the two planning inputs.  A* still
    receives the SLAM occupancy and deformation overlay separately so their
    hard/soft semantics are not inferred back from a rendered image.
    """
    occupancy = np.asarray(occupancy)
    deformation = np.asarray(deformation)
    if occupancy.ndim != 2 or deformation.shape != occupancy.shape:
        raise ValueError("occupancy and deformation shapes differ")
    if (not np.isfinite(occupancy).all()
            or not np.isfinite(deformation).all()
            or np.any(occupancy < -1) or np.any(occupancy > 100)
            or np.any(deformation < 0) or np.any(deformation > 100)):
        raise ValueError("planning layers contain invalid costs")
    output = np.array(occupancy, dtype=np.int16, copy=True)
    known = occupancy >= 0
    output[known] = np.maximum(
        occupancy[known], np.minimum(99, deformation[known])).astype(np.int16)
    return output


def nearest_traversable(occupancy, requested, overlay=None,
                        unknown_is_traversable=True):
    """Find a geometrically admissible endpoint, regardless of soft overlay."""
    if requested is None:
        return None
    height, width = occupancy.shape
    start = (min(width - 1, max(0, requested[0])),
             min(height - 1, max(0, requested[1])))
    queue = deque([start])
    visited = {start}
    while queue:
        column, row = queue.popleft()
        if occupancy[row, column] < 100 and (
                unknown_is_traversable or occupancy[row, column] >= 0):
            return column, row
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            candidate = (column + dx, row + dy)
            if candidate in visited:
                continue
            if 0 <= candidate[0] < width and 0 <= candidate[1] < height:
                visited.add(candidate)
                queue.append(candidate)
    return None


def astar(occupancy, start, goal, overlay=None, warning_weight=4.0,
          unknown_penalty=4.0, unknown_is_traversable=True):
    """Eight-connected A*: geometry is hard, deformation is finite soft cost.

    Accept both separate layers and a fused grid. Edge costs integrate the
    endpoint-average soft cost, including exposure at a high-cost start.
    No start clearing, snapping out of deformation, or corner cutting of walls.
    """
    occupancy = np.asarray(occupancy)
    if occupancy.ndim != 2 or start is None or goal is None:
        return None, float("inf")
    height, width = occupancy.shape
    deformation = (np.zeros_like(occupancy, dtype=float) if overlay is None
                   else np.asarray(overlay, dtype=float))
    if deformation.shape != occupancy.shape:
        raise ValueError("overlay and occupancy shapes differ")
    if (not np.isfinite(occupancy).all() or not np.isfinite(deformation).all()
            or np.any(occupancy < -1) or np.any(occupancy > 100)
            or np.any(deformation < 0) or np.any(deformation > 100)
            or not math.isfinite(warning_weight) or warning_weight < 0
            or not math.isfinite(unknown_penalty) or unknown_penalty < 1):
        raise ValueError("invalid planning costs or weights")
    if any(not (0 <= p[0] < width and 0 <= p[1] < height) for p in (start, goal)):
        return None, float("inf")
    soft_cost = np.maximum(np.clip(occupancy, 0, 99), np.minimum(deformation, 99))
    if (occupancy[start[1], start[0]] >= 100
            or occupancy[goal[1], goal[0]] >= 100
            or (not unknown_is_traversable
                and (occupancy[start[1], start[0]] < 0
                     or occupancy[goal[1], goal[0]] < 0))):
        return None, float("inf")
    frontier = [(0.0, 0.0, start)]
    cost = {start: 0.0}
    parent = {}
    moves = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
             (-1, 1, math.sqrt(2.0)), (-1, -1, math.sqrt(2.0)))
    while frontier:
        _, current_cost, current = heapq.heappop(frontier)
        if current_cost > cost.get(current, float("inf")) + 1.0e-12:
            continue
        if current == goal:
            path = [current]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            return path, current_cost
        for dx, dy, step in moves:
            neighbor = (current[0] + dx, current[1] + dy)
            if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
                continue
            if occupancy[neighbor[1], neighbor[0]] >= 100:
                continue
            if (not unknown_is_traversable
                    and occupancy[neighbor[1], neighbor[0]] < 0):
                continue
            if dx != 0 and dy != 0:
                # Only actual geometry/UNKNOWN can exclude diagonal corners.
                adjacent = ((current[0] + dx, current[1]),
                            (current[0], current[1] + dy))
                if any(occupancy[y, x] >= 100
                       or (not unknown_is_traversable
                           and occupancy[y, x] < 0)
                       for x, y in adjacent):
                    continue
            cell_penalty = (unknown_penalty if occupancy[neighbor[1], neighbor[0]] < 0
                            else 1.0)
            deform_penalty = 1.0 + warning_weight * (
                soft_cost[current[1], current[0]]
                + soft_cost[neighbor[1], neighbor[0]]) / 198.0
            candidate_cost = current_cost + step * cell_penalty * deform_penalty
            if candidate_cost + 1.0e-12 >= cost.get(neighbor, float("inf")):
                continue
            cost[neighbor] = candidate_cost
            parent[neighbor] = current
            heuristic = math.hypot(goal[0] - neighbor[0],
                                   goal[1] - neighbor[1])
            heapq.heappush(frontier,
                           (candidate_cost + heuristic, candidate_cost, neighbor))
    return None, float("inf")


def path_length_m(path, resolution):
    if not path or len(path) < 2:
        return 0.0
    return resolution * sum(math.hypot(
        path[index][0] - path[index - 1][0],
        path[index][1] - path[index - 1][1])
        for index in range(1, len(path)))


def first_rotation_clearance_index(occupancy, path, resolution,
                                   minimum_path_distance_m,
                                   clearance_radius_m):
    """Find the first known-free path cell with circular turn clearance.

    ``occupancy`` is already inflated for translational footprint tracking.
    The additional circular neighborhood is deliberately conservative: it
    selects where an arbitrary-yaw rectangular footprint may turn, rather
    than assuming that every translationally valid path cell is a valid
    in-place rotation centre.  UNKNOWN is never counted as clearance.
    """
    grid = np.asarray(occupancy)
    if (grid.ndim != 2 or not path or len(path) < 2
            or not math.isfinite(float(resolution)) or resolution <= 0.0
            or not math.isfinite(float(minimum_path_distance_m))
            or minimum_path_distance_m < 0.0
            or not math.isfinite(float(clearance_radius_m))
            or clearance_radius_m <= 0.0):
        raise ValueError("invalid rotation-clearance request")
    radius_cells = int(math.ceil(clearance_radius_m / resolution))
    offsets = tuple(
        (dx, dy)
        for dy in range(-radius_cells, radius_cells + 1)
        for dx in range(-radius_cells, radius_cells + 1)
        if math.hypot(dx, dy) * resolution
        <= clearance_radius_m + 1.0e-12)
    height, width = grid.shape
    distance = 0.0
    for index, (column, row) in enumerate(path):
        if index:
            previous = path[index - 1]
            distance += resolution * math.hypot(
                column - previous[0], row - previous[1])
        if distance + 1.0e-12 < minimum_path_distance_m:
            continue
        clear = True
        for dx, dy in offsets:
            x_value = column + dx
            y_value = row + dy
            if (not 0 <= x_value < width or not 0 <= y_value < height
                    or grid[y_value, x_value] < 0
                    or grid[y_value, x_value] >= 100):
                clear = False
                break
        if clear and index + 1 < len(path):
            return index, distance
    return None


def choose_comparison_goal(occupancy, overlay, start, geometry,
                           minimum_radius_m=2.0, maximum_radius_m=5.0):
    """Choose a nearby reachable goal that maximizes the deformation detour."""
    height, width = occupancy.shape
    candidates = []
    for radius_m in np.linspace(minimum_radius_m, maximum_radius_m, 4):
        radius_cells = radius_m / geometry.resolution
        for angle in np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False):
            requested = (int(round(start[0] + radius_cells * math.cos(angle))),
                         int(round(start[1] + radius_cells * math.sin(angle))))
            if not (0 <= requested[0] < width and 0 <= requested[1] < height):
                continue
            goal = nearest_traversable(occupancy, requested, overlay)
            if goal is not None:
                candidates.append(goal)
    best = None
    seen = set()
    for goal in candidates:
        if goal in seen or goal == start:
            continue
        seen.add(goal)
        baseline, baseline_cost = astar(occupancy, start, goal)
        avoided, avoided_cost = astar(occupancy, start, goal, overlay)
        if not baseline or not avoided:
            continue
        baseline_overlay_max = max(float(overlay[row, column])
                                   for column, row in baseline)
        length_difference = (path_length_m(avoided, geometry.resolution)
                             - path_length_m(baseline, geometry.resolution))
        score = 0.1 * baseline_overlay_max + 5.0 * max(0.0, length_difference)
        score += 0.01 * max(0.0, avoided_cost - baseline_cost)
        tie = (score, length_difference, baseline_overlay_max,
               -goal[0], -goal[1])
        if best is None or tie > best[0]:
            best = (tie, goal, baseline, avoided, baseline_cost, avoided_cost)
    if best is None:
        return None
    return {
        "goal": best[1],
        "baseline_path": best[2],
        "avoidance_path": best[3],
        "baseline_cost": best[4],
        "avoidance_cost": best[5],
        "score": best[0][0],
        "baseline_overlay_max": best[0][2],
        "length_difference_m": best[0][1],
    }


def deformation_changes_route(paths_differ, baseline_max_deformation_cost):
    """Return true only for a changed route causally exposed to deformation."""
    try:
        cost = float(baseline_max_deformation_cost)
    except (TypeError, ValueError):
        return False
    return bool(paths_differ) and math.isfinite(cost) and cost > 0.0


def snapshot_identity_matches(first_snapshot_id, first_frame_index,
                              first_stamp_s, second_snapshot_id,
                              second_frame_index, second_stamp_s,
                              stamp_tolerance_s=1.0e-9):
    """Return whether two products identify the same atomic source snapshot."""
    if (not isinstance(first_snapshot_id, str)
            or not isinstance(second_snapshot_id, str)
            or not first_snapshot_id
            or first_snapshot_id != second_snapshot_id):
        return False
    try:
        first_index = int(first_frame_index)
        second_index = int(second_frame_index)
        first_stamp = float(first_stamp_s)
        second_stamp = float(second_stamp_s)
        tolerance = float(stamp_tolerance_s)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(
        first_index == second_index
        and first_index >= 0
        and math.isfinite(first_stamp)
        and math.isfinite(second_stamp)
        and math.isfinite(tolerance)
        and tolerance >= 0.0
        and abs(first_stamp - second_stamp) <= tolerance)


def complete_forecast_horizons(horizons, overlay_count,
                               expected_horizons,
                               horizon_tolerance_s=1.0e-9):
    """Validate a one-to-one, duplicate-free set of required horizons."""
    try:
        actual = [float(value) for value in horizons]
        expected = [float(value) for value in expected_horizons]
        count = int(overlay_count)
        tolerance = float(horizon_tolerance_s)
    except (TypeError, ValueError, OverflowError):
        return False
    if (count != len(actual) or not expected
            or not math.isfinite(tolerance) or tolerance < 0.0
            or any(not math.isfinite(value) or value <= 0.0
                   for value in actual + expected)):
        return False
    unmatched = list(actual)
    for expected_value in expected:
        matches = [index for index, value in enumerate(unmatched)
                   if abs(value - expected_value) <= tolerance]
        if len(matches) != 1:
            return False
        unmatched.pop(matches[0])
    return not unmatched


def prediction_window_covers_horizons(source_stamp_s,
                                      avoidance_valid_until_s,
                                      evolution_valid_until_s,
                                      expected_horizons,
                                      tolerance_s=1.0e-9):
    """Validate equal nonzero prediction TTLs spanning every horizon."""
    try:
        source = float(source_stamp_s)
        avoidance_until = float(avoidance_valid_until_s)
        evolution_until = float(evolution_valid_until_s)
        horizons = [float(value) for value in expected_horizons]
        tolerance = float(tolerance_s)
    except (TypeError, ValueError, OverflowError):
        return False
    if (not horizons or any(not math.isfinite(value) or value <= 0.0
                            for value in horizons)
            or not all(math.isfinite(value) for value in (
                source, avoidance_until, evolution_until, tolerance))
            or source < 0.0 or avoidance_until <= 0.0
            or evolution_until <= 0.0 or tolerance < 0.0
            or abs(avoidance_until - evolution_until) > tolerance):
        return False
    return avoidance_until + tolerance >= source + max(horizons)


def select_latest_forecast_candidate(candidates, cutoff_frame_index):
    """Select the newest complete paired candidate at or before the cutoff."""
    try:
        cutoff = int(cutoff_frame_index)
    except (TypeError, ValueError, OverflowError):
        return None
    eligible = []
    for candidate in candidates:
        try:
            frame_index = int(candidate["source_accepted_frame_index"])
            stamp_s = float(candidate["source_stamp_s"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if frame_index <= cutoff and math.isfinite(stamp_s):
            eligible.append((frame_index, stamp_s, candidate))
    return max(eligible, key=lambda item: (item[0], item[1]))[2] if eligible else None


def crop_occupancy_grid(data, source_resolution, source_origin,
                        source_width, source_height, target_geometry):
    source = np.asarray(data, dtype=np.int16).reshape(source_height, source_width)
    output = np.zeros((target_geometry.height, target_geometry.width),
                      dtype=np.int16)
    for row in range(target_geometry.height):
        y = target_geometry.origin_y + (row + 0.5) * target_geometry.resolution
        source_row = int(math.floor((y - source_origin[1]) / source_resolution))
        if not 0 <= source_row < source_height:
            continue
        for column in range(target_geometry.width):
            x = (target_geometry.origin_x
                 + (column + 0.5) * target_geometry.resolution)
            source_column = int(math.floor(
                (x - source_origin[0]) / source_resolution))
            if 0 <= source_column < source_width:
                output[row, column] = source[source_row, source_column]
    return output
