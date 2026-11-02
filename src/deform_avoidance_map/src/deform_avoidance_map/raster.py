"""Pure, deterministic occupancy-grid operations used by the ROS node."""

import math

import numpy as np
from scipy.spatial import cKDTree


class GridSpec:
    def __init__(self, resolution, width, height, origin_x, origin_y):
        self.resolution = float(resolution)
        self.width = int(width)
        self.height = int(height)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        if (not math.isfinite(self.resolution) or self.resolution <= 0.0
                or self.width <= 0 or self.height <= 0
                or not math.isfinite(self.origin_x)
                or not math.isfinite(self.origin_y)):
            raise ValueError("invalid grid specification")

    @classmethod
    def from_metric_bounds(cls, resolution, width_m, height_m,
                           origin_x, origin_y):
        width = int(math.ceil(float(width_m) / float(resolution)))
        height = int(math.ceil(float(height_m) / float(resolution)))
        return cls(resolution, width, height, origin_x, origin_y)

    def world_to_cell(self, x, y):
        column = int(math.floor((x - self.origin_x) / self.resolution))
        row = int(math.floor((y - self.origin_y) / self.resolution))
        if column < 0 or row < 0 or column >= self.width or row >= self.height:
            return None
        return column, row

    def index(self, column, row):
        return row * self.width + column

    def circle_within_grid(self, x, y, radius):
        if not all(math.isfinite(value) for value in (x, y, radius)):
            return False
        radius = max(0.0, radius)
        maximum_x = self.origin_x + self.width * self.resolution
        maximum_y = self.origin_y + self.height * self.resolution
        return (x - radius >= self.origin_x and x + radius <= maximum_x
                and y - radius >= self.origin_y and y + radius <= maximum_y)


def _cost_at_distance(distance, hard_radius, warning_radius,
                      hard_cost, warning_peak_cost):
    """Evaluate the frozen radial policy at an exact metric distance."""
    if distance <= hard_radius + 1.0e-12:
        return hard_cost
    if distance >= warning_radius - 1.0e-12:
        return 0
    if warning_radius <= hard_radius:
        return 0
    fraction = (warning_radius - distance) / (
        warning_radius - hard_radius)
    return max(1, min(warning_peak_cost,
                      int(round(warning_peak_cost * fraction))))


def _candidate_bounds(spec, coordinate, radius, origin, count):
    """Return cells whose centers can lie within a metric source radius."""
    lower = int(math.ceil(
        (coordinate - radius - origin) / spec.resolution - 0.5))
    upper = int(math.floor(
        (coordinate + radius - origin) / spec.resolution - 0.5))
    return max(0, lower), min(count - 1, upper)


def _accumulate_source(output, spec, x, y, support_radius, hard_radius,
                       warning_radius, hard_cost, warning_peak_cost):
    support = max(0.0, support_radius)
    hard = hard_radius + support
    warning = warning_radius + support
    first_column, last_column = _candidate_bounds(
        spec, x, warning, spec.origin_x, spec.width)
    first_row, last_row = _candidate_bounds(
        spec, y, warning, spec.origin_y, spec.height)
    if first_column > last_column or first_row > last_row:
        return
    for row in range(first_row, last_row + 1):
        center_y = spec.origin_y + (row + 0.5) * spec.resolution
        for column in range(first_column, last_column + 1):
            center_x = spec.origin_x + (column + 0.5) * spec.resolution
            distance = math.hypot(center_x - x, center_y - y)
            cost = _cost_at_distance(
                distance, hard, warning, hard_cost, warning_peak_cost)
            if cost > 0:
                index = spec.index(column, row)
                if cost > output[index]:
                    output[index] = cost


def rasterize_sources(spec, sources, hard_radius_m, warning_radius_m,
                      hard_cost=99, warning_peak_cost=98):
    """Rasterize finite core/halo costs; legacy ``hard_*`` names mean core.

    Each source is ``(x, y, support_radius)``.  The support radius is part of
    the detected geometry; the configured avoidance radius remains fixed and
    carries no material or collapse semantics.
    """
    if (not all(math.isfinite(value)
                for value in (hard_radius_m, warning_radius_m))
            or hard_radius_m < 0.0 or warning_radius_m < hard_radius_m
            or not 1 <= int(warning_peak_cost) < int(hard_cost) <= 99):
        raise ValueError("avoidance radii are inconsistent")
    return rasterize_policy_sources(
        spec,
        ((x, y, support_radius, hard_radius_m, warning_radius_m)
         for x, y, support_radius in sources),
        hard_cost, warning_peak_cost)


def _rasterize_policy_sources_scalar(spec, sources, hard_cost=99,
                                     warning_peak_cost=98):
    """Reference implementation used to audit the vectorized rasterizer."""
    output = [0] * (spec.width * spec.height)
    for x, y, support_radius, hard_radius, warning_radius in sources:
        values = (x, y, support_radius, hard_radius, warning_radius)
        if not all(math.isfinite(value) for value in values):
            continue
        if hard_radius < 0.0 or warning_radius < hard_radius:
            raise ValueError("source-specific avoidance radii are inconsistent")
        _accumulate_source(
            output, spec, x, y, support_radius, hard_radius, warning_radius,
            hard_cost, warning_peak_cost)
    return output


def rasterize_policy_sources(spec, sources, hard_cost=99,
                             warning_peak_cost=98):
    """Rasterize exact continuous-coordinate radial policies.

    Sources with identical effective hard/warning radii share a monotone cost
    function.  Their pointwise maximum is therefore exactly that function
    evaluated at the nearest source distance.  A deterministic single-worker
    KD-tree query replaces repeated evaluation of overlapping source discs;
    no support point, cell, radius, probability, or forecast is removed.
    """
    if not 1 <= int(warning_peak_cost) < int(hard_cost) <= 99:
        raise ValueError("avoidance costs are inconsistent")
    groups = {}
    for x, y, support_radius, hard_radius, warning_radius in sources:
        values = (x, y, support_radius, hard_radius, warning_radius)
        if not all(math.isfinite(value) for value in values):
            continue
        if hard_radius < 0.0 or warning_radius < hard_radius:
            raise ValueError("source-specific avoidance radii are inconsistent")
        support = max(0.0, support_radius)
        hard = hard_radius + support
        warning = warning_radius + support
        groups.setdefault((hard, warning), []).append((float(x), float(y)))

    output = np.zeros((spec.height, spec.width), dtype=np.uint8)
    if not groups:
        return output.reshape(-1).tolist()
    column_centers = (spec.origin_x
                      + (np.arange(spec.width, dtype=np.float64) + 0.5)
                      * spec.resolution)
    row_centers = (spec.origin_y
                   + (np.arange(spec.height, dtype=np.float64) + 0.5)
                   * spec.resolution)
    for (hard, warning), group_sources in groups.items():
        points = np.asarray(group_sources, dtype=np.float64)
        first_column, _ = _candidate_bounds(
            spec, float(np.min(points[:, 0])), warning,
            spec.origin_x, spec.width)
        _, last_column = _candidate_bounds(
            spec, float(np.max(points[:, 0])), warning,
            spec.origin_x, spec.width)
        first_row, _ = _candidate_bounds(
            spec, float(np.min(points[:, 1])), warning,
            spec.origin_y, spec.height)
        _, last_row = _candidate_bounds(
            spec, float(np.max(points[:, 1])), warning,
            spec.origin_y, spec.height)
        if first_column > last_column or first_row > last_row:
            continue
        local_x = column_centers[first_column:last_column + 1]
        local_y = row_centers[first_row:last_row + 1]
        query = np.empty((len(local_x) * len(local_y), 2), dtype=np.float64)
        query[:, 0] = np.tile(local_x, len(local_y))
        query[:, 1] = np.repeat(local_y, len(local_x))
        # eps=0 is the exact nearest-neighbour query.  Omitting the optional
        # parallel-worker argument keeps deterministic behavior across the
        # SciPy versions distributed with ROS Noetic.
        distance, _ = cKDTree(points).query(query, k=1, eps=0.0)
        distance = distance.reshape((len(local_y), len(local_x)))
        cost = np.zeros(distance.shape, dtype=np.uint8)
        hard_mask = distance <= hard + 1.0e-12
        cost[hard_mask] = int(hard_cost)
        if warning > hard:
            warning_mask = ((~hard_mask)
                            & (distance < warning - 1.0e-12))
            if np.any(warning_mask):
                warning_cost = np.rint(
                    warning_peak_cost * (warning - distance[warning_mask])
                    / (warning - hard))
                warning_cost = np.clip(
                    warning_cost, 1, warning_peak_cost).astype(
                        np.uint8, copy=False)
                cost[warning_mask] = warning_cost
        target = output[first_row:last_row + 1,
                        first_column:last_column + 1]
        np.maximum(target, cost, out=target)
    return output.reshape(-1).tolist()


def _fuse_overlay_scalar(base, overlay):
    """Reference implementation used to audit vectorized max fusion."""
    if len(base) != len(overlay):
        raise ValueError("base and overlay sizes differ")
    fused = []
    for base_value, overlay_value in zip(base, overlay):
        if base_value < 0:
            fused.append(-1)
        else:
            fused.append(max(int(base_value), min(99, int(overlay_value))))
    return fused


def fuse_overlay(base, overlay):
    """Max-fuse finite deformation costs; only SLAM may supply lethal 100.

    Cap legacy overlays at 99 at this boundary as well. UNKNOWN stays UNKNOWN.
    """
    if len(base) != len(overlay):
        raise ValueError("base and overlay sizes differ")
    base_values = np.asarray(base)
    base_integer = base_values.astype(np.int64, copy=False)
    overlay_integer = np.asarray(overlay).astype(np.int64, copy=False)
    fused = np.maximum(base_integer, np.minimum(99, overlay_integer))
    # UNKNOWN is decided from the original base value before integer
    # conversion, matching the scalar contract even for non-integral inputs.
    fused[base_values < 0] = -1
    return fused.tolist()
