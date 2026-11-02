#!/usr/bin/env python3
"""Pure geometry and frontier planning helpers for the ICRA2027 mission."""

import math
from collections import deque

import numpy as np


def recorder_finalization_result(document):
    """Return ``(ready, error)`` for the recorder's terminal handshake."""
    if not isinstance(document, dict):
        return False, ""
    error = str(document.get("finalization_error", "") or "")
    if error:
        return False, error
    return bool(document.get("finalized", False)), ""


def normalize_angle(angle):
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def compose_se2(first, second):
    """Return T_ac = T_ab * T_bc for (x, y, yaw) tuples."""
    ax, ay, aa = (float(value) for value in first)
    bx, by, ba = (float(value) for value in second)
    cosine = math.cos(aa)
    sine = math.sin(aa)
    return (ax + cosine * bx - sine * by,
            ay + sine * bx + cosine * by,
            normalize_angle(aa + ba))


def inverse_se2(transform):
    x_value, y_value, angle = (float(value) for value in transform)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (-cosine * x_value - sine * y_value,
            sine * x_value - cosine * y_value,
            normalize_angle(-angle))


def transform_point(transform, point):
    x_value, y_value, angle = (float(value) for value in transform)
    px, py = (float(value) for value in point)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (x_value + cosine * px - sine * py,
            y_value + sine * px + cosine * py)


def map_from_world_anchor(map_robot, world_robot):
    """Calibrate T_map_world from simultaneous robot poses."""
    return compose_se2(map_robot, inverse_se2(world_robot))


def compose_planar_body_pose(model_world_pose, model_to_body):
    """Compose a planar Gazebo model pose with a rigid body calibration.

    Both arguments are ``(x, y, z, yaw)``.  The ICRA2027 planar mover keeps
    roll and pitch zero, so the calibrated z translation is additive while
    the horizontal offset is rotated by the model yaw.
    """
    if len(model_world_pose) != 4 or len(model_to_body) != 4:
        raise ValueError("model and body poses must contain x,y,z,yaw")
    model = tuple(float(value) for value in model_world_pose)
    calibration = tuple(float(value) for value in model_to_body)
    if not all(math.isfinite(value) for value in model + calibration):
        raise ValueError("model-to-body calibration must be finite")
    xy_yaw = compose_se2(
        (model[0], model[1], model[3]),
        (calibration[0], calibration[1], calibration[3]))
    return (xy_yaw[0], xy_yaw[1], model[2] + calibration[2], xy_yaw[2])


def pose_within_tolerance(pose, target, xy_tolerance_m,
                          target_yaw=None, yaw_tolerance_rad=None):
    """Return whether a planar pose satisfies the explicit task tolerances."""
    if pose is None or target is None:
        return False
    if (not math.isfinite(float(xy_tolerance_m))
            or float(xy_tolerance_m) < 0.0):
        raise ValueError("xy tolerance must be finite and non-negative")
    distance = math.hypot(float(pose[0]) - float(target[0]),
                          float(pose[1]) - float(target[1]))
    if distance > float(xy_tolerance_m):
        return False
    if target_yaw is None:
        return True
    if (yaw_tolerance_rad is None
            or not math.isfinite(float(yaw_tolerance_rad))
            or float(yaw_tolerance_rad) < 0.0):
        raise ValueError("yaw tolerance must be finite and non-negative")
    return abs(normalize_angle(float(pose[2]) - float(target_yaw))) \
        <= float(yaw_tolerance_rad)


def waypoint_passed(segment_start, waypoint, current,
                    cross_track_tolerance_m):
    """Test whether an intermediate directed-path waypoint was crossed."""
    values = tuple(float(value) for point in (
        segment_start, waypoint, current) for value in point[:2])
    tolerance = float(cross_track_tolerance_m)
    if (not all(math.isfinite(value) for value in values)
            or not math.isfinite(tolerance) or tolerance < 0.0):
        raise ValueError("invalid waypoint pass-through geometry")
    start_x, start_y, target_x, target_y, current_x, current_y = values
    delta_x = target_x - start_x
    delta_y = target_y - start_y
    squared_length = delta_x * delta_x + delta_y * delta_y
    if squared_length <= 1.0e-12:
        raise ValueError("pass-through segment has zero length")
    relative_x = current_x - start_x
    relative_y = current_y - start_y
    along_fraction = (relative_x * delta_x + relative_y * delta_y) \
        / squared_length
    cross_track = abs(delta_x * relative_y - delta_y * relative_x) \
        / math.sqrt(squared_length)
    return along_fraction >= 1.0 and cross_track <= tolerance + 1.0e-12


def reverse_path_tracking_command(path, pose, previous_index,
                                  lookahead_m=0.75, speed_mps=0.35,
                                  heading_gain=1.2,
                                  maximum_angular_speed=0.60,
                                  search_window=30):
    """Return a reverse-only pure-pursuit command on an ordered path.

    The path direction is the direction of travel.  A reversing body must
    therefore face ``travel_heading + pi``.  The nearest-path index is
    monotone, preventing a self-near route from jumping backwards.
    """
    parsed = tuple((float(point[0]), float(point[1])) for point in path)
    state = tuple(float(value) for value in pose[:3])
    index = int(previous_index)
    scalars = (float(lookahead_m), float(speed_mps), float(heading_gain),
               float(maximum_angular_speed))
    if (len(parsed) < 2 or len(state) != 3
            or not all(math.isfinite(value) for point in parsed
                       for value in point)
            or not all(math.isfinite(value) for value in state + scalars)
            or not 0 <= index < len(parsed)
            or lookahead_m <= 0.0 or speed_mps <= 0.0
            or heading_gain <= 0.0 or maximum_angular_speed <= 0.0
            or int(search_window) <= 0):
        raise ValueError("invalid reverse path tracking input")
    stop = min(len(parsed), index + int(search_window) + 1)
    index = min(range(index, stop), key=lambda candidate: math.hypot(
        parsed[candidate][0] - state[0],
        parsed[candidate][1] - state[1]))
    target_index = index
    accumulated = 0.0
    while target_index + 1 < len(parsed) and accumulated < lookahead_m:
        accumulated += math.hypot(
            parsed[target_index + 1][0] - parsed[target_index][0],
            parsed[target_index + 1][1] - parsed[target_index][1])
        target_index += 1
    travel_heading = math.atan2(parsed[target_index][1] - state[1],
                                parsed[target_index][0] - state[0])
    desired_body_heading = normalize_angle(travel_heading + math.pi)
    heading_error = normalize_angle(desired_body_heading - state[2])
    angular = max(-maximum_angular_speed,
                  min(maximum_angular_speed, heading_gain * heading_error))
    # Do not command forward motion.  Large heading errors retain a small
    # reverse component so steering occurs as a swept curve, not an unverified
    # in-place recovery rotation.
    scale = max(0.15, math.cos(abs(heading_error)))
    return index, -float(speed_mps) * scale, angular, heading_error


def forward_path_tracking_command(path, pose, previous_index,
                                  lookahead_m=0.75, speed_mps=0.50,
                                  heading_gain=1.2,
                                  maximum_angular_speed=0.30,
                                  search_window=30):
    """Return a forward-only pure-pursuit command on an ordered path."""
    parsed = tuple((float(point[0]), float(point[1])) for point in path)
    state = tuple(float(value) for value in pose[:3])
    index = int(previous_index)
    scalars = (float(lookahead_m), float(speed_mps), float(heading_gain),
               float(maximum_angular_speed))
    if (len(parsed) < 2 or len(state) != 3
            or not all(math.isfinite(value) for point in parsed
                       for value in point)
            or not all(math.isfinite(value) for value in state + scalars)
            or not 0 <= index < len(parsed)
            or lookahead_m <= 0.0 or speed_mps <= 0.0
            or heading_gain <= 0.0 or maximum_angular_speed <= 0.0
            or int(search_window) <= 0):
        raise ValueError("invalid forward path tracking input")
    stop = min(len(parsed), index + int(search_window) + 1)
    index = min(range(index, stop), key=lambda candidate: math.hypot(
        parsed[candidate][0] - state[0],
        parsed[candidate][1] - state[1]))
    target_index = index
    accumulated = 0.0
    while target_index + 1 < len(parsed) and accumulated < lookahead_m:
        accumulated += math.hypot(
            parsed[target_index + 1][0] - parsed[target_index][0],
            parsed[target_index + 1][1] - parsed[target_index][1])
        target_index += 1
    desired_heading = math.atan2(parsed[target_index][1] - state[1],
                                 parsed[target_index][0] - state[0])
    heading_error = normalize_angle(desired_heading - state[2])
    angular = max(-maximum_angular_speed,
                  min(maximum_angular_speed, heading_gain * heading_error))
    scale = max(0.15, math.cos(abs(heading_error)))
    return index, float(speed_mps) * scale, angular, heading_error


def integrate_unicycle(pose, linear_speed, angular_speed, duration_s):
    """Exact constant-command planar unicycle integration."""
    x_value, y_value, yaw = (float(value) for value in pose[:3])
    linear = float(linear_speed)
    angular = float(angular_speed)
    duration = float(duration_s)
    if not all(math.isfinite(value) for value in (
            x_value, y_value, yaw, linear, angular, duration)) \
            or duration < 0.0:
        raise ValueError("invalid unicycle integration input")
    if abs(angular) <= 1.0e-9:
        return (x_value + linear * math.cos(yaw) * duration,
                y_value + linear * math.sin(yaw) * duration,
                normalize_angle(yaw))
    final_yaw = yaw + angular * duration
    return (x_value + linear / angular
            * (math.sin(final_yaw) - math.sin(yaw)),
            y_value - linear / angular
            * (math.cos(final_yaw) - math.cos(yaw)),
            normalize_angle(final_yaw))


def persistent_evidence_update(first_stamp_s, current_stamp_s, present,
                               required_duration_s):
    """Update a continuous-evidence timer without using wall-clock time."""
    current = float(current_stamp_s)
    required = float(required_duration_s)
    if (not math.isfinite(current) or not math.isfinite(required)
            or required <= 0.0):
        raise ValueError("invalid persistent evidence timer input")
    if not present:
        return None, 0.0, False
    first = current if first_stamp_s is None else float(first_stamp_s)
    if not math.isfinite(first) or current < first:
        first = current
    duration = max(0.0, current - first)
    return first, duration, duration + 1.0e-9 >= required


def retained_source_persistence_update(source_intervals, current_stamp_s,
                                       retained_source_ids,
                                       required_duration_s):
    """Time how long each causal GDU remains retained by the risk map.

    ``source_intervals`` maps a GDU/source ID to the first and latest accepted
    source timestamps for one uninterrupted *retained* lifetime.  The caller
    supplies the IDs whose current evolution record has
    ``retained_for_avoidance=true``.  Once a GDU has passed the evolution
    layer's birth confirmation, temporary loss of ``active_now`` or current
    observability does not erase the already-retained causal hazard.  Its
    timer therefore survives irregular sampling intervals and advances in
    source time until the GDU is absent from a valid complete snapshot (the
    evolution layer has explicitly cleared/removed it).

    Timers are never transferred between IDs.  A non-monotone source clock
    restarts the affected ID, while invalid/incomplete snapshots are rejected
    and reset by the caller.  The helper consumes algorithm output only and
    has no concept of Gazebo model names, semantic classes or registered
    spatial boxes.

    Returns ``(updated_intervals, durations, ready_source_ids)``.
    """
    current = float(current_stamp_s)
    required = float(required_duration_s)
    if (not all(math.isfinite(value) for value in (current, required))
            or required <= 0.0):
        raise ValueError("invalid retained-source persistence timer input")
    if source_intervals is None:
        source_intervals = {}
    if not isinstance(source_intervals, dict):
        raise ValueError("source intervals must be a mapping")
    sources = tuple(sorted(set(str(value) for value in retained_source_ids
                               if str(value))))
    updated = {}
    durations = {}
    ready = []
    for source_id in sources:
        first = current
        old_interval = source_intervals.get(source_id)
        if old_interval is not None:
            if (not isinstance(old_interval, (list, tuple))
                    or len(old_interval) != 2):
                raise ValueError("source interval must contain first and last")
            old_first, old_last = (float(value) for value in old_interval)
            if (math.isfinite(old_first) and math.isfinite(old_last)
                    and old_first <= old_last <= current):
                first = old_first
        duration = max(0.0, current - first)
        updated[source_id] = (first, current)
        durations[source_id] = duration
        if duration + 1.0e-9 >= required:
            ready.append(source_id)
    return updated, durations, tuple(sorted(ready))


def circular_footprint_is_traversable(data, width, height, resolution,
                                      origin_xy, pose_xy, radius_m,
                                      lethal_cost=100):
    """Require every grid cell under a circular footprint to be known/free."""
    width = int(width)
    height = int(height)
    resolution = float(resolution)
    origin_x, origin_y = (float(value) for value in origin_xy[:2])
    center_x, center_y = (float(value) for value in pose_xy[:2])
    radius = float(radius_m)
    values = data
    if (width <= 0 or height <= 0 or len(values) != width * height
            or resolution <= 0.0 or radius < 0.0
            or not all(math.isfinite(value) for value in (
                resolution, origin_x, origin_y, center_x, center_y, radius))):
        raise ValueError("invalid occupancy footprint input")
    center_column = int(math.floor((center_x - origin_x) / resolution))
    center_row = int(math.floor((center_y - origin_y) / resolution))
    extent = int(math.ceil(radius / resolution))
    for row in range(center_row - extent - 1, center_row + extent + 2):
        for column in range(center_column - extent - 1,
                            center_column + extent + 2):
            cell_x = origin_x + (column + 0.5) * resolution
            cell_y = origin_y + (row + 0.5) * resolution
            # Exact circle-versus-axis-aligned-cell intersection.  Testing
            # cell centres against radius + half the diagonal is conservative
            # by an extra 0.5*sqrt(2)*resolution and can reject a route that
            # was correctly inflated for the configured robot radius.
            delta_x = max(abs(cell_x - center_x) - 0.5 * resolution, 0.0)
            delta_y = max(abs(cell_y - center_y) - 0.5 * resolution, 0.0)
            if math.hypot(delta_x, delta_y) > radius + 1.0e-12:
                continue
            if not (0 <= column < width and 0 <= row < height):
                return False
            value = values[row * width + column]
            if value < 0 or value >= int(lethal_cost):
                return False
    return True


def swept_reverse_command_is_traversable(data, width, height, resolution,
                                         origin_xy, pose, radius_m,
                                         linear_speed, angular_speed,
                                         horizon_s=0.8, step_s=0.1):
    """Check the current and predicted footprint for one reverse command."""
    horizon = float(horizon_s)
    step = float(step_s)
    if (not math.isfinite(horizon) or not math.isfinite(step)
            or horizon < 0.0 or step <= 0.0):
        raise ValueError("invalid swept-footprint horizon")
    count = int(math.ceil(horizon / step))
    for index in range(count + 1):
        predicted = integrate_unicycle(
            pose, linear_speed, angular_speed, min(horizon, index * step))
        if not circular_footprint_is_traversable(
                data, width, height, resolution, origin_xy,
                predicted[:2], radius_m):
            return False
    return True


def rectangular_footprint_is_traversable(data, width, height, resolution,
                                         origin_xy, pose,
                                         half_length_m, half_width_m,
                                         lethal_cost=100):
    """Exact SAT overlap test for an oriented robot rectangle and grid cells."""
    width = int(width)
    height = int(height)
    resolution = float(resolution)
    origin_x, origin_y = (float(value) for value in origin_xy[:2])
    center_x, center_y, yaw = (float(value) for value in pose[:3])
    half_length = float(half_length_m)
    half_width = float(half_width_m)
    values = data
    if (width <= 0 or height <= 0 or len(values) != width * height
            or resolution <= 0.0 or half_length <= 0.0 or half_width <= 0.0
            or not all(math.isfinite(value) for value in (
                resolution, origin_x, origin_y, center_x, center_y, yaw,
                half_length, half_width))):
        raise ValueError("invalid rectangular footprint input")
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    half_cell = 0.5 * resolution
    radius = math.hypot(half_length, half_width) + math.sqrt(2.0) * half_cell
    min_column = int(math.floor((center_x - radius - origin_x) / resolution))
    max_column = int(math.floor((center_x + radius - origin_x) / resolution))
    min_row = int(math.floor((center_y - radius - origin_y) / resolution))
    max_row = int(math.floor((center_y + radius - origin_y) / resolution))
    for row in range(min_row, max_row + 1):
        for column in range(min_column, max_column + 1):
            cell_x = origin_x + (column + 0.5) * resolution
            cell_y = origin_y + (row + 0.5) * resolution
            delta_x = cell_x - center_x
            delta_y = cell_y - center_y
            projection_world_x = (half_length * abs(cosine)
                                  + half_width * abs(sine))
            projection_world_y = (half_length * abs(sine)
                                  + half_width * abs(cosine))
            if (abs(delta_x) > projection_world_x + half_cell
                    or abs(delta_y) > projection_world_y + half_cell):
                continue
            along = delta_x * cosine + delta_y * sine
            lateral = -delta_x * sine + delta_y * cosine
            cell_on_body_axis = half_cell * (abs(cosine) + abs(sine))
            if (abs(along) > half_length + cell_on_body_axis
                    or abs(lateral) > half_width + cell_on_body_axis):
                continue
            if not (0 <= column < width and 0 <= row < height):
                return False
            value = values[row * width + column]
            if value < 0 or value >= int(lethal_cost):
                return False
    return True


def swept_rectangular_command_is_traversable(
        data, width, height, resolution, origin_xy, pose,
        half_length_m, half_width_m, linear_speed, angular_speed,
        horizon_s=0.8, step_s=0.1):
    """Check an oriented rectangular footprint over a constant command."""
    horizon = float(horizon_s)
    step = float(step_s)
    if (not math.isfinite(horizon) or not math.isfinite(step)
            or horizon < 0.0 or step <= 0.0):
        raise ValueError("invalid rectangular swept-footprint horizon")
    count = int(math.ceil(horizon / step))
    for index in range(count + 1):
        predicted = integrate_unicycle(
            pose, linear_speed, angular_speed, min(horizon, index * step))
        if not rectangular_footprint_is_traversable(
                data, width, height, resolution, origin_xy, predicted,
                half_length_m, half_width_m):
            return False
    return True


def select_safe_reverse_command(
        data, width, height, resolution, origin_xy, pose, path, transition,
        half_length_m, half_width_m, horizon_s=0.8, step_s=0.1):
    """Deterministically select one swept-safe reverse lattice command.

    This is a bounded local controller, not a new risk model.  Every candidate
    remains reverse-only and is rejected on UNKNOWN or lethal occupancy by the
    same oriented-rectangle test used for the nominal tracker.
    """
    points = tuple((float(point[0]), float(point[1])) for point in path)
    state = tuple(float(value) for value in pose[:3])
    goal = tuple(float(value) for value in transition[:2])
    if len(points) < 2 or len(state) != 3 or len(goal) != 2:
        raise ValueError("invalid reverse command lattice input")
    nearest = min(range(len(points)), key=lambda index: math.hypot(
        points[index][0] - state[0], points[index][1] - state[1]))
    target = points[min(len(points) - 1, nearest + 4)]
    candidates = []
    for linear in (-0.20, -0.15, -0.10, -0.05):
        for angular in (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15):
            if not swept_rectangular_command_is_traversable(
                    data, width, height, resolution, origin_xy, state,
                    half_length_m, half_width_m, linear, angular,
                    horizon_s, step_s):
                continue
            predicted = integrate_unicycle(state, linear, angular, horizon_s)
            travel_heading = math.atan2(
                target[1] - predicted[1], target[0] - predicted[0])
            desired_body_heading = normalize_angle(travel_heading + math.pi)
            heading_error = abs(normalize_angle(
                desired_body_heading - predicted[2]))
            score = (math.hypot(predicted[0] - target[0],
                                predicted[1] - target[1])
                     + 0.25 * math.hypot(predicted[0] - goal[0],
                                         predicted[1] - goal[1])
                     + 0.10 * heading_error
                     - 0.05 * abs(linear))
            candidates.append((score, -abs(linear), abs(angular),
                               angular, linear))
    if not candidates:
        return None
    selected = min(candidates)
    return selected[4], selected[3]


def select_safe_forward_command(
        data, width, height, resolution, origin_xy, pose, path, goal,
        half_length_m, half_width_m, horizon_s=0.8, step_s=0.1):
    """Deterministically select one swept-safe forward lattice command."""
    points = tuple((float(point[0]), float(point[1])) for point in path)
    state = tuple(float(value) for value in pose[:3])
    destination = tuple(float(value) for value in goal[:2])
    if len(points) < 2 or len(state) != 3 or len(destination) != 2:
        raise ValueError("invalid forward command lattice input")
    nearest = min(range(len(points)), key=lambda index: math.hypot(
        points[index][0] - state[0], points[index][1] - state[1]))
    target = points[min(len(points) - 1, nearest + 4)]
    candidates = []
    for linear in (0.50, 0.35, 0.20, 0.10, 0.05):
        for angular in (-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30):
            if not swept_rectangular_command_is_traversable(
                    data, width, height, resolution, origin_xy, state,
                    half_length_m, half_width_m, linear, angular,
                    horizon_s, step_s):
                continue
            predicted = integrate_unicycle(state, linear, angular, horizon_s)
            desired_heading = math.atan2(
                target[1] - predicted[1], target[0] - predicted[0])
            heading_error = abs(normalize_angle(
                desired_heading - predicted[2]))
            score = (math.hypot(predicted[0] - target[0],
                                predicted[1] - target[1])
                     + 0.25 * math.hypot(predicted[0] - destination[0],
                                         predicted[1] - destination[1])
                     + 0.10 * heading_error
                     - 0.05 * linear)
            candidates.append((score, -linear, abs(angular), angular, linear))
    if not candidates:
        return None
    selected = min(candidates)
    return selected[4], selected[3]


def reverse_path_waypoints(path, start, goal, spacing_m=2.0,
                           first_spacing_m=None,
                           endpoint_tolerance_m=0.75,
                           maximum_step_m=0.35):
    """Validate a strict evacuation polyline and sample reverse waypoints.

    Each returned tuple is ``(x, y, body_yaw)``.  ``body_yaw`` is opposite
    the local direction of travel so a nonholonomic robot can track the route
    in reverse without being simultaneously commanded to turn toward home.
    The source path is expected to come from the UNKNOWN-forbidden fused-map
    planner; continuity and both endpoint identities are checked here before
    any command is issued.
    """
    first_spacing = (float(spacing_m) if first_spacing_m is None
                     else float(first_spacing_m))
    if (not math.isfinite(float(spacing_m)) or spacing_m <= 0.0
            or not math.isfinite(first_spacing) or first_spacing <= 0.0
            or first_spacing > spacing_m
            or not math.isfinite(float(endpoint_tolerance_m))
            or endpoint_tolerance_m < 0.0
            or not math.isfinite(float(maximum_step_m))
            or maximum_step_m <= 0.0):
        raise ValueError("invalid reverse path sampling parameters")
    parsed = []
    for point in path or ():
        if len(point) < 2:
            raise ValueError("evacuation path point must contain x and y")
        value = (float(point[0]), float(point[1]))
        if not all(math.isfinite(component) for component in value):
            raise ValueError("evacuation path contains a non-finite point")
        parsed.append(value)
    if len(parsed) < 2:
        raise ValueError("evacuation path must contain at least two points")
    if math.hypot(parsed[0][0] - float(start[0]),
                  parsed[0][1] - float(start[1])) > endpoint_tolerance_m:
        raise ValueError("evacuation path does not start at the robot")
    if math.hypot(parsed[-1][0] - float(goal[0]),
                  parsed[-1][1] - float(goal[1])) > endpoint_tolerance_m:
        raise ValueError("evacuation path does not end at recorded home")
    waypoints = []
    anchor_index = 0
    accumulated = 0.0
    for index in range(1, len(parsed)):
        step = math.hypot(parsed[index][0] - parsed[index - 1][0],
                          parsed[index][1] - parsed[index - 1][1])
        if step <= 0.0 or step > maximum_step_m:
            raise ValueError("evacuation path is not a continuous grid path")
        accumulated += step
        is_last = index == len(parsed) - 1
        required_spacing = first_spacing if not waypoints else spacing_m
        if accumulated < required_spacing and not is_last:
            continue
        delta_x = parsed[index][0] - parsed[anchor_index][0]
        delta_y = parsed[index][1] - parsed[anchor_index][1]
        if math.hypot(delta_x, delta_y) <= 1.0e-9:
            raise ValueError("evacuation waypoint has no travel direction")
        reverse_yaw = normalize_angle(math.atan2(delta_y, delta_x) + math.pi)
        waypoints.append((parsed[index][0], parsed[index][1], reverse_yaw))
        anchor_index = index
        accumulated = 0.0
    if not waypoints:
        raise ValueError("evacuation path produced no waypoint")
    return tuple(waypoints)


def inflate_hard_obstacles(occupancy, radius_cells, hard_threshold=100):
    occupancy = np.asarray(occupancy, dtype=np.int16)
    output = np.array(occupancy, copy=True)
    radius_cells = max(0, int(radius_cells))
    if radius_cells == 0:
        return output
    occupied = np.argwhere(occupancy >= int(hard_threshold))
    offsets = []
    for delta_y in range(-radius_cells, radius_cells + 1):
        for delta_x in range(-radius_cells, radius_cells + 1):
            if math.hypot(delta_x, delta_y) <= radius_cells + 1.0e-12:
                offsets.append((delta_x, delta_y))
    height, width = occupancy.shape
    for row, column in occupied:
        for delta_x, delta_y in offsets:
            candidate_x = column + delta_x
            candidate_y = row + delta_y
            if 0 <= candidate_x < width and 0 <= candidate_y < height:
                output[candidate_y, candidate_x] = 100
    return output


def _neighbors(cell, width, height, diagonal=True):
    moves = ((1, 0), (-1, 0), (0, 1), (0, -1))
    if diagonal:
        moves += ((1, 1), (1, -1), (-1, 1), (-1, -1))
    for delta_x, delta_y in moves:
        candidate = (cell[0] + delta_x, cell[1] + delta_y)
        if 0 <= candidate[0] < width and 0 <= candidate[1] < height:
            yield candidate


def bounded_disk_cells(center, radius_cells, width, height):
    """Return deterministic in-bounds cells in a closed Euclidean disk."""
    radius = max(0, int(radius_cells))
    cells = []
    for delta_y in range(-radius, radius + 1):
        for delta_x in range(-radius, radius + 1):
            if math.hypot(delta_x, delta_y) > radius + 1.0e-12:
                continue
            candidate = (int(center[0]) + delta_x,
                         int(center[1]) + delta_y)
            if 0 <= candidate[0] < int(width) and 0 <= candidate[1] < int(height):
                cells.append(candidate)
    return tuple(cells)


def _reconstruct(parent, endpoint):
    path = [endpoint]
    while path[-1] in parent:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def goal_or_frontier(occupancy, start, requested_goal, resolution,
                     arrival_tolerance_m=0.7, robot_radius_m=0.4,
                     hard_threshold=100, frontier_backoff_m=0.35,
                     blacklisted=None):
    """Select a goal/frontier through known geometrically traversable space.

    Returns ``(mode, endpoint, path)`` where mode is ``GOAL`` or ``FRONTIER``.
    Finite 1..99 costs remain reachable; move_base optimizes the actual route.
    UNKNOWN cells are never traversed.  If no known-free cell is within the
    target tolerance, the selected endpoint is a reachable frontier that makes
    goal-directed progress.  ``None`` means no safe progress is available.
    """
    occupancy = np.asarray(occupancy, dtype=np.int16)
    if occupancy.ndim != 2 or not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("invalid occupancy grid")
    height, width = occupancy.shape
    if (start is None or requested_goal is None
            or not (0 <= start[0] < width and 0 <= start[1] < height)):
        return None
    if not 0 <= occupancy[start[1], start[0]] < hard_threshold:
        return None
    inflated = inflate_hard_obstacles(
        occupancy, int(math.ceil(robot_radius_m / resolution)), hard_threshold)
    # The current pose can lie inside an inflated footprint boundary; it is a
    # valid BFS seed, but all subsequent cells must be observed FREE.
    queue = deque([start])
    parent = {}
    distance_steps = {start: 0}
    reachable = []
    while queue:
        current = queue.popleft()
        reachable.append(current)
        for neighbor in _neighbors(current, width, height, diagonal=True):
            if neighbor in distance_steps:
                continue
            column, row = neighbor
            if not 0 <= inflated[row, column] < hard_threshold:
                continue
            if neighbor[0] != current[0] and neighbor[1] != current[1]:
                adjacent = ((neighbor[0], current[1]),
                            (current[0], neighbor[1]))
                if any(not 0 <= inflated[y_value, x_value] < hard_threshold
                       for x_value, y_value in adjacent):
                    continue
            distance_steps[neighbor] = distance_steps[current] + 1
            parent[neighbor] = current
            queue.append(neighbor)
    if not reachable:
        return None

    tolerance_cells = max(1.0, arrival_tolerance_m / resolution)
    goal_candidates = [
        cell for cell in reachable
        if math.hypot(cell[0] - requested_goal[0],
                      cell[1] - requested_goal[1]) <= tolerance_cells
    ]
    if goal_candidates:
        endpoint = min(
            goal_candidates,
            key=lambda cell: (math.hypot(cell[0] - requested_goal[0],
                                         cell[1] - requested_goal[1]),
                              distance_steps[cell], cell[1], cell[0]))
        return "GOAL", endpoint, _reconstruct(parent, endpoint)

    blacklist = set(blacklisted or ())
    frontier_candidates = []
    cardinal = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for cell in reachable:
        if cell in blacklist:
            continue
        is_frontier = False
        for delta_x, delta_y in cardinal:
            neighbor = (cell[0] + delta_x, cell[1] + delta_y)
            if (0 <= neighbor[0] < width and 0 <= neighbor[1] < height
                    and occupancy[neighbor[1], neighbor[0]] < 0):
                is_frontier = True
                break
        if is_frontier:
            frontier_candidates.append(cell)
    if not frontier_candidates:
        return None

    # Distance to the requested target dominates; path length breaks ties and
    # prevents choosing a remote frontier that gives negligible progress.
    frontier = min(
        frontier_candidates,
        key=lambda cell: (math.hypot(cell[0] - requested_goal[0],
                                     cell[1] - requested_goal[1])
                          + 0.12 * distance_steps[cell],
                          distance_steps[cell], cell[1], cell[0]))
    path = _reconstruct(parent, frontier)
    backoff_cells = int(math.ceil(frontier_backoff_m / resolution))
    endpoint_index = max(0, len(path) - 1 - backoff_cells)
    endpoint = path[endpoint_index]
    if endpoint == start and len(path) > 1:
        endpoint = path[min(len(path) - 1, 1)]
    return "FRONTIER", endpoint, path[:endpoint_index + 1]
