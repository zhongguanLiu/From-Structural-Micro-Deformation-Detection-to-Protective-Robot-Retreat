"""Deterministic temporal logic independent of ROS transport.

The module deliberately separates three notions that must not be conflated:
deformation magnitude change on a fixed GDU, motion of that GDU's current
geometry, and activation/deactivation of spatial support.
"""

import math


MAGNITUDE_UNKNOWN = 0
MAGNITUDE_INCREASING = 1
MAGNITUDE_STABLE = 2
MAGNITUDE_DECREASING = 3

LIFECYCLE_INACTIVE = 0
LIFECYCLE_NEWLY_DETECTED = 1
LIFECYCLE_PERSISTENT = 2
LIFECYCLE_CLEAR_CANDIDATE = 3
LIFECYCLE_INCONCLUSIVE = 4
LIFECYCLE_PENDING_CONFIRMATION = 5


def fresh_observable_measurement(validity, observable_rank):
    """Whether this snapshot contains a new usable geometric measurement.

    A physical sample with rank zero is still a known frozen GDU carrier, but
    its posterior contains no newly observed deformation coordinate.  It is
    therefore temporary missing information, not OTHER and not counter-
    evidence for a pending detection.
    """
    return (validity in (0, 1)
            and isinstance(observable_rank, int)
            and observable_rank > 0)


def activation_decision(strict_mode, admission, validity, probability,
                        magnitude, minimum_effect, observable_rank,
                        all_task_directions_detectable,
                        probability_threshold=0.8,
                        clear_probability_threshold=0.05):
    """Return the current active/clear decision without semantic priors.

    A deformation source may become active when at least one deformation
    coordinate is observable.  Clearing is deliberately stronger: every
    registered task direction must be detectable, otherwise a small posterior
    can mean ``not observable`` instead of ``not changing``.  The strict mode
    keeps the detector admission decision but still checks these two numerical
    observability invariants so an inconsistent message cannot clear a cause.
    """
    measurement_valid = validity in (0, 1)
    observable = fresh_observable_measurement(validity, observable_rank)
    all_directions = bool(all_task_directions_detectable)
    if strict_mode:
        return (admission == 0 and validity == 0 and observable,
                admission == 1 and validity == 0 and observable
                and all_directions)
    active = (measurement_valid and observable
              and math.isfinite(probability)
              and probability >= probability_threshold
              and math.isfinite(magnitude)
              and magnitude >= minimum_effect)
    cleared = (measurement_valid and observable and all_directions
               and math.isfinite(probability)
               and probability <= clear_probability_threshold
               and math.isfinite(magnitude)
               and magnitude < minimum_effect)
    return active, cleared


def _finite_vector(vector):
    return len(vector) == 3 and all(math.isfinite(value) for value in vector)


def _finite_matrix3_flat(values):
    if len(values) != 9 or not all(math.isfinite(value) for value in values):
        return None
    return tuple(float(value) for value in values)


def _symmetric_psd3_flat(matrix, tolerance=1.0e-10):
    """Check a finite symmetric 3x3 covariance without forming an inverse."""
    if max(abs(matrix[1] - matrix[3]),
           abs(matrix[2] - matrix[6]),
           abs(matrix[5] - matrix[7])) > tolerance:
        return False
    a, b, c = matrix[0], matrix[1], matrix[2]
    d, e, f = matrix[4], matrix[5], matrix[8]
    scale = max(1.0e-30, abs(a), abs(b), abs(c), abs(d), abs(e), abs(f))
    tol = tolerance * scale
    principal_2 = (a * d - b * b, a * f - c * c, d * f - e * e)
    determinant = (a * (d * f - e * e)
                   - b * (b * f - c * e)
                   + c * (b * e - c * d))
    return (a >= -tol and d >= -tol and f >= -tol
            and all(value >= -tolerance * scale * scale
                    for value in principal_2)
            and determinant >= -tolerance * scale * scale * scale)


def observable_sample(displacement, velocity, displacement_covariance,
                      velocity_covariance, projector, tolerance=1.0e-7):
    """Validate and project one detector sample into its observable subspace.

    The detector represents unobservable directions by a projector/nullspace,
    not by artificial small noise.  Downstream magnitude and kinematic
    prediction must therefore use ``P u``, ``P v`` and ``P Sigma P`` only.
    A posterior mean may retain a prior component in the current information
    nullspace.  That component is not evidence and is deliberately discarded;
    only an invalid covariance/projector makes the complete input frame fail.
    ``None`` means the complete input frame violates that numerical contract.
    """
    if not (_finite_vector(displacement) and _finite_vector(velocity)):
        return None
    covariance_d = _finite_matrix3_flat(displacement_covariance)
    covariance_v = _finite_matrix3_flat(velocity_covariance)
    observable = _finite_matrix3_flat(projector)
    if covariance_d is None or covariance_v is None or observable is None:
        return None
    if (not _symmetric_psd3_flat(covariance_d)
            or not _symmetric_psd3_flat(covariance_v)):
        return None
    if max(abs(observable[1] - observable[3]),
           abs(observable[2] - observable[6]),
           abs(observable[5] - observable[7])) > tolerance:
        return None
    p = observable
    projector_squared = (
        p[0] * p[0] + p[1] * p[3] + p[2] * p[6],
        p[0] * p[1] + p[1] * p[4] + p[2] * p[7],
        p[0] * p[2] + p[1] * p[5] + p[2] * p[8],
        p[3] * p[0] + p[4] * p[3] + p[5] * p[6],
        p[3] * p[1] + p[4] * p[4] + p[5] * p[7],
        p[3] * p[2] + p[4] * p[5] + p[5] * p[8],
        p[6] * p[0] + p[7] * p[3] + p[8] * p[6],
        p[6] * p[1] + p[7] * p[4] + p[8] * p[7],
        p[6] * p[2] + p[7] * p[5] + p[8] * p[8],
    )
    if max(abs(projector_squared[index] - p[index])
           for index in range(9)) > tolerance:
        return None
    rank = int(round(p[0] + p[4] + p[8]))
    # Rank zero is a valid local nullspace carrier inside an otherwise
    # observable GDU.  It contributes neither motion nor uncertainty after
    # projection.  Whole-GDU rank-zero carriers are removed by the node before
    # aggregation; rejecting a mixed GDU here would make one unobservable
    # support sample invalidate every observable support in the snapshot.
    if rank < 0 or rank > 3:
        return None
    projected_d = (
        p[0] * displacement[0] + p[1] * displacement[1] + p[2] * displacement[2],
        p[3] * displacement[0] + p[4] * displacement[1] + p[5] * displacement[2],
        p[6] * displacement[0] + p[7] * displacement[1] + p[8] * displacement[2],
    )
    projected_v = (
        p[0] * velocity[0] + p[1] * velocity[1] + p[2] * velocity[2],
        p[3] * velocity[0] + p[4] * velocity[1] + p[5] * velocity[2],
        p[6] * velocity[0] + p[7] * velocity[1] + p[8] * velocity[2],
    )

    def projected_covariance_trace(covariance):
        ps = (
            p[0] * covariance[0] + p[1] * covariance[3] + p[2] * covariance[6],
            p[0] * covariance[1] + p[1] * covariance[4] + p[2] * covariance[7],
            p[0] * covariance[2] + p[1] * covariance[5] + p[2] * covariance[8],
            p[3] * covariance[0] + p[4] * covariance[3] + p[5] * covariance[6],
            p[3] * covariance[1] + p[4] * covariance[4] + p[5] * covariance[7],
            p[3] * covariance[2] + p[4] * covariance[5] + p[5] * covariance[8],
            p[6] * covariance[0] + p[7] * covariance[3] + p[8] * covariance[6],
            p[6] * covariance[1] + p[7] * covariance[4] + p[8] * covariance[7],
            p[6] * covariance[2] + p[7] * covariance[5] + p[8] * covariance[8],
        )
        diagonal_0 = ps[0] * p[0] + ps[1] * p[3] + ps[2] * p[6]
        diagonal_1 = ps[3] * p[1] + ps[4] * p[4] + ps[5] * p[7]
        diagonal_2 = ps[6] * p[2] + ps[7] * p[5] + ps[8] * p[8]
        return max(0.0, diagonal_0 + diagonal_1 + diagonal_2)

    trace_d = projected_covariance_trace(covariance_d)
    trace_v = projected_covariance_trace(covariance_v)
    return projected_d, projected_v, trace_d, trace_v, rank


def magnitude_statistics(displacements, covariance_traces):
    """Return RMS magnitude, mean vector, and a conservative sigma bound.

    Surfel samples from one GDU are spatially correlated.  Consequently the
    bound is the largest per-sample trace standard deviation; it is never
    divided by the sample count as if the samples were independent.
    """
    if not displacements or len(displacements) != len(covariance_traces):
        return 0.0, (0.0, 0.0, 0.0), float("inf")
    squared_norm_sum = 0.0
    coordinate_sum = [0.0, 0.0, 0.0]
    maximum_trace = 0.0
    for vector, covariance_trace in zip(displacements, covariance_traces):
        if (not _finite_vector(vector)
                or not math.isfinite(covariance_trace)
                or covariance_trace < 0.0):
            return 0.0, (0.0, 0.0, 0.0), float("inf")
        squared_norm_sum += sum(value * value for value in vector)
        coordinate_sum[0] += vector[0]
        coordinate_sum[1] += vector[1]
        coordinate_sum[2] += vector[2]
        maximum_trace = max(maximum_trace, covariance_trace)
    rms = math.sqrt(max(0.0, squared_norm_sum / len(displacements)))
    mean = tuple(value / len(displacements) for value in coordinate_sum)
    sigma = math.sqrt(maximum_trace)
    return rms, mean, sigma


def direction_change(previous, current, epsilon=1.0e-12):
    if not _finite_vector(previous) or not _finite_vector(current):
        return float("nan")
    previous_norm = math.sqrt(sum(value * value for value in previous))
    current_norm = math.sqrt(sum(value * value for value in current))
    if previous_norm <= epsilon or current_norm <= epsilon:
        return float("nan")
    cosine = sum(previous[i] * current[i] for i in range(3)) / (
        previous_norm * current_norm)
    return math.acos(max(-1.0, min(1.0, cosine)))


def classify_magnitude(previous_magnitude, previous_sigma,
                       current_magnitude, current_sigma,
                       minimum_change_m, sigma_factor):
    values = (previous_magnitude, previous_sigma,
              current_magnitude, current_sigma)
    if not all(math.isfinite(value) for value in values):
        return MAGNITUDE_UNKNOWN
    # No cross-epoch covariance is exported by V1.  The triangle bound is
    # valid for every unknown temporal correlation, unlike root-sum-square
    # which silently assumes independence.
    uncertainty = sigma_factor * (previous_sigma + current_sigma)
    threshold = max(minimum_change_m, uncertainty)
    difference = current_magnitude - previous_magnitude
    if difference > threshold:
        return MAGNITUDE_INCREASING
    if difference < -threshold:
        return MAGNITUDE_DECREASING
    return MAGNITUDE_STABLE


def predict_support(reference_support, displacements, velocities, horizon_s):
    if not math.isfinite(horizon_s) or horizon_s < 0.0:
        return [], [], False
    if not (len(reference_support) == len(displacements) == len(velocities)):
        return [], [], False
    positions = []
    predicted_displacements = []
    for reference, displacement, velocity in zip(
            reference_support, displacements, velocities):
        if not (_finite_vector(reference) and _finite_vector(displacement)
                and _finite_vector(velocity)):
            return [], [], False
        predicted = tuple(displacement[axis] + horizon_s * velocity[axis]
                          for axis in range(3))
        positions.append(tuple(reference[axis] + predicted[axis]
                               for axis in range(3)))
        predicted_displacements.append(predicted)
    return positions, predicted_displacements, True


def predict_support_horizons(reference_support, displacements, velocities,
                             horizons_s):
    """Predict several fixed horizons with one validation pass.

    The returned list has one ``(positions, displacements)`` pair per input
    horizon, in the same order.  Every value is computed by the same constant-
    velocity expression as :func:`predict_support`; only repeated validation
    and traversal of the 8k-scale support arrays are removed.
    """
    horizons = tuple(float(value) for value in horizons_s)
    if (not horizons
            or any(not math.isfinite(value) or value < 0.0
                   for value in horizons)
            or not (len(reference_support) == len(displacements)
                    == len(velocities))):
        return [], False
    positions_by_horizon = [[] for _ in horizons]
    displacements_by_horizon = [[] for _ in horizons]
    for reference, displacement, velocity in zip(
            reference_support, displacements, velocities):
        if not (_finite_vector(reference) and _finite_vector(displacement)
                and _finite_vector(velocity)):
            return [], False
        for index, horizon_s in enumerate(horizons):
            predicted = (
                displacement[0] + horizon_s * velocity[0],
                displacement[1] + horizon_s * velocity[1],
                displacement[2] + horizon_s * velocity[2],
            )
            positions_by_horizon[index].append((
                reference[0] + predicted[0],
                reference[1] + predicted[1],
                reference[2] + predicted[2],
            ))
            displacements_by_horizon[index].append(predicted)
    return list(zip(positions_by_horizon, displacements_by_horizon)), True


def kinematic_forecast_allowed(evidence_observable_rank,
                               sample_observable_ranks,
                               velocity_mde_defined,
                               velocity_mde_mps,
                               displacement_sigma,
                               velocity_sigma):
    """Return whether a submitted velocity posterior may be extrapolated.

    ``continuous_valid_observation_duration_s`` is intentionally absent.  It
    is the duration of the *velocity-threshold event*, not the duration for
    which the velocity state has been observed.  A small, observable velocity
    may stay below that event threshold and is still a valid input to a
    kinematic forecast; its uncertainty and MDE communicate its weak power.
    """
    # Evidence rank is the rank of the complete GDU coefficient posterior and
    # may exceed three.  Per-support projectors alone are three-dimensional;
    # rank-zero supports are valid local nullspaces as long as at least one
    # support direction in the GDU is observable.
    if (not isinstance(evidence_observable_rank, int)
            or evidence_observable_rank < 1
            or not sample_observable_ranks
            or any(not isinstance(rank, int) or rank < 0 or rank > 3
                   for rank in sample_observable_ranks)
            or not any(rank > 0 for rank in sample_observable_ranks)
            or not bool(velocity_mde_defined)
            or not math.isfinite(velocity_mde_mps)
            or velocity_mde_mps < 0.0):
        return False
    return (math.isfinite(displacement_sigma)
            and displacement_sigma >= 0.0
            and math.isfinite(velocity_sigma)
            and velocity_sigma >= 0.0)


class EvolutionTracker:
    """Track fixed-reference GDU evolution and deterministic birth/clear hysteresis."""

    def __init__(self, clear_confirmations=3, retained_birth_confirmations=3,
                 minimum_change_m=0.0005, sigma_factor=1.0,
                 pending_confirmation_timeout_s=15.0):
        if clear_confirmations < 1:
            raise ValueError("clear_confirmations must be positive")
        if retained_birth_confirmations < 1:
            raise ValueError("retained_birth_confirmations must be positive")
        if (not math.isfinite(pending_confirmation_timeout_s)
                or pending_confirmation_timeout_s <= 0.0):
            raise ValueError(
                "pending_confirmation_timeout_s must be finite and positive")
        self.clear_confirmations = int(clear_confirmations)
        self.retained_birth_confirmations = int(retained_birth_confirmations)
        self.pending_confirmation_timeout_s = float(
            pending_confirmation_timeout_s)
        self.minimum_change_m = float(minimum_change_m)
        self.sigma_factor = float(sigma_factor)
        self.reference_epoch = None
        self._states = {}

    def reset(self, reference_epoch=None):
        self.reference_epoch = reference_epoch
        self._states = {}

    def retained_ids(self):
        """Return retained GDU identities without exposing mutable state."""
        return sorted(key for key, state in self._states.items()
                      if state["retained"])

    def tracked_ids(self):
        """Return every identity whose birth or retained state is pending."""
        return sorted(self._states)

    def mark_missing(self, gdu_id, reference_epoch, stamp_s):
        """Advance a tracked GDU through one inconclusive observation.

        Absence from a complete detector tile is not clearing evidence.  Keep
        a retained GDU and pause a pending birth until its bounded timeout;
        only a fresh non-active observation or timeout rejects that birth.
        The caller owns the last retained geometry and must not extrapolate it.
        """
        if self.reference_epoch != reference_epoch:
            self.reset(reference_epoch)
        previous = self._states.get(gdu_id)
        if not previous:
            return None
        return self.update(
            gdu_id, reference_epoch, stamp_s,
            False, False, False,
            previous["magnitude"], previous["sigma"],
            previous["mean_displacement"])

    def update(self, gdu_id, reference_epoch, stamp_s, active_now,
               cleared_now, measurement_valid, magnitude, sigma,
               mean_displacement):
        """Commit one temporal step.

        ``measurement_valid`` is intentionally narrower than frame validity:
        it means a fresh usable deformation measurement is present.  Callers
        must pass false for CTMC-only, missing, or rank-zero observations.
        """
        if self.reference_epoch != reference_epoch:
            self.reset(reference_epoch)
        previous = self._states.get(gdu_id)
        previously_retained = bool(previous and previous["retained"])
        clear_streak = int(previous["clear_streak"]) if previous else 0
        retained_birth_streak = int(
            previous.get("retained_birth_streak", 0)) if previous else 0
        pending_started = (previous.get(
            "pending_confirmation_started_stamp_s") if previous else None)
        pending_timed_out = False

        # A pending birth is deliberately bounded in wall/measurement time.
        # CTMC-only or temporarily unobservable frames pause the number of
        # independent confirmations, but cannot preserve it forever.
        if (not previously_retained and retained_birth_streak > 0
                and pending_started is not None
                and float(stamp_s) - float(pending_started)
                > self.pending_confirmation_timeout_s):
            retained_birth_streak = 0
            pending_started = None
            pending_timed_out = True

        if active_now and measurement_valid:
            clear_streak = 0
            if previously_retained:
                retained = True
                retained_birth_streak = self.retained_birth_confirmations
                pending_started = None
                lifecycle = LIFECYCLE_PERSISTENT
            else:
                if retained_birth_streak == 0:
                    pending_started = float(stamp_s)
                retained_birth_streak += 1
                retained = (retained_birth_streak
                            >= self.retained_birth_confirmations)
                if retained:
                    retained_birth_streak = self.retained_birth_confirmations
                    pending_started = None
                    lifecycle = LIFECYCLE_NEWLY_DETECTED
                else:
                    lifecycle = LIFECYCLE_PENDING_CONFIRMATION
        elif cleared_now and measurement_valid:
            pending_started = None
            if previously_retained:
                clear_streak += 1
                retained = clear_streak < self.clear_confirmations
                lifecycle = (LIFECYCLE_CLEAR_CANDIDATE if retained
                             else LIFECYCLE_INACTIVE)
                if not retained:
                    retained_birth_streak = 0
            else:
                retained = False
                clear_streak = 0
                retained_birth_streak = 0
                lifecycle = LIFECYCLE_INACTIVE
        elif not measurement_valid:
            retained = previously_retained
            lifecycle = (LIFECYCLE_INCONCLUSIVE
                         if retained or retained_birth_streak > 0
                         else LIFECYCLE_INACTIVE)
        else:
            # A fresh, observable non-active result is counter-evidence for a
            # pending birth.  This is intentionally different from a frame
            # that carries only CTMC prediction or has observable_rank == 0.
            retained = previously_retained
            pending_started = None
            lifecycle = (LIFECYCLE_PERSISTENT if retained
                         else LIFECYCLE_INACTIVE)
            if not retained:
                retained_birth_streak = 0

        trend = MAGNITUDE_UNKNOWN
        previous_magnitude = float("nan")
        angle = float("nan")
        observation_interval = float("nan")
        displacement_rate = float("nan")
        if previous and measurement_valid and previous["measurement_valid"]:
            previous_magnitude = previous["magnitude"]
            trend = classify_magnitude(
                previous["magnitude"], previous["sigma"], magnitude, sigma,
                self.minimum_change_m, self.sigma_factor)
            angle = direction_change(previous["mean_displacement"],
                                     mean_displacement)
            observation_interval = float(stamp_s) - previous["stamp_s"]
            if observation_interval > 0.0 and math.isfinite(observation_interval):
                displacement_rate = (
                    magnitude - previous_magnitude) / observation_interval

        self._states[gdu_id] = {
            "retained": retained,
            "clear_streak": clear_streak,
            "retained_birth_streak": retained_birth_streak,
            "measurement_valid": measurement_valid,
            "pending_confirmation_started_stamp_s": pending_started,
            "magnitude": magnitude,
            "sigma": sigma,
            "mean_displacement": tuple(mean_displacement),
            "stamp_s": float(stamp_s),
        }
        return {
            "retained": retained,
            "clear_streak": clear_streak,
            "retained_birth_streak": retained_birth_streak,
            "pending_confirmation_started_stamp_s": pending_started,
            "pending_confirmation_timed_out": pending_timed_out,
            "lifecycle": lifecycle,
            "trend": trend,
            "previous_magnitude": previous_magnitude,
            "magnitude_change": (magnitude - previous_magnitude
                                 if math.isfinite(previous_magnitude)
                                 else float("nan")),
            "direction_change": angle,
            "observation_interval": observation_interval,
            "displacement_rate": displacement_rate,
        }
