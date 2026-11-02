#!/usr/bin/env python3
"""Build a non-semantic temporal evolution product from V1 detection tiles."""

import copy
import math

import rospy
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from deform_monitor_v3.msg import V1DetectionTile
from deform_evolution_map.model import (
    activation_decision,
    EvolutionTracker,
    fresh_observable_measurement,
    LIFECYCLE_CLEAR_CANDIDATE,
    LIFECYCLE_INCONCLUSIVE,
    LIFECYCLE_NEWLY_DETECTED,
    LIFECYCLE_PENDING_CONFIRMATION,
    LIFECYCLE_PERSISTENT,
    MAGNITUDE_DECREASING,
    MAGNITUDE_INCREASING,
    kinematic_forecast_allowed,
    magnitude_statistics,
    observable_sample,
    predict_support_horizons,
)
from deform_evolution_map.msg import (
    DeformationEvolutionFrame,
    GduEvolution,
    GduForecast,
)


def _id_string(stable_id):
    return "{:016x}{:016x}".format(stable_id.high, stable_id.low)


def _point_tuple(point):
    return (float(point.x), float(point.y), float(point.z))


def _vector_tuple(vector):
    return (float(vector.x), float(vector.y), float(vector.z))


def _point(values):
    return Point(x=values[0], y=values[1], z=values[2])


def _vector(values):
    return Vector3(x=values[0], y=values[1], z=values[2])


def _mean(vectors):
    if not vectors:
        return (0.0, 0.0, 0.0)
    return tuple(sum(vector[axis] for vector in vectors) / len(vectors)
                 for axis in range(3))


def _finite(value):
    return math.isfinite(value)


class DeformationEvolutionNode:
    STRICT = "STRICT_ADMISSION"
    RESEARCH = "RESEARCH_POSTERIOR"

    def __init__(self):
        prefix = "~deform_evolution/"
        self.input_topic = rospy.get_param(prefix + "input_topic",
                                           "/deform/v1/detection_tiles")
        output_topic = rospy.get_param(prefix + "output_topic",
                                       "/deform/evolution/frame")
        marker_topic = rospy.get_param(prefix + "marker_topic",
                                       "/deform/evolution/markers")
        self.activation_mode = rospy.get_param(
            prefix + "activation_mode", self.RESEARCH).upper()
        if self.activation_mode not in (self.STRICT, self.RESEARCH):
            raise ValueError("activation_mode must be STRICT_ADMISSION or RESEARCH_POSTERIOR")
        self.probability_threshold = float(rospy.get_param(
            prefix + "research_probability_threshold", 0.80))
        self.clear_probability_threshold = float(rospy.get_param(
            prefix + "research_clear_probability_threshold", 0.05))
        self.minimum_displacement_m = float(rospy.get_param(
            prefix + "research_minimum_displacement_m", 0.0005))
        self.research_require_mde = bool(rospy.get_param(
            prefix + "research_require_mde", False))
        self.horizons = [float(value) for value in rospy.get_param(
            prefix + "prediction_horizons_s", [0.5, 1.0, 2.0])]
        if (not self.horizons or any(not _finite(value) or value <= 0.0
                                     for value in self.horizons)
                or self.horizons != sorted(set(self.horizons))):
            raise ValueError("prediction_horizons_s must be finite, positive, unique and sorted")
        self.display_scale = float(rospy.get_param(
            prefix + "displacement_display_scale", 50.0))
        self.maximum_arrow_m = float(rospy.get_param(
            prefix + "maximum_display_arrow_m", 1.0))
        self.tracker = EvolutionTracker(
            clear_confirmations=int(rospy.get_param(
                prefix + "clear_confirmations", 3)),
            retained_birth_confirmations=int(rospy.get_param(
                prefix + "retained_birth_confirmations", 3)),
            minimum_change_m=float(rospy.get_param(
                prefix + "magnitude_change_minimum_m", 0.0005)),
            sigma_factor=float(rospy.get_param(
                prefix + "magnitude_change_sigma_factor", 1.0)),
            pending_confirmation_timeout_s=float(rospy.get_param(
                prefix + "pending_confirmation_timeout_s", 15.0)),
        )
        self.last_reference_epoch = None
        self.last_accepted_frame_index = None
        self.last_stamp = None
        self.last_snapshot_id = None
        self.last_retained_gdus = {}
        self.publisher = rospy.Publisher(
            output_topic, DeformationEvolutionFrame, queue_size=2, latch=True)
        self.marker_publisher = rospy.Publisher(
            marker_topic, MarkerArray, queue_size=1, latch=True)
        self.subscriber = rospy.Subscriber(
            self.input_topic, V1DetectionTile,
            self._callback, queue_size=2)
        rospy.loginfo(
            "[deform_evolution] activation=%s retained_birth=%d pending_timeout=%.3fs input=%s",
            self.activation_mode, self.tracker.retained_birth_confirmations,
            self.tracker.pending_confirmation_timeout_s, self.input_topic)

    def _reject_frame(self, frame, detail):
        frame.validity = 2  # V1 Validity::UNKNOWN
        frame.complete_snapshot = False
        frame.input_contract_valid = False
        frame.status_detail = detail
        self.publisher.publish(frame)
        rospy.logerr_throttle(
            2.0, "[deform_evolution] rejected complete LOD2 frame: %s",
            detail)

    def _activation(self, admission, validity, probability, magnitude,
                    mde_defined, mde, observable_rank,
                    all_task_directions_detectable):
        minimum_effect = self.minimum_displacement_m
        if (self.research_require_mde and mde_defined
                and _finite(mde) and mde > 0.0):
            minimum_effect = max(minimum_effect, mde)
        return activation_decision(
            self.activation_mode == self.STRICT,
            admission, validity, probability, magnitude, minimum_effect,
            observable_rank, all_task_directions_detectable,
            self.probability_threshold, self.clear_probability_threshold)

    def _callback(self, tile):
        if tile.lod != 2:
            return
        samples_by_gdu = {}
        support_radius_by_gdu = {}
        evidence_by_gdu = {}
        for sample in tile.physical_deformation:
            samples_by_gdu.setdefault(_id_string(sample.gdu_id), []).append(sample)
        for support in tile.reference_support:
            key = _id_string(support.gdu_id)
            support_radius_by_gdu[key] = max(
                support_radius_by_gdu.get(key, 0.0),
                float(support.support_radius_m))
        for evidence in tile.detection_evidence:
            key = _id_string(evidence.gdu_id)
            evidence_by_gdu.setdefault(key, evidence)

        # Upstream LOD2 emits physical_deformation only for frozen,
        # monitorable GDUs.  A carrier whose current evidence rank is zero is
        # therefore temporarily unobservable, not an OTHER GDU.  Keep it so
        # the temporal tracker can pause (rather than erase) a pending birth.

        frame = DeformationEvolutionFrame()
        frame.header = tile.common.header
        frame.source_snapshot_id = _id_string(tile.source_snapshot_id)
        frame.reference_epoch = tile.common.reference_epoch
        frame.source_accepted_frame_index = tile.common.accepted_frame_index
        frame.valid_until = tile.common.valid_until
        frame.prediction_valid_until = (
            tile.common.header.stamp + rospy.Duration(max(self.horizons)))
        frame.activation_policy = self.activation_mode
        frame.model_revision = "fixed-gdu-observable-vector-evolution-v3"
        frame.validity = tile.common.validity
        frame.complete_snapshot = True
        frame.input_contract_valid = True
        frame.status_detail = "VALID_COMPLETE_LOD2_SNAPSHOT"
        stamp_s = tile.common.header.stamp.to_sec()

        common_snapshot_id = _id_string(tile.common.snapshot_id)
        accepted_index = int(tile.common.accepted_frame_index)
        sequence_error = ""
        if frame.source_snapshot_id != common_snapshot_id:
            sequence_error = "SNAPSHOT_ID_MISMATCH"
        elif (not _finite(stamp_s) or stamp_s <= 0.0
              or not tile.common.header.frame_id):
            sequence_error = "INVALID_HEADER_OR_STAMP"
        elif self.last_accepted_frame_index is not None:
            if (accepted_index == self.last_accepted_frame_index
                    and frame.source_snapshot_id == self.last_snapshot_id):
                return  # Idempotent retransmission of the committed snapshot.
            if accepted_index <= self.last_accepted_frame_index:
                sequence_error = "NON_MONOTONIC_ACCEPTED_FRAME_INDEX"
            elif self.last_stamp is not None and stamp_s <= self.last_stamp:
                sequence_error = "NON_MONOTONIC_MEASUREMENT_STAMP"
        if sequence_error:
            self._reject_frame(frame, sequence_error)
            return

        reference_positions_by_gdu = {}
        for support in tile.reference_support:
            key = _id_string(support.gdu_id)
            reference_positions_by_gdu.setdefault(key, []).append(
                _point_tuple(support.position))
        for key, samples in samples_by_gdu.items():
            sample_positions = sorted(_point_tuple(sample.position)
                                      for sample in samples)
            reference_positions = sorted(reference_positions_by_gdu.get(key, []))
            if (key not in evidence_by_gdu
                    or sample_positions != reference_positions
                    or not _finite(support_radius_by_gdu.get(key, float("nan")))
                    or support_radius_by_gdu.get(key, -1.0) < 0.0):
                self._reject_frame(frame, "INCOMPLETE_GDU_SNAPSHOT:" + key)
                return

        prepared_by_gdu = {}
        contract_error = ""
        temporarily_unobservable = 0
        for key in sorted(samples_by_gdu):
            prepared = []
            for sample in samples_by_gdu[key]:
                result = observable_sample(
                    _vector_tuple(sample.displacement_mean),
                    _vector_tuple(sample.velocity_mean),
                    sample.displacement_covariance_row_major,
                    sample.velocity_covariance_row_major,
                    sample.observable_projector_row_major)
                if result is None or not all(_finite(value) for value in
                                             _point_tuple(sample.position)):
                    contract_error = "INVALID_OBSERVABLE_SAMPLE:" + key
                    break
                prepared.append(result)
            if contract_error:
                break
            prepared_by_gdu[key] = prepared
        if contract_error:
            # Atomic rollback: do not mutate the tracker and do not publish a
            # marker DELETEALL.  The avoidance consumer preserves its last
            # complete source snapshot when it sees this status frame.
            self._reject_frame(frame, contract_error)
            return

        # Build the complete next temporal state off to the side.  A later
        # GDU failure must not leave earlier GDU histories committed.
        candidate_tracker = copy.deepcopy(self.tracker)
        # Retained messages are immutable snapshots.  A shallow dictionary
        # view is sufficient for rollback; only a missing GDU that must be
        # modified below is deep-copied.  Cloning all support/forecast points
        # here would otherwise duplicate the complete 8k-scale product before
        # any contract check can commit it.
        candidate_last_retained = self.last_retained_gdus
        if candidate_tracker.reference_epoch != frame.reference_epoch:
            candidate_last_retained = {}
        for key in sorted(samples_by_gdu):
            samples = samples_by_gdu[key]
            prepared = prepared_by_gdu[key]
            evidence = evidence_by_gdu.get(key)
            references = [_point_tuple(sample.position) for sample in samples]
            displacements = [result[0] for result in prepared]
            velocities = [result[1] for result in prepared]
            covariance_traces = [result[2] for result in prepared]
            velocity_covariance_traces = [result[3] for result in prepared]
            observable_ranks = [result[4] for result in prepared]
            magnitude, mean_displacement, sigma = magnitude_statistics(
                displacements, covariance_traces)
            mean_velocity = _mean(velocities)
            velocity_sigma = math.sqrt(max(velocity_covariance_traces))
            admission = int(samples[0].detection_admission)
            validity = max(int(sample.validity) for sample in samples)
            probability = 0.0
            probability_mean_identified = False
            mde_defined = bool(samples[0].minimum_detectable_effect_defined)
            mde = float(samples[0].minimum_detectable_effect_m)
            if evidence is not None:
                admission = int(evidence.detection_admission)
                validity = max(validity, int(evidence.validity))
                probability_mean_identified = bool(
                    evidence.task_scale_probability_mean_identified)
                probability = float(
                    evidence.task_scale_probability_mean
                    if probability_mean_identified
                    else evidence.task_scale_probability_upper)
                mde_defined = bool(evidence.minimum_detectable_effect_defined)
                mde = float(evidence.minimum_detectable_effect_m)
            elif samples[0].active_probability_mean_identified:
                probability = float(samples[0].active_probability_mean)
                probability_mean_identified = True
            else:
                probability = float(samples[0].active_probability_upper)

            active, cleared = self._activation(
                admission, validity, probability, magnitude, mde_defined, mde,
                int(evidence.observable_rank),
                bool(evidence.all_task_directions_detectable))
            measurement_valid = fresh_observable_measurement(
                validity, int(evidence.observable_rank))
            state = candidate_tracker.update(
                key, frame.reference_epoch, stamp_s, active, cleared,
                measurement_valid, magnitude, sigma, mean_displacement)

            output = GduEvolution()
            output.gdu_id = key
            output.detection_admission = admission
            output.validity = validity
            output.activation_mode = (GduEvolution.ACTIVATION_STRICT_ADMISSION
                                      if self.activation_mode == self.STRICT
                                      else GduEvolution.ACTIVATION_RESEARCH_POSTERIOR)
            output.active_now = active
            output.retained_for_avoidance = state["retained"]
            output.retained_birth_streak = state["retained_birth_streak"]
            output.retained_birth_confirmations = (
                candidate_tracker.retained_birth_confirmations)
            output.clear_streak = state["clear_streak"]
            output.lifecycle_state = state["lifecycle"]
            output.magnitude_trend = state["trend"]
            output.task_scale_probability = probability
            output.task_scale_probability_mean_identified = (
                probability_mean_identified)
            output.task_scale_probability_statistic = (
                "POSTERIOR_MEAN" if probability_mean_identified
                else "CONSERVATIVE_UPPER_BOUND")
            output.displacement_rms_m = magnitude
            output.previous_displacement_rms_m = (
                state["previous_magnitude"]
                if _finite(state["previous_magnitude"]) else 0.0)
            output.displacement_change_m = (
                state["magnitude_change"]
                if _finite(state["magnitude_change"]) else 0.0)
            output.observation_interval_s = (
                state["observation_interval"]
                if _finite(state["observation_interval"]) else 0.0)
            output.displacement_rate_mps = (
                state["displacement_rate"]
                if _finite(state["displacement_rate"]) else 0.0)
            output.displacement_sigma_upper_m = sigma if _finite(sigma) else 0.0
            output.velocity_sigma_upper_mps = (
                velocity_sigma if _finite(velocity_sigma) else 0.0)
            output.displacement_uncertainty_defined = (
                measurement_valid and _finite(sigma))
            output.velocity_uncertainty_defined = (
                measurement_valid and _finite(velocity_sigma))
            output.direction_change_rad = (
                state["direction_change"]
                if _finite(state["direction_change"]) else 0.0)
            output.support_radius_m = support_radius_by_gdu.get(key, 0.0)
            output.mean_displacement = _vector(mean_displacement)
            output.mean_velocity = _vector(mean_velocity)
            output.reference_support = [_point(value) for value in references]
            current_support = [
                tuple(reference[axis] + displacement[axis] for axis in range(3))
                for reference, displacement in zip(references, displacements)
            ]
            output.current_support = [_point(value) for value in current_support]
            output.displacement_samples = [_vector(value) for value in displacements]
            output.velocity_samples = [_vector(value) for value in velocities]
            if active and measurement_valid:
                forecasts, prediction_valid = predict_support_horizons(
                    references, displacements, velocities, self.horizons)
                forecast_allowed = (
                    prediction_valid and evidence is not None
                    and kinematic_forecast_allowed(
                        int(evidence.observable_rank), observable_ranks,
                        evidence.velocity_minimum_detectable_effect_defined,
                        float(evidence.velocity_minimum_detectable_effect_mps),
                        sigma, velocity_sigma))
                for horizon, forecast_values in zip(self.horizons, forecasts):
                    positions, predicted = forecast_values
                    forecast = GduForecast()
                    forecast.horizon_s = horizon
                    forecast.valid = forecast_allowed
                    forecast.model = "CONSTANT_VELOCITY_KINEMATIC_ONLY"
                    forecast.position_sigma_upper_m = (
                        sigma + horizon * velocity_sigma
                        if _finite(sigma) and _finite(velocity_sigma) else 0.0)
                    forecast.predicted_support = [_point(value) for value in positions]
                    forecast.predicted_displacement = [_vector(value) for value in predicted]
                    output.forecasts.append(forecast)

            if not measurement_valid:
                temporarily_unobservable += 1
                output.active_now = False
                output.validity = max(1, int(output.validity))
                output.lifecycle_state = LIFECYCLE_INCONCLUSIVE
                output.magnitude_trend = 0  # MAGNITUDE_UNKNOWN
                output.previous_displacement_rms_m = output.displacement_rms_m
                output.displacement_change_m = 0.0
                output.observation_interval_s = 0.0
                output.displacement_rate_mps = 0.0
                output.direction_change_rad = 0.0
                output.forecasts = []

                # Once a GDU has entered the retained layer, absence of a new
                # observable coordinate cannot move it back to its reference
                # surface.  Carry exactly the last measured geometry while
                # suppressing stale velocity prediction.
                previous = candidate_last_retained.get(key)
                if state["retained"] and previous is not None:
                    output.support_radius_m = previous.support_radius_m
                    output.displacement_rms_m = previous.displacement_rms_m
                    output.previous_displacement_rms_m = (
                        previous.displacement_rms_m)
                    output.displacement_sigma_upper_m = (
                        previous.displacement_sigma_upper_m)
                    output.displacement_uncertainty_defined = (
                        previous.displacement_uncertainty_defined)
                    output.mean_displacement = copy.deepcopy(
                        previous.mean_displacement)
                    output.reference_support = copy.deepcopy(
                        previous.reference_support)
                    output.current_support = copy.deepcopy(
                        previous.current_support)
                    output.displacement_samples = copy.deepcopy(
                        previous.displacement_samples)
                output.mean_velocity = Vector3()
                output.velocity_samples = [Vector3()
                                           for _ in output.displacement_samples]
                output.velocity_sigma_upper_mps = 0.0
                output.velocity_uncertainty_defined = False
            frame.gdus.append(output)

        # A complete detector tile may omit physical samples for a previously
        # detected GDU when its current support is temporarily missing or has
        # moved beyond the local association radius.  That omission is not a
        # clear observation.  Carry the last valid geometry as explicitly
        # INCONCLUSIVE so the downstream hard-avoidance cause persists; never
        # extrapolate its now-unobserved velocity.
        current_keys = set(samples_by_gdu)
        carried = 0
        for key in sorted(set(candidate_tracker.tracked_ids()) - current_keys):
            previous = candidate_last_retained.get(key)
            state = candidate_tracker.mark_missing(
                key, frame.reference_epoch, stamp_s)
            if previous is None or state is None:
                continue
            output = copy.deepcopy(previous)
            output.active_now = False
            output.retained_for_avoidance = bool(state["retained"])
            output.retained_birth_streak = int(
                state["retained_birth_streak"])
            output.retained_birth_confirmations = (
                candidate_tracker.retained_birth_confirmations)
            output.clear_streak = int(state["clear_streak"])
            output.lifecycle_state = LIFECYCLE_INCONCLUSIVE
            output.magnitude_trend = 0  # MAGNITUDE_UNKNOWN
            output.previous_displacement_rms_m = output.displacement_rms_m
            output.displacement_change_m = 0.0
            output.observation_interval_s = 0.0
            output.displacement_rate_mps = 0.0
            output.direction_change_rad = 0.0
            output.mean_velocity = Vector3()
            output.velocity_samples = [Vector3()
                                       for _ in output.displacement_samples]
            output.velocity_sigma_upper_mps = 0.0
            output.velocity_uncertainty_defined = False
            output.validity = max(1, int(output.validity))
            output.detection_admission = 3  # INCONCLUSIVE_ASSOCIATION
            output.forecasts = []
            frame.gdus.append(output)
            carried += 1
        if carried:
            frame.status_detail += ";CARRIED_INCONCLUSIVE={}".format(carried)
        if temporarily_unobservable:
            frame.status_detail += ";TEMPORARILY_UNOBSERVABLE={}".format(
                temporarily_unobservable)

        markers = self._markers(frame)
        self.tracker = candidate_tracker
        # Frame messages are never mutated after publication.  Retain those
        # immutable objects directly; a later missing-observation transition
        # deep-copies only the affected GDU before changing its status fields.
        self.last_retained_gdus = {
            gdu.gdu_id: gdu for gdu in frame.gdus
            if gdu.retained_for_avoidance
        }
        self.last_reference_epoch = frame.reference_epoch
        self.last_accepted_frame_index = accepted_index
        self.last_stamp = stamp_s
        self.last_snapshot_id = frame.source_snapshot_id
        self.publisher.publish(frame)
        self.marker_publisher.publish(markers)

    def _trend_color(self, gdu):
        if gdu.lifecycle_state == LIFECYCLE_PENDING_CONFIRMATION:
            return ColorRGBA(1.0, 0.85, 0.05, 0.8)
        if gdu.lifecycle_state == LIFECYCLE_INCONCLUSIVE:
            return ColorRGBA(0.55, 0.55, 0.55, 0.75)
        if gdu.lifecycle_state == LIFECYCLE_NEWLY_DETECTED:
            return ColorRGBA(1.0, 0.05, 0.05, 0.95)
        if gdu.magnitude_trend == MAGNITUDE_INCREASING:
            return ColorRGBA(1.0, 0.0, 0.7, 0.95)
        if gdu.magnitude_trend == MAGNITUDE_DECREASING:
            return ColorRGBA(0.1, 0.45, 1.0, 0.9)
        if gdu.lifecycle_state in (LIFECYCLE_PERSISTENT,
                                  LIFECYCLE_CLEAR_CANDIDATE):
            return ColorRGBA(1.0, 0.55, 0.0, 0.9)
        return ColorRGBA(0.2, 0.8, 0.2, 0.7)

    def _markers(self, frame):
        markers = MarkerArray()
        clear = Marker()
        clear.header = frame.header
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        marker_id = 0
        for gdu in frame.gdus:
            if not gdu.retained_for_avoidance and not gdu.active_now:
                continue
            color = self._trend_color(gdu)
            support = Marker()
            support.header = frame.header
            support.ns = "deformation_current_support"
            support.id = marker_id
            marker_id += 1
            support.type = Marker.SPHERE_LIST
            support.action = Marker.ADD
            support.pose.orientation.w = 1.0
            diameter = max(0.05, 2.0 * gdu.support_radius_m)
            support.scale.x = diameter
            support.scale.y = diameter
            support.scale.z = diameter
            support.color = color
            support.points = list(gdu.current_support)
            markers.markers.append(support)

            reference = Marker()
            reference.header = frame.header
            reference.ns = "deformation_fixed_reference_support"
            reference.id = marker_id
            marker_id += 1
            reference.type = Marker.SPHERE_LIST
            reference.action = Marker.ADD
            reference.pose.orientation.w = 1.0
            reference.scale.x = max(0.025, 0.65 * diameter)
            reference.scale.y = reference.scale.x
            reference.scale.z = reference.scale.x
            reference.color = ColorRGBA(0.05, 0.75, 1.0, 0.30)
            reference.points = list(gdu.reference_support)
            markers.markers.append(reference)

            for forecast_index, forecast in enumerate(gdu.forecasts):
                if not forecast.valid:
                    continue
                predicted = Marker()
                predicted.header = frame.header
                predicted.ns = "deformation_kinematic_forecast"
                predicted.id = marker_id
                marker_id += 1
                predicted.type = Marker.SPHERE_LIST
                predicted.action = Marker.ADD
                predicted.pose.orientation.w = 1.0
                predicted.scale.x = max(0.035, 0.75 * diameter)
                predicted.scale.y = predicted.scale.x
                predicted.scale.z = predicted.scale.x
                fade = 1.0 / (1.0 + float(forecast_index))
                predicted.color = ColorRGBA(0.75, 0.15, 1.0,
                                            0.38 * fade)
                predicted.points = list(forecast.predicted_support)
                markers.markers.append(predicted)

                trail = Marker()
                trail.header = frame.header
                trail.ns = "deformation_forecast_correspondence"
                trail.id = marker_id
                marker_id += 1
                trail.type = Marker.LINE_LIST
                trail.action = Marker.ADD
                trail.pose.orientation.w = 1.0
                trail.scale.x = 0.01
                trail.color = ColorRGBA(0.75, 0.15, 1.0, 0.25 * fade)
                for current_point, predicted_point in zip(
                        gdu.current_support, forecast.predicted_support):
                    trail.points.extend([current_point, predicted_point])
                markers.markers.append(trail)

            if not gdu.reference_support:
                continue
            center = _mean([_point_tuple(point) for point in gdu.reference_support])
            displacement = _vector_tuple(gdu.mean_displacement)
            physical_norm = math.sqrt(sum(value * value for value in displacement))
            if physical_norm <= 0.0:
                continue
            display_length = min(self.maximum_arrow_m,
                                 self.display_scale * physical_norm)
            scale = display_length / physical_norm
            arrow = Marker()
            arrow.header = frame.header
            arrow.ns = "deformation_magnitude_trend"
            arrow.id = marker_id
            marker_id += 1
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.pose.orientation.w = 1.0
            arrow.scale.x = 0.018
            arrow.scale.y = 0.035
            arrow.scale.z = 0.045
            arrow.color = color
            arrow.points = [
                _point(center),
                _point(tuple(center[axis] + scale * displacement[axis]
                             for axis in range(3))),
            ]
            markers.markers.append(arrow)
        return markers


if __name__ == "__main__":
    rospy.init_node("deformation_evolution")
    DeformationEvolutionNode()
    rospy.spin()
