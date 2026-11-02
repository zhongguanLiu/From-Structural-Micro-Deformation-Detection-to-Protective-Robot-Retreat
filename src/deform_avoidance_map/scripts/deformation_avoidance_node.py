#!/usr/bin/env python3
"""Publish finite global-frame deformation avoidance costs, never obstacles."""

import copy
import math

import rospy
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid, MapMetaData

from deform_evolution_map.msg import DeformationEvolutionFrame, GduEvolution
from deform_avoidance_map.msg import (
    DeformationAvoidanceAlert,
    DeformationAvoidanceMap,
)
from deform_avoidance_map.raster import (
    GridSpec,
    fuse_overlay,
    rasterize_policy_sources,
)


def _point_tuple(point):
    return (float(point.x), float(point.y), float(point.z))


def _mean_point(points):
    if not points:
        return Point()
    return Point(
        x=sum(point.x for point in points) / len(points),
        y=sum(point.y for point in points) / len(points),
        z=sum(point.z for point in points) / len(points),
    )


class DeformationAvoidanceNode:
    def __init__(self):
        prefix = "~deform_avoidance/"
        self.world_frame = rospy.get_param(prefix + "world_frame", "camera_init")
        self.hard_radius = float(rospy.get_param(
            prefix + "hard_avoidance_radius_m", 0.75))
        self.warning_radius = float(rospy.get_param(
            prefix + "warning_radius_m", 1.25))
        self.elevated_min_z = float(rospy.get_param(
            prefix + "elevated_source_min_z_m", 2.70))
        self.elevated_hard_radius = float(rospy.get_param(
            prefix + "elevated_hard_projection_radius_m", 0.25))
        self.elevated_warning_radius = float(rospy.get_param(
            prefix + "elevated_warning_projection_radius_m", 0.60))
        # Keep legacy parameter names for existing launch/radius overrides.
        self.hard_cost = int(rospy.get_param(prefix + "hard_cost", 99))
        self.warning_peak_cost = int(rospy.get_param(
            prefix + "warning_peak_cost", 98))
        self.overlay_as_fused = bool(rospy.get_param(
            prefix + "publish_overlay_as_fused_without_base", False))
        self.include_reference = bool(rospy.get_param(
            prefix + "include_reference_and_current_support", True))
        self.vertical_projection_policy = str(rospy.get_param(
            prefix + "vertical_projection_policy",
            "ALL_ACTIVE_SUPPORT_TO_GROUND"))
        if self.vertical_projection_policy != "ALL_ACTIVE_SUPPORT_TO_GROUND":
            raise ValueError("unsupported vertical projection policy")
        self.forecast_visualization_only = bool(rospy.get_param(
            prefix + "forecast_is_visualization_only", True))
        configured_forecast_topics = rospy.get_param(
            prefix + "forecast_overlay_topics", {
                "0.5": "/deform/avoidance/forecast_0p5s",
                "1.0": "/deform/avoidance/forecast_1p0s",
                "2.0": "/deform/avoidance/forecast_2p0s",
            })
        self.forecast_publishers = {}
        for horizon_text, topic in configured_forecast_topics.items():
            horizon = round(float(horizon_text), 9)
            self.forecast_publishers[horizon] = rospy.Publisher(
                str(topic), OccupancyGrid, queue_size=2, latch=True)
        if (self.hard_radius < 0.0 or self.warning_radius < self.hard_radius
                or not math.isfinite(self.elevated_min_z)
                or self.elevated_hard_radius < 0.0
                or self.elevated_warning_radius < self.elevated_hard_radius
                or not 1 <= self.warning_peak_cost < self.hard_cost <= 99):
            raise ValueError("invalid avoidance radii or costs")
        self.config_spec = GridSpec.from_metric_bounds(
            float(rospy.get_param(prefix + "resolution_m", 0.10)),
            float(rospy.get_param(prefix + "width_m", 40.0)),
            float(rospy.get_param(prefix + "height_m", 40.0)),
            float(rospy.get_param(prefix + "origin_x_m", -20.0)),
            float(rospy.get_param(prefix + "origin_y_m", -20.0)),
        )
        input_topic = rospy.get_param(prefix + "input_topic",
                                      "/deform/evolution/frame")
        base_topic = rospy.get_param(prefix + "base_map_topic", "/map")
        self.overlay_publisher = rospy.Publisher(
            rospy.get_param(prefix + "current_overlay_topic",
                            "/deform/avoidance/current_overlay"),
            OccupancyGrid, queue_size=2, latch=True)
        self.fused_publisher = rospy.Publisher(
            rospy.get_param(prefix + "fused_map_topic",
                            "/deform/avoidance/fused_map"),
            OccupancyGrid, queue_size=2, latch=True)
        self.frame_publisher = rospy.Publisher(
            rospy.get_param(prefix + "frame_topic", "/deform/avoidance/frame"),
            DeformationAvoidanceMap, queue_size=2, latch=True)
        self.alert_publisher = rospy.Publisher(
            rospy.get_param(prefix + "alert_topic", "/deform/avoidance/alerts"),
            DeformationAvoidanceAlert, queue_size=16, latch=True)
        self.base_subscriber = rospy.Subscriber(
            base_topic, OccupancyGrid, self._base_callback, queue_size=1)
        self.evolution_subscriber = rospy.Subscriber(
            input_topic, DeformationEvolutionFrame,
            self._evolution_callback, queue_size=2)
        self.base_map = None
        self.source_cache = {}
        self.last_trend = {}
        self.map_revision = 0
        self.last_accepted_frame_index = None
        self.last_snapshot_id = None
        self.last_stamp = None
        rospy.loginfo(
            "[deform_avoidance] finite soft costs; ordinary core/warning=%.3f/%.3f m; "
            "elevated z>=%.3f m projection core/warning=%.3f/%.3f m",
            self.hard_radius, self.warning_radius, self.elevated_min_z,
            self.elevated_hard_radius, self.elevated_warning_radius)

    def _base_callback(self, message):
        orientation = message.info.origin.orientation
        axis_aligned = (abs(orientation.x) < 1.0e-9
                        and abs(orientation.y) < 1.0e-9
                        and abs(orientation.z) < 1.0e-9
                        and abs(abs(orientation.w) - 1.0) < 1.0e-9)
        if (message.header.frame_id != self.world_frame or not axis_aligned
                or message.info.resolution <= 0.0
                or len(message.data) != message.info.width * message.info.height):
            rospy.logwarn_throttle(
                5.0, "[deform_avoidance] base map ignored: frame, orientation or dimensions mismatch")
            return
        self.base_map = copy.deepcopy(message)

    def _spec(self):
        if self.base_map is None:
            return self.config_spec
        info = self.base_map.info
        return GridSpec(info.resolution, info.width, info.height,
                        info.origin.position.x, info.origin.position.y)

    def _grid(self, header, spec, data):
        message = OccupancyGrid()
        message.header = copy.deepcopy(header)
        message.header.frame_id = self.world_frame
        message.info = MapMetaData()
        message.info.map_load_time = header.stamp
        message.info.resolution = spec.resolution
        message.info.width = spec.width
        message.info.height = spec.height
        message.info.origin.position.x = spec.origin_x
        message.info.origin.position.y = spec.origin_y
        message.info.origin.orientation.w = 1.0
        # Raster/fusion helpers return a fresh list owned by this transaction;
        # transfer it directly instead of duplicating every 400x400 cell.
        message.data = data if isinstance(data, list) else list(data)
        return message

    def _cached_source(self, gdu):
        # Deserialized evolution messages are immutable after this callback.
        # Holding their Point objects keeps the exact coordinates while
        # avoiding four deep copies of every physical/forecast sample.
        reference = tuple(gdu.reference_support)
        current = tuple(gdu.current_support)
        forecasts = {}
        for forecast in gdu.forecasts:
            if forecast.valid:
                forecasts[round(float(forecast.horizon_s), 9)] = {
                    "points": tuple(forecast.predicted_support),
                    "position_sigma": max(
                        0.0, float(forecast.position_sigma_upper_m)),
                }
        geometry = current if current else reference
        median_z = 0.0
        if geometry:
            ordered_z = sorted(float(point.z) for point in geometry)
            middle = len(ordered_z) // 2
            median_z = (ordered_z[middle] if len(ordered_z) % 2
                        else 0.5 * (ordered_z[middle - 1] + ordered_z[middle]))
        elevated = bool(geometry and median_z >= self.elevated_min_z)
        return {
            "reference": reference,
            "current": current,
            "forecasts": forecasts,
            "support_radius": max(0.0, float(gdu.support_radius_m)),
            "center": _mean_point(current if current else reference),
            "magnitude": float(gdu.displacement_rms_m),
            "trend": int(gdu.magnitude_trend),
            "median_z": median_z,
            "elevated_projection": elevated,
        }

    def _publish_alert(self, header, source_id, source, event_type):
        alert = DeformationAvoidanceAlert()
        alert.header = copy.deepcopy(header)
        alert.header.frame_id = self.world_frame
        alert.map_revision = self.map_revision
        alert.event_type = event_type
        alert.source_id = source_id
        alert.position = copy.deepcopy(source.get("center", Point()))
        alert.displacement_rms_m = float(source.get("magnitude", 0.0))
        alert.magnitude_trend = int(source.get("trend", 0))
        alert.policy = "DETECTED_DEFORMATION_HEIGHT_AWARE_GROUND_PROJECTION"
        self.alert_publisher.publish(alert)

    def _sources(self, cached_sources, selector):
        output = []
        for source in cached_sources.values():
            points = selector(source)
            for point in points:
                if source["elevated_projection"]:
                    # The observed points already describe the horizontal
                    # footprint.  Adding the GDU bounding radius here would
                    # replace that footprint by one large disc and could
                    # unnecessarily close every side passage.
                    output.append((point.x, point.y, 0.0,
                                   self.elevated_hard_radius,
                                   self.elevated_warning_radius))
                else:
                    output.append((point.x, point.y, source["support_radius"],
                                   self.hard_radius, self.warning_radius))
        return output

    def _policy_radii(self, source):
        if source["elevated_projection"]:
            return (0.0, self.elevated_hard_radius,
                    self.elevated_warning_radius)
        return (source["support_radius"], self.hard_radius,
                self.warning_radius)

    def _out_of_bounds_ids(self, spec, cached_sources, selector):
        output = []
        for source_id, source in cached_sources.items():
            support, _, warning = self._policy_radii(source)
            radius = support + warning
            if any(not spec.circle_within_grid(point.x, point.y, radius)
                   for point in selector(source)):
                output.append(source_id)
        return sorted(output)

    def _forecast_sources(self, cached_sources, horizon):
        output = []
        for source in cached_sources.values():
            forecast = source["forecasts"].get(horizon)
            if forecast is None:
                continue
            support, hard, warning = self._policy_radii(source)
            radius = support + forecast["position_sigma"]
            for point in forecast["points"]:
                output.append((point.x, point.y, radius, hard, warning))
        return output

    def _forecast_out_of_bounds_ids(self, spec, cached_sources, horizon):
        output = []
        for source_id, source in cached_sources.items():
            forecast = source["forecasts"].get(horizon)
            if forecast is None:
                continue
            support, _, warning = self._policy_radii(source)
            radius = support + warning + forecast["position_sigma"]
            if any(not spec.circle_within_grid(point.x, point.y, radius)
                   for point in forecast["points"]):
                output.append(source_id)
        return sorted(output)

    def _evolution_callback(self, frame):
        if frame.header.frame_id != self.world_frame:
            rospy.logerr_throttle(
                2.0, "[deform_avoidance] evolution frame does not match configured world frame")
            return
        if (not frame.complete_snapshot or not frame.input_contract_valid
                or int(frame.validity) >= 2):
            rospy.logwarn_throttle(
                2.0,
                "[deform_avoidance] incomplete/invalid evolution frame ignored; last complete map retained: %s",
                frame.status_detail)
            return
        accepted_index = int(frame.source_accepted_frame_index)
        stamp_s = frame.header.stamp.to_sec()
        if (not frame.source_snapshot_id or not math.isfinite(stamp_s)
                or stamp_s <= 0.0
                or (frame.valid_until != rospy.Time(0)
                    and frame.valid_until < frame.header.stamp)):
            rospy.logerr_throttle(
                2.0, "[deform_avoidance] invalid snapshot identity or validity interval")
            return
        if self.last_accepted_frame_index is not None:
            if (accepted_index == self.last_accepted_frame_index
                    and frame.source_snapshot_id == self.last_snapshot_id):
                return
            if (accepted_index <= self.last_accepted_frame_index
                    or (self.last_stamp is not None and stamp_s <= self.last_stamp)):
                rospy.logerr_throttle(
                    2.0, "[deform_avoidance] non-monotonic evolution snapshot ignored")
                return

        candidate_revision = self.map_revision + 1
        # Cached source messages are immutable after construction.  A shallow
        # dictionary copy therefore preserves transactional rollback without
        # cloning thousands of support points on every evolution frame.  The
        # only in-place-looking change below (dropping stale forecasts) uses
        # copy-on-write for that one source dictionary.
        candidate_cache = dict(self.source_cache)
        candidate_last_trend = dict(self.last_trend)
        retained_ids = set()
        newly_detected = []
        updated = []
        for gdu in frame.gdus:
            source_id = gdu.gdu_id
            # active_now is the raw instantaneous detector decision.  A new
            # source is admitted to this planning layer only after the generic
            # evolution birth streak sets retained_for_avoidance; pending
            # candidates remain visible in evolution output/markers but never
            # enter the source cache, alerts, or any current/forecast raster.
            if gdu.active_now and gdu.retained_for_avoidance:
                if source_id not in candidate_cache:
                    newly_detected.append(source_id)
                previous_trend = candidate_last_trend.get(source_id)
                if (previous_trend is not None
                        and previous_trend != GduEvolution.MAGNITUDE_INCREASING
                        and gdu.magnitude_trend == GduEvolution.MAGNITUDE_INCREASING):
                    updated.append(source_id)
                candidate_cache[source_id] = self._cached_source(gdu)
                candidate_last_trend[source_id] = int(gdu.magnitude_trend)
            if gdu.retained_for_avoidance:
                retained_ids.add(source_id)
                if source_id not in candidate_cache:
                    candidate_cache[source_id] = self._cached_source(gdu)
                elif not gdu.active_now:
                    # Retention is a protective current-geometry hysteresis,
                    # not permission to extend an older kinematic forecast.
                    # A forecast is anchored to the detector/evolution
                    # snapshot that made the source active.  Once the newest
                    # complete snapshot is only CLEAR_CANDIDATE,
                    # INCONCLUSIVE, or otherwise non-active, keep the last
                    # committed soft-avoidance geometry but remove all stale
                    # future nodes until a new active snapshot supplies them.
                    retained_source = dict(candidate_cache[source_id])
                    retained_source["forecasts"] = {}
                    candidate_cache[source_id] = retained_source

        removed = sorted(set(candidate_cache) - retained_ids)
        removed_sources = {source_id: candidate_cache[source_id]
                           for source_id in removed}
        for source_id in removed:
            candidate_cache.pop(source_id, None)
            candidate_last_trend.pop(source_id, None)

        active_cache = {source_id: candidate_cache[source_id]
                        for source_id in sorted(retained_ids)
                        if source_id in candidate_cache}
        spec = self._spec()
        current_sources = self._sources(
            active_cache,
            lambda source: ((source["reference"] + source["current"])
                            if self.include_reference else source["current"]))
        current_data = rasterize_policy_sources(
            spec, current_sources, self.hard_cost, self.warning_peak_cost)
        overlay = self._grid(frame.header, spec, current_data)

        map_frame = DeformationAvoidanceMap()
        map_frame.header = copy.deepcopy(overlay.header)
        map_frame.source_snapshot_id = frame.source_snapshot_id
        map_frame.source_accepted_frame_index = frame.source_accepted_frame_index
        map_frame.map_revision = candidate_revision
        map_frame.valid_until = frame.valid_until
        map_frame.prediction_valid_until = frame.prediction_valid_until
        map_frame.policy = "FINITE_SOFT_DEFORMATION_HEIGHT_AWARE_GROUND_PROJECTION"
        map_frame.source_geometry_policy = (
            "UNION_OF_FIXED_REFERENCE_AND_CURRENT_DETECTED_SUPPORT;"
            + self.vertical_projection_policy
            + ";ELEVATED_SUPPORT_USES_POINT_FOOTPRINT_CLEARANCE"
            if self.include_reference
            else "CURRENT_DETECTED_SUPPORT_ONLY;"
            + self.vertical_projection_policy
            + ";ELEVATED_SUPPORT_USES_POINT_FOOTPRINT_CLEARANCE")
        map_frame.amplitude_policy = (
            "MAGNITUDE_TREND_IS_ATTRIBUTE_NOT_SPATIAL_EXPANSION")
        map_frame.forecast_visualization_only = self.forecast_visualization_only
        map_frame.hard_avoidance_radius_m = self.hard_radius
        map_frame.warning_radius_m = self.warning_radius
        map_frame.elevated_source_min_z_m = self.elevated_min_z
        map_frame.elevated_hard_projection_radius_m = self.elevated_hard_radius
        map_frame.elevated_warning_projection_radius_m = self.elevated_warning_radius
        map_frame.current_overlay = overlay
        map_frame.active_source_ids = sorted(active_cache)
        map_frame.ordinary_projection_source_ids = sorted(
            source_id for source_id, source in active_cache.items()
            if not source["elevated_projection"])
        map_frame.elevated_projection_source_ids = sorted(
            source_id for source_id, source in active_cache.items()
            if source["elevated_projection"])
        current_selector = (
            (lambda source: source["reference"] + source["current"])
            if self.include_reference else (lambda source: source["current"]))
        current_out_of_bounds = self._out_of_bounds_ids(
            spec, active_cache, current_selector)
        map_frame.out_of_bounds_current_source_ids = current_out_of_bounds
        map_frame.all_current_sources_within_map = not current_out_of_bounds

        horizons = sorted({horizon for source in active_cache.values()
                           for horizon in source["forecasts"]})
        forecast_publications = []
        for horizon in horizons:
            # The policy raster is a pointwise maximum over independent
            # sources.  Reuse the already exact current raster and max-fuse a
            # forecast-only raster instead of rasterizing every current
            # support again for every horizon.  This is algebraically
            # identical to rasterizing their concatenated source list.
            forecast_only_data = rasterize_policy_sources(
                spec, self._forecast_sources(active_cache, horizon),
                self.hard_cost, self.warning_peak_cost)
            forecast_data = fuse_overlay(current_data, forecast_only_data)
            forecast_grid = self._grid(frame.header, spec, forecast_data)
            map_frame.forecast_horizons_s.append(horizon)
            map_frame.forecast_overlays.append(forecast_grid)
            publisher = self.forecast_publishers.get(round(horizon, 9))
            if publisher is not None:
                forecast_publications.append((publisher, forecast_grid))
        forecast_out_of_bounds = set()
        for horizon in horizons:
            forecast_out_of_bounds.update(self._forecast_out_of_bounds_ids(
                spec, active_cache, horizon))
        map_frame.out_of_bounds_forecast_source_ids = sorted(
            forecast_out_of_bounds)
        map_frame.all_forecast_sources_within_map = not forecast_out_of_bounds

        fused = None
        if self.base_map is not None:
            if (self.base_map.info.width == overlay.info.width
                    and self.base_map.info.height == overlay.info.height
                    and abs(self.base_map.info.resolution
                            - overlay.info.resolution) < 1.0e-12):
                # Rebuild the message shell instead of first copying the full
                # base cell array and immediately replacing it with fusion.
                fused = OccupancyGrid()
                fused.header = copy.deepcopy(overlay.header)
                fused.info = copy.deepcopy(self.base_map.info)
                fused.data = fuse_overlay(self.base_map.data, overlay.data)
        elif self.overlay_as_fused:
            fused = overlay

        # Commit only after every source, raster and message has been built.
        self.source_cache = candidate_cache
        self.last_trend = candidate_last_trend
        self.map_revision = candidate_revision
        self.last_accepted_frame_index = accepted_index
        self.last_snapshot_id = frame.source_snapshot_id
        self.last_stamp = stamp_s
        self.overlay_publisher.publish(overlay)
        for publisher, forecast_grid in forecast_publications:
            publisher.publish(forecast_grid)
        self.frame_publisher.publish(map_frame)
        if fused is not None:
            self.fused_publisher.publish(fused)

        for source_id in newly_detected:
            if source_id in self.source_cache:
                self._publish_alert(
                    frame.header, source_id, self.source_cache[source_id],
                    DeformationAvoidanceAlert.NEW_DEFORMATION_SOURCE)
                rospy.logwarn(
                    "[deform_avoidance] new deformation source %s: finite soft-cost map updated",
                    source_id)
        for source_id in updated:
            if source_id in self.source_cache:
                self._publish_alert(
                    frame.header, source_id, self.source_cache[source_id],
                    DeformationAvoidanceAlert.UPDATED_DEFORMATION_SOURCE)
        for source_id, source in removed_sources.items():
            self._publish_alert(
                frame.header, source_id, source,
                DeformationAvoidanceAlert.CLEARED_DEFORMATION_SOURCE)


if __name__ == "__main__":
    rospy.init_node("deformation_avoidance")
    DeformationAvoidanceNode()
    rospy.spin()
