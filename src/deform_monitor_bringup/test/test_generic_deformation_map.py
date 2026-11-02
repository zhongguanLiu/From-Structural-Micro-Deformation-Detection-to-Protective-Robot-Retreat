#!/usr/bin/env python3
"""ROS integration test with a generic, non-semantic deformation source."""

import math
import threading
import time
import unittest

import rospy

from deform_monitor_v3.msg import (
    V1DeformationSample,
    V1DetectionTile,
    V1ReferenceSupportSample,
    V1SpatialDetectionEvidence,
    V1StableId128,
)
from deform_evolution_map.msg import DeformationEvolutionFrame, GduEvolution
from deform_avoidance_map.msg import DeformationAvoidanceMap


class GenericDeformationMapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rospy.init_node("test_generic_deformation_map", anonymous=True)
        cls.lock = threading.Lock()
        cls.evolution = None
        cls.avoidance = None
        cls.evolution_sub = rospy.Subscriber(
            "/deform/evolution/frame", DeformationEvolutionFrame,
            cls._evolution_callback, queue_size=10)
        cls.avoidance_sub = rospy.Subscriber(
            "/deform/avoidance/frame", DeformationAvoidanceMap,
            cls._avoidance_callback, queue_size=10)
        cls.publisher = rospy.Publisher(
            "/deform/v1/detection_tiles", V1DetectionTile,
            queue_size=2, latch=True)
        deadline = time.monotonic() + 5.0
        while (cls.publisher.get_num_connections() == 0
               and time.monotonic() < deadline):
            time.sleep(0.02)

    @classmethod
    def _evolution_callback(cls, message):
        with cls.lock:
            cls.evolution = message

    @classmethod
    def _avoidance_callback(cls, message):
        with cls.lock:
            cls.avoidance = message

    def wait_for(self, predicate, timeout=8.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not rospy.is_shutdown():
            with self.lock:
                evolution = self.evolution
                avoidance = self.avoidance
            if predicate(evolution, avoidance):
                return evolution, avoidance
            time.sleep(0.02)
        self.fail("timed out waiting for evolution/avoidance output")

    @staticmethod
    def tile(stamp, displacement, velocity, probability, snapshot_low):
        tile = V1DetectionTile()
        tile.common.header.stamp = stamp
        tile.common.header.frame_id = "camera_init"
        tile.common.reference_epoch = 7
        tile.common.accepted_frame_index = snapshot_low
        tile.common.snapshot_id = V1StableId128(high=9, low=snapshot_low)
        tile.common.valid_until = stamp + rospy.Duration(1.0)
        tile.common.validity = 1  # INCONCLUSIVE: expected uncalibrated research mode.
        tile.source_snapshot_id = V1StableId128(high=9, low=snapshot_low)
        tile.lod = 2
        tile.tile_key = "roi-root/lod-2"

        gdu_id = V1StableId128(high=1, low=2)
        support = V1ReferenceSupportSample()
        support.gdu_id = gdu_id
        support.surfel_id = V1StableId128(high=3, low=4)
        support.position.x = 2.0
        support.position.y = 1.0
        support.support_radius_m = 0.10
        tile.reference_support.append(support)

        sample = V1DeformationSample()
        sample.gdu_id = gdu_id
        sample.position = support.position
        sample.displacement_mean.x = displacement
        sample.velocity_mean.x = velocity
        sample.displacement_covariance_row_major = [
            1.0e-8, 0.0, 0.0,
            0.0, 1.0e-8, 0.0,
            0.0, 0.0, 1.0e-8,
        ]
        sample.velocity_covariance_row_major = [
            1.0e-8, 0.0, 0.0,
            0.0, 1.0e-8, 0.0,
            0.0, 0.0, 1.0e-8,
        ]
        sample.observable_projector_row_major = [
            1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]
        sample.active_probability_upper = probability
        sample.validity = 1
        sample.detection_admission = 6
        tile.physical_deformation.append(sample)

        evidence = V1SpatialDetectionEvidence()
        evidence.gdu_id = gdu_id
        evidence.position = support.position
        evidence.task_scale_probability_upper = probability
        evidence.observable_rank = 1
        # The clear portion of this fixture represents an actually observed
        # one-direction task coordinate.  Without this flag a low posterior is
        # inconclusive, not valid evidence that the source has cleared.
        evidence.all_task_directions_detectable = True
        evidence.velocity_minimum_detectable_effect_defined = True
        evidence.velocity_minimum_detectable_effect_mps = 0.001
        # This is the duration of a velocity-threshold event, not an
        # observability duration.  A zero value must still permit forecasting
        # when velocity MDE, rank and posterior uncertainty are defined.
        evidence.continuous_valid_observation_duration_s = 0.0
        evidence.validity = 1
        evidence.detection_admission = 6
        tile.detection_evidence.append(evidence)
        return tile

    @staticmethod
    def cost_at(grid, x, y):
        column = int(math.floor(
            (x - grid.info.origin.position.x) / grid.info.resolution))
        row = int(math.floor(
            (y - grid.info.origin.position.y) / grid.info.resolution))
        return grid.data[row * grid.info.width + column]

    def test_t0_t1_forecast_and_clear(self):
        base = rospy.Time.now()
        self.publisher.publish(self.tile(base, 0.002, 0.001, 0.99, 1))
        pending1, pending_map1 = self.wait_for(
            lambda e, a: e is not None and a is not None
            and e.source_snapshot_id.endswith("0000000000000001")
            and a.source_snapshot_id.endswith("0000000000000001"))
        self.assertTrue(pending1.gdus[0].active_now)
        self.assertFalse(pending1.gdus[0].retained_for_avoidance)
        self.assertEqual(pending1.gdus[0].retained_birth_streak, 1)
        self.assertEqual(pending1.gdus[0].retained_birth_confirmations, 3)
        self.assertEqual(
            pending1.gdus[0].lifecycle_state,
            GduEvolution.LIFECYCLE_PENDING_CONFIRMATION)
        self.assertEqual(list(pending_map1.active_source_ids), [])
        self.assertEqual(max(pending_map1.current_overlay.data), 0)

        self.publisher.publish(self.tile(
            base + rospy.Duration(1.0), 0.002, 0.001, 0.99, 2))
        pending2, pending_map2 = self.wait_for(
            lambda e, a: e is not None and a is not None
            and e.source_snapshot_id.endswith("0000000000000002")
            and a.source_snapshot_id.endswith("0000000000000002"))
        self.assertTrue(pending2.gdus[0].active_now)
        self.assertFalse(pending2.gdus[0].retained_for_avoidance)
        self.assertEqual(pending2.gdus[0].retained_birth_streak, 2)
        self.assertEqual(list(pending_map2.active_source_ids), [])

        self.publisher.publish(self.tile(
            base + rospy.Duration(2.0), 0.002, 0.001, 0.99, 3))
        evolution0, avoidance0 = self.wait_for(
            lambda e, a: e is not None and a is not None
            and e.source_snapshot_id.endswith("0000000000000003")
            and a.source_snapshot_id.endswith("0000000000000003"))
        self.assertEqual(len(evolution0.gdus), 1)
        self.assertTrue(evolution0.gdus[0].active_now)
        self.assertTrue(evolution0.gdus[0].retained_for_avoidance)
        self.assertEqual(evolution0.gdus[0].retained_birth_streak, 3)
        self.assertEqual(evolution0.gdus[0].lifecycle_state,
                         GduEvolution.LIFECYCLE_NEWLY_DETECTED)
        self.assertFalse(
            evolution0.gdus[0].task_scale_probability_mean_identified)
        self.assertEqual(
            evolution0.gdus[0].task_scale_probability_statistic,
            "CONSERVATIVE_UPPER_BOUND")
        self.assertTrue(evolution0.gdus[0].displacement_uncertainty_defined)
        self.assertTrue(evolution0.gdus[0].velocity_uncertainty_defined)
        self.assertAlmostEqual(evolution0.gdus[0].current_support[0].x,
                               2.002, places=9)
        self.assertTrue(evolution0.gdus[0].forecasts[0].valid)
        self.assertAlmostEqual(
            evolution0.gdus[0].forecasts[0].predicted_support[0].x,
            2.0025, places=9)
        self.assertEqual(self.cost_at(
            avoidance0.current_overlay, 2.0, 1.0), 99)
        self.assertTrue(avoidance0.forecast_visualization_only)
        self.assertTrue(evolution0.complete_snapshot)
        self.assertTrue(evolution0.input_contract_valid)
        self.assertTrue(avoidance0.all_current_sources_within_map)
        self.assertTrue(avoidance0.all_forecast_sources_within_map)
        self.assertEqual(len(avoidance0.forecast_overlays), 3)
        self.publisher.publish(self.tile(
            base + rospy.Duration(3.0), 0.005, 0.001, 0.99, 4))
        evolution1, avoidance1 = self.wait_for(
            lambda e, a: e is not None and a is not None
            and e.source_snapshot_id.endswith("0000000000000004")
            and a.source_snapshot_id.endswith("0000000000000004"))
        self.assertEqual(evolution1.gdus[0].magnitude_trend,
                         GduEvolution.MAGNITUDE_INCREASING)
        self.assertAlmostEqual(evolution1.gdus[0].displacement_rate_mps,
                               0.003, places=9)
        # Amplitude is not a radius multiplier: the frozen policy clearances
        # and source identity stay fixed.  The exact continuous current
        # support does move from x=2.002 to x=2.005, so warning-edge cell
        # values are allowed to change without implying spatial propagation.
        self.assertAlmostEqual(
            evolution1.gdus[0].current_support[0].x, 2.005, places=9)
        self.assertEqual(avoidance0.active_source_ids,
                         avoidance1.active_source_ids)
        self.assertAlmostEqual(avoidance1.hard_avoidance_radius_m,
                               avoidance0.hard_avoidance_radius_m, places=12)
        self.assertAlmostEqual(avoidance1.warning_radius_m,
                               avoidance0.warning_radius_m, places=12)
        self.assertEqual(self.cost_at(
            avoidance1.current_overlay, 2.0, 1.0), 99)
        self.assertEqual(self.cost_at(
            avoidance1.current_overlay, 3.5, 1.0), 0)

        # An out-of-order snapshot is reported by the evolution status stream,
        # but the avoidance node must retain the last complete committed map.
        self.publisher.publish(self.tile(
            base + rospy.Duration(3.5), 0.009, 0.001, 0.99, 3))
        invalid_evolution, retained_avoidance = self.wait_for(
            lambda e, a: e is not None and a is not None
            and not e.input_contract_valid
            and e.status_detail == "NON_MONOTONIC_ACCEPTED_FRAME_INDEX"
            and a.source_snapshot_id.endswith("0000000000000004"))
        self.assertFalse(invalid_evolution.complete_snapshot)
        self.assertEqual(list(retained_avoidance.current_overlay.data),
                         list(avoidance1.current_overlay.data))

        for offset in range(3):
            self.publisher.publish(self.tile(
                base + rospy.Duration(4.0 + offset),
                0.0, 0.0, 0.01, 5 + offset))
            expected_suffix = "{:016x}".format(5 + offset)
            evolution_clear, avoidance_clear = self.wait_for(
                lambda e, a, suffix=expected_suffix:
                e is not None and a is not None
                and e.source_snapshot_id.endswith(suffix)
                and a.source_snapshot_id.endswith(suffix))
            if offset == 0:
                # The retained current hard source survives clear hysteresis,
                # but forecasts from the preceding active snapshot are no
                # longer anchored to the newest complete snapshot.
                self.assertTrue(
                    evolution_clear.gdus[0].retained_for_avoidance)
                self.assertFalse(evolution_clear.gdus[0].active_now)
                self.assertEqual(list(avoidance_clear.active_source_ids),
                                 list(avoidance1.active_source_ids))
                self.assertEqual(list(avoidance_clear.forecast_horizons_s), [])
                self.assertEqual(len(avoidance_clear.forecast_overlays), 0)
        self.assertFalse(evolution_clear.gdus[0].retained_for_avoidance)
        self.assertEqual(list(avoidance_clear.active_source_ids), [])
        self.assertEqual(max(avoidance_clear.current_overlay.data), 0)


if __name__ == "__main__":
    import rostest
    rostest.rosrun("deform_monitor_bringup",
                   "generic_deformation_map",
                   GenericDeformationMapTest)
