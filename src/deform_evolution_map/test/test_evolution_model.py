#!/usr/bin/env python3
import math
import random
import unittest

from deform_evolution_map.model import (
    activation_decision,
    EvolutionTracker,
    LIFECYCLE_CLEAR_CANDIDATE,
    LIFECYCLE_INCONCLUSIVE,
    LIFECYCLE_INACTIVE,
    LIFECYCLE_NEWLY_DETECTED,
    LIFECYCLE_PENDING_CONFIRMATION,
    LIFECYCLE_PERSISTENT,
    MAGNITUDE_DECREASING,
    MAGNITUDE_INCREASING,
    MAGNITUDE_STABLE,
    classify_magnitude,
    fresh_observable_measurement,
    kinematic_forecast_allowed,
    magnitude_statistics,
    observable_sample,
    predict_support,
    predict_support_horizons,
)


class EvolutionModelTest(unittest.TestCase):
    def test_activation_requires_an_observable_coordinate(self):
        self.assertEqual(
            activation_decision(False, 4, 1, 0.99, 0.004, 0.0005,
                                0, True),
            (False, False))
        self.assertEqual(
            activation_decision(False, 4, 1, 0.99, 0.004, 0.0005,
                                1, False),
            (True, False))

    def test_rank_zero_physical_carrier_has_no_fresh_measurement(self):
        self.assertFalse(fresh_observable_measurement(0, 0))
        self.assertFalse(fresh_observable_measurement(1, 0))
        self.assertFalse(fresh_observable_measurement(2, 1))
        self.assertTrue(fresh_observable_measurement(0, 1))
        self.assertTrue(fresh_observable_measurement(1, 3))

    def test_clear_requires_every_task_direction_detectable(self):
        self.assertEqual(
            activation_decision(False, 4, 0, 0.01, 0.0001, 0.0005,
                                1, False),
            (False, False))
        self.assertEqual(
            activation_decision(False, 4, 0, 0.01, 0.0001, 0.0005,
                                1, True),
            (False, True))

    def test_strict_admission_is_guarded_by_observability(self):
        self.assertEqual(
            activation_decision(True, 0, 0, 1.0, 0.01, 0.0005,
                                0, True),
            (False, False))
        self.assertEqual(
            activation_decision(True, 0, 0, 1.0, 0.01, 0.0005,
                                1, False),
            (True, False))
        self.assertEqual(
            activation_decision(True, 1, 0, 0.0, 0.0, 0.0005,
                                1, False),
            (False, False))
        self.assertEqual(
            activation_decision(True, 1, 0, 0.0, 0.0, 0.0005,
                                1, True),
            (False, True))

    def test_flat_observable_contraction_matches_dense_reference(self):
        random_generator = random.Random(72813)
        for _ in range(100):
            direction = [random_generator.uniform(-1.0, 1.0)
                         for _ in range(3)]
            norm = math.sqrt(sum(value * value for value in direction))
            direction = [value / norm for value in direction]
            projector = [direction[row] * direction[column]
                         for row in range(3) for column in range(3)]
            factor = [[random_generator.uniform(-0.001, 0.001)
                       for _ in range(3)] for _ in range(3)]
            covariance = [sum(factor[row][inner] * factor[column][inner]
                              for inner in range(3))
                          for row in range(3) for column in range(3)]
            displacement = tuple(random_generator.uniform(-0.01, 0.01)
                                 for _ in range(3))
            velocity = tuple(random_generator.uniform(-0.002, 0.002)
                             for _ in range(3))
            actual = observable_sample(
                displacement, velocity, covariance, covariance, projector)

            dense_projector = tuple(
                tuple(projector[3 * row + column] for column in range(3))
                for row in range(3))
            dense_covariance = tuple(
                tuple(covariance[3 * row + column] for column in range(3))
                for row in range(3))

            def matrix_vector(matrix, vector):
                return tuple(sum(matrix[row][column] * vector[column]
                                 for column in range(3))
                             for row in range(3))

            def matrix_product(first, second):
                return tuple(tuple(sum(
                    first[row][inner] * second[inner][column]
                    for inner in range(3)) for column in range(3))
                    for row in range(3))

            expected_displacement = matrix_vector(
                dense_projector, displacement)
            expected_velocity = matrix_vector(dense_projector, velocity)
            projected_covariance = matrix_product(
                matrix_product(dense_projector, dense_covariance),
                dense_projector)
            expected_trace = max(0.0, sum(
                projected_covariance[index][index] for index in range(3)))
            self.assertIsNotNone(actual)
            for observed, expected in zip(actual[0], expected_displacement):
                self.assertAlmostEqual(observed, expected, places=15)
            for observed, expected in zip(actual[1], expected_velocity):
                self.assertAlmostEqual(observed, expected, places=15)
            self.assertAlmostEqual(actual[2], expected_trace, places=15)
            self.assertAlmostEqual(actual[3], expected_trace, places=15)

    def test_magnitude_is_not_spatial_support(self):
        first, _, _ = magnitude_statistics([(0.002, 0.0, 0.0)], [1.0e-8])
        second, _, _ = magnitude_statistics([(0.005, 0.0, 0.0)], [1.0e-8])
        self.assertAlmostEqual(first, 0.002)
        self.assertAlmostEqual(second, 0.005)
        self.assertEqual(
            classify_magnitude(first, 1.0e-4, second, 1.0e-4,
                               5.0e-4, 1.0),
            MAGNITUDE_INCREASING)

    def test_uncertainty_prevents_false_trend(self):
        self.assertEqual(
            classify_magnitude(0.002, 0.001, 0.0024, 0.001,
                               1.0e-4, 1.0),
            MAGNITUDE_STABLE)
        self.assertEqual(
            classify_magnitude(0.005, 1.0e-4, 0.002, 1.0e-4,
                               5.0e-4, 1.0),
            MAGNITUDE_DECREASING)

    def test_unknown_cross_epoch_correlation_uses_triangle_bound(self):
        # |delta|=1.6 mm would exceed sqrt(2)*1 mm, but not the universally
        # valid 1 mm + 1 mm bound when temporal covariance is unavailable.
        self.assertEqual(
            classify_magnitude(0.002, 0.001, 0.0036, 0.001,
                               1.0e-4, 1.0),
            MAGNITUDE_STABLE)

    def test_correlated_support_uses_largest_covariance_not_sample_count(self):
        _, _, sigma = magnitude_statistics(
            [(0.001, 0.0, 0.0), (0.001, 0.0, 0.0)],
            [1.0e-8, 9.0e-8])
        self.assertAlmostEqual(sigma, 3.0e-4)

    def test_observable_projection_excludes_nullspace_uncertainty(self):
        result = observable_sample(
            (0.002, 0.0, 0.0), (0.001, 0.0, 0.0),
            [1.0e-8, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0],
            [4.0e-8, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 9.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertIsNotNone(result)
        self.assertEqual(result[4], 1)
        self.assertAlmostEqual(result[2], 1.0e-8)
        self.assertAlmostEqual(result[3], 4.0e-8)

    def test_nullspace_vector_is_discarded_but_indefinite_covariance_rejects(self):
        projector = [1.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0]
        covariance = [1.0e-8, 0.0, 0.0,
                      0.0, 1.0e-8, 0.0,
                      0.0, 0.0, 1.0e-8]
        projected = observable_sample(
            (0.002, 0.001, 0.0), (0.003, 0.0, -0.004),
            covariance, covariance, projector)
        self.assertEqual(projected[0], (0.002, 0.0, 0.0))
        self.assertEqual(projected[1], (0.003, 0.0, 0.0))
        indefinite = [1.0e-8, 2.0e-8, 0.0,
                      2.0e-8, 1.0e-8, 0.0,
                      0.0, 0.0, 1.0e-8]
        self.assertIsNone(observable_sample(
            (0.002, 0.0, 0.0), (0.0, 0.0, 0.0),
            indefinite, covariance, projector))

    def test_rank_zero_sample_is_valid_zero_space_not_frame_failure(self):
        covariance = [1.0e-8, 0.0, 0.0,
                      0.0, 2.0e-8, 0.0,
                      0.0, 0.0, 3.0e-8]
        result = observable_sample(
            (0.002, -0.003, 0.004), (0.01, 0.02, -0.03),
            covariance, covariance, [0.0] * 9)
        self.assertEqual(result[0], (0.0, 0.0, 0.0))
        self.assertEqual(result[1], (0.0, 0.0, 0.0))
        self.assertEqual(result[2], 0.0)
        self.assertEqual(result[3], 0.0)
        self.assertEqual(result[4], 0)

    def test_partial_invalid_gdu_statistics_do_not_change_sample_set(self):
        magnitude, _, sigma = magnitude_statistics(
            [(0.001, 0.0, 0.0), (float("nan"), 0.0, 0.0)],
            [1.0e-8, 1.0e-8])
        self.assertEqual(magnitude, 0.0)
        self.assertTrue(math.isinf(sigma))

    def test_fixed_reference_kinematic_prediction(self):
        positions, displacements, valid = predict_support(
            [(1.0, 2.0, 0.0)], [(0.002, 0.0, 0.0)],
            [(0.001, 0.0, 0.0)], 2.0)
        self.assertTrue(valid)
        self.assertAlmostEqual(displacements[0][0], 0.004)
        self.assertAlmostEqual(positions[0][0], 1.004)

    def test_multi_horizon_prediction_matches_independent_calls(self):
        references = [(1.0 + 0.01 * index, 2.0, -0.5)
                      for index in range(37)]
        displacements = [(0.002, -0.001, 0.0005)] * len(references)
        velocities = [(0.001, 0.0003, -0.0002)] * len(references)
        horizons = [0.5, 1.0, 2.0]
        combined, valid = predict_support_horizons(
            references, displacements, velocities, horizons)
        self.assertTrue(valid)
        self.assertEqual(len(combined), len(horizons))
        for horizon, (positions, predicted) in zip(horizons, combined):
            expected_positions, expected_predicted, expected_valid = (
                predict_support(references, displacements, velocities,
                                horizon))
            self.assertTrue(expected_valid)
            self.assertEqual(positions, expected_positions)
            self.assertEqual(predicted, expected_predicted)

    def test_forecast_capability_is_not_velocity_event_duration(self):
        # The detector may observe a finite velocity posterior whose mean is
        # below the registered velocity-event threshold.  Event duration is
        # then exactly zero, but it must not suppress the kinematic forecast.
        self.assertTrue(kinematic_forecast_allowed(
            1, [1], True, 1.0e-3, 1.0e-4, 2.0e-4))
        self.assertTrue(kinematic_forecast_allowed(
            5, [0, 1, 0], True, 1.0e-3, 1.0e-4, 2.0e-4))
        self.assertFalse(kinematic_forecast_allowed(
            1, [1], False, 1.0e-3, 1.0e-4, 2.0e-4))
        self.assertFalse(kinematic_forecast_allowed(
            1, [0], True, 1.0e-3, 1.0e-4, 2.0e-4))
        self.assertFalse(kinematic_forecast_allowed(
            1, [1], True, float("nan"), 1.0e-4, 2.0e-4))
        self.assertFalse(kinematic_forecast_allowed(
            1, [1], True, 1.0e-3, 1.0e-4, float("inf")))

    def test_three_valid_active_frames_admit_then_delayed_clear(self):
        tracker = EvolutionTracker(clear_confirmations=3,
                                   retained_birth_confirmations=3,
                                   minimum_change_m=0.0001,
                                   sigma_factor=0.0)
        first = tracker.update("gdu", 1, 1.0, True, False, True,
                               0.002, 0.0001, (0.002, 0.0, 0.0))
        self.assertFalse(first["retained"])
        self.assertEqual(first["retained_birth_streak"], 1)
        self.assertEqual(first["lifecycle"], LIFECYCLE_PENDING_CONFIRMATION)
        second = tracker.update("gdu", 1, 2.0, True, False, True,
                                0.003, 0.0001, (0.003, 0.0, 0.0))
        self.assertFalse(second["retained"])
        self.assertEqual(second["retained_birth_streak"], 2)
        self.assertEqual(second["lifecycle"], LIFECYCLE_PENDING_CONFIRMATION)
        self.assertAlmostEqual(second["observation_interval"], 1.0)
        self.assertAlmostEqual(second["displacement_rate"], 0.001)
        third = tracker.update("gdu", 1, 3.0, True, False, True,
                               0.004, 0.0001, (0.004, 0.0, 0.0))
        self.assertTrue(third["retained"])
        self.assertEqual(third["retained_birth_streak"], 3)
        self.assertEqual(third["lifecycle"], LIFECYCLE_NEWLY_DETECTED)
        persistent = tracker.update("gdu", 1, 4.0, True, False, True,
                                    0.005, 0.0001, (0.005, 0.0, 0.0))
        self.assertEqual(persistent["lifecycle"], LIFECYCLE_PERSISTENT)
        for index in range(2):
            state = tracker.update("gdu", 1, 5.0 + index, False, True, True,
                                   0.0, 0.0001, (0.0, 0.0, 0.0))
            self.assertTrue(state["retained"])
            self.assertEqual(state["lifecycle"], LIFECYCLE_CLEAR_CANDIDATE)
        state = tracker.update("gdu", 1, 7.0, False, True, True,
                               0.0, 0.0001, (0.0, 0.0, 0.0))
        self.assertFalse(state["retained"])
        self.assertEqual(state["retained_birth_streak"], 0)
        self.assertEqual(state["lifecycle"], LIFECYCLE_INACTIVE)

    def test_pending_birth_pauses_across_unobservable_and_missing_frames(self):
        tracker = EvolutionTracker(
            retained_birth_confirmations=3,
            pending_confirmation_timeout_s=15.0)

        def update(stamp, active, valid=True):
            return tracker.update(
                "gdu", 1, stamp, active, False, valid,
                0.002, 0.0001, (0.002, 0.0, 0.0))

        first = update(1.0, True)
        self.assertEqual(first["retained_birth_streak"], 1)
        self.assertEqual(first["pending_confirmation_started_stamp_s"], 1.0)

        # A rank-zero/CTMC-only frame has no fresh measurement.  It pauses the
        # pending count and explicitly reports inconclusive information.
        unobservable = update(2.0, False, False)
        self.assertEqual(unobservable["retained_birth_streak"], 1)
        self.assertEqual(unobservable["lifecycle"], LIFECYCLE_INCONCLUSIVE)
        self.assertFalse(unobservable["pending_confirmation_timed_out"])

        second = update(3.0, True)
        self.assertEqual(second["retained_birth_streak"], 2)
        self.assertEqual(second["pending_confirmation_started_stamp_s"], 1.0)
        self.assertIn("gdu", tracker.tracked_ids())

        missing = tracker.mark_missing("gdu", 1, 4.0)
        self.assertIsNotNone(missing)
        self.assertEqual(missing["retained_birth_streak"], 2)
        self.assertEqual(missing["lifecycle"], LIFECYCLE_INCONCLUSIVE)

        confirmed = update(5.0, True)
        self.assertTrue(confirmed["retained"])
        self.assertEqual(confirmed["lifecycle"], LIFECYCLE_NEWLY_DETECTED)

    def test_fresh_nonactive_measurement_resets_pending_birth(self):
        tracker = EvolutionTracker(retained_birth_confirmations=3)
        tracker.update("gdu", 1, 1.0, True, False, True,
                       0.002, 0.0001, (0.002, 0.0, 0.0))
        reset = tracker.update("gdu", 1, 2.0, False, False, True,
                               0.002, 0.0001, (0.002, 0.0, 0.0))
        self.assertEqual(reset["retained_birth_streak"], 0)
        self.assertIsNone(reset["pending_confirmation_started_stamp_s"])
        self.assertEqual(reset["lifecycle"], LIFECYCLE_INACTIVE)
        restarted = tracker.update("gdu", 1, 3.0, True, False, True,
                                   0.002, 0.0001, (0.002, 0.0, 0.0))
        self.assertEqual(restarted["retained_birth_streak"], 1)

    def test_pending_confirmation_timeout_is_bounded_and_restarts(self):
        tracker = EvolutionTracker(
            retained_birth_confirmations=3,
            pending_confirmation_timeout_s=5.0)
        tracker.update("gdu", 1, 1.0, True, False, True,
                       0.002, 0.0001, (0.002, 0.0, 0.0))
        boundary = tracker.update("gdu", 1, 6.0, False, False, False,
                                  0.002, 0.0001, (0.002, 0.0, 0.0))
        self.assertEqual(boundary["retained_birth_streak"], 1)
        self.assertEqual(boundary["lifecycle"], LIFECYCLE_INCONCLUSIVE)

        expired = tracker.update("gdu", 1, 6.001, False, False, False,
                                 0.002, 0.0001, (0.002, 0.0, 0.0))
        self.assertTrue(expired["pending_confirmation_timed_out"])
        self.assertEqual(expired["retained_birth_streak"], 0)
        self.assertEqual(expired["lifecycle"], LIFECYCLE_INACTIVE)
        restarted = tracker.update("gdu", 1, 7.0, True, False, True,
                                   0.002, 0.0001, (0.002, 0.0, 0.0))
        self.assertEqual(restarted["retained_birth_streak"], 1)
        self.assertEqual(
            restarted["pending_confirmation_started_stamp_s"], 7.0)

    def test_active_measurement_after_timeout_restarts_at_one(self):
        tracker = EvolutionTracker(
            retained_birth_confirmations=3,
            pending_confirmation_timeout_s=5.0)
        tracker.update("gdu", 1, 1.0, True, False, True,
                       0.002, 0.0001, (0.002, 0.0, 0.0))
        restarted = tracker.update("gdu", 1, 7.0, True, False, True,
                                   0.003, 0.0001, (0.003, 0.0, 0.0))
        self.assertTrue(restarted["pending_confirmation_timed_out"])
        self.assertFalse(restarted["retained"])
        self.assertEqual(restarted["retained_birth_streak"], 1)
        self.assertEqual(
            restarted["pending_confirmation_started_stamp_s"], 7.0)

    def test_confirmation_counts_must_be_positive(self):
        with self.assertRaises(ValueError):
            EvolutionTracker(retained_birth_confirmations=0)
        with self.assertRaises(ValueError):
            EvolutionTracker(clear_confirmations=0)
        for invalid_timeout in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                EvolutionTracker(
                    pending_confirmation_timeout_s=invalid_timeout)

    def test_reference_epoch_resets_identity_history(self):
        tracker = EvolutionTracker(retained_birth_confirmations=1)
        tracker.update("gdu", 1, 1.0, True, False, True,
                       0.002, 0.0001, (0.002, 0.0, 0.0))
        state = tracker.update("gdu", 2, 2.0, True, False, True,
                               0.002, 0.0001, (0.002, 0.0, 0.0))
        self.assertEqual(state["lifecycle"], LIFECYCLE_NEWLY_DETECTED)
        self.assertTrue(math.isnan(state["previous_magnitude"]))

    def test_missing_observation_retains_geometry_cause_until_explicit_clear(self):
        tracker = EvolutionTracker(
            clear_confirmations=2, retained_birth_confirmations=1)
        tracker.update("gdu", 1, 1.0, True, False, True,
                       0.003, 0.0001, (0.003, 0.0, 0.0))
        self.assertEqual(tracker.retained_ids(), ["gdu"])
        missing = tracker.mark_missing("gdu", 1, 2.0)
        self.assertIsNotNone(missing)
        self.assertTrue(missing["retained"])
        self.assertEqual(missing["lifecycle"], LIFECYCLE_INCONCLUSIVE)
        self.assertEqual(missing["clear_streak"], 0)
        # Missing data breaks the valid temporal-difference chain.
        observed = tracker.update("gdu", 1, 3.0, False, False, True,
                                  0.003, 0.0001, (0.003, 0.0, 0.0))
        self.assertTrue(math.isnan(observed["previous_magnitude"]))
        self.assertTrue(observed["retained"])
        self.assertIsNone(tracker.mark_missing("unknown", 1, 4.0))
        self.assertIsNone(tracker.mark_missing("gdu", 2, 5.0))
        self.assertEqual(tracker.retained_ids(), [])


if __name__ == "__main__":
    unittest.main()
