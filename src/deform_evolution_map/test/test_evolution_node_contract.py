#!/usr/bin/env python3
"""Static wiring contracts for the ROS adapter around the pure tracker."""

import ast
import os
import unittest


SCRIPT = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "scripts",
    "deformation_evolution_node.py"))
with open(SCRIPT, "r", encoding="utf-8") as source_file:
    SOURCE = source_file.read()
TREE = ast.parse(SOURCE, filename=SCRIPT)


class EvolutionNodeContractTest(unittest.TestCase):
    def test_rank_zero_physical_carrier_is_not_deleted_as_other(self):
        calls = [node for node in ast.walk(TREE) if isinstance(node, ast.Call)]
        self.assertTrue(any(
            isinstance(call.func, ast.Name)
            and call.func.id == "fresh_observable_measurement"
            for call in calls))
        self.assertFalse(any(
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "samples_by_gdu"
            and call.func.attr == "pop"
            for call in calls))
        self.assertNotIn("projector_traces", SOURCE)

    def test_temporary_unobservable_retained_gdu_carries_geometry(self):
        for field in (
                "reference_support", "current_support",
                "displacement_samples", "mean_displacement"):
            self.assertIn(
                "output.{0} = copy.deepcopy(\n"
                "                        previous.{0})".format(field),
                SOURCE)
        self.assertIn("output.forecasts = []", SOURCE)
        self.assertIn("output.velocity_uncertainty_defined = False", SOURCE)
        self.assertIn("TEMPORARILY_UNOBSERVABLE", SOURCE)

    def test_bounded_pending_timeout_parameter_reaches_tracker(self):
        self.assertIn(
            'prefix + "pending_confirmation_timeout_s", 15.0', SOURCE)
        self.assertIn(
            "pending_confirmation_timeout_s=float(rospy.get_param(", SOURCE)

    def test_online_adapter_contains_no_simulation_target_prior(self):
        lowered = SOURCE.lower()
        for forbidden in ("m01", "m02", "m03", "m04", "m05", "m06",
                          "laser_retro", "model_states", "gazebo"):
            self.assertNotIn(forbidden, lowered)


if __name__ == "__main__":
    unittest.main()
