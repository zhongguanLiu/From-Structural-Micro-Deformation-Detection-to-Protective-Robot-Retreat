#!/usr/bin/env python3

import pathlib
import unittest

import rosparam


CONFIG_ROOT = pathlib.Path(__file__).resolve().parents[1] / "config" / "v1"
CONFIG_FILES = (
    "algorithm_v1.yaml",
    "resource_v1.yaml",
    "calibration_uncalibrated.yaml",
)


def load_rosparam_source(name):
    loaded = rosparam.load_file(str(CONFIG_ROOT / name))
    if len(loaded) != 1:
        raise AssertionError("each source must produce exactly one parameter tree")
    tree, namespace = loaded[0]
    if namespace not in (None, "/"):
        raise AssertionError("unexpected source namespace: {}".format(namespace))
    if list(tree.keys()) != ["v1"] or not isinstance(tree["v1"], dict):
        raise AssertionError("source must contain exactly one mapping root named v1")
    return tree["v1"]


def leaf_paths(value, prefix=""):
    if not isinstance(value, dict):
        return {prefix}
    output = set()
    for key, child in value.items():
        child_prefix = key if not prefix else prefix + "/" + key
        child_leaves = leaf_paths(child, child_prefix)
        if output.intersection(child_leaves):
            raise AssertionError("duplicate leaf inside one source")
        output.update(child_leaves)
    return output


class V1ProductionYamlTest(unittest.TestCase):
    def test_real_overrides_preserve_task_scale_and_known_parameter_types(self):
        overrides = load_rosparam_source("real_overrides.yaml")
        known = set().union(*(leaf_paths(source) for source in self.sources))
        self.assertTrue(leaf_paths(overrides) <= known)
        self.assertNotIn("validated", overrides["calibration"])
        self.assertNotIn("extrinsic", overrides["resource"])
        self.assertEqual(overrides["resource"]["reference_min_frames"], 10)
        self.assertEqual(overrides["resource"]["reference_min_surfel_exposures"], 6)
        self.assertTrue(overrides["resource"]["bounded_reference_search"])
        self.assertFalse(self.resource["bounded_reference_search"])
        self.assertEqual(overrides["resource"]["reference_min_duration_s"], 2.0)
        for name in ("task_displacement_scale_m", "task_interface_jump_scale_m",
                     "task_velocity_scale_mps"):
            self.assertEqual(overrides["calibration"][name], self.calibration[name])
        self.assertAlmostEqual(
            overrides["calibration"]["reference_stability_sigma"] *
            overrides["calibration"]["independent_point_sigma_m"],
            self.calibration["reference_stability_sigma"] *
            self.calibration["independent_point_sigma_m"])

    @classmethod
    def setUpClass(cls):
        cls.sources = [load_rosparam_source(name) for name in CONFIG_FILES]
        cls.algorithm = cls.sources[0]["algorithm"]
        cls.resource = cls.sources[1]["resource"]
        cls.calibration = cls.sources[2]["calibration"]

    def test_rosparam_types_match_loader_contract(self):
        self.assertIs(type(self.algorithm["schema_version"]), int)
        self.assertIs(type(self.calibration["pose_control_max_condition_number"]),
                      float)
        self.assertIs(type(self.calibration["pose_control_huber_scale_m"]), float)
        self.assertIs(type(self.calibration["validated"]), bool)
        self.assertEqual(self.calibration["pose_control_max_condition_number"],
                         100000000.0)

    def test_three_detector_sources_have_one_disjoint_closed_leaf_set(self):
        all_leaves = set()
        for source in self.sources:
            leaves = leaf_paths(source)
            self.assertFalse(all_leaves.intersection(leaves))
            all_leaves.update(leaves)
        # Includes the digested, real-profile bounded-reference search switch.
        self.assertEqual(len(all_leaves), 163)

    def test_frozen_algorithm_and_resource_profile(self):
        self.assertEqual(self.algorithm["schema_version"], 3)
        self.assertEqual(self.algorithm["support_partition_percent"],
                         [20, 20, 40, 20])
        self.assertNotIn("hazard_aggregation", self.algorithm)
        self.assertNotIn("policy", self.sources[1])
        self.assertNotIn("robot_geometry_revision", self.resource)
        for obsolete in (
                "nb_dispersion", "nb_clutter_mean", "nb_occupied_mean_scale",
                "new_occupancy_prior", "novel_occupancy_birth_rate_hz",
                "novel_occupancy_death_rate_hz",
                "reference_persistence_birth_rate_hz",
                "reference_persistence_death_rate_hz",
                "unknown_occupancy_birth_rate_hz",
                "unknown_occupancy_death_rate_hz",
                "visibility_exposure_scale"):
            self.assertNotIn(obsolete, self.calibration)
        self.assertEqual(self.resource["prediction_horizons_s"],
                         [0.0, 0.5, 1.0, 2.0])
        self.assertEqual(self.resource["pose_control_exclusion_radius_m"],
                         5.0)
        self.assertEqual(
            (self.resource["max_gdu"],
             self.resource["min_controls_per_gdu"],
             self.resource["max_controls_per_gdu"],
             self.resource["max_interfaces_per_gdu"],
             self.resource["association_max_candidates"],
             self.resource["mixture_max_components"]),
            (64, 6, 16, 1, 3, 24))


if __name__ == "__main__":
    unittest.main()
