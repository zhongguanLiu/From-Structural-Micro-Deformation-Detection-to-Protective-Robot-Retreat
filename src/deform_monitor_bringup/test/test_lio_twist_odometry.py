#!/usr/bin/env python3

import importlib.util
import math
import pathlib
import unittest

from geometry_msgs.msg import Pose
import tf.transformations as transformations


SCRIPT = (pathlib.Path(__file__).resolve().parents[1] / "scripts"
          / "lio_twist_odometry.py")
SPEC = importlib.util.spec_from_file_location("lio_twist_odometry", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def pose(x, y, yaw):
    message = Pose()
    message.position.x = x
    message.position.y = y
    quaternion = transformations.quaternion_from_euler(0.0, 0.0, yaw)
    message.orientation.x = quaternion[0]
    message.orientation.y = quaternion[1]
    message.orientation.z = quaternion[2]
    message.orientation.w = quaternion[3]
    return message


class LioTwistTest(unittest.TestCase):
    def test_navigation_base_translation_is_composed_in_body_frame(self):
        result = MODULE.compose_pose_with_translation(
            pose(2.0, 3.0, math.pi / 2.0), (-0.28, 0.0, -0.60))
        self.assertAlmostEqual(result.position.x, 2.0, places=9)
        self.assertAlmostEqual(result.position.y, 2.72, places=9)
        self.assertAlmostEqual(result.position.z, -0.60, places=9)

    def test_world_motion_is_rotated_into_body_frame(self):
        result = MODULE.estimate_body_twist(
            pose(0.0, 0.0, math.pi / 2.0),
            pose(0.0, 1.2, math.pi / 2.0), 1.0)
        self.assertAlmostEqual(result[0], 1.2, places=6)
        self.assertAlmostEqual(result[1], 0.0, places=6)
        self.assertAlmostEqual(result[3], 0.0, places=6)

    def test_yaw_difference_wraps_at_pi(self):
        result = MODULE.estimate_body_twist(
            pose(0.0, 0.0, math.radians(179.0)),
            pose(0.0, 0.0, math.radians(-179.0)), 0.5)
        self.assertAlmostEqual(result[3], math.radians(4.0), places=6)

    def test_nonpositive_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.estimate_body_twist(pose(0, 0, 0), pose(0, 0, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
