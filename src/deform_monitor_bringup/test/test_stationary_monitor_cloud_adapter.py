#!/usr/bin/env python3

import importlib.util
import json
import pathlib
import struct
import threading
import unittest

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String


SCRIPT = (pathlib.Path(__file__).resolve().parents[1] / "scripts"
          / "stationary_monitor_cloud_adapter.py")
SPEC = importlib.util.spec_from_file_location("stationary_adapter", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class StationaryCloudTransformTest(unittest.TestCase):
    def test_vectorized_voxel_selection_matches_formal_tuple_order(self):
        random = np.random.RandomState(17)
        points = random.uniform(-1.5, 1.5, size=(500, 3))
        # Exercise duplicate coordinates and exact distance ties explicitly.
        points[10] = points[9]
        points[20] = [0.11, 0.11, 0.11]
        points[21] = [0.19, 0.19, 0.19]
        source = np.arange(700, 1200, dtype=np.int64)
        size = 0.20
        candidates = []
        for point, index in zip(points, source):
            key = np.floor(point / size).astype(np.int64)
            center = (key.astype(np.float64) + 0.5) * size
            distance_squared = float(np.dot(point - center, point - center))
            candidates.append((int(key[0]), int(key[1]), int(key[2]),
                               distance_squared, float(point[0]),
                               float(point[1]), float(point[2]), int(index)))
        candidates.sort()
        expected = []
        previous = None
        for candidate in candidates:
            if candidate[:3] == previous:
                continue
            previous = candidate[:3]
            expected.append(candidate[-1])
        actual = MODULE.select_world_voxel_representatives(
            points, source, size)
        self.assertEqual(actual.tolist(), expected)

    def test_body_self_filter_removes_only_calibrated_volume(self):
        message = PointCloud2()
        message.header.frame_id = "body"
        message.height = 1
        message.width = 3
        message.is_bigendian = False
        message.is_dense = True
        message.point_step = 16
        message.row_step = 48
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12,
                       datatype=PointField.FLOAT32, count=1),
        ]
        message.data = b"".join([
            # Calibrated A2 body return: removed.
            struct.pack("<ffff", -0.20, 0.10, -0.08, 11.0),
            # Nearby scene return outside the lateral body envelope: kept.
            struct.pack("<ffff", -0.20, 0.50, -0.08, 22.0),
            # Forward scene return outside the body envelope: kept.
            struct.pack("<ffff", 0.50, 0.00, 0.00, 33.0),
        ])
        output = MODULE.filter_cloud_outside_box(
            message, [-0.70, -0.38, -0.35], [0.12, 0.38, 0.35])
        values = struct.unpack("<ffffffff", output.data)
        self.assertEqual(output.header.frame_id, "body")
        self.assertEqual(output.width, 2)
        self.assertAlmostEqual(values[3], 22.0)
        self.assertAlmostEqual(values[7], 33.0)
        self.assertEqual(message.width, 3)

    def test_body_self_filter_rejects_inverted_volume(self):
        message = PointCloud2()
        with self.assertRaises(ValueError):
            MODULE.filter_cloud_outside_box(
                message, [1.0, 0.0, 0.0], [0.0, 1.0, 1.0])

    def test_transform_preserves_fields_and_applies_frozen_pose(self):
        message = PointCloud2()
        message.header.frame_id = "body"
        message.height = 1
        message.width = 2
        message.is_bigendian = False
        message.is_dense = True
        message.point_step = 16
        message.row_step = 32
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12,
                       datatype=PointField.FLOAT32, count=1),
        ]
        message.data = b"".join([
            struct.pack("<ffff", 1.0, 2.0, 3.0, 7.0),
            struct.pack("<ffff", -1.0, 0.0, 2.0, 9.0),
        ])
        matrix = np.eye(4)
        matrix[:3, 3] = [10.0, -2.0, 0.5]
        output = MODULE.transform_cloud_xyz(
            message, matrix, "camera_init", min_range_m=0.5,
            max_range_m=20.0)
        values = struct.unpack("<ffffffff", output.data)
        self.assertEqual(output.header.frame_id, "camera_init")
        self.assertEqual(values[:4], (11.0, 0.0, 3.5, 7.0))
        self.assertEqual(values[4:], (9.0, -2.0, 2.5, 9.0))
        self.assertEqual(message.header.frame_id, "body")

    def test_radial_range_compacts_records_and_preserves_extra_fields(self):
        message = PointCloud2()
        message.header.frame_id = "body"
        message.height = 1
        message.width = 3
        message.is_bigendian = False
        message.is_dense = True
        message.point_step = 16
        message.row_step = 48
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12,
                       datatype=PointField.FLOAT32, count=1),
        ]
        message.data = b"".join([
            struct.pack("<ffff", 0.1, 0.0, 0.0, 10.0),
            struct.pack("<ffff", 5.0, 5.0, 5.0, 20.0),
            struct.pack("<ffff", 21.0, 0.0, 0.0, 30.0),
        ])
        output = MODULE.transform_cloud_xyz(
            message, np.eye(4), "camera_init",
            apply_radial_range_filter=True,
            min_range_m=0.5, max_range_m=20.0)
        values = struct.unpack("<ffff", output.data)
        self.assertEqual(values, (5.0, 5.0, 5.0, 20.0))
        self.assertEqual(output.height, 1)
        self.assertEqual(output.width, 1)
        self.assertEqual(output.row_step, 16)
        self.assertEqual(output.point_step, message.point_step)

    def test_radial_range_rejects_inverted_bounds(self):
        message = PointCloud2()
        with self.assertRaises(ValueError):
            MODULE.transform_cloud_xyz(
                message, np.eye(4), "camera_init",
                apply_radial_range_filter=True,
                min_range_m=20.0, max_range_m=0.5)

    def test_formal_default_does_not_repeat_single_origin_range_cut(self):
        message = PointCloud2()
        message.header.frame_id = "body"
        message.height = 1
        message.width = 3
        message.is_bigendian = False
        message.is_dense = True
        message.point_step = 16
        message.row_step = 48
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12,
                       datatype=PointField.FLOAT32, count=1),
        ]
        message.data = b"".join([
            struct.pack("<ffff", 0.10, 0.00, 0.00, 10.0),
            struct.pack("<ffff", 5.00, 0.00, 0.00, 20.0),
            struct.pack("<ffff", 21.0, 0.00, 0.00, 30.0),
        ])
        output = MODULE.transform_cloud_xyz(
            message, np.eye(4), "camera_init", voxel_size_m=0.0)
        values = struct.unpack("<ffffffffffff", output.data)
        self.assertEqual(output.width, 3)
        self.assertAlmostEqual(values[0], 0.10)
        self.assertEqual(values[3], 10.0)
        self.assertEqual(values[7], 20.0)
        self.assertEqual(values[11], 30.0)

    def test_world_fixed_voxel_keeps_one_real_center_nearest_record(self):
        message = PointCloud2()
        message.header.frame_id = "body"
        message.height = 1
        message.width = 3
        message.is_bigendian = False
        message.is_dense = True
        message.point_step = 16
        message.row_step = 48
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12,
                       datatype=PointField.FLOAT32, count=1),
        ]
        message.data = b"".join([
            struct.pack("<ffff", 0.101, 0.101, 0.101, 10.0),
            struct.pack("<ffff", 0.149, 0.149, 0.149, 20.0),
            struct.pack("<ffff", 0.251, 0.251, 0.251, 30.0),
        ])
        output = MODULE.transform_cloud_xyz(
            message, np.eye(4), "camera_init",
            min_range_m=0.0, max_range_m=1.0,
            voxel_size_m=0.1)
        values = struct.unpack("<ffffffff", output.data)
        self.assertEqual(output.width, 2)
        self.assertAlmostEqual(values[0], 0.149, places=6)
        self.assertEqual(values[3], 20.0)
        self.assertAlmostEqual(values[4], 0.251, places=6)
        self.assertEqual(values[7], 30.0)

    def test_voxel_size_rejects_negative_values(self):
        message = PointCloud2()
        message.width = 0
        with self.assertRaises(ValueError):
            MODULE.transform_cloud_xyz(
                message, np.eye(4), "camera_init", voxel_size_m=-0.1)

    def test_mission_status_target_boxes_are_ignored(self):
        adapter = MODULE.StationaryMonitorCloudAdapter.__new__(
            MODULE.StationaryMonitorCloudAdapter)
        adapter.lock = threading.RLock()
        adapter.latest_odometry = None
        adapter.frozen_matrix = None
        adapter.mission_state = "MONITORING"
        message = String(data=json.dumps({
            "state": "MONITORING",
            "monitor_regions_map": "deliberately-invalid-target-prior",
            "stable_support_regions_map": [[6, 5, 4, 3, 2, 1]],
        }))
        adapter._mission_callback(message)
        self.assertEqual(adapter.mission_state, "MONITORING")
        self.assertFalse(hasattr(adapter, "include_boxes"))


if __name__ == "__main__":
    unittest.main()
