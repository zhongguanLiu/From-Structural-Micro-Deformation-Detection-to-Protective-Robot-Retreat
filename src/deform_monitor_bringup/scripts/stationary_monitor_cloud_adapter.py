#!/usr/bin/env python3
"""Freeze the LiDAR world transform while a stopped robot monitors deformation.

FAST-LIO remains the mapping/localization source.  Before monitoring this node
passes its registered cloud through unchanged.  During the mission's MONITORING
state it transforms the current body-frame scan with the last pre-motion
FAST-LIO pose, preventing target points from moving the monitoring coordinate
frame.  Every finite merged return is eligible; no target, truth or manually
selected spatial region is consulted.  The two Gazebo ray sensors enforce their
own physical ranges before their returns are transformed into the shared
main-LiDAR/IMU frame, so the formal adapter does not apply a second radial cut
about that shared origin.  The contract is intentionally limited to a
commanded-stationary robot.
"""

import copy
import json
import threading

import numpy as np
import rospy
import tf.transformations as transformations
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String


PRE_MONITOR_STATES = {
    "", "WAITING_FOR_INPUTS", "WAITING_FOR_OPERATOR", "OPERATOR_STARTED",
    "NAVIGATING_OUTBOUND", "STABILIZING_AT_TARGET",
}
FROZEN_MONITOR_STATES = {
    "BUILDING_REFERENCE", "MONITORING", "MONITORING_COMPLETE",
}
POST_MONITOR_STATES = {
    "DEFORMATION_DETECTED", "MONITORING_COMPLETE",
    "RETURNING_HOME", "COMPLETED", "FAILED",
}


def filter_cloud_outside_box(message, minimum_xyz, maximum_xyz):
    """Remove returns inside one calibrated body-frame self volume.

    The input coordinates are expressed in the FAST-LIO ``body`` frame.  The
    operation preserves the complete binary point record and therefore does
    not alter intensity, ring or timing fields.  This product is for mapping
    and navigation only; the registered detector stream remains untouched.
    """
    minimum = np.asarray(minimum_xyz, dtype=np.float64)
    maximum = np.asarray(maximum_xyz, dtype=np.float64)
    if (minimum.shape != (3,) or maximum.shape != (3,)
            or not np.all(np.isfinite(minimum))
            or not np.all(np.isfinite(maximum))
            or np.any(minimum >= maximum)):
        raise ValueError("self-filter bounds must be finite ordered xyz triples")
    if message.is_bigendian:
        raise ValueError("big-endian PointCloud2 is unsupported")
    fields = {field.name: field for field in message.fields}
    for name in ("x", "y", "z"):
        field = fields.get(name)
        if field is None or field.datatype != PointField.FLOAT32 or field.count != 1:
            raise ValueError("PointCloud2 requires scalar FLOAT32 x/y/z fields")
    count = int(message.width) * int(message.height)
    point_step = int(message.point_step)
    expected = count * point_step
    if count <= 0 or point_step <= 0 or len(message.data) < expected:
        raise ValueError("invalid or empty PointCloud2 storage")
    storage = memoryview(message.data)[:expected]
    coordinates = np.column_stack([
        np.ndarray(shape=(count,), dtype=np.dtype("<f4"), buffer=storage,
                   offset=int(fields[name].offset), strides=(point_step,))
        for name in ("x", "y", "z")
    ]).astype(np.float64, copy=False)
    finite = np.all(np.isfinite(coordinates), axis=1)
    inside = finite & np.all((coordinates >= minimum)
                             & (coordinates <= maximum), axis=1)
    retained = finite & ~inside
    records = np.frombuffer(storage, dtype=np.uint8,
                            count=count * point_step).reshape((count, point_step))
    compact = records[retained]
    if compact.shape[0] == 0:
        raise ValueError("self filter removed every finite point")
    output = copy.deepcopy(message)
    output.height = 1
    output.width = int(compact.shape[0])
    output.row_step = output.width * point_step
    output.is_dense = True
    output.data = compact.tobytes()
    return output


def select_world_voxel_representatives(points, source_indices, voxel_size_m):
    """Return the exact tuple-sort representative of every occupied voxel.

    The historical implementation materialized and Python-sorted one tuple
    ``(kx, ky, kz, d2, x, y, z, source_index)`` per point.  ``numpy.lexsort``
    applies the same lexicographic keys in compiled code.  Keeping the source
    index as the final tie breaker preserves determinism even for duplicate
    returns; no centroid, averaging, random choice or target-dependent crop is
    introduced.
    """
    coordinates = np.asarray(points, dtype=np.float64)
    indices = np.asarray(source_indices, dtype=np.int64)
    size = float(voxel_size_m)
    if (coordinates.ndim != 2 or coordinates.shape[1] != 3
            or indices.ndim != 1 or indices.shape[0] != coordinates.shape[0]
            or coordinates.shape[0] == 0
            or not np.all(np.isfinite(coordinates))
            or not np.isfinite(size) or size <= 0.0):
        raise ValueError("voxel inputs must be finite, non-empty and aligned")
    keys = np.floor(coordinates / size).astype(np.int64)
    centers = (keys.astype(np.float64) + 0.5) * size
    delta = coordinates - centers
    distance_squared = np.einsum("ij,ij->i", delta, delta)
    order = np.lexsort((
        indices,
        coordinates[:, 2], coordinates[:, 1], coordinates[:, 0],
        distance_squared,
        keys[:, 2], keys[:, 1], keys[:, 0],
    ))
    ordered_keys = keys[order]
    first = np.empty(order.shape[0], dtype=bool)
    first[0] = True
    first[1:] = np.any(ordered_keys[1:] != ordered_keys[:-1], axis=1)
    return indices[order[first]]


def transform_cloud_xyz(message, matrix, output_frame, voxel_size_m=0.0,
                        apply_radial_range_filter=False, min_range_m=0.5,
                        max_range_m=20.0):
    """Transform all finite merged returns and optionally voxelize them.

    Every retained point keeps its complete binary record, including intensity
    and any extra fields.  Voxel keys are fixed in the output world frame.  A
    single-origin radial cut is available only as an explicitly enabled helper;
    it is disabled for the formal dual-LiDAR stream because its points originate
    from two sensors before being merged into one coordinate frame.  The result
    is always compacted to a dense, one-row cloud because the detector rejects
    non-finite placeholder points.
    """
    if not isinstance(apply_radial_range_filter, (bool, np.bool_)):
        raise ValueError("apply_radial_range_filter must be boolean")
    minimum_range = float(min_range_m)
    maximum_range = float(max_range_m)
    if apply_radial_range_filter:
        if (not np.isfinite(minimum_range)
                or not np.isfinite(maximum_range)
                or minimum_range < 0.0
                or minimum_range >= maximum_range):
            raise ValueError("monitor radial range must be finite and ordered")
    if not np.isfinite(float(voxel_size_m)) or float(voxel_size_m) < 0.0:
        raise ValueError("monitor voxel size must be finite and non-negative")
    if message.is_bigendian:
        raise ValueError("big-endian PointCloud2 is unsupported")
    fields = {field.name: field for field in message.fields}
    for name in ("x", "y", "z"):
        field = fields.get(name)
        if field is None or field.datatype != PointField.FLOAT32 or field.count != 1:
            raise ValueError("PointCloud2 requires scalar FLOAT32 x/y/z fields")
    count = int(message.width) * int(message.height)
    expected = count * int(message.point_step)
    if count <= 0 or message.point_step <= 0 or len(message.data) < expected:
        raise ValueError("invalid or empty PointCloud2 storage")
    storage = bytearray(message.data)
    coordinates = []
    for name in ("x", "y", "z"):
        coordinates.append(np.ndarray(
            shape=(count,), dtype=np.dtype("<f4"), buffer=storage,
            offset=int(fields[name].offset), strides=(int(message.point_step),)))
    xyz = np.column_stack(coordinates).astype(np.float64, copy=False)
    finite = np.all(np.isfinite(xyz), axis=1)
    retained = finite.copy()
    if apply_radial_range_filter:
        squared_range = np.zeros(count, dtype=np.float64)
        squared_range[finite] = np.einsum(
            "ij,ij->i", xyz[finite], xyz[finite])
        retained &= ((squared_range >= minimum_range * minimum_range)
                     & (squared_range <= maximum_range * maximum_range))
    if np.any(retained):
        rotated = xyz[retained].dot(np.asarray(matrix[:3, :3]).T)
        rotated += np.asarray(matrix[:3, 3])
        for axis in range(3):
            coordinates[axis][retained] = rotated[:, axis].astype(np.float32)
    if float(voxel_size_m) > 0.0 and np.any(retained):
        transformed = np.column_stack(coordinates).astype(np.float64, copy=False)
        finite_indices = np.flatnonzero(retained)
        # World-fixed voxel key is primary.  The actual measured point nearest
        # its centre is retained; coordinates and source index only resolve
        # exact deterministic ties.  No synthetic centroid is created and no
        # unmodelled averaging precision is claimed.
        selected = select_world_voxel_representatives(
            transformed[finite_indices], finite_indices, voxel_size_m)
        retained[:] = False
        retained[selected] = True
    records = np.frombuffer(storage, dtype=np.uint8,
                            count=count * int(message.point_step))
    records = records.reshape((count, int(message.point_step)))
    compact = records[retained]
    if compact.shape[0] == 0:
        raise ValueError("monitor input contains no retained finite points")
    output = copy.deepcopy(message)
    output.header.frame_id = output_frame
    output.height = 1
    output.width = int(compact.shape[0])
    output.row_step = output.width * int(message.point_step)
    output.is_dense = True
    output.data = compact.tobytes()
    return output


def odometry_matrix(message):
    pose = message.pose.pose
    quaternion = pose.orientation
    matrix = transformations.quaternion_matrix((
        quaternion.x, quaternion.y, quaternion.z, quaternion.w))
    matrix[0, 3] = pose.position.x
    matrix[1, 3] = pose.position.y
    matrix[2, 3] = pose.position.z
    return matrix


class StationaryMonitorCloudAdapter:
    def __init__(self):
        self.lock = threading.RLock()
        self.latest_odometry = None
        self.frozen_matrix = None
        self.monitor_voxel_size_m = float(rospy.get_param(
            "~monitor_voxel_size_m", 0.10))
        if (not np.isfinite(self.monitor_voxel_size_m)
                or self.monitor_voxel_size_m <= 0.0):
            raise ValueError("monitor_voxel_size_m must be finite and positive")
        self.monitor_min_range_m = float(rospy.get_param(
            "~monitor_min_range_m", 0.5))
        self.monitor_max_range_m = float(rospy.get_param(
            "~monitor_max_range_m", 20.0))
        self.monitor_apply_radial_range_filter = rospy.get_param(
            "~monitor_apply_radial_range_filter", False)
        if not isinstance(self.monitor_apply_radial_range_filter, bool):
            raise ValueError("monitor_apply_radial_range_filter must be boolean")
        if self.monitor_apply_radial_range_filter:
            if (not np.isfinite(self.monitor_min_range_m)
                    or not np.isfinite(self.monitor_max_range_m)
                    or self.monitor_min_range_m < 0.0
                    or self.monitor_min_range_m >= self.monitor_max_range_m):
                raise ValueError("monitor radial range must be finite and ordered")
        self.mission_state = ""
        self.world_frame = rospy.get_param("~world_frame", "camera_init")
        self.publisher = rospy.Publisher(
            rospy.get_param("~output_topic", "/monitor/cloud_registered"),
            PointCloud2, queue_size=3)
        self.navigation_publisher = rospy.Publisher(
            rospy.get_param("~navigation_output_topic",
                            "/navigation/cloud_registered_body"),
            PointCloud2, queue_size=3)
        self.self_filter_min = tuple(float(value) for value in rospy.get_param(
            "~self_filter_min_xyz", [-0.70, -0.38, -0.35]))
        self.self_filter_max = tuple(float(value) for value in rospy.get_param(
            "~self_filter_max_xyz", [0.12, 0.38, 0.35]))
        # Validate at startup instead of failing only after navigation begins.
        minimum = np.asarray(self.self_filter_min, dtype=np.float64)
        maximum = np.asarray(self.self_filter_max, dtype=np.float64)
        if (minimum.shape != (3,) or maximum.shape != (3,)
                or not np.all(np.isfinite(minimum))
                or not np.all(np.isfinite(maximum))
                or np.any(minimum >= maximum)):
            raise ValueError("invalid calibrated body self-filter volume")
        rospy.Subscriber(rospy.get_param("~registered_topic", "/cloud_registered"),
                         PointCloud2, self._registered_callback, queue_size=3)
        rospy.Subscriber(rospy.get_param("~body_topic", "/cloud_registered_body"),
                         PointCloud2, self._body_callback, queue_size=3)
        rospy.Subscriber(rospy.get_param("~odometry_topic", "/Odometry"),
                         Odometry, self._odometry_callback, queue_size=5)
        rospy.Subscriber(rospy.get_param("~mission_status_topic",
                                         "/deform/mission/status"),
                         String, self._mission_callback, queue_size=3)

    def _odometry_callback(self, message):
        with self.lock:
            self.latest_odometry = copy.deepcopy(message)

    def _mission_callback(self, message):
        try:
            document = json.loads(message.data)
            state = str(document.get("state", ""))
        except (TypeError, ValueError):
            rospy.logerr_throttle(2.0, "invalid closed-loop mission status JSON")
            return
        with self.lock:
            if (state == "BUILDING_REFERENCE"
                    and self.mission_state != "BUILDING_REFERENCE"):
                if self.latest_odometry is None:
                    rospy.logerr("cannot freeze monitoring pose without FAST-LIO odometry")
                    self.frozen_matrix = None
                else:
                    self.frozen_matrix = odometry_matrix(self.latest_odometry)
                    rospy.loginfo("[monitor_cloud] froze stationary FAST-LIO pose")
            self.mission_state = state

    def _registered_callback(self, message):
        with self.lock:
            state = self.mission_state
        if state in PRE_MONITOR_STATES:
            self.publisher.publish(message)

    def _body_callback(self, message):
        with self.lock:
            state = self.mission_state
            matrix = None if self.frozen_matrix is None else self.frozen_matrix.copy()
        try:
            navigation_cloud = filter_cloud_outside_box(
                message, self.self_filter_min, self.self_filter_max)
            self.navigation_publisher.publish(navigation_cloud)
        except ValueError as error:
            rospy.logerr_throttle(2.0, "navigation self filter rejected cloud: %s",
                                  error)
        if state not in FROZEN_MONITOR_STATES or matrix is None:
            return
        try:
            output = transform_cloud_xyz(
                message, matrix, self.world_frame,
                apply_radial_range_filter=self.monitor_apply_radial_range_filter,
                min_range_m=self.monitor_min_range_m,
                max_range_m=self.monitor_max_range_m,
                voxel_size_m=self.monitor_voxel_size_m)
        except ValueError as error:
            rospy.logerr_throttle(2.0, "stationary monitoring cloud rejected: %s", error)
            return
        self.publisher.publish(output)


def main():
    rospy.init_node("stationary_monitor_cloud_adapter")
    StationaryMonitorCloudAdapter()
    rospy.spin()


if __name__ == "__main__":
    main()
