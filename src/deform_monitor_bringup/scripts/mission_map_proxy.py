#!/usr/bin/env python3
"""Publish one navigation-map topic from SLAM and deformation-aware inputs."""

import copy
import hashlib
import json
import os
import sys
import threading
import rospy
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String

_SCRIPT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
if _SCRIPT_DIRECTORY not in sys.path:
    sys.path.insert(0, _SCRIPT_DIRECTORY)
from planning_core import fuse_planning_layers
import numpy as np


def compatible(first, second):
    if first is None or second is None:
        return False
    return (first.header.frame_id == second.header.frame_id
            and first.info.width == second.info.width
            and first.info.height == second.info.height
            and abs(first.info.resolution - second.info.resolution) < 1.0e-12
            and abs(first.info.origin.position.x
                    - second.info.origin.position.x) < 1.0e-9
            and abs(first.info.origin.position.y
                    - second.info.origin.position.y) < 1.0e-9
            and len(first.data) == len(second.data))


def overlay_on_base(base, overlay):
    """Copy a retained overlay onto the same-resolution expanding SLAM lattice.

    Zero outside the old overlay means no retained deformation cost, not
    observed free space: fuse_planning_layers preserves SLAM UNKNOWN there.
    """
    if (base.header.frame_id != overlay.header.frame_id
            or base.info.resolution <= 0
            or abs(base.info.resolution - overlay.info.resolution) > 1.e-9):
        raise ValueError("retained overlay must share the live SLAM frame/resolution")
    for grid in (base, overlay):
        q = grid.info.origin.orientation
        if abs(q.x) + abs(q.y) + abs(q.z) > 1.e-9:
            raise ValueError("only axis-aligned SLAM grids are supported")
    offset = np.array([
        overlay.info.origin.position.x - base.info.origin.position.x,
        overlay.info.origin.position.y - base.info.origin.position.y,
    ]) / base.info.resolution
    rounded = np.rint(offset)
    if not np.all(np.isfinite(offset)) or np.max(np.abs(offset-rounded)) > 1.e-4:
        raise ValueError("retained overlay does not align with live SLAM lattice")
    dx, dy = rounded.astype(int)
    source = np.asarray(overlay.data).reshape(overlay.info.height, overlay.info.width)
    result = np.zeros((base.info.height, base.info.width), dtype=np.int16)
    x0, y0 = max(0, dx), max(0, dy)
    x1 = min(base.info.width, dx + overlay.info.width)
    y1 = min(base.info.height, dy + overlay.info.height)
    if x1 > x0 and y1 > y0:
        result[y0:y1, x0:x1] = source[y0-dy:y1-dy, x0-dx:x1-dx]
    return result


class MissionMapProxy:
    def __init__(self):
        self.lock = threading.RLock()
        self.base = None
        self.overlay = None
        self.freeze_requested = False
        self.snapshot_locked = False
        self.frozen_base = None
        self.frozen_overlay = None
        self.snapshot_directory = rospy.get_param("~snapshot_directory", "")
        self.base_publisher = rospy.Publisher(
            rospy.get_param("~normalized_base_topic", "/map"),
            OccupancyGrid, queue_size=1, latch=True)
        self.publisher = rospy.Publisher(
            rospy.get_param("~output_topic", "/mission/navigation_map"),
            OccupancyGrid, queue_size=1, latch=True)
        rospy.Subscriber(rospy.get_param("~base_topic", "/map"),
                         OccupancyGrid, self._base_callback, queue_size=1)
        rospy.Subscriber(rospy.get_param(
            "~overlay_topic", "/deform/avoidance/current_overlay"),
            OccupancyGrid, self._overlay_callback, queue_size=1)
        rospy.Subscriber(rospy.get_param(
            "~mission_status_topic", "/deform/mission/status"),
            String, self._mission_callback, queue_size=2)

    def _mission_callback(self, message):
        try:
            state = str(json.loads(message.data).get("state", ""))
        except (TypeError, ValueError):
            rospy.logwarn_throttle(2.0, "invalid mission status JSON")
            return
        with self.lock:
            if state in ("DEFORMATION_DETECTED", "MONITORING_COMPLETE"):
                # Freeze the scientific decision and retained deformation,
                # NOT the navigation geometry needed during robot motion.
                self.freeze_requested = True
                self._try_lock_snapshot()

    def _base_callback(self, message):
        with self.lock:
            self.base = copy.deepcopy(message)
            # octomap_server's axis-aligned origin may use a zero quaternion.
            self.base.info.origin.orientation.x = 0.0
            self.base.info.origin.orientation.y = 0.0
            self.base.info.origin.orientation.z = 0.0
            self.base.info.origin.orientation.w = 1.0
            self.base_publisher.publish(self.base)
            self._try_lock_snapshot()
            self._publish()

    def _overlay_callback(self, message):
        with self.lock:
            self.overlay = copy.deepcopy(message)
            self._try_lock_snapshot()
            self._publish()

    def _try_lock_snapshot(self):
        if (not self.freeze_requested or self.snapshot_locked
                or not compatible(self.base, self.overlay)):
            return False
        self.frozen_base = copy.deepcopy(self.base)
        self.frozen_overlay = copy.deepcopy(self.overlay)
        if self.snapshot_directory:
            # Persist the owner's exact layers, not asynchronous recorder copies.
            shape = (self.base.info.height, self.base.info.width)
            base = np.asarray(self.frozen_base.data, dtype=np.int16).reshape(shape)
            overlay = np.asarray(self.frozen_overlay.data, dtype=np.int16).reshape(shape)
            fused = fuse_planning_layers(base, overlay)
            metadata = dict(frame_id=self.base.header.frame_id,
                            resolution=self.base.info.resolution,
                            origin=[self.base.info.origin.position.x,
                                    self.base.info.origin.position.y],
                            base_stamp_s=self.base.header.stamp.to_sec(),
                            overlay_stamp_s=self.overlay.header.stamp.to_sec(),
                            freeze_time_s=rospy.Time.now().to_sec(),
                            policy="FINITE_SOFT_DEFORMATION", frozen=True)
            digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()
                                    + base.tobytes() + overlay.tobytes()).hexdigest()
            metadata["snapshot_id"] = digest
            os.makedirs(self.snapshot_directory, exist_ok=True)
            path = os.path.join(self.snapshot_directory, "decision_map.npz")
            with open(path + ".tmp", "wb") as stream:
                np.savez_compressed(stream, base=base, overlay=overlay, fused=fused,
                                    metadata=json.dumps(metadata, sort_keys=True))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(path + ".tmp", path)
        self.snapshot_locked = True
        rospy.loginfo(
            "[mission_map_proxy] locked deformation-decision navigation "
            "snapshot at %.6f",
            max(self.frozen_base.header.stamp.to_sec(),
                self.frozen_overlay.header.stamp.to_sec()))
        self._publish()
        return True

    def _publish(self):
        selected_base = self.base
        selected_overlay = (self.frozen_overlay
                            if self.snapshot_locked else self.overlay)
        if selected_base is None:
            return
        output = copy.deepcopy(selected_base)
        if selected_overlay is not None:
            shape = (selected_base.info.height, selected_base.info.width)
            # Never carve an escape hole. The whole overlay remains finite,
            # including under the initial footprint; geometry stays lethal.
            try:
                output.data = fuse_planning_layers(
                    np.asarray(selected_base.data).reshape(shape),
                    overlay_on_base(selected_base, selected_overlay)).ravel().tolist()
            except ValueError as error:
                rospy.logerr_throttle(2.0, "invalid navigation layers: %s", error)
                return
        self.publisher.publish(output)


def main():
    rospy.init_node("mission_map_proxy")
    MissionMapProxy()
    rospy.spin()


if __name__ == "__main__":
    main()
