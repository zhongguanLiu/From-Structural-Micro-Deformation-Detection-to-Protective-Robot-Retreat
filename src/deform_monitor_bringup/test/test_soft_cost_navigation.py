#!/usr/bin/env python3
"""Actual proxy -> StaticLayer -> Navfn regression; no motion goals are sent."""
import copy
import time
import unittest

import numpy as np
import rospy
import rostest
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetPlan
from sensor_msgs.msg import PointCloud2
from sensor_msgs.point_cloud2 import create_cloud_xyz32
from std_msgs.msg import Header, String


class SoftCostNavigationTest(unittest.TestCase):
    def test_soft_egress_detour_and_geometric_barriers(self):
        latest = {}
        rospy.Subscriber("/mission/navigation_map", OccupancyGrid,
                         lambda msg: latest.update(fused=msg))
        rospy.Subscriber("/move_base/global_costmap/costmap", OccupancyGrid,
                         lambda msg: latest.update(costmap=msg))
        rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid,
                         lambda msg: latest.update(local=msg))
        base_pub = rospy.Publisher("/test/base", OccupancyGrid, queue_size=1, latch=True)
        overlay_pub = rospy.Publisher("/test/overlay", OccupancyGrid, queue_size=1, latch=True)
        template = OccupancyGrid()
        template.header.frame_id = "camera_init"
        template.info.resolution = .1
        template.info.width, template.info.height = 120, 110
        template.info.origin.orientation.w = 1.
        shape = (110, 120)
        free = np.zeros(shape, dtype=np.int16)

        def wait_until(predicate, label, timeout=12.):
            deadline = time.monotonic() + timeout
            while not rospy.is_shutdown() and time.monotonic() < deadline:
                if predicate():
                    return
                rospy.sleep(.05)
            self.fail("Timeout: " + label)

        def publish(base, overlay, check_cell, expect):
            previous = latest.get("costmap")
            base_msg, overlay_msg = copy.deepcopy(template), copy.deepcopy(template)
            base_msg.header.stamp = overlay_msg.header.stamp = rospy.Time.now()
            base_msg.data, overlay_msg.data = base.ravel().tolist(), overlay.ravel().tolist()
            base_pub.publish(base_msg)
            overlay_pub.publish(overlay_msg)
            expected = np.where(base < 0, -1, np.maximum(base, np.minimum(99, overlay)))
            index = check_cell[1] * 120 + check_cell[0]
            wait_until(lambda: "fused" in latest and
                       list(latest["fused"].data) == expected.ravel().tolist(), "proxy fusion")
            wait_until(lambda: "costmap" in latest and latest["costmap"] is not previous
                       and len(latest["costmap"].data) == free.size
                       and expect(latest["costmap"].data[index]), "StaticLayer cost conversion")

        def pose(column, row):
            message = PoseStamped()
            message.header.frame_id = "camera_init"
            message.pose.position.x, message.pose.position.y = (column+.5)*.1, (row+.5)*.1
            message.pose.orientation.w = 1.
            return message

        wait_until(lambda: base_pub.get_num_connections() and overlay_pub.get_num_connections(),
                   "proxy subscribers")
        publish(free, free, (50, 50), lambda value: value == 0)
        rospy.wait_for_service("/move_base/make_plan", timeout=15.)
        planner = rospy.ServiceProxy("/move_base/make_plan", GetPlan)

        def plan():
            return planner(pose(10, 50), pose(100, 50), 0.).plan.poses

        baseline = plan()
        self.assertGreater(len(baseline), 2)
        overlay = free.copy()
        overlay[:, :60] = 99  # Start is metres inside, not a one-cell boundary case.
        publish(free, overlay, (40, 50), lambda value: 0 < value < 100)
        def local_soft_at_start():
            grid = latest.get("local")
            if grid is None:
                return False
            column = int((1.05-grid.info.origin.position.x)/grid.info.resolution)
            row = int((5.05-grid.info.origin.position.y)/grid.info.resolution)
            return (0 <= column < grid.info.width and 0 <= row < grid.info.height
                    and 0 < grid.data[row*grid.info.width+column] < 100)
        wait_until(local_soft_at_start, "DWA local costmap preserves soft start cost")
        egress = plan()
        self.assertGreater(len(egress), 2)
        self.assertEqual(latest["fused"].data[50*120+10], 99)  # No carved start hole.

        overlay[:] = 99  # No zero-cost route or zero-cost goal exists.
        publish(free, overlay, (90, 50), lambda value: 0 < value < 100)
        self.assertGreater(len(plan()), 2)

        overlay[:] = 0
        overlay[35:66, 45:66] = 99
        publish(free, overlay, (90, 50), lambda value: value == 0)
        wait_until(lambda: 0 < latest["costmap"].data[50*120+50] < 100, "detour core")
        detour = plan()
        self.assertGreater(len(detour), 2)
        self.assertGreater(max(abs(p.pose.position.y-5.05) for p in detour), 1.)

        for barrier in (100, -1):
            base = free.copy()
            base[:, 60:65] = barrier
            publish(base, overlay, (62, 50), lambda value: value == barrier)
            self.assertEqual(len(plan()), 0, "Real obstacle/UNKNOWN barrier crossed")
        # Real LiDAR obstacle missing from the frozen base must reach BOTH
        # planners, otherwise Navfn repeatedly plans through a DWA-only wall.
        publish(free, free, (62, 50), lambda value: value == 0)
        cloud_pub = rospy.Publisher("/navigation/cloud_registered_body", PointCloud2,
                                    queue_size=1)
        # Both plugins share move_base's transport subscription to this topic.
        wait_until(lambda: cloud_pub.get_num_connections() >= 1, "live obstacle subscriber")
        def send_wall(_event=None):
            cloud_pub.publish(create_cloud_xyz32(
                Header(stamp=rospy.Time.now(), frame_id="camera_init"),
                [(3.05, y, .5) for y in np.arange(4.05, 6.06, .05)]))
        timer = rospy.Timer(rospy.Duration(.1), send_wall)
        def wall_in_both_maps():
            for key in ("costmap", "local"):
                grid = latest.get(key)
                if grid is None:
                    return False
                column = int((3.05-grid.info.origin.position.x)/grid.info.resolution)
                row = int((5.05-grid.info.origin.position.y)/grid.info.resolution)
                if grid.data[row*grid.info.width+column] != 100:
                    return False
            return True
        try:
            wait_until(wall_in_both_maps, "same live obstacle in global/local maps")
            live_detour = plan()
            self.assertGreater(len(live_detour), 2)
            self.assertGreater(max(abs(p.pose.position.y-5.05) for p in live_detour), 1.)
        finally:
            timer.shutdown()
        # Freeze the deformation decision; a later SLAM update must still
        # reach navigation, and an expanding grid must retain world locations.
        state_pub = rospy.Publisher("/deform/mission/status", String, queue_size=1, latch=True)
        overlay[:] = 0
        overlay[80, 90] = 99
        publish(free, overlay, (90, 80), lambda value: value > 0)
        state_pub.publish(String(data='{"state":"DEFORMATION_DETECTED"}'))
        rospy.sleep(.3)
        expanded = copy.deepcopy(template)
        expanded.header.stamp = rospy.Time.now()
        expanded.info.origin.position.x = -1.
        expanded.info.width = 130
        expanded_base = np.zeros((110, 130), dtype=np.int16)
        expanded_base[80, 95] = 100
        expanded.data = expanded_base.ravel().tolist()
        base_pub.publish(expanded)
        wait_until(lambda: latest.get("fused") is not None
                   and latest["fused"].info.width == 130
                   and latest["fused"].data[80*130+95] == 100
                   and latest["fused"].data[80*130+100] == 99,
                   "live expanded SLAM retains frozen deformation at world position")
        rospy.loginfo("PASS: actual Navfn soft-start egress, all-soft map, detour, wall and UNKNOWN")


if __name__ == "__main__":
    rospy.init_node("test_soft_cost_navigation")
    rostest.rosrun("deform_monitor_bringup", "soft_cost_navigation", SoftCostNavigationTest)
