#!/usr/bin/env python3
"""Derive navigation-base pose/Twist from FAST-LIO sensor odometry.

FAST-LIO intentionally publishes pose/covariance on /Odometry but leaves the
Twist field zero.  DWA uses Twist to construct its dynamic velocity window, so
feeding the unmodified message permanently limits translation to min_vel_x.
FAST-LIO's ``body`` origin is the front IMU, not the geometric centre used by
the robot footprint.  This adapter composes a frozen rigid body-to-navigation
base calibration, differentiates that centre pose, and publishes its TF.
Gazebo truth is neither subscribed nor used.
"""

import copy
import math
import threading

import rospy
import tf2_ros
import tf.transformations as transformations
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def pose_yaw(pose):
    q = pose.orientation
    return transformations.euler_from_quaternion(
        (q.x, q.y, q.z, q.w))[2]


def estimate_body_twist(previous_pose, current_pose, dt):
    """Return (vx, vy, vz, wz) expressed in the current body frame."""
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("odometry interval must be positive")
    yaw = pose_yaw(current_pose)
    previous_yaw = pose_yaw(previous_pose)
    vx_world = (current_pose.position.x - previous_pose.position.x) / dt
    vy_world = (current_pose.position.y - previous_pose.position.y) / dt
    vz_world = (current_pose.position.z - previous_pose.position.z) / dt
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return (
        cosine * vx_world + sine * vy_world,
        -sine * vx_world + cosine * vy_world,
        vz_world,
        normalize_angle(yaw - previous_yaw) / dt,
    )


def compose_pose_with_translation(pose, translation_xyz):
    """Return the rigidly offset child pose in the same world frame.

    ``translation_xyz`` is the child origin expressed in the input pose frame.
    The full measured quaternion rotates the translation, so small FAST-LIO
    roll/pitch estimates do not silently become planar calibration errors.
    """
    translation = tuple(float(value) for value in translation_xyz)
    if len(translation) != 3 or not all(math.isfinite(v) for v in translation):
        raise ValueError("navigation-base translation must be finite xyz")
    output = copy.deepcopy(pose)
    quaternion = (pose.orientation.x, pose.orientation.y,
                  pose.orientation.z, pose.orientation.w)
    if not all(math.isfinite(value) for value in quaternion):
        raise ValueError("input pose quaternion must be finite")
    rotation = transformations.quaternion_matrix(quaternion)
    rotated = rotation[:3, :3].dot(translation)
    output.position.x += float(rotated[0])
    output.position.y += float(rotated[1])
    output.position.z += float(rotated[2])
    return output


class LioTwistOdometry:
    def __init__(self):
        self.lock = threading.RLock()
        self.previous = None
        self.maximum_linear_speed = float(rospy.get_param(
            "~maximum_linear_speed_mps", 3.0))
        self.maximum_angular_speed = float(rospy.get_param(
            "~maximum_angular_speed_rad_s", 3.0))
        self.maximum_interval = float(rospy.get_param(
            "~maximum_interval_s", 0.5))
        calibration = rospy.get_param(
            "~body_to_navigation_base_xyz", [-0.280, 0.0, -0.617102])
        if len(calibration) != 3:
            raise ValueError("~body_to_navigation_base_xyz must contain xyz")
        self.body_to_navigation_base = tuple(float(value)
                                             for value in calibration)
        if not all(math.isfinite(value)
                   for value in self.body_to_navigation_base):
            raise ValueError("navigation-base calibration must be finite")
        self.navigation_base_frame = str(rospy.get_param(
            "~navigation_base_frame", "navigation_base"))
        if not self.navigation_base_frame:
            raise ValueError("~navigation_base_frame cannot be empty")
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.publisher = rospy.Publisher(
            rospy.get_param("~output_topic", "/navigation/odometry"),
            Odometry, queue_size=10)
        rospy.Subscriber(rospy.get_param("~input_topic", "/Odometry"),
                         Odometry, self._callback, queue_size=10)

    def _callback(self, message):
        output = copy.deepcopy(message)
        output.pose.pose = compose_pose_with_translation(
            message.pose.pose, self.body_to_navigation_base)
        output.child_frame_id = self.navigation_base_frame
        with self.lock:
            previous = self.previous
            self.previous = copy.deepcopy(output)
        if previous is not None:
            dt = (output.header.stamp - previous.header.stamp).to_sec()
            if 0.0 < dt <= self.maximum_interval:
                try:
                    vx, vy, vz, wz = estimate_body_twist(
                        previous.pose.pose, output.pose.pose, dt)
                except ValueError:
                    vx = vy = vz = wz = 0.0
                linear_norm = math.sqrt(vx * vx + vy * vy + vz * vz)
                if (all(math.isfinite(value) for value in (vx, vy, vz, wz))
                        and linear_norm <= self.maximum_linear_speed
                        and abs(wz) <= self.maximum_angular_speed):
                    output.twist.twist.linear.x = vx
                    output.twist.twist.linear.y = vy
                    output.twist.twist.linear.z = vz
                    output.twist.twist.angular.z = wz
        self.publisher.publish(output)
        transform = TransformStamped()
        transform.header = copy.deepcopy(output.header)
        transform.child_frame_id = self.navigation_base_frame
        transform.transform.translation.x = output.pose.pose.position.x
        transform.transform.translation.y = output.pose.pose.position.y
        transform.transform.translation.z = output.pose.pose.position.z
        transform.transform.rotation = copy.deepcopy(output.pose.pose.orientation)
        self.tf_broadcaster.sendTransform(transform)


def main():
    rospy.init_node("lio_twist_odometry")
    LioTwistOdometry()
    rospy.spin()


if __name__ == "__main__":
    main()
