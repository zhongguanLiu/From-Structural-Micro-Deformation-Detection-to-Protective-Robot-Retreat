#!/usr/bin/env python3

"""Unpause Gazebo only after the complete dual-Livox chain is connected.

The two simulated Mid360 plugins each load an 800k-ray scan table.  If physics
is already advancing, independent 10 Hz CPU RaySensor schedules drift under
unequal load.  In the ICRA2027 model the primary plugin therefore owns one
shared sampling event and evaluates both real ray tables at that same Gazebo
epoch.  Starting paused here additionally guarantees that neither source can
publish before the merger has completed both transport handshakes.
"""

import time

import rospy
from std_msgs.msg import Bool
from std_srvs.srv import Empty


def wait_for_true(topic, deadline):
    """Wait for a latched Bool(true) without extending the shared deadline."""
    while not rospy.is_shutdown():
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return False
        try:
            message = rospy.wait_for_message(
                topic, Bool, timeout=min(remaining, 1.0))
        except rospy.ROSException:
            continue
        if bool(message.data):
            return True
        time.sleep(0.01)
    return False


def main():
    rospy.init_node("unpause_synchronized_dual_lidar")
    required = [
        str(rospy.get_param(
            "~primary_ready_topic", "/livox/front/lidar/ready")),
        str(rospy.get_param(
            "~secondary_ready_topic", "/livox/rear/lidar/ready")),
        str(rospy.get_param(
            "~merge_ready_topic", "/livox/merge/ready")),
    ]
    if len(set(required)) != len(required) or any(not name for name in required):
        raise ValueError("dual Livox readiness topics must be distinct and non-empty")
    timeout = float(rospy.get_param("~wall_timeout_s", 30.0))
    if not 0.0 < timeout <= 120.0:
        raise ValueError("wall_timeout_s must be in (0,120]")
    deadline = time.monotonic() + timeout
    for topic in required:
        if not wait_for_true(topic, deadline):
            raise RuntimeError(
                "dual Livox chain did not become ready before timeout: {}".format(
                    topic))
    service_timeout = deadline - time.monotonic()
    if service_timeout <= 0.0:
        raise RuntimeError("dual Livox readiness exhausted the unpause deadline")
    rospy.wait_for_service(
        "/gazebo/unpause_physics", timeout=min(service_timeout, 5.0))
    rospy.ServiceProxy("/gazebo/unpause_physics", Empty)()
    rospy.loginfo(
        "Gazebo unpaused after both shared-epoch ray tables and merger "
        "transports were ready")
    rospy.spin()


if __name__ == "__main__":
    main()
