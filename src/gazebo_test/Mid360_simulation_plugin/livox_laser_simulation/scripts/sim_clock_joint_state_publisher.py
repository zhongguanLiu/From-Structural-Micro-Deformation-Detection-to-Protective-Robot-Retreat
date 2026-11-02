#!/usr/bin/env python3
"""Publish fixed visualization joints only at valid Gazebo clock epochs.

The stock joint_state_publisher emits once immediately.  During Gazebo
startup that first stamp can be wall time, followed by much smaller simulated
stamps.  robot_state_publisher then reports TF_OLD_DATA continuously.  This
small publisher is deliberately driven by /clock and therefore cannot publish
before simulation time exists.  It is visualization-only; no value is fed to
Gazebo, FAST-LIO, deformation detection, or planning.
"""

import math

import rospy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState


def stamp_to_nanoseconds(stamp):
    return int(stamp.secs) * 1_000_000_000 + int(stamp.nsecs)


def publish_due(last_publish_ns, current_ns, period_ns):
    """Return whether a valid clock epoch is due for publication."""
    if current_ns <= 0 or period_ns <= 0:
        return False
    if last_publish_ns is None:
        return True
    # A new Gazebo epoch must be re-anchored instead of waiting for the old
    # epoch to catch up.  In the normal experiment clock never regresses.
    if current_ns < last_publish_ns:
        return True
    return current_ns - last_publish_ns >= period_ns


class SimClockJointStatePublisher:
    def __init__(self):
        zeros = rospy.get_param("~zeros", {})
        if not isinstance(zeros, dict) or not zeros:
            raise ValueError("~zeros must be a non-empty joint:value mapping")

        names = sorted(str(name) for name in zeros.keys())
        positions = [float(zeros[name]) for name in names]
        if any(not math.isfinite(value) for value in positions):
            raise ValueError("all fixed joint values must be finite")

        rate_hz = float(rospy.get_param("~publish_rate_hz", 2.0))
        if not math.isfinite(rate_hz) or rate_hz <= 0.0 or rate_hz > 20.0:
            raise ValueError("~publish_rate_hz must be in (0, 20]")

        self._names = names
        self._positions = positions
        self._period_ns = max(1, int(round(1.0e9 / rate_hz)))
        self._last_clock_ns = None
        self._last_publish_ns = None
        self._publisher = rospy.Publisher("joint_states", JointState,
                                          queue_size=1)
        self._clock_subscriber = rospy.Subscriber(
            "/clock", Clock, self._on_clock, queue_size=1,
            tcp_nodelay=True)
        rospy.loginfo(
            "sim-clock fixed joint publisher armed: joints=%d rate=%.3f Hz; "
            "waiting for valid /clock", len(names), rate_hz)

    def _on_clock(self, message):
        current_ns = stamp_to_nanoseconds(message.clock)
        if current_ns <= 0:
            return

        if (self._last_clock_ns is not None
                and current_ns < self._last_clock_ns):
            rospy.logwarn(
                "Gazebo /clock regressed; re-anchoring fixed joint publisher "
                "to the new simulation epoch")
            self._last_publish_ns = None
        self._last_clock_ns = current_ns

        if not publish_due(self._last_publish_ns, current_ns,
                           self._period_ns):
            return

        output = JointState()
        output.header.stamp = message.clock
        output.name = self._names
        output.position = self._positions
        output.velocity = []
        output.effort = []
        self._publisher.publish(output)
        self._last_publish_ns = current_ns


def main():
    rospy.init_node("sim_clock_fixed_joint_state_publisher")
    try:
        SimClockJointStatePublisher()
    except (TypeError, ValueError) as error:
        rospy.logfatal("invalid fixed joint publisher configuration: %s", error)
        raise
    rospy.spin()


if __name__ == "__main__":
    main()
