#!/usr/bin/env python3
"""Summarize Mid360 scan health and observed laser_retro IDs."""

import argparse
import json
import sys
import time
from collections import Counter

import rospy
from livox_ros_driver.msg import CustomMsg


class ScanStatistics:
    def __init__(self):
        self.message_count = 0
        self.point_count = 0
        self.first_wall_time = None
        self.last_wall_time = None
        self.reflectivity_counts = Counter()

    def callback(self, message):
        now = time.monotonic()
        if self.first_wall_time is None:
            self.first_wall_time = now
        self.last_wall_time = now
        self.message_count += 1
        self.point_count += len(message.points)
        self.reflectivity_counts.update(int(point.reflectivity) for point in message.points)

    def report(self, topic, duration):
        span = 0.0
        if self.first_wall_time is not None and self.last_wall_time is not None:
            span = self.last_wall_time - self.first_wall_time
        rate = ((self.message_count - 1) / span
                if self.message_count > 1 and span > 0.0 else 0.0)
        return {
            "topic": topic,
            "measurement_wall_time_s": duration,
            "message_count": self.message_count,
            "message_rate_hz": rate,
            "total_points": self.point_count,
            "average_points_per_message": (
                self.point_count / self.message_count if self.message_count else 0.0
            ),
            "observed_laser_retro_ids": sorted(self.reflectivity_counts),
            "points_by_laser_retro_id": {
                str(key): self.reflectivity_counts[key]
                for key in sorted(self.reflectivity_counts)
            },
        }


def main():
    parser = argparse.ArgumentParser(
        description="Measure Mid360 topic health and laser_retro visibility"
    )
    parser.add_argument("--topic", default="/livox/lidar")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--minimum-messages", type=int, default=2)
    parser.add_argument(
        "--require-id", action="append", type=int, default=[],
        help="Fail unless this laser_retro ID is observed; may be repeated",
    )
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    if args.duration <= 0.0 or args.minimum_messages < 1:
        parser.error("duration and minimum-messages must be positive")

    rospy.init_node("icra2027_scan_check", anonymous=True)
    statistics = ScanStatistics()
    subscriber = rospy.Subscriber(args.topic, CustomMsg, statistics.callback, queue_size=20)
    deadline = time.monotonic() + args.duration
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        rospy.rostime.wallsleep(min(0.05, max(0.0, deadline - time.monotonic())))
    subscriber.unregister()

    report = statistics.report(args.topic, args.duration)
    print(json.dumps(report, indent=2, sort_keys=True))
    missing = sorted(set(args.require_id) - set(statistics.reflectivity_counts))
    if statistics.message_count < args.minimum_messages:
        print("ERROR: insufficient lidar messages", file=sys.stderr)
        return 1
    if statistics.point_count == 0:
        print("ERROR: lidar messages contain no points", file=sys.stderr)
        return 1
    if missing:
        print("ERROR: required laser_retro IDs not observed: {}".format(missing), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
