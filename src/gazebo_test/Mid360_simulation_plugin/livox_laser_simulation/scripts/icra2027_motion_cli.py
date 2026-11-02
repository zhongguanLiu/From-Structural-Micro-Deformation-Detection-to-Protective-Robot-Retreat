#!/usr/bin/env python3
"""Command-line control for the deterministic ICRA2027 motion experiment."""

import argparse
import json
import sys
import time

import rospy
from std_msgs.msg import String
from std_srvs.srv import Trigger


ALLOWED_LINEAR_SPEEDS = (1.0, 3.0, 5.0, 10.0)
ALLOWED_ANGULAR_SPEEDS = (0.0015, 0.0025)


class MotionClient:
    def __init__(self, namespace):
        self.namespace = "/" + namespace.strip("/")
        self.latest_status = None
        self.status_subscriber = rospy.Subscriber(
            self.namespace + "/status", String, self._status_callback, queue_size=10
        )

    def _status_callback(self, message):
        try:
            self.latest_status = json.loads(message.data)
        except (TypeError, ValueError):
            rospy.logwarn_throttle(2.0, "Received invalid motion status JSON")

    def call(self, service_name, timeout=15.0):
        full_name = self.namespace + "/" + service_name
        rospy.wait_for_service(full_name, timeout=timeout)
        response = rospy.ServiceProxy(full_name, Trigger)()
        if not response.success:
            raise RuntimeError("{} failed: {}".format(full_name, response.message))
        return response.message

    def status(self):
        return json.loads(self.call("get_status"))

    def wait_for_reset(self, previous_count, timeout):
        deadline = time.monotonic() + timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            status = self.status()
            if int(status.get("reset_count", 0)) > previous_count:
                return status
            rospy.sleep(0.05)
        raise RuntimeError("Timed out waiting for Gazebo to apply the reset")

    def wait_for_completion(self, timeout):
        deadline = time.monotonic() + timeout
        last_state = None
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            status = self.latest_status or self.status()
            state = status.get("state", "UNKNOWN")
            if state != last_state:
                print("state: {} (sim_time={:.3f})".format(
                    state, float(status.get("sim_time", 0.0))))
                last_state = state
            if state == "COMPLETED":
                return status
            if state == "ERROR":
                raise RuntimeError(status.get("message", "Motion plugin entered ERROR"))
            rospy.sleep(0.1)
        raise RuntimeError("Timed out waiting for the motion experiment to complete")


def speed_choice(value, allowed, label):
    parsed = float(value)
    if not any(abs(parsed - candidate) < 1.0e-12 for candidate in allowed):
        raise argparse.ArgumentTypeError(
            "{} must be one of {}".format(label, ", ".join(map(str, allowed)))
        )
    return parsed


def print_status(status):
    print(json.dumps(status, indent=2, sort_keys=True, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(
        description="Control and monitor the six moving targets in ICRA2027.world"
    )
    parser.add_argument("action", choices=("run", "start", "stop", "reset", "reload", "status"))
    parser.add_argument("--namespace", default="/icra2027_motion")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--linear-speed-mm-s",
        type=lambda value: speed_choice(value, ALLOWED_LINEAR_SPEEDS, "linear speed"),
    )
    parser.add_argument(
        "--angular-speed-rad-s",
        type=lambda value: speed_choice(value, ALLOWED_ANGULAR_SPEEDS, "angular speed"),
    )
    parser.add_argument(
        "--no-reset", action="store_true",
        help="Start from the current target poses instead of restoring the captured scene poses",
    )
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    rospy.init_node("icra2027_motion_cli", anonymous=True, disable_signals=True)
    client = MotionClient(args.namespace)

    try:
        if args.action == "status":
            print_status(client.status())
            return 0

        if args.action == "reset":
            before = client.status()
            print(client.call("reset"))
            print_status(client.wait_for_reset(int(before.get("reset_count", 0)), args.timeout))
            return 0

        if args.action in ("start", "stop", "reload"):
            print(client.call(args.action))
            print_status(client.status())
            return 0

        if not args.no_reset:
            before = client.status()
            print(client.call("reset"))
            client.wait_for_reset(int(before.get("reset_count", 0)), args.timeout)

        parameter_namespace = "/" + args.namespace.strip("/")
        if args.linear_speed_mm_s is not None:
            rospy.set_param(parameter_namespace + "/linear_speed_mm_s", args.linear_speed_mm_s)
        if args.angular_speed_rad_s is not None:
            rospy.set_param(parameter_namespace + "/angular_speed_rad_s", args.angular_speed_rad_s)

        print(client.call("reload"))
        print(client.call("start"))
        print_status(client.wait_for_completion(args.timeout))
        return 0
    except (rospy.ROSException, rospy.ServiceException, RuntimeError, ValueError) as error:
        try:
            client.call("stop", timeout=2.0)
        except Exception:
            pass
        print("ERROR: {}".format(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        try:
            print(client.call("stop", timeout=2.0))
        except Exception:
            pass
        return 130


if __name__ == "__main__":
    sys.exit(main())
