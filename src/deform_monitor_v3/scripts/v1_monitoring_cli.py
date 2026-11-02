#!/usr/bin/env python3
"""Reliable phase control for the V1 LiDAR deformation monitor."""

import argparse
import sys
import time
import threading

import rospy
from deform_monitor_v3.msg import V1DetectionStatus, V1StableId128
from deform_monitor_v3.srv import SetV1MonitoringPhase, SetV1MonitoringPhaseRequest


PHASE_NAMES = {
    0: "MAPPING_CONTEXT",
    1: "MONITOR_REFERENCE",
    2: "MONITOR_UPDATE",
    3: "CLOSED",
}

REASON_NAMES = {
    1: "INPUT_SCHEMA_INVALID",
    5: "INPUT_TIME_INVALID",
    20: "REFERENCE_NOT_FROZEN",
    21: "REFERENCE_INSUFFICIENT_EXPOSURE",
    22: "REFERENCE_RESOURCE_LIMIT",
    23: "PRE_REFERENCE_CHANGE_UNOBSERVABLE",
    24: "REFERENCE_SELF_CONSISTENCY_FAILED",
    42: "ASSOCIATION_AMBIGUOUS",
    63: "RESOURCE_LIMIT",
    64: "DEADLINE_EXCEEDED",
    65: "UNKNOWN_PREDICTION",
    66: "SNAPSHOT_STALE",
    67: "QUERY_CONTRACT_INVALID",
}


def phase_name(value):
    return PHASE_NAMES.get(value, "INVALID({})".format(value))


def reason_text(values):
    if not values:
        return "NONE"
    return ",".join(REASON_NAMES.get(value, str(value)) for value in values)


def reference_detail():
    return rospy.get_param("/deform_monitor_v1/reference_diagnostics", "")


def call_phase_with_progress(service, request, label):
    # ROS service calls are synchronous and cannot be cancelled safely. Keep
    # progress visible while the server builds the reference or filter caches.
    done = threading.Event()
    result, errors = [], []
    def invoke():
        try:
            result.append(service(request))
        except Exception as error:
            errors.append(error)
        finally:
            done.set()
    worker = threading.Thread(target=invoke, daemon=True)
    started = time.monotonic()
    print("{} ...".format(label), flush=True)
    worker.start()
    while not done.wait(2.0):
        print("{}: {:.1f}s wall time; keep bag before target motion".format(
            label, time.monotonic() - started), flush=True)
        if rospy.is_shutdown():
            raise RuntimeError("CLI stopped; server phase request may still be running")
    if errors:
        raise errors[0]
    print("{} completed in {:.2f}s".format(label, time.monotonic() - started), flush=True)
    return result[0]


def wait_status(predicate, timeout_s, progress=False):
    deadline = time.monotonic() + timeout_s
    next_progress = time.monotonic()
    last = None
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        if progress and time.monotonic() >= next_progress:
            print("Waiting for reference: {}".format(reference_detail() or
                  "waiting for input/status; continue only the static baseline"), flush=True)
            next_progress = time.monotonic() + 2.0
        remaining = max(0.05, min(1.0, deadline - time.monotonic()))
        try:
            last = rospy.wait_for_message(
                "/deform/v1/status", V1DetectionStatus, timeout=remaining)
        except rospy.ROSException:
            continue
        if predicate(last):
            return last
        # Wall sleep is intentional: bag replay may reset /clock before the
        # first sensor bundle, while this polling loop must remain usable.
        time.sleep(0.02)
    detail = "no status" if last is None else (
        "phase={}, frozen={}, stamp={:.3f}, reasons={}".format(
            phase_name(last.phase), last.reference_frozen,
            last.last_attempt_stamp.to_sec(),
            reason_text(last.last_attempt_reason_codes)))
    raise RuntimeError("timeout while waiting for monitor: " + detail)


def make_request(args, target_phase, effective_stamp, selection_cutoff):
    request = SetV1MonitoringPhaseRequest()
    request.session_id = args.session_id
    request.target_phase = target_phase
    request.roi_min.x, request.roi_min.y, request.roi_min.z = args.roi_min
    request.roi_max.x, request.roi_max.y, request.roi_max.z = args.roi_max
    request.effective_stamp = effective_stamp
    request.trigger_source = 0
    request.selection_support_cutoff = selection_cutoff
    request.expected_previous_snapshot = V1StableId128()
    return request


def print_status(status):
    print("phase={} reference_frozen={} GDU={} "
          "last_stamp={:.3f} published={} reasons={}".format(
              phase_name(status.phase), status.reference_frozen,
              status.gdu_count,
              status.last_attempt_stamp.to_sec(),
              status.last_attempt_published,
              reason_text(status.last_attempt_reason_codes)))


def start_monitoring(args):
    rospy.wait_for_service("/deform/v1/set_monitoring_phase", timeout=args.timeout)
    service = rospy.ServiceProxy(
        "/deform/v1/set_monitoring_phase", SetV1MonitoringPhase)
    initial = wait_status(
        lambda status: status.last_attempt_stamp.to_sec() > 0.0,
        args.timeout)
    if initial.phase == 2:
        print("monitor is already in MONITOR_UPDATE")
        return 0
    if initial.phase != 0:
        raise RuntimeError(
            "start requires a fresh MAPPING_CONTEXT session; current phase is {}. "
            "Restart the detection launch and run this command once.".format(
                phase_name(initial.phase)))

    # Trigger 0 first tries pre-cutoff history; only if that is unavailable
    # does it collect a post-request baseline. Neither is the trigger-2 5N dwell.
    selection_cutoff = rospy.Time.from_sec(max(
        args.minimum_sim_time, initial.last_attempt_stamp.to_sec()))
    print("Monitoring ROI in camera_init: {} -> {}".format(args.roi_min, args.roi_max), flush=True)
    reference_response = call_phase_with_progress(service, make_request(
        args, 1, selection_cutoff, selection_cutoff), "Selecting/building reference")
    if not reference_response.accepted:
        raise RuntimeError("MONITOR_REFERENCE rejected: " +
                           reason_text(reference_response.reason_codes))
    print("reference request accepted; waiting for a frozen reference ...", flush=True)

    frozen = wait_status(
        lambda status: status.phase == 1 and status.reference_frozen,
        args.timeout, progress=True)
    detail = reference_detail()
    if detail:
        print("Reference coverage: " + detail, flush=True)
    if 23 in frozen.last_attempt_reason_codes or "post_request_baseline" in detail:
        print("WARNING: reference uses scans AFTER your request. If the target already "
              "moved, restart from a static bag segment; earlier displacement is not recoverable.",
              flush=True)
    # Use the sensor stamp of the exact status that reports the frozen
    # reference.  This preserves the configured max_update_gap_s contract and
    # avoids the race caused by a fixed effective_stamp in manual commands.
    update_stamp = frozen.last_attempt_stamp
    update_response = call_phase_with_progress(service, make_request(
        args, 2, update_stamp, selection_cutoff), "Initializing deformation monitoring")
    if not update_response.accepted:
        raise RuntimeError("MONITOR_UPDATE rejected: " +
                           reason_text(update_response.reason_codes))
    active = wait_status(lambda status: status.phase == 2, args.timeout)
    print_status(active)
    print("MONITORING ACTIVE: reference frozen; now resume the motion segment "
          "or start the physical motion.", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Control V1 reference freeze and monitoring phases")
    parser.add_argument("command", choices=("start", "status"))
    parser.add_argument("--session-id",
                        default="deform-monitor-research-session-v1")
    parser.add_argument("--roi-min", type=float, nargs=3,
                        default=(-10.0, -10.0, -1.0),
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--roi-max", type=float, nargs=3,
                        default=(10.0, 10.0, 5.0),
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--minimum-sim-time", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    rospy.init_node("deform_monitor_v1_phase_cli", anonymous=True,
                    disable_signals=True)
    try:
        if args.command == "status":
            print_status(wait_status(lambda _: True, args.timeout))
            return 0
        return start_monitoring(args)
    except (RuntimeError, rospy.ROSException, rospy.ServiceException) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
