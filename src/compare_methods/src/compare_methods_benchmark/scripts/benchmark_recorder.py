#!/usr/bin/env python3
"""Compact, detector-independent recorder for the ICRA2027 comparison runs."""

import csv
import gzip
import hashlib
import json
import math
import os
import pathlib
import tempfile
import threading
import time
from collections import deque

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Float64MultiArray, String
from std_srvs.srv import Trigger


METHOD_TOPICS = {
    "m3c2": ("/compare/m3c2/dynamic_points", "/compare/m3c2/runtime"),
    "m_detector": (
        "/compare/m_detector/dynamic_points",
        "/compare/m_detector/runtime"),
    "spatiotemporal_normals": (
        "/compare/spatiotemporal_normals/dynamic_points",
        "/compare/spatiotemporal_normals/runtime"),
    "dynablox": (
        "/compare/dynablox/dynamic_points",
        "/compare/dynablox/runtime"),
}
MODEL_NAMES = ("m01", "m02", "m03", "m04", "m05", "m06")
GROUP_NAMES = MODEL_NAMES + ("OTHER",)
AGGREGATE_FIELDS = (
    "source_stamp_s", "arrival_sim_time_s", "truth_sim_time_s",
    "truth_time_delta_s", "truth_experiment_elapsed_s",
    "truth_motion_elapsed_s", "truth_state", "source_frame_id",
    "group_name", "point_count", "score_mean", "score_max", "centroid_x",
    "centroid_y", "centroid_z", "min_x", "min_y", "min_z", "max_x",
    "max_y", "max_z")


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def atomic_json(path, payload):
    path = pathlib.Path(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Aggregate:
    def __init__(self):
        self.count = 0
        self.score_sum = 0.0
        self.score_count = 0
        self.score_max = None
        self.sum_xyz = [0.0, 0.0, 0.0]
        self.min_xyz = [float("inf")] * 3
        self.max_xyz = [float("-inf")] * 3

    def add(self, xyz, score):
        self.count += 1
        score = finite(score)
        if score is not None:
            self.score_sum += score
            self.score_count += 1
            self.score_max = score if self.score_max is None else max(
                self.score_max, score)
        for axis, value in enumerate(xyz):
            self.sum_xyz[axis] += value
            self.min_xyz[axis] = min(self.min_xyz[axis], value)
            self.max_xyz[axis] = max(self.max_xyz[axis], value)

    def row(self, metadata, group_name):
        if not self.count:
            return dict(metadata, group_name=group_name, point_count=0,
                        score_mean="", score_max="", centroid_x="",
                        centroid_y="", centroid_z="", min_x="", min_y="",
                        min_z="", max_x="", max_y="", max_z="")
        return dict(
            metadata, group_name=group_name, point_count=self.count,
            score_mean=("" if not self.score_count else
                        self.score_sum / self.score_count),
            score_max="" if self.score_max is None else self.score_max,
            centroid_x=self.sum_xyz[0] / self.count,
            centroid_y=self.sum_xyz[1] / self.count,
            centroid_z=self.sum_xyz[2] / self.count,
            min_x=self.min_xyz[0], min_y=self.min_xyz[1],
            min_z=self.min_xyz[2], max_x=self.max_xyz[0],
            max_y=self.max_xyz[1], max_z=self.max_xyz[2])


class BenchmarkRecorder:
    def __init__(self):
        self.method = rospy.get_param("~method")
        self.configured_speed = float(rospy.get_param(
            "~configured_speed_mm_s"))
        if self.method not in METHOD_TOPICS:
            raise ValueError("unsupported comparison method: " + self.method)
        self.output_dir = pathlib.Path(os.path.abspath(os.path.expanduser(
            rospy.get_param("~output_directory"))))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "truth").mkdir(exist_ok=True)
        (self.output_dir / "detections").mkdir(exist_ok=True)
        (self.output_dir / "runtime").mkdir(exist_ok=True)
        (self.output_dir / "configuration").mkdir(exist_ok=True)
        self.world_file = rospy.get_param("~world_file")
        self.motion_config_file = rospy.get_param("~motion_config_file")
        self.world_file_hash = file_sha256(self.world_file)
        self.motion_config_file_hash = file_sha256(self.motion_config_file)
        with gzip.open(
                self.output_dir / "configuration" / "ros_parameters.json.gz",
                "wt", encoding="utf-8") as stream:
            json.dump(rospy.get_param("/"), stream, sort_keys=True)
            stream.write("\n")
        self.margin = float(rospy.get_param("~aabb_margin_m", 0.10))
        origin = rospy.get_param("~camera_init_origin_gazebo_xyzyaw")
        if len(origin) != 4 or any(finite(value) is None for value in origin):
            raise ValueError("camera_init_origin_gazebo_xyzyaw must have four finite values")
        self.camera_origin = tuple(float(value) for value in origin)
        self.input_stride = max(1, int(rospy.get_param(
            "~input_metadata_stride", 5)))
        self.input_callback_count = 0
        self.raw_input_callback_count = 0
        self.motion_start_after_frames = max(1, int(rospy.get_param(
            "~motion_start_after_input_frames", 10)))
        self.motion_start_requested = False
        self.reference_ready = self.method != "m3c2"
        self.motion_start_error = None
        self.truth_snapshots = deque(maxlen=2000)
        self.output_lock = threading.RLock()
        self.counts = {"truth_messages": 0, "input_frames": 0,
                       "detection_frames": 0, "runtime_frames": 0,
                       "odometry_messages": 0}
        self.coverage = {
            name: {"first_source_stamp_s": None, "last_source_stamp_s": None}
            for name in ("input", "detection", "runtime", "odometry")}
        self.source_frame_ids = {"input": set(), "detection": set()}
        self.truth_association_max_delta_s = {
            "input": 0.0, "detection": 0.0}
        self.truth_association_missing = {"input": 0, "detection": 0}
        self.completed = False
        self.final_truth = None
        self.closed = False

        self.truth_stream = gzip.open(
            self.output_dir / "truth" / "motion_truth.jsonl.gz", "wt",
            encoding="utf-8", newline="")
        self.input_stream, self.input_writer = self._csv(
            self.output_dir / "runtime" / "input_frames.csv.gz")
        self.detection_stream, self.detection_writer = self._csv(
            self.output_dir / "detections" / "detection_frames.csv.gz")
        self.runtime_stream, self.runtime_writer = self._csv(
            self.output_dir / "runtime" / "algorithm_runtime.csv.gz",
            fields=("source_stamp_s", "frame_index", "input_point_count",
                    "dynamic_point_count", "processing_time_ms",
                    "arrival_sim_time_s"))
        self.odometry_stream, self.odometry_writer = self._csv(
            self.output_dir / "runtime" / "fastlio_odometry.csv.gz",
            fields=("source_stamp_s", "frame_id", "child_frame_id",
                    "position_x", "position_y", "position_z",
                    "orientation_x", "orientation_y", "orientation_z",
                    "orientation_w", "covariance_xx", "covariance_yy",
                    "covariance_zz", "covariance_roll_roll",
                    "covariance_pitch_pitch", "covariance_yaw_yaw"))

        truth_topic = rospy.get_param(
            "~truth_topic", "/icra2027_motion/truth")
        input_topic = rospy.get_param("~input_topic", "/cloud_registered")
        detection_topic, runtime_topic = METHOD_TOPICS[self.method]
        self.truth_sub = rospy.Subscriber(
            truth_topic, String, self._truth, queue_size=100)
        self.input_sub = rospy.Subscriber(
            input_topic, PointCloud2, self._input, queue_size=2)
        self.detection_sub = rospy.Subscriber(
            detection_topic, PointCloud2, self._detection, queue_size=10)
        self.runtime_sub = rospy.Subscriber(
            runtime_topic, Float64MultiArray, self._runtime, queue_size=50)
        self.odometry_sub = rospy.Subscriber(
            rospy.get_param("~odometry_topic", "/Odometry"), Odometry,
            self._odometry, queue_size=50)
        if self.method == "m3c2":
            self.ready_sub = rospy.Subscriber(
                "/compare/m3c2/reference_ready", Bool, self._ready, queue_size=1)
        rospy.on_shutdown(self.close)
        rospy.loginfo("comparison recorder method=%s output=%s",
                      self.method, self.output_dir)

    def _csv(self, path, fields=AGGREGATE_FIELDS):
        stream = gzip.open(path, "wt", encoding="utf-8", newline="")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        return stream, writer

    def _truth(self, message):
        try:
            document = json.loads(message.data)
            sim_time = float(document["sim_time"])
            if {model.get("model_name") for model in document.get(
                    "models", [])} != set(MODEL_NAMES):
                raise ValueError("truth contains an unexpected model")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            rospy.logerr_throttle(2.0, "invalid comparison truth: %s", error)
            return
        with self.output_lock:
            if self.closed:
                return
            self.truth_snapshots.append((sim_time, document))
            self.final_truth = document
            self.truth_stream.write(json.dumps(document, sort_keys=True) + "\n")
            self.counts["truth_messages"] += 1
            if document.get("state") == "COMPLETED" and not self.completed:
                self.completed = True
                rospy.Timer(rospy.Duration(0.5), self._finish, oneshot=True)

    def _nearest_truth(self, stamp):
        # A buffered detector can emit a pre-experiment source stamp after the
        # first truth message arrives.  Never label that frame with future
        # truth; it belongs to detector warm-up and is excluded from metrics.
        with self.output_lock:
            snapshots = list(self.truth_snapshots)
        if (not snapshots or stamp < snapshots[0][0] - 1e-9):
            return None, None
        truth_time, document = min(
            snapshots, key=lambda item: abs(item[0] - stamp))
        return truth_time, document

    def _boxes(self, document):
        boxes = []
        if not document:
            return boxes
        for model in document.get("models", []):
            lower = model.get("aabb_min", ())
            upper = model.get("aabb_max", ())
            if len(lower) == 3 and len(upper) == 3:
                boxes.append((model["model_name"], lower, upper))
        return boxes

    def _group(self, xyz, boxes):
        matches = []
        for name, lower, upper in boxes:
            if all(lower[axis] - self.margin <= xyz[axis]
                   <= upper[axis] + self.margin for axis in range(3)):
                center = [(lower[i] + upper[i]) * 0.5 for i in range(3)]
                distance = sum((xyz[i] - center[i]) ** 2 for i in range(3))
                matches.append((distance, name))
        return min(matches)[1] if matches else "OTHER"

    def _to_gazebo(self, xyz):
        """Map FAST-LIO camera_init coordinates into Gazebo world coordinates."""
        x0, y0, z0, yaw = self.camera_origin
        cosine, sine = math.cos(yaw), math.sin(yaw)
        return (x0 + cosine * xyz[0] - sine * xyz[1],
                y0 + sine * xyz[0] + cosine * xyz[1], z0 + xyz[2])

    def _observe_stamp(self, stream_name, stamp):
        with self.output_lock:
            coverage = self.coverage[stream_name]
            if coverage["first_source_stamp_s"] is None:
                coverage["first_source_stamp_s"] = stamp
            coverage["last_source_stamp_s"] = stamp

    def _aggregate_cloud(self, message, with_score, stream_name):
        stamp = message.header.stamp.to_sec()
        self._observe_stamp(stream_name, stamp)
        with self.output_lock:
            self.source_frame_ids[stream_name].add(
                message.header.frame_id.lstrip("/"))
        truth_time, document = self._nearest_truth(stamp)
        with self.output_lock:
            if truth_time is None:
                self.truth_association_missing[stream_name] += 1
            else:
                self.truth_association_max_delta_s[stream_name] = max(
                    self.truth_association_max_delta_s[stream_name],
                    abs(truth_time - stamp))
        boxes = self._boxes(document)
        groups = {name: Aggregate() for name in GROUP_NAMES}
        fields = [field.name for field in message.fields]
        read_fields = ("x", "y", "z", "intensity") if (
            with_score and "intensity" in fields) else ("x", "y", "z")
        for point in point_cloud2.read_points(
                message, field_names=read_fields, skip_nans=True):
            xyz = self._to_gazebo(tuple(float(value) for value in point[:3]))
            score = point[3] if len(point) > 3 else None
            groups[self._group(xyz, boxes)].add(xyz, score)
        metadata = {
            "source_stamp_s": stamp,
            "arrival_sim_time_s": rospy.Time.now().to_sec(),
            "truth_sim_time_s": "" if truth_time is None else truth_time,
            "truth_time_delta_s": "" if truth_time is None else
                abs(truth_time - stamp),
            "truth_experiment_elapsed_s": "" if document is None else
                document.get("experiment_elapsed", ""),
            "truth_motion_elapsed_s": "" if document is None else
                document.get("motion_elapsed", ""),
            "truth_state": "" if document is None else
                document.get("state", ""),
            "source_frame_id": message.header.frame_id.lstrip("/"),
        }
        return groups, metadata

    def _input(self, message):
        if self.closed:
            return
        self.raw_input_callback_count += 1
        if (self.reference_ready and not self.motion_start_requested and
                self.raw_input_callback_count >= self.motion_start_after_frames):
            self.motion_start_requested = True
            threading.Thread(target=self._start_motion, daemon=True).start()
        self.input_callback_count += 1
        if self.input_callback_count % self.input_stride:
            return
        groups, metadata = self._aggregate_cloud(message, False, "input")
        with self.output_lock:
            if self.closed:
                return
            for name in GROUP_NAMES:
                self.input_writer.writerow(groups[name].row(metadata, name))
            self.counts["input_frames"] += 1

    def _ready(self, message):
        self.reference_ready = bool(message.data)

    def _start_motion(self):
        namespace = "/icra2027_motion"
        try:
            for service in ("get_status", "reset", "reload", "start"):
                rospy.wait_for_service(namespace + "/" + service, timeout=30.0)
            call = {name: rospy.ServiceProxy(namespace + "/" + name, Trigger)
                    for name in ("get_status", "reset", "reload", "start")}
            before = json.loads(call["get_status"]().message)
            response = call["reset"]()
            if not response.success:
                raise RuntimeError(response.message)
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                status = json.loads(call["get_status"]().message)
                if int(status.get("reset_count", 0)) > int(
                        before.get("reset_count", 0)):
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError("motion reset was not applied")
            for operation in ("reload", "start"):
                response = call[operation]()
                if not response.success:
                    raise RuntimeError(response.message)
            rospy.loginfo("COMPARISON_MOTION_STARTED after_input_frames=%d",
                          self.raw_input_callback_count)
        except Exception as error:
            self.motion_start_error = str(error)
            rospy.logfatal("comparison motion start failed: %s", error)
            rospy.signal_shutdown("comparison motion start failed")

    def _detection(self, message):
        if self.closed:
            return
        groups, metadata = self._aggregate_cloud(message, True, "detection")
        with self.output_lock:
            if self.closed:
                return
            for name in GROUP_NAMES:
                self.detection_writer.writerow(groups[name].row(metadata, name))
            self.counts["detection_frames"] += 1

    def _runtime(self, message):
        if self.closed:
            return
        if len(message.data) != 5:
            rospy.logerr_throttle(2.0, "invalid comparison runtime record")
            return
        self._observe_stamp("runtime", float(message.data[0]))
        row = dict(zip(self.runtime_writer.fieldnames,
                       list(message.data) + [rospy.Time.now().to_sec()]))
        with self.output_lock:
            if self.closed:
                return
            self.runtime_writer.writerow(row)
            self.counts["runtime_frames"] += 1

    def _odometry(self, message):
        if self.closed:
            return
        stamp = message.header.stamp.to_sec()
        self._observe_stamp("odometry", stamp)
        pose = message.pose.pose
        covariance = message.pose.covariance
        row = {
            "source_stamp_s": stamp,
            "frame_id": message.header.frame_id.lstrip("/"),
            "child_frame_id": message.child_frame_id.lstrip("/"),
            "position_x": pose.position.x,
            "position_y": pose.position.y,
            "position_z": pose.position.z,
            "orientation_x": pose.orientation.x,
            "orientation_y": pose.orientation.y,
            "orientation_z": pose.orientation.z,
            "orientation_w": pose.orientation.w,
            "covariance_xx": covariance[0],
            "covariance_yy": covariance[7],
            "covariance_zz": covariance[14],
            "covariance_roll_roll": covariance[21],
            "covariance_pitch_pitch": covariance[28],
            "covariance_yaw_yaw": covariance[35],
        }
        with self.output_lock:
            if self.closed:
                return
            self.odometry_writer.writerow(row)
            self.counts["odometry_messages"] += 1

    def _finish(self, _event):
        # Drain a slower M3C2 callback without shortening its last observation.
        if self.method == "m3c2":
            target = float(self.final_truth["sim_time"]) - 0.5
            deadline = time.monotonic() + 60.0
            drained = False
            while not rospy.is_shutdown() and time.monotonic() < deadline:
                with self.output_lock:
                    drained = all((self.coverage[name]["last_source_stamp_s"] or 0) >= target
                                  for name in ("detection", "runtime"))
                if drained:
                    break
                time.sleep(0.1)
            if not drained:
                self.completed = False
        self.close()
        rospy.loginfo("COMPARISON_RUN_COMPLETE method=%s", self.method)
        rospy.signal_shutdown("comparison timeline completed")

    def close(self):
        with self.output_lock:
            if self.closed:
                return
            self.closed = True
            for stream in (self.truth_stream, self.input_stream,
                           self.detection_stream, self.runtime_stream,
                           self.odometry_stream):
                stream.flush()
                stream.close()
            payload = {
                "schema_version": 1,
                "method": self.method,
                "configured_speed_mm_s": self.configured_speed,
                "truth_speed_mm_s": (None if self.final_truth is None else
                                      self.final_truth.get("linear_speed_mm_s")),
                "truth_final_state": (None if self.final_truth is None else
                                      self.final_truth.get("state")),
                "truth_final_experiment_elapsed_s": (
                    None if self.final_truth is None else
                    self.final_truth.get("experiment_elapsed")),
                "status": "COMPLETED" if self.completed else "INCOMPLETE",
                "single_front_lidar": True,
                "fastlio_input": True,
                "rosbag_recorded": False,
                "object_names_used_by_detector": False,
                "truth_usage": "evaluation_only",
                "reference_ready_before_motion": self.reference_ready,
                "world_file": self.world_file,
                "world_file_sha256": self.world_file_hash,
                "motion_config_file": self.motion_config_file,
                "motion_config_file_sha256": self.motion_config_file_hash,
                "ros_parameters_recorded": True,
                "motion_start_error": self.motion_start_error,
                "camera_init_origin_gazebo_xyzyaw": self.camera_origin,
                "source_frame_ids": {
                    name: sorted(values)
                    for name, values in self.source_frame_ids.items()},
                "source_stamp_coverage": self.coverage,
                "truth_association_max_delta_s":
                    self.truth_association_max_delta_s,
                "truth_association_missing_frames":
                    self.truth_association_missing,
                "counts": self.counts,
            }
            atomic_json(self.output_dir / "run_manifest.json", payload)


def main():
    rospy.init_node("compare_methods_benchmark_recorder")
    BenchmarkRecorder()
    rospy.spin()


if __name__ == "__main__":
    main()
