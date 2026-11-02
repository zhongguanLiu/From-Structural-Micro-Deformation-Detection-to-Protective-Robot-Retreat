#!/usr/bin/env python3
"""Record V1 real-data estimates, not RViz markers or another sensor bag."""
import datetime
import gzip
import hashlib
import json
import math
import pathlib
import queue
import threading
import time
import uuid
from collections import Counter

import rospy
from nav_msgs.msg import Odometry
from deform_monitor_v3.msg import V1DetectionStatus, V1DetectionTile


def plain(value):
    """Lossless scalar/field conversion; nonfinite values remain explicit strings."""
    if hasattr(value, "secs") and hasattr(value, "nsecs"):
        return {"secs": value.secs, "nsecs": value.nsecs}
    if hasattr(value, "__slots__"):
        # Per-component filter checkpoints are not required for output-trajectory
        # plots. Preserve all published marginal uncertainties and validity flags.
        return {key: plain(getattr(value, key)) for key in value.__slots__ if key != "components"}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def id_key(value):
    return "{:016x}{:016x}".format(value.high, value.low)


class Recorder:
    def __init__(self):
        root = pathlib.Path(rospy.get_param("~output_root"))
        name = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
        self.directory = root / name
        for child in ("metadata", "measurements", "events", "analysis", "figures"):
            (self.directory / child).mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.items = queue.Queue(maxsize=256)
        self.lock = threading.Lock()
        self.closed = False
        self.error = None
        self.received, self.written = Counter(), Counter()
        self.overflow = 0
        self.max_queue = 0
        self.snapshot_ids, self.published_ids, self.reference_epochs = set(), set(), set()
        self.previous_phase = None
        self.previous_input_index = None
        self.input_status_index_gaps = 0
        self.streams = {name: gzip.open(str(self.directory / "measurements" / (name + ".jsonl.gz")),
                                      "wt", encoding="utf-8", compresslevel=1)
                        for name in ("snapshots", "references", "status", "odometry")}
        self.events = (self.directory / "events" / "algorithm_events.jsonl").open("w")
        (self.directory / "events" / "external_events.csv").write_text(
            "event,sensor_time_s,uncertainty_s,source,notes\n", encoding="utf-8")
        self.manifest = {
            "schema": "v1-real-observations-1", "complete": False,
            "source_bag": rospy.get_param("~source_bag", ""),
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "omitted": ["per-component filter checkpoints", "input point clouds", "RViz markers"],
            "raw_estimates_before_display_filter": True, "ground_truth_available": False,
            "external_events_annotated": False,
            "completeness_scope": "Clean writer shutdown, no queue/write failures, paired published status/snapshot IDs. Does not imply every input scan was processed.",
            "notes": "Output estimates, not independent displacement ground truth. No bag is recorded.",
        }
        self.save_metadata()
        self.save_manifest(False)
        self.worker = threading.Thread(target=self.write_loop, daemon=True)
        self.worker.start()
        self.subscribers = [
            rospy.Subscriber("/deform/v1/detection_tiles", V1DetectionTile,
                             lambda msg: self.enqueue("snapshots", msg), queue_size=50, buff_size=128*1024*1024),
            rospy.Subscriber("/deform/v1/status", V1DetectionStatus,
                             lambda msg: self.enqueue("status", msg), queue_size=2000),
            rospy.Subscriber("/Odometry", Odometry,
                             lambda msg: self.enqueue("odometry", msg), queue_size=2000)]
        rospy.set_param("~run_directory", str(self.directory))
        rospy.set_param("~ready", True)
        rospy.on_shutdown(self.close)
        rospy.loginfo("[real_recorder] ready: %s", self.directory)

    def save_metadata(self):
        params = {name: rospy.get_param(name, {}) for name in
                  ("/deform_monitor_v1", "/deform_monitor_v1_rviz", "/mapping", "/preprocess", "/common")}
        (self.directory / "metadata" / "parameters.json").write_text(json.dumps(params, indent=2))
        package = pathlib.Path(__file__).resolve().parents[1]
        hashes = {}
        for folder in ("src/v1", "include/deform_monitor_v3/v1", "config/v1", "launch"):
            for path in sorted((package / folder).glob("*")):
                if path.is_file(): hashes[str(path.relative_to(package))] = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes["scripts/v1_real_recorder.py"] = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
        (self.directory / "metadata" / "source_sha256.json").write_text(json.dumps(hashes, indent=2))

    def enqueue(self, kind, msg):
        if kind == "snapshots" and msg.lod != 2:
            return
        receipt = {"receive_ros": plain(rospy.Time.now()), "receive_unix_s": time.time(),
                   "receive_wall_elapsed_s": time.monotonic() - self.started}
        with self.lock:
            if self.closed: return
            self.received[kind] += 1
            try:
                self.items.put_nowait((kind, msg, receipt))
                self.max_queue = max(self.max_queue, self.items.qsize())
            except queue.Full:
                self.overflow += 1
                self.error = "recorder queue overflow; run is incomplete"
                rospy.logerr("[real_recorder] %s", self.error)

    def emit(self, kind, record):
        self.streams[kind].write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        self.written[kind] += 1

    def process(self, kind, msg, receipt):
        if kind == "snapshots":
            key = (msg.common.session_id, msg.common.reference_epoch, id_key(msg.common.snapshot_id))
            if key in self.snapshot_ids:
                self.written["duplicate_snapshots"] += 1
                return
            self.snapshot_ids.add(key)
            epoch = (msg.common.session_id, msg.common.reference_epoch)
            if epoch not in self.reference_epochs:
                self.reference_epochs.add(epoch)
                self.emit("references", dict(receipt, common=plain(msg.common),
                         reference_support=plain(msg.reference_support)))
            self.emit(kind, dict(receipt, common=plain(msg.common), tile_key=msg.tile_key,
                      physical_deformation=plain(msg.physical_deformation),
                      detection_evidence=plain(msg.detection_evidence), interface_motion=plain(msg.interface_motion)))
        else:
            self.emit(kind, dict(receipt, message=plain(msg)))
            if kind == "status":
                if msg.last_attempt_published:
                    self.published_ids.add((msg.common.session_id, msg.common.reference_epoch,
                                            id_key(msg.common.snapshot_id)))
                if msg.last_attempt_input_point_count > 0:
                    index = msg.last_attempt_frame_index
                    if self.previous_input_index is not None and index > self.previous_input_index + 1:
                        self.input_status_index_gaps += index - self.previous_input_index - 1
                    self.previous_input_index = index
                phase = (msg.common.reference_epoch, msg.phase, msg.reference_frozen)
                if phase != self.previous_phase:
                    self.previous_phase = phase
                    self.events.write(json.dumps(dict(receipt, event="phase_change", phase=msg.phase,
                        reference_epoch=msg.common.reference_epoch, reference_frozen=msg.reference_frozen,
                        sensor_stamp=plain(msg.last_attempt_stamp))) + "\n")

    def write_loop(self):
        last_flush = time.monotonic()
        while True:
            item = self.items.get()
            try:
                if item is None: return
                if self.error is None:
                    self.process(*item)
                if time.monotonic() - last_flush > 5:
                    for stream in self.streams.values(): stream.flush()
                    self.events.flush()
                    self.save_manifest(False)
                    last_flush = time.monotonic()
            except Exception as error:
                self.error = repr(error)
                rospy.logerr("[real_recorder] write failed: %s", error)
            finally:
                self.items.task_done()

    def save_manifest(self, closed):
        self.manifest.update(received=dict(self.received), written=dict(self.written),
            queue_overflow=self.overflow, max_queue_depth=self.max_queue,
            input_status_index_gaps=self.input_status_index_gaps,
            published_snapshots_missing_records=len(self.published_ids-self.snapshot_ids),
            recorded_snapshots_missing_status=len(self.snapshot_ids-self.published_ids),
            error=self.error, closed_cleanly=closed)
        self.manifest["complete"] = bool(closed and self.error is None and not self.overflow
            and self.snapshot_ids and self.published_ids == self.snapshot_ids and not self.input_status_index_gaps)
        destination = self.directory / "metadata" / "manifest.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.manifest, indent=2, allow_nan=False))
        temporary.replace(destination)

    def close(self):
        with self.lock:
            if self.closed: return
            self.closed = True
        for sub in self.subscribers: sub.unregister()
        self.items.put(None)
        self.worker.join()
        for stream in self.streams.values(): stream.close()
        self.events.close()
        try: self.save_metadata()
        except Exception as error: rospy.logwarn("final metadata refresh: %s", error)
        self.save_manifest(True)
        rospy.loginfo("[real_recorder] closed: %s", self.directory)


if __name__ == "__main__":
    rospy.init_node("deform_v1_real_recorder")
    recorder = Recorder()
    rospy.spin()
