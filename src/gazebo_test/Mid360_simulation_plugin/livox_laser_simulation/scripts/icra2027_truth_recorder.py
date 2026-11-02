#!/usr/bin/env python3
"""Record ICRA2027 motion ground truth messages as analysis-ready CSV rows."""

import csv
import json
import os
from datetime import datetime

import rospy
from std_msgs.msg import String


CSV_FIELDS = [
    "schema_version", "state", "sim_time_s", "experiment_elapsed_s",
    "waiting_elapsed_s", "motion_elapsed_s", "model_motion_elapsed_s",
    "motion_mode", "linear_speed_mm_s", "angular_speed_rad_s",
    "laser_retro_id", "model_name",
    "position_x_m", "position_y_m", "position_z_m",
    "orientation_x", "orientation_y", "orientation_z", "orientation_w",
    "aabb_min_x_m", "aabb_min_y_m", "aabb_min_z_m",
    "aabb_max_x_m", "aabb_max_y_m", "aabb_max_z_m",
    "translation_x_m", "translation_y_m", "translation_z_m",
    "expected_translation_x_m", "expected_translation_y_m",
    "expected_translation_z_m",
    "translation_magnitude_m", "translation_expected_m", "translation_error_m",
    "rotation_angle_rad", "rotation_expected_rad", "rotation_error_rad",
    "surface_rms_displacement_m", "surface_max_displacement_m", "surface_point_count",
]


def _component(sequence, index, default=0.0):
    try:
        return sequence[index]
    except (IndexError, TypeError):
        return default


def rows_from_truth(payload):
    """Flatten one plugin truth payload into one CSV row per controlled model."""
    document = json.loads(payload) if isinstance(payload, str) else payload
    rows = []
    for model in document.get("models", []):
        position = model.get("position", [])
        orientation = model.get("orientation", [])
        aabb_min = model.get("aabb_min", [])
        aabb_max = model.get("aabb_max", [])
        translation = model.get("translation", [])
        expected_translation = model.get("expected_translation_vector", [])
        rows.append({
            "schema_version": document.get("schema_version", 1),
            "state": document.get("state", ""),
            "sim_time_s": document.get("sim_time", 0.0),
            "experiment_elapsed_s": document.get("experiment_elapsed", 0.0),
            "waiting_elapsed_s": document.get("waiting_elapsed", 0.0),
            "motion_elapsed_s": document.get("motion_elapsed", 0.0),
            "model_motion_elapsed_s": model.get("motion_elapsed", 0.0),
            "motion_mode": document.get("motion_mode", ""),
            "linear_speed_mm_s": document.get("linear_speed_mm_s", 0.0),
            "angular_speed_rad_s": document.get("angular_speed_rad_s", 0.0),
            "laser_retro_id": model.get("laser_retro_id", ""),
            "model_name": model.get("model_name", ""),
            "position_x_m": _component(position, 0),
            "position_y_m": _component(position, 1),
            "position_z_m": _component(position, 2),
            "orientation_x": _component(orientation, 0),
            "orientation_y": _component(orientation, 1),
            "orientation_z": _component(orientation, 2),
            "orientation_w": _component(orientation, 3, 1.0),
            "aabb_min_x_m": _component(aabb_min, 0, ""),
            "aabb_min_y_m": _component(aabb_min, 1, ""),
            "aabb_min_z_m": _component(aabb_min, 2, ""),
            "aabb_max_x_m": _component(aabb_max, 0, ""),
            "aabb_max_y_m": _component(aabb_max, 1, ""),
            "aabb_max_z_m": _component(aabb_max, 2, ""),
            "translation_x_m": _component(translation, 0),
            "translation_y_m": _component(translation, 1),
            "translation_z_m": _component(translation, 2),
            "expected_translation_x_m": _component(expected_translation, 0),
            "expected_translation_y_m": _component(expected_translation, 1),
            "expected_translation_z_m": _component(expected_translation, 2),
            "translation_magnitude_m": model.get("translation_magnitude", 0.0),
            "translation_expected_m": model.get("translation_expected", 0.0),
            "translation_error_m": model.get("translation_error", 0.0),
            "rotation_angle_rad": model.get("rotation_angle", 0.0),
            "rotation_expected_rad": model.get("rotation_expected", 0.0),
            "rotation_error_rad": model.get("rotation_error", 0.0),
            "surface_rms_displacement_m": model.get("surface_rms_displacement", 0.0),
            "surface_max_displacement_m": model.get("surface_max_displacement", 0.0),
            "surface_point_count": model.get("surface_point_count", 0),
        })
    return rows


class TruthRecorder:
    def __init__(self):
        topic = rospy.get_param("~truth_topic", "/icra2027_motion/truth")
        output_directory = os.path.abspath(os.path.expanduser(
            rospy.get_param("~output_directory", "/tmp/icra2027_truth")
        ))
        os.makedirs(output_directory, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(output_directory, "icra2027_motion_truth_{}.csv".format(stamp))
        self.stream = open(self.path, "w", newline="")
        self.writer = csv.DictWriter(self.stream, fieldnames=CSV_FIELDS)
        self.writer.writeheader()
        self.stream.flush()
        self.subscriber = rospy.Subscriber(topic, String, self._callback, queue_size=100)
        rospy.on_shutdown(self.close)
        rospy.loginfo("ICRA2027 truth CSV: %s", self.path)

    def _callback(self, message):
        try:
            for row in rows_from_truth(message.data):
                self.writer.writerow(row)
            self.stream.flush()
        except (ValueError, TypeError, OSError) as error:
            rospy.logerr_throttle(2.0, "Could not record ICRA2027 truth: %s", error)

    def close(self):
        if getattr(self, "stream", None) is not None and not self.stream.closed:
            self.stream.flush()
            self.stream.close()


def main():
    rospy.init_node("icra2027_truth_recorder")
    TruthRecorder()
    rospy.spin()


if __name__ == "__main__":
    main()
