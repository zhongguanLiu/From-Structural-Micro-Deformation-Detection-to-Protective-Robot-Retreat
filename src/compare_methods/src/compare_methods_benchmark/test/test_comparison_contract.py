#!/usr/bin/env python3
import collections
import csv
import io
import math
import pathlib
import subprocess
import xml.etree.ElementTree as ET


ROOT = pathlib.Path(__file__).resolve().parents[4]
COMPARE = ROOT / "compare_methods"
LAUNCH = (COMPARE / "src/compare_methods_benchmark/launch" /
          "stationary_single_lidar.launch")
BATCH = ROOT / "batch_script/run_compare_methods_single_lidar_speed_sweep.sh"


def main():
    root = ET.parse(LAUNCH).getroot()
    simulation = root.find("include")
    forwarded = {node.get("name"): node.get("value")
                 for node in simulation.findall("arg")}
    assert forwarded["enable_rear_lidar"] == "false"
    assert forwarded["record_truth"] == "false"
    assert forwarded["world"] == "$(arg world)"
    assert forwarded["linear_speed_mm_s"] == "$(arg linear_speed_mm_s)"

    recorder = root.findall("node")[0]
    recorder_params = {node.get("name"): node.get("value")
                       for node in recorder.findall("param")}
    assert recorder_params["input_metadata_stride"] == "1"
    transform = next(node for node in recorder.findall("rosparam")
                     if node.get("param") ==
                     "camera_init_origin_gazebo_xyzyaw")
    values = [float(value) for value in transform.text.strip(" []\n").split(",")]
    expected = (-3.7902 + math.cos(2.0688) * 0.2801,
                12.1308 + math.sin(2.0688) * 0.2801,
                1.505898 + 0.616928878, 2.0688)
    assert len(values) == 4
    assert max(abs(a - b) for a, b in zip(values, expected)) < 1e-9

    output = subprocess.check_output(
        ["bash", str(BATCH), "--print-matrix"], text=True)
    rows = list(csv.DictReader(io.StringIO(output)))
    assert len(rows) == 120
    counts = collections.Counter((row["method"], row["speed_mm_s"])
                                 for row in rows)
    assert set(counts.values()) == {10}
    assert len({row["scenario_id"] for row in rows}) == 120

    m3c2_output = subprocess.check_output(
        ["bash", str(ROOT / "batch_script/run_m3c2_speed_sweep.sh"), "--print-matrix"], text=True)
    m3c2_rows = list(csv.DictReader(io.StringIO(m3c2_output)))
    assert len(m3c2_rows) == 40
    assert {row["method"] for row in m3c2_rows} == {"m3c2"}
    assert collections.Counter(row["speed_mm_s"] for row in m3c2_rows) == {
        "1": 10, "3": 10, "5": 10, "10": 10}
    assert len({row["scenario_id"] for row in m3c2_rows}) == 40

    detector_files = [
        COMPARE / "src/M-detector/launch/fastlio_mid360.launch",
        COMPARE / "src/dynamic_object_detection/launch/fastlio_mid360.launch",
        COMPARE / "src/dynablox/dynablox_ros/launch/fastlio_mid360.launch",
        COMPARE / "src/m3c2_baseline/launch/fastlio_mid360.launch",
    ]
    detector_text = "\n".join(path.read_text() for path in detector_files)
    assert not any(name in detector_text for name in
                   ("m01", "m02", "m03", "m04", "m05", "m06"))
    recorder_text = (COMPARE / "src/compare_methods_benchmark/scripts" /
                     "benchmark_recorder.py").read_text()
    assert "motion_start_after_input_frames" in recorder_text
    assert 'call["reset"]' in recorder_text
    assert 'for operation in ("reload", "start")' in recorder_text
    for field in ("truth_experiment_elapsed_s", "truth_motion_elapsed_s",
                  "truth_state", "source_frame_id",
                  "truth_association_max_delta_s", "source_stamp_coverage",
                  "ros_parameters.json.gz", "fastlio_odometry.csv.gz",
                  "world_file_sha256", "motion_config_file_sha256"):
        assert field in recorder_text
    assert "stamp < snapshots[0][0]" in recorder_text
    cluster_header = (COMPARE / "src/M-detector/include/m-detector" /
                      "DynObjCluster.h").read_text()
    cluster_source = (COMPARE / "src/M-detector/src" /
                      "DynObjCluster.cpp").read_text()
    assert "std::unordered_map<int, Point_Cloud> umap" in cluster_header
    assert "umap.resize(GridMapsize)" not in cluster_source
    print("comparison contract: PASS (original 120 + independent M3C2 40 runs, no truth leakage)")


if __name__ == "__main__":
    main()
