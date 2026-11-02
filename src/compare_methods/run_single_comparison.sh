#!/usr/bin/env bash
set -euo pipefail

readonly RESEARCH_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly COMPARE_WORKSPACE="${RESEARCH_WORKSPACE}/src/compare_methods"

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 METHOD SPEED_MM_S OUTPUT_DIRECTORY [gui:=false] [rviz:=false]" >&2
  exit 2
fi
readonly METHOD="$1"
readonly SPEED_MM_S="$2"
readonly RUN_DIRECTORY="$3"
shift 3
case "${METHOD}" in
  m_detector|spatiotemporal_normals|dynablox|m3c2) ;;
  *) echo "Unsupported method: ${METHOD}" >&2; exit 2 ;;
esac
[[ "${SPEED_MM_S}" =~ ^(1|3|5|10)(\.0+)?$ ]] || {
  echo "Speed must be 1, 3, 5, or 10 mm/s" >&2; exit 2; }

source /opt/ros/noetic/setup.bash
source "${RESEARCH_WORKSPACE}/devel/setup.bash"
source "${COMPARE_WORKSPACE}/devel/setup.bash" --extend

if [[ -e "${RUN_DIRECTORY}/run_manifest.json" || -e "${RUN_DIRECTORY}/m3c2/frames.csv.gz" ]]; then
  echo "Refusing to overwrite an existing experiment: ${RUN_DIRECTORY}" >&2
  exit 2
fi

mkdir -p "${RUN_DIRECTORY}/ros_logs"
export ROS_LOG_DIR="${RUN_DIRECTORY}/ros_logs"
ROS_MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')
export ROS_MASTER_URI="http://127.0.0.1:${ROS_MASTER_PORT}"
export ROS_HOSTNAME=127.0.0.1
unset ROS_IP || true

ROS_MASTER_PID=""
cleanup() {
  if [[ -n "${ROS_MASTER_PID}" ]] && kill -0 "${ROS_MASTER_PID}" 2>/dev/null; then
    kill -INT "${ROS_MASTER_PID}" 2>/dev/null || true
    wait "${ROS_MASTER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
roscore -p "${ROS_MASTER_PORT}" >"${RUN_DIRECTORY}/roscore.log" 2>&1 &
ROS_MASTER_PID=$!
for _ in $(seq 1 100); do
  rosparam list >/dev/null 2>&1 && break
  kill -0 "${ROS_MASTER_PID}" 2>/dev/null || exit 3
  sleep 0.1
done

echo "COMPARE_RUN_START method=${METHOD} speed=${SPEED_MM_S}mm/s output=${RUN_DIRECTORY}"
set +e
roslaunch compare_methods_benchmark stationary_single_lidar.launch \
  method:="${METHOD}" linear_speed_mm_s:="${SPEED_MM_S}" \
  output_directory:="${RUN_DIRECTORY}" "$@" 2>&1 \
  | tee "${RUN_DIRECTORY}/roslaunch_console.log" \
  | awk '/COMPARE_RUN_|COMPARISON_RUN_COMPLETE|\[ERROR\]|\[FATAL\]|process has died/ {print; fflush()}'
LAUNCH_STATUS=${PIPESTATUS[0]}
set -e
cleanup

python3 -B - "${RUN_DIRECTORY}" "${METHOD}" "${SPEED_MM_S}" <<'PY'
import csv
import gzip
import json
import math
import pathlib
import sys

run = pathlib.Path(sys.argv[1])
method = sys.argv[2]
speed = float(sys.argv[3])
manifest = json.loads((run / "run_manifest.json").read_text())
if manifest.get("status") != "COMPLETED":
    raise SystemExit("run manifest is not COMPLETED")
if manifest.get("method") != method or not manifest.get("single_front_lidar"):
    raise SystemExit("method or single-LiDAR contract mismatch")
if manifest.get("rosbag_recorded") is not False:
    raise SystemExit("rosbag must remain disabled")
if not manifest.get("ros_parameters_recorded"):
    raise SystemExit("ROS parameter provenance is missing")
if not manifest.get("fastlio_input") or manifest.get("motion_start_error"):
    raise SystemExit("FAST-LIO or motion-start contract failed")
if manifest.get("object_names_used_by_detector") is not False:
    raise SystemExit("truth leaked into detector contract")
if method == "m3c2" and not manifest.get("reference_ready_before_motion"):
    raise SystemExit("M3C2 reference was not ready")
if not math.isclose(float(manifest["truth_speed_mm_s"]), speed, abs_tol=1e-12):
    raise SystemExit("truth speed mismatch")
if float(manifest["truth_final_experiment_elapsed_s"]) < 60.0 - 1e-9:
    raise SystemExit("truth timeline is incomplete")
with gzip.open(run / "configuration/ros_parameters.json.gz", "rt") as stream:
    parameters = json.load(stream)
merger = parameters.get("dual_livox_merge", {})
if (merger.get("secondary_enabled") is not False or
        not math.isclose(float(merger.get("configured_min_range_m", -1)), 0.5) or
        not math.isclose(float(merger.get("configured_max_range_m", -1)), 20.0)):
    raise SystemExit("single-LiDAR range contract mismatch")
counts = manifest.get("counts", {})
for name in ("truth_messages", "input_frames", "detection_frames",
             "runtime_frames", "odometry_messages"):
    if int(counts.get(name, 0)) < 10:
        raise SystemExit("insufficient {} records".format(name))
with gzip.open(run / "truth/motion_truth.jsonl.gz", "rt") as stream:
    truth = [json.loads(line) for line in stream]
final = truth[-1] if truth else None
if final is None or final.get("state") != "COMPLETED":
    raise SystemExit("truth stream has no COMPLETED record")
expected_models = {"m01", "m02", "m03", "m04", "m05", "m06"}
previous_sim_time = None
previous_experiment_elapsed = None
maximum_truth_gap = 0.0
for document in truth:
    if document.get("schema_version") != 2:
        raise SystemExit("truth schema mismatch")
    if {item["model_name"] for item in document["models"]} != expected_models:
        raise SystemExit("truth model set mismatch")
    sim_time = float(document["sim_time"])
    elapsed = float(document["experiment_elapsed"])
    if previous_sim_time is not None:
        if sim_time < previous_sim_time or elapsed < previous_experiment_elapsed:
            raise SystemExit("truth timeline is not monotonic")
        maximum_truth_gap = max(maximum_truth_gap, sim_time - previous_sim_time)
    previous_sim_time = sim_time
    previous_experiment_elapsed = elapsed
    for item in document["models"]:
        if (float(item["translation_error"]) > 1e-7 or
                float(item["rotation_angle"]) > 1e-8 or
                float(item["rotation_error"]) > 1e-8):
            raise SystemExit("invalid truth trajectory for " + item["model_name"])
if maximum_truth_gap > 0.5:
    raise SystemExit("truth sampling gap exceeded 0.5 s")
if float(truth[0]["experiment_elapsed"]) > 0.5:
    raise SystemExit("truth stream missed the experiment baseline start")
expected_translation = speed * 40.0 / 1000.0
for item in final["models"]:
    if not math.isclose(float(item["translation_magnitude"]),
                        expected_translation, abs_tol=1e-8):
        raise SystemExit("truth displacement mismatch for " + item["model_name"])
    if float(item["translation_error"]) > 1e-8 or float(item["rotation_angle"]) > 1e-8:
        raise SystemExit("truth motion contract mismatch for " + item["model_name"])
visible_counts = {name: 0 for name in
                  ("m01", "m02", "m03", "m04", "m05", "m06")}
phase_visible = {
    phase: {name: 0 for name in visible_counts}
    for phase in ("baseline", "motion", "hold")}
with gzip.open(run / "runtime/input_frames.csv.gz", "rt", newline="") as stream:
    for row in csv.DictReader(stream):
        if row["group_name"] in visible_counts and row["truth_sim_time_s"]:
            name = row["group_name"]
            count = int(row["point_count"])
            visible_counts[name] += count
            elapsed = float(row["truth_experiment_elapsed_s"])
            phase = ("baseline" if elapsed < 10.0 else
                     "motion" if elapsed < 50.0 else "hold")
            phase_visible[phase][name] += count
            if float(row["truth_time_delta_s"]) > 0.5:
                raise SystemExit("input/truth timestamp association exceeded 0.5 s")
if any(count == 0 for count in visible_counts.values()):
    raise SystemExit("one or more m01-m06 targets had no registered input points")
for phase, counts_by_model in phase_visible.items():
    if any(count == 0 for count in counts_by_model.values()):
        raise SystemExit("one or more m01-m06 targets were invisible in " + phase)
with gzip.open(run / "detections/detection_frames.csv.gz", "rt", newline="") as stream:
    for row in csv.DictReader(stream):
        if row["truth_sim_time_s"] and float(row["truth_time_delta_s"]) > 0.5:
            raise SystemExit("detection/truth timestamp association exceeded 0.5 s")
if manifest.get("source_frame_ids") != {
        "input": ["camera_init"], "detection": ["camera_init"]}:
    raise SystemExit("comparison output frame contract mismatch")
coverage = manifest.get("source_stamp_coverage", {})
final_sim_time = float(final["sim_time"])
for stream_name in ("input", "detection", "runtime", "odometry"):
    span = coverage.get(stream_name, {})
    first = span.get("first_source_stamp_s")
    last = span.get("last_source_stamp_s")
    if first is None or last is None or float(last) - float(first) < 55.0:
        raise SystemExit(stream_name + " stream did not cover the experiment")
    if float(last) < final_sim_time - 2.5:
        raise SystemExit(stream_name + " stream ended before the experiment")
for relative in ("runtime/input_frames.csv.gz",
                 "runtime/algorithm_runtime.csv.gz",
                 "runtime/fastlio_odometry.csv.gz",
                 "detections/detection_frames.csv.gz"):
    with gzip.open(run / relative, "rt", newline="") as stream:
        if not next(csv.reader(stream), None):
            raise SystemExit("empty CSV: " + relative)
PY

if [[ "${METHOD}" == m3c2 ]]; then
  python3 -B "${COMPARE_WORKSPACE}/src/m3c2_baseline/scripts/check_run.py" "${RUN_DIRECTORY}"
fi

if (( LAUNCH_STATUS != 0 )); then
  exit "${LAUNCH_STATUS}"
fi
rm -rf -- "${RUN_DIRECTORY}/ros_logs"
gzip -f "${RUN_DIRECTORY}/roslaunch_console.log" "${RUN_DIRECTORY}/roscore.log"
echo "COMPARE_RUN_COMPLETE method=${METHOD} speed=${SPEED_MM_S}mm/s"
