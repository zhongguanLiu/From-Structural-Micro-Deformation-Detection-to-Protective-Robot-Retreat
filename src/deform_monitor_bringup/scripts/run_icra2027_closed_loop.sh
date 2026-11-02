#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT_ROOT=${DEFORM_MONITOR_OUTPUT_ROOT:-${WORKSPACE}/experiment_output}
RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_DIRECTORY=${OUTPUT_ROOT}/closed_loop/icra2027_${RUN_ID}

# Interactive GUI runs wait after the first scene is visible. Batch runs
# remain automatic; either behavior can be selected explicitly.
WAIT_FOR_OPERATOR=auto
INTERACTIVE_GUI=true
for argument in "$@"; do
  case "${argument}" in
    gui:=false|gui:=0|rviz:=false|rviz:=0) INTERACTIVE_GUI=false ;;
    wait_for_operator:=*) WAIT_FOR_OPERATOR=${argument#*:=} ;;
  esac
done
if [[ "${WAIT_FOR_OPERATOR}" == auto ]]; then
  WAIT_FOR_OPERATOR=false
  if [[ -t 0 && -t 1 && "${INTERACTIVE_GUI}" == true ]]; then
    WAIT_FOR_OPERATOR=true
  fi
fi
case "${WAIT_FOR_OPERATOR}" in
  true|1) WAIT_FOR_OPERATOR=true ;;
  false|0) WAIT_FOR_OPERATOR=false ;;
  *) echo "wait_for_operator must be true or false" >&2; exit 2 ;;
esac
if [[ "${WAIT_FOR_OPERATOR}" == true && ! -t 0 ]]; then
  echo "Manual start needs an interactive terminal; use wait_for_operator:=false for automation." >&2
  exit 2
fi

source /opt/ros/noetic/setup.bash
cd "${WORKSPACE}"

# Always make incrementally so launched binaries match current source/config.
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash

mkdir -p "${RUN_DIRECTORY}" "${RUN_DIRECTORY}/ros_logs"
export ROS_LOG_DIR="${RUN_DIRECTORY}/ros_logs"

python3 "${WORKSPACE}/src/deform_monitor_bringup/scripts/experiment_source_snapshot.py" \
  "${WORKSPACE}" "${RUN_DIRECTORY}" "$@"

# Own one private ROS graph for the complete experiment.  Sharing the user's
# default master allows an unrelated Gazebo or rosbag --clock process to inject
# a second /clock stream; that makes FAST-LIO timestamps run backwards and can
# irreversibly corrupt its state.  An optional fixed port is useful only for
# deliberate debugging; normal runs select a free loopback port.
if [[ -n "${DEFORM_MONITOR_ROS_MASTER_PORT:-}" ]]; then
  ROS_MASTER_PORT="${DEFORM_MONITOR_ROS_MASTER_PORT}"
  if [[ ! "${ROS_MASTER_PORT}" =~ ^[0-9]+$ ]] \
      || (( ROS_MASTER_PORT < 1024 || ROS_MASTER_PORT > 65535 )); then
    echo "Invalid DEFORM_MONITOR_ROS_MASTER_PORT: ${ROS_MASTER_PORT}" >&2
    exit 2
  fi
else
  ROS_MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')
fi
export ROS_MASTER_URI="http://127.0.0.1:${ROS_MASTER_PORT}"
export ROS_HOSTNAME=127.0.0.1
unset ROS_IP || true

ROS_MASTER_PID=""
OPERATOR_PID=""
cleanup_ros_master() {
  if [[ -n "${OPERATOR_PID}" ]]; then
    kill -TERM "${OPERATOR_PID}" 2>/dev/null || true
    wait "${OPERATOR_PID}" 2>/dev/null || true
    OPERATOR_PID=""
  fi
  if [[ -n "${ROS_MASTER_PID}" ]] && kill -0 "${ROS_MASTER_PID}" 2>/dev/null; then
    kill -INT "${ROS_MASTER_PID}" 2>/dev/null || true
    wait "${ROS_MASTER_PID}" 2>/dev/null || true
  fi
  ROS_MASTER_PID=""
}
trap cleanup_ros_master EXIT INT TERM
roscore -p "${ROS_MASTER_PORT}" \
  >"${RUN_DIRECTORY}/roscore_console.log" 2>&1 &
ROS_MASTER_PID=$!
MASTER_READY=false
for _attempt in $(seq 1 100); do
  if rosparam list >/dev/null 2>&1; then
    MASTER_READY=true
    break
  fi
  if ! kill -0 "${ROS_MASTER_PID}" 2>/dev/null; then
    break
  fi
  sleep 0.1
done
if [[ "${MASTER_READY}" != true ]]; then
  echo "Private ROS master failed to start on port ${ROS_MASTER_PORT}" >&2
  exit 3
fi

echo "ICRA2027 closed-loop output: ${RUN_DIRECTORY}"
echo "Isolated ROS master: ${ROS_MASTER_URI}"
echo "No rosbag recorder is started by this script."
echo "Terminal shows mission milestones only; full console is saved in ros_logs/roslaunch_console.log."
echo "Do not press Ctrl+C until RUN_COMPLETE is shown."

# A diagnostic run may explicitly disable the structured experiment recorder.
# In that case no run_manifest.json is expected, but ROS logs still have to be
# scrubbed.  Formal/default runs keep the strict manifest requirement.
RECORD_EXPERIMENT_ENABLED=true
for argument in "$@"; do
  case "${argument}" in
    record_experiment:=false|record_experiment:=0)
      RECORD_EXPERIMENT_ENABLED=false
      ;;
    record_experiment:=true|record_experiment:=1)
      RECORD_EXPERIMENT_ENABLED=true
      ;;
  esac
done

if [[ "${WAIT_FOR_OPERATOR}" == true ]]; then
  echo "Preparing Gazebo/RViz; wait for READY_TO_RECORD before pressing Enter."
  python3 "${WORKSPACE}/src/deform_monitor_bringup/scripts/icra2027_closed_loop_mission.py" \
    --operator-start &
  OPERATOR_PID=$!
fi

LAUNCH_STATUS=0
set +e
# Interrupted runs still need time to flush numeric streams and finalize
# diagnostics. ROS's default 15 s SIGINT grace killed the recorder mid-analysis.
roslaunch --sigint-timeout=180 --sigterm-timeout=30 deform_monitor_bringup icra2027_closed_loop.launch \
  mission_output_directory:="${RUN_DIRECTORY}" wait_for_operator:="${WAIT_FOR_OPERATOR}" "$@" 2>&1 \
  | tee "${RUN_DIRECTORY}/ros_logs/roslaunch_console.log" \
  | awk '/\[closed_loop\]/ || /\[ERROR\]/ || /\[FATAL\]/ { print; fflush(); }'
LAUNCH_STATUS=${PIPESTATUS[0]}
set -e

cleanup_ros_master

# roslaunch writes the complete process environment into its debug log.  Run
# credential scrubbing only after every logger has closed, so atomic file
# replacement cannot detach a live writer or lose the shutdown tail.
SCRUB_STATUS=0
SCRUB_ARGUMENTS=(--run-directory "${RUN_DIRECTORY}")
if [[ "${RECORD_EXPERIMENT_ENABLED}" == false ]]; then
  SCRUB_ARGUMENTS+=(--allow-missing-manifest)
fi
python3 "${WORKSPACE}/src/deform_monitor_bringup/scripts/ros_log_scrubber.py" \
  "${SCRUB_ARGUMENTS[@]}" || SCRUB_STATUS=$?

if [[ -f "${RUN_DIRECTORY}/run_manifest.json" ]] && (( SCRUB_STATUS == 0 )); then
  echo "RUN_COMPLETE: valid manifest saved at ${RUN_DIRECTORY}/run_manifest.json"
elif [[ "${RECORD_EXPERIMENT_ENABLED}" == false ]]; then
  echo "DIAGNOSTIC_COMPLETE: recorder was disabled; no manifest was requested."
else
  echo "RUN_INCOMPLETE: missing or failed final manifest; do not use this run for metrics." >&2
fi

if (( LAUNCH_STATUS != 0 )); then
  exit "${LAUNCH_STATUS}"
fi
exit "${SCRUB_STATUS}"
