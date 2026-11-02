#!/usr/bin/env bash
set -euo pipefail

readonly WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly RUNNER="${WORKSPACE}/src/deform_monitor_bringup/scripts/run_icra2027_stationary_full_process.sh"
readonly MOTION_CONFIG="${WORKSPACE}/src/gazebo_test/Mid360_simulation_plugin/livox_laser_simulation/config/icra2027_stationary_full_process_motion.yaml"
readonly OUTPUT_ROOT="${DEFORM_SPEED_SWEEP_OUTPUT_ROOT:-${WORKSPACE}/experiment_output/stationary_speed_sweep}"
readonly REPEATS="${DEFORM_SPEED_SWEEP_REPEATS:-10}"
readonly RETRY_DELAY_S="${DEFORM_SPEED_SWEEP_RETRY_DELAY_S:-3}"
readonly GUI="${DEFORM_SPEED_SWEEP_GUI:-false}"
readonly RVIZ="${DEFORM_SPEED_SWEEP_RVIZ:-false}"
readonly ANGULAR_SPEED_RAD_S=0.0015
readonly SPEEDS=(1.0 3.0 5.0 10.0)
readonly TOTAL_RUNS=$((${#SPEEDS[@]} * REPEATS))
INTERRUPTED=false

number_token() {
  printf '%s' "${1//./p}"
}

print_matrix() {
  local run_id=1 repeat offset position speed
  printf 'run_id,speed_mm_s,repeat,block_position,scenario_id\n'
  for ((repeat = 1; repeat <= REPEATS; ++repeat)); do
    offset=$(((repeat - 1) % ${#SPEEDS[@]}))
    for ((position = 0; position < ${#SPEEDS[@]}; ++position)); do
      speed="${SPEEDS[$(((offset + position) % ${#SPEEDS[@]}))]}"
      printf '%03d,%s,%02d,%d,stationary_l%smmps_r%02d\n' \
        "${run_id}" "${speed}" "${repeat}" "$((position + 1))" \
        "$(number_token "${speed}")" "${repeat}"
      run_id=$((run_id + 1))
    done
  done
}

lookup_run() {
  local requested="$1"
  [[ "${requested}" =~ ^[0-9]+$ ]] || return 1
  requested=$((10#${requested}))
  ((requested >= 1 && requested <= TOTAL_RUNS)) || return 1
  print_matrix | awk -F, -v requested="${requested}" \
    'NR > 1 && ($1 + 0) == requested { print; exit }'
}

trial_command() {
  local run_id="$1" speed="$2" repeat="$3" batch_dir="$4"
  local scenario="stationary_l$(number_token "${speed}")mmps_r$(printf '%02d' "${repeat}")"
  local run_dir="${batch_dir}/runs/${scenario}"
  printf 'DEFORM_MONITOR_SKIP_BUILD=true DEFORM_MONITOR_RUN_DIRECTORY=%q %q gui:=%q rviz:=%q linear_speed_mm_s:=%q angular_speed_rad_s:=%q' \
    "${run_dir}" "${RUNNER}" "${GUI}" "${RVIZ}" "${speed}" \
    "${ANGULAR_SPEED_RAD_S}"
}

validate_run() {
  local run_dir="$1" requested_speed="$2"
  python3 -B - "${run_dir}" "${requested_speed}" <<'PY'
import csv
import json
import math
import pathlib
import sys

run_dir = pathlib.Path(sys.argv[1])
requested_speed = float(sys.argv[2])
manifest_path = run_dir / "run_manifest.json"
truth_path = run_dir / "truth" / "m01_m06_motion_truth.csv"
if not manifest_path.is_file() or not truth_path.is_file():
    raise SystemExit("missing manifest or m01-m06 truth")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("final_output_contract_status") != "COMPLETED":
    raise SystemExit("final output contract is not COMPLETED")
if manifest.get("finalization_outcome") != "COMPLETED":
    raise SystemExit("recorder finalization is not COMPLETED")
if manifest.get("required_artifact_errors"):
    raise SystemExit("required artifact errors are present")
if manifest.get("rosbag_recorded") is not False:
    raise SystemExit("rosbag_recorded must be false")
truth_contract = manifest.get("gazebo_motion_truth_contract", {})
if truth_contract.get("status") != "COMPLETED":
    raise SystemExit("Gazebo motion truth contract is not COMPLETED")

rows = list(csv.DictReader(truth_path.open(newline="", encoding="utf-8")))
if not rows or {row["model_name"] for row in rows} != {
        "m01", "m02", "m03", "m04", "m05", "m06"}:
    raise SystemExit("truth does not contain the complete m01-m06 set")
speeds = {float(row["linear_speed_mm_s"]) for row in rows}
if len(speeds) != 1 or not math.isclose(
        next(iter(speeds)), requested_speed, rel_tol=0.0, abs_tol=1.0e-12):
    raise SystemExit("recorded motion speed does not match the requested speed")
if max(float(row["experiment_elapsed_s"]) for row in rows) < 60.0 - 1.0e-9:
    raise SystemExit("truth timeline did not reach 60 seconds")
if any(abs(float(row["rotation_expected_rad"])) > 1.0e-12 for row in rows):
    raise SystemExit("a stationary speed-sweep fixture requested rotation")
PY
}

validate_pure_translation_config() {
  python3 -B - "${MOTION_CONFIG}" <<'PY'
import sys
import yaml

document = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
models = document.get("models", [])
if len(models) != 6 or {entry.get("model_name") for entry in models} != {
        "m01", "m02", "m03", "m04", "m05", "m06"}:
    raise SystemExit("stationary motion config does not contain exactly m01-m06")
if any(any(abs(float(value)) > 1.0e-12
           for value in entry.get("angular_axis", [])) for entry in models):
    raise SystemExit("stationary speed sweep requires zero angular axes")
PY
}

append_status() {
  local ledger="$1" run_id="$2" speed="$3" repeat="$4" scenario="$5"
  local attempts="$6" run_dir="$7"
  printf '%s,%03d,%s,%02d,%s,%d,SUCCESS,%s\n' \
    "$(date --iso-8601=seconds)" "${run_id}" "${speed}" "${repeat}" \
    "${scenario}" "${attempts}" "${run_dir}" \
    >>"${ledger}"
}

reset_failed_run() {
  local batch_dir="$1" run_dir="$2"
  case "${run_dir}" in
    "${batch_dir}/runs/"*) rm -rf -- "${run_dir}" ;;
    *) echo "Refusing to remove unsafe run directory: ${run_dir}" >&2; return 2 ;;
  esac
}

write_summary() {
  local batch_dir="$1" ledger="$2"
  python3 -B - "${batch_dir}" "${ledger}" "${TOTAL_RUNS}" <<'PY'
import csv
import json
import pathlib
import sys

batch_dir = pathlib.Path(sys.argv[1])
rows = list(csv.DictReader(open(sys.argv[2], newline="", encoding="utf-8")))
total = int(sys.argv[3])
successful = {}
for row in rows:
    successful[int(row["run_id"])] = row
payload = {
    "schema_version": 1,
    "status": "COMPLETED" if len(successful) == total else "INCOMPLETE",
    "scheduled_run_count": total,
    "successful_run_count": len(successful),
    "missing_run_ids": sorted(set(range(1, total + 1)) - set(successful)),
    "successful_runs": [successful[key] for key in sorted(successful)],
    "detection_outcome_used_as_success_gate": False,
    "rosbag_recorded": False,
}
(batch_dir / "batch_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    encoding="utf-8")
print("BATCH_{}: {}/{} successful runs; summary={}".format(
    payload["status"], len(successful), total,
    batch_dir / "batch_summary.json"))
raise SystemExit(0 if payload["status"] == "COMPLETED" else 1)
PY
}

usage() {
  cat <<'EOF'
Usage:
  run_icra2027_stationary_speed_sweep.sh
  run_icra2027_stationary_speed_sweep.sh --print-matrix
  run_icra2027_stationary_speed_sweep.sh --print-launch-command RUN_ID
EOF
}

run_batch() {
  (($# == 0)) || { usage >&2; return 2; }
  [[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid repeat count" >&2; return 2; }
  [[ "${RETRY_DELAY_S}" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    echo "Invalid retry delay" >&2; return 2; }
  [[ -x "${RUNNER}" ]] || { echo "Missing runner: ${RUNNER}" >&2; return 2; }
  validate_pure_translation_config

  local batch_dir="${OUTPUT_ROOT}/batch_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${batch_dir}/runs"
  print_matrix >"${batch_dir}/matrix.csv"
  local ledger="${batch_dir}/batch_status.csv"
  printf 'timestamp,run_id,speed_mm_s,repeat,scenario_id,attempt_count,status,run_directory\n' \
    >"${ledger}"

  echo "Building workspace once before ${TOTAL_RUNS} trials..."
  (cd "${WORKSPACE}" && source /opt/ros/noetic/setup.bash && \
    catkin_make -DCMAKE_BUILD_TYPE=Release)

  trap 'INTERRUPTED=true' INT TERM
  local line run_id speed repeat block_position scenario attempt run_dir
  while IFS= read -r line; do
    IFS=, read -r run_id speed repeat block_position scenario <<<"${line}"
    run_id=$((10#${run_id}))
    repeat=$((10#${repeat}))
    attempt=1
    run_dir="${batch_dir}/runs/${scenario}"
    while true; do
      if [[ "${INTERRUPTED}" == true ]]; then
        reset_failed_run "${batch_dir}" "${run_dir}"
        write_summary "${batch_dir}" "${ledger}" || true
        return 130
      fi
      reset_failed_run "${batch_dir}" "${run_dir}"
      echo "RUN ${run_id}/${TOTAL_RUNS}: ${scenario}, attempt ${attempt}"
      local runner_succeeded=false
      if DEFORM_MONITOR_SKIP_BUILD=true \
          DEFORM_MONITOR_RUN_DIRECTORY="${run_dir}" \
          "${RUNNER}" gui:="${GUI}" rviz:="${RVIZ}" \
          linear_speed_mm_s:="${speed}" \
          angular_speed_rad_s:="${ANGULAR_SPEED_RAD_S}"; then
        runner_succeeded=true
      fi
      if [[ "${INTERRUPTED}" == true ]]; then
        reset_failed_run "${batch_dir}" "${run_dir}"
        write_summary "${batch_dir}" "${ledger}" || true
        return 130
      fi
      if [[ "${runner_succeeded}" == true ]]; then
        if validate_run "${run_dir}" "${speed}"; then
          append_status "${ledger}" "${run_id}" "${speed}" "${repeat}" \
            "${scenario}" "${attempt}" "${run_dir}"
          echo "SUCCESS ${run_id}/${TOTAL_RUNS}: ${scenario} after ${attempt} attempt(s)"
          break
        fi
      fi
      echo "RETRY ${run_id}/${TOTAL_RUNS}: deleting incomplete output and retrying in ${RETRY_DELAY_S}s" >&2
      reset_failed_run "${batch_dir}" "${run_dir}"
      attempt=$((attempt + 1))
      sleep "${RETRY_DELAY_S}" || true
    done
  done < <(print_matrix | tail -n +2)

  write_summary "${batch_dir}" "${ledger}"
}

main() {
  case "${1:-}" in
    --print-matrix) print_matrix ;;
    --print-launch-command)
      [[ $# -eq 2 ]] || { usage >&2; return 2; }
      local line run_id speed repeat block_position scenario preview_dir
      line="$(lookup_run "$2")" || { echo "Invalid run id: $2" >&2; return 2; }
      IFS=, read -r run_id speed repeat block_position scenario <<<"${line}"
      preview_dir="${OUTPUT_ROOT}/PREVIEW"
      trial_command "$((10#${run_id}))" "${speed}" "$((10#${repeat}))" \
        "${preview_dir}"
      printf '\n'
      ;;
    -h|--help) usage ;;
    *) run_batch "$@" ;;
  esac
}

main "$@"
