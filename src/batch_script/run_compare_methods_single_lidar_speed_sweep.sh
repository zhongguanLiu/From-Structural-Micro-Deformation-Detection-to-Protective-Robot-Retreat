#!/usr/bin/env bash
set -euo pipefail

readonly RESEARCH_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly COMPARE_WORKSPACE="${RESEARCH_WORKSPACE}/src/compare_methods"
readonly RUNNER="${COMPARE_WORKSPACE}/run_single_comparison.sh"
readonly OUTPUT_ROOT="${COMPARE_METHODS_OUTPUT_ROOT:-${RESEARCH_WORKSPACE}/experiment_output/compare_methods_single_lidar}"
readonly REPEATS="${COMPARE_METHODS_REPEATS:-10}"
readonly RETRY_DELAY_S="${COMPARE_METHODS_RETRY_DELAY_S:-3}"
read -r -a METHODS <<<"${COMPARE_METHODS_METHODS:-m_detector spatiotemporal_normals dynablox}"
readonly METHODS
for method in "${METHODS[@]}"; do
  case "${method}" in m_detector|spatiotemporal_normals|dynablox|m3c2) ;;
    *) echo "Unsupported method: ${method}" >&2; exit 2 ;; esac
done
readonly MAX_ATTEMPTS="${COMPARE_METHODS_MAX_ATTEMPTS:-0}"
readonly SPEEDS=(1 3 5 10)
readonly TOTAL_RUNS=$((${#METHODS[@]} * ${#SPEEDS[@]} * REPEATS))
INTERRUPTED=false

print_matrix() {
  local id=1 repeat method_offset speed_offset mi si method speed
  printf 'run_id,method,speed_mm_s,repeat,scenario_id\n'
  for ((repeat=1; repeat<=REPEATS; ++repeat)); do
    method_offset=$(((repeat - 1) % ${#METHODS[@]}))
    speed_offset=$(((repeat - 1) % ${#SPEEDS[@]}))
    for ((mi=0; mi<${#METHODS[@]}; ++mi)); do
      method="${METHODS[$(((method_offset + mi) % ${#METHODS[@]}))]}"
      for ((si=0; si<${#SPEEDS[@]}; ++si)); do
        speed="${SPEEDS[$(((speed_offset + si) % ${#SPEEDS[@]}))]}"
        printf '%03d,%s,%s,%02d,%s_%smmps_r%02d\n' \
          "${id}" "${method}" "${speed}" "${repeat}" \
          "${method}" "${speed}" "${repeat}"
        id=$((id + 1))
      done
    done
  done
}

safe_remove_attempt() {
  local batch_dir="$1" run_dir="$2"
  case "${run_dir}" in
    "${batch_dir}/runs/"*) rm -rf -- "${run_dir}" ;;
    *) echo "Refusing unsafe cleanup: ${run_dir}" >&2; return 2 ;;
  esac
}

write_summary() {
  local batch_dir="$1" ledger="$2"
  python3 -B - "${batch_dir}" "${ledger}" "${TOTAL_RUNS}" "${REPEATS}" <<'PY'
import csv, json, pathlib, sys
batch = pathlib.Path(sys.argv[1])
rows = list(csv.DictReader(open(sys.argv[2], newline="")))
total = int(sys.argv[3])
repeats = int(sys.argv[4])
ids = {int(row["run_id"]) for row in rows if row["status"] == "SUCCESS"}
payload = {
    "schema_version": 1,
    "status": "COMPLETED" if len(ids) == total else "INCOMPLETE",
    "scheduled_run_count": total,
    "successful_run_count": len(ids),
    "missing_run_ids": sorted(set(range(1, total + 1)) - ids),
    "methods": sorted({row["method"] for row in csv.DictReader(open(batch / "matrix.csv"))}),
    "speeds_mm_s": [1, 3, 5, 10],
    "repeats_per_condition": repeats,
    "single_front_lidar": True,
    "detection_outcome_used_as_success_gate": False,
    "rosbag_recorded": False,
}
(batch / "batch_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n")
print("BATCH_{} {}/{}".format(payload["status"], len(ids), total))
raise SystemExit(0 if payload["status"] == "COMPLETED" else 1)
PY
}

if [[ "${1:-}" == "--print-matrix" ]]; then
  print_matrix
  exit 0
fi
[[ $# -eq 0 ]] || { echo "Usage: $0 [--print-matrix]" >&2; exit 2; }
[[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ -x "${RUNNER}" ]] || { echo "Missing runner: ${RUNNER}" >&2; exit 2; }

if [[ "${COMPARE_METHODS_SKIP_MAIN_BUILD:-false}" != true ]]; then
echo "Building main workspace once..."
(cd "${RESEARCH_WORKSPACE}" && source /opt/ros/noetic/setup.bash && \
  catkin_make -DCMAKE_BUILD_TYPE=Release)
fi
echo "Building comparison workspace once..."
(source /opt/ros/noetic/setup.bash && \
  source "${RESEARCH_WORKSPACE}/devel/setup.bash" && \
  catkin_make -C "${COMPARE_WORKSPACE}" -DCMAKE_BUILD_TYPE=Release -j2)

readonly BATCH_DIRECTORY="${OUTPUT_ROOT}/batch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${BATCH_DIRECTORY}/runs"
print_matrix >"${BATCH_DIRECTORY}/matrix.csv"
readonly LEDGER="${BATCH_DIRECTORY}/batch_status.csv"
printf 'timestamp,run_id,method,speed_mm_s,repeat,scenario_id,attempt_count,status,run_directory\n' >"${LEDGER}"
trap 'INTERRUPTED=true' INT TERM

while IFS=, read -r run_id method speed repeat scenario; do
  attempt=1
  run_dir="${BATCH_DIRECTORY}/runs/${scenario}"
  while true; do
    if [[ "${INTERRUPTED}" == true ]]; then
      safe_remove_attempt "${BATCH_DIRECTORY}" "${run_dir}"
      write_summary "${BATCH_DIRECTORY}" "${LEDGER}" || true
      exit 130
    fi
    safe_remove_attempt "${BATCH_DIRECTORY}" "${run_dir}"
    echo "RUN $((10#${run_id}))/${TOTAL_RUNS} ${scenario} attempt=${attempt}"
    succeeded=false
    if "${RUNNER}" "${method}" "${speed}" "${run_dir}" \
        gui:=false rviz:=false; then
      succeeded=true
    fi
    if [[ "${INTERRUPTED}" == true ]]; then
      safe_remove_attempt "${BATCH_DIRECTORY}" "${run_dir}"
      write_summary "${BATCH_DIRECTORY}" "${LEDGER}" || true
      exit 130
    fi
    if [[ "${succeeded}" == true ]]; then
      printf '%s,%03d,%s,%s,%02d,%s,%d,SUCCESS,%s\n' \
        "$(date --iso-8601=seconds)" "$((10#${run_id}))" "${method}" \
        "${speed}" "$((10#${repeat}))" "${scenario}" "${attempt}" \
        "${run_dir}" >>"${LEDGER}"
      break
    fi
    safe_remove_attempt "${BATCH_DIRECTORY}" "${run_dir}"
    if (( MAX_ATTEMPTS > 0 && attempt >= MAX_ATTEMPTS )); then
      write_summary "${BATCH_DIRECTORY}" "${LEDGER}" || true
      echo "Infrastructure failed ${attempt} times; stopped without accepting missing detections as failure." >&2
      exit 3
    fi
    attempt=$((attempt + 1))
    sleep "${RETRY_DELAY_S}" || true
  done
done < <(print_matrix | tail -n +2)

write_summary "${BATCH_DIRECTORY}" "${LEDGER}"
