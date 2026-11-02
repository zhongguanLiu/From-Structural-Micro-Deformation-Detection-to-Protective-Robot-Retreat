#!/usr/bin/env bash
set -euo pipefail
export COMPARE_METHODS_METHODS=m3c2
WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export COMPARE_METHODS_OUTPUT_ROOT="${M3C2_OUTPUT_ROOT:-${WORKSPACE}/experiment_output/m3c2_speed_sweep}"
export COMPARE_METHODS_MAX_ATTEMPTS="${COMPARE_METHODS_MAX_ATTEMPTS:-3}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_compare_methods_single_lidar_speed_sweep.sh" "$@"
