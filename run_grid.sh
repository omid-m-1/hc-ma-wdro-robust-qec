#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if (( $# < 3 || $# > 4 )); then
  echo 'Usage: bash run_grid.sh SUITE DISTANCES SEEDS [PRESET]' >&2
  exit 2
fi
suite="$1"
distances="$2"
seeds="$3"
preset="${4:-v10-main}"
profile="${PROFILE:-matched}"
workers="${WORKERS:-8}"
task_timeout="${TASK_TIMEOUT:-1800}"
python -u study.py plan --suite "$suite" --preset "$preset" --profile "$profile" --distances "$distances" --seeds "$seeds"
python -u study.py run --suite "$suite" --preset "$preset" --profile "$profile" --distances "$distances" --seeds "$seeds" --workers "$workers" --timeout-seconds "$task_timeout"
python -u study.py tables --suite "$suite" --preset "$preset" --profile "$profile"
echo 'RUN AND TABLE EXPORT COMPLETED'
