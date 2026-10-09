#!/usr/bin/env bash
# Run engine sessions until each engine x model has one stable session
# (acceptance per check-stable.py: 3 recorded runs agree on decode/prefill).
# State = presence of results/accepted-<engine>-<model>.json.
# Every attempt is archived in results/att-*; GPU snapshots in results/gpu-log.txt.
set -uo pipefail
ROOT="${BENCH_ROOT:-$HOME/uzu-bench}"
cd "$ROOT"
MAX_ROUNDS=${MAX_ROUNDS:-6}

run_eng() {  # eng model
  case "$1" in
    mlx)      ONLY="$2" ./venv/bin/python bench-mlx.py ;;
    llamacpp) ONLY="$2" ./venv/bin/python bench-llamacpp.py ;;
    uzu)      ONLY="$2" ./bench-uzu.sh ;;
  esac
}

rawfile() {  # eng model -> file the harness just wrote
  case "$1" in
    mlx|llamacpp) echo "results/$1.json" ;;
    uzu)          echo "results/uzu-$2.json" ;;
  esac
}

gpu_snap() {
  { echo "--- $(date -u +%FT%TZ) $*"
    sudo -n powermetrics --samplers gpu_power -i 1000 -n 2 2>/dev/null \
      | grep "GPU HW active residency"
  } >> results/gpu-log.txt
}

stable() { ./venv/bin/python check-stable.py "$1" "$2" >/dev/null 2>&1; }

for round in $(seq 1 "$MAX_ROUNDS"); do
  echo "=== round $round $(date -u +%T)"
  done_all=1
  for eng in mlx llamacpp uzu; do
    for model in 0.8B 4B; do
      acc="results/accepted-${eng}-${model}.json"
      [[ -f "$acc" ]] && continue
      done_all=0
      if ! run_eng "$eng" "$model" >/dev/null 2>&1; then
        echo "  $eng $model run FAILED"; gpu_snap "$eng $model round $round FAILED"; continue
      fi
      f="$(rawfile "$eng" "$model")"
      cp "$f" "results/att-${eng}-${model}-r${round}.json"
      gpu_snap "$eng $model round $round"
      if stable "$eng" "$f"; then
        cp "$f" "$acc"
        echo "  $eng $model ACCEPTED (round $round)"
      else
        echo "  $eng $model unstable (round $round, attempt archived)"
      fi
    done
  done
  if [[ $done_all -eq 1 ]]; then echo "all accepted after round $round"; break; fi
done
ls results/accepted-*.json 2>/dev/null
[[ $(ls results/accepted-*.json 2>/dev/null | wc -l) -eq 6 ]]
