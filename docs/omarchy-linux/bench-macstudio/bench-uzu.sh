#!/usr/bin/env bash
# uzu engine harness: explicit warmup run + 3 recorded runs per model via `cli bench`.
# (`cli bench` additionally does its own internal 1-token warmup per invocation.)
set -euo pipefail
ROOT="${BENCH_ROOT:-$HOME/uzu-bench}"
CLI="$ROOT/uzu/target/release/cli"
mkdir -p "$ROOT/results"

find_model_dir() {  # registry entry dir -> checkpoint version dir
  local d
  d="$(find "$HOME/.cache/mirai/models/mirai" -maxdepth 2 -type d -name "$1" 2>/dev/null | head -1)"
  [[ -n "$d" ]] || return 1
  ls -d "$d"/*/ 2>/dev/null | head -1
}

D08="$(find_model_dir 'alibaba-qwen3.5-0.8b-mirai-mirai-m-4')"
D4B="$(find_model_dir 'alibaba-qwen3.5-4b-mirai-mirai-m-4')"
[[ -n "$D08" && -n "$D4B" ]] || { echo "registry model cache dirs not found" >&2; exit 1; }
echo "0.8B dir: $D08"
echo "4B dir:   $D4B"

ONLY="${ONLY:-}"
for M in "0.8B:$D08" "4B:$D4B"; do
  label="${M%%:*}"; dir="${M#*:}"
  if [[ -n "$ONLY" && "$ONLY" != *"$label"* ]]; then continue; fi
  "$ROOT/venv/bin/python" - "$ROOT/prompt.json" "$ROOT/task-$label.json" "$label" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
label = sys.argv[3]
task = {
    "identifier": "mac-studio-bench",
    "repo_id": "trymirai/Qwen3.5-0.8B-M" if label == "0.8B" else "trymirai/Qwen3.5-4B-M",
    "number_of_runs": 3,
    "tokens_limit": 128,
    "messages": [
        {"role": "system", "content": p["system"], "reasoning_content": None},
        {"role": "user", "content": p["user"], "reasoning_content": None},
    ],
    "greedy": True,
}
json.dump(task, open(sys.argv[2], "w"), indent=2)
PY
  echo "=== uzu warmup $label $(date -u +%T)"
  "$CLI" bench "$dir" "$ROOT/task-$label.json" "$ROOT/results/uzu-$label-warmup.json"
  echo "=== uzu recorded $label $(date -u +%T)"
  "$CLI" bench "$dir" "$ROOT/task-$label.json" "$ROOT/results/uzu-$label.json"
done
echo "uzu harness done"
