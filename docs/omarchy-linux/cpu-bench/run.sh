#!/bin/bash
# CPU only, no GPU. Usage: run.sh OUTDIR PROMPT_WORDS TOKENS RUNS
# env: UZU_CLI (built cli binary), UZU_MODEL (downloaded model directory), LLAMA_CLI (llama.cpp CPU llama-cli), GGUF (model file),
#      UZU_SKIP=1 to run only llama.cpp against an existing OUTDIR (reuses its prompt.txt), UZU_CAP (seconds, default 1500)
set -u
OUT=${1:?outdir}
WORDS=${2:-100}
TOK=${3:-32}
RUNS=${4:-2}
H=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
[ -n "${UZU_SKIP:-}" ] || python3 "$H/make-task.py" "$OUT" "$WORDS" "$TOK" "$RUNS" > "$OUT/task.txt"
{
  echo "== host $(hostname -s) kernel $(uname -r) cpus $(nproc) $(date -u +%FT%TZ)"
  echo "load before: $(cut -d' ' -f1-3 /proc/loadavg)"
  if [ -z "${UZU_SKIP:-}" ]; then
    echo "== uzu CPU backend ($UZU_CLI)"
    T0=$(date +%s)
    timeout "${UZU_CAP:-1500}" "$UZU_CLI" bench "$UZU_MODEL" "$OUT/task.json" "$OUT/uzu-result.json" > "$OUT/uzu.log" 2>&1
    echo "uzu rc=$? wall_s=$(( $(date +%s) - T0 ))"
  fi
  for T in 1 8; do
    echo "== llama.cpp CPU, $T thread(s)"
    T0=$(date +%s)
    timeout 600 "$LLAMA_CLI" -m "$GGUF" -f "$OUT/prompt.txt" -n "$TOK" -t "$T" --temp 0 > "$OUT/llama-t$T.out" 2> "$OUT/llama-t$T.log" < /dev/null
    echo "llama t=$T rc=$? wall_s=$(( $(date +%s) - T0 ))"
    grep -E "prompt eval time|eval time|load time|llama_print_timings" "$OUT/llama-t$T.log" | cut -c1-170
  done
  echo "load after: $(cut -d' ' -f1-3 /proc/loadavg)"
} > "$OUT/summary-llama.txt" 2>&1
cat "$OUT/summary-llama.txt"
