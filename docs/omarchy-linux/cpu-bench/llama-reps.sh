#!/bin/bash
# Usage: llama-reps.sh PROMPT_FILE TOKENS REPS    env: LLAMA_CLI, GGUF. Prints one line per run: threads, prompt t/s, generation t/s.
PROMPT=$1
TOK=${2:-32}
REPS=${3:-3}
for T in 1 4 8; do
  for i in $(seq "$REPS"); do
    line=$("$LLAMA_CLI" -m "$GGUF" -f "$PROMPT" -n "$TOK" -t "$T" --temp 0 < /dev/null 2> /dev/null | grep -o "\[ Prompt:.*\]" | tail -1)
    echo "threads=$T run=$i $line"
  done
done
