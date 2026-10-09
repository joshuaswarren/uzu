#!/usr/bin/env bash
# Provision private toolchains in $BENCH_ROOT (default ~/uzu-bench) on the Mac Studio for the
# uzu 0.6.1 vs llama.cpp vs MLX benchmark (M1 Ultra).
#   - uzu 0.6.1 CLI built from the fork tag 0.6.1 (registry models fetched via the CLI)
#   - llama.cpp Metal build pinned to published engine commit b29c606e2 (0.4.1-dev+b29c606e2)
#   - uv venv with current mlx + mlx-lm (versions recorded in logs/models.log)
# Nothing outside $BENCH_ROOT (default ~/uzu-bench) is modified; the Mac Studio system is untouched.
set -euo pipefail
ROOT="${BENCH_ROOT:-$HOME/uzu-bench}"
mkdir -p "$ROOT/logs"
cd "$ROOT"

{
  echo "=== env $(date -u +%FT%TZ) on $(hostname)"
  rustc --version; cargo --version; cmake --version | head -1; uv --version
  xcodebuild -version | head -1
} >> logs/env.txt

job_uzu() {
  exec >> logs/uzu.log 2>&1
  echo "=== uzu start $(date -u +%FT%TZ)"
  git clone --depth 1 --branch 0.6.1 https://github.com/joshuaswarren/uzu uzu
  echo "uzu 0.6.1 commit: $(git -C uzu rev-parse HEAD)"
  (cd uzu && cargo build --release -p cli)
  CLI="$ROOT/uzu/target/release/cli"
  "$CLI" --version
  # Fetch the uzu registry models via the CLI; the download cache dir is what
  # `cli bench` takes as model_path.
  "$CLI" --model trymirai/Qwen3.5-0.8B-M --message "Say hi in five words."
  "$CLI" --model trymirai/Qwen3.5-4B-M --message "Say hi in five words."
  echo "=== uzu done $(date -u +%FT%TZ)"
}

job_llamacpp() {
  exec >> logs/llamacpp.log 2>&1
  echo "=== llama.cpp start $(date -u +%FT%TZ)"
  git clone https://github.com/ggml-org/llama.cpp llama-cpp
  (
    cd llama-cpp
    git checkout b29c606e2
    echo "llama.cpp commit: $(git rev-parse HEAD)"
    cmake -B build -DLLAMA_CURL=OFF
    cmake --build build --config Release -j 12 --target llama-server llama-cli
    ./build/bin/llama-server --version
  )
  echo "=== llama.cpp done $(date -u +%FT%TZ)"
}

job_models() {
  exec >> logs/models.log 2>&1
  echo "=== models start $(date -u +%FT%TZ)"
  uv venv --python 3.12 venv
  uv pip install --python venv/bin/python mlx mlx-lm 'huggingface_hub[hf_transfer]'
  "$ROOT/venv/bin/python" - <<'PY'
import importlib.metadata as md
import mlx.core as mx  # noqa: F401
print("mlx", md.version("mlx"), "| mlx-lm", md.version("mlx-lm"))
import os
os.environ.setdefault("HF_HOME", os.path.expanduser("$BENCH_ROOT (default ~/uzu-bench)/hf"))
from huggingface_hub import snapshot_download, hf_hub_download, HfApi
for repo in ["mlx-community/Qwen3.5-0.8B-4bit", "mlx-community/Qwen3.5-4B-4bit"]:
    print(repo, "->", snapshot_download(repo))
for repo in ["unsloth/Qwen3.5-0.8B-GGUF", "unsloth/Qwen3.5-4B-GGUF"]:
    files = sorted(f for f in HfApi().list_repo_files(repo) if f.endswith("Q4_K_S.gguf"))
    for f in files:
        print(repo, f, "->", hf_hub_download(repo, f))
PY
  echo "=== models done $(date -u +%FT%TZ)"
}

job_uzu & U=$!
job_llamacpp & L=$!
job_models & M=$!
RC=0
wait "$U" || { echo "uzu job FAILED"; RC=1; }
wait "$L" || { echo "llama.cpp job FAILED"; RC=1; }
wait "$M" || { echo "models job FAILED"; RC=1; }
exit $RC
