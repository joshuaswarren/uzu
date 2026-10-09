# uzu vs llama.cpp vs MLX on the Mac Studio (M1 Ultra) - receipt

Date: 2026-10-09 · Host: the Mac Studio (Mac13,2, Apple M1 Ultra, 128 GB, macOS 26.6.2 25G83, up 14 d)
Branch: `omarchy-linux` (this directory) · Runs: private toolchains in `$BENCH_ROOT (default ~/uzu-bench)`, removed afterwards.

## Question

Reproduce, on an M1 Ultra, the engine comparison published at
trymirai.com/local-models (uzu 0.6.1 vs MLX vs llama.cpp, benchmarked 2026-10-07),
for Qwen3.5 0.8B and 4B in 4-bit, and judge whether the published M1 ordering
holds on our hardware. Prompt: exactly **1357 tokens** (chat-formatted, Qwen3.5
tokenizer; `prompt.json`). Output: 128 tokens, greedy (`temperature` 0 /
sampler temp 0 / `greedy: true`). Per engine: **1 warmup + 3 recorded runs**.

the Mac Studio's GPU is shared with the owner's inference service (`a local inference server`),
which saturates it in bursts. All engine×model pairs therefore ran under an
identical **stable-session protocol** (`run-until-stable.sh`, `check-stable.py`):
a session (warmup + 3 recorded runs) is accepted only when its 3 recorded runs
agree - decode within 25 % of median (the claim-bearing metric, strict) and
prefill within 75 % of median (co-tenant bursts hit prefill hardest);
otherwise it is retried, round-robin across engines so all three sample the
same box conditions. **Every attempt is kept** (`results/att-*.json`, GPU
snapshots in `results/gpu-log.txt`); the first attempt passing the final rule,
in round order, is what the table reports (`select-from-archive.py`).

## Toolchains (all private, built this run)

| Tool | Version | Source | Commit |
|---|---|---|---|
| uzu CLI | 0.6.1 (`cli bench` binary) | fork tag `0.6.1` | `8d01581474760bd1fc7f4a547ff1d02f7dada437` |
| llama.cpp (Metal) | 0.4.1-dev build 10964 - **same build as published** | ggml-org/llama.cpp | `b29c606e2` (checked out) |
| MLX | 0.32.3 | PyPI via uv | (published used 0.32.2) |
| mlx-lm | 0.32.0 | PyPI via uv | - |
| rust/cargo | 1.94.0-nightly (env); uzu's `rust-toolchain.toml` pulled 1.101.0-nightly into `~/.rustup` for this build only | - | - |
| cmake / uv / Xcode | 4.4.4 / 0.12.3 / 27.0 (27A266a) | preinstalled, unmodified | - |

Build log snippets: `results/toolchain.txt`. No system package was upgraded; the
only machine-side writes were inside `$BENCH_ROOT (default ~/uzu-bench)` and `~/.rustup`.

## Models (sizes and sha256 in `results/models-manifest.txt`)

| Engine | Model | Size |
|---|---|---|
| uzu (registry, via CLI) | trymirai/Qwen3.5-0.8B-M | ~0.5 GB |
| uzu (registry, via CLI) | trymirai/Qwen3.5-4B-M | ~2.7 GB |
| MLX | mlx-community/Qwen3.5-0.8B-4bit | 0.5 GB |
| MLX | mlx-community/Qwen3.5-4B-4bit | 2.5 GB |
| llama.cpp | unsloth/Qwen3.5-0.8B-Q4_K_S.gguf | 508,104,960 B |
| llama.cpp | unsloth/Qwen3.5-4B-Q4_K_S.gguf | 2,590,430,368 B |

uzu registry models were fetched with the CLI itself
(`cli --model trymirai/Qwen3.5-<size>-M --message ...`), cached under
`~/.cache/mirai/models/mirai/…`, and passed to `cli bench <dir> <task> <out>`.

## Results

Means ± sd over the 3 recorded runs of the accepted session (warmup discarded;
uzu's explicit warmup was a separate `cli bench` invocation). Published M1 16 GB
numbers from trymirai.com/local-models (2026-10-07) in the last two columns.

| Model | Engine | Input tok | Prefill tok/s | Decode tok/s | TTFT ms | Pub. M1 16GB prefill | Pub. M1 16GB decode |
|---|---|---|---|---|---|---|---|
| Qwen3.5 0.8B | uzu 0.6.1 | 1357 | 2240 ± 293 | 150.1 ± 2.3 | 613 ± 87 | 895 | 98.3 |
| Qwen3.5 0.8B | MLX 0.32.3 | 1357 | 1203 ± 73 | 137.3 ± 4.7 | 1130 ± 67 | 928 | 79.4 |
| Qwen3.5 0.8B | llama.cpp b29c606e2 | 1357 | 2667 ± 1054 | 125.3 ± 3.2 | 572 ± 184 | 1089 | 72.2 |
| Qwen3.5 4B | uzu 0.6.1 | 1357 | 387 ± 41 | 43.1 ± 0.8 | 3535 ± 361 | 160 | 21.2 |
| Qwen3.5 4B | MLX 0.32.3 | 1357 | 239 ± 31 | 38.5 ± 0.2 | 5753 ± 806 | 169 | 20.4 |
| Qwen3.5 4B | llama.cpp b29c606e2 | 1357 | 256 ± 29 | 24.4 ± 0.6 | 5351 ± 656 | 197 | 18.1 |

Measured ratios, uzu / other (published M1 16 GB ratio in parens):

| | uzu/MLX | uzu/llama.cpp | winner measured | winner published M1 |
|---|---|---|---|---|
| 0.8B prefill | 1.86 (0.96) | 0.84 (0.82) | llama.cpp | llama.cpp |
| 0.8B decode | 1.09 (1.24) | 1.20 (1.36) | uzu | uzu |
| 4B prefill | 1.62 (0.94) | 1.51 (0.81) | uzu | llama.cpp |
| 4B decode | 1.12 (1.04) | 1.76 (1.17) | uzu | uzu |

Absolute scale check: measured/published M1 16 GB is 1.5-2.5× across all cells
(decode 150/98 = 1.53, 43/21 = 2.0; prefill 2240/895 = 2.5, 387/160 = 2.4) -
consistent with an M1 Ultra GPU being roughly 2-3× an M1.

## Verdict

- **Decode - claim CONFIRMED.** The published claim that uzu has the fastest
  decode on M1 reproduces on M1 Ultra for both models: uzu 150.1 t/s (0.8B) and
  43.1 t/s (4B), ahead of MLX (+9 % / +12 %) and llama.cpp (+20 % / +76 %).
  The published M1 ordering (uzu > MLX > llama.cpp on both models) matches our
  measurement; margins are smaller than published at 0.8B and larger at 4B.
- **Prefill - claim SPLIT.** The published M1 16 GB card does NOT show uzu
  fastest on prefill (llama.cpp wins both models there). On M1 Ultra we measure
  the same at 0.8B (llama.cpp 2667 t/s > uzu 2240, ratio 0.84 vs published
  0.82 - reproduced), but the ordering flips at 4B (uzu 387 t/s, 1.51×
  llama.cpp; published M1 had llama.cpp ahead, 0.81). The 4B prefill result
  instead agrees with the M5 Max card, where uzu leads prefill.
- TTFT (not published): uzu wins at 4B (3.5 s vs 5.4/5.8 s); at 0.8B
  llama.cpp and uzu tie (~0.6 s), MLX trails (~1.1 s).

## Fairness notes

- Same machine, same prompt, same 128-token greedy budget, sequential runs, one model loaded at a time.
- Each engine's own timing instrumentation is used (`cli bench` stats, llama-server `timings`, mlx_lm step timing). Prefill/decode definitions follow each engine's own convention (±1 token boundary, <1 % effect).
- Input token counts: 1357 for all three engines on both models (verified in per-run JSONs).
- llama.cpp runs used a cold `llama-server` per run so no KV prefix is reused between runs; uzu bench resets its session per run; MLX gets a fresh prompt cache per run. Model load time is excluded everywhere; TTFT is first token after prefill.
- Engine versions are the published ones where possible: llama.cpp pinned to the exact published commit b29c606e2; MLX is 0.32.3 vs published 0.32.2 (minor drift, noted).
- Quantizations differ by design: Mirai-M (uzu, custom 4-bit format with Hadamard), MLX 4-bit, GGUF Q4_K_S - same checkpoints family, different formats, as in the published comparison.
- Qwen3.5 is a reasoning model; outputs may include reasoning tokens. Token throughput is unaffected by content.

## Load and thermal notes

- the Mac Studio was NOT idle: owner services run on it (`a local inference server` inference daemon ~35-60 % CPU, fseventsd, AssetCache, openclaw agent scans). Load average during runs: 8.7-11.9 (20 cores). The owner's services were not touched.
- **GPU contention is the dominant measurement hazard**: `powermetrics` samples taken mid-task showed `GPU HW active residency: 100 %` while nothing of ours was running. Slow runs (e.g. decode 4-5 t/s vs 22-27 t/s siblings, prefill 50 vs 500 t/s) are co-tenant GPU bursts, not thermals: `pmset -g therm` reports "No thermal warning level has been recorded. No performance warning level has been recorded."
- First llama.cpp sessions were badly contaminated (bimodal runs); the stable-session protocol above was introduced for ALL engines equally and accepted sessions are used everywhere. All rejected attempts and GPU samples are in `results/`.

## What was not run

- Linux/Vulkan: explicitly out of scope (non-goal).
- Resident-memory and joules-per-token metrics (published pages also show them): `cli bench` reports them; llama.cpp/MLX harnesses do not measure them - skipped as not required by the task.
- Published numbers are per-device from trymirai.com (M1 16 GB and M5 Max 128 GB cards); the M1 16 GB card is the comparison baseline. M1 Ultra itself has no published card.
