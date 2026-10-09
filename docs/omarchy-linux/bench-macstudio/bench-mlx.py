#!/usr/bin/env python3
"""MLX engine harness: warmup + 3 runs per model in one process.
TTFT = time to first token; prefill_tps mirrors mlx_lm's own convention
(prompt tokens / prompt time); decode_tps = output tokens / time after
first token (matches mlx_lm.generate's printed generation t/s).
"""
import json
import os
import sys
import time

ROOT = os.environ.get("BENCH_ROOT", os.path.expanduser("~/uzu-bench"))
MAX_TOKENS = 128


def snapshot(repo):
    base = f"{ROOT}/hf/hub/models--{repo.replace('/', '--')}/snapshots"
    return f"{base}/{[d for d in os.listdir(base) if not d.startswith('.')][0]}"


def main():
    from mlx_lm.generate import stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load

    sampler = make_sampler(temp=0.0)

    prompt = json.load(open(f"{ROOT}/prompt.json"))
    msgs = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"]},
    ]
    out = {}
    only = os.environ.get("ONLY")
    for repo in ["mlx-community/Qwen3.5-0.8B-4bit", "mlx-community/Qwen3.5-4B-4bit"]:
        if only and only not in repo:
            continue
        path = snapshot(repo)
        model, tok = load(path)
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
        runs = []
        for r in range(4):  # run 0 = warmup
            cache = make_prompt_cache(model)
            t0 = time.perf_counter()
            first = None
            n_out = 0
            for _seg in stream_generate(model, tok, ids, max_tokens=MAX_TOKENS,
                                        sampler=sampler, prompt_cache=cache):
                if first is None:
                    first = time.perf_counter()
                n_out += 1
            t_end = time.perf_counter()
            ttft = first - t0
            entry = {
                "run": r,
                "prompt_tokens": len(ids),
                "output_tokens": n_out,
                "ttft_s": round(ttft, 4),
                "prefill_tps": round(len(ids) / ttft, 2),
                "decode_tps": round(n_out / (t_end - first), 2),
            }
            runs.append(entry)
            print(repo, entry, flush=True)
        out[repo] = runs
        del model
    json.dump(out, open(f"{ROOT}/results/mlx.json", "w"), indent=2)
    print("wrote results/mlx.json")


if __name__ == "__main__":
    main()
