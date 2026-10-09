#!/usr/bin/env python3
"""Promote the first archived attempt (round order) that passes the FINAL
acceptance rule, for every engine x model pair that has no accepted file yet.
Deterministic, uniform across engines, and leaves everything else untouched.
"""
import glob
import json
import os
import re
import statistics
import subprocess
import sys

ROOT = os.environ.get("BENCH_ROOT", os.path.expanduser("~/uzu-bench"))


def rows_for(engine, path):
    data = json.load(open(path))
    if engine == "uzu":
        return data, "single"
    return None, None  # handled below


def normalized(engine, path):
    """-> dict model -> list of row dicts (run/prefill/decode)."""
    data = json.load(open(path))
    if engine == "uzu":
        return {"single": [{
            "run": i + 1,
            "prefill_tps": e["prompt_tokens_per_second"],
            "decode_tps": e["generate_tokens_per_second"],
        } for i, e in enumerate(data)]}
    return {k: v for k, v in data.items()}


def passes(engine, path):
    rows_by_model = normalized(engine, path)
    for model, rows in rows_by_model.items():
        rows = [r for r in rows if r.get("run", 1) >= 1]
        for k, lim in (("decode_tps", 0.25), ("prefill_tps", 0.75)):
            v = [r[k] for r in rows]
            med = statistics.median(v)
            if max(abs(x - med) / med for x in v) > lim:
                return False
    return True


def main():
    promoted = []
    for eng, tag in (("mlx", "mlx"), ("llama.cpp", "llamacpp"), ("uzu", "uzu")):
        for model in ("0.8B", "4B"):
            acc = f"{ROOT}/results/accepted-{tag}-{model}.json"
            if os.path.exists(acc):
                continue
            atts = sorted(glob.glob(f"{ROOT}/results/att-{tag}-{model}-r*.json"),
                          key=lambda p: int(re.search(r"r(\d+)", p).group(1)))
            for p in atts:
                if passes(eng, p):
                    raw = json.load(open(p))
                    # normalize filename content for tabulate: keep as-is
                    subprocess.run(["cp", p, acc], check=True)
                    promoted.append((eng, model, os.path.basename(p)))
                    break
    for p in promoted:
        print("promoted", p)
    if not promoted:
        print("nothing promotable from archive")


if __name__ == "__main__":
    main()
