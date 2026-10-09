#!/usr/bin/env python3
"""Stability check for one engine+model result file under co-tenant GPU load.

Accepts a session (warmup + 3 recorded runs) when the 3 recorded runs' decode
t/s are within 25% and prefill t/s within 75% of their median (co-tenant GPU
bursts on this shared box hit prefill hardest; decode is the claim-bearing
metric and stays strict). Prints ACCEPT or RETRY plus the per-run values.
Usage: check-stable.py <engine: mlx|llamacpp|uzu> <file>
"""
import json
import statistics
import sys

engine, path = sys.argv[1], sys.argv[2]
data = json.load(open(path))

if engine == "uzu":  # list of BenchResult; recorded file has no warmup run
    entries = [{
        "run": i + 1,
        "prefill_tps": e["prompt_tokens_per_second"],
        "decode_tps": e["generate_tokens_per_second"],
        "ttft_s": e["time_to_first_token"],
    } for i, e in enumerate(data)]
    models = {"single": entries}
elif engine == "mlx":
    models = data
else:
    models = data

ok = True
for model, rows in models.items():
    rows = [r for r in rows if r.get("run", 1) >= 1]
    vals = {}
    for k, lim in (("decode_tps", 0.25), ("prefill_tps", 0.75)):
        v = [r[k] for r in rows]
        med = statistics.median(v)
        spread = max(abs(x - med) / med for x in v)
        vals[k] = (v, round(spread, 2))
        if spread > lim:
            ok = False
    print(f"  {model}: decode {vals['decode_tps'][0]} spread {vals['decode_tps'][1]} | "
          f"prefill {vals['prefill_tps'][0]} spread {vals['prefill_tps'][1]}")
print("ACCEPT" if ok else "RETRY")
sys.exit(0 if ok else 1)
