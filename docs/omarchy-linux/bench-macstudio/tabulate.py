#!/usr/bin/env python3
"""Tabulate results/*.json into the receipt table (means over recorded runs;
run 0 = warmup discarded where applicable) and print verdicts against the
published trymirai numbers.
"""
import json
import os
import statistics

ROOT = os.environ.get("BENCH_ROOT", os.path.expanduser("~/uzu-bench"))

# Published numbers, trymirai.com/local-models (benchmarked 2026-10-07),
# uzu 0.6.1 / MLX 0.32.2 / llama.cpp 0.4.1-dev+b29c606e2.
# Values: (prefill t/s, decode t/s).
PUB_M1 = {
    "0.8B": {"uzu": (895.0, 98.3), "mlx": (928.2, 79.4), "llama.cpp": (1089.2, 72.2)},
    "4B": {"uzu": (159.6, 21.2), "mlx": (169.2, 20.4), "llama.cpp": (196.7, 18.1)},
}
PUB_M5MAX = {
    "0.8B": {"uzu": (22111.3, 524.7), "mlx": (9721.7, 443.7), "llama.cpp": (13339.7, 260.7)},
    "4B": {"uzu": (5462.0, 166.0), "mlx": (4158.0, 159.0), "llama.cpp": (3716.0, 123.0)},
}
MODELS = {"0.8B": "0.8b", "4B": "4b"}


def norm_uzu(entry, idx):
    return {
        "run": idx + 1,  # recorded file: all runs count (warmup was a separate invocation)
        "prompt_tokens": entry["tokens_count_input"],
        "output_tokens": entry["tokens_count_output"],
        "ttft_s": entry["time_to_first_token"],
        "prefill_tps": entry["prompt_tokens_per_second"],
        "decode_tps": entry["generate_tokens_per_second"],
    }


def mean_runs(rows):
    rows = [r for r in rows if r.get("run", 1) >= 1]  # run 0 = warmup
    agg = {}
    for k in ("prefill_tps", "decode_tps", "ttft_s"):
        vals = [r[k] for r in rows if r.get(k) is not None]
        agg[k] = (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)
    agg["prompt_tokens"] = rows[0]["prompt_tokens"]
    agg["output_tokens"] = statistics.mean([r["output_tokens"] for r in rows])
    return agg


def load_results():
    res = {"uzu": {}, "mlx": {}, "llama.cpp": {}}
    for model in MODELS:
        uzu_raw = json.load(open(f"{ROOT}/results/accepted-uzu-{model}.json"))
        res["uzu"][model] = mean_runs([norm_uzu(e, i) for i, e in enumerate(uzu_raw)])
        for eng, tag in (("mlx", "mlx"), ("llama.cpp", "llamacpp")):
            raw = json.load(open(f"{ROOT}/results/accepted-{tag}-{model}.json"))
            key = next(k for k in raw if f"Qwen3.5-{model}" in k)
            res[eng][model] = mean_runs(raw[key])
    return res


def fmt(agg):
    return (f"{agg['prefill_tps'][0]:.0f} ± {agg['prefill_tps'][1]:.0f}",
            f"{agg['decode_tps'][0]:.1f} ± {agg['decode_tps'][1]:.1f}",
            f"{agg['ttft_s'][0]*1000:.0f} ± {agg['ttft_s'][1]*1000:.0f}")


def main():
    res = load_results()
    rows = ["| Model | Engine | Input tok | Prefill tok/s | Decode tok/s | TTFT ms | Pub. M1 16GB prefill | Pub. M1 16GB decode |",
            "|---|---|---|---|---|---|---|---|"]
    for model in MODELS:
        for eng in ("uzu", "mlx", "llama.cpp"):
            p, d, t = fmt(res[eng][model])
            pub = PUB_M1[model][eng]
            rows.append(
                f"| Qwen3.5 {model} | {eng} | {res[eng][model]['prompt_tokens']} | {p} | {d} | {t} | {pub[0]:.0f} | {pub[1]:.1f} |")
    print("\n".join(rows))

    print("\nMeasured ratios, uzu / other (published M1 16GB ratio in parens):")
    verdicts = {}
    for model in MODELS:
        for metric in ("prefill", "decode"):
            u = res["uzu"][model][f"{metric}_tps"][0]
            m = res["mlx"][model][f"{metric}_tps"][0]
            l = res["llama.cpp"][model][f"{metric}_tps"][0]
            idx = ("prefill", "decode").index(metric)
            pm, pl = PUB_M1[model]["mlx"][idx], PUB_M1[model]["llama.cpp"][idx]
            pu = PUB_M1[model]["uzu"][idx]
            winner = max(("uzu", u), ("mlx", m), ("llama.cpp", l), key=lambda x: x[1])[0]
            pub_winner = max(("uzu", pu), ("mlx", pm), ("llama.cpp", pl), key=lambda x: x[1])[0]
            verdicts[f"{model}/{metric}"] = {
                "measured": {"uzu": round(u, 1), "mlx": round(m, 1), "llama.cpp": round(l, 1)},
                "published_m1": {"uzu": pu, "mlx": pm, "llama.cpp": pl},
                "measured_winner": winner,
                "published_m1_winner": pub_winner,
                "uzu_vs_mlx": round(u / m, 2), "pub_m1_uza_vs_mlx": round(pu / pm, 2),
                "uzu_vs_llama": round(u / l, 2), "pub_m1_uzu_vs_llama": round(pu / pl, 2),
            }
            print(f"  {model} {metric}: uzu/mlx {u/m:.2f} ({pu/pm:.2f})  uzu/llama {u/l:.2f} ({pu/pl:.2f})  winner {winner} (published M1: {pub_winner})")
    json.dump({"table": rows, "verdicts": verdicts},
              open(f"{ROOT}/results/table.json", "w"), indent=2)
    print("\nwrote results/table.json")


if __name__ == "__main__":
    main()
