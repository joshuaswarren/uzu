#!/usr/bin/env python3
"""llama.cpp engine harness: cold server per run (no KV prefix reuse),
warmup + 3 measured streaming requests, TTFT from first content chunk,
prefill/decode t/s from llama-server's own timings block.
Port 8013 on purpose (never touch 8002).
"""
import glob
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.environ.get("BENCH_ROOT", os.path.expanduser("~/uzu-bench"))
PORT = 8013
MAX_TOKENS = 128


def ggufs():
    out = {}
    for key, pat in [
        ("Qwen3.5-0.8B-Q4_K_S", f"{ROOT}/hf/hub/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots/*/Qwen3.5-0.8B-Q4_K_S.gguf"),
        ("Qwen3.5-4B-Q4_K_S", f"{ROOT}/hf/hub/models--unsloth--Qwen3.5-4B-GGUF/snapshots/*/Qwen3.5-4B-Q4_K_S.gguf"),
    ]:
        hits = sorted(glob.glob(pat))
        if not hits:
            sys.exit(f"missing GGUF for {pat}")
        out[key] = hits[0]
    return out


def wait_health(proc):
    for _ in range(300):
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited rc={proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if json.load(r).get("status") == "ok":
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError("llama-server never became healthy")


def chat(msgs):
    payload = json.dumps({
        "messages": msgs, "max_tokens": MAX_TOKENS,
        "temperature": 0, "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    timings = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if "timings" in obj:
                timings = obj["timings"]
            for ch in obj.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    if first is None:
                        first = time.perf_counter()
    t_end = time.perf_counter()
    return first - t0, timings, t_end


def bench(gguf, label):
    prompt = json.load(open(f"{ROOT}/prompt.json"))
    msgs = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"]},
    ]
    runs = []
    for r in range(4):  # run 0 = warmup
        proc = subprocess.Popen([
            f"{ROOT}/llama-cpp/build/bin/llama-server",
            "-m", gguf, "-ngl", "999", "-c", "8192",
            "--host", "127.0.0.1", "--port", str(PORT),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            wait_health(proc)
            ttft, timings, _ = chat(msgs)
            entry = {
                "run": r,
                "ttft_s": round(ttft, 4),
                "prompt_tokens": timings.get("prompt_n"),
                "output_tokens": timings.get("predicted_n"),
                "prefill_tps": round(timings["prompt_per_second"], 2),
                "decode_tps": round(timings["predicted_per_second"], 2),
            }
            runs.append(entry)
            print(label, entry, flush=True)
        finally:
            proc.terminate()
            proc.wait(timeout=60)
    return runs


def main():
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    out = {}
    only = os.environ.get("ONLY")
    for label, gguf in ggufs().items():
        if only and only not in label:
            continue
        out[label] = bench(gguf, label)
    json.dump(out, open(f"{ROOT}/results/llamacpp.json", "w"), indent=2)
    print("wrote results/llamacpp.json")


if __name__ == "__main__":
    main()
