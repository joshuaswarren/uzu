# uzu on aarch64 Linux (Apple silicon, Omarchy)

Branch `omarchy-linux` of this fork. Base: upstream `main` at `c92723bd`. Status: the CPU backend builds and runs on an
Apple M1 Max under Linux with no source changes except one real credential store (below). The GPU backend is Metal only upstream;
the Vulkan work is in [metal2vk](https://github.com/joshuaswarren/metal2vk) and scoped in [vulkan-scope.md](vulkan-scope.md).

## Build

uzu needs a nightly toolchain (`rust-toolchain.toml`, edition 2024, rust 1.98 or newer). Build only what exists on Linux:

```
RUSTUP_TOOLCHAIN=nightly cargo build --release -p uzu-engine --no-default-features --features cpu    # 23 s
RUSTUP_TOOLCHAIN=nightly cargo build --release -p cli --no-default-features --features backend-cpu    # 1 min 42 s
```

Measured with rustc 1.101.0-nightly (2026-10-08) installed through `rustup-init` into a private directory. The distribution's
packaged rustc on the test machine was unusable at the time (linked against an LLVM major version that had been replaced by a
partial upgrade), which is why a self-contained nightly was used. System packages needed: a C toolchain, `alsa-lib` (the `rodio`
audio crate), `pkgconf`, `openssl`.

Run the quick start model:

```
target/release/cli --model alibaba:qwen3.5:0.8b:mirai:mirai-m:4 -m "Tell me a short, funny story about a robot"
```

The model (439 MB) downloads through the registry and produces coherent text; `list-models` works against the live registry.

## macOS-only crates and APIs

| item | where | on Linux |
| --- | --- | --- |
| `metal` (mtl-rs), `objc2`, `obfstr`, `xxhash-rust` for Metal | `uzu-engine` feature `metal`, target-gated to Apple | not built; `cpu` backend only |
| build-time Metal compiler tooling (`clang-ast`, `depfile`, `rayon`, `blake3`) | `uzu-engine` build-dependencies, target-gated to Apple | not built |
| `objc2`, `objc2-foundation` | `uzu`, `download-manager`, target_vendor apple | compiled out |
| download manager Apple backend (NSURLSession, `block2`) | `download-manager/src/backends/apple` | upstream's `UniversalBackend` (reqwest) is used; verified by a real 439 MB model download |
| `apple-native-keyring-store` (Keychain) | `crates/legacy/uzu/src/settings/mod.rs` | upstream falls back to `keyring_core::mock`, an in-memory store that loses every secret at process exit. This branch replaces that with a real store: the Secret Service over D-Bus (`zbus-secret-service-keyring-store`, pure Rust crypto) when a session bus provides one, otherwise the kernel keyring (`linux-keyutils-keyring-store`, no daemon, keys do not survive a reboot). Test: `crates/legacy/uzu/tests/settings_linux.rs` (a second `Settings` instance reads back what the first saved, which the mock store cannot do); passes against GNOME Keyring through the Secret Service on the test machine, the kernel keyring fallback was not exercised |
| `keisoku` (IOReport, SMC, HID) | dev-dependency, macOS only; GPU temperature for bench cool-down | not built; there is no thermal gating on Linux |

## CPU backend against llama.cpp CPU

Same model class (Qwen3.5 0.8B, 4-bit), Apple M1 Max (8 performance cores), Linux, the machine shared with other jobs
(load average about 2 to 3, so these are not quiet-machine numbers). uzu: `cli bench` with a 128 token prompt and 16 generated
tokens, one run, greedy. llama.cpp: build `65840ed`, Qwen3.5-0.8B Q4_0 GGUF (`ggml-org/Qwen3.5-0.8B-GGUF`), a 106 word prompt,
32 generated tokens, three runs per thread count.

| engine | threads | prompt tok/s | generation tok/s |
| --- | --- | --- | --- |
| uzu CPU backend (bfloat16) | 1 (single threaded) | 0.57 | 0.38 |
| llama.cpp | 1 | 158 | 52 |
| llama.cpp | 4 | 388 to 482 | 93 to 95 |
| llama.cpp | 8 | 484 to 521 | 97 to 98 |

uzu's CPU backend is a correctness reference, not a performance path: it has no threads and no SIMD (`rayon` is only a
dependency of the Metal build) and the matrix multiply is a scalar loop over bfloat16 (`backends/cpu/kernel/matmul/reference.rs`).
It is 280 times slower than one llama.cpp thread on prompt processing and 140 times slower on generation. Time to first token
for the 128 token prompt was 225 s. The two models are not the same quantization (Mirai-M 4-bit with bfloat16 activations
against Q4_0), so the ratio is a bound on the CPU path, not a statement about the formats.

## Files

- `vulkan-scope.md`: Metal kernels mapped to omarchy-mlx's Vulkan kernels and to metal2vk status, with an estimate.
- `data/census-uzu-55.txt`: parse level census of all 55 `.metal` files through the metal2vk shim (39 of 55 parse today).
- `cpu-bench/`: scripts for the CPU comparison and the keyring test.
- `data/uzu-published-*.txt`: uzu's published benchmark tables for three devices (from trymirai.com/metrics, 7 Oct 2026).
- `bench-macstudio/`: replication of the published comparison on an M1 Ultra, summarized below.

## Published speed claim, replicated on an M1 Ultra

uzu's page (trymirai.com/metrics, 7 Oct 2026) reports uzu 0.6.1 fastest in output speed against MLX 0.32.2 and llama.cpp. They did not
publish an M1 Ultra. Run on one (macOS, 1357 token prompt, 128 greedy tokens, 1 warmup and 3 recorded runs per engine, uzu 0.6.1,
MLX 0.32.3 with mlx-lm 0.32.0, llama.cpp build 10964 at the commit they used, 4-bit class checkpoints; full method, every attempt
and the caveats in `bench-macstudio/RECEIPT.md`). The machine's GPU was shared with another inference service, so the runs used a
retry-until-stable protocol and the numbers carry that noise.

| model | engine | prefill tok/s | decode tok/s |
| --- | --- | --- | --- |
| Qwen3.5 0.8B | uzu | 2240 | 150.1 |
| Qwen3.5 0.8B | MLX | 1203 | 137.3 |
| Qwen3.5 0.8B | llama.cpp | 2667 | 125.3 |
| Qwen3.5 4B | uzu | 387 | 43.1 |
| Qwen3.5 4B | MLX | 239 | 38.5 |
| Qwen3.5 4B | llama.cpp | 256 | 24.4 |

Decode: the claim holds on both models (uzu ahead of MLX by 9 and 12 percent, of llama.cpp by 20 and 76 percent). Prefill: it does
not hold at 0.8B, where llama.cpp is ahead (uzu 0.84 of llama.cpp, in line with the published M1 card at 0.82), and it does at 4B.
This is the target for any Vulkan port of uzu on the same model class.
