# Scoping a Vulkan backend for uzu

uzu's engine has a backend split (`crates/uzu-engine/src/backends/`): `cpu` (76 Rust files, about 4.4k lines of kernels),
`metal` (47 Rust files, about 5.8k lines of host code, 55 `.metal` kernel files, about 6.8k lines of MSL plus about 5.8k lines of
shared headers) and `common` (the `Backend`, `Context`, `CommandBuffer`, `Kernels` traits). A Vulkan backend implements the same
traits. This note maps the Metal kernels to what omarchy-mlx's hand written Vulkan kernels already cover, and says which path
each family takes.

Two ways to get there, and why both are used:

1. Translate: [metal2vk](https://github.com/joshuaswarren/metal2vk) compiles the `.metal` files unmodified (clang C++ for OpenCL
   through a `metal_stdlib` shim, then clspv). No per-kernel porting; correctness is checked against the Metal source itself.
2. Route: for the hot ops (GEMM and quantized matmul, attention, DeltaNet) the host layer calls omarchy-mlx's tuned kernels when
   the shape and dtype match, and logs which path ran. Tuned kernels use the 8x8 matrix unit and shapes the translation does not
   reach yet.

## Kernel families

Parse column: does the file parse through the shim today (`census.py`, parse level only, templates not instantiated).
Hand kernel column: the omarchy-mlx shader that covers the same operation (`overlay/mlx/backend/omarchy/shaders/`).

| uzu family (files) | what it is | hand kernel in omarchy-mlx | parse | path |
| --- | --- | --- | --- | --- |
| activation, gated_act_mul, tensor_add_bias, tensor_add_scale (4) | elementwise | `unary_vec`, `binary_vec`, `swiglu`, `fused_chain` | 3 of 4 | translate |
| softmax (1) | row softmax with sinks | `softmax_suffix` | yes | translate (runs, correct, parity with hand kernel on the first measurement) |
| normalization, qkv_norm (2) | RMS/layer norm | `fast_norm`, `fast_norm_gated`, `fast_rope_norm` | 1 of 2 | translate |
| embedding, pooling, logit_transform (3) | gather, reduce | `gather_axis`, `gather_take`, `reduce_*` | yes | translate |
| matmul/gemm (3), full precision and 4/8-bit scale-bias, zero-point, symmetric | tiled GEMM, simdgroup_matrix core | `matmul_coopmat`, `qmm_coopmat`, `qmm_tile`, `int8_matmul*` | yes (front end instantiates it) | translate once clspv accepts the IR (in progress), route until then |
| matmul/gemv (1) | decode matvec, quantized | `matmul_vec`, `qmm_vec`, `bonsai_qmv*`, `gather_qmm` | no (declarations) | route; translate later |
| gemm_trellis, activation_transform/trellis (2) | Mirai trellis weight format | none | no | translate only; no hand kernel |
| attention single/two-pass, prepare, kv_cache_update, sigmoid_gate (6) | decode and prefill attention | `sdpa_decode_native*`, `sdpa_prefill_flash`, `fast_rope` | 4 of 6 | route; translate for the rest |
| attention_gemm, gemm_grouped, attention_fallback (3) | prefill attention on the GEMM core | `sdpa_prefill_flash` | yes | route |
| ancestor_attention (1), gdn tree_verify (6), weaver (3) | speculative decoding trees | none | 3 of 10 | translate only |
| gdn prefill/update/conv/chunked (12) | Gated DeltaNet (Qwen3.5) | `gated_delta_decode*`, `gated_delta_prefill*`, `gdn_conv_decode` | 8 of 12 | route; Qwen3.5 needs this family |
| ssm (4), short_conv (1), separable_causal_conv (1) | Mamba2 SSD, LFM2 short conv | `conv_dw1d`, `scan_*` (no SSD kernel) | 5 of 6 | translate |
| sampling (3), radix_top_k_small (1) | sampling, top-k | `sort_*`, `argreduce_suffix`, `random_bits` | 2 of 4 | translate |

Counts are files; one file holds several kernels and variants (`VARIANTS(...)` expands each file into tens of pipelines).

## What the translation needs beyond the shim

- `simdgroup_matrix` is lowered to `VK_KHR_cooperative_matrix`: Honeykrisp's 8x8 fragment layout is Apple's, so uzu's per-lane code
  maps one to one (validated, see the metal2vk README). Only the fork driver exposes it today.
- Metal 4 host API (argument tables with raw GPU addresses, residency sets, sparse buffers, function constants) needs a Rust shim
  over a Vulkan layer (slice `host` in metal2vk): 47 files, about 5.8k lines, of which the engine calls about 60 distinct methods.
- uzu's `build.rs` generates the Rust kernel traits and bindings from the DSL annotations on macOS only; a Linux path has to run
  the same analysis with clang through the shim and emit bindings plus the kernel library.
- bfloat16 is the model dtype for most Mirai checkpoints. Vulkan has no bf16 arithmetic on this hardware, so the shim widens to f32
  at load and narrows at store; this costs bandwidth on the translated path only.

## Estimate

| piece | state | remaining |
| --- | --- | --- |
| shim + clspv lowering, elementwise/reduction/softmax | done | per-kernel tuning |
| simdgroup_matrix to cooperative matrix | done | f16/bf16 inputs, resident accumulators |
| uzu Gemm through clspv | front end done, IR accepted for the three small tilings after the clang IR fix | correctness against the CPU backend, speed |
| remaining parse failures (16 of 55 files) | listed in `data/census-uzu-55.txt` | vector brace initialisers, bfloat4, a few missing builtins |
| host API shim | started | kernel library format, buffer address mapping, queues |
| Linux `build.rs` path | not started | analysis through clang, binding generation |
| end to end Qwen3.5-0.8B on the translated path | not started | everything above plus the GDN family |

## Targets

The numbers to beat are the published ones for the M1 family (`data/uzu-published-macos-m1-pool.txt`): Qwen3.5-0.8B-M decode
98 tok/s on an M1 16GB, against 79 for MLX and 72 for llama.cpp; prefill 895, 928 and 1089 tok/s respectively. On Linux the
comparison point is omarchy-mlx on the same chip.
