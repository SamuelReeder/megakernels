# Milestone 6: the kernel inside vLLM

## Current: milestone 4 kernel (41443a0), vLLM `0ad8a7c56`

Measured 2026-10-27 20:15-20:34 EDT on banff-cyxtera-s70-4 (one MI300X; SLURM job
68193015), same image, settings and scripts as the milestone 3 rounds below. The
kernels were exported from `fee3171`; `41443a0` differs only by an unused import and
a comment (the committed export is from `41443a0`; same Cfg, same kernel name). The
runtime needed no change: it packs with the Cfg's new chunk defaults (c_o 16, c_lm 1)
and sizes the counters from `CNT_REGION`. Results in `m4/`.

TPOT ms (512-token prompts, 256 generated, median of 3), rounds 1 / 2:

| Backend | Path | B = 1 | B = 2 | B = 4 | B = 8 |
|---|---|---|---|---|---|
| TRITON_ATTN | vLLM (off) | 5.25 / 5.55 | 5.22 / 5.81 | 6.32 / 6.31 | 8.00 / 8.03 |
| TRITON_ATTN | kernel (on) | 3.48 / 3.50 | 4.10 / 3.54 | 4.60 / 4.62 | 6.69 / 6.72 |
| ROCM_ATTN | vLLM (off) | 6.30 / 6.59 | 6.58 / 6.61 | 7.09 / 6.55 | 8.23 / 8.89 |
| ROCM_ATTN | kernel (on) | 3.16 / 3.20 | 4.05 / 4.55 | 5.07 / 5.13 | 6.56 / 6.58 |

Speedup, rounds 1 / 2: TRITON_ATTN 1.51 / 1.59, 1.27 / 1.64, 1.37 / 1.36, 1.20 / 1.20;
ROCM_ATTN 1.99 / 2.06, 1.62 / 1.45, 1.40 / 1.28, 1.25 / 1.35 (B = 1, 2, 4, 8). Against
the milestone 3 kernel (below) kernel-on TPOT drops by 0.2-0.5 ms at B = 1, 0.4-1.0 ms
at B = 4 and about 0.55 ms at B = 8 (B = 2 is within the round-to-round noise).

Kernel steps: 159 during the 8-prompt generate and 255 in every 256-token generate
with the kernel on, 0 off (`m4/raw/*.json`).

Greedy, 8 prompts x 160 tokens (`m4/compare_*_r1.txt`; round 2 reproduced every token
of round 1 in all four configurations; no non-finite logprob):

| Prompt | TRITON_ATTN identical | first div. | gap off / on | ROCM_ATTN identical | first div. | gap off / on |
|---|---|---|---|---|---|---|
| 0 pangram | 95/160 | 94 | 0.625 / 0.375 | 59/160 | 59 | 0.125 / 0.750 |
| 1 fibonacci | 160/160 | - | - | 160/160 | - | - |
| 2 planets | 160/160 | - | - | 32/160 | 23 | 0.000 / 1.250 |
| 3 Apollo 11 | 30/160 | 23 | 0.125 / 1.625 | 19/160 | 19 | 0.500 / 1.250 |
| 4 French | 17/17 | - | - | 17/17 | - | - |
| 5 derivative | 9/160 | 8 | 0.375 / 0.000 | 120/160 | 50 | 0.250 / 0.250 |
| 6 Romeo | 99/109 | 86 | 0.000 / 0.375 | 89/113 | 89 | 0.125 / 0.250 |
| 7 GPU tips | 119/160 | 119 | 0.125 / 0.625 | 23/160 | 22 | 0.000 / 0.500 |
| total | 689/1086 (63.4 %) | | | 519/1090 (47.6 %) | | |

The gaps stay within vLLM's own backend-to-backend gaps (up to 1.375, at the same
Apollo 11 token pair that gives the largest gap here). Kernel on, TRITON_ATTN vs
ROCM_ATTN: 1019/1086 identical (93.8 %, `m4/kernel_triton_vs_rocm.txt`). The
milestone 4 kernel agrees with the milestone 3 kernel on 42.8 % (TRITON_ATTN) and
40.6 % (ROCM_ATTN) of the tokens: its reductions add in another order (o_proj in 16
chunks, the attention merge in the prologue, DPP wave sums), which moves the same
near-ties. vLLM's kernel-off tokens are the same as in the milestone 3 session.

Unit tests with this export: 140 passed (`m4/unit_tests_cpu.txt`).

## Milestone 3 kernel (history)

Measured 2026-10-27 on banff-cyxtera-s70-4 (one MI300X, 304 CUs; SLURM job 68192237),
image `vllm-rocm-nightly-00b6c4c5.sqsh` (vLLM 0.32.1.dev594+gfe77d0906, torch
2.13.0+rocm10.0.0) with the branch's `vllm` package on `PYTHONPATH`,
`FLYDSL_RUNTIME_ENABLE_CACHE=0`, checkpoint `Qwen/Qwen3.6-35B-A3B-FP8`.

- vLLM branch: `users/sareeder/qwen36-gfx942-specialized-decode` on
  github.com/SamuelReeder/vllm, head `c482702dc`, stacked on the Qwen3.8 branch
  (`0b5a1a0ae`, itself on vllm#61311).
- Kernel: `mk/*.py` as exported by `scripts/export_vllm.py` at 16:50 EDT (milestone 3
  kernel with the ruff fixes; not the milestone 4 speed work).

## Commits

| Commit | Scope |
|---|---|
| `529e83b25` | Qwen3.8 spec (separate): `kv_geometry(kv, backend, nkv, head_dim)` takes the KV head shape (defaults unchanged). |
| `b41c03750` | #61311 library (separate): `SpecializedDecode.plan` sends the first eager step of each width to vLLM's path when piecewise CUDA graphs are captured. Bug found here; see "Capture bug". Library test added. |
| `c482702dc` | The Qwen3.6-35B-A3B spec, runtime, exported kernels, the hook on `Qwen3_5MoeForCausalLM`, and its tests. |

## Kernel-on vs kernel-off, greedy, 8 prompts x 160 tokens

`bench/vllm_e2e.py run ... on|off`, then `compare`. Logprob gaps are vLLM's logprobs
of the two candidate tokens at the first divergence; vLLM computes them from bf16
logits, so they come in steps of 0.125 and 0.000 is an exact bf16 tie.

| Prompt | TRITON_ATTN identical | first div. | gap off / on | ROCM_ATTN identical | first div. | gap off / on |
|---|---|---|---|---|---|---|
| 0 pangram | 85/160 | 78 | 0.125 / 0.250 | 160/160 | - | - |
| 1 fibonacci | 160/160 | - | - | 160/160 | - | - |
| 2 planets | 31/160 | 23 | 0.250 / 0.000 | 74/160 | 73 | 0.375 / 0.125 |
| 3 Apollo 11 | 30/160 | 23 | 0.125 / 0.875 | 19/160 | 19 | 0.500 / 1.625 |
| 4 French | 17/17 | - | - | 17/17 | - | - |
| 5 derivative | 12/160 | 12 | 0.500 / 0.125 | 12/160 | 8 | 0.500 / 0.250 |
| 6 Romeo | 42/122 | 42 | 0.000 / 0.125 | 42/122 | 42 | 0.000 / 0.125 |
| 7 GPU tips | 21/160 | 21 | 0.625 / 0.125 | 22/160 | 21 | 1.000 / 0.125 |
| total | 398/1099 (36.2 %) | | | 506/1099 (46.0 %) | | |

Every text is coherent (`compare_*.txt` print the first 160 characters of both).
The divergences are near-ties of vLLM's own path, as the two references show:

- vLLM against vLLM (kernel off, TRITON_ATTN vs ROCM_ATTN): 409/1090 identical
  (37.5 %), first divergences at gaps up to 1.375 (prompt 3: the same token pair as the
  kernel's 1.625 divergence) (`baseline_vllm_triton_vs_rocm.txt`).
- Kernel on, TRITON_ATTN vs ROCM_ATTN: 1000/1099 identical (91.0 %), one divergence
  (prompt 0, token 59, a bf16 tie): the kernel reads both KV layouts alike; the
  remaining difference comes from vLLM's prefill (`kernel_triton_vs_rocm.txt`).
- Both modes are run-to-run deterministic: round 2 reproduced round 1's tokens
  exactly in all four configurations, and no logprob is non-finite.

Layer check (`bench/vllm_debug.py`, eager, TRITON_ATTN, `debug_triton.txt`): at decode
steps 1 to 3 the kernel's normed hidden states after 1, 3, 4, 8 and 40 layers against
vLLM's own modules on the same caches: cosine 0.9996 / 0.9995 / 0.9993 / 0.9987 /
0.9983 (step 1), 0.9903 after 40 layers at step 3.

Weights (`debug_triton.txt`): vLLM's FP8 tensors equal the checkpoint's e4m3fn bits
with 0x80 -> 0 and its block scales x 2 for in_proj_qkv / in_proj_z (fused
in_proj_qkvz), k_proj (fused qkv_proj), the shared expert's up_proj, and experts 0, 77,
255 (w13 gate and up, w2); the router and in_proj_b are the checkpoint's bf16. The
repack (`runtime/weights.py`) halves vLLM's scales back and reuses the kernel's packer.

## TPOT, 512-token prompts, 256 generated

TPOT = (time(256 tokens) - time(1 token)) / 255, median of 3; batch B runs B distinct
512-token windows. Two rounds (`compare_*.txt`, `compare_*_r2.txt`):

| Backend | Path | B = 1 | B = 2 | B = 4 | B = 8 |
|---|---|---|---|---|---|
| TRITON_ATTN | vLLM (off) | 5.55 / 5.29 | 6.42 / 5.81 | 6.26 / 6.25 | 7.94 / 7.97 |
| TRITON_ATTN | kernel (on) | 3.72 / 3.75 | 4.36 / 4.40 | 9.24* / 5.56 | 7.23 / 7.30 |
| ROCM_ATTN | vLLM (off) | 6.32 / 6.58 | 6.57 / 7.16 | 7.06 / 7.08 | 8.79 / 8.75 |
| ROCM_ATTN | kernel (on) | 3.66 / 3.67 | 4.31 / 3.74 | 5.44 / 5.49 | 7.13 / 7.15 |

\* round 1 outlier, not reproduced in round 2.

Speedup (round 2): TRITON_ATTN 1.41 / 1.32 / 1.13 / 1.09, ROCM_ATTN 1.79 / 1.91 /
1.29 / 1.22.

The kernel ran every decode step: its own step counter (the grid barrier epoch on the
device, `Qwen35MoeGfx942Decode.steps_run()`, read through `LLM.apply_model`) advanced by
159 during the 8-prompt generate of 160 tokens (1 prefill + 159 decode steps; the batch
shrinks from 8 rows as prompts 4 and 6 end, so padded rows ran too) and by 255 in every
256-token TPOT generate; 0 with the kernel off (`raw/*.json`, `kernel_steps`).

Memory: model loading takes 65.22 GiB with the kernel on, 33.51 GiB off (the repacked
copy is 31.7 GiB); KV cache memory 93.45 GiB against 125.39 GiB.

## Unit tests

`unit_tests_cpu.txt`: `tests/models/qwen3_5/` (Qwen3.8 and Qwen3.6 specs),
`tests/models/common/specialized_decode/` (library, with the new test) and
`tests/models/test_dsv41_specialized_decode.py`: 140 passed (CPU, torch 2.14.0+cpu,
the branch's sources). The Qwen3.6 file runs the #61311 conformance kit
(`TestQwen35MoeGfx942Spec`: no kernel backend imported by the spec module, every cache
refusal reported, build does not move weights) plus 20 tests of its own choices.

## Capture bug (fixed in `b41c03750`)

With the kernel on, the first engine start failed in CUDA graph capture
(`capture_failure_before_fix.txt`): Model Runner V2 warms every graph eagerly (mode
NONE) before it captures it, piecewise graphs first. For widths 1, 2, 4, 8 the eager
warm-up took the kernel, while the piecewise capture runs vLLM's path, so vLLM's path
first ran inside the capture. At width 1 the shared expert's torch.compile'd SiluAndMul
recompiled (size-1 specialization) and autotuned under capture: "operation not
permitted when stream is capturing". The library now sends the first eager step of
each width to vLLM's path when piecewise graphs are captured; the FULL graph's warm-up
then takes the kernel. Qwen3.8 has the same exposure but no lazily compiled shape on
that path.

## Limitations

- FP8 checkpoint only. The BF16 checkpoint, an fp8 KV cache, TP/PP/DP/EP > 1, EPLB,
  LoRA, MTP / speculative decoding and other GPUs are refused before load.
- The routed experts must be in vLLM's plain layout (the TRITON FP8 MoE backend, the
  default here). AITER's MoE backend shuffles them and AITER's fused shared expert
  removes `mlp.shared_expert`: both are refused at load (`VLLM_ROCM_USE_AITER=1` was not
  run).
- About 32 GiB of extra weights (the repacked copy), 25 % less KV cache.
- Widths 1, 2, 4, 8 only; the CUDA graph pads 3, 5, 6, 7 rows up and the kernel skips
  the padded rows. Steps with prefill (mixed or chunked) and steps over 8 rows run
  vLLM's path.
- The first eager step of each width runs vLLM's path when piecewise graphs are
  captured (capture warm-up only in a FULL_AND_PIECEWISE deployment).
- Start-up compiles the four widths in the capture warm-up (about 12 s each with the
  FlyDSL cache off); the runtime does not implement `DecodeWarmable`.
- TPOT varies by up to about 0.6 ms between rounds at the same setting; one round 1
  value (TRITON_ATTN, kernel on, B = 4: 9.24 ms) did not reproduce.
- Measured on MI300X only; text-only (`language_model_only=True`); images through the
  multimodal path were not run.
- TRITON_ATTN's KV view has the same shape in both of its layouts with 2 KV heads; the
  spec reads the backend's (blocks, heads, block, K | V) order. The kernel-on outputs
  agree across TRITON_ATTN and ROCM_ATTN, which checks it end to end.
