# Gemma 4 26B-A4B decode megakernel for gfx942 (FlyDSL): design

Status: milestones 1 to 4, 6 and 7 done; milestone 5: second node done (MI325X), second
day queued (see "Milestone 5"). The kernel runs every layer of a decode step
in one persistent launch, passes the correctness suite (`bench/results/m4/suite/`) and runs
inside vLLM ("Milestone 6"). Milestones 1 and 2 were signed off on 2026-10-27; from
milestone 3 on, the decisions below were made without sign-off, each with its evidence and
the rejected alternative.

## Fixed point

One decode step of the Gemma 4 26B-A4B text model for M = 1..8 independent sequences runs as
one persistent FlyDSL launch on one gfx942 GPU (MI300X, 304 CUs, SPX). The launch reads the
token ids, the positions, vLLM's paged KV caches (sliding-window and full-attention groups) and
the FP8 weights. It writes either the soft-capped fp32 logits (standalone engine) or the bf16
final-normed hidden state of every row (a serving engine's own LM head), and the new K and V of
every layer.

Properties that stay true if the implementation is discarded:

- P1. Identical inputs give identical output bits.
- P2. A row's output bits do not depend on the other rows of the batch (their tokens, their
  experts, M, padding rows).
- P3. The kernel output is as close to the exact reference as the emulating reference (which
  copies the kernel's rounding) is.
- P4. Every compile-time setting is a field of `Cfg` (or a debug global), and the kernel name
  holds a hash of all of them.

## Scope

- In: embedding (with the `sqrt(hidden)` scale), 30 decoder layers (25 sliding-window, 5
  full-attention), router, 128 routed experts, the dense MLP that runs beside them, every norm,
  `layer_scalar`, final norm, and (option `Cfg.lm_head`) the tied LM head with the final logit
  soft-cap. M = 1..8. Decode only, greedy (argmax on the host), one GPU, TP1.
- Out: prefill (the standalone harness feeds a prompt one token per step), speculative decoding,
  TP > 1, vision and audio, sampling in the kernel.
- Weights: the FP8 checkpoint `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic`. The BF16 checkpoint is
  the quality reference only.

## Model facts (from the checkpoints and the code, 2026-10-27)

Sources: `google/gemma-4-26B-A4B-it/config.json`, safetensors headers of
`google/gemma-4-26B-A4B-it` and `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic`, vLLM `origin/main`
228b1e3d2 `vllm/model_executor/models/gemma4.py`, `.../rotary_embedding/gemma4_rope.py`,
HF transformers `models/gemma4/modeling_gemma4.py` and `masking_utils.py`.

| Item | Value |
|---|---|
| Layers | 30. `layer_types`: full attention at 5, 11, 17, 23, 29; sliding at the other 25 |
| Hidden size | 2816 |
| Sliding window | 1024. Key `j` is visible to query `i` when `i - 1024 < j <= i` (HF `kv_idx > q_idx - sliding_window`; vLLM TRITON_ATTN `(w-1, 0)`) |
| Sliding attention | 16 q heads, 8 KV heads, head_dim 256; q, k, v projections (q 4096, k 2048, v 2048) |
| Full attention | 16 q heads, 2 KV heads, head_dim 512; `attention_k_eq_v`: no v_proj, V comes from the k_proj output (q 8192, k 1024) |
| Attention math | `q_norm`, `k_norm` = RMSNorm with weight; `v_norm` = RMSNorm without weight; RoPE on q and k only; scaling 1.0; no attention soft-cap (`attn_logit_softcapping` absent) |
| RoPE, sliding | NeoX, theta 1e4, all 256 dims rotate |
| RoPE, full | NeoX "proportional", theta 1e6, partial factor 0.25: 64 angle pairs with `inv_freq = theta^(-2i/512)`, the other 192 pairs are identity (vLLM `Gemma4RotaryEmbedding`) |
| Dense MLP | gate/up 2112, down 2112 -> 2816, GeGLU with `gelu_pytorch_tanh`. Runs on every layer beside the MoE. This is the "shared expert" |
| MoE | 128 experts, top-8, expert size 704, GeGLU (`gelu_tanh`). No other shared expert |
| Router | `x = rmsnorm_noweight(h) * 2816^-0.5 * scale`; `logits = x @ W_r^T` (bf16 W, fp32 out); softmax over all 128; top-8; renormalize; multiply by `per_expert_scale[e]` |
| Norms | All `x * rstd * w` (not zero-centred), eps 1e-6. Per layer: `input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm_1` (dense), `pre_feedforward_layernorm_2` (MoE input), `post_feedforward_layernorm_2` (MoE), `post_feedforward_layernorm`, plus `layer_scalar` (one value per layer) |
| Soft-capping | Final logits only: `30 * tanh(logits / 30)` |
| Per-layer embeddings | None (`hidden_size_per_layer_input` = 0) |
| KV sharing | None (`num_kv_shared_layers` = 0) |
| Embedding | 262144 x 2816 bf16, times `bf16(sqrt(2816))` = 53.0 |
| Vocabulary, LM head | 262144, tied to the embedding |
| Checkpoints | BF16 `google/gemma-4-26B-A4B-it` (51.6 GB). FP8 `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic`: E4M3FN weights with one bf16 scale per output row for every Linear including all experts; dynamic per-token FP8 activations; router, norms, embedding bf16. Others: `Firworks/...-fp8` (same scheme), NVFP4 (NVIDIA), AWQ-INT4, GGUF, MLX, ONNX (`amd/..._gpuep...`) |

Decoder layer (vLLM `Gemma4DecoderLayer.forward`):

```
a  = o_proj(attn(input_ln(h)))
h1 = h + post_attn_ln(a)
d  = mlp(pre_ff_ln(h1))                                  # dense, 2112
e  = sum_k w_k * expert_k(pre_ff_ln_2(h1))               # routed on router(h1)
h' = (h1 + post_ff_ln(post_ff_ln_1(d) + post_ff_ln_2(e))) * layer_scalar
```

Every norm after a projection needs the complete row of that projection. This is the main
structural difference from Qwen3.8, where each sub-block adds its output to the residual
directly.

## Bytes per step (as built: FP8 weights, M rows)

Each fp8 unit (16 rows x 128 K, 2 KiB) carries 16 fp32 row scales (64 B, +3.1 %). The expert
down projection is padded from K = 704 to 768 (+9 % of the down bytes, about 2 % of the step).

| M | attention (q, k, v, o) | dense MLP | routed experts (8 M distinct) | total without / with the LM head (1.48 GB bf16) |
|---|---|---|---|---|
| 1 | 1.15 GB | 0.57 GB | 1.52 GB | 3.25 / 4.73 GB |
| 2 | 1.15 GB | 0.57 GB | 3.03 GB | 4.77 / 6.25 GB |
| 4 | 1.15 GB | 0.57 GB | 6.07 GB | 7.80 / 9.28 GB |
| 8 | 1.15 GB | 0.57 GB | 12.13 GB | 13.87 / 15.34 GB |

Rows share experts on real text, so the routed-expert bytes at M > 1 are an upper bound. KV:
8 KiB per token per sliding layer (at most 1024 tokens), 4 KiB per token per full layer.

## Reuse

Copied (copy and diverge, D2) from `qwen3.6-35b-a3b-gfx942/` at megakernels `e2a6fb8`; each
file names its source at the top:

- `mk/dev.py`: unchanged, plus `tanh`, `gelu_tanh`, `bf16_rne`, `pack_bf2`.
- `mk/gemv.py`: the flat unit-stream GEMV (split-K, fixed-order last arriver, `fp8` and `bf16`
  kinds). Changed: the `fp8` kind has one scale per output row (16 fp32 per unit), because the
  Gemma checkpoint has per-channel scales, not 128 x 128 blocks.
- `mk/moe.py`: routing in every CTA, the grouped expert GEMV over the unique selected experts
  with piece-aligned wave ranges, per-slot partials, two-level S6 counters. Changed: see "MoE
  stages".
- `mk/stages.py`: `store_quant8`, `epi_raw`, `prologue_scaled`, the rstd reduction.
- `mk/kernel.py`, `mk/runner.py`, `mk/pack.py`: the grid, the barriers with the device epoch,
  opaque thread ids, pointer tables, `Cfg`, the unit-order packing.
- `mk/vendor/`: FlyDSL's `buffer_ops` (identical in the Qwen3.8 and Qwen3.6 directories).
- Harness: `tests/test_decode.py`, `tests/test_determinism.py`, `tests/eval_text.py`,
  `bench/bench_decode.py`, `bench/profile_decode.py`, `bench/stock/`, `tools/`.

New for Gemma: `mk/attention.py` (milestone 2 work), the whole-row prologues with the deferred
residual (`stages.rebuild_h`, `prologue_router`, `prologue_gu`, `prologue_final`), the
dense MLP as pseudo-experts, the reference (`mk/reference.py`).

## Stages and barriers (as built)

```
step  = embed | L0 | L1 | ... | L29 | final norm (+ LM head)
layer = qkv | attn | o_proj | router | gate_up | down
```

| Stage | Prologue (every CTA) | Work | Last-arriver epilogue | Output |
|---|---|---|---|---|
| embed | none | CTA b: 16-column groups b, b + 304, ... | none | `h1` = bf16(embed x 53.0), `d` = `e` = 0 |
| qkv | layer 0: `rebuild_h` (h = h1, the scaled embedding); x = input_ln(h), fp8. Layers > 0: copy x, published by the previous layer's combine CTAs | fp8 GEMV 8192 (sliding) or 9216 (full) x 2816 | raw | y [M][QKV_N] |
| attn | none | wave tasks (row, KV head, q-head group, 64 positions); K / V into the cache | merge of the records | o fp32 [M][16 HD] |
| o_proj | o to fp8 | fp8 GEMV 2816 x 4096 / 8192 | raw + 16-column sums of squares | a, ssq_a |
| router | CTAs with router units: h1 = h + post_attn_ln(a); x = bf16(rms(h1) x 2816^-0.5 x scale). CTAs NR .. NR + M - 1 (`publish_row`): row m of h1, rstd(h1), and the two gate_up inputs, fp8, to global memory | bf16 GEMV 128 x 2816 | raw | fp32 logits; h1, rstd(h1), gate_up inputs |
| gate_up | routing (every CTA); copy the published inputs (rows 0..M-1: fp8(pre_ff_ln_2(h1)), rows M..2M-1: fp8(pre_ff_ln(h1))) | grouped fp8 GEMV, (U + 3) x 1408 x 2816 | GeGLU per (position, group) | act fp32 [M][11][768] |
| down | act of the CTA's <= 2 positions, fp8 | grouped fp8 GEMV, (U + 3) x 2816 x 768 | e = sum over slots j < 8 of w_j x part_j, d = part_8 + part_9 + part_10; raw + sums of squares; count the group. Then CTAs cus - M + m (not after the last layer) wait for all groups and run `combine_row`: row m of h (as `rebuild_h`), x = input_ln(h) of the next layer, fp8 | e, d, ssq_e, ssq_d; h, next qkv input |
| final | `rebuild_h` from the last layer; x = bf16(norm(h)) | bf16 LM head GEMV 262144 x 2816 (or CTA 0 stores x) | 30 tanh(z / 30) | fp32 logits or bf16 hidden |

Barriers: 1 + 6 x 30 = 181 per step (Qwen3.6: 241, Qwen3.8: 321).

Whole-row work: h (layer 0 and final stage: every CTA; other layers: one combine CTA per row)
and h1 (router CTAs: all rows; publisher CTAs: one row each) are computed from the raw stage
outputs: thread t < 352 holds columns 8 t .. 8 t + 7 of a row; the row sums go through a DPP
wave sum and a fixed-order sum over the 8 waves. Every producer of a row runs the same
arithmetic in the same order, so all copies of a row hold the same bits (checked:
`tests/ab_bitwise.py` against the all-CTA prologues, `bench/results/m4/`). The values of a row
do not depend on M or on the other rows (P2).

## MoE stages

From the Qwen3.6 MoE stage (grouped GEMV over the unique selected experts, piece-aligned wave
ranges, per-slot partials, routing in every CTA, routing hash in debug builds). Changes:

- 128 experts (`NE // 64` logits per lane), router GEMV of 128 rows, no shared-expert gate.
- Routing weights: softmax over the 128 fp32 logits, top-8 (ties to the lower expert id),
  renormalized: `exp(l_j - l_0) / sum`, then times `per_expert_scale[expert]` (as HF).
- The dense MLP (2112 = 3 x 704) runs as 3 pseudo-experts (experts 128, 129, 130 of each
  stack: columns 704 k .. 704 k + 703 of gate_proj / up_proj and of down_proj). They are the
  last 3 positions for every row (slots 8, 9, 10, weight 1). They read the
  pre_feedforward_layernorm input rows M..2M-1 in the gate_up LDS; the routed experts read
  rows 0..M-1 (pre_feedforward_layernorm_2). Positions: U + 3, U <= min(8 M, 128).
- GeGLU: act = gelu_tanh(gate) x up in the gate_up last arriver.
- The down last arriver hands two sums to its epilogue: e (weighted, slots 0..7 in rank order)
  and d (slots 8..10 in order), each raw with its sums of squares: the post-feedforward norms
  need the whole rows.
- Chunks: gate_up C = 11 (2 units per piece), down C = 2 (3 units per piece).

## Attention over vLLM's paged cache

`mk/attention.py` (milestone 2; facts from vLLM `origin/main`, to be confirmed on gfx942 in
milestone 6):

- Per-layer pointers to the KV tensor, the block table of the layer's group and the slot
  mapping. `Cfg` holds one geometry (block size, element strides) per layer type.
- A task covers 64 positions and can span several blocks (16 or 32 positions each), one
  block-table lookup per 16-position tile.
- Sliding layers: tasks start at the chunk that holds `max(0, pos - 1023)`; positions below are
  masked; no task reads a block-table entry below the window (`tests/test_attention.py` fills
  those blocks with NaN).
- Full layers: the 8 q heads of a KV head are split into 2 groups of 4 (LDS).
- q, K, V round to bf16 with round to nearest even (`dev.pack_bf2`).

## Numerics

| Quantity | Format |
|---|---|
| Linear weights | FP8 E4M3FN of the RedHatAI checkpoint, re-read as E4M3FNUZ (exact), one fp32 scale per row |
| Router weight, norms, embedding, LM head | bf16, as in the checkpoint |
| GEMV inputs (fp8 weights) | e4m3fnuz, power-of-two scale per (token, 128 K) |
| Router and LM head inputs | bf16 (round to nearest even) |
| Router logits, routing weights | fp32 |
| Accumulation, residual stream, raw stage outputs, norms | fp32 |
| Expert act | fp32 in global memory, fp8 in the down GEMV input |
| KV cache, q in QK^T | bf16 |

The checkpoint was calibrated with per-token activation scales. Measured in the reference
(`tests/ref_modes.py`, 400 tokens, FP8 checkpoint, milestone 2 with 64-K blocks): perplexity with
the kernel's fp8 activations / with bf16 activations = 0.994. `tests/eval_text.py` measures the
kernel (128-K blocks) against the same reference.

## Invariants

- I1: every CTA is resident (grid = CU count, LDS <= 64 KiB: the builder asserts it).
- I2: a stage reads only data published before the preceding barrier, or by a last arriver whose
  counter it observed.
- I3: P3 on all 30 layers (`test_decode.py`).
- I4: P1 and P2 (`test_determinism.py`).
- I5: every CTA computes the same routing tables in a step (debug hash, `test_decode.py`).
- I6: a CTA's down range touches at most 2 positions (asserted at build time).
- I7: padding rows select no routed expert and write no KV.
- I8: no sliding-layer task reads a position at or below `pos - 1024`.

## Correctness oracle (as built)

1. `mk/reference.py`: PyTorch decoder from safetensors. `step` decodes one token for M rows;
   `prefill` runs a whole prompt (exact mode, causal and window masks) into the same caches.
   Modes: exact (bf16 GEMV inputs, fp32 elsewhere, FP8 weights dequantized exactly), emulate
   (the kernel's fp8 GEMV inputs, 128-K blocks, the expert down input padded, the dense MLP as 3
   pseudo-experts), fp32 (no bf16 rounding: for HF in fp32). The expert sum runs in rank order,
   as the kernel's slots.
2. `tests/ref_vs_hf.py` (milestone 2): fp32 mode against HF transformers fp32 on 1300 tokens:
   cosine min 0.99942, top-1 0.999.
3. `tests/test_attention.py`: the attention stage alone against a PyTorch model of the paged step.
4. `tests/test_decode.py`: the reference prefills CTX tokens of real text per row; the kernel's
   paged caches and the emulating reference start from that cache; then 16 greedy steps. Gates:
   kernel~exact worst cosine and mean top-1 within 2x the emulate~exact distance (P3);
   kernel~emulate is printed; in every step and layer, the kernel's routing tables (CTA 0)
   equal `route_logits` of its own router logits; all 304 CTAs build the same tables (I5).
   Run at M = 1, 2, 4, 8 and CTX = 512, 2048, 4096.
5. `tests/test_determinism.py`: three runs bitwise equal; row 0 alone and inside M = 2, 4, 8 and
   with padding rows bitwise equal; 1100 steps (past the window).
6. `tests/eval_text.py`: teacher-forced perplexity of the kernel, the FP8 reference and the BF16
   reference on five passages and one 1536-token text; batch invariance.

Changes from the Qwen3.6 gates: (a) "every top-1 change against the exact model is a near-tie"
is replaced by a P3 gate on top-1 agreement. Evidence: at M = 1, CTX = 512, step 0 the emulating
reference itself changes the exact model's top-1 token at an exact logit gap of 7.1 (and the
kernel the same token). (b) The absolute floor "kernel~emulate cosine > 0.95" is dropped (the
value is printed). Evidence: at M = 2, CTX = 512 the emulating reference drifts to cosine 0.890
from the exact model (kernel~exact 0.935, kernel~emulate 0.936): fp8 activation rounding moves
routing decisions, and over 30 layers this model's logits drift farther than Qwen3.6's. A
relative bound (kernel~emulate within 2x emulate~exact) was tried and dropped: per step the two
angles to the exact model bound the angle between them (triangle inequality), and with 16 steps
the worst-step ratio is noise (it failed M = 1, CTX = 4096 with kernel~exact inside the P3 bound;
`bench/results/m3/test_decode_m1_ctx4096_relative_gate.txt`). The perplexity gate of
`eval_text.py` is the quality gate.

## Milestone 3 results

One MI300X (banff-cyxtera-s70-4, job 68197433), 2026-10-28; logs in `bench/results/m3/`.

`tests/test_decode.py`, 30 layers, 16 greedy steps after CTX tokens of real text (worst-step
logits cosine / mean top-1 agreement):

| M | CTX | kernel~exact | emulate~exact | kernel~emulate | result |
|---|---|---|---|---|---|
| 1 | 512 | 0.9929 / 0.69 | 0.9911 / 0.81 | 0.9908 / 0.81 | PASS |
| 1 | 2048 | 0.9968 / 0.81 | 0.9853 / 0.75 | 0.9840 / 0.63 | PASS |
| 1 | 4096 | 0.9911 / 0.88 | 0.9948 / 0.88 | 0.9858 / 0.88 | PASS |
| 2 | 512 | 0.9350 / 0.84 | 0.8901 / 0.88 | 0.9359 / 0.91 | PASS |
| 2 | 2048 | 0.9836 / 0.97 | 0.9870 / 0.91 | 0.9869 / 0.88 | PASS |
| 2 | 4096 | 0.9861 / 0.88 | 0.9926 / 0.91 | 0.9972 / 0.94 | PASS |
| 4 | 512 | 0.9410 / 0.83 | 0.9317 / 0.84 | 0.9387 / 0.83 | PASS |
| 4 | 2048 | 0.9454 / 0.91 | 0.9678 / 0.91 | 0.9775 / 0.94 | PASS |
| 4 | 4096 | 0.9912 / 0.92 | 0.9905 / 0.94 | 0.9858 / 0.95 | PASS |
| 8 | 512 | 0.8142 / 0.98 | 0.8894 / 0.97 | 0.9092 / 0.96 | PASS |
| 8 | 2048 | 0.9794 / 0.92 | 0.9616 / 0.94 | 0.9672 / 0.92 | PASS |
| 8 | 4096 | 0.8574 / 0.95 | 0.9018 / 0.95 | 0.9064 / 0.95 | PASS |

In every run: routing tables equal in all 304 CTAs (I5) and equal to `route_logits` of the
kernel's own router logits in every layer and step. A 6-layer run at M = 8, CTX = 2048 (less
routing drift) gives kernel~emulate 0.9968.

- `tests/test_determinism.py` at M = 1 and 8, 1100 steps (past the window): three runs bitwise
  equal; row 0 bitwise equal alone and inside M = 2, 4, 8 and with 4 padding rows. PASS.
- `tests/eval_text.py` (teacher-forced perplexity): passages kernel 216.2, FP8 reference 226.8,
  BF16 reference 242.7; 1536-token text kernel 85.3, FP8 reference 98.5, BF16 reference 63.0
  (positions 1024..1535: 27.59 / 27.58 / 13.13). Batch invariance bitwise. PASS. The
  instruction-tuned model on raw text has high perplexity (HF bf16 itself: 459 on 400 tokens,
  milestone 2), so the comparison is relative. The gap between the FP8 reference and the BF16
  reference past the window is the checkpoint's (the kernel matches the FP8 reference); it is
  checked against vLLM's own FP8 path in milestone 6.
- `tests/test_attention.py`: PASS (worst 9.12e-4, decision 7).
- Speed (`bench/bench_decode.py`, 512-token prompt, 256 steps, LM head included): TPOT 3.68 /
  4.25 / 5.02 / 6.54 ms at M = 1 / 2 / 4 / 8; 3.81 ms and 3.90 ms at M = 1 after 2048 and 4096
  tokens.

## Milestone 4: speed

### Stock baselines (same node, banff-cyxtera-s70-4, 2026-10-28)

`bench/stock/` (logs and JSON in `bench/results/stock/`): vLLM 0.32.1rc1.dev505+g00b6c4c56
(image `vllm-rocm-nightly-00b6c4c5`), `vllm bench serve`, 512 input / 256 output tokens,
concurrency = batch; SGLang 0.5.10.post1 `bench_one_batch`, the same lengths. Median TPOT, ms:

| Engine | Checkpoint | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| vLLM (default, TRITON_ATTN) | FP8 | 4.54 | 4.74 | 5.24 | 5.98 |
| vLLM (default) | BF16 | 4.44 | 4.98 | 6.35 | 7.43 |
| vLLM + AITER | BF16 | 4.89 | 5.45 | 6.77 | 7.93 |
| vLLM + AITER | FP8 | fails: AITER's FP8 MoE has no `GELU_TANH` | | | |
| vLLM + AITER without AITER MoE, memory 0.8 | FP8 | 5.06 | 5.32 | 5.80 | 6.68 |
| vLLM + AITER without AITER MoE, memory 0.8 | BF16 | 5.01 | 5.52 | 7.04 | 8.02 |
| SGLang | BF16 | 9.16 | 9.83 | 10.45 | 11.76 |
| SGLang | FP8 | fails: `This GEMM is not supported!` (compressed-tensors FP8) | | | |

The fastest stock engine at every batch with the FP8 checkpoint is vLLM's default path.
`vllm bench serve` uses random token ids as prompts. vLLM + AITER without its MoE
(`VLLM_ROCM_USE_AITER_MOE=0`) fails at startup with an out-of-memory error in the KV cache
allocation at 0.9 GPU memory (twice, `failed_oom*`); the rows above ran at 0.8.
`bench/results/stock/summary.md` lists every run.

### Optimization log

Each change is measured with `bench/bench_decode.py` (512-token prompt, 128 steps, 2 or 3
repeats) and, if it only moves work, checked bitwise against the old configuration
(`tests/ab_bitwise.py`, `Cfg` flags through `MK_CFG`). Logs in `bench/results/m4/`.

| Change | M = 1 | M = 8 | Result |
|---|---|---|---|
| Start (milestone 3 kernel) | 3.65 ms | 6.49 ms | `profile_v0.txt`: e1 55 us, e2 36 us, qkv 30 us per layer at M = 8 |
| K-chunk sweep of o, e1, e2 (`MK_CHUNKS`) | o = 2: -1.5 % | all worse | kept the defaults (`chunk_sweep.txt`) |
| o_proj prologue merges the attention records, one head per chunk (`attn_merge_o`) | +6.7 % | +4.3 % | rejected, removed (`flags_alone.txt`) |
| Router publishers: only the CTAs with router units run the router prologue; M other CTAs publish h1 and the quantized gate_up inputs, which the gate_up prologue copies (`router_publish`) | -0.5 % | -5.6 % | kept; bitwise equal (`ab_v2.txt`) |
| One CTA (the last expert-down arriver) rebuilds h and publishes the qkv input (`combine_last`) | -1.8 % | +4 % with router_publish | replaced (`combine_vs_m.txt`) |
| M CTAs wait for all expert-down groups (device counter), each rebuilds one row of h and publishes its qkv input; the qkv prologue copies it (`combine_rows`) | -1.1 % | -3.4 % | kept; bitwise equal at M = 1 and 8 (`combine_rows.txt`) |
| V loads one batch ahead in the attention task | 0 % | +6 % | rejected, reverted (`attn_vpipe*.txt`) |

Prompt content matters at M = 8. The first benchmarks fed random token ids (as `vllm bench
serve` does); greedy decoding after them routes the rows to fewer distinct experts than real
text does. `bench/bench_decode.py` and `bench/profile_decode.py` now feed windows of real
text (`tests/texts.text_prompts`, `MK_PROMPT=random` for the old prompts). Real text, TPOT
(`text_prompts.txt`): 3.58 / 4.15 / 5.18 / 6.70 ms at M = 1 / 2 / 4 / 8; random prompts
(`suite/bench_m*.txt`): 3.56 / 4.06 / 4.80 / 5.92 ms. At M = 8 on real text the expert stages
take 67 us (gate_up) and 66 us (down, 33 us busy) per layer.

Where the time goes now (`profile_v1.txt`, random prompts, per layer, spans include the stage's barrier wait):
at M = 8 the expert stages read about 216 MB (gate_up) and 111 MB (down) per layer for about 52
distinct experts, so e1 (55 us) and e2 (27 us busy) run near the HBM bandwidth; the LM head
(1.48 GB bf16) takes 337 us. At M = 1 the weight stream is about 100 MB per layer (about 23 us at
4.3 TB/s) but a layer takes about 100 us: 6 barriers (about 2 us each), the attention task's
serial load chain (`probes_attn_v1.txt`: q/k prep 3.5 us, scores 6 us, P V 7 us, merge 5 us on
the full-attention layer) and the stage prologues are the fixed cost.

## Milestone 6: vLLM

Branch `users/sareeder/gemma4-gfx942-specialized-decode` of the vLLM fork (on the Qwen3.6
branch, which holds vllm#61311's library). Touch points as listed in "vLLM integration"
below. Run in the vLLM nightly image `vllm-rocm-nightly-00b6c4c5` with the branch's Python
files overlaid (`~/mk/vllmg4-src`, compiled extensions of the image). Logs in
`bench/results/m6/` (banff-cyxtera-s70-4, 2026-10-28).

vLLM's layout as read at run time (`on*_summary.txt`, `runtime/decode.py` log line): backend
TRITON_ATTN, 6 KV cache groups (5 sliding-window groups and 1 full group), block 16 for the
sliding groups and 32 for the full group, K and V separate in each block. The kernel binds the
groups' block tables and slot mappings per layer. `spec._refuse_caches` checks the geometry.

- Conformance: `tests/models/gemma4/test_gemma4_specialized_decode.py` and the #61311 kit
  (`tests/models/common/specialized_decode`): 75 passed (`pytest.txt`).
- Kernel on vs kernel off (`bench/vllm_e2e.py`, chat template, greedy, each prompt alone at
  batch 1; `compare4.txt`): 46 % identical tokens over 10 prompts, 2 of them 2438 and 2442
  tokens long (past the 1024 window). Both runs answer the long retrieval prompt with the
  same 52 tokens. Every divergence is at a logprob gap of 1.25 or less. vLLM's own path
  differs from itself as much: the same prompts at batch 8 and at batch 1, kernel off, agree on
  41 % of the tokens (`compare_off_b8_b1.txt`).
- Speed in vLLM, real-text 512-token prompts, TPOT ms (`compare4.txt`, widths 1 and 2 served):

| Batch | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| vLLM, kernel off | 4.43 | 4.85 | 5.02 | 6.22 |
| vLLM, kernel on | 3.59 | 4.23 | 5.01 (vLLM's path) | 6.16 (vLLM's path) |

- With widths 1, 2, 4 and 8 served (`on_summary.txt`, `off_summary.txt`,
  `compare_rawtext_all_widths.txt`, raw-text prompts at batch 8 for the quality part): kernel on
  3.57 / 4.20 / 5.20 / 6.94 ms, kernel off 4.41 / 4.64 / 5.22 / 6.12 ms. The kernel launch alone
  in vLLM takes 6.33 ms at width 8 (`kernel_time_in_vllm.txt`).

Decision 9 (milestone 6): the spec serves widths 1 and 2 only (`spec._gemma_widths`); wider
steps take vLLM's path with its CUDA graphs. Evidence: the table above. Rejected: serving
width 4 (no gain) and 8 (13 % slower). The kernel itself stays correct and tested at M = 1..8
(milestone 3 and 4 suites).

Decision 10 (milestone 6): `spec._refuse_config` does not check `tie_word_embeddings`. vLLM's
text config of the checkpoint reads False although `config.json` says true; the kernel does not
read the LM head in vLLM (`lm_head = False`).

## Milestone 5: reproducibility

Second node (`bench/results/m5/quanta-ccs-aus-j02-45_68199254/`, 2026-10-28): the full
suite (`scripts/suite_austin.sh`, commit `767d759`, the same kernel as milestone 4) on an
AUSTIN node with an **MI325X** (gfx942, 304 CUs), checkpoints downloaded to node `/tmp`.
Every test passes: test_decode at M = 1, 2, 4, 8 and contexts 512 / 2048 / 4096,
determinism and batch invariance at M = 1 and 8, eval_text (perplexity 216.183 on the
passages and 85.291 on the long text, the same values as on the MI300X), test_attention.
TPOT with real-text prompts: 3.45 / 4.03 / 5.03 / 6.45 ms at M = 1 / 2 / 4 / 8 (MI300X:
3.58 / 4.18 / 5.29 / 6.77 ms, `bench/results/m7/`); the MI325X has faster HBM.

Second day and a second MI300X: `scripts/repro_sbatch.sh` runs `tests/run_suite.sh` from a
source tarball of commit `d22bcd4` (milestone 4 kernel, random-prompt benchmark): job
68198890 starts on 2026-10-29, job 68198889 waits for a MARKHAM MI300X other than
banff-cyxtera-s70-4 (the only one, banff-cyxtera-s83-2, was fully allocated on 2026-10-28).

## Decisions

Milestone 1 (signed off 2026-10-27):

- One persistent launch with grid barriers. Rejected: per-layer launches.
- FP8 RedHatAI checkpoint. Rejected: requantize to MXFP4.
- Router as its own stage with bf16 input. Rejected: fp8 router input (flips top-8 choices;
  vLLM uses bf16); router in the gate_up stage (LDS at M = 8).
- Deferred residual (each consumer rebuilds h and h1 from raw stage outputs). Rejected: a
  combine stage per post-norm (2 to 3 more barriers per layer).
- D1: the MoE stage comes from the Qwen3.6 kernel. D2: copy and diverge. D3: attention tasks span
  several KV blocks (vLLM's default block sizes). Storage: `~/mk/models`.

Milestone 3 (made without sign-off; evidence and rejected alternative each):

1. fp8 GEMV unit of 16 x 128 with one scale per row (16 fp32 per unit). Reason: the Qwen3.6
   `fp8` kind and its grouped GEMV are proven; only the scale load changes. Cost: +3.1 % of the
   fp8 bytes (about 0.1 GB per step). Rejected: the 16 x 64 unit of the milestone 1 plan (a new
   stream geometry in the shared GEMV), the row scale in each epilogue (every epilogue changes,
   and the expert down sum would need it per slot).
2. Expert down K padded 704 -> 768 with zero weights; the act buffer keeps zero columns 704..767.
   Cost: about 2 % of the step's bytes. Rejected: 16 x 64 units (as 1).
3. The dense MLP as 3 pseudo-experts of the MoE stages. Reason: no second GEMV code path in the
   gate_up and down stages; the dense weights stream with the experts' load balance. Rejected:
   dense GEMVs beside the grouped GEMV in the same stages (two stream kinds per stage); the
   Qwen3.6 shared-expert slot (widths differ: 2112 vs 704, and the input norm differs).
4. Activation scale per (token, 128 K), not 64 K: the Qwen3.6 GEMV's superstep. The milestone 2
   measurement used 64 K; eval_text measures the effect.
5. The embedding is its own stage with a barrier (181 barriers, not 180): layer 0's rebuild then
   needs no special case (table row 0 holds ones and layer_scalar 1.0, and d = e = 0, so h = h1
   exactly). Rejected: the embedding inside layer 0's prologue (a runtime branch in the rebuild
   or an extra embedding load in every layer).
6. The LM head and the soft-cap are in the kernel when `Cfg.lm_head` (the standalone engine and
   its benchmark; stock engines include the LM head in their TPOT). vLLM runs with
   `lm_head = False` and applies its own LM head and soft-cap. Changed from milestone 1, which
   had both outside the kernel.
7. f32 -> bf16 conversions round to nearest even (`dev.bf16_rne`), as torch and vLLM. Evidence
   (`bench/results/m3/bf16_rne_probe.txt`): FlyDSL's `.to(BFloat16)` on gfx942 already rounds to
   nearest even (0 mismatches against torch on exact ties, +-1 ulp and random values), so the
   Qwen kernels (which use `.to(BFloat16)`) also round to nearest even; their code is unchanged.
   The 1.2e-3 attention case of milestone 2 (`full M8`, row 2, head 7) stays at 9.12e-4 with RNE
   and with three rsqrt variants in the q norm: its cause is a one-ulp fp32 difference in q before
   the rounding (the norm's sum order differs from torch's), not the rounding mode. The cache
   therefore matches vLLM's path up to such one-ulp differences, not bit for bit.
8. Router logits fp32, top-8 ties to the lower expert id (as Qwen3.6). Rejected: bf16 logits
   (ties become common and the tie order of torch.topk is not defined).

## vLLM integration: touch points (milestone 6, recorded before the edits)

Base: branch `users/sareeder/qwen36-gfx942-specialized-decode` of the vLLM fork (vllm#61311's
library with the Qwen3.8 and Qwen3.6 specs). New branch `users/sareeder/gemma4-gfx942-specialized-decode`.
Gemma 4 lives in `vllm/model_executor/models/gemma4.py` (`Gemma4ForCausalLM`, `Gemma4Model`,
decoder layers) and `gemma4_mm.py` (`Gemma4ForConditionalGeneration`, the class the checkpoints
name). Edits, mirroring `qwen3_5.py`:

1. `gemma4.py` `Gemma4ForCausalLM`: `select_decode_spec` in `__init__`,
   `specialized_decode_specs()` (ROCm only), `process_weights_after_loading` building
   `SpecializedDecode(self, ...)`, `run_specialized_decode`, the hook at the top of `forward`,
   `get_model_state_cls`.
2. `gemma4_mm.py` `Gemma4ForConditionalGeneration.forward` calls
   `language_model.model(...)` directly (it bypasses `Gemma4ForCausalLM.forward`): the hook goes
   there too, plus a `specialized_decode` property, the `process_weights_after_loading` delegate
   and `get_model_state_cls`.
3. A model state (`vllm/models/gemma4/amd/model_state.py`, a `DefaultModelState`): after the KV
   cache is allocated, `set_cache_groups` and `bind_caches`.
4. New package `vllm/models/gemma4/amd/specialized_decode/`: `spec.py` (refusals, widths,
   skip_step), `runtime/decode.py` (buffers, per-group block tables and slot mappings, launch),
   `runtime/weights.py` (packing from vLLM's loaded tensors), `kernels/` (exported from this
   directory by `scripts/export_vllm.py`).
5. Tests: `tests/models/gemma4/test_gemma4_specialized_decode.py` (the #61311 conformance kit,
   CPU, stub kernel).

Facts the runtime relies on (read from code; checked at run time by `refuse_caches`):
`Gemma4Config.verify_and_update_config` selects TRITON_ATTN on ROCm; KV view per layer
`[blocks, kv_heads, block, 2 head_dim]` (K | V halves), physical order from the KV layout;
5 sliding groups and 1 full group, the full group's block twice the sliding one (page-size
unification); the V2 runner nulls sliding block-table entries below the window. vLLM's FP8
linears hold e4m3fnuz weights as a [K, N] view with fp32 [N, 1] scales doubled; full-attention
`qkv_proj` holds [q | k | v = k] (10240 rows: the kernel reads q and k); the experts'
`w13_weight` [128, 1408, 2816], `w2_weight` [128, 2816, 704] with `*_weight_scale` [E, N, 1].

## Open questions

- vLLM's attention backend, KV layout and block sizes on gfx942 for this model: milestone 6.
- The standalone engine's host process segfaults (in a torch indexing op of `Engine.step`) when
  about 2048 decode steps are queued without a host synchronization; with a synchronization every
  256 steps it does not (`bench/results/m3/queue_depth_crash.txt`). The kernel's results are
  bitwise reproducible in both cases up to the crash, and AMD_SERIALIZE_KERNEL=3 also removes it,
  so the cause is on the host side (HIP or FlyDSL launch path with a deep queue behind a
  persistent kernel) [INFERENCE]. The harness synchronizes every 256 steps (prompt loops of
  `bench_decode.py`, `test_determinism.py`); vLLM synchronizes every step.
- Fixed per-stage latency with 6 stages per layer: milestone 4.
