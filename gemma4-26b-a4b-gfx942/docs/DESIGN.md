# Gemma 4 26B-A4B decode megakernel for gfx942 (FlyDSL): design

Status: milestone 1 (design). No kernel code exists yet. Decisions marked **OPEN** need
sign-off before milestone 2 starts.

## Fixed point

One decode step of the Gemma 4 26B-A4B text model for M = 1..8 independent sequences runs as
one persistent FlyDSL launch on one gfx942 GPU (MI300X, 304 CUs, SPX). The launch reads the
token ids, vLLM's paged KV caches (sliding-window and full-attention groups) and the FP8
weights. It writes the bf16 final-normed hidden state of every row and the new K and V of
every layer. The result is as close to an fp32 reference as an emulating reference is, and a
row's result does not depend on the other rows.

## Scope

- In: embedding (with the `sqrt(hidden)` scale), 30 decoder layers (25 sliding-window, 5
  full-attention), router, 128 routed experts, the dense MLP that runs beside them, every norm,
  `layer_scalar`, final norm. M = 1..8. Decode only, greedy, one GPU, TP1.
- Out: prefill (the standalone harness feeds a prompt one token per step), speculative decoding,
  TP > 1, vision and audio, sampling, the LM head and the final logit soft-cap (vLLM applies
  both; the standalone harness applies them in PyTorch).

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

## Bytes per step (FP8 weights, M rows)

| Part | MB per layer |
|---|---|
| Sliding attention (q, k, v, o) | 34.6 |
| Full attention (q, k, o) | 49.0 |
| Dense MLP | 17.8 |
| Router (bf16) | 0.7 |
| One expert | 5.9 |

Total weight stream: 3.09 GB at M = 1, 4.52 at M = 2, 7.38 at M = 4, 13.09 at M = 8 (assuming
8M distinct experts; fewer when rows share experts). KV: 8 KiB per token per sliding layer
(capped at 1024 tokens = 8.4 MB per row), 4 KiB per token per full layer.

## Reuse

The `qwen3.6-35b-a3b-gfx942/` directory named in the task does not exist (see Open decision
D1). The list below is against `qwen3.8-27b-gfx942/mk/` at megakernels `913f790`.

Reused unchanged:

- Persistent launch: grid = CU count, 512 threads, one resident CTA per CU, runtime layer loop,
  opaque thread ids, int64 per-layer and global pointer tables, device-side epoch.
- Grid barrier `dev.grid_arrive` / `grid_wait` (XCD leaders) and device-scope stores.
- Last-arriver counters (`dev.atomic_add_dev`) and the fixed-order split-K reduction.
- GEMV unit stream, `mx` and `bf16` kinds, ping-pong blocks of P = 2 units, `CHUNKS`.
- Activation quantizer: e4m3fnuz with a power-of-two scale (`stages.store_quant8`,
  `dev.pow2_scale`); the `fp8` kind takes one scale per (token, 64 K) instead of 128 K.
- Flash-decoding structure of the attention task: partial records per (row, KV head, chunk),
  last arriver merges with an online softmax.
- Kernel naming: `name = f"gemma4_decode_m{M}_{crc32(repr(cfg))}"`; every compile-time
  setting is a `Cfg` field.
- Harness shape: `reference.py` exact and emulate modes, `test_decode.py`,
  `test_determinism.py`, `eval_text.py`, `ref_vs_hf.py`, `scripts/mkrun`, `setup_env.sh`.

Copied into this directory (D2: copy and diverge until the Qwen3.6 agent moves the shared code
to `common/` after its milestone 3): `mk/dev.py`, `mk/gemv.py`, `mk/fp4.py`, `mk/vendor/`.
Changes made to the copies here:

- GEMV: new kind `fp8`. Unit = 16 rows x 64 K of e4m3fnuz = 1 KiB, the same unit size as
  `mx`. One fp32 scale per output row, applied in the last-arriver epilogue. K must be a
  multiple of 64, so 2112 (33 x 64) and 704 (11 x 64) need no padding. One activation scale
  per (token, 64 K).
  Checkpoint E4M3FN bytes are re-read as E4M3FNUZ, which halves each value exactly; the
  packer doubles the row scale and maps 0x80 (-0) to 0x00. The packer refuses NaN codes.
  The Qwen3.6 design also adds an `fp8` kind; the two are reconciled when the code moves to
  `common/`.
- Norm prologues: `x * w` (Gemma), not `x * (1 + w)`.

New for Gemma (model directory):

- Deferred-residual prologue: rebuilds the residual from the raw outputs of the preceding
  stage (post-norms, sums, `layer_scalar`), then applies the next pre-norm. See "Stages".
- Attention task for two geometries in one kernel: (head_dim 256, GQA 2) and (head_dim 512,
  GQA 8); sliding-window mask; `x * w` q/k norms; weightless v norm; V = raw K projection on
  full layers; NeoX RoPE with full rotation (sliding) or proportional 64-pair rotation (full);
  scaling 1.0; no output gate.
- Embedding scale and the window-aware paged-KV addressing for two block sizes.

From the Qwen3.6 kernel (D1, not yet available): the MoE stage (grouped GEMV over the unique
experts, per-slot partial sums, batch-invariant). Gemma adds its routing function (softmax
over all 128, top-8, renormalize, `per_expert_scale`) and GeGLU (gelu tanh) if the stage does
not already take them as parameters.

## Stages and barriers

`|` is a grid barrier.

```
step  = L0 | L1 | ... | L29 | final norm
layer = qkv | attn | o_proj | router | gate_up | down
```

| Stage | Prologue (every CTA, whole row) | Work | Output |
|---|---|---|---|
| qkv | Rebuild `h` from the previous layer's `d`, `e`, `h1` (layer 0: embedding x 53.0); `input_ln`; fp8 | GEMV q, k, v (full: q, k) | raw q, k, v fp32 |
| attn | none | One wave per (row, KV head, q-head group, 64 positions): q/k/v norms, RoPE, write new K and V to vLLM's cache, QK^T, softmax, PV; last arriver merges | o bf16 |
| o_proj | o to fp8 | GEMV o_proj | raw `a`, 16-column sums of squares |
| router | `h1 = h + post_attn_ln(a)`; router norm x `2816^-0.5` x `scale`; bf16 | bf16 GEMV, N = 128 | fp32 logits, `h1` slice |
| gate_up | `pre_ff_ln(h1)` and `pre_ff_ln_2(h1)` to fp8 (2 x 22.5 KiB at M = 8); routing from the logits | dense gate_up GEMV + gathered expert gate_up GEMV; GeGLU in the epilogue | dense act, expert act per (row, expert) |
| down | dense act to fp8 in LDS; expert act read per unit from L2 | dense down GEMV + gathered expert down GEMV; fixed-order expert sum weighted by `w_k` | raw `d`, raw `e` |
| final norm | Rebuild `h` (as in qkv); `norm` | none | bf16 hidden [M, 2816] |

Barrier count: 6 per layer, 180 per step (Qwen3.8: 5 per layer, 321 per step). The embedding
needs no barrier because the layer-0 qkv prologue reads the embedding rows itself.

The router is a separate stage because its bf16 input (45 KiB at M = 8) and the two fp8 inputs
of gate_up (45 KiB) do not fit together in 64 KiB of LDS.

Rough cost at M = 1 [INFERENCE from Qwen3.8 measurements]: weight stream 3.09 GB at 3.3-3.7
TB/s = 0.85-0.95 ms; fixed latency about 8 us per stage x 180 = 1.4 ms. Total about 2.3 ms per
token. Fixed latency dominates, so the stage count is the main speed lever (milestone 4).

## Attention over vLLM's paged cache

Facts (vLLM `origin/main`, to be confirmed by a startup log on gfx942 in milestone 6):

- `Gemma4Config.verify_and_update_config` selects TRITON_ATTN when FA4 is not available
  (the ROCm case) and no backend is set.
- The hybrid manager makes 6 KV cache groups: 5 sliding groups of 5 layers, 1 full group of 5
  layers. Page sizes differ (8192 B per token sliding, 4096 B full), so the full group gets
  twice the block size (default 16 sliding, 32 full) unless the cache layout is block-outermost.
- Per-layer view `[num_blocks, num_kv_heads, block_size, 2 * head_dim]`, K then V in the last
  dimension, physical strides depend on the layout. K and V are stored separately on full
  layers too (K is rotated and normed, V is only normed).
- Block-table entries wholly below `pos - (window - 1)` are the null block 0 (V2 runner
  `gather_block_tables`); freed blocks go to other requests.

Design:

- Per-layer pointers to the KV tensor, the block table of the layer's group and the slot
  mapping (Qwen3.8 already has per-layer `T_IDX` / `T_SLOT`). Two KV geometries in `Cfg`
  (block size and strides for sliding and for full), read from tensor strides like Qwen3.8's
  `kv_geometry`.
- A task covers 64 positions and can span several blocks (block 16 or 32), one block-table
  lookup per block.
- Sliding layers: tasks start at the first chunk that holds `max(0, pos - 1023)`; positions
  below it are masked. The last-arriver count is the number of in-window chunks. The kernel
  never reads a block-table entry below the window.
- Full layers: q heads split into 2 groups of 4 per task, so the per-wave q slice in LDS stays
  below the 64 KiB CTA limit (8 heads x 512 would need 74 KiB for 8 waves in bf16).

## Numerics

| Quantity | Format |
|---|---|
| Linear weights | FP8 E4M3FN of the RedHatAI checkpoint, re-read as E4M3FNUZ (exact) |
| Router weight, norms, embedding | bf16, as in the checkpoint |
| GEMV inputs | e4m3fnuz, power-of-two scale per (token, 64 K); router input bf16 |
| Accumulation, residual stream, raw stage outputs | fp32 |
| Router logits, routing weights | fp32 |
| KV cache | bf16 (vLLM `kv_cache_scheme: null`) |

The checkpoint was calibrated with per-token activation scales. The kernel uses one
power-of-two scale per (token, 64 K). Measured in the reference (`tests/ref_modes.py`, 400
tokens, FP8 checkpoint): perplexity with the kernel's fp8 activations / with bf16 activations
= 0.994.

## Invariants

- I1: every CTA is resident.
- I2: a stage reads only data published before the preceding barrier, or by a last arriver
  whose counter it observed.
- I3: the kernel is as close to the exact reference as the emulating reference is, on all 30
  layers (`test_decode.py`).
- I4: identical inputs give identical bits; a row does not depend on the other rows. For the
  expert GEMVs this is a property the Qwen3.6 MoE stage must provide (its design states it).
- I5: the routing is computed identically by every CTA (same instructions on the same fp32
  logits). Routing is chaotic across the 30 layers: with the same FP8 weights, the reference's
  emulate and exact modes pick different top-8 sets in 33 % of (token, layer) pairs after a few
  layers of divergence (`tests/ref_modes.py`; median 8th-9th logit gap 0.048, < 1e-3 in 1.2 %
  of decisions). So routing equality is checked per layer on the kernel's own layer input,
  not end to end.
- I6: no sliding-layer task reads a position at or below `pos - 1024`. Check: the harness puts
  NaN in the null block and in every out-of-window block.

## Correctness oracle

1. `mk/reference.py`: PyTorch decoder from safetensors, one token per step, contiguous KV.
   Modes: exact (bf16 GEMV inputs, fp32 everything else, FP8 weights exactly dequantized),
   emulate (the kernel's fp8 GEMV inputs), fp32 (no bf16 rounding anywhere, fp32 KV cache: for
   the HF fp32 comparison).
2. `tests/ref_vs_hf.py`: reference (BF16 checkpoint) against HF transformers on a 1300-token
   prompt (> window). Against HF fp32 with the fp32 mode: pass rule cosine > 0.999 and top-1
   >= 0.9, plus per-layer residual distance at 6 positions. Against HF bf16 (exact mode):
   reported only; it measures HF's own bf16 noise.
3. Per-op checks before fusion: fp8 GEMV against `torch`; the attention stage
   (`tests/test_attention.py`) against a PyTorch model of the paged step, at window-boundary
   positions, block sizes 16 and 32, two physical KV layouts, padding rows, NaN in the null
   and out-of-window blocks, three consecutive steps.
4. Dumps: per-layer `h`, router top-8 ids and weights. Routing check: per layer, the kernel's
   top-8 against the reference router run on the kernel's own layer input (I5).
5. `tests/test_decode.py` at M = 1, 2, 4, 8, contexts 512, 2048, 4096 (> window), Qwen3.8 pass
   rule.
6. `tests/test_determinism.py`: 3 runs, bitwise-equal outputs, M = 1 and 8.
7. `tests/eval_text.py`: teacher-forced perplexity of kernel, FP8 reference, BF16 HF; top-1
   agreement; KL against BF16; batch invariance (a passage at M = 1 against the same passage in a
   batch of 5).

## Decisions

- One persistent launch with grid barriers, as Qwen3.8. Rejected: per-layer launches (a
  launch costs more than a barrier); a dataflow graph with readiness counters for v1 (more
  states to verify; kept as a milestone-4 option).
- FP8 RedHatAI checkpoint. Rejected: requantize to MXFP4 (halves the stream, but the task
  says FP8 when an FP8 checkpoint exists, and it would change quality).
- fp8 GEMV unit of 16 x 64 (1 KiB). Rejected: 16 x 128 (2 KiB) needs K padding for 2112 and
  704 and changes the unit size the stream code assumes.
- LM head and soft-cap outside the kernel. Rejected: in-kernel bf16 LM head (out of scope;
  1.48 GB more per step in the launch; vLLM computes logits itself).
- Router as its own stage with bf16 input. Rejected: fp8 router input (can flip top-8
  choices; vLLM uses bf16); router in the gate_up stage (LDS overflow at M = 8).
- Expert GEMV mechanism: superseded by D1 (the Qwen3.6 MoE stage). The gate_up and down
  stages above keep their place in the stage list; their expert half is that stage.
- Deferred residual (each consumer rebuilds `h` and `h1` from raw stage outputs). Rejected: a
  separate combine stage per post-norm (2 to 3 more barriers per layer).

## Signed-off decisions (2026-10-27, revised the same day)

- **D1.** The MoE stage comes from the Qwen3.6 kernel (approved design there: grouped GEMV
  over the unique experts, per-slot partial sums, batch-invariant). Until it lands, this model
  does milestones 1-2 and the attention work, which has no MoE in it. Rejected: a second MoE
  stage written here in parallel (two implementations to keep correct).
- **D2.** Copy and diverge for now: `dev.py`, `gemv.py`, `fp4.py`, `vendor/` are copied into
  `mk/`. After Qwen3.6 milestone 3, the Qwen3.6 agent moves the shared code to `common/` in one
  separate commit and re-runs the Qwen3.8 and Qwen3.6 tests. Rejected: moving it now (the
  Qwen3.6 design changes `Gemv`: an `fp8` kind, piece-aligned wave ranges, every knob in `Cfg`;
  two agents editing `gemv.py` would conflict). Note: the Qwen3.8 kernel that vLLM exports comes
  from `repos/qwen38-gfx942-mk`, not from megakernels.
- **D3.** Attention tasks of 64 positions span several KV blocks; the kernel accepts vLLM's
  default block sizes (one block-table lookup per block). Rejected: require `--block-size 64`.
- Storage: `~/mk/models` on Alola is the one store for both agents' checkpoints. The home is
  at its quota; the user frees space. Until then this allocation reads the checkpoints from
  the node's `/tmp/g4models` (`scripts/stage_models.sh`).

## Open questions

- vLLM's attention backend and block sizes on gfx942 for this model: read the startup log of
  the vLLM image (milestone 4 baseline run).
- f32 -> bf16 rounding at exact ties: the attention stage's bf16 q (and K, V written to the
  cache) differs from `torch`'s round-to-nearest-even when an fp32 value lies exactly on a
  bf16 midpoint. Evidence (`tests/test_attention.py`, case "full M8", row 2 head 7): the
  model's o differs from the kernel's by 1.2e-3; with the 2 dims of q that sit exactly on a
  midpoint rounded the other way, by 3.1e-6. Effect: one-ulp differences on exact ties only.
  Open: which conversion FlyDSL emits on gfx942, and whether to match RNE (vLLM's kernels
  round to nearest even).
- Fixed per-stage latency with 6 stages per layer: measure in milestone 4; candidates are
  readiness counters for router -> gate_up and merging o_proj into the attention last arriver.
- Load balance of the gathered GEMV under I4 at M = 1 (8 experts x 3872 fp8 units of 1 KiB
  for gate_up): measure in milestone 3.
