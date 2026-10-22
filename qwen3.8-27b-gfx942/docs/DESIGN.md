# Qwen3.8-27B decode megakernel for gfx942 (FlyDSL)

## Fixed point

One decode step of the Qwen3.8-27B text model for M independent sequences
(M = 1, 2, 4, 8) runs as ONE persistent FlyDSL launch on one MI300X/MI325X
(gfx942, SPX, 304 CUs). The launch reads the token ids, the KV caches, the
GDN conv and SSM states, and the weights. It writes the logits (fp32), the
greedy token per row, and the updated caches and states. The result matches
an independent PyTorch reference of the same math within stated tolerances.

## Scope

In scope: embedding, 48 GDN layers, 16 full-attention layers, 64 MLPs,
final norm, LM head, greedy argmax, all state updates.

Out of scope: prefill (the reference fills caches and states), sampling
other than greedy, speculative decoding, vision, MTP, vLLM integration.

## Decisions

1. Weights: MXFP4 (e2m1, block 32, e8m0 scale) from
   `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`, re-laid offline. LM head and
   embedding stay bf16 (the checkpoint keeps them bf16).
   Evidence: decode is weight-bandwidth-bound; MXFP4 halves bytes vs FP8.
   Rejected: FP8 weights (2x bytes); MXFP4 activations (gfx942 has no MX
   MFMA, and A4 loses accuracy).
2. Activations: bf16 (W4A16). Accumulation: fp32. The checkpoint was
   calibrated for A-MXFP4; bf16 activations are strictly more precise.
3. Matrix math: `v_mfma_f32_16x16x16_bf16` (gfx942). A = activations
   (rows m, padded to 16), B = weights (16 output columns n). Weights are
   decoded e2m1 -> bf16 (unscaled, exact) with `v_perm_b32` byte lookups.
   The e8m0 scale is applied to the fp32 partial sum of each 32-wide K block,
   because the D layout gives each lane one fixed n.
4. Execution: grid = number of CUs (304), all CTAs co-resident, one CTA per
   CU. Stages are separated by device-scope grid barriers. Each CTA issues
   the weight loads of its next stage before it waits at a barrier, so HBM
   stays busy during the barrier.
5. Cross-XCD coherence: data written for other CTAs uses device-scope stores
   (`sc1`); readers use device-scope loads. Weights use normal or
   non-temporal loads.
6. Determinism: no float atomics. Split-K partial sums go to scratch and are
   summed in a fixed order. The same inputs give bitwise-equal outputs.
7. Caches: contiguous per-layer KV cache `[layer][row][kvh][pos][256]` bf16.
   GDN SSM state `[layer][row][48][128 v][128 k]` fp32, conv state
   `[layer][row][10240][3]` bf16.

## Invariants

- I1: every CTA of the grid is resident (grid <= CUs, LDS and VGPR use
  allow 1 CTA/CU). Checked at launch: CU count, kernel resource usage.
- I2: a stage reads only data published before the preceding barrier.
- I3: the megakernel output equals the reference within tolerance for
  every stage, checked per stage with debug dumps.
- I4: identical inputs give identical bits (determinism test).

## Correctness oracle

- `mk/reference.py`: PyTorch fp32 implementation of the decode step, using
  the same dequantized weights. It is checked against HF transformers bf16
  on the real checkpoint (logits correlation, greedy tokens).
- Stage dumps: the megakernel can write each stage's output to a debug
  buffer; tests compare each against the reference.

## Baselines

- B1: PyTorch eager decode step (reference module, bf16 weights).
- B2: stock serving engine on gfx942 (vLLM or SGLang ROCm), Qwen3.8-27B
  BF16 and FP8, decode TPOT at batch 1..8.

## Open questions

- Achievable HBM bandwidth from one persistent FlyDSL grid on MI300X:
  microbenchmark.
- Grid barrier latency on 304 CTAs across 8 XCDs: microbenchmark.
- Whether stock engines run Qwen3.8 (qwen3_5) on gfx942: try.
