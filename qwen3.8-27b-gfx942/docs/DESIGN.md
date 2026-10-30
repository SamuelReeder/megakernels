# Qwen3.8-27B decode megakernel for gfx942 (FlyDSL)

## Result

Batch 1 on one MI300X: 7.56 ms per output token (132 tok/s) against 15.59 ms for the
fastest stock engine on the same GPU (vLLM 0.29.1 + AITER, FP8 checkpoint): 2.06x.
Batch 2 / 4 / 8: 2.07x / 1.98x / 1.68x. Perplexity +0.8 % against the same MXFP4
weights with bf16 activations. Details and reproducibility runs: `bench/RESULTS.md`.

## Fixed point

One decode step of the Qwen3.8-27B text model for M independent sequences runs as
one persistent FlyDSL launch on one gfx942 GPU (MI300X: 304 CUs, SPX). The launch
reads the token ids, the KV caches, the GDN conv and SSM states and the weights. It
writes the fp32 logits and updates every cache and state.

## Scope

- In scope: embedding, 48 Gated DeltaNet (GDN) layers, 16 full-attention layers, 64
  MLPs, final norm, LM head, and every state update of a decode step. M = 1 to 16 rows
  (tested 1, 2, 4, 8).
- Out of scope: prefill (a prompt goes in one token per step), sampling (the host
  takes the argmax of the logits), speculative decoding, vision, MTP, multi-GPU,
  vLLM integration.

## Architecture

```
step = embed | L0 | L1 | ... | L63 | final norm + LM head
GDN layer  (i % 4 != 3): in_proj GEMV | GDN tasks       | out_proj GEMV | gate_up GEMV | down GEMV
attn layer (i % 4 == 3): qkv GEMV     | attention tasks | o_proj GEMV   | gate_up GEMV | down GEMV
```

`|` is a grid barrier: 5 per layer, 321 per step.

- Grid: one CTA per CU (grid = CU count), 512 threads (8 waves) per CTA. Every CTA
  must be resident: the kernel uses up to 256 VGPRs and the LDS of one CTA per CU.
- Grid barrier (`dev.grid_arrive` / `grid_wait`): every CTA stores a monotonic epoch
  to its flag line. CTA x < 8 waits for the CTAs b % 8 == x (one XCD under round-robin
  dispatch) and stores the epoch to leader line x. Every CTA waits for the 8 leader
  lines. Data for other CTAs is stored and loaded at device scope (`sc1`).
- GEMV (`mk/gemv.py`): `out[m, n] = sum_k x[m, k] W[n, k]` for M <= 16.
  - Unit = 16 output rows x 128 K = one 1 KiB MXFP4 tile + one scale dword per row.
    Units are ordered (K chunk, group, superstep) and stored in that order (`pack.
    unit_order`), so unit u is tile u.
  - Wave gw streams units [gw U / NW, (gw + 1) U / NW): one contiguous range, ping-pong
    blocks of P = 2 units. Blocks inside one piece (group x chunk) take a select-free
    path.
  - Split-K: each wave stores the partial of each piece it touches. A device-scope
    counter per group finds the last arriver, which sums the segments in a fixed order
    and runs the epilogue.
  - Math: e2m1 decodes to e4m3fnuz exactly (one `v_perm_b32` lookup per 4 values).
    `v_mfma_f32_16x16x32_fp8_fp8` with A = activations (fp8, one power-of-two scale
    per token and 128 K) and B = weights. The e8m0 scale multiplies each 32-K MFMA
    result. The LM head keeps bf16 weights and uses `v_mfma_f32_16x16x16_bf16`.
- Prologues (`mk/stages.py`) fill the CTA's LDS with the fp8 GEMV input:
  - norm (in_proj, qkv, gate_up): bf16 copy of the residual x rstd (from the fp32
    16-column sums of squares) x (1 + w);
  - act (down): the act vector written by the gate_up epilogue;
  - scaled (out_proj, o_proj): the GDN output with its per-head RMS, or the attention
    output.
- Epilogues: `EpiConv` (in_proj: causal conv + SiLU on the q/k/v channels, conv
  state shift), `epi_raw` (qkv, LM head), `epi_act` (gate_up: rows interleave gate
  and up per 8 columns, so one group gives 8 act columns), `EpiRes` (out/o/down:
  residual add, 16-column sums of squares, bf16 copy).
- GDN task (`mk/mixer.py`): one wave per (row, value head, 16 value rows). l2norm of
  q, k, the gated delta rule on a 16 x 128 fp32 state slice (stored lane-interleaved:
  one contiguous KiB per load instruction), the gated RMSNorm numerator. Consecutive
  tasks go to different CUs, last waves first.
- Attention task: one wave per (row, kv head, 64 positions). RMSNorm + RoPE of q and
  k_new, QK^T on the bf16 MFMA, softmax, P V on the VALU, a partial record. The last
  arriver of (row, kv head) merges the records and applies the output gate.

## Reuse

From `common/mkcore` (`common/DESIGN.md`): `dev` (barrier, DPP reductions, buffer
loads), `gemv.Gemv` (the split-K MXFP4 / bf16 stream), `fp4` (MXFP4 tile layout),
`pack` (`gu_order`, `pack_bf16`), `stagelib` (`store_quant8`), `key` (compile key) and
`vendor.buffer_ops`. In `mk/`: `dims.py`, `kernel.py` (`Cfg`, `build`, stage schedule),
`stages.py` (`rstd_from_ssq`, `prologue_scaled`, the norm prologues, the TP all-reduce
epilogues), `mixer.py`, `reference.py`, `runner.py`, `pack.py` (`Packed`, TP sharding),
and the TP code `sync.py`, `peer.py`, `tp.py`, `tpdev.py` (trace state, R-state).

## Numerics

| Quantity | Format |
|---|---|
| Linear weights | MXFP4 of `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`, decoded exactly |
| LM head, embedding | bf16 (as in the checkpoint) |
| GEMV inputs | e4m3fnuz, power-of-two scale per (token, 128 K) |
| Accumulation, residual stream, GDN state, conv state | fp32 |
| Norm input | bf16 copy of the fp32 residual; rstd from the fp32 residual |
| KV cache | bf16 |
| q in QK^T | bf16 |

`mk/reference.py` implements the same step in PyTorch. Exact mode uses bf16 GEMV
inputs (the model dtype). Emulate mode applies the kernel's fp8 quantization and bf16
norm input.

## Decisions and evidence

1. MXFP4 weights. Decode is weight-bandwidth-bound; the linear layers read 12.9 GB
   per step in MXFP4 (with scales) against 25.1 GB in FP8. The bf16 LM head adds 2.5 GB.
2. fp8 activations with the fp8 MFMA. gfx942 has no MX MFMA. Decoding e2m1 to e4m3fnuz
   is exact and costs one byte permute per 4 values; a bf16 path would need twice the
   MFMA work. Cost on real text: perplexity +0.8 % against the same weights with bf16
   activations (`tests/eval_text.py`, 592 tokens).
3. Split-K with a fixed-order last arriver, no float atomics. Same inputs give
   bitwise-equal logits (`tests/test_determinism.py`); a row's logits do not depend on
   the other rows of the batch (`tests/eval_text.py`, batch-invariance check).
4. Grid barriers, not a dataflow graph. With 304 CTAs a barrier costs about 2 us
   (microbenchmark) and 3 us in the kernel including the store drain.
5. K chunks: out/o_proj 2, down 4, LM head 8, the others 1 (measured; see below).

## Measured and rejected (M = 1 unless noted)

| Change | Result |
|---|---|
| Prefetch the next GEMV's first units before the barrier arrival (NPRE 2/4/8, stores drained by `vmcnt`) | 8.06 / 8.17 / 8.22 ms against 7.89: slower |
| Deeper prefetch after the arrival (NPRE 4 / 8) | 8.01 / 8.09 ms against 7.89 |
| Whole out/o_proj weight slice in registers during the mixer stage | 7.94 ms against 7.90 |
| Consumers sum the raw GEMV partials (no last arriver) for in_proj, qkv, gate_up | 8.28 ms against 8.21 |
| Weighted unit split between the two waves of a SIMD (9:7, 5:4) | the second wave still ends at the same time; tails slower |
| P = 1 / 3 / 4 units per pipeline block | P = 3 equal to P = 2; P = 1 and 4 slower |
| Cache policy of weight loads (default, nt, sc0) | no difference |
| K chunks 4 or 8 for in_proj / qkv / gate_up | M = 8 slightly faster, M = 1 slower: not taken |
| Ping-pong of the attention K and V loads | no gain (QK was compute-bound until it moved to the MFMA) |

Measured and kept, in order:

| Change | Effect |
|---|---|
| Opaque thread ids per stage (no hoisting of per-stage address math out of the layer loop) | VGPR spills 424 -> 66 (0 after the GEMV changes below) |
| Wave index made wave-uniform; unit cursors; unit-order tiles; select-free blocks | GEMV loop SALU per 2-unit block 352 -> 20; gate_up stream (slowest wave) 29.9 -> 26.8 us |
| Mixer tasks spread over CUs | GDN stage 12.4 -> 7.6 us |
| Attention QK^T on the bf16 MFMA | attention stage 44 -> 25 us (ctx 512) |
| Residual epilogue issues its h load with the partial loads | out/o/down hand-off -1 round trip |
| Causal conv in the in_proj epilogue | GDN stage 7.3 -> 6.4 us |
| GDN state lane-interleaved | GDN stage 6.4 -> 4.7 us (M = 8: 38 -> 15.5 us) |
| DPP row reductions | norm prologue quantization M = 8: 6.0 -> 4.3 us |
| gate/up interleaved per 8 columns, act in the epilogue | down stage 20.4 -> 18.4 us (M = 8: 33.8 -> 24.2 us) |
| bf16 copy of the residual for the norm prologues | M = 8: 11.1 -> 10.1 ms |

## Where the time goes (M = 1, ctx 512, `bench/profile_decode.py`)

Per GDN layer: in_proj 19 us + barrier 5, GDN tasks 5 + 4, out_proj 12 + 4, gate_up
32 + 4, down 18 + 6. The weight stream runs at 3.3 to 3.7 TB/s; the rest is fixed
per-stage latency (prologue about 2.5 us, last-arriver hand-off 2 to 5 us, barrier
about 3 us). The LM head (2.5 GB bf16) takes 0.65 ms. Inside a CU the older wave of
each SIMD finishes its stream about 4 us before the younger one; the CU's total
throughput sets the stage end (weighted splits do not help).

## Invariants

- I1: every CTA is resident (grid = CU count, one CTA per CU by LDS and VGPR use).
- I2: a stage reads only data published before the preceding barrier, or by a last
  arriver whose counter it observed.
- I3: the kernel is as close to the exact reference as the emulating reference is
  (`tests/test_decode.py`), on all 64 layers.
- I4: identical inputs give identical bits; a row does not depend on the other rows.

## Correctness oracle

- `tests/ref_vs_hf.py`: reference against HF transformers on the BF16 checkpoint.
- `tests/test_decode.py`: kernel against the reference in both modes, every step.
- `tests/eval_text.py`: teacher-forced perplexity on four real passages (kernel, the
  MXFP4 reference with bf16 activations, the BF16 model), top-1 agreement, KL, batch
  invariance.
- `tests/test_determinism.py`: three runs, bitwise-equal logits.

## Run-time geometry (rt2, 2026-10-29)

`Cfg.rt_geom` 2: `maxlen` and `maxblocks` are read from glob slot `G_GEOM` at run time,
so one build serves every `max_model_len`. Default 2 at TP1, 0 at TP2 (TP2 not checked).
Evidence (`bench/results/rt2/`, MI325X, job 68211382):
- Logits hash rt0 == rt2 at M = 1, 2, 4, 8; `test_decode` and determinism pass at M = 1, 8.
- TPOT rt2 vs rt0 (median of 3, ctx 512): M = 1 −0.47 %, M = 8 +0.29 %.
- One ELF for maxlen 4096, 8192, 16384 (`isa_local.txt`); rt0 ISA equals main's.
- vLLM (`vllm/`, TP1, greedy, 128 tokens): base vs rt2 tokens equal at 4096 and 8192.
  Prompt 1's logprobs differ by up to 0.145 in both. Cause: vLLM run-to-run variation,
  not rt2. Prompt 1 has two outputs in vLLM (logprob at position 1 −1.212131 or
  −1.250210, the rest follows), and rt0 kernels give both: the common/ migration's M8
  base run (`cm8/tp1_base_r1.json`, pre-migration rt0 export) equals the rt2 runs on
  prompt 1 bit for bit (`vllm/modes.txt`). The spike saw the same 0.145 once at 16384,
  between two runs of one build. At vLLM's geometry (block 784, x-packed KV, maxlen 4096
  and 8192, 1800 steps from position 0, M = 1, 2, 4, 8) rt0 and rt2 logits are bitwise
  equal every step (`common/results/rt2/vgeom.py`, `common/results/rt2/rebase/`).
- After the rebase onto main `e72a981` (common/ migration): rt0 ISA equals main's, one
  rt2 ELF across maxlens (`isa_rebase.txt`); hash rt0 == rt2, every hard gate PASS, TP2
  (rt0) gates PASS (job 68213193, `common/results/rt2/rebase/summary.txt`).

## Limits and open questions

- Prefill runs one token per step; a real deployment needs a prefill path.
- The fixed per-stage latency (about 40 % of the step at M = 1) is the next target:
  fewer stages per layer (for example the GDN core inside the in_proj tail) or
  readiness counters in place of grid barriers.
- An fp8 LM head would save about 0.3 ms but changes the checkpoint's bf16 LM head.
- MI325X (AUSTIN nodes) was not run: those nodes do not share the home directory.
