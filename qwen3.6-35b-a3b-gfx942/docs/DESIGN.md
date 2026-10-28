# Qwen3.6-35B-A3B decode megakernel for gfx942 (FlyDSL)

Status: milestones 1 to 7 done. The full suite passes on a second day and on a second node,
an MI325X (`bench/results/m5/`). The kernel decodes at 3.42 / 3.99 / 4.97 / 6.37 ms per token
at M = 1 / 2 / 4 / 8 (512-token context, MI300X), which is 1.49x to 1.17x the fastest stock
engine on the same GPU ("Optimization log"). The design
reuses the Qwen3.8-27B kernel (`../qwen3.8-27b-gfx942/`) for every stage that the two
models share, and adds a mixture-of-experts (MoE) block in place of the dense MLP.

## Fixed point

One decode step of the Qwen3.6-35B-A3B text model for M independent sequences
(1 <= M <= 8) runs as one persistent FlyDSL launch on one gfx942 GPU (MI300X: 304 CUs,
SPX mode). The launch reads the token ids, the positions, the KV cache, the GDN conv
and SSM states and the weights. It writes the fp32 logits (or the normed hidden states
for a serving engine's own LM head) and updates every cache and state in place, in
vLLM's cache layouts.

Properties that stay true if the implementation is discarded:

- P1. Identical inputs give identical output bits.
- P2. A row's output bits do not depend on the other rows of the batch: not on their
  tokens, not on the experts they select, and not on M.
- P3. The kernel output is as close to the exact fp32 reference as the emulating
  reference (which copies the kernel's rounding) is.
- P4. Every compile-time setting is a field of `Cfg`, and the kernel name contains a
  hash of `repr(cfg)`. FlyDSL's disk cache keys on the name, not on dataclass fields.

## Scope

- In scope: embedding, 30 Gated DeltaNet (GDN) layers, 10 gated full-attention layers,
  40 MoE blocks (router, top-8, routed experts, shared expert with its sigmoid gate),
  final norm, LM head, and every state update of a decode step. M = 1 to 8 rows
  (tested 1, 2, 4, 8). Greedy decoding on the host (argmax of the logits). One GPU, TP1.
- Weights: the FP8 checkpoint `Qwen/Qwen3.6-35B-A3B-FP8` is the main path. The BF16
  checkpoint runs through the same GEMV with the `bf16` weight format (see "Weight
  formats"); it is added only after the FP8 path passes milestone 3, and only if it
  needs no new stage code.
- Out of scope: prefill (a prompt goes in one token per step), MTP and speculative
  decoding, TP > 1, sampling in the kernel, the vision tower.

## Model facts

Read from `config.json` of both checkpoints and from the safetensors headers on
2026-10-27. The kernel asserts each value against the config at load time.

| Quantity | Value |
|---|---|
| Layers | 40; layer i is full attention iff i % 4 == 3 (30 GDN, 10 attention) |
| Hidden H | 2048; RMSNorm eps 1e-6, zero-centred (`x * (1 + w)`) |
| Vocabulary | 248320; embedding and LM head separate (not tied), bf16 in both checkpoints |
| GDN | 16 K heads, 32 V heads (V head h uses K head h // 2), head dims 128, conv kernel 4 |
| GDN projections | `in_proj_qkv` 8192 x 2048 (q 2048, k 2048, v 4096), `in_proj_z` 4096 x 2048, `in_proj_b` / `in_proj_a` 32 x 2048, `out_proj` 2048 x 4096 |
| GDN extras | `conv1d` [8192, 1, 4], `A_log` [32], `dt_bias` [32], gated RMSNorm weight [128] (plain `w`, not `1 + w`), state fp32 (`mamba_ssm_dtype`) |
| Attention | 16 Q heads, 2 KV heads (8 Q heads per KV head), head_dim 256, output gate |
| Attention projections | `q_proj` 8192 x 2048 (per head [q 256, gate 256]), `k_proj` / `v_proj` 512 x 2048, `o_proj` 2048 x 4096; q_norm / k_norm [256] |
| RoPE | theta 1e7, partial factor 0.25: NeoX rotation of dims 0..63. `mrope_interleaved` with sections [11, 11, 10]: for text the three position ids are equal, so it is plain RoPE |
| MoE | 256 experts, top-8, expert gate / up / down 512 x 2048, 512 x 2048, 2048 x 512 |
| Routing (HF) | softmax over 256 in fp32, top-8, divide by the sum of the 8 (always), cast to bf16 |
| Shared expert | gate / up / down 512 x 2048, 512 x 2048, 2048 x 512; output times `sigmoid(shared_expert_gate(x))`, gate [1, 2048] |
| FP8 checkpoint | e4m3fn (OCP), block scales 128 x 128 (`weight_scale_inv`, bf16). Not quantized: embedding, LM head, router `mlp.gate`, `shared_expert_gate`, `in_proj_a`, `in_proj_b`, `conv1d`, norms |
| MTP | 1 layer; not used |

Per-step weight bytes (FP8 path; GDN state read and write included):

| M | GDN layers | attention layers | MoE (unique experts U) | LM head (bf16) | total | at 4 TB/s |
|---|---|---|---|---|---|---|
| 1 | 1.14 GB | 0.27 GB | 1.18 GB (U = 8) | 1.02 GB | 3.60 GB | 0.90 ms |
| 2 | 1.27 GB | 0.27 GB | 2.15-2.18 GB (U <= 16) | 1.02 GB | 4.7 GB | 1.18 ms |
| 4 | 1.52 GB | 0.27 GB | 4.0-4.2 GB (U <= 32) | 1.02 GB | 6.8-7.0 GB | 1.70-1.75 ms |
| 8 | 2.02 GB | 0.27 GB | 7.4-8.2 GB (U <= 64) | 1.02 GB | 10.7-11.5 GB | 2.68-2.88 ms |

The low MoE value assumes uniform random routing; real routing is correlated, so U is
measured, not assumed (see "Open questions"). One routed expert is 3.15 MB in FP8; the
router is 1.05 MB in bf16. The GDN SSM state is 2 MB per row per layer (fp32), read and
written every step, so at M = 8 it adds 0.96 GB.

## Architecture

```
step = embed | L0 | L1 | ... | L39 | final norm + LM head
GDN layer  (i % 4 != 3): in_proj | GDN tasks       | out_proj | router | experts gate_up | experts down
attn layer (i % 4 == 3): qkv     | attention tasks | o_proj   | router | experts gate_up | experts down
```

`|` is a grid barrier: 6 per layer, 1 + 6 x 40 = 241 per step (Qwen3.8: 321).

Unchanged from Qwen3.8 (with the dimensions above): the grid (one 512-thread CTA per
CU, grid = CU count, all CTAs resident), the two-level grid barrier with the device
epoch, the flat unit-stream GEMV with split-K and a fixed-order last arriver, the stage
prologues (norm, scaled, act), the residual epilogue `EpiRes`, the conv epilogue
`EpiConv`, the GDN task, the attention task and its merge, the vLLM cache layouts
(paged KV for TRITON_ATTN and ROCM_ATTN, slot-indexed GDN state), and the pointer
tables.

### Per-layer stage list

| # | Stage | Prologue (LDS input) | Work | Last-arriver epilogue | Output |
|---|---|---|---|---|---|
| S1 | in_proj (GDN) | norm(h0) -> fp8 x and bf16 x | fp8 GEMV 12288 x 2048 (qkv, z); bf16 GEMV 64 x 2048 (b, a) | `EpiConv` (conv + SiLU on q/k/v, conv state shift), raw z, b, a | y_in fp32 [M][12352] |
| S1 | qkv (attention) | norm(h0) -> fp8 x | fp8 GEMV 9216 x 2048 (q+gate 8192, k 512, v 512) | raw | y_qkv fp32 [M][9216] |
| S2 | GDN tasks | none | M x 32 x 8 wave tasks (Qwen3.8 `gdn_task`) | none | y, per-head ssq |
| S2 | attention tasks | none | M x 2 x chunks wave tasks; KV write | none | partial records [M][2][chunk] (max, sum, o of 8 heads) |
| S3 | out_proj / o_proj | scaled(y), or the merge of the head's records x sigmoid(gate) (`prologue_attn`) -> fp8 x | fp8 GEMV 2048 x 4096 | `EpiRes` | h1, ssq1, bf16 copy |
| S4 | router | norm(h1) -> bf16 x | bf16 GEMV 272 x 2048 (256 router rows, 1 shared-gate row, 15 zero rows) | raw | logits fp32 [M][272] |
| S5 | experts gate_up | routing; norm(h1) -> fp8 x | grouped fp8 GEMV, (U + 1) x 1024 x 2048 | `EpiAct` per (position, group) | act fp32 [M][9][512] |
| S6 | experts down | act of the CTA's <= 2 positions -> fp8 x | grouped fp8 GEMV, (U + 1) x 2048 x 512 | `EpiMoe`: weighted slot sum + `EpiRes` | h0, ssq0, bf16 copy |

Final stage: norm(h0) -> bf16 x, bf16 LM head GEMV 248320 x 2048, fp32 logits. With
`lm_head = False` the stage writes the normed hidden states (Qwen3.8 `final_norm_stage`).

Small GEMVs (fewer units than the 2432 waves: the router, 272 units, and in_proj b / a,
64 units) give wave gw < U unit gw instead of the Qwen3.8 spread
[gw U / NW, (gw + 1) U / NW). The spread leaves empty waves inside a piece, and the
last-arriver count assumes the segments of a piece sit in consecutive waves. The
in_proj b / a GEMV uses its own partial region after the fp8 GEMV's (both run in S1).

### MoE stage design

Terms:

- Slot (m, j): row m's j-th expert, j = 0..7 in top-k rank order. Slot j = 8 is the
  shared expert for every row.
- Position p: an index into the step's list of unique selected experts, sorted by
  expert id, followed by the shared expert as the last position. U unique routed
  experts give U + 1 positions. U <= min(8 M_valid, 256).

#### Routing (S4, then the start of S5)

1. S4 computes the router logits as one bf16 GEMV of 272 rows: rows 0..255 are
   `mlp.gate`, row 256 is `shared_expert_gate`, rows 257..271 are zero. The input is
   bf16(norm(h1)), the model dtype. The epilogue writes fp32 logits (no bf16 rounding).
   Split-K and the last arriver are the Qwen3.8 GEMV.
2. After the S4 barrier, every CTA computes the routing for all M rows. Wave m does
   row m: top-8 of the 256 fp32 logits (8 rounds of a wave max; ties go to the lower
   expert id), weights w[m][j] = exp(l_j - l_0) / sum over the 8 in fp32 (equal to HF's
   softmax, top-8, renormalize), shared weight w[m][8] = sigmoid(l_256).
3. Wave m stores row m's 256-bit expert set (8 words, built from its wave-uniform
   picks; a padding row selects none). Wave 0 ORs the rows' words, builds the position
   list (expert ids in ascending order, then the shared expert), U, and slot_of[m][p]
   (the slot j that row m uses for position p, or -1). The tables stay in a reserved
   LDS region (about 1.5 KB at M = 8) for S5 and S6.
4. A padding row (vLLM marks it with slot index 0 / slot < 0) selects no routed expert.
   Its outputs are don't-care. Without this rule, a batch of 3 padded to 4 would stream
   8 extra experts per layer.

All CTAs compute the same tables: they run the same code on the same logits, which S4
published at device scope before the barrier. A debug mode stores a hash of each CTA's
tables; `tests/test_decode.py` checks that all 304 hashes are equal (invariant I5).

Rejected: one CTA computes the routing (S4 last arriver of the last arriver) and
publishes it. It adds one more device-scope round trip on the critical path, and the
redundant computation costs about 1 us in each CTA, in parallel.

#### Expert selection for M > 1, and how expert weights stream (S5, S6)

The grouped GEMV is the Qwen3.8 flat unit stream with three changes:

1. The stream runs over the U + 1 selected positions. Piece (p, g, c) is position p,
   output group g (16 rows), K chunk c. Pieces are ordered (position, chunk, group).
   The piece count T = (U + 1) x G x C is known only at run time, and every CTA has
   the same value.
2. Wave ranges are whole pieces: global wave gw takes pieces [gw T / NW, (gw + 1) T /
   NW), NW = 304 x 8. One wave computes all units of a piece, in K order. The value of a
   piece partial therefore does not depend on T, which depends on the other rows'
   routing (property P2). The Qwen3.8 dense GEMV can split a piece between two waves
   at a T-dependent point; that would break P2 here.
3. The weight address of a unit is `base[expert(p)] + local unit offset`. The CTA reads
   expert(p) from the LDS position list once per piece (wave-uniform). Within an expert
   the tiles are in unit order, so the loads inside a piece are contiguous, as in
   Qwen3.8.

Each piece is computed for all M rows: the MFMA has 16 token rows, and M <= 8, so the
rows that did not select the expert cost no extra weight traffic. Their results are
dropped by the epilogue. Every selected expert is read once per step, however many
rows selected it.

| GEMV | per expert | G | C (start value) | units per piece | pieces at M = 1 / M = 8 (U = 64) |
|---|---|---|---|---|---|
| gate_up (S5) | 1024 x 2048, 2 MB | 64 | 4 (512 K) | 4 | 2304 / 16640 |
| down (S6) | 2048 x 512, 1 MB | 128 | 1 (512 K) | 4 | 1152 / 8320 |

With 2432 waves, M = 1 gives at most one piece (8 KB) per wave, and M = 8 at most 7.
A CTA's range covers at most 110 down units (M = 8), fewer than one expert's 512, so a
CTA touches at most 2 positions in S6. The builder asserts this bound.

gate_up rows are interleaved per 8 columns (`gu_order`, as in Qwen3.8): one 16-row
group holds the gate and up rows of 8 act columns.

#### How partial sums reduce

S5 (gate_up): a device-scope counter per (position, group) counts C arrivals. The last
arriver sums the C chunk partials in chunk order, computes act = silu(gate) x up for
the group's 8 columns, and stores act[m][slot_of[m][p]][8 columns] for each row m with
slot_of[m][p] >= 0. Other rows store nothing.

S6 (down): the S6 prologue loads, for each of the CTA's <= 2 positions and each row m,
act[m][slot_of[m][p]] and quantizes it to fp8 (pow2 scale per 128 K) into LDS. Rows
that did not select p get zero fp8 values and a zero scale. A wave that finishes piece
(p, g, c) stores, for each row m that selected p, its 16 values to
`part[g][slot][c][col][m]` (slot = slot_of[m][p]). Counting has two levels: counter
(p mod 4, g) counts C arrivals per position of its sub-set; the last arriver of each
non-empty sub-counter increments counter g, which counts min(U + 1, 4) arrivals. (One
counter per g with (U + 1) x C arrivals serialized about 40 atomics on one address at
M = 8.) The last arriver of g loads, per lane, the 9 slots x C float4
values of its rows, and computes for each row m:

```
moe[m] = sum over j = 0..8 in order of w[m][j] * (sum over c in order of part[g][j][c][m])
h0[m]  = h1[m] + moe[m]      (EpiRes: fp32, ssq0, bf16 copy)
```

The order of the sum depends only on row m's own slots, so the sum is batch-invariant
(P2), and no float atomics are used (P1). The routing weight is applied in fp32 after
the down projection, in the order HF uses.

Rejected for S6: partials stored per position and summed over all U + 1 positions in
the last arriver. At M = 8 the last arriver would load 65 partials per lane (about 15
us of serial L2 round trips); the per-slot layout needs 9.

### Weight formats and the GEMV interface

`Gemv(kind=...)` selects the weight format. A format defines the unit's tile bytes,
its scale data, the load of one unit, and the compute of one unit (4 fp32 values per
lane: 4 tokens of one output row). Packing (`mk/pack.py`) produces tiles in unit order.

| kind | unit (16 rows x 128 K) | scales | A operand | MFMA |
|---|---|---|---|---|
| `fp8` (new, main path) | 2 KB e4m3fnuz | 2 fp32 per unit: rows 0-7 and rows 8-15 (the 128 x 128 block scales x 2) | fp8 x, pow2 scale per (token, 128 K) | `v_mfma_f32_16x16x32_fp8_fp8`, 4 per unit |
| `bf16` (Qwen3.8) | 4 KB bf16 | none | bf16 x | `v_mfma_f32_16x16x16_bf16`, 8 per unit |
| `mx` (Qwen3.8, later) | 1 KB e2m1 | 1 dword of 4 e8m0 per row | fp8 x, pow2 scale | fp8 MFMA after an exact e2m1 -> e4m3fnuz decode |

FP8 GEMV math per unit: the 4 MFMAs accumulate in one fp32 vector t (no per-32-K
scaling, unlike `mx`), then `acc += t * (ws[col < 8 ? 0 : 1] * xs[token])`. The two
weight scales are one 8-byte wave-uniform load per unit.

FP8 conversion at pack time: gfx942 MFMAs take e4m3fnuz, whose exponent bias is 8 (OCP
e4m3fn: 7). The same bits are worth half in e4m3fnuz, so the packer keeps the bits,
maps 0x80 (-0 in e4m3fn, NaN in e4m3fnuz) to 0x00, and doubles the scale. This is
exact. The packer asserts that the checkpoint holds no e4m3fn NaN (0x7F, 0xFF).

The 2-scale unit covers every FP8 matrix: block rows are 128-aligned in all of them
(8192, 4096, 512 and 2048 are multiples of 128), and the gate_up interleave puts 8 gate
rows and 8 up rows in one group. Rejected: one fp32 scale per row and unit (64 B per 2
KB unit, 3 % extra traffic) and one scale per unit (wrong for the interleaved gate_up).

MXFP4 later: add the `mx` format from Qwen3.8 (`mk/fp4.py`) and a packer path. No stage
changes, because stages call only `Gemv.load` / `Gemv.compute` / `Gemv.run`.

BF16 checkpoint: every linear uses `kind = bf16` and the bf16 prologues. LDS at M = 8
stays below 64 KB (S5 x 32 KB, S6 x 16 KB, in_proj 32 KB). Weight bytes double.

### LDS budget (M = 8, gfx942 limit 64 KB per CTA)

| Region | Bytes |
|---|---|
| S1 in_proj: fp8 x 16.5 KB + scales 1 KB + bf16 x 32.9 KB | 50.4 KB |
| S4 router: bf16 x | 32.9 KB |
| S5: fp8 x + scales | 17.5 KB |
| S6: 2 positions x 8 rows fp8 x + scales | 9.0 KB |
| S2 attention: 8 waves x (q bf16 8 heads x 256 = 4 KB, k_new / v_new 1 KB) | 40 KB |
| S2 GDN: 8 waves x 1 KB | 8 KB |
| Routing tables (persistent across S5, S6: pick, w, eid, posof [256], slot [M][65]) + `red` | 3.5 KB |

The stage regions form a union (as in Qwen3.8); the maximum is 53.9 KB. The builder
computes the total and stops with `STOP: I1` above 64 KB.

Required change from Qwen3.8: the attention task keeps q in LDS as bf16, not fp32.
With 8 Q heads per KV head the fp32 layout needs 8 x (8 KB + 1 KB) = 72 KB. q is
rounded to bf16 before the score MFMA anyway, so the numerics do not change.

### Barrier count

241 grid barriers per step: 1 after the embedding, then 6 per layer (after S1..S6;
the barrier after S6 of layer 39 precedes the LM head). Qwen3.8 measured about 3 us
per barrier in the kernel plus 2.5 us prologue and 2 to 5 us hand-off per stage.

Expected time at M = 1: 0.9 ms of weight stream plus about 240 x 7 to 9 us of fixed
stage cost (1.7 to 2.2 ms), so about 2.6 to 3.1 ms per token. The fixed cost dominates;
milestone 4 attacks it (see "Open questions"). This is an estimate, not a measurement.

## Numerics

| Quantity | Format |
|---|---|
| Linear weights | FP8 checkpoint bits as e4m3fnuz, 2 fp32 scales per unit (exact) |
| Router, shared gate, in_proj_a / in_proj_b, LM head, embedding | bf16 (as in the checkpoint) |
| GEMV inputs, fp8 weights | e4m3fnuz, power-of-two scale per (token, 128 K) |
| GEMV inputs, bf16 weights | bf16 |
| Router logits, shared-gate logit | fp32 (HF and vLLM round them to bf16) |
| Routing weights, shared gate sigmoid | fp32 (HF casts the weights to bf16; vLLM keeps fp32) |
| Expert act (silu(g) x u) | fp32 in global memory, fp8 in the down GEMV input |
| Accumulation, residual stream, GDN state | fp32 |
| Conv state, KV cache | bf16 (vLLM's layouts) |
| Norm input | bf16 copy of the fp32 residual; rstd from the fp32 residual |
| q in QK^T | bf16 |

Decision (confirmed 2026-10-27): fp32 router logits, ties to the lower expert id.
Rejected: bf16 logits to copy HF and vLLM. bf16 logits make exact ties common, and the
tie order of `torch.topk` and of vLLM's top-k kernel is not defined, so tests would fail
at random. The kernel's GEMV also adds in a different order from rocBLAS, so bf16
rounding would not make the kernel select bit-for-bit the same experts as vLLM anyway.
The tests allow near-ties (see "Correctness oracle").

## Decisions and evidence

1. Grid barriers and stages as in Qwen3.8 - the Qwen3.8 kernel is correct, determinate
   and 2.06x the fastest stock engine at batch 1 with this structure.
2. Grouped GEMV over unique selected experts with a run-time piece count - every
   selected expert is read once; all 2432 waves share the work at every U.
3. Piece-aligned wave ranges in S5 and S6 - required by P2 (see "MoE stage design").
4. Shared expert as position U, slot 8 - S5 and S6 need no second code path.
   Rejected: shared gate_up in S4 and shared down in S5. It balances S4 better but
   needs two formats in one stage. Kept as a milestone 4 experiment.
5. Routing computed in every CTA - no extra round trip; checked by the routing hash.
6. Per-slot partials with a fixed-order weighted sum - P1, P2, 9 loads per hand-off.
7. FP8 weights repacked into unit order (a copy) - the Qwen3.8 GEMV address math and
   coalescing are proven. In vLLM the copy costs about 34 GB of the 192 GB HBM.
   Rejected for now: reading vLLM's row-major tensors in place (see "Open questions").
8. bf16 GEMV for in_proj b / a inside S1 - keeps the checkpoint's bf16 weights.
   Rejected: quantizing b / a to fp8 (changes the decay and beta gates; vLLM runs them
   in bf16), and computing them in the GDN tasks (8 tasks per head would each read the
   8 KB of weights and the input, and the GDN stage would get 2 more round trips).

## Rejected alternatives

| Alternative | Reason |
|---|---|
| Stream all 256 experts as a dense GEMV | 805 MB per layer instead of 25 MB at M = 1 |
| Static partition by (row, slot), no deduplication | Rows that share an expert read it again: up to 8x for a popular expert at M = 8 |
| Static partition over the maximum number of positions (8 M + 1) | Waves of unused positions idle; at M = 8 with U = 50, 14 of 65 positions (22 %) idle |
| Router GEMV in every CTA (no S4 barrier) | 304 x 1 MB of L2 reads per layer, about 10 us |
| Router GEMV in one CTA | 1 MB on one CU, about 10 to 20 us |
| Float atomics to add expert outputs into h | Not determinate (P1) |
| Qwen3.8 flat stream (T-dependent split-K) for the expert GEMVs | Breaks P2 |
| Routing weight folded into act before the fp8 quantization | Different rounding from HF; no gain |
| Token sorting and padding per expert (fused-MoE style grouped GEMM) | M <= 8 fits in one MFMA tile; masking is free |
| One CTA per expert | At M = 1 only 9 experts: 9 of 304 CUs stream |
| fp32 q in the attention LDS | 72 KB > 64 KB |

## Stock baselines (milestone 4)

One MI300X (banff-cyxtera-s70-4), 2026-10-27, 512 random input tokens, 256 output
tokens, TP1, prefix caching off, engine-default graphs. vLLM: nightly image
`vllm-rocm-nightly-00b6c4c5` (vLLM 0.32.1rc1.dev505+g00b6c4c56), `vllm bench serve` at
max-concurrency B. SGLang 0.5.10.post1 (mi30x image), `sglang.bench_one_batch`.
Median TPOT in ms (batch 1: median of 3 runs). Raw data and env: `bench/results/stock/`,
table: `bench/results/stock_summary.md`, scripts: `bench/stock/`.

| Checkpoint | Engine | B = 1 | B = 2 | B = 4 | B = 8 |
|---|---|---|---|---|---|
| FP8 | vLLM | 6.43 | 6.67 | 7.20 | 8.95 |
| FP8 | vLLM + `VLLM_ROCM_USE_AITER=1` | **5.11** | **5.51** | **6.17** | **7.44** |
| FP8 | SGLang | 7.79 | 7.98 | 8.51 | 9.23 |
| BF16 | vLLM | 5.24 | 5.78 | 7.00 | 9.86 |
| BF16 | vLLM + AITER | 4.93 | 5.38 | 6.48 | 8.81 |
| BF16 | SGLang | 6.13 | 6.09 | 6.83 | 8.79 |

The fastest FP8 baseline (vLLM + AITER) is the reference for the speedups.

## Optimization log (milestone 4)

Start: the milestone 3 kernel, 3.72 ms at M = 1 and 6.71 ms at M = 8
(`bench/results/m3/bench_untuned_m*.txt`). End: 3.42 / 3.99 / 4.97 / 6.37 ms at
M = 1 / 2 / 4 / 8 (`bench/results/m4/bench_v5_m*.txt`), 1.49x / 1.38x / 1.24x / 1.17x
vLLM + AITER on the FP8 checkpoint.

Tools: `bench/profile_decode.py` (per-stage span, busy and barrier time from per-CTA
stamps; `PROBE=<stage>` adds per-wave stamps), `scripts/isa_stats` (VGPRs, spills).
The profile measures one greedy step after a 512-token random prompt per row (distinct
rows: an early version decoded token 0 in every row, which gave U = 9 at M = 8 and
hid the MoE cost). Stage numbers: median over layers, us.

| Change | Measured effect | Kept |
|---|---|---|
| LM head K chunks 8 -> 1 (`c_lm`) | LM head 315 -> 224 us at M = 1: with 8 chunks each wave ended 51 pieces (51 partial hand-offs, 78 us). TPOT sweep 8/4/2/1: 3.361 / 3.310 / 3.283 / 3.263 ms (ctr node, before the other changes) | yes |
| Attention merge in the o_proj prologue (`mixer.prologue_attn`), o_proj K chunks 2 -> 16 (one head per CTA) | attention stage 34.4 -> 20.6 us (the last-arriver merge read 8 records in 4 dependent rounds, 10 us), o_proj 13.7 -> 19.5 us; -8 us per attention layer. Bitwise the same merge order | yes |
| e2 first piece loaded between the grid arrive and wait (routing tables already in LDS) | S6 8.3 vs 8.4 us at M = 1 (A/B in one build, `MK_CFG`) | no, removed |
| Router logits loaded before the S5 norm prologue, routing after it | S5 17.6 vs 17.7 us | no, removed |
| Routing reductions with offsets 1, 2, 4, 8 first (DPP) instead of 32 first (6 ds_bpermute) | S5 17.7 -> 15.2 us at M = 1 | yes |
| DPP-only wave reductions (`dev.wave_max`, `wave_sum`: row butterfly, row_bcast:15 / :31, readlane 63) in the routing, attention softmax, q/k norm and rstd | TPOT 3.465 -> 3.404 ms (M = 1), 6.53 -> 6.44 ms (M = 8) | yes |
| Routing expert set from per-row 256-bit LDS masks instead of a 64-iteration pick loop | routing at M = 8 7.3 -> 4.6 us (`PROBE=route`), S5 37.0 -> 34.6 us | yes |
| Two-level S6 counters (4 sub-counters per group) | S6 at M = 8 24.8 -> 20.3 us (atomic contention: 40 arrivals per counter); M = 1 8.4 -> 8.7 us | yes |
| K chunks: out 1 or 4, in_proj 2, qkv 2, S6 2, router 2 | within 1 % at M = 1 and M = 8, or slower (out 4: +4 %, S6 2: +6 % / +11 %) (`bench/results/m4/chunk_sweep.txt`) | no |

Where the time goes now (`bench/results/m4/profile_v4_m*.txt` plus the later changes):
at M = 1 each of the 241 stages costs a fixed 7 to 10 us (prologue, partial hand-off
with a device atomic, grid barrier) against about 0.8 ms of weight stream in total. At
M = 8 the MoE stages take 54 us per layer: the expert gate_up stream is bandwidth-bound
(about 20 us), the rest is routing (4.6 us), the norm prologue and hand-offs. The kernel
spills about 50 VGPRs in the attention task (`scripts/isa_stats`); not removed.

## Invariants

- I1: every CTA is resident (grid = CU count; LDS <= 64 KB and VGPRs <= 256 per lane,
  one CTA per CU). The builder asserts the LDS size; the compile log reports VGPRs and
  spills.
- I2: a stage reads only data published before the preceding barrier, or by a last
  arriver whose counter it observed.
- I3: the kernel is as close to the exact reference as the emulating reference is, on
  all 40 layers (P3).
- I4: identical inputs give identical bits; a row does not depend on the other rows
  or on M (P1, P2).
- I5: every CTA computes the same routing tables in a step.
- I6: a CTA's S6 range touches at most 2 positions (asserted at build time from the
  maximum T).
- I7: padding rows select no experts.
- I8: every compile-time setting is a `Cfg` field (P4). No module-level knob (such as
  Qwen3.8's `CHUNKS` dict) affects code generation.

## Correctness oracle

As built (milestones 2 and 3; results in `bench/results/m2`, `bench/results/m3`):

1. `mk/reference.py`: plain PyTorch fp32 decoder of one step for M rows, with a `trace`
   option that returns per-layer h0 / h1, the router logits, and the selected experts
   and weights.
   - Exact mode: fp32 everywhere (activations, caches); FP8 weights dequantized exactly.
   - Emulate mode: the kernel's rounding - fp8 GEMV inputs with pow2 scales per (token,
     128 K), bf16 inputs to bf16 weights, bf16 norm input, fp32 routing, fp8 act input
     to the down projection, bf16 conv state, q and KV, the fixed slot order of the
     expert sum.
2. `tests/ref_vs_hf.py`: exact reference against HF transformers (`Qwen3_5Moe`, fp32,
   BF16 checkpoint), every position of a 52-token prompt: gate cosine >= 0.9999 and the
   same top-1 at every position. Measured: cosine min 1.000000, top-1 100 %, KL 1.7e-9.
   Information: the FP8 checkpoint (exact) against HF BF16: cosine min 0.992, top-1
   100 %.
3. `tests/test_decode.py`: the kernel against the reference in both modes, at M = 1, 2,
   4, 8 over 32 greedy steps (or a truncated model with LAYERS < 40). Gates: the
   kernel is as close to the exact reference as the emulating reference is (worst-row
   cosine); every top-1 change against the exact model is a near-tie (exact logit gap
   <= 0.5); in every step the routing tables of all 304 CTAs hash equal (I5,
   `Cfg.debug_route`); and CTA 0's tables (experts in rank order, weights, shared gate)
   equal `reference.route` of the kernel's own router logits. The expert sets against
   the emulating reference are printed only: fp8 drift alone moves router logits
   across the top-8 boundary.
4. `tests/test_determinism.py`: three runs give bitwise-equal logits; a row decoded
   alone (M = 1) and inside batches of M = 2, 4, 8 with other rows that select other
   experts gives bitwise-equal logits (I4); the same with rows 4..7 marked padding (I7).
5. `tests/eval_text.py`: teacher-forced perplexity on five real passages for the
   kernel, the FP8 reference (fp32 activations) and the BF16 reference; top-1 agreement,
   KL against BF16, and batch invariance.

Changed from the milestone 1 plan: the unit-level grouped-GEMV test
(`tests/test_gemv_moe.py`) was not built. The grouped GEMV is checked end to end by
items 3 and 4 (U from 8 at M = 1 to about 57 at M = 8, padding rows), not for chosen
routings such as expert ids 0 and 255 or U = 64.

Reproducibility gate (milestone 5, `scripts/finish_m5.sh`). A suite run from another node
passes when these conditions are true:

1. Every gate prints PASS, and no step has NaN logits.
2. The node is not the day-1 node.
3. The median TPOT at batch 1, 2, 4, 8 meets a limit against day 1
   (`bench/results/m4/bench_v5_m*.txt`, MI300X):
   - On the same GPU model, the TPOT is within 3 %.
   - On another gfx942 model with 304 CUs (MI325X), the TPOT is at most 3 % slower. A
     faster run passes, because MI325X has more memory bandwidth (approved 2026-10-28).

## Stop conditions

The implementation stops with `STOP: <violated invariant>` and evidence when:

- the CTA does not fit one per CU (LDS > 64 KB, or VGPR spills that the stage structure
  cannot remove), so the grid cannot stay resident (I1);
- a CTA's S6 range can touch more than 2 positions (I6);
- the routing hashes of two CTAs differ (I5);
- a row's logits change with the other rows of the batch (I4).

## Open questions

- Stage latency at M = 1: is the estimate of 7 to 9 us per stage correct for the
  smaller GEMVs of this model? Measure with `bench/profile_decode.py` (per-stage
  timestamps) in milestone 4.
- Fewer stages: (a) router logits from the out_proj / o_proj last arrivers (the logits
  are linear in h once rstd is known), which removes S4 (40 barriers); (b) the GDN core
  in the in_proj tail (Qwen3.8 open question); (c) readiness counters in place of some
  barriers. Each needs its own correctness argument for P2.
- The C values of S5 and S6: measured in milestone 4, the defaults stay (see
  "Optimization log").
- Closed 2026-10-28: one MI300X GPU (ctr-cx63-mi300x-12, job 68192231) gave
  run-to-run different results and NaN logits for this kernel and for the Qwen3.8
  kernel. On that GPU, the PyTorch reference alone also gave different results run to
  run, and the same Qwen3.8 code gave a different step-0 reference result than on banff.
  Random delays at every barrier and counter atomic (`Cfg.chaos`) leave the kernel's
  bits unchanged on a healthy GPU. The fault is in that GPU or node, not in the kernels.
  Slurm drained the node ("Kill task failed"). Evidence:
  `bench/results/m5/ctr-cx63-mi300x-12_68192231/README.md`.
- The distribution of U on real text at M = 8. Log U per layer in `tests/eval_text.py`.
- Reading weights in place (row-major [N][K], no copy) with a permuted K order in the
  A operand. Saves about 34 GB in vLLM. Needs a stream bandwidth microbenchmark against
  the unit-order tiles before any decision.
- vLLM's FP8 MoE weights on ROCm: the layout after loading (e4m3fnuz conversion, any
  AITER shuffle) decides how `runtime/weights.py` repacks them. Check in milestone 6
  before the repack code is written.

## Disposable

None yet. Spikes in later milestones go under `scratch/` and are deleted before the
milestone ends.
