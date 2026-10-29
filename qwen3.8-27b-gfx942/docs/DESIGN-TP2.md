# Qwen3.8-27B decode megakernel, tensor parallel 2 (TP2) on gfx942: design

Status: milestone 2 (design, no kernel code), 2026-10-27. Milestone 1 (spike) result: GO.
Source of every number: the log path next to it. `scratch/` is the disposable spike.

## 1. Fixed point

One decode step of Qwen3.8-27B for M <= 8 rows runs as one persistent FlyDSL launch on each
of two MI300X GPUs (304 CUs each, XGMI between them). Each launch holds the vLLM TP2 shard of
the weights, caches and states of its rank. The launches do their own all-reduces through
peer memory. Both ranks hold the full residual stream and produce bit-identical residuals,
normed hidden states and (standalone) logits halves.

TP1 does not change: `Cfg.tp = 1` gives today's kernel and passes today's tests.

## 2. Required properties (invariants)

| Id | Property | Check |
|---|---|---|
| T1 | Every CTA of both launches is resident (grid = CU count, one CTA per CU). | TP1 rule I1, both ranks |
| T2 | Both ranks compute every reduced value with the same fp32 adds in the same order (rank 0 partial + rank 1 partial). The residual, its sums of squares, its bf16 copy and the final normed hidden state are bit-identical on the two ranks. | rank comparison every step (milestone 3) |
| T3 | No rank overwrites a peer's mailbox data that the peer has not read. | proof in section 6, spike exactness |
| T4 | A row's bits do not depend on the other rows (batch invariance). | `eval_text` batch-invariance pass |
| T5 | Identical inputs give identical bits (3 runs). | `test_determinism` |
| T6 | Every rank takes the kernel path or vLLM's path for the same step. | section 8 |
| T7 | TP2 is as close to the exact reference as TP1 is (bound in section 9). | `test_decode` |
| T8 | TP2 is at least 15 % faster than TP1 at M = 1 on the same node. | milestone 4 |

## 3. Milestone 1 result (spike)

Code `scratch/ar_spike/`, summary `scratch/logs/SUMMARY.md`, node banff-cyxtera-s70-4, GPUs
on one XGMI link (`scratch/logs/20261027a/env.log`).

Spike kernel: 304 CTAs x 512 threads per GPU, 128 all-reduces per launch. One round: each
16-column group's owner CTA pushes its partial as (value, tag) pairs to both ranks' regions
(DSv4.1 `Mailbox`, system scope), polls its own region, sums in rank order, stores, then the
TP1 grid barrier. Added cost = round time minus the barrier-only round. All 128 rounds of the
checked launches were bit-exact and rank-equal.

| M | fp32 pairs, added median / p99 us | bf16-pair words, median / p99 us |
|---:|---|---|
| 1 | 1.15 / 1.64 (run a), 0.93 / 1.82 (run b) | 0.81 / 1.11 |
| 2 | 1.64 / 2.08 | 0.96 / 1.68 |
| 4 | 3.55 / 3.96 | 1.84 / 2.31 |
| 8 | 7.07 / 7.72 (a), 7.02 / 7.83 (b) | 3.57 / 4.44 |

Logs: `scratch/logs/20261027a/predict_r0.txt`, `scratch/logs/20261027b/predict_r0.txt`.
The cost grows with the bytes: about 46 GB/s in both formats, the rate of one XGMI link.
fp32 pairs move 8 bytes a value, bf16-pair words 4 bytes a value.

For comparison (one all-reduce in a CUDA graph of 128, median us, fp32 / bf16): RCCL 9.75 /
10.88 (M = 1), 16.14 / 15.52 (M = 8); vLLM custom all-reduce 4.14 / 4.00 and 7.47 / 5.59;
vLLM with AITER 5.64 / 3.34 and 9.10 / 5.86 (`scratch/logs/20261027a/coll_*_r0.json`,
`scratch/logs/20261027b/coll_vllm-noaiter_r0.json`).

Prediction (milestone 1 formula, W the weight-stream time, F the fixed cost):

| M | TP1 | 40 % split | bytes / LM-head-rate split |
|---:|---:|---|---|
| 1 | 7.56 ms | 5.41-5.50 ms (+27 to +28 %) | 5.67-5.78 ms (+24 to +25 %) |
| 8, fp32 | 10.26 ms | 8.08-8.18 ms (+20 to +21 %) | 8.88-8.98 ms (+12.5 to +13.5 %) |
| 8, bf16 | 10.26 ms | 7.64-7.75 ms (+25 %) | 8.43-8.55 ms (+17 to +18 %) |

## 4. Sharding (vLLM's TP2 layout)

Facts from the vLLM worktree `worktrees/vllm/qwen38-gfx942-specialized-decode` (paths relative
to `vllm/`). The kernel's per-rank packing must give these shards in this order, because in
vLLM the kernel repacks the weights vLLM loaded on that rank.

| Part | vLLM layer | Rank r holds (in this order) | Source |
|---|---|---|---|
| GDN in_proj_qkvz | MergedColumnParallel [2048, 2048, 6144, 6144] | [q 1024 \| k 1024 \| v 3072 \| z 3072]: key heads 8r..8r+7, value heads 24r..24r+23 | `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:803-826`, `model_executor/layers/linear.py:837-863` |
| GDN in_proj_ba | MergedColumnParallel [48, 48] (separate layer) | [b 24 \| a 24] | `qwen_gdn_linear_attn.py:829-851, 871-872` |
| GDN conv1d | ColumnParallel 10240 x 4, per-segment shard | [q 1024 \| k 1024 \| v 3072] channels = 5120 | `qwen_gdn_linear_attn.py:597-640`, `layers/mamba/mamba_mixer2.py:202-246` |
| GDN A_log, dt_bias | sharded dim 0 | 24 value heads | `qwen_gdn_linear_attn.py:646-657` |
| GDN gated RMSNorm weight | not sharded | full [128], shared by all heads | `qwen_gdn_linear_attn.py:666-673` |
| GDN out_proj | RowParallel 6144 -> 5120 | K slice [3072 r, 3072 (r + 1)) | `qwen_gdn_linear_attn.py:675-682` |
| GDN conv / SSM state | per rank | conv (3, 5120) bf16 SD; SSM (24, 128, 128) fp32 | `model_executor/layers/mamba/mamba_utils.py:272-295` |
| Attention qkv | QKVParallel, 48 "heads" (24 q + 24 gate), 4 KV heads | [12 x (q 256 \| gate 256) \| k 2 x 256 \| v 2 x 256] = 7168; q heads 12r..12r+11, KV heads 2r, 2r+1 | `model_executor/models/qwen3_next.py:316-325, 436-444`, `linear.py:1064-1077, 1313-1317` |
| Attention q_norm, k_norm | not sharded | full [256] | `qwen3_next.py:370-371` |
| Attention o_proj | RowParallel 6144 -> 5120 | K slice of 3072 | `qwen3_next.py:327-334` |
| MLP gate_up | MergedColumnParallel [17408, 17408] | [gate 8704 \| up 8704] (kernel interleaves per 8 columns as in TP1) | `model_executor/models/qwen2_moe.py:91-97` |
| MLP down | RowParallel 17408 -> 5120 | K slice of 8704 | `qwen2_moe.py:99-107` |
| Embedding | VocabParallelEmbedding | rows [124160 r, 124160 (r + 1)); other ids masked to 0, then all-reduce | `model_executor/layers/vocab_parallel_embedding.py:304-306, 522-564` |
| LM head | ParallelLMHead, vocab split | rows [124160 r, ...); LogitsProcessor all-gathers on dim -1 | `vocab_parallel_embedding.py:578`, `model_executor/layers/logits_processor.py:122-134, 193-198` |
| Norms (input, post-attention, final) | replicated | full | `qwen3_next.py:568-571, 606, 739` |

MXFP4: a RowParallel rank holds a contiguous K/2 slice of the packed weight and of the e8m0
scales [N, K/32] (`model_executor/layers/parameter.py:216-220`,
`quantization/quark/schemes/quark_ocp_mx.py:224-234`). No 32-element scale group crosses
ranks. The kernel's 128-K unit divides each K slice (3072 = 24 x 128, 8704 = 68 x 128).

Per-rank GEMV shapes (N x K) and their units of 16 rows x 128 K:

| GEMV | TP1 | TP2 per rank | Bytes per rank | Units per wave (2432 waves) |
|---|---|---|---|---|
| in_proj (qkvz + ba) | 16480 x 5120 | 8240 x 5120 | 22.4 MB | 8.5 |
| out_proj / o_proj | 5120 x 6144 | 5120 x 3072 | 8.4 MB | 3.2 |
| qkv | 14336 x 5120 | 7168 x 5120 | 19.5 MB | 7.4 |
| gate_up | 34816 x 5120 | 17408 x 5120 | 47.3 MB | 17.9 |
| down | 5120 x 17408 | 5120 x 8704 | 23.6 MB | 9.0 |
| LM head (bf16) | 248320 x 5120 | 124160 x 5120 | 1.27 GB | - |

Group counts stay whole: 8240 / 16 = 515 (b and a start at column 8192), 7168 / 16 = 448,
8704 / 8 = 1088 gate/up pairs, 124160 / 16 = 7760.

Mixer tasks per rank: GDN tasks (row, value head, 16 value rows) over 24 heads instead of 48;
attention tasks (row, KV head, 64 positions) over 2 KV heads (GQA 6 is unchanged). The KV
cache of a rank holds its 2 KV heads (vLLM allocates it per rank).

Kernel changes this needs (milestone 3):
- `dims.py`: a `Dims(tp)` value with the per-rank widths; `tp = 1` gives today's constants.
  The kernel, stages and mixer read the widths from it instead of module constants.
- `Cfg`: new fields `tp: int = 1` and `ar_dtype: str = "fp32"` (`"fp32"` or `"bf16"`), and a
  debug-only field `ar_local: bool = False` (section 11). The kernel name hashes `repr(cfg)`,
  so every field is in the name (TP1 binaries get a new name and the same code).
- The rank is not in `Cfg`: both ranks run the same binary. The rank, the peer table address
  and the vocab offset go into the global pointer table.
- `pack.py`: `Packed(W, tp, rank)` slices the checkpoint into the shards above, then packs
  them as TP1 does. `reference.py` keeps the full single-device model.
- `gemv.CHUNKS` for out/o (K 3072) and down (K 8704) is tuned again (milestone 4).

## 5. Where the all-reduce sits

TP1 residual epilogue (`stages.EpiRes`, run by the last arriver of each 16-column group):
`h_out = h_in + tot` with `tot` the sum of the split-K segments in a fixed order, then the
16-column sum of squares, the bf16 copy, then the grid barrier. TP2 changes only `tot`:

1. The last arriver of group g sums its split-K segments in TP1's order: `p_rank`.
2. It puts `p_rank` (fp32 pairs, or bf16-pair words for `ar_dtype = "bf16"`) to region
   [src = rank][row][col] of both ranks' peer buffers (its own rank included).
3. It polls its own region for both sources and sums `p_0 + p_1` in fp32, rank order.
   For `bf16`, the own partial is the rounded word from the region too, so both ranks add the
   same values.
4. `h_out = h_in + (p_0 + p_1)`, sum of squares, bf16 copy, as in TP1. Then the TP1 barrier.

The all-reduce adds no barrier: it runs inside the epilogue, before the barrier that TP1
already has. Thus the spike's "added cost" is the cost per all-reduce.

Can it merge with the next stage's prologue? Every stage after an all-reduce (gate_up after
out/o_proj, in_proj or qkv after down) starts with a norm prologue that needs the full
residual. A merged design pushes in the epilogue, does not poll, and lets every CTA poll all
5120 columns in the prologue. That hides the wait behind the barrier, but:
- 304 CTAs each read M x 5120 x 2 tagged pairs of uncached memory (24 MB at M = 1) where TP1
  reads a 10 KB bf16 copy;
- the prologue needs rstd, which comes from the per-group sums of squares the epilogue writes;
  every CTA would compute the sum of squares over 5120 columns itself;
- someone must still write `h_out` for the next residual epilogue.
Rejected for milestone 3. It is a milestone 4 candidate only if the measured wait (section 11)
is large compared with the barrier (3 us).

Embedding: a token's row lives on one rank. The embed stage writes the row on the owning rank
and zero on the other, and exchanges it with the same mailbox (one more all-reduce, exact:
x + 0 = x). It uses its own region (section 6). LM head: each rank computes its vocab half.
In vLLM the kernel runs with `lm_head = False` (normed hidden states out, as today); vLLM's
ParallelLMHead, all-gather and sampler follow.

## 6. Peer-memory protocol

Copied from DSv4.1 (`models/deepseek_v41/amd/mono/common/sync.py`, `peer_memory.py`):
- `PeerBuffer`: one uncached buffer per rank (`UncachedIpcHeap.alloc_uncached`), IPC handles
  exchanged over the gloo group, every rank holds every peer's address, `tp_agree` on the
  mapping (`peer_memory.py:40-76`).
- `Mailbox`: each 32-bit value is stored next to a 32-bit tag (8-byte pair, `sc0 sc1` system
  scope), the consumer polls until every tag matches. No flags, no store drain, one round trip.
- Every rank pushes to every rank's region, its own included, and sums the sources in rank
  order 0 .. tp - 1 (`ops.sum_partials`).
- Two parity halves, so a producer never writes the half a peer may still read.

Adaptation (no new mechanism): DSv4.1 runs one hand-off set per launch, tag `(epoch << 1) + 2`
and parity `epoch & 1`. This kernel runs 128 + 1 hand-offs per launch. Thus:
- tag = the grid-barrier epoch that follows the hand-off (unique, grows by 321 a step);
- layer all-reduce k (k = 0 .. 127 in a step) uses half `k & 1`. 128 is even, so consecutive
  all-reduces use different halves across steps too;
- the embedding exchange has its own region (no parity).

Layout per half: [src rank][M_MAX = 8][5120] pairs: 640 KB fp32, 320 KB bf16. Total with both
halves and the embedding region: below 2 MB.

No overwrite of unread data (T3). Rank A writes half p for all-reduce k + 2 only after it
finished all-reduce k + 1. That needs B's push of k + 1. B pushes k + 1 in a later stage than
its read of k, after the grid barrier that ends the stage of k, so B finished every read of
half p. The same argument covers the embedding region between steps: A's embed exchange of
step s + 1 needs A to finish step s, which needs B's last push of step s, issued after B read
its step s embed region. A stale pair cannot match: a tag repeats only after 2^32 barriers.
Spike evidence: 128-round launches, back to back, bit-exact on both ranks
(`scratch/logs/20261027a/spike_r*.log`).

The epochs of the two ranks must agree: both flag buffers start at zero, and both ranks run
the same sequence of launches with the same layer count. A rank whose launch fails leaves its
peer spinning (DSv4.1 behaves the same). The standalone harness runs each launch under a
timeout; vLLM fails the engine.

Source of truth:
- vLLM: the TP2 runtime imports `PeerBuffer` from
  `vllm/models/deepseek_v41/amd/mono/common/peer_memory.py` (aiter `UncachedIpcHeap`), as
  MiniMax-M3 does (`models/minimax_m3/amd/specialized_decode/runtime/executor.py:57`). The
  exported kernel imports `Mailbox` and `preg` from
  `vllm/models/deepseek_v41/amd/mono/common/sync.py`. These two files are the source of truth.
- Standalone repository: the kernel image has no aiter. `mk/peer.py` holds a copy of
  `PeerBuffer`, `tp_agree` and aiter v0.1.26's `UncachedIpcHeap` (the methods `PeerBuffer`
  uses), and `mk/sync.py` a copy of `Mailbox` (put, poll) with the same `Addr` API. Both name
  the vLLM commit they copy. `scripts/export_vllm.py` does not export them: it rewrites
  `from .sync import ...` to the DSv4.1 import, and it fails when the copied `Mailbox` method
  bodies differ from the vLLM file.

## 7. Determinism and batch invariance

- Within a rank, split-K stays the fixed-order last arriver of TP1: no float atomics.
- Across ranks, both compute `p_0 + p_1` from the same bits in the same order (T2). The
  replicated state after the all-reduce (residual, sums of squares, bf16 copy, norm prologue
  input) is identical on both ranks, so every column-parallel GEMV input is identical too.
- The quantized GEMV inputs are the same as TP1's: column GEMVs quantize the full replicated
  x; row GEMVs quantize per (token, 128 K) and each K slice is a whole number of 128-K blocks.
  For `fp32`, TP2 differs from TP1 only in the grouping of the fp32 sums of the row GEMVs.
- A row's mailbox slots and sums use only that row's values (T4). The region offset depends
  on M, the values do not.
- TP2 is not bitwise-equal to TP1 and need not be. The oracle is the reference (section 9).

## 8. Path agreement (T6)

- Build: `SpecializedDecode.__init__` agrees on model refusal with `tp_agree`
  (`models/common/specialized_decode/dispatch.py:78-84`). The Qwen spec refuses TP != 1 today
  (`models/qwen3_5/amd/specialized_decode/spec.py:142-143`); it changes to accept TP 1 and 2.
- Peer memory: `PeerBuffer` construction is collective and agrees on the mapping. As in
  DSv4.1 (`mono/runner.py:241`), it is constructed at model load, after the build
  `tp_agree`; only the per-width scratch is allocated at the first eager step
  (`mono/runner.py:250-266`). Approved 2026-10-27.
- Epoch: the mailbox tags come from the barrier epoch, so every kernel that shares a peer
  buffer must share one epoch word. The standalone engine has one width per engine; the vLLM
  runtime (milestone 6) keeps one flag buffer for all widths.
- Caches: `bind_caches` raises on refusal (`dispatch.py:94-107`); it does not fall back, so a
  one-rank refusal stops the engine instead of splitting the paths. The cache checks become
  per-rank shapes (SSM (24, 128, 128), 2 KV heads, conv 5120 channels).
- Each step: `plan` decides from the width, the forward context and the attention metadata
  (`dispatch.py:116-143`, `spec.py:224-240`). All of it comes from the scheduler output,
  which every TP rank receives identically. No per-step collective (it would cost more than
  an all-reduce). Milestone 6 adds a debug check (an all-reduce of the decision) to the
  validation runs.
- Sampling: every rank samples the same token from the all-gathered logits
  (`v1/worker/gpu/model_runner.py:1673-1675`, seed/position-deterministic Gumbel noise,
  `v1/worker/gpu/sample/gumbel.py:105-116`). The kernel's hidden-state outputs are
  bit-identical on both ranks (T2), so the logits are too.

## 9. Oracle

- Reference: `mk/reference.py`, the single-device fp32 step (exact: bf16 GEMV inputs;
  emulate: the kernel's fp8 inputs). For `ar_dtype = "bf16"`, emulate mode gets an option that
  splits each row GEMV into the two K halves, rounds each half to bf16, and adds them.
- `test_decode` TP2 (2 processes): rank 0 also runs the reference; logits are the two vocab
  halves gathered over gloo. Pass:
  1. the TP1 criterion (`tests/test_decode.py:58-60`) with the TP2 logits; and
  2. the TP1 bound: on the same node, seed and steps, 1 - worst cos(TP2, exact) <=
     F x (1 - worst cos(TP1, exact)) + 1e-5, and mean top-1 agreement with exact >= TP1's
     - 0.05. F = 1.5, or the largest TP1-to-TP1 ratio of 1 - worst cos over 3 seeds at
     M = 1 and 8 if that is larger (approved 2026-10-27).
- Local check before the 64-layer runs: 1 layer, TP2 `fp32` against TP1, max relative
  difference of the residual <= 1e-5 (only the fp32 sum grouping differs).
- Ranks: the residual buffer and the normed hidden state compared bitwise across ranks after
  every step of `test_decode` and `test_determinism`.
- `test_determinism`: 3 runs bitwise-equal per rank, M = 1 and 8.
- `eval_text`: perplexity, top-1, KL(bf16 || kernel) per passage, and the batch-invariance pass.

Gates from milestone 4 on (approved 2026-10-27, after milestone 3):
- Hard gates: median per-step 1 - cos(kernel, exact) <= 1.2 x TP1's (same node, same
  seeds); eval_text KL(bf16 || kernel) within 10 % of TP1's; ranks bitwise equal at every
  step; determinism (3 runs); batch invariance.
- Report only: the test_decode criterion (c1) and the TP1 bound (c2). With F = 55 the bound
  does not bind at M = 8.
- fp32 is at +9.6 % KL against the 10 % limit: re-checked in milestone 5 on a second node.
- The 1-layer check stays: TP2 against an emulating reference with TP2's sum order
  (`reference.step(..., ar_tp=2)`: each row-parallel GEMV as the ranks' K-slice fp32
  partials, added in rank order). Expected agreement about 1e-7, as TP1 against its own
  order. test_decode emulates TP2's order by default (MK_EMU_SPLIT=1).
- 1-layer check result (2026-10-28, job 68193288, `scratch/logs/l1a/`, M = 8, seeds 0..2):
  relative error of the residual per row, TP2 fp32 against the reference in TP2's order:
  median 7.6e-8, 4.4e-8, 6.8e-8, max 2.3e-7, 1.2e-7 and 3.6e-4; TP1 against its own order:
  median 7.3e-8, 6.5e-8, 9.6e-8, max 1.8e-7, 1.6e-7 and 7.3e-4. The seed-2 maximum is one row
  in both kernels: one fp8 input rounding flip (TP1 has it too). TP2 agrees as TP1 does; the
  milestone 3 value 3.2e-5 was such a flip.

Partial format choice (milestone 3, both formats run): default `bf16` if it passes the TP1
bound above and its `eval_text` KL against BF16 is within 10 % of TP1's; otherwise `fp32`.

### Milestone 3 results (2026-10-27, banff-cyxtera-s70-4, job 68191827)

Code: commit `19e5983` and the embedding fix of the next commit (`fx.Float32(v)` in the embedding put; without it
`Mailbox.put` converted the value to an integer and every activation was 0). Container
hipdnn_20261024_gfx942. Logs: `scratch/logs/m3a/` (seeds 0..2, determinism, eval_text) and
`scratch/logs/m3b/` (seeds 3..7); summaries `scratch/logs/bound_m3.txt`,
`scratch/logs/m3a/steps_summary.txt`.

The TP1-to-TP1 spread (TP1, seeds 0, 1, 2): 1 - worst cos(kernel, exact) is 0.000557,
0.000532, 0.000546 at M = 1 (ratio 1.05) and 0.012213, 0.002636, 0.145837 at M = 8 (ratio
55.3). The worst cosine is one row at one step, and fp8 rounding is chaotic over 64 layers,
so at M = 8 it spans 55x. Per the approved rule, F = 55.3.

| Format | test_decode criterion (c1), 18 runs | TP1 bound (c2), 18 runs | median per-step 1 - cos(kernel, exact), 128 steps | eval_text KL(bf16 \|\| kernel) | ppl kernel (w4, bf16) | ranks bitwise equal | determinism (3 runs, M = 1, 8) | batch invariance | Decision |
|---|---|---|---|---|---|---|---|---|---|
| TP1 | 17/18 | 18/18 (reference) | 0.000377 | 0.03904 | 1.661 (1.645, 1.580) | - | PASS, PASS | True | reference |
| TP2 fp32 | 16/18 | 18/18 | 0.000406 | 0.04279 (+9.6 %) | 1.657 | every step of every run | PASS, PASS | True | **chosen** |
| TP2 bf16 | 13/18 | 16/18 | 0.000501 | 0.03708 (-5.0 %) | 1.650 | every step of every run | PASS, PASS | True | rejected: fails the bound |

Runs: M = 1 and 2 at seeds 0 and 3..7, M = 1 and 8 at seeds 1 and 2, M = 4 and 8 at seed 0
(test_decode, 64 layers, 16 steps). The median column counts the 128 steps of M = 1, 2, 4,
8 at seeds 0..2 (`steps_summary.txt`).

- The c1 failures are single chaotic rows. TP1 fails c1 once too (M = 1, seed 7). TP2 fp32
  fails it at M = 1 seed 2 (kernel~emulate 0.942 < 0.95; the emulating reference itself
  drops to 0.951 against exact at that step, the kernel stays at 0.997) and at M = 2 seed 0
  (1 - cos 0.000895 against 2 x 0.000402 + 1e-5 = 0.000814; TP1 0.000579).
- bf16 fails the bound at M = 1 seed 2 (ratio 89 to TP1) and M = 1 seed 7 (top-1 0.812
  against TP1 0.875); its median per-step error is 33 % above TP1's.
- Decision: `ar_dtype = "fp32"` is the default. bf16 passes the KL condition but not the
  bound.
- Local 1-layer check (`scratch/m3/dbg.py`, `scratch/logs/dbg2/`, TP1:
  `scratch/logs/dbg1/dbg_tp1_l1.r0.log`): relative error of the residual against the emulating
  reference is 3.2e-5 for TP2 fp32 and 6.5e-8 for TP1, above the planned 1e-5. The planned
  bound assumed no fp8 rounding flips; one flipped e4m3 input element in gate_up or down
  gives this size. The 64-layer statistics above show no systematic error.

## 10. Fixed cost per stage at TP2

The go/no-go formula keeps TP1's fixed cost. Evidence and limits:

Fit of the TP1 GEMV stage span against bytes (6 GEMV kinds, median over layers;
`scratch/logs/stage_fit.txt`, from `bench/results/timeline/timeline_m{1,8}.json`):
- M = 1: span = 6.52 us + bytes / 3.66 TB/s, max residual 1.27 us.
- M = 8: span = 11.93 us + bytes / 3.41 TB/s, max residual 3.74 us.

The intercept is the per-stage fixed cost. A stage model that keeps every intercept, every
barrier and every mixer stage, halves the stream part of each GEMV and the LM head, and adds
128 all-reduces gives (stamped steps, so totals are the profiled ones):

| M | TP1 profiled | TP2 without all-reduce | + fp32 | + bf16 |
|---:|---:|---:|---|---|
| 1 | 7.927 ms | 5.828 ms | 5.947-5.975 ms (+24.6 to +25.0 %) | 5.932 ms (+25.2 %) |
| 8 | 10.258 ms | 7.986 ms | 8.885-8.891 ms (+13.3 %) | 8.443 ms (+17.7 %) |

Why the fixed parts stay:
- Grid barrier: same 304 CTAs, same protocol; independent of GEMV size.
- Norm prologue: every CTA reads and quantizes the full replicated residual (K = 5120), as in
  TP1. The prologues of the row GEMVs read half the K (act 8704, mixer output 3072): cheaper.
- Last-arriver hand-off: each group keeps its split-K segments; with half the units a wave
  touches the same number of pieces or fewer. Counter atomics per stage halve for column GEMVs.
- GDN and attention tasks: half the heads per rank, the same per-task latency. Kept at the
  TP1 value (conservative; at M = 8 the GDN state traffic halves, up to about 0.5 ms less).

What can grow, and its cost:
- Stream ramp on small GEMVs. TP2 out/o_proj is 8.4 MB, 3.2 units per wave, below the smallest
  TP1 GEMV of the fit (16.7 MB, 6.3 units per wave) and near the ping-pong depth (P = 2). The
  fit is an extrapolation there.
- Cross-GPU skew: an all-reduce waits for the slower GPU's last arriver of the group. TP1 has
  no such term. The spike ran identical work on idle GPUs, so it does not contain it.
- Tolerance: at M = 1, the 15 % gate allows 6.738 ms (stamped scale) against 5.975 ms
  predicted: 0.76 ms of slack, that is 3.0 us more fixed cost per GEMV stage (256 stages) or
  5.9 us more per all-reduce, or a mix. At M = 8 (`bf16`) the slack to 15 % is 0.28 ms.

## 11. Milestone 4 measurements (planned)

1. Per-stage profile of TP2 (`bench/profile_decode.py` on both ranks): span and barrier per
   stage kind at M = 1, 2, 4, 8. Compare each GEMV span with the fit of section 10
   (`a + bytes / 2 / B`). Report the measured fixed cost per stage.
2. All-reduce cost under load. Build the same TP2 kernel with `ar_local = True` (debug only,
   wrong numbers, right timing: each rank adds its own partial to itself, no peer traffic).
   Under-load cost per all-reduce = (span of out/o_proj and down stages with the all-reduce)
   minus (the same spans with `ar_local`), per M and per format. Also per-wave probe stamps in
   the residual epilogue (push issued, poll satisfied) give the wait distribution and the
   cross-GPU skew. Compare with the spike (1.15 / 7.07 us fp32, 0.81 / 3.57 us bf16).
3. TP2 against TP1 on the same node (`bench/bench_decode.py`, 3 runs, M = 1, 2, 4, 8), and
   stock vLLM TP2 (nightly image, MXFP4 and FP8 checkpoints) at batch 1, 2, 4, 8.
4. Tuning candidates, each logged here with its effect: `CHUNKS` for out/o/down at TP2,
   16-byte pushes (two columns a lane through a DPP exchange), the prologue merge of section 5.

### Milestone 4 results (2026-10-28, banff-cyxtera-s70-4, job 68193288): STOP

`STOP: all-reduce under load slower than the section 13 limit` (the measured cost at M = 1 is
more than twice the spike's, in both formats). The 15 % gate (T8) holds: TP2 fp32 is 25.9 %
faster than TP1 at M = 1. No optimization was tried after the STOP. Resolved 2026-10-28: the
measured end-to-end result replaces the prediction that the limit protected (section 13).

Speed, same node, TP1 on one GPU alone (`scratch/logs/m4_bench_summary.txt`; raw
`scratch/logs/m4a/`, `m4b/`). `bench/bench_decode.py`, ctx 512, 64 steps, median of 3 runs.
TPOT includes the greedy token: TP2 takes it from the ranks' vocab slices with one RCCL
all-gather of (max, index) per step (`mk/tp.py next_token`). "Fixed" holds the tokens
fixed: launches only.

| M | TP1 TPOT (fixed) ms | TP2 fp32 TPOT (fixed) ms | vs TP1 | TP2 bf16 TPOT (fixed) ms | vs TP1 |
|---:|---|---|---:|---|---:|
| 1 | 7.895 (7.891) | 5.852 (5.777) | -25.9 % | 6.137 (6.064) | -22.3 % |
| 2 | 8.358 (8.358) | 6.434 (6.366) | -23.0 % | 6.577 (6.491) | -21.3 % |
| 4 | 9.163 (9.166) | 7.562 (7.473) | -17.5 % | 7.369 (7.287) | -19.6 % |
| 8 | 10.526 (10.512) | 8.635 (8.552) | -18.0 % | 8.235 (8.177) | -21.8 % |

TP1 on this node today: 7.895 ms at M = 1 against 7.56 ms on 2026-10-22 (same node, 4 %
slower; the comparison above uses the same day and node).

All-reduce cost under load: (fixed TPOT with the all-reduce - fixed TPOT of the same kernel
with `ar_local`) / 128.

| Format | M | Under load, us per all-reduce | Spike (idle GPUs), us | Section 13 limit |
|---|---:|---:|---:|---|
| fp32 | 1 | 2.74 | 0.93-1.15 | 2.3: exceeded |
| fp32 | 8 | 9.34 | 7.02-7.07 | - |
| bf16 | 1 | 3.35 | 0.81 | 1.6: exceeded |
| bf16 | 8 | 6.34 | 3.57 | - |

The `ar_local` runs vary by up to 0.35 ms between repeats (bf16 M = 1: 5.458, 5.636,
5.736 ms), so the bf16 M = 1 value is uncertain to about +-1.5 us; fp32 repeats agree within
0.04 ms.

Per-stage profile (one stamped step per variant, rank 0; `scratch/logs/m4c/stages_summary.txt`,
timelines `scratch/logs/m4c/tl_*.json*` and `m4b/tl_tp1_m1.json`, not in git):
- Fixed cost per GEMV stage, M = 1, against the section 10 fit (a + bytes / 2 / B): inp
  14.11 us (fit 12.64), qkv 12.81 (11.85), gu 19.78 (19.46), and without the all-reduce
  (`ar_local`) out 8.98 (8.80), o 9.21 (8.80), down 14.91 (12.99). All within 1.9 us of the
  fit: the fixed-cost stop condition (3 us) does not apply. At M = 8 inp is 4.2 us and qkv
  3.0 us above the fit (not a gate).
- Where the all-reduce cost appears (M = 1, fp32): out +1.6 us, o +1.2 us in the span; for
  down in the barrier after it (6.08 us against 4.20 with `ar_local`; bf16 9.51 us). One
  stamped step per variant is noisy (the other stages differ by up to 1.8 us between the
  variants), so these split the cost only roughly.
- Not measured yet: the per-wave probe stamps of the push and the poll (item 2 above).

Candidate causes (not verified, no fix tried): the grid arrival drains every outstanding
store (`s_waitcnt vmcnt(0)`), so it waits for the system-scope stores to the peer to
complete over XGMI; each all-reduce waits for the slower GPU's last arriver of the group (the
cross-GPU skew of the stage end), which the spike did not have; at M = 1 bf16 is slower than
fp32 although it moves half the bytes, so the cost at M = 1 is latency, not bandwidth.

Stock vLLM TP2, nightly image `vllm-rocm-nightly-00b6c4c5` (vLLM 0.32.1rc1.dev505,
`VLLM_ROCM_USE_AITER=1`, CUDA graphs, `vllm bench serve` 512 in / 256 out), median TPOT ms
(`scratch/logs/m4c/vllm_aiter_tp2_Qwen3.8-27B-Quark-AWQ-MXFP4/`,
`scratch/logs/m4d/vllm_aiter_tp2_Qwen3.8-27B/`):

| Checkpoint | batch 1 (3 runs) | 2 | 4 | 8 |
|---|---|---:|---:|---:|
| MXFP4 (vLLM emulates MXFP4 on gfx942) | 25.15 (25.14-25.17) | 25.32 | 25.52 | 26.25 |
| BF16 (in place of FP8) | 9.81 (9.77-9.82) | 10.61 | 12.21 | 13.32 |
| FP8 | missing on 2026-10-28 (deleted in a disk clean-up); restored and measured below | | | |

TP2 kernel (fp32) against stock vLLM TP2 BF16: 9.81 / 5.852 = 1.68x at batch 1, 13.32 /
8.635 = 1.54x at batch 8. The two methods differ (server TPOT against engine steps, see
`bench/RESULTS.md`).

Bugs found during milestone 4 (fixed, both in the harness or the timing-only build):
- `bench/profile_decode.py` allocated its stamp buffers before `TP.init` selected the rank's
  GPU; rank 1's kernel stored to GPU 0 memory and its first step hung (`scratch/logs/pdiag/`
  and `scratch/logs/m4b/`).
- `ar_local` builds still ran the embedding exchange; without the layer all-reduces the
  ranks drift apart and the one embedding region was overwritten before it was read (hang
  in `m4a`). The timing-only build now skips the exchange.

### All-reduce timing split (2026-10-28, banff-cyxtera-s70-4, job 68195787)

Method: debug probes `dev.ARPROBE` (off by default, compile-time). Per CTA and barrier:
arrive (thread 0), drained (after the CTA's `s_waitcnt vmcnt(0)` and `s_barrier`: every
store of every wave acknowledged, peer stores included), released. Per wave: push, poll
start, poll end of its last all-reduce epilogue in each stage (lane 0). Driver
`scratch/m3/arsplit.py`: 512 warm-up steps, then 5 stamped steps per variant, both ranks,
fp32 and bf16, with the all-reduce and with `ar_local`, M = 1 and 8. Analysis
`scratch/m3/arsplit_an.py` and `scratch/m3/arsplit_parts.py`; output
`scratch/logs/ars/split.txt` and `parts.txt` (raw stamps `*.pt` stay in
`/cluster/logs/qwen38-tp2/ars/`).

- Clocks: each GPU's 100 MHz counter has its own origin. The offset comes from the
  all-reduce itself (NTP style, symmetric latency assumed): 0.275 ms, the same in every
  step. The minimum one-way latency push -> peer poll end is L = 0.80 us at M = 1 (fp32 and
  bf16), 2.4 us at M = 8 fp32, 1.5 us at M = 8 bf16.
- Layer 0 is excluded: it absorbs the launch skew of the two processes (about 1 ms in a
  stamped step, because the host syncs the ranks before that step). In back-to-back steps
  the launch queue hides this skew.
- Probe cost: the probes inflate the added cost per all-reduce (stamped / unstamped):
  fp32 3.77 / 2.74 us at M = 1, 12.6 / 9.34 us at M = 8; bf16 5.96 / 3.35 and 9.08 / 6.34 us.
  The shares below are of the stamped cost; scaling them to the unstamped cost is an
  inference, not a measurement.
- Closure: the sum of the stage segments (out, o, down) matches the whole-step difference:
  3.83 / 3.77, 6.36 / 5.96, 13.66 / 12.62, 9.11 / 9.08 us per all-reduce (fp32 M = 1, bf16
  M = 1, fp32 M = 8, bf16 M = 8).

Segments of an all-reduce stage, rank-0 clock: gemv = stage start -> the rank's last push
(the GEMV stream, with every epilogue's push and poll inside it); wait = the slower rank's
last push - this rank's; xfer = last poll end - the slower rank's last push; drain = last
poll end -> last CTA drained; bar = drained -> released. Mean over the layers of each kind
after layer 0 (out 47, o 16, down 63), 2 ranks, 5 steps; AR / `ar_local`, us:

| M, format | stage | gemv | wait | xfer | drain | bar | added |
|---|---|---|---|---|---|---|---:|
| 1, fp32 | out | 10.83 / 8.23 | 0.34 / 0 | 1.18 / 0 | 0.73 / 0.83 | 2.38 / 2.60 | 3.79 |
| 1, fp32 | o | 10.73 / 8.31 | 0.46 / 0 | 1.16 / 0 | 0.73 / 0.83 | 2.38 / 2.59 | 3.72 |
| 1, fp32 | down | 17.28 / 14.36 | 0.35 / 0 | 1.15 / 0 | 0.70 / 1.01 | 2.48 / 2.71 | 3.89 |
| 1, bf16 | down | 22.19 / 14.28 | 0.62 / 0 | 1.20 / 0 | 0.77 / 1.12 | 2.51 / 2.83 | 9.06 |

Per epilogue (one group, one wave), push -> poll end at M = 1: 1.72 to 1.89 us in both
formats = peer-late wait 0.4 to 0.65 us (the peer's push + L comes after this rank's poll
start, in 51 to 68 % of the groups) + own push / poll 1.2 to 1.4 us. Push issue (push ->
poll start) 0.44 to 0.52 us fp32, 0.72 to 0.76 us bf16. At M = 8 fp32 the push issue is
3.6 to 4.0 us (8 system-scope 8-byte stores per lane, 4 rows x 2 ranks), bf16 1.1 to 1.2 us.
About 250 of the 320 groups' epilogues are the last one of their wave, so some waves run two
or more epilogues in a stage.

The split (`scratch/logs/ars/parts.txt`). push / poll = xfer + the own push / poll share of
the gemv growth; slower GPU = wait + the peer-late share of the gemv growth; drain and
barrier = the growth of those segments. The gemv growth is split by the per-epilogue ratio
of peer-late wait to own push / poll.

| M, format | added per all-reduce (stamped) | push / poll | slower GPU | peer-store drain | barrier |
|---|---:|---:|---:|---:|---:|
| 1, fp32 | 3.83 us | +2.97 (78 %) | +1.28 (33 %) | -0.20 (-5 %) | -0.23 (-6 %) |
| 1, bf16 | 6.36 us | +4.81 (76 %) | +2.13 (34 %) | -0.25 (-4 %) | -0.33 (-5 %) |
| 8, fp32 | 13.66 us | +12.16 (89 %) | +2.01 (15 %) | -0.28 (-2 %) | -0.24 (-2 %) |
| 8, bf16 | 9.11 us | +6.69 (73 %) | +2.95 (32 %) | -0.37 (-4 %) | -0.16 (-2 %) |

Findings:
1. The peer-store drain at the barrier costs nothing measurable: the time from the last
   poll end to the last CTA drained is shorter with the all-reduce than with `ar_local`.
   [Inference] The peer stores are acknowledged during the poll round trip, so the drain
   finds nothing outstanding.
2. The largest part is the push / poll itself, and most of it sits inside the GEMV stream:
   the last arriver of a group runs the epilogue in the middle of its unit stream and waits
   about 1.8 us for the round trip before it streams its next units. The stage's last push
   then comes 2.4 to 2.9 us later than with `ar_local` (M = 1, fp32), plus about 1.2 us
   for the last partial's transport and poll (xfer).
3. The wait for the slower GPU is about one third at M = 1. It is jitter per stage, not a
   slow GPU: with `ar_local` the two GPUs take the same time per step (4.99 and 5.01 ms
   median at M = 1 fp32). The peer is late for 51 to 68 % of the groups, by 0.4 to 0.65 us
   on average.
4. Why bf16 is slower than fp32 at M = 1: per epilogue the cost is the same (1.8 us; bf16
   issues its pushes 0.25 us slower, through the lane exchange, and polls 0.3 us faster).
   The whole difference is in the down stage: its last push comes 4.7 us later than with
   fp32, in every layer (median 21.9 against 17.2 us), and even its first push comes 2.2 us
   later. Out and o are equal in both formats. The cause inside down is not found yet.
   Excluded: the GEMV loop code (the MFMA loop blocks are instruction-for-instruction
   equal in the fp32, bf16 and `ar_local` ISA, `scratch/logs/isa/`), VGPR spills (1 in every
   build), the per-wave stream time of the last down layer (7.3 us median in both,
   `scratch/logs/pdown/`). The AR builds have more SGPR spills than `ar_local` (fp32 608,
   bf16 733, against 373 and 350; spilled to VGPR lanes). With the probes the down
   difference is about 5 us per layer; unstamped, bf16 costs 0.6 us more than fp32 per
   all-reduce at M = 1.

Next optimization (not started; for review): the largest part is the epilogue's round trip
inside the GEMV stream. Candidate: the last arriver pushes in the epilogue but does not
poll there; it streams its remaining units and polls its pending groups at the end of its
stream, before the barrier arrival. This removes the in-stream wait and keeps the barrier
count and the protocol (the same tags, regions and sum order). The prologue merge of
section 5 also removes it, but it adds peer reads to every CTA's next prologue.

Hard gates after the probe change (probes off; `scratch/logs/gates/`, 2026-10-28, job
68195787), TP2 fp32 against TP1 on the same node and seed 0: median per-step 1 - cos to
exact 0.000341 against 0.000342 (ratio 1.00; per M 0.97, 1.06, 1.20, 1.02 at M = 1, 2, 4, 8);
`eval_text` KL 0.04279 against 0.03904 (+9.6 %); ranks bitwise-equal every step;
determinism M = 1 and 8 PASS; batch-invariant rows. All pass. The test_decode values equal
the milestone 3 values to six digits (for example 1 - worst cos 0.000897 at M = 1), so the
probes-off build computes the same bits. Report only: the old test_decode criterion fails at
M = 1 as in milestone 3.

### FP8 baseline and engine-level comparison (2026-10-28, banff-cyxtera-s70-4, job 68195787)

`Qwen/Qwen3.8-27B-FP8` restored to `~/mk/models/Qwen3.8-27B-FP8` (29 GB, `hf download`,
after a test write; `~/mk/models/dl_Qwen3.8-27B-FP8.log`). All runs on one node in one
allocation (`scratch/logs/fp8/`).

Timing methods:
- Kernel: `bench/bench_decode.py`, a 512-token prompt fed through the kernel, then 64 greedy
  steps back to back; TPOT = wall time of the 64 steps / 64 (CUDA events), host argmax and
  token feedback included; median of 3 runs.
- vLLM, engine level: `bench/stock/vllm_latency_bench.sh` runs `vllm bench latency`
  (offline `LLM` engine, CUDA graphs, prefix caching off, nightly image, AITER on) with a
  512-token prompt and 1 or 65 output tokens; TPOT = (median latency with 65 - median with
  1) / 64, 10 iterations each. It covers the same context range (512 to 576) and includes
  vLLM's per-step scheduler, sampler and output processing, which the kernel benchmark does
  not have (the kernel's host loop is an argmax and a copy).
- vLLM, server: `vllm bench serve`, 512 in / 256 out, median TPOT (for reference; it adds
  HTTP and detokenization).

Engine-level TPOT, ms:

| Batch | TP2 kernel (fp32 partials) | TP1 kernel (1 GPU) | vLLM TP2 FP8 | FP8 / TP2 kernel | vLLM TP2 BF16 | vLLM TP2 MXFP4 (emulated) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.896 | 7.865 | 9.814 | **1.66x** | 9.325 | 24.78 |
| 2 | 6.501 | 8.353 | 10.825 | **1.67x** | 10.626 | 25.62 |
| 4 | 7.645 | 9.119 | 10.385 | **1.36x** | 11.847 | 25.47 |
| 8 | 8.686 | 10.468 | 11.905 | **1.37x** | 13.216 | 25.80 |

vLLM TP2 FP8 server TPOT (`vllm_aiter_tp2_Qwen3.8-27B-FP8/`): 10.16 (10.15 to 10.17, 3
runs), 10.52, 10.95, 11.48 ms at batch 1, 2, 4, 8. The FP8 engine value at batch 4 is below
batch 2 (one measurement each; the latency difference method is sensitive to the prefill
time of the 1-token run). The kernel numbers repeat the milestone 4 values within 0.05 ms
(TP2 fp32 5.896 against 5.852 at M = 1).

FP8 repeated 3 times (2026-10-28, banff-cyxtera-s70-4, job 68197384, `scratch/logs/fp8x3/`,
three independent `vllm_latency_bench.sh` runs; each value is already the median of 10
iterations), engine-level TPOT in ms, median (range):

| Batch | vLLM TP2 FP8, 3 runs | TP2 kernel, same allocation (`scratch/logs/opt1/`, before round 1) | ratio |
|---:|---:|---:|---:|
| 1 | 9.822 (9.721 to 9.858) | 5.766 | 1.70x |
| 2 | 10.716 (10.578 to 11.055) | 6.340 | 1.69x |
| 4 | 10.660 (10.468 to 11.349) | 7.424 | 1.44x |
| 8 | 11.152 (11.006 to 12.185) | 8.562 | 1.30x |

vLLM's time includes its per-step scheduler, sampler and output processing; the kernel's
host loop is an argmax and a token copy. The spread between runs (up to 1.2 ms at batch 8)
comes from the 65-token latency, not from the 1-token run. Batch 4 is still at or below
batch 2 in two of the three runs; the medians differ by 0.06 ms, inside the spread.
The final table with the optimized kernel is in section 11, optimization log.

### Milestone 5: reproducibility

Each run repeats `scratch/m3/plans/final.sh` (gates and speed, TP1 on GPU 0 of the same
node) from a frozen copy of the final source, started by `scratch/m3/m5_watch.sh` when the
held job starts. `scratch/m3/final_summary.py` gives the table.

| Run | Node, date, job | G1 ratio | G2 KL | G3-G5 | Logits hash (M = 1, 8) | TP2 faster than TP1, M = 1 / 2 / 4 / 8 | TP2 ms, M = 1 / 2 / 4 / 8 |
|---|---|---:|---:|---|---|---|---|
| 1 | banff-cyxtera-s70-4, 2026-10-28, 68197384 (`final/`) | 1.07 | +9.6 % | PASS | dc59fa9e, eca6810e | 27.7 / 26.9 / 24.6 / 23.2 % | 5.593 / 5.972 / 6.727 / 7.912 |
| 2 | banff-cyxtera-s70-4, 2026-10-29, 68198806 (`m5_day2/`) | 1.07 | +9.6 % | PASS | dc59fa9e, eca6810e | 28.3 / 27.7 / 25.1 / 24.1 % | 5.613 / 5.993 / 6.742 / 7.909 |
| 3 | quanta-ccs-aus-j02-45 (MI325X, AUSTIN), 2026-10-29, 68206615 (`m5_node2/`) | 1.07 | +9.6 % | PASS | dc59fa9e, eca6810e | 23.5 / 24.2 / 21.4 / 19.2 % | 5.212 / 5.524 / 6.248 / 7.317 |

Milestone 5 passes. The second node (run 3, MI325X, `scratch/m5/austin.sh`, checkpoints
downloaded from HF there) computes the same bits as banff: the same logits hashes and the
same gate values to six digits, so the downloaded weights equal ours (the size listing
check in its log differs only in form; the hashes are the proof). The speed gate for a
different GPU model passes: TP2 is 6.8 / 7.5 / 7.2 / 7.5 % faster than on banff, not
slower. TP1 there: 6.813 / 7.283 / 7.949 / 9.058 ms, so TP2's gain over TP1 is 19-24 %
on MI325X. The vLLM rank check did not run there (section 14).

The second day computes the same bits (same hashes, same gate values) and the same speed
within 0.4 %. TP1 on that day: 7.824 / 8.284 / 9.004 / 10.415 ms. The rank check in vLLM
ran twice more on day 2: 4230 and 4237 launches, 0 with differing ranks
(`m5_day2/r*/rankcheck_r*.txt`); those two runs also summed every GDN state each step (a
diagnosis since dropped: the sum cost 3.5 ms a step, and the logged sums were those of
warm-up launches), so their TPOT is not a measurement.

### Milestone 6: vLLM at TP2 (2026-10-28, banff-cyxtera-s70-4, job 68197384)

Branch `users/sareeder/qwen38-gfx942-tp2` of vLLM (fork `SamuelReeder/vllm`), on top of
`users/sareeder/qwen38-gfx942-specialized-decode`. Commits: the exported TP-aware kernel
(`scripts/export_vllm.py`; the mailbox comes from DSv4.1's `sync.py`, and the export fails
if `mk/sync.py` differs from it), TP2 in the spec and runtime (`PeerBuffer` of DSv4.1 at
model load after a `tp_agree`; one flag buffer and one peer buffer for all widths; per-rank
cache checks; weights packed from vLLM's local shards, checked on CPU against
`mk/pack.py` for both ranks), and an opt-in check that both ranks plan the same step
(`VLLM_SPECIALIZED_DECODE_CHECK_PATHS=1`). Ruff 0.14 clean on every changed file.
Conformance kit and Qwen spec tests on the node, in the nightly image with the branch
overlaid: 71 passed (`scratch/logs/m6b/m6_kit.r0.log`), including the kit at TP2.

Kernel on against kernel off, `bench/vllm_e2e.py` (8 prompts, greedy, 128 tokens; TPOT =
(time of 256 tokens - time of 1 token) / 255, median of 3; both runs MRv2, MXFP4
checkpoint; `scratch/logs/m6a/`, `m6b/`, `m6c/`, `m6d/`):

| Batch | TPOT kernel on (3 runs) | kernel off (3 runs) | speedup | TP1 kernel on, same overlay method (2 runs) |
|---:|---:|---:|---:|---:|
| 1 | 5.58 (5.56 to 5.59) | 24.48 (24.42 to 24.51) | 4.38x | 7.68 / 7.72 |
| 2 | 6.06 (6.02 to 6.06) | 24.76 (24.72 to 24.82) | 4.09x | 8.19 / 8.54 |
| 4 | 6.87 (6.84 to 6.88) | 25.10 (25.06 to 25.13) | 3.65x | 8.94 / 8.99 |
| 8 | 8.15 (8.08 to 8.17) | 25.75 (25.69 to 26.00) | 3.16x | 10.47 / 10.53 |

(Kernel-on runs: `m6a/on`, `m6b/on_r1`, `m6b/on_r2`; off: `m6a/off`, `m6b/off_r1`,
`m6b/off_r2`. vLLM's own path emulates MXFP4 on gfx942, so "off" is slow.)

Greedy outputs. Kernel on against kernel off: 0 of 8 prompts token-identical; the first
differing token is at positions 3, 1, 48, 2, 23, 28, 1, 16. The TP1 kernel inside vLLM
gives the same picture against its kernel-off run (0 of 8; positions 25, 16, 24, 1, 9, 33,
1, 20; `scratch/logs/tp1_e2e_ref/`): the kernel's fp8 activations are not vLLM's bf16
emulation, and section 9 (not token identity) is the accuracy oracle. The largest logprob
difference over the common prefixes is 0.38 at TP2 and 0.43 at TP1.

Rank equality inside vLLM (T2), `scratch/m6/rankcheck/` (a sitecustomize hook that
all-gathers the kernel's output after every launch, CUDA-graph replays included): 4233
launches, 0 with any differing bit between the ranks, and both ranks end at epoch
1358793 = 4233 x 321 (`scratch/logs/m6d/rankcheck_r*.txt`). The path check ran in
`m6a/on_check` with no failure.

Run-to-run repeatability inside vLLM at TP2 (open, not a kernel defect as far as measured):
- kernel off, TP2: two runs give 6 or 7 of 8 identical prompts (first logprob difference
  at positions 16, 66, 98), with vLLM's custom all-reduce and with RCCL only (`m6c/`);
- kernel on, TP2: 6 of 8 identical; the first logprob difference is at decode position 1
  or 2, never at position 0 (the prefill token);
- kernel on, TP1: 8 of 8 identical in two runs and against the run of 2026-10-23
  (`m6b/tp1on_r*`);
- standalone, the same TP2 kernel is bitwise deterministic (section 9) and its ranks are
  equal in vLLM (above).
vLLM's own TP2 path is thus not repeatable, and the fp8 activations of the kernel turn a
small input difference into a different token sooner. A check that the GDN states before
the first kernel launch differ between runs (checksums in the same hook) did not run: the
node's container scratch (/var/tmp) was full (`scratch/logs/m6d/driver_m6d3.log`). It is
planned for the second node (milestone 5).

## 12. Rejected alternatives

| Alternative | Reason |
|---|---|
| RCCL or vLLM all-reduce between launches (one launch per layer or per stage) | 128 x 3.3-16 us plus a launch boundary each; gives up the one-launch design |
| Reduce-scatter + all-gather | for 2 ranks each link direction carries the same bytes; one more round trip |
| Data parallel (rows split over GPUs) | does not cut the weight stream per GPU at M = 1 |
| Replicated LM head | 0.33 ms more per step at M = 1; vLLM splits it anyway |
| Replicated embedding in the standalone engine | does not match vLLM's shards; the exchange costs about 1 us |
| Float atomics for the reduction | not deterministic (T2, T5) |
| 16-byte records (3 values + 1 tag) | 16-byte single-copy atomicity over XGMI is not given; DSv4.1 tags every word; copy, do not invent |
| One region per all-reduce instead of two halves | not needed (section 6 proof); DSv4.1 uses parity |
| Merge the all-reduce into the next prologue now | see section 5; milestone 4 candidate only |
| Choose fp32 or bf16 now | decided by milestone 3 measurements (section 9) |

## 13. Stop conditions

Return `STOP: <invariant>` with evidence, no workaround:
- T1: a launch is not fully resident on either GPU.
- T2: ranks differ in any bit of the residual or the normed hidden state.
- T3 or T5: a non-exact all-reduce result, a hang, or run-to-run bit differences.
- T7: neither format passes the TP1 bound.
- Fixed cost: a TP2 GEMV stage's fixed cost exceeds the TP1 fit by more than 3 us at M = 1
  (milestone 4 item 1).
- All-reduce under load: no longer a gate (resolved 2026-10-28, after milestone 4). This limit
  (> 2x the spike: 2.3 us fp32, 1.6 us bf16 at M = 1) protected the go/no-go prediction.
  The measured end-to-end result replaced that prediction: TP2 fp32 is 25.9 % faster at
  M = 1 and 18.0 % faster at M = 8 than TP1 (same node, same day, section 11). The cost
  under load is now a measured value (2.74 us fp32, 3.35 us bf16 at M = 1, section 11) and
  an optimization target. T8 stays the speed gate.
- T8: TP2 at M = 1 is not at least 15 % faster than TP1 on the same node:
  `STOP: TP2 not faster`.
- vLLM: the peer handshake fails, or kernel-on and kernel-off greedy outputs differ beyond
  the kernel's TP1 behaviour.

## 14. Decisions and open questions

Decided 2026-10-27: PeerBuffer at model load (section 8); the bound factor rule (section 9);
`ar_local` is a hashed `Cfg` field that the vLLM spec refuses; `ar_dtype = "fp32"` after
milestone 3 (section 9).

Decided after milestone 3: the gates of section 9 (hard: median per-step error <= 1.2 x
TP1, KL within 10 % of TP1, rank equality, determinism, batch invariance; report only: c1
and the TP1 bound). The 1-layer check stays, against the reference in TP2's sum order.

Decided 2026-10-28 (review of section 11): the agent decides from here on, takes the
conservative option, and records each decision below with its evidence and the rejected
alternative. Optimization order: deferred poll (two variants at most), the second
`grid_wait` of `barrier_then`, the SGPR spill gap, K-chunk tuning, 16-byte pushes, and the
prologue merge only if the wait is still the largest part. Optimization stops when two
candidates in a row gain less than 1 % at M = 1 and M = 8. A candidate that breaks a hard
gate is reverted and logged.

- `ar_dtype = "bf16"` stays an option, not the default. Evidence: it fails the hard gate of
  median per-step error (section 9, milestone 3: bound 13/18 against fp32 18/18; c1 13/18).
  The cause of its later down-stage push at M = 1 (section 11, timing split: +4.7 us on the
  last push of every down stage, same GEMV code, same stream time) is still open; it is not
  investigated, because bf16 cannot become the default. Rejected: investigate it now.

Optimization round 1 (2026-10-28, banff-cyxtera-s70-4, job 68197384, `scratch/logs/opt1/`;
`bench/bench_decode.py` ctx 512, 64 steps, median of 3 fixed-token TPOT, ms):

- Deferred poll, kept. The GEMV's last arriver pushes every counter group it completes
  first, then polls and stores them, so the round trips of a wave's groups overlap. TP2
  fp32: 5.766 -> 5.683 (M = 1, -1.4 %), 6.340 -> 6.158 (M = 2, -2.9 %), 7.424 -> 7.088
  (M = 4, -4.5 %), 8.562 -> 8.207 (M = 8, -4.1 %). Logits bitwise equal (determinism hash
  dc59fa9ecba1e506 at M = 1, eca6810ec44f5829 at M = 8; ranks equal). The order of the
  adds does not change, so the section 9 gates keep their values. It is now the only TP
  all-reduce path (no Cfg switch). Second variant not tried: the remaining choice (one
  combined poll over a wave's groups instead of one poll each after all pushes) only
  changes how the already-overlapped polls are issued; the conservative option is the
  simpler code. Rejected: keep the in-epilogue poll as an option (dead code).
- Second `grid_wait` in `barrier_then`, removed. It came with the profiling commit 6f01725
  without a comment; the second wait is always satisfied at once (epochs only grow), so it
  only adds a leader store, a poll and a workgroup barrier per barrier. TP1: 7.840 -> 7.743
  (M = 1, -1.2 %), 10.425 -> 10.366 (M = 8, -0.6 %); TP2 (before the deferred poll):
  5.766 -> 5.698 (M = 1), 8.562 -> 8.539 (M = 8). Logits bitwise equal in both TP1 and TP2
  (hashes 2a0a1919674a3562, 49c5e7cb71e4be94 for TP1). Separate TP1 commit 5e151e9 on
  `master` of `repos/qwen38-gfx942-mk`. TP1 suite on master + 5e151e9
  (`scratch/logs/tp1fix/`): test_decode M = 1, 2, 4, 8 PASS with the same values to six
  digits as before; determinism M = 1, 8 PASS; eval_text per-passage lines identical
  (ppl 1.661, KL 0.03904).

Optimization round 2 (2026-10-28, same node and allocation, `scratch/logs/opt2/`; base =
round 1 result: 5.599 / 6.094 / 7.026 / 8.141 ms at M = 1 / 2 / 4 / 8):

- SGPR spill gap, rejected. Candidate: the peer bases pass through opaque moves, so the
  peer address math is not hoisted out of the layer loop. SGPR spills 607 -> 499 (ISA of
  the M = 1 build), but TPOT 5.599 -> 5.602 (M = 1, +0.1 %), 8.141 -> 8.209 (M = 8,
  +0.8 %). Logits bitwise equal. Gain below 1 %: not kept (no code).
- K chunks at TP2 (MK_CHUNKS override), first look: lm = 4 gives 5.525 (M = 1, -1.3 %) and
  8.078 (M = 8, -0.8 %); out = 1, o = 1, down = 2 gives 5.505 (-1.7 %) but 8.390 (+3.1 %);
  down = 2 alone is slower (5.713, 8.265). The gains are close to the run-to-run spread
  (about 0.8 % at M = 1), so round 3 repeats lm = 4 and lm = 4 + out = 1 + o = 1
  interleaved with the base before a decision.
- 16-byte pushes, kept. The even lane of a column pair stores both (value, tag) pairs in
  one 16-byte store (`Mailbox.put_words`, two tagged words; the protocol is unchanged:
  every word keeps its tag), and the poll reads the pair of pairs. TPOT 5.599 -> 5.588
  (M = 1, -0.2 %), 8.141 -> 7.895 (M = 8, -3.0 %). Logits bitwise equal (same hashes as
  round 1 at M = 1 and 8, ranks equal). Now the only fp32 path.

Optimization round 3 (2026-10-28, same node and allocation, `scratch/logs/opt3/`; 16-byte
pushes in the base; each variant 2 interleaved runs of 3 reps, median of 6):

- K chunks, rejected. TPOT ms (base / lm = 4 / lm = 4 + out = 1 + o = 1): M = 1 5.575 /
  5.527 (-0.8 %) / 5.518 (-1.0 %); M = 2 5.979 / 5.942 / 5.950; M = 4 6.712 / 6.650 / 6.697;
  M = 8 7.883 / 7.849 (-0.4 %) / 7.979 (+1.2 %). lm = 4 gains less than 1 % at M = 1 and
  M = 8; the other variant is slower at M = 8. TP2 keeps TP1's chunk table.
- Timing split of the round 3 base (`scratch/logs/opt3/parts.txt`, `split.txt`): the added
  cost per all-reduce is 2.78 us at M = 1 (was 3.77 us before rounds 1-3, same method) and
  7.17 us at M = 8. Push/poll is still the largest part (85 % at M = 1, 89 % at M = 8);
  the wait for the slower GPU is 23 % / 15 %; drain and barrier are below zero. Per
  epilogue at M = 1: push -> poll end 2.4 us, of which 0.4-0.6 us is the peer arriving
  later (in 39-44 % of the groups). So the prologue merge is tried next (rule above).
- Prologue merge, rejected after a spike (`scratch/ar_spike/arkernel.py` mode `ar_merged`,
  `scratch/logs/merge/`, same node and allocation). Each round: the put only, the grid
  barrier, then every CTA polls all M x 5120 pairs of both ranks (what a merged norm
  prologue must read). Round time per all-reduce, median of 200 launches of 128 rounds:
  M = 1: barrier only 2.03 us, in-epilogue all-reduce 3.47 us, merged 5.00 us; M = 8: 2.23,
  9.87, 22.52 us. The merged round is slower by 1.5 us (M = 1) and 12.7 us (M = 8) before
  it adds the sum of squares over 5120 columns and the h_out stores it would also need.
  The 304-fold read of the uncached region costs more than the poll it hides. Results
  exact and equal on both ranks.

Optimization stops here: every listed candidate is measured. Kept: deferred poll, single
`grid_wait`, 16-byte pushes. Rejected: opaque peer bases (SGPR), K chunks, prologue merge.

Final kernel (after rounds 1-3), run 1: banff-cyxtera-s70-4, 2026-10-28, job 68197384
(`scratch/logs/final/`, `summary.txt`; TP1 on GPU 0 of the same node):

| Gate (section 9) | Result |
|---|---|
| Median per-step 1 - cos to exact, 128 steps (M 1 x 3 seeds, 2, 4, 8 x 3 seeds) | TP2 0.000406, TP1 0.000377, ratio 1.07 (<= 1.2): PASS |
| eval_text KL(bf16 \|\| kernel) | TP2 0.04279, TP1 0.03904, +9.6 % (<= +10 %): PASS |
| Ranks bitwise equal, every step of every TP2 run | PASS |
| Determinism (3 runs), M = 1 and 8 | PASS; logits sha256 dc59fa9ecba1e506, eca6810ec44f5829 (same as round 1) |
| Batch-invariant rows | PASS |

| M | TP1 ms | TP2 ms | TP2 faster by |
|---:|---:|---:|---:|
| 1 | 7.734 | 5.593 | 27.7 % |
| 2 | 8.173 | 5.972 | 26.9 % |
| 4 | 8.919 | 6.727 | 24.6 % |
| 8 | 10.307 | 7.912 | 23.2 % |

Against stock vLLM TP2 FP8 on the same node and day (3-run medians above): 9.822 / 5.593 =
1.76x at batch 1, 10.716 / 5.972 = 1.79x, 10.660 / 6.727 = 1.58x, 11.152 / 7.912 = 1.41x.

Decided 2026-10-29 (milestone 5, second node):
- Allocation. `sbatch --test-only` gave later estimates for the smaller requests (2 GPUs,
  16 CPUs, 64 GB, 1:15) than for the held 8 h jobs: defq 2026-11-16, miopen 2027-03-09.
  One `salloc --immediate=60` round over four requests (MI325X / MI300X, defq / miopen,
  excluding banff-cyxtera-s70-4 and ctr-cx63-mi300x-12) got job 68206615 on
  quanta-ccs-aus-j02-45 (MI325X, AUSTIN) at once. So no new sbatch job and no retry loop
  were started, and the held jobs 68198219 / 68198220 and the queued salloc 68206625 were
  cancelled. Rejected: keep the held jobs as a fall-back (the instruction was to cancel
  as soon as one job starts). `rocke` holds only gfx950 nodes, so it cannot give a gfx942
  second node.
- AUSTIN run. The node has its own `/cluster` and no home, so the frozen copy
  (`/cluster/logs/qwen38-tp2/src-m5`) is not visible there. `scratch/m5/austin.sh` (from
  the 35B `suite_austin.sh`) clones megakernels `users/sareeder/qwen38-tp2` at 57d6a48
  (the same md5 for `mk/*.py`, `tests/*.py`, `bench/bench_decode.py` as `src-m5`, the
  NaN check included), builds the venv and downloads both checkpoints into the
  container's `/tmp`, puts our MXFP4 `config.json` over the downloaded one, runs
  `scratch/m3/plans/final.sh` with local one/two helpers, and returns the logs as a base64
  tar in the job log. The vLLM rank check of `m5.sh` is not run there (it needs the
  vLLM image in the home); milestone 6 covers it on banff.
- Speed gate on a different GPU model (MI325X): the run fails only when TP2 is more than
  3 % slower than on banff-cyxtera-s70-4 (run 1 of section 11, milestone 5 table).
