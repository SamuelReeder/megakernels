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
- Peer memory: `PeerBuffer` construction is collective and agrees on the mapping. In DSv4.1
  it is constructed in the runner constructor at model load (`mono/runner.py:241`), not at the
  first eager step; only the per-width scratch is allocated at the first eager step
  (`mono/runner.py:250-266`). The objective says "bind the peer buffers at the first eager
  decode step, as DSv4.1 does". The design follows what DSv4.1 does: allocate and map at
  build (after the build `tp_agree`), bind the per-width scratch at the first eager step.
  This needs your confirmation.
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
     1.5 x (1 - worst cos(TP1, exact)) + 1e-5, and mean top-1 agreement with exact >= TP1's
     - 0.05. The factor 1.5 absorbs the run-to-run spread of the chaotic worst cosine; review
     it.
- Local check before the 64-layer runs: 1 layer, TP2 `fp32` against TP1, max relative
  difference of the residual <= 1e-5 (only the fp32 sum grouping differs).
- Ranks: the residual buffer and the normed hidden state compared bitwise across ranks after
  every step of `test_decode` and `test_determinism`.
- `test_determinism`: 3 runs bitwise-equal per rank, M = 1 and 8.
- `eval_text`: perplexity, top-1, KL(bf16 || kernel) per passage, and the batch-invariance pass.

Partial format choice (milestone 3, both formats run): default `bf16` if it passes the TP1
bound above and its `eval_text` KL against BF16 is within 10 % of TP1's
(`bench/results/banff-cyxtera-s70-4_suite_41b0956/eval_text.txt`); otherwise `fp32`. The
comparison table goes here:

| Format | test_decode worst cos (exact) | top-1 | eval_text KL(bf16 \|\| kernel) | ppl | Decision |
|---|---|---|---|---|---|
| TP1 | | | | | reference |
| TP2 fp32 | | | | | |
| TP2 bf16 | | | | | |

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
- All-reduce under load: the measured cost at M = 1 is more than twice the spike's
  (> 2.3 us fp32, > 1.6 us bf16), or it pushes the projection below 15 %.
- T8: TP2 at M = 1 is not at least 15 % faster than TP1 on the same node:
  `STOP: TP2 not faster`.
- vLLM: the peer handshake fails, or kernel-on and kernel-off greedy outputs differ beyond
  the kernel's TP1 behaviour.

## 14. Open questions

1. PeerBuffer timing in vLLM: build time (what DSv4.1 does) or first eager step (what the
   objective states). Section 8 proposes build time.
2. The TP1-bound factor (1.5 on 1 - worst cos, 0.05 on top-1) in section 9.
3. The `ar_local` debug field: in `Cfg` (hashed, refused by the vLLM spec) as proposed, or a
   separate profiling build path.
