# gemma4-26b-a4b-gfx942 megakernel

The entire decode step of **Gemma 4 26B-A4B** (FP8 checkpoint) in a single persistent
FlyDSL kernel on an AMD **MI300X (gfx942)**. The embedding, all 30 layers (25
sliding-window and 5 full-attention layers, the 128-expert top-8 MoE beside a dense GeGLU
MLP, the sandwich norms) and the final norm run inside one launch per step, for 1 to 8
sequences. The kernel reads and writes vLLM's paged KV cache in place.

On one MI300X, TP1, 512-token prompts of real text, 256 output tokens, 2026-10-29, with
the node's other 7 GPUs idle before and after the run (`bench/results/m8/quiet/`):

- **Batch 1: 3.74 ms per token (267 tok/s); batch 8: 5.88 ms (1360 tok/s total).**
- **Faster than vLLM at every batch, 1 to 8.** vLLM's own path in the same job took 6.13 /
  6.53 / 6.86 / 8.09 ms at batch 1 / 2 / 4 / 8 (1.64x / 1.54x / 1.49x / 1.37x). That day
  vLLM ran 27 to 38 % slower than on 2026-10-28 with idle neighbours too (cause not
  found). Against its 2026-10-28 numbers (4.43 / 4.85 / 5.02 / 6.22 ms) the kernel is
  1.18x / 1.14x / 1.09x / 1.06x faster: the conservative claim.
- **Milestone 8:** the expert stages no longer hand partial sums through last-arriver
  chains: the down stage reduces by row and the gate_up stage folds its K chunks inside
  each CTA. Against the previous kernel in one job: -3 % / -10 % / -25 % / -24 % at batch
  1 / 2 / 4 / 8.
- **Inside vLLM** (vllm#61311 specialized decode, now every width 1 to 8): 3.67 / 4.38 /
  4.68 / 5.77 ms at batch 1 / 2 / 4 / 8 in the same job.
- **FP8 weights** (`RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic`, one scale per output row)
  with fp8 activations: perplexity on five passages 216.2 against 226.8 for the same
  weights with bf16 activations (GEMV inputs) and 242.7 for the BF16 checkpoint.
- **Bitwise deterministic** and batch-invariant: a row gives the same bits alone or in a
  batch of 2, 4 or 8, with or without padding rows. No float atomics. The milestone-8
  changes give the same bits as the previous kernel.

![Decode speed vs vLLM and SGLang on the same MI300X](docs/throughput.png)

## Results

| Batch | vLLM FP8, kernel off (10-28) | vLLM FP8, kernel off (10-29) | vLLM FP8 + megakernel (10-29) | Megakernel FP8 (10-29) | Speedup vs 10-28 / 10-29 vLLM | Total tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.43 ms | 6.13 ms | **3.67 ms** | **3.74 ms** | **1.18x** / 1.64x | 267 |
| 2 | 4.85 ms | 6.53 ms | **4.38 ms** | **4.25 ms** | **1.14x** / 1.54x | 471 |
| 4 | 5.02 ms | 6.86 ms | **4.68 ms** | **4.61 ms** | **1.09x** / 1.49x | 868 |
| 8 | 6.22 ms | 8.09 ms | **5.77 ms** | **5.88 ms** | **1.06x** / 1.37x | 1360 |

Time per output token on banff-cyxtera-s70-4 (MI300X).

- 10-29 columns: one Slurm job (68207195), third try; every GPU of the node read 0 to 0.1 %
  busy (20 samples of `gpu_busy_percent`) before and after it
  (`bench/results/m8/quiet/try3_*`). Tries 1 and 2 had a busy neighbour and gave the same
  numbers within 4 %.
- Megakernel: `bench/bench_decode.py`, real-text prompts, median of 3 runs
  (`bench/results/m8/quiet/bench_m*.txt`). Before milestone 8 (main 13b25e7, same day,
  `bench/results/m8/bench_main_m*.txt` against `bench_new_m*.txt`): 3.87 / 4.63 / 6.12 /
  7.76 ms vs 3.75 / 4.18 / 4.59 / 5.92 ms.
- vLLM 0.32.1rc1 nightly, kernel off and on: `bench/vllm_e2e.py`, the same real-text
  prompts (`quiet/try3_vllm_off.json`, `try3_vllm_on.json`, `try3_vllm_compare.txt`);
  10-28: `bench/results/m6/off4.json`.
- `vllm bench serve` (random-token prompts) and SGLang 0.5.10 `bench_one_batch`
  (2026-10-28): `bench/results/stock/summary.md`. SGLang's FP8 path fails on this checkpoint
  (`This GEMM is not supported!`); vLLM + AITER fails on FP8 (no GELU_TANH in its FP8 MoE).

Quality (`bench/results/m2/` to `m4/`): the PyTorch reference (`mk/reference.py`) matches
HF transformers in fp32 at logits cosine 0.99942 (worst position) and top-1 99.9 % over a
1300-token prompt (`bench/results/m2/ref_vs_hf_fp32.txt`). The kernel against the
reference at M = 1, 2, 4, 8 and contexts 512, 2048 and 4096 (past the 1024 window): all
pass (`bench/results/m4/suite/test_decode_*.txt`). Determinism: three runs bitwise
equal at M = 1 and 8 over 1100 steps. Milestone 8 (`bench/results/m8/final_gates.txt`):
test_decode at M = 1, 2, 4, 8 (CTX 512), determinism and batch invariance at M = 1 and 8
over 1100 steps, and eval_text (the same perplexities, 216.183 and 85.291) pass.

Second node (`bench/results/m5/quanta-ccs-aus-j02-45_68199254/`): every test passes on an
MI325X, with the same perplexities as on the MI300X; TPOT 3.45 / 4.03 / 5.03 / 6.45 ms at
batch 1 / 2 / 4 / 8. Second day on the day-1 MI300X
(`bench/results/m5/banff-cyxtera-s70-4_68198890_day2/`): every test passes, TPOT within
0.6 % of day 1 (random-token prompts, as in `bench/results/m4/suite/`).

## How it works

![Architecture: stages per layer, attention, the MoE block, the deferred residual](docs/architecture.svg)

- **One CTA per CU**, 512 threads, all resident. The embedding is one stage; every layer
  is six (qkv, attention, o_proj, router, routing + gate_up, down), separated by grid
  barriers: 181 per step. Each GEMV's weight units are divided among all 2,432 waves.
- **Attention** runs as wave tasks of (row, KV head, q-head group, 64 positions) over
  vLLM's paged cache: q/k RMSNorm, RoPE (full rotation at θ 1e4 for the sliding layers,
  proportional 64-pair rotation at θ 1e6 for the full layers), scores on the bf16 MFMA,
  partial softmax records merged by the last arriver. Sliding layers read only the last
  1024 positions; full layers use V = K (`attention_k_eq_v`).
- **MoE + dense MLP.** Every CTA computes the top-8 routing of all rows (softmax over 128
  fp32 logits, renormalized, times `per_expert_scale`). One grouped FP8 GEMV streams each
  selected expert once, and the dense 2112-wide GeGLU MLP rides along as three 704-wide
  pseudo-experts. From the Qwen3.6-35B-A3B kernel (copy and diverge), then (milestone 8):
  at batch >= 2 each CTA owns whole output groups of gate_up and folds their K chunks in
  its waves (`e1_cta`), and the down stage's partials are reduced by row in the combine
  CTAs (`e2_rows`): no last-arriver chains, the same bits.
- **Deferred residual.** Gemma's sandwich norms need whole rows after every projection.
  Stages store raw outputs with their sums of squares; consumers rebuild h1 and h. A few
  CTAs publish each rebuilt row in FP8 for the next stage (`router_publish`,
  `combine_rows`), bitwise equal to every CTA computing it.
- **FP8.** A unit is 16 rows x 128 K of e4m3fnuz (the checkpoint's e4m3fn bits) plus 16
  fp32 row scales, packed in the order the waves stream it.

The design record, the numerics, the invariants, the decisions with their evidence and the
optimization log are in [`docs/DESIGN.md`](docs/DESIGN.md).

## Where the time goes

![Per-CU timeline of the first two layers of one decode step at batch 1](docs/timeline_m1_zoom.png)

Each row is one CU; colored bars are a stage's work, grey is waiting at the grid barrier
for the slowest CU. At batch 1 the 180 layer stages take about 3.1 ms and the LM head
0.29 ms (`bench/results/m7/profile_m1.txt`). A layer streams about 100 MB of weights (about
23 us at 4.3 TB/s) but takes about 100 us: the fixed cost per stage (barrier, prologue,
partial hand-off) and the attention task's serial load chain dominate. At batch 8 the
expert stages took 74 us (gate_up) and 68 us (down) per layer before milestone 8, about
twice their weight-stream time (`bench/results/m8/base_profile_m4_m8.txt`); the analysis
and the changes are in `docs/DESIGN.md`, "Milestone 8". Whole step:
[`docs/timeline_m1.png`](docs/timeline_m1.png).

## vLLM integration

Branch [`users/sareeder/gemma4-moe-batch`](https://github.com/SamuelReeder/vllm/tree/users/sareeder/gemma4-moe-batch)
(on `users/sareeder/gemma4-gfx942-specialized-decode`, stacked on the Qwen3.6 branch and
vllm#61311) adds a DecodeSpec for this model; the kernel serves every width, 1 to 8. The
kernel reads vLLM's KV cache in place (TRITON_ATTN, 6 KV cache groups: 5 sliding-window
groups with 16-token blocks, 1 full group with 32-token blocks) and returns the normed
hidden states; vLLM keeps prefill, the LM head with the logit soft-cap, and sampling.

- Conformance: the Gemma 4 spec tests and the #61311 kit pass (75 tests,
  `bench/results/m6/pytest.txt`).
- Greedy outputs with the kernel on follow vLLM's own path until a near-tie: every first
  divergence is at a logprob gap of 1.25 or less, including two prompts of about 2440
  tokens (past the window). vLLM's own path at batch 8 and at batch 1 diverges as often
  (`bench/results/m6/compare4.txt`, `compare_off_b8_b1.txt`).
- The kernel keeps its own repacked copy of the weights: 24.4 GiB, which reduces vLLM's
  KV cache by 18 % (`bench/results/m6/memory.txt`).

## Quick start (Alola cluster)

```bash
salloc -p miopen -A miopen -J mkdev --constraint="GFX942&MARKHAM" --gres=gpu:1 \
       --cpus-per-task=16 --mem=200G -t 12:00:00 --no-shell
# models in ~/mk/models: gemma-4-26B-A4B-it (BF16) and gemma-4-26B-A4B-it-FP8-dynamic
../common/scripts/mkrun . 'python tests/test_decode.py $G4/gemma-4-26B-A4B-it-FP8-dynamic 1 30 16 2048'
../common/scripts/mkrun . 'python bench/bench_decode.py $G4/gemma-4-26B-A4B-it-FP8-dynamic 1 512 256 3'
```

`../common/scripts/mkrun . '<command>'` ships this directory and `common/` to the held allocation (the job named
`mkdev`, or `MK_JOB=<jobid>`) and runs the command in the hipdnn gfx942 image with the venv
of `common/scripts/setup_env.sh` (torch + ROCm, FlyDSL 0.4.0).

## Repository layout

```
mk/kernel.py       the kernel: stage sequence, grid barriers, Cfg (every compile-time setting)
mk/gemv.py         the split-K FP8 / bf16 GEMV stream (one scale per row)
mk/moe.py          routing, the grouped expert GEMV, the expert epilogues
mk/attention.py    attention tasks over the paged KV cache and the record merge
mk/stages.py       GEMV prologues and epilogues, row rebuilds and publishers
mk/dev.py, dims.py device helpers (barrier, DPP reductions), model dimensions
mk/pack.py         FP8 checkpoint -> unit-order tiles
mk/runner.py       host engine: buffers, pointer tables, one launch per step
mk/reference.py    PyTorch reference of the decode step (exact, emulating and fp32 modes)
tests/             kernel vs reference, attention stage, determinism, perplexity, vs HF
bench/             decode benchmark, per-stage profiler, stock baselines, vLLM harnesses
tools/             plots: throughput chart, per-CU timeline
scripts/           cluster runner, reproducibility job, vLLM export
```

## Tests and benchmarks

```bash
bash tests/run_suite.sh $MODELS OUT_DIR       # everything, about 1.5 h on one MI300X
python bench/profile_decode.py $MODELS/gemma-4-26B-A4B-it-FP8-dynamic 1 512 timeline.json
python tools/timeline.py timeline.json docs/timeline_m1_zoom.png --zoom 0 220
python tools/plot_results.py 'bench/results/m7/bench_text_m{b}.txt' \
       bench/results/m6/on4.json bench/results/m6/off4.json
```

## Limits

- Decode only: the prompt goes in one token per step outside vLLM; in vLLM, vLLM runs
  prefill. Greedy argmax outside the kernel.
- One GPU (TP1), 1 to 8 rows, gfx942 (MI300X) only. FP8 checkpoint only.
- Text only: no vision or audio inputs.
- The kernel's weight copy costs 24.4 GiB inside vLLM.
