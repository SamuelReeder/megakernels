# qwen3.8-27b-gfx942 megakernel

The entire decode step of **Qwen3.8-27B** in a single persistent FlyDSL kernel on an AMD
**MI300X (gfx942)**. All 64 layers (48 Gated DeltaNet + 16 gated attention), every state and
cache update and the 248K-vocab LM head run inside one launch per step, for 1 to 16 sequences.

On one MI300X, TP1, 512-token context:

- **Batch 1: 7.56 ms per token, 132 tok/s**, against 15.59 ms (64 tok/s) for the fastest stock
  engine on the same GPU (vLLM 0.29.1 + AITER on the FP8 checkpoint): **2.06×**.
- **Batch 2 / 4 / 8: 2.07× / 1.98× / 1.68×** per user against the same baseline.
- **MXFP4 weights** (`amd/Qwen3.8-27B-Quark-AWQ-MXFP4`), decoded exactly to fp8 for the gfx942
  fp8 matrix cores (gfx942 has no MX MFMA). The kernel's fp8 activations add 0.8% perplexity
  against the same weights with bf16 activations.
- **Bitwise deterministic** and batch-invariant: no float atomics, split-K sums in a fixed order.
- Reproduced on a second MI300X, on an MI300A (228 CUs) and five days later on the same node.

![Decode speed vs stock engines on the same MI300X](docs/throughput.png)

## Quick start (Alola cluster)

```bash
salloc -p miopen -A miopen -J mkdev --constraint="GFX942&MARKHAM" --gres=gpu:1 \
       --cpus-per-task=16 --mem=200G -t 12:00:00 --no-shell
# container: /cluster/images/hipdnn/hipdnn_latest_gfx942.sqsh (pyxis), then once:
bash scripts/setup_env.sh                     # venv: torch 2.14.1+rocm7.2, flydsl 0.4.0
python tests/test_decode.py $M/Qwen3.8-27B-Quark-AWQ-MXFP4 1 64 16      # kernel vs reference
python bench/bench_decode.py $M/Qwen3.8-27B-Quark-AWQ-MXFP4 1 512 64 3  # TPOT, batch 1
```

`scripts/mkrun '<command>'` syncs this directory to the cluster and runs the command in the
held allocation. Tested on MI300X and MI300A in SPX mode (every CU must hold one resident CTA) with
ROCm 7.2, PyTorch 2.14.1 and FlyDSL 0.4.0.

## Results

| Batch | vLLM FP8 + AITER | SGLang BF16 | Megakernel MXFP4 | Speedup |
|---:|---:|---:|---:|---:|
| 1 | 15.59 ms | 16.48 ms | **7.56 ms** | **2.06×** |
| 2 | 16.71 ms | 17.17 ms | **8.08 ms** | **2.07×** |
| 4 | 17.39 ms | 17.74 ms | **8.80 ms** | **1.98×** |
| 8 | 17.21 ms | 18.69 ms | **10.26 ms** | **1.68×** |

Time per output token, one MI300X (banff-cyxtera-s70-4). The stock engines were measured on
the same GPU; vLLM through `vllm bench serve` (scheduler included), SGLang with
`bench_one_batch`. Context 2048 / 4096: 7.86 / 8.18 ms at batch 1. Every run, the other GPUs,
the quality checks and the failed runs are in [`bench/RESULTS.md`](bench/RESULTS.md); the
stock baselines for every checkpoint and engine are in
[`bench/stock/results.md`](bench/stock/results.md).

## How it works

![Architecture: stages per layer, the MXFP4 weight stream inside one CU, the grid barrier](docs/architecture.svg)

- **One CTA per CU**, 512 threads, all resident. Every layer is five stages separated by grid
  barriers; each GEMV's weight tiles are divided among all 2,432 waves.
- **MXFP4 → fp8.** A unit is 16 rows × 128 K (a 1 KiB e2m1 tile plus e8m0 scales), packed in
  the order the waves stream it. One `v_perm_b32` turns 4 e2m1 values into e4m3fnuz exactly;
  `v_mfma_f32_16x16x32_fp8_fp8` multiplies them with the fp8 activations the stage prologue
  wrote to LDS, and the e8m0 scale multiplies each 32-K result.
- **Split-K with a last arriver.** Each wave stores its partial sums; a device-scope counter
  per output group finds the last wave, which sums the partials in a fixed order and runs the
  epilogue (residual add, SiLU-mul, the causal conv of the DeltaNet inputs).
- **Grid barrier:** two levels, one leader per XCD, about 3 µs with 304 CTAs.
- **Mixers** run as wave tasks: the gated delta rule on a 16 × 128 fp32 slice of the state per
  wave; attention as (row, kv head, 64 positions) chunks with QK^T on the bf16 MFMA, merged by
  the last arriver.

The design record, the numerics, the invariants and every measured-and-rejected change are in
[`docs/DESIGN.md`](docs/DESIGN.md).

## Where the time goes

![Per-CU timeline of the first four layers of one decode step](docs/timeline_m1_zoom.png)

Each row is one CU; colored bars are a stage's work, grey is waiting at the barrier for the
slowest CU. The weight stream runs at 3.3-3.7 TB/s, so about 40% of the step is fixed
per-stage latency (barrier, prologue, hand-off). The attention stage at a 512-token context
uses only 36 wave tasks, and the bf16 LM head takes 0.6 ms of the 7.5 ms. The whole step:
[`docs/timeline_m1.png`](docs/timeline_m1.png).

## Repository layout

```
mk/kernel.py       the kernel: stage sequence, grid barriers
mk/gemv.py         the split-K MXFP4 / bf16 GEMV stream
mk/stages.py       GEMV prologues (LDS input) and epilogues
mk/mixer.py        Gated DeltaNet and attention tasks
mk/dev.py, fp4.py  device helpers, MXFP4 tile layout
mk/pack.py         offline weight packing
mk/runner.py       host engine: buffers, pointer tables, one launch per step
mk/reference.py    PyTorch reference of the decode step (exact and fp8-emulating modes)
mk/vendor/         FlyDSL buffer_ops.py (Apache-2.0, header kept)
tests/             kernel vs reference, real-text perplexity, determinism, reference vs HF
bench/             decode benchmark, per-stage profiler, reproducibility script, stock baselines
tools/             plots: throughput chart, per-CU timeline
```

## Tests and benchmarks

```bash
bash tests/run_suite.sh $MODELS OUT_DIR       # everything in bench/RESULTS.md, about 1 h
bash bench/repro.sh $MODELS/Qwen3.8-27B-Quark-AWQ-MXFP4 OUT_DIR   # speed + smoke on another node
python bench/profile_decode.py $MODELS/Qwen3.8-27B-Quark-AWQ-MXFP4 1 512 timeline.json
python tools/timeline.py timeline.json docs/timeline_m1_zoom.png --zoom 0 500
python tools/plot_results.py
```

## Limits

Decode only (the prompt goes in one token per step), greedy argmax outside the kernel, one GPU,
M ≤ 16 rows, gfx942 only. No serving-engine integration yet.
