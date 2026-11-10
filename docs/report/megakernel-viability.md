# Decode megakernels on MI300X: results and viability

*November 2026 · Samuel Reeder · code, logs and design docs: [SamuelReeder/megakernels](https://github.com/SamuelReeder/megakernels)*

## TL;DR

- We wrote whole-model decode megakernels for three current models on MI300X (gfx942): one
  persistent GPU launch runs every layer of a decode step.
- **Each user gets tokens 1.2-2.1x faster at batch 1, and 1.06-1.7x faster at batch 8**, than with stock vLLM on the
  same GPU. Above 8 concurrent requests the kernel hands off to vLLM, so the gain is zero.
- Output accuracy matches vLLM's. The kernels are bitwise deterministic, and they run inside vLLM today on our branches.
- **The tradeoff:** the gain is for interactive, low-concurrency serving only, and each model needs its own kernel.
  Landing it depends on vLLM PR [#61311](https://github.com/vllm-project/vllm/pull/61311).

## What a megakernel changes

A normal engine launches hundreds of small kernels for each generated token. At small batch, the GPU spends
much of the step between kernels instead of reading weights. A megakernel keeps one workgroup resident on each
of the 304 compute units for the whole step. It streams every weight once and passes data from one stage to
the next through grid barriers inside the launch. AMD's own megakernel work (ATOM, aitermk, vLLM's DeepSeek-V4.1
"mono" decode) targets MI355X. The only other MI300X megakernel we know of is DeepSeek-V4.1 at TP2/TP4 in #61311.

## Results

![Speedup per batch size](IMG_SPEEDUP)

| Model (all layers in one launch) | Weights | Batch 1, per user | Speedup vs stock vLLM, batch 1 → 8 | Inside vLLM ([#61311](https://github.com/vllm-project/vllm/pull/61311) branch) |
|---|---|---|---|---|
| [Qwen3.6-35B-A3B](LINK_Q6) (hybrid attention, 256-expert MoE) | FP8 | 3.11 ms, 322 tok/s | **1.64x → 1.13x** (same-day baseline: 1.72x → 1.25x) | 1.3-2.0x vLLM's own path |
| [Gemma 4 26B-A4B](LINK_G) (sliding window, 128-expert MoE) | FP8 | 3.74 ms, 267 tok/s | **1.18x → 1.06x** (same-day baseline: 1.64x → 1.37x) | same speed as standalone, every batch 1-8 |
| [Qwen3.8-27B](LINK_Q8) (hybrid attention, dense) | MXFP4 * | 7.56 ms, 132 tok/s | **2.06x → 1.68x** | same speed as standalone |
| [Qwen3.8-27B, 2 GPUs (TP2)](LINK_TP2) | MXFP4 * | 5.59 ms, 179 tok/s | **1.76x → 1.41x** vs vLLM TP2 | same speed as standalone |

One MI300X (two for TP2), 512-token prompts, time per output token, October 22-29. Baseline: stock vLLM
(0.29.1 or 0.32 nightly) with its fastest settings (FP8 checkpoint, AITER where it works). The
bold number uses the best day we measured for stock vLLM. On other days stock vLLM ran 4-38 % slower on the
same node for reasons we did not find.

\* Qwen3.8 runs the MXFP4 checkpoint against vLLM's FP8, because vLLM has no fast MXFP4 path on gfx942. MXFP4
reads half the weight bytes, so part of that gain is the format, not the kernel. Qwen3.6 and Gemma compare FP8
with FP8.

**Correctness.** Perplexity is no more than 1 % above a reference with the same weights (Gemma's is lower). For
example, Qwen3.6 scores 1.589, against 1.585 for the FP8 reference and 1.584 for BF16. Every kernel is bitwise deterministic, and a request
gets the same bits alone or in a batch. Results repeat on a second day and on a second node (MI325X). This
determinism also found a faulty MI300X on the cluster, where plain PyTorch gave different answers on each run.

## Where it helps, and where it does not

| Serving situation | Effect |
|---|---|
| 1-8 concurrent requests (chat, agents, latency SLAs) | **Faster**: 1.06-2.1x tokens/s per user, and total throughput rises by the same factor |
| More than 8 concurrent requests | **None**: the step runs on vLLM's normal path |
| Time to first token (prefill) | **None**: prefill always runs on vLLM's path |
| Peak server throughput | **None** |

## Tradeoffs to weigh

1. **One kernel per model.** Each model needs about 2,000 lines of kernel code. A shared library of
   about 1,900 lines covers the GEMV stream, FP8/MXFP4 packing and the compile key. With that library and agent-driven
   work, a new model took 1-2 days (Qwen3.6, Gemma) and TP2 took 3. New checkpoint formats and new
   cache layouts in vLLM also need kernel work.
2. **Strict deployment needs.** The kernel needs the whole GPU: every compute unit must hold one resident
   workgroup, so partitioned modes (CPX) are refused. A wrong setup would hang instead of failing, so each
   kernel checks the setup at load and refuses. Binaries are specific to the GPU model, the KV layout and the
   batch width (4 per model per configuration). Tuning is gfx942-specific; MI355X would need a port.
3. **Upstream dependency.** The vLLM integration builds on [#61311](https://github.com/vllm-project/vllm/pull/61311),
   which is open, not merged. After it lands, each model is one more upstream PR. An optional route that ships
   precompiled kernels through hipDNN works end to end (same bits, speed within 0.8 %), but it depends on
   [ROCm/rocm-libraries#14255](https://github.com/ROCm/rocm-libraries/pull/14255), which is still a draft.
4. **Benchmarks drift.** Stock vLLM's speed varied by up to 38 % between days on one node. Any external claim
   needs a same-day, same-node baseline.

**Upside not yet used.** About 40 % of a Qwen3.8 step is fixed cost per stage (barriers and hand-offs), and
the weight stream reaches 3.3-3.7 of the GPU's 5.3 TB/s. Speculative decoding (several tokens checked per
launch) is not in these kernels yet. AMD's largest MI355X gains combine it with their megakernels.

## Recommendation

1. **Land through vLLM.** Push for review of [#61311](https://github.com/vllm-project/vllm/pull/61311), then upstream
   Qwen3.6 first. It has the cleanest result: FP8 against FP8, 1.64x at batch 1.
2. **Choose models by demand.** Build a kernel only for models that serve interactive traffic on MI300X/MI325X.
   Before starting one, agree the split with AMD's MI355X megakernel teams.
3. **Next engineering bet:** speculative-decoding verify inside the kernel, then a cut in the fixed per-stage
   cost. Defer an MI355X port until someone asks for it.
4. **Before any public claim:** re-run all baselines on one node on one day.

## Links

- Per-model results, every run and log: [Qwen3.6](LINK_Q6) · [Gemma 4](LINK_G) · [Qwen3.8](LINK_Q8) · [Qwen3.8 TP2 design](LINK_TP2D)
- hipDNN distribution route (design and results): [docs/hipdnn-route.md](LINK_HIPDNN)
- Chart data and sources: [docs/report/plot_speedup.py](LINK_PLOT)
