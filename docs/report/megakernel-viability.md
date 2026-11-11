# Decode megakernels on MI300X: results and viability

*November 2026 · Samuel Reeder · code, logs and design docs: [SamuelReeder/megakernels](https://github.com/SamuelReeder/megakernels)*

## TL;DR

**Verdict: a clear gain for low-latency serving at small batch; the relative speedup shrinks as the batch grows.**

- I wrote decode megakernels for three current models on MI300X (gfx942). One persistent GPU launch runs
  every layer of a decode step, and the kernels run inside vLLM on my branches.
- **With the same FP8 weights, each user gets tokens 1.2-1.6x faster at batch 1 and 1.06-1.13x faster at
  batch 8** than with stock vLLM on the same GPU. Qwen3.8 reaches 2.1x, partly because it also uses the
  smaller MXFP4 checkpoint, which costs accuracy.
- **The relative speedup falls as the batch grows, and batches above 8 are untested.** The kernels are built for
  batch 1-8, and vLLM's own path runs above that and during prefill. The kernel's own weight copy cuts KV-cache
  space by 18-25 %.
- **Cost:** a new kernel for each model. The vLLM integration builds on vLLM PR
  [#61311](https://github.com/vllm-project/vllm/pull/61311), which is not merged yet.

## What a megakernel changes

A normal engine launches hundreds of small kernels for each generated token. At small batch, the GPU spends
much of the step between kernels instead of reading weights. A megakernel keeps one workgroup resident on each
of the 304 compute units for the whole step. It streams every weight once and passes data from one stage to the
next through grid barriers inside the launch. AMD's ATOM and aitermk megakernels target MI355X. vLLM main already
runs a DeepSeek-V4.1 decode megakernel on MI300X (TP4); #61311 moves it onto shared hooks that my kernels plug into.

## Results

![Speedup per batch size](https://raw.githubusercontent.com/SamuelReeder/megakernels/afdf29ed5809e01ddcd17449ed3767433c4f4165/docs/report/speedup.png)

| Model | Weights | Batch 1 per user | Speedup vs stock vLLM, batch 1 → 8 | In vLLM, kernel on, batch 1 |
|---|---|---|---|---|
| [Qwen3.6-35B-A3B](https://github.com/SamuelReeder/megakernels/blob/afdf29ed5809e01ddcd17449ed3767433c4f4165/qwen3.6-35b-a3b-gfx942/README.md) (MoE) | FP8 | 3.11 ms, 322 tok/s | **1.64x → 1.13x** (same-day baseline: 1.72x → 1.25x) | 2.9-3.2 ms |
| [Gemma 4 26B-A4B](https://github.com/SamuelReeder/megakernels/blob/afdf29ed5809e01ddcd17449ed3767433c4f4165/gemma4-26b-a4b-gfx942/README.md) (MoE) | FP8 | 3.74 ms, 267 tok/s | **1.18x → 1.06x** (same-day baseline: 1.64x → 1.37x) | 3.67 ms |
| [Qwen3.8-27B](https://github.com/SamuelReeder/megakernels/blob/afdf29ed5809e01ddcd17449ed3767433c4f4165/qwen3.8-27b-gfx942/README.md) (dense) | MXFP4 * | 7.56 ms, 132 tok/s | **2.06x → 1.68x** | 7.8 ms |
| [Qwen3.8-27B, 2 GPUs (TP2)](https://github.com/SamuelReeder/megakernels/blob/afdf29ed5809e01ddcd17449ed3767433c4f4165/qwen3.8-27b-gfx942/README.md#tensor-parallel-2-two-mi300x-over-xgmi) | MXFP4 * | 5.59 ms, 179 tok/s | **1.76x → 1.41x** vs vLLM TP2 | 5.58 ms |

**How this was measured.**
- **Setup:** one MI300X (two for TP2), 512-token prompts, time per output token, October 22-29.
- **Baseline:** stock vLLM (0.29.1 or 0.32 nightly) on the FP8 checkpoint, with AITER where it works.
  The BF16 checkpoint is up to 5 % faster than FP8 at batch 1 for Qwen3.6 and TP2.
- **Harnesses:** the kernel was timed standalone, and vLLM through its own benchmark, which includes its
  scheduler. The last column is the kernel running inside vLLM.
- **Which day:** the bold ratio uses the best day for stock vLLM. On other days it ran 4-38 % slower on the
  same node, for reasons I did not find.

\* Qwen3.8 runs MXFP4 against vLLM's FP8, because vLLM has no fast MXFP4 path on gfx942. MXFP4 reads half the
bytes of FP8, so part of the gain is the format. The MXFP4 checkpoint scores 4.9 % higher perplexity than BF16
(1.658 against 1.581).

**Correctness.**
- **Accuracy:** perplexity is no more than 1 % above a reference with the same weights (Gemma's is lower).
  Qwen3.6, for example, scores 1.589, against 1.585 for the FP8 reference and 1.584 for BF16.
- **Inside vLLM:** greedy output follows vLLM's own path until the first near-tied token. I have not run a
  task-level evaluation.
- **Determinism:** every kernel is bitwise deterministic, and a request gets the same bits alone or in a batch.
  Earlier versions of each kernel repeated on a second day and on an MI325X; the current Qwen3.6 and Gemma
  kernels have passed their checks on one node only.

## How the gain changes with batch size

For every model, the relative speedup is lower at batch 8 than at batch 1. There are three reasons:

- **Fixed savings:** the megakernel removes a mostly fixed cost per step (launches and the gaps between kernels).
  That cost is a large share of a batch-1 step and a smaller share of a batch-8 step.
- **Shared work:** the work that grows with the batch costs about the same in both engines. In the MoE models,
  more requests select more distinct experts: Gemma reads about 8 experts per layer at batch 1 and about 33
  at batch 8. That extra weight stream is bandwidth-bound in both engines. For Qwen3.6, the kernel's step grows
  from 3.11 to 6.60 ms and vLLM's from 5.33 to 8.23 ms, so the ratio falls from 1.72x to 1.25x.
- **Small-batch design:** the kernels were designed for small batches. Some of their own costs grow with the
  batch: partial-sum hand-offs, routing, and on-chip memory, which is near its limit at batch 8.

**Not yet measured:**
- Batches above 8. The kernels are not built for them; inside vLLM those steps run on vLLM's own path.
- Mixed traffic. Any step that includes a new prompt (prefill) also runs on vLLM's path, so under real
  traffic the gain is smaller than in the table.

**Other effects:** time to first token does not change. KV-cache capacity is 18-25 % lower (Qwen3.6, Gemma),
because the kernel keeps its own copy of the weights. This lowers the possible peak batch and context length.

**Headroom.** These kernels are not at the hardware limit. Barriers and hand-offs take about 40 % of a Qwen3.8
step, and the weight stream reaches 3.3-3.7 TB/s of the GPU's 5.3 TB/s.

## Tradeoffs to weigh

1. **One kernel per model.**
   - Each model has about 2,000-2,400 lines of kernel code, plus a PyTorch reference and the vLLM integration.
   - Qwen3.6 and Gemma took 1-2 days each, with agents, by adapting the earlier kernels. TP2 took
     3 days. The shared parts now live in one common library.
   - New checkpoint formats or new vLLM cache layouts need kernel work.
2. **Strict deployment needs.**
   - The kernel must have every compute unit to itself. A partitioned GPU, or other work on the same GPU,
     can make it hang.
   - A refusal check exists on the hipDNN route but is untested; the default vLLM path has none.
   - Each model needs one binary per batch width (1, 2, 4, 8) for each GPU model, KV layout and TP size
     (Gemma and TP2 also for each max_model_len).
   - The tuning is for gfx942; MI355X would need a port.
3. **Upstream dependency.**
   - My vLLM integration builds on [#61311](https://github.com/vllm-project/vllm/pull/61311), which is open,
     not merged. After it lands, each model needs its own upstream PR.
   - A hipDNN route for shipping precompiled kernels works end to end (same bits, speed within 0.8 %), but
     it waits on the draft [ROCm/rocm-libraries#14255](https://github.com/ROCm/rocm-libraries/pull/14255).

## Links

- Per-model results, every run and log: [Qwen3.6](https://github.com/SamuelReeder/megakernels/blob/afdf29ed5809e01ddcd17449ed3767433c4f4165/qwen3.6-35b-a3b-gfx942/README.md) · [Gemma 4](https://github.com/SamuelReeder/megakernels/blob/afdf29ed5809e01ddcd17449ed3767433c4f4165/gemma4-26b-a4b-gfx942/README.md) · [Qwen3.8](https://github.com/SamuelReeder/megakernels/blob/afdf29ed5809e01ddcd17449ed3767433c4f4165/qwen3.8-27b-gfx942/README.md) · [Qwen3.8 TP2 design](https://github.com/SamuelReeder/megakernels/blob/afdf29ed5809e01ddcd17449ed3767433c4f4165/qwen3.8-27b-gfx942/docs/DESIGN-TP2.md)
- hipDNN distribution route (design and results): [docs/hipdnn-route.md](https://github.com/SamuelReeder/megakernels/blob/afdf29ed5809e01ddcd17449ed3767433c4f4165/docs/hipdnn-route.md)
- Chart data and sources: [docs/report/plot_speedup.py](https://github.com/SamuelReeder/megakernels/blob/afdf29ed5809e01ddcd17449ed3767433c4f4165/docs/report/plot_speedup.py)
