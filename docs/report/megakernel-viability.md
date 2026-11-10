# Decode megakernels on MI300X: results and viability

*November 2026 · Samuel Reeder · code, logs and design docs: [SamuelReeder/megakernels](https://github.com/SamuelReeder/megakernels)*

## TL;DR

- We wrote decode megakernels for three current models on MI300X (gfx942). One persistent GPU launch runs
  every layer of a decode step, and the kernels run inside vLLM on our branches.
- **With the same FP8 weights, each user gets tokens 1.2-1.6x faster at batch 1 and 1.06-1.13x faster at
  batch 8** than with stock vLLM on the same GPU. Qwen3.8 reaches 2.1x, partly because it also uses the
  smaller MXFP4 checkpoint, which costs accuracy.
- **The gain is for interactive, low-concurrency decode only.** Above 8 concurrent requests, and during
  prefill, vLLM's own path runs. The kernel's own weight copy cuts KV-cache space by 18-25 %.
- **Cost:** a new kernel for each model, and upstream landing waits on vLLM PR
  [#61311](https://github.com/vllm-project/vllm/pull/61311).

## What a megakernel changes

A normal engine launches hundreds of small kernels for each generated token. At small batch, the GPU spends
much of the step between kernels instead of reading weights. A megakernel keeps one workgroup resident on each
of the 304 compute units for the whole step. It streams every weight once and passes data from one stage to the
next through grid barriers inside the launch. AMD's ATOM and aitermk megakernels target MI355X. For MI300X,
#61311 adds a DeepSeek-V4.1 decode megakernel (TP4) and the hooks that our kernels plug into.

## Results

![Speedup per batch size](https://raw.githubusercontent.com/SamuelReeder/megakernels/25948a617c66239c74240abfa0121aa871e1a190/docs/report/speedup.png)

| Model | Weights | Batch 1 per user | Speedup vs stock vLLM, batch 1 → 8 | In vLLM, kernel on, batch 1 |
|---|---|---|---|---|
| [Qwen3.6-35B-A3B](https://github.com/SamuelReeder/megakernels/blob/25948a617c66239c74240abfa0121aa871e1a190/qwen3.6-35b-a3b-gfx942/README.md) (hybrid attention, 256-expert MoE) | FP8 | 3.11 ms, 322 tok/s | **1.64x → 1.13x** (same-day baseline: 1.72x → 1.25x) | 2.9-3.2 ms |
| [Gemma 4 26B-A4B](https://github.com/SamuelReeder/megakernels/blob/25948a617c66239c74240abfa0121aa871e1a190/gemma4-26b-a4b-gfx942/README.md) (sliding window, 128-expert MoE) | FP8 | 3.74 ms, 267 tok/s | **1.18x → 1.06x** (same-day baseline: 1.64x → 1.37x) | 3.67 ms |
| [Qwen3.8-27B](https://github.com/SamuelReeder/megakernels/blob/25948a617c66239c74240abfa0121aa871e1a190/qwen3.8-27b-gfx942/README.md) (hybrid attention, dense) | MXFP4 * | 7.56 ms, 132 tok/s | **2.06x → 1.68x** | 7.8 ms |
| [Qwen3.8-27B, 2 GPUs (TP2)](https://github.com/SamuelReeder/megakernels/blob/25948a617c66239c74240abfa0121aa871e1a190/qwen3.8-27b-gfx942/README.md#tensor-parallel-2-two-mi300x-over-xgmi) | MXFP4 * | 5.59 ms, 179 tok/s | **1.76x → 1.41x** vs vLLM TP2 | 5.58 ms |

**How this was measured.**
- **Setup:** one MI300X (two for TP2), 512-token prompts, time per output token, October 22-29.
- **Baseline:** stock vLLM (0.29.1 or 0.32 nightly) on the FP8 checkpoint, with AITER where it works.
  The BF16 checkpoint is up to 5 % faster than FP8 at batch 1 for Qwen3.6 and TP2.
- **Harnesses:** the kernel was timed standalone, and vLLM through its own benchmark, which includes its
  scheduler. The last column is the kernel running inside vLLM.
- **Which day:** the bold ratio uses the best day for stock vLLM. On other days it ran 4-38 % slower on the
  same node, for reasons we did not find.

\* Qwen3.8 runs MXFP4 against vLLM's FP8, because vLLM has no fast MXFP4 path on gfx942. MXFP4 reads half the
bytes of FP8, so part of the gain is the format. The MXFP4 checkpoint scores 4.9 % higher perplexity than BF16
(1.658 against 1.581).

**Correctness.**
- **Accuracy:** perplexity is no more than 1 % above a reference with the same weights (Gemma's is lower).
  Qwen3.6, for example, scores 1.589, against 1.585 for the FP8 reference and 1.584 for BF16.
- **Inside vLLM:** greedy output follows vLLM's own path until the first near-tied token. We have not run a
  task-level evaluation.
- **Determinism:** every kernel is bitwise deterministic, and a request gets the same bits alone or in a batch.
- **Repeat runs:** earlier versions of each kernel gave the same results on a second day and on an MI325X. The
  current Qwen3.6 and Gemma kernels have passed their checks on one node only.

## Where it helps, and where it does not

| Serving situation | Effect |
|---|---|
| 1-8 concurrent requests in decode (chat, agents, latency targets) | **Faster**, by the ratios above |
| More than 8 concurrent requests, or any step that includes prefill | **None**: vLLM's own path runs |
| Time to first token | **None** |
| KV-cache capacity, and therefore peak throughput and long context | **Lower**: the kernel's weight copy uses 18-25 % of the KV cache (Qwen3.6, Gemma) |

All speed numbers come from steady decode with a fixed batch. Under real traffic, any step that mixes in a new
prompt runs on vLLM's path, so the gain will be smaller. We have not measured how much.

## Tradeoffs to weigh

1. **One kernel per model.**
   - Each model has about 1,900-2,300 lines of kernel code, plus a PyTorch reference and the vLLM integration.
   - Qwen3.6 and Gemma took 1-2 days each, with agents, by adapting the earlier kernels. TP2 took
     3 days. The shared parts now live in one common library.
   - New checkpoint formats or new vLLM cache layouts need kernel work.
2. **Strict deployment needs.**
   - The kernel must have every compute unit to itself. A partitioned GPU, or other work on the same GPU,
     can make it hang.
   - A check that turns this into a refusal is designed but not yet tested.
   - Each model needs one binary per batch width (1, 2, 4, 8) for each GPU model, KV layout and TP size.
   - The tuning is for gfx942; MI355X would need a port.
3. **Upstream dependency.**
   - Our vLLM integration builds on [#61311](https://github.com/vllm-project/vllm/pull/61311), which is open,
     not merged. After it lands, each model needs its own upstream PR.
   - An optional route ships precompiled kernels through hipDNN. It works end to end, with the same bits and
     speed within 0.8 %. It waits on [ROCm/rocm-libraries#14255](https://github.com/ROCm/rocm-libraries/pull/14255),
     which is a draft.

## Recommendation

1. **Land through vLLM.** Push for review of [#61311](https://github.com/vllm-project/vllm/pull/61311). Then
   upstream Qwen3.6 first, because it has the clearest result: FP8 against FP8, 1.64x at batch 1.
2. **Measure real serving before any public claim.**
   - Run `vllm bench serve` with the kernel on and off, with arrivals at concurrency 1-8, on one node and day.
   - Report the KV-cache cost next to the speedup.
3. **Build kernels only where there is demand,** meaning models that serve interactive traffic on
   MI300X/MI325X. Agree the split with AMD's MI355X megakernel teams before starting one.
4. **Next engineering work, if we continue:**
   - Cut the fixed cost per stage: barriers and hand-offs are about 40 % of a Qwen3.8 step, and the weight
     stream reaches 3.3-3.7 TB/s of the GPU's 5.3 TB/s.
   - Share one weight copy with vLLM, which recovers the KV cache.
   - Verify several speculative tokens in each launch.

## Links

- Per-model results, every run and log: [Qwen3.6](https://github.com/SamuelReeder/megakernels/blob/25948a617c66239c74240abfa0121aa871e1a190/qwen3.6-35b-a3b-gfx942/README.md) · [Gemma 4](https://github.com/SamuelReeder/megakernels/blob/25948a617c66239c74240abfa0121aa871e1a190/gemma4-26b-a4b-gfx942/README.md) · [Qwen3.8](https://github.com/SamuelReeder/megakernels/blob/25948a617c66239c74240abfa0121aa871e1a190/qwen3.8-27b-gfx942/README.md) · [Qwen3.8 TP2 design](https://github.com/SamuelReeder/megakernels/blob/25948a617c66239c74240abfa0121aa871e1a190/qwen3.8-27b-gfx942/docs/DESIGN-TP2.md)
- hipDNN distribution route (design and results): [docs/hipdnn-route.md](https://github.com/SamuelReeder/megakernels/blob/25948a617c66239c74240abfa0121aa871e1a190/docs/hipdnn-route.md)
- Chart data and sources: [docs/report/plot_speedup.py](https://github.com/SamuelReeder/megakernels/blob/25948a617c66239c74240abfa0121aa871e1a190/docs/report/plot_speedup.py)
