# Results: Qwen3.8-27B decode on one MI300X

All dates are 2026-10-22 (US Eastern). Raw outputs are in `bench/results/` (megakernel)
and `bench/stock/raw/` (stock engines).

## Setup

- Megakernel: commit `96466b3` (kernel code; later commits change docs and tests
  only), container `hipdnn_20261017_gfx942.sqsh`, torch 2.14.1+rocm7.2, FlyDSL 0.4.0,
  checkpoint `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`.
- Stock baseline (`bench/stock/results.md`): vLLM 0.29.1 (ROCm 10.1.0 image) and
  SGLang 0.5.10.post1, TP1, CUDA/HIP graphs on. The fastest stock configuration on
  MI300X is vLLM + `VLLM_ROCM_USE_AITER=1` on `Qwen/Qwen3.8-27B-FP8`.
- Workload: 512 tokens of context, then decode. TPOT = time per output token of one
  sequence; tok/s/user = 1000 / TPOT.

Methods differ: `vllm bench serve` measures TPOT through the server (scheduler,
sampler, detokenizer included). `bench/bench_decode.py` times 64 back-to-back decode
steps of the engine in `mk/runner.py` (launch + argmax on the GPU, no scheduler).
SGLang's `bench_one_batch` (a static batch, no server) is the closest stock analogue.

## Speed, same GPU (banff-cyxtera-s70-4)

The stock rerun ran on the GPU of the megakernel allocation at 23:08, 9 hours after
the first stock run; it reproduced the first run within 0.6 % (0.1 % at batch 1).

| Batch | vLLM FP8 + AITER TPOT ms | SGLang BF16 (one batch) TPOT ms | Megakernel TPOT ms | Speedup vs vLLM FP8 + AITER |
|---:|---:|---:|---:|---:|
| 1 | 15.59 (15.59-15.60, 3 runs) | 16.48 | **7.56** (7.54-7.58, 3 runs) | **2.06x** |
| 2 | 16.71 | 17.17 | **8.08** | **2.07x** |
| 4 | 17.39 | 17.74 | **8.80** | **1.98x** |
| 8 | 17.21 | 18.69 | **10.26** | **1.68x** |

Batch 1: 132 tok/s/user against 64 tok/s/user.

Context length (megakernel, batch 1): 7.56 ms at 512, 7.86 ms at 2048, 8.18 ms at 4096.

## Speed on other gfx942 GPUs (same kernel commit)

| Node | GPU | Batch 1 | Batch 2 | Batch 4 | Batch 8 |
|---|---|---:|---:|---:|---:|
| ctr-cx66-mi300x-13, megakernel | MI300X | 8.48 | 8.96 | 10.04 | 12.01 |
| ctr-cx66-mi300x-13, vLLM FP8 + AITER, same GPU | MI300X | 16.47 | 18.07 | 18.82 | 18.60 |
| speedup | | 1.94x | 2.02x | 1.88x | 1.55x |
| ctr-cx66-mi300x-13, megakernel, separate earlier job | MI300X | 8.46 | 8.94 | 9.99 | 12.01 |
| ctr-rack33-mi300a-27, megakernel | MI300A (228 CUs) | 7.49 | 7.89 | 8.67 | 9.60 |

On ctr-cx66-mi300x-13 the stock engine is 6 to 8 % slower and the megakernel 11 to 17 %
slower than on banff-cyxtera-s70-4; the speedup there is 1.55x to 2.02x. The kernel adapts the grid
to the CU count (228 on MI300A). Run-to-run spread is below 1 % everywhere.

## Correctness and quality

| Check | Result |
|---|---|
| Reference vs HF transformers, BF16 checkpoint (`tests/ref_vs_hf.py`) | logits cosine min 0.99989, top-1 agreement 95.7 % (one tie, "Moon" vs "moon") |
| Kernel vs reference, 64 layers, 16 steps (`tests/test_decode.py`) M = 1, 2, 4, 8 | PASS: the kernel is as close to the exact reference as the emulating reference is (worst cosine 0.989 to 0.9995) |
| Perplexity on 5 passages, 592 tokens (`tests/eval_text.py`) | kernel 1.658; same MXFP4 weights with bf16 activations 1.645 (+0.8 %); BF16 model 1.581 |
| Top-1 agreement with the BF16 model | 92.9 % to 98.7 % per passage (MXFP4 reference: 93.9 % to 99.3 %) |
| Batch invariance (5 passages at M = 5 vs alone) | bitwise equal |
| Determinism (3 runs, 64 steps, M = 1 and 8) | bitwise equal |
| Determinism and kernel vs reference on ctr-cx66-mi300x-13 and MI300A | PASS |

Most of the quality loss against the BF16 model comes from the MXFP4 checkpoint
itself; the kernel's fp8 activations add 0.8 % perplexity.

## Reproduce

```bash
bash tests/run_suite.sh $HOME/mk/models OUT_DIR            # everything above on one GPU
bash bench/repro.sh $HOME/mk/models/Qwen3.8-27B-Quark-AWQ-MXFP4 OUT_DIR   # speed + smoke on another node
```

A third same-GPU stock + megakernel run is queued for 2026-10-23 (Slurm job
68152179, output `~/mk/repro_out/pair_*_68152179` and `~/mk/pair_68152179.log`).
