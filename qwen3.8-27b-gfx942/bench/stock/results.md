# Stock serving baseline: Qwen3.8-27B decode TPOT on one MI300X (gfx942)

All runs were made on 2026-10-22 on node `banff-cyxtera-s70-4` (Alola, MARKHAM, partition `miopen`). Each run used one GPU: AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, 192 GiB. All runs used TP1, text-only, and the engine-default CUDA/HIP graphs. Prefix caching was off, or not applicable. The workload is 512 input tokens and 256 output tokens.

## Result

The batch 1 rows give the median of 3 runs. The range of the 3 runs is in brackets. The rows for batch 2, 4 and 8 come from one run. tok/s/user = 1000 / TPOT.

| checkpoint | engine+version | batch | TPOT median ms | tok/s/user | node | date |
|---|---|---:|---:|---:|---|---|
| BF16 `Qwen/Qwen3.8-27B` | vLLM 0.29.1.dev0+g98dff2a81 (default) | 1 | 16.42 [16.29–16.42] | 60.9 | banff-cyxtera-s70-4 | 2026-10-22 |
| BF16 | vLLM 0.29.1 (default) | 2 | 18.11 | 55.2 | banff-cyxtera-s70-4 | 2026-10-22 |
| BF16 | vLLM 0.29.1 (default) | 4 | 20.54 | 48.7 | banff-cyxtera-s70-4 | 2026-10-22 |
| BF16 | vLLM 0.29.1 (default) | 8 | 21.78 | 45.9 | banff-cyxtera-s70-4 | 2026-10-22 |
| BF16 | vLLM 0.29.1 + `VLLM_ROCM_USE_AITER=1` | 1 | 16.35 [16.25–16.38] | 61.2 | banff-cyxtera-s70-4 | 2026-10-22 |
| BF16 | vLLM 0.29.1 + AITER | 2 | 18.05 | 55.4 | banff-cyxtera-s70-4 | 2026-10-22 |
| BF16 | vLLM 0.29.1 + AITER | 4 | 20.54 | 48.7 | banff-cyxtera-s70-4 | 2026-10-22 |
| BF16 | vLLM 0.29.1 + AITER | 8 | 21.74 | 46.0 | banff-cyxtera-s70-4 | 2026-10-22 |
| BF16 | SGLang 0.5.10.post1.dev20260428+g144038fba | 1 | 16.48 [16.43–16.49] | 60.7 | banff-cyxtera-s70-4 | 2026-10-22 |
| BF16 | SGLang 0.5.10.post1 | 2 | 17.17 | 58.2 | banff-cyxtera-s70-4 | 2026-10-22 |
| BF16 | SGLang 0.5.10.post1 | 4 | 17.74 | 56.4 | banff-cyxtera-s70-4 | 2026-10-22 |
| BF16 | SGLang 0.5.10.post1 | 8 | 18.69 | 53.5 | banff-cyxtera-s70-4 | 2026-10-22 |
| FP8 `Qwen/Qwen3.8-27B-FP8` | vLLM 0.29.1 (default) | 1 | 28.44 [28.41–28.45] | 35.2 | banff-cyxtera-s70-4 | 2026-10-22 |
| FP8 | vLLM 0.29.1 (default) | 2 | 30.75 | 32.5 | banff-cyxtera-s70-4 | 2026-10-22 |
| FP8 | vLLM 0.29.1 (default) | 4 | 31.39 | 31.9 | banff-cyxtera-s70-4 | 2026-10-22 |
| FP8 | vLLM 0.29.1 (default) | 8 | 31.24 | 32.0 | banff-cyxtera-s70-4 | 2026-10-22 |
| FP8 | **vLLM 0.29.1 + AITER** | 1 | **15.58** [15.56–15.58] | **64.2** | banff-cyxtera-s70-4 | 2026-10-22 |
| FP8 | vLLM 0.29.1 + AITER | 2 | 16.71 | 59.9 | banff-cyxtera-s70-4 | 2026-10-22 |
| FP8 | vLLM 0.29.1 + AITER | 4 | 17.30 | 57.8 | banff-cyxtera-s70-4 | 2026-10-22 |
| FP8 | vLLM 0.29.1 + AITER | 8 | 17.15 | 58.3 | banff-cyxtera-s70-4 | 2026-10-22 |
| FP8 | SGLang 0.5.10.post1 | 1 | 35.41 [35.40–35.41] | 28.2 | banff-cyxtera-s70-4 | 2026-10-22 |
| FP8 | SGLang 0.5.10.post1 | 2 | 38.44 | 26.0 | banff-cyxtera-s70-4 | 2026-10-22 |
| FP8 | SGLang 0.5.10.post1 | 4 | 38.98 | 25.7 | banff-cyxtera-s70-4 | 2026-10-22 |
| FP8 | SGLang 0.5.10.post1 | 8 | 39.92 | 25.1 | banff-cyxtera-s70-4 | 2026-10-22 |
| MXFP4 `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` | vLLM 0.29.1 (default), emulated MXFP4 | 1 | 43.03 [42.92–43.05] | 23.2 | banff-cyxtera-s70-4 | 2026-10-22 |
| MXFP4 | vLLM 0.29.1 (default), emulated | 2 | 44.49 | 22.5 | banff-cyxtera-s70-4 | 2026-10-22 |
| MXFP4 | vLLM 0.29.1 (default), emulated | 4 | 45.14 | 22.2 | banff-cyxtera-s70-4 | 2026-10-22 |
| MXFP4 | vLLM 0.29.1 (default), emulated | 8 | 44.69 | 22.4 | banff-cyxtera-s70-4 | 2026-10-22 |
| MXFP4 | vLLM 0.29.1 + AITER, emulated | 1 | 43.49 [43.38–43.51] | 23.0 | banff-cyxtera-s70-4 | 2026-10-22 |
| MXFP4 | vLLM 0.29.1 + AITER, emulated | 2 | 45.17 | 22.1 | banff-cyxtera-s70-4 | 2026-10-22 |
| MXFP4 | vLLM 0.29.1 + AITER, emulated | 4 | 45.99 | 21.7 | banff-cyxtera-s70-4 | 2026-10-22 |
| MXFP4 | vLLM 0.29.1 + AITER, emulated | 8 | 45.33 | 22.1 | banff-cyxtera-s70-4 | 2026-10-22 |
| MXFP4 | SGLang 0.5.10.post1 | – | **fails** (see below) | – | banff-cyxtera-s70-4 | 2026-10-22 |

The best stock decode baselines on one MI300X are:

- BF16: about 16.3–16.5 ms at batch 1 (about 61 tok/s/user) with every engine. AITER does not change the BF16 result.
- FP8: 15.6 ms at batch 1 (64 tok/s/user) with vLLM + `VLLM_ROCM_USE_AITER=1`. The default vLLM FP8 path is slower than BF16 (28.4 ms). The default path selects `TritonFp8BlockScaledMMKernel`, and the server log warns: `Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at .../N=16384,K=5120,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json` (one warning per GEMM shape). With AITER, vLLM selects `AiterFp8BlockScaledMMKernel`. The SGLang FP8 result (35.4 ms) is also slower than its BF16 result. The cause in SGLang was not examined.
- MXFP4: gfx942 has no native MXFP4. vLLM runs the checkpoint by emulation: `The current platform does not support native MXFP4 computation. Using EmulationMxfp4LinearKernel, with simulated weight dequantization and activation QDQ (quantize and dequantize), with the linear layers computed in high precision.` The result (43 ms) is slower than BF16 and is not a useful baseline. SGLang fails on MXFP4:
  ```
  AssertionError: Required config file doesn't exist: /sgl-workspace/aiter/aiter/ops/triton/configs/gemm/gfx942-GEMM-AFP4WFP4.json
  ```
  The error comes from `get_gemm_config("GEMM-AFP4WFP4", M, N, K)` in the AITER Triton FP4 GEMM. It occurs in all 3 SGLang MXFP4 runs (`raw/sglang_Qwen3.8-27B-Quark-AWQ-MXFP4/*.log`).

Run-to-run variation at batch 1 is small: the spread is 0.15 ms or less (≤1%) for all 8 engine/checkpoint pairs.

### Differences between the two methods

- **vLLM** (`vllm bench serve`): TPOT is the per-request `(e2e − TTFT) / (output_tokens − 1)`. The median is taken over the requests (8×B requests at max-concurrency B). Requests are admitted continuously, so at B>1 a decode step can share its batch with a prefill of another request. This partly explains why vLLM is slower than SGLang at B=4 and B=8. The p90 TPOT and the median TTFT of each run are in `raw/vllm*/c*.json` (run `python3 summarize.py raw`).
- **SGLang** (`sglang.bench_one_batch`): this is a static batch. The value is the median latency of one decode step over 256 steps, with no prefill in the decode steps. It is the cleaner pure-decode number.

At batch 1 the two methods agree within 1%.

### Notes on the engine configuration (from the server logs)

- vLLM: `Using Triton/FLA GDN prefill kernel`, `Falling back to the Triton GDN decode path: torch.ops._C.fused_gdn_decode_post_conv_mtp is not built`, `Cannot use ROCm custom paged attention kernel, falling back to Triton implementation`. Graph mode is the default (PIECEWISE plus FULL captures, 51 sizes). `max_model_len` is 4096. The `mamba_ssm_dtype: float32` setting comes from the checkpoint config. No other hybrid-model flags were needed.
- SGLang: AITER is on by default in the image (`SGLANG_USE_AITER=1`). `Linear attention kernel backend: decode=triton, prefill=triton`, `GDN kernel dispatcher: decode=TritonGDNKernel`. The CUDA graphs captured are bs [1, 2, 4, 8, …, 87]. The overlap schedule is off (`mamba no_buffer is not compatible with overlap schedule`). bench_one_batch loads the full multimodal checkpoint. This does not change the decode step time. [INFERENCE]

## Software

| | vLLM | SGLang |
|---|---|---|
| image | `docker://rocm/vllm:rocm10.1.0_ubuntu24.04_py3.14-pytorch_2.13.0_vllm-0.29.0`. Docker Hub digest `sha256:e0ea9a10b729573ea8598bdc073bda0ea382fe1037eec7c0761b6fa7f46baaf4`; tag updated 2026-10-05. `rocm/vllm:latest` was pushed 4 s later, and it is probably the same image [INFERENCE]. Imported with enroot to `/home/AMD/sareeder/mk/images/rocm+vllm+rocm10.1.0_vllm-0.29.0.sqsh` | `/cluster/images/lmsysorg+sglang-rocm+v0.5.10.post1-rocm700-mi30x-20260428.sqsh` (sha256 prefix `2e808977ae1ea31f`) |
| engine | vllm 0.29.1.dev0+g98dff2a81.d20261002 | sglang 0.5.10.post1.dev20260428+g144038fba |
| torch / HIP | 2.13.0+rocm10.1.0 / HIP 7.16.26385 | 2.9.0a0+git7bcbafe / HIP 7.0.51831 |
| ROCm | ROCm SDK 10.1.0 (Python wheels) | 7.0.0 |
| AITER | amd-aiter 0.1.22.post2.dev0+gb4d9154d1 (off by default; on with `VLLM_ROCM_USE_AITER=1`) | v0.1.12.post1 (on by default) |

Per-run versions and environment: `raw/<run>/env.txt`.

## Commands

Jobs were started from tmux on the alola login node. Each job used one GPU:

```bash
# vLLM (default and AITER variants). Script args = checkpoint dir names under ~/mk/models
bash submit_vllm.sh Qwen3.8-27B Qwen3.8-27B-FP8                                          # runA
VARIANT=aiter VLLM_ROCM_USE_AITER=1 bash submit_vllm.sh Qwen3.8-27B Qwen3.8-27B-FP8      # runB
bash submit_vllm.sh Qwen3.8-27B-Quark-AWQ-MXFP4                                          # run3 (MXFP4 default)
VARIANT=aiter VLLM_ROCM_USE_AITER=1 bash submit_vllm.sh Qwen3.8-27B-Quark-AWQ-MXFP4      # runD
bash submit_sglang.sh                                                                    # runC (BF16, FP8)
bash submit_sglang.sh Qwen3.8-27B-Quark-AWQ-MXFP4                                        # runE
```

`submit_vllm.sh` and `submit_sglang.sh` use this command:

```bash
srun -p miopen -A miopen --constraint="GFX942&MARKHAM" --gres=gpu:1 --cpus-per-task=16 --mem=192G -t 02:00:00 \
  --container-image=<image> --container-mounts=/cluster:/cluster,/home/AMD/sareeder:/home/AMD/sareeder \
  --container-writable ...
```

`vllm_serve_bench.sh` runs these commands in the container:

```bash
vllm serve /home/AMD/sareeder/mk/models/<ckpt> --served-model-name m --tensor-parallel-size 1 \
  --max-model-len 4096 --no-enable-prefix-caching --language-model-only --port <port>
# one warmup (max-concurrency 8, 16 prompts), then for B in 1 2 4 8, then B=1 twice more:
vllm bench serve --backend vllm --model m --tokenizer <ckpt dir> --port <port> \
  --dataset-name random --random-input-len 512 --random-output-len 256 --ignore-eos \
  --max-concurrency B --num-prompts $((8*B)) --seed $RANDOM \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --save-result --result-dir <out> --result-filename cB.json
```

`sglang_one_batch.sh` runs these commands in the container:

```bash
python3 -m sglang.bench_one_batch --model-path <ckpt dir> --tp-size 1 --batch-size 1 2 4 8 \
  --input-len 512 --output-len 256 --run-name sweep --result-filename <out>/sweep.jsonl
python3 -m sglang.bench_one_batch ... --batch-size 1 ... --run-name b1_rep2   # and b1_rep3
```

bench_one_batch has no radix/prefix cache. It runs one warmup pass before the sweep. `summarize.py` skips the warmup when it reads the logs.

## Raw data (`raw/`)

| path | content |
|---|---|
| `vllm_<ckpt>/`, `vllm_aiter_<ckpt>/` | `server.log`, `env.txt`, `c{1,2,4,8}.{json,log}`, `c1_rep{2,3}.{json,log}`, `warmup.*` |
| `sglang_<ckpt>/` | `env.txt`, `sweep.log/.jsonl`, `b1_rep{2,3}.log/.jsonl` |
| `vllm_Qwen3.8-27B_run1_partial/` | Earlier valid BF16 default run: B=1,2,4,8 (16.27 / 18.02 / 20.45 / 21.63 ms). It agrees with runA within 0.7%. Its repeats were lost (see attempt 3). |
| `INVALID_portclash_vllm_Qwen3.8-27B-FP8/` | **Invalid; do not use.** The FP8 server failed with `OSError: [Errno 98] Address already in use`, and the benchmark measured the BF16 server that was still running. |
| `run*.log` | Job console logs. `enroot_import.log` is the image import log. |

`python3 summarize.py raw` prints every measurement as a table row, with the p90 TPOT and the TTFT.

## Attempt log

1. Run 1 (job 68148833) was cancelled by me. It was waiting for a download marker that the restarted downloader no longer writes. No data.
2. Run 2 (job 68148891) failed. Slurm sets `ROCR_VISIBLE_DEVICES`, and `vllm serve` fails at the `ray` import with `RuntimeError: Please use HIP_VISIBLE_DEVICES instead of ROCR_VISIBLE_DEVICES`. Fix: `vllm_serve_bench.sh` converts it to `HIP_VISIBLE_DEVICES`.
3. Run 3 (job 68148946) produced MXFP4 default (valid) and BF16 B=1..8 (valid, `run1_partial`). I replaced the script on NFS while the job ran, and bash lost the script it was reading. The BF16 server was not stopped, so the FP8 server could not bind its port, and the FP8 numbers measured the BF16 server (invalid). Fixes: a per-job port, and the readiness check now verifies that `/v1/models` serves the expected checkpoint path.
4. runA, runB, runC (jobs 68149166/67/68) and runD, runE (jobs 68149288/89) completed. They produced all the numbers above, except SGLang MXFP4, which fails on gfx942 as quoted above.
