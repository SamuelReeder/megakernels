# Spike C: Qwen3.8 decode megakernel served through hipDNN, standalone and in vLLM (disposable)

Node banff-cyxtera-s70-4 (MI300X, gfx942, 304 CUs), job 68209893 (released). All runs are in the vLLM
nightly image `vllm-rocm-nightly-00b6c4c5.sqsh` (torch 2.13.0+rocm10.0.0, FlyDSL 0.4.0). Logs: `logs/`.

## Commits

| repo | branch (local) | commit |
|---|---|---|
| rocm-libraries | users/sareeder/megakernel-spike | 102d947a342 (geometry key, vLLM code objects) on 665e9800a8f; diff `rocm-libraries-spike-c.diff` (no .co) |
| vllm | users/sareeder/qwen38-hipdnn-spike (worktree `worktrees/vllm/qwen38-hipdnn-spike`) | 2473830c3, 6e0656478 on cfc4c8c5b; diff `vllm-spike.diff` |

## 1. Environment

- (a) Build hipDNN in the vLLM image: works (`logs/c0_vllmimg.log`). The image has cmake 3.31.10, ninja
  1.13.2, gcc 11.4, and ROCm 10.0 as the `_rocm_sdk_devel` wheel (clang 23, HIP 7.15). Use
  `-DROCM_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel` (`lib/llvm/bin` fits the
  toolchain file) and `-DHIPKERNELPROVIDER_KPACK_PYTHON_DIR=<copy of the hipDNN image's
  /opt/rocm-kpack/python>`. Configure 5 s, build 49 s on 32 cores. Script `scripts/c0_build.sh`.
- Problem: torch 2.13+rocm10 loads `_rocm_sdk_libraries/lib/libhipdnn_backend.so` at `import torch`. Then
  the spike frontend binds to that older backend: `undefined symbol: hipdnnGetEngineNameById_ext`
  (`logs/c1_import_error_no_preload.log`). Fix: `LD_PRELOAD=<spike>/lib/libhipdnn_backend.so`. Both
  backends stay mapped; one `libamdhip64` (the `_rocm_sdk_devel` one) serves torch and hipDNN (`HIP libs`
  line in `logs/c1_final.log`).
- (b) The hipDNN-image build (Ubuntu 24.04) in the vLLM image (Ubuntu 22.04) with LD_LIBRARY_PATH fails:
  `GLIBC_2.38` and `GLIBCXX_3.4.32` not found (`logs/c1b_hipdnnimg_in_vllm.log`).

## 2. Oracle (standalone, `scripts/c1_engine_hipdnn.py`, `logs/c1_final.log`)

`mk.runner.Engine` builds tab/glob/state; per step: tok copy, `Graph.execute` on torch's current
stream, `pos += 1`. Payload `{"model":"qwen3.8-27b-flydsl","M":M,"nlayers":64,"geom":"e5cc820d",...}`.

| M | FlyDSL | hipDNN execute | hipDNN in torch.cuda.graph | 96 greedy steps vs FlyDSL |
|---|---|---|---|---|
| 1 | 4db972e0ec306d4f | 4db972e0ec306d4f | 4db972e0ec306d4f | bitwise equal every step |
| 8 | cf305425fce9de61 | cf305425fce9de61 | cf305425fce9de61 | bitwise equal every step |

## 3. Overhead and TPOT (standalone, `logs/c1_final.log`)

Host us per launch: median of 200, sync before each. TPOT: 64 greedy steps, host reads each token;
median of 7 rounds, paths interleaved. Async: same loop, one sync at the end.

| path | host us M1 | host us M8 | TPOT ms M1 | TPOT ms M8 | async ms M1 | async ms M8 |
|---|---|---|---|---|---|---|
| FlyDSL JitFunction call | 69.7 | 67.1 | 8.46 | 12.24 | 8.40 | 12.25 |
| FlyDSL flyc.compile fast path | 3.9 | 3.9 | 8.37 | 12.16 | 8.40 | 12.27 |
| hipModuleLaunchKernel (ctypes) | 7.2 | 13.1 (p10 7.1) | 8.45 | 12.14 | 8.40 | 12.44 |
| hipDNN execute | 4.4 | 7.0 (p10 4.4) | 8.34 | 12.13 | 8.40 | 12.45 |
| hipDNN execute in graph, replay | 6.3 | 4.5 | 8.45 | 12.16 | 8.42 | 12.27 |

The device time (8.4 / 12.2 ms) hides every host cost; the path differences are inside the round min/max.

## 4. vLLM (`VLLM_MK_HIPDNN=1`, `scripts/c4_vllm.sh`, `bench/vllm_e2e.py` kernel on, TP1)

- Export: `VLLM_MK_HIPDNN_EXPORT=DIR` compiles each width with FlyDSL COMPILE_ONLY after `bind_caches`
  and writes the ELF from `gpu.binary` plus `manifest.jsonl` (`export/`). vLLM geometry:
  `Cfg(cus=304, maxlen=4096, block=784, maxblocks=6, kv_s0=1605632, kv_s1=200704, kv_s2=8,
  kv_sv=802816, kv_xpack=True, conv_s0=ssm_s0=3211264, lm_head=False)`, key `e00e8bc9`, M = 1, 2, 4, 8.
  `scripts/gen_kdp.py` writes `megakernel_vllm.kdp.json` (4 UKDs). Incremental rebuild 15 s.
- Every width ran through hipDNN, eager and under vLLM's graph capture (`SPIKE hipDNN execute: first at
  width W, capturing True/False` in `logs/c4_hipdnn_r*.log`).
- Greedy tokens: 8/8 prompts x 128 tokens equal to the FlyDSL runs in all 3 hipDNN runs. Logprobs:
  prompt 1 has two bit patterns from token 1 (max |diff| 0.145) that occur in FlyDSL runs too
  (fly_export ~ hipdnn ~ hipdnn_r1; fly_r1 ~ fly_r2 ~ hipdnn_r2); every hipDNN run equals a FlyDSL run bitwise.
- TPOT ms (B1 / B2 / B4 / B8): FlyDSL 8.34-8.37 / 9.04-9.10 / 9.61-10.55 / 12.60-14.04; hipDNN
  8.38-8.48 / 9.10-9.56 / 9.69-10.30 / 12.80-12.84 (3 runs each). B1 hipDNN is 0.01-0.14 ms higher in
  each pair; graph replay does not call hipDNN, so [INFERENCE] noise; not resolved.

## Geometry key design

- Key = `crc32(repr(Cfg with M=0))` (8 hex): cus, maxlen, KV block, block-table columns, KV strides,
  x-pack, conv/SSM slot strides, lm_head. Payload field `geom`; KMD field `geom` (string, mandatory for
  every UKD); KERNEL-scope match on model + M + geom. The test geometry key is `e5cc820d`
  (`logs/geom_test_cfg.log`); stand-in UKDs use `standin`. An unknown key fails at
  `create_execution_plans: No engine configurations available for the graph.` (`logs/b_standin_geom.log`).
- The model id must also carry the kernel source version: the symbol of megakernels hashes the compile
  key (Cfg + sources), not repr(Cfg) alone.
- Fixed-geometry list (what a shipped pack would hold instead of a per-deployment export):

```
{"model": "qwen3.8-27b-flydsl-vllm@<kernel src hash>", "arch": "gfx942", "M": [1, 2, 4, 8],
 "geometries": [
   {"cus": 304, "attn": "ROCM_ATTN", "kv_xpack": true,  "block": 784, "maxlen": 4096},
   {"cus": 304, "attn": "ROCM_ATTN", "kv_xpack": true,  "block": 784, "maxlen": 32768},
   {"cus": 304, "attn": "TRITON_ATTN", "kv_xpack": false, "block": 784, "maxlen": 4096}]}
```

  Only the first entry was exported and run. [INFERENCE] The other two entries (block 784 under
  TRITON_ATTN, maxlen 32768) are illustrations, not observed geometries. Strides and slot strides follow
  from block, KV dtype and the model, so cus x attention layout x maxlen x M spans the list (3 x 4 = 12
  objects of 107-140 KB, about 1.5 MB). maxlen enters only through maxblocks and the RoPE table length;
  [INFERENCE] a runtime maxlen argument in the kernel would remove that axis.

## Reproduce

1. `git archive` rocm-libraries (CMakeLists.txt CMakePresets.json cmake shared/ctest projects/hipdnn
   dnn-providers/cmake dnn-providers/hip-kernel-provider) to node `/tmp/mkC/src`; `TAG=vllmimg c0_build.sh`.
2. `c1_run.sh c1_engine_hipdnn.py OUT MODEL ../spike/hipdnn/a/run2 1,8 8 96`.
3. `c4_vllm.sh NAME [VLLM_MK_HIPDNN_EXPORT=/work/export | VLLM_MK_HIPDNN=1]`; `c4_compare.py A.json B.json`.
