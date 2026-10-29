# Spike: serve the megakernels through hipDNN (2026-10-29)

Question: can a FlyDSL megakernel ship as a precompiled code object in hipDNN and run
through a hipDNN custom op, standalone and inside vLLM, with no change to the result?

Answer: yes, for Qwen3.8-27B TP1 on MI300X (gfx942). Every oracle passed. The code is
disposable. The details are in `a/`, `engine/` and `c/` (each has NOTES.md and logs).

## Path that works

```
vLLM DecodeSpec (VLLM_MK_HIPDNN=1)
  -> hipdnn_frontend Graph.custom_op("megakernel.qwen3_8_27b_decode",
       inputs [tab, glob] (INT64 tensors), data = JSON {model, M, nlayers, geom, grid, block})
  -> hip-kernel-provider ingestor pack "megakernel" (ENABLE_MEGAKERNEL_ENGINE, default OFF)
       UMD (native, GRAPH scope): one CUSTOM_OP node, id prefix "megakernel."
       UMD (KERNEL scope): model + M + geom match the KMD fields of one UKD
       UDD (native): hipModuleLaunchKernel(fn, 304, 512, args {tab, glob, nlayers}, handle stream)
       UKD source: hsaco code object from FlyDSL, packed unchanged into kpack by hkp_pack.py
```

## Results

| Check | Result | Log |
|---|---|---|
| Code object out of FlyDSL | `COMPILE_ONLY=1`, take the ELF from the `gpu.binary` op. Plain AMDGPU ELF, deterministic sha256. 237 KB (M=1), 279 KB (M=8). | `a/run2/` |
| Kernel ABI | 20 B kernarg: tab i64, glob i64, nlayers i32. No hidden args, no dynamic LDS. | `a/run2/` |
| Logits hash, standalone | M=1 `4db972e0ec306d4f`, M=8 `cf305425fce9de61` for FlyDSL, hipModuleLaunchKernel, hipDNN execute and hipDNN in `torch.cuda.graph` | `c/logs/c1_final.log` |
| 96 greedy steps | Bitwise equal to the FlyDSL launch at M=1 and M=8 | `c/logs/c1_final.log` |
| Binary selection | KERNEL-scope match on M and a geometry key. An unknown key declines: "No engine configurations available". | `engine/logs/g1.log`, `c/NOTES.md` |
| Host cost per step | FlyDSL JIT call 70 us; hipDNN execute 4.4 / 7.0 us (M=1 / M=8); FlyDSL compiled fast path 3.9 us. TPOT equal within noise (8.34-8.46 ms M=1, 12.13-12.24 ms M=8). | `c/logs/` |
| vLLM e2e | Widths 1, 2, 4, 8 run through hipDNN, eager and under vLLM graph capture. Greedy outputs equal to the FlyDSL path, 8 prompts x 128 tokens, 3 of 3 runs. TPOT within run-to-run spread. | `c/logs/c4_*.log` |
| Build | hipDNN + hip-kernel-provider + Python bindings, ingestor ON: about 40-50 s. Also builds inside the vLLM image. | `engine/logs/b0.log`, `c/logs/c0_vllmimg.log` |

## Problems found

1. **Geometry is part of the binary.** The code object fixes max_model_len, KV block size,
   KV layout and CU count. The spike keys binaries by `geom = crc32(repr(Cfg with M=0))`.
   The vLLM run needed a second set (maxlen 4096, block 784, x-packed KV, no LM head).
   A product needs a fixed list of supported geometries, with fallback to vLLM's own path,
   or geometry passed at run time (a kernel change).
2. **Two hipDNN backends in one process.** torch 2.13+rocm10 loads its own
   `libhipdnn_backend.so`. The spike extension then fails with
   `undefined symbol: hipdnnGetEngineNameById_ext`. The spike uses `LD_PRELOAD`. A product
   must build against the hipDNN that torch ships, or use a separate library name.
3. **ABI of the dev image.** The hipDNN image (Ubuntu 24.04) build does not load in the vLLM
   image (Ubuntu 22.04): `GLIBC_2.38` not found. Build inside the target image.
4. **Feature flag needs two places.** `HKP_DESCRIPTOR_FAMILIES` gets
   `megakernel=ENABLE_MEGAKERNEL_ENGINE`, and the native symbols sit behind
   `#ifdef HIPKERNELPROVIDER_ENABLE_MEGAKERNEL_ENGINE`. Without the first, the loader sees the
   descriptors but refuses the engine.
5. **Custom-op friction.** compute_data_type is required but unused. Inputs must be typed
   tensors with dims. Python `set_data` takes one int per byte. Each distinct payload is
   a separate catalog entry (fine: one per width and geometry).
6. **Not covered:** TP2 (no peer-memory API for plugins), co-residency check in the matcher
   (the spike trusts grid = CU count), concurrent execute from two threads, kpack size for a
   full geometry list, the frontend wheel (`python -m build` is missing from the image).

## What hipDNN does not remove

The vLLM side still needs the per-model DecodeSpec: the step checks, cache binding, weight
repacking, and the tab/glob tables. hipDNN replaces only the FlyDSL compile and launch.
`hipdnn_vllm` does not exist yet.

## Branches (all local, not pushed upstream)

- megakernels `users/sareeder/hipdnn-spike`: this directory.
- rocm-libraries `users/sareeder/megakernel-spike` (102d947a342, from develop 470e78bcfb0);
  diff copy in `c/rocm-libraries-spike-c.diff`.
- vLLM `users/sareeder/qwen38-hipdnn-spike` (6e0656478); diff copy in `c/vllm-spike.diff`.
