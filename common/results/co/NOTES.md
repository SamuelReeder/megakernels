# Code-object sets for hipDNN (docs/hipdnn-route.md step 1, section 6 items 1-2)

A set is the lock that hipDNN packs: `manifest.json` plus one `<model>_M<M>_<src[:16]>.co`
per entry. Scripts: `common/scripts/export_co.py` (export), `common/scripts/h1_co.py` (H1
check), `run_set.sh` (driver on this host) and `node_model.sh` (one model on the node).

## Method

1. `export_co.py` starts vLLM on a clean checkout of the set's vLLM commit (`vllm/` first on
   `PYTHONPATH`, vLLM nightly image, `common/scripts/vrun`) with the engine arguments of the
   models' `bench/vllm_e2e.py` and `load_format="dummy"`. The DecodeSpec binds the caches,
   so `kernel.cfgs[M]` is the `Cfg` that the vLLM runtime builds for that deployment. In
   each TP rank (`llm.apply_model`) the script compiles each `Cfg` in a fresh Python
   process (`export_co.py --compile`, CPU only): the checkout's `kernels/kernel.build(cfg)`
   with `COMPILE_ONLY=1` and FlyDSL's IR dump; the lowered module before
   `gpu-module-to-binary` is printed without locations, parsed again, and serialized with
   FlyDSL's own binary pass (canonical serialization, see "FlyDSL serialization" below).
   `src` = `kernels/mkcore/key.compile_key(cfg.compiled(), <kernels dir>)` of the checkout
   (`cfg` where the checkout has no `compiled()`, rt0). Every TP rank must give the same
   bytes.
2. Each deployment is exported twice (runs `a` and `b`, two vLLM processes each);
   `logs/determinism.log` compares the manifests and every file byte for byte.
3. `h1_co.py` starts vLLM with the real weights on the same checkout and deployment. Per
   rank and width it: selects the manifest entry by the runtime `Cfg`'s `src` (a missing
   entry fails); compares the key fields; checks the file sha256; runs 16 steps from zeroed
   cache pages (seeded tokens, positions 0-15, `kernel.stage` under a forward context with
   disjoint block tables, slot mappings and state indices) with the FlyDSL launch, then
   with the code object (`hipModuleLoadData`, `hipModuleLaunchKernel`, grid `cus`, block
   `block_threads`, `lds_bytes`, the runtime's `tab` and `glob`, `nlayers` = the model's
   layer count), then FlyDSL again; and compiles the runtime `Cfg` again as step 1 does
   and requires the manifest sha256. PASS: equal output hashes, finite outputs, equal
   fields, equal sha256, on every rank. Reported only: whether the runtime launcher's own
   JIT ELF has the manifest bytes and kernel name.

## Field definitions (shared with MkEngine and the vLLM payload)

| Field | Definition |
|---|---|
| `model` | DecodeSpec name |
| `src` | as above, lowercase hex |
| `arch` | `gcnArchName` without features; equals the ELF's `amdhsa.target` |
| `cus` | `Cfg.cus` (the device's `multi_processor_count`) |
| `kv` | first attention layer: `ROCM_ATTN` (x-packed, the spec's `attention_backend()` name), `TRITON_ATTN:1` (view `[blocks, nkv, block, 2 hd]`) or `TRITON_ATTN:2` (`[blocks, 2, block, nkv hd]`); the TRITON test order of the specs' `kv_geometry` |
| `block` | KV block of the first attention layer (`kv_cache.shape[2]`); Gemma: the sliding layers' block (the full layers' block is in `src`) |
| `M`, `tp`, `rt`, `lm_head` | width, TP size, `Cfg.rt_geom` (0 where `Cfg` has none), `Cfg.lm_head` |
| `symbol` | kernel name (ELF metadata `.name`, equal to FlyDSL's `gpu.launch_func`) |
| `block_threads` | `.reqd_workgroup_size[0]` |
| `lds_bytes` | dynamic LDS for `hipModuleLaunchKernel` (`sharedMemBytes`): 0; the export fails if the launch has dynamic LDS. Static LDS is in the ELF. |
| `kernarg_bytes` | `.kernarg_segment_size`; the export asserts the args `(0, 8) (8, 8) (16, 4)`, all by value (no hidden arguments) |

## Decisions

- The `Cfg` comes from the vLLM runtime object, not from a formula. vLLM picks the KV block
  and the page strides (hybrid page unification; Gemma has two KV block sizes), and a second
  implementation of that would drift. H1 then checks, with real weights, that the runtime
  `Cfg` of each deployment selects a manifest entry.
- Qwen3.8 TP1 and TP2 use one vLLM commit, the head of `users/sareeder/qwen38-gfx942-tp2`:
  that branch holds the TP1 path too, and at the rt2 set the TP1 branch
  (`0332fdc96`) has the same `vllm/` tree as `9fc7bc243`.
- Qwen3.6: `TRITON_ATTN` (block 1056 at max_model_len 4096), the backend of the validated
  vLLM runs (`common/results/m8q/drive.sh`). Qwen3.8: `ROCM_ATTN` (vLLM's default on
  gfx942 in those runs; block 784). Gemma: `TRITON_ATTN` (block 16 sliding, 32 full).
- Gemma widths 1, 2, 4, 8: the latest Gemma branch (`users/sareeder/gemma4-moe-batch`)
  serves all four (section 4 of the design lists 1 and 2 from the older branch).
- The rt2 set exports Qwen3.8 TP1 and Qwen3.6 at max_model_len 4096 and 8192; the export
  asserts one object (one `src`, one sha256) per width for both.

## FlyDSL serialization (Qwen3.6 rt0 determinism FAIL, 2026-10-30)

First pipeline run (`f9e41ae`, in-process COMPILE_ONLY in the vLLM worker): Qwen3.6 rt0
export runs a and b differ at M = 1 and 2 (same size, `.text` differs: register
allocation, one extra instruction), and no H1 width had the runtime JIT's bytes, although
all output hashes were equal (alola `~/coexport/sets/rt0/qwen36-35b-a3b-fp8-gfx942/logs/`,
`~/mk/out/pipeline/c_rt0_q6.log`). Two causes, both outside the exporter's key:

1. Kernel name. `build()` hashes `mkcore.dev.PROBE` into the key that names the kernel;
   tracing sets `PROBE["on"]`, so a build after any trace in the process gets another
   name (Qwen3.6 M = 1: `fe4610df` first build, `91b513f2` after). vLLM builds every
   width before it traces any, so its names are the first-build names; an export after
   vLLM's warmup is not. Same code in Qwen3.8 and Gemma (`D.PROBE` in their keys).
2. Serialization. CPU-only compiles of one Qwen3.6 `Cfg` in fresh processes on the node:
   the printed IR of every stage up to `reconcile-unrealized-casts`, the LLVM IR and the
   ISA dump are identical (4 of 4), but FlyDSL's `gpu-module-to-binary` output varies
   (M = 2: two variants in 4 processes; in one process too). The same pass on the printed
   module parsed again gives one ELF: 12 of 12 (2 dumps x 3 processes x 2). [INFERENCE]
   The in-memory module carries state that printing drops (for example use-list order),
   and LLVM codegen depends on it.

Fix in the exporter: one fresh process per object (first-build name) and serialization
of the parsed module (`canonical_compile`). The vLLM runtime JIT keeps FlyDSL's own
serialization, so its bytes may differ from the manifest; H1 reports that and requires
equal outputs and an equal rebuild. Scripts and logs of the investigation: `det/`. The
rerun with `ac6c4a1` (Results) has the runtime's kernel names in every row.

## Sets

| Set | Qwen3.8 (TP1, TP2) | Qwen3.6 | Gemma4 |
|---|---|---|---|
| `rt0` | `fd8a77a5f` (rt0, rt0) | `70f57a0ae` (rt0) | `7f586cd2a` (rt0) |
| `rt2` | `9fc7bc243` (rt2, rt0) | `6657a8352` (rt2) | `5a25d336a` (rt0) |

## Results (exporter `ac6c4a1`, MI300X ctr-cx66-mi300x-31, FlyDSL 0.4.0, vLLM nightly image)

Each set has `<set>/final/` (`manifest.json` and the `.co` files, 2.2 MB per set, committed)
and `<set>/<model>/logs/` (`env.txt`, `export_{a,b}.log`, `determinism.log`, `sha_{a,b}.txt`,
`h1.log`) plus `<set>/<model>/a/{manifest.json,export.log}`. On Alola: `~/coexport/sets/<set>/`.

| Set | `final/manifest.json` sha256 | Objects | Determinism a == b | H1 (rank x width rows) |
|---|---|---|---|---|
| `rt0` | `e689d054dfca5f05…` | 16 | EQUAL (3 models) | 20/20 PASS: Qwen3.8 12 (TP1 4, TP2 2 x 4), Qwen3.6 4, Gemma4 4 |
| `rt2` | `5b83b31430d7f43e…` | 16 | EQUAL (3 models) | 28/28 PASS: Qwen3.8 16 (TP1 at 4096 and 8192, TP2 2 x 4), Qwen3.6 8 (4096 and 8192), Gemma4 4 |

rt2: Qwen3.8 TP1 and Qwen3.6 give one object per width at max_model_len 4096 and 8192
(`rt2/*/a/export.log`, OBJECT lines). The runtime JIT's own bytes equal the manifest in
every row except Qwen3.6 rt0 M = 2 and rt2 M = 1, 2, 8 (4096) / M = 1 (8192), the
FlyDSL serialization variance above; the output hashes are equal in every row.
