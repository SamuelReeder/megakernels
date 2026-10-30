# Code-object sets for hipDNN (docs/hipdnn-route.md step 1, section 6 items 1-2)

A set is the lock that hipDNN packs: `manifest.json` plus one `<model>_M<M>_<src[:16]>.co`
per entry. Scripts: `common/scripts/export_co.py` (export), `common/scripts/h1_co.py` (H1
check), `run_set.sh` (driver on this host) and `node_model.sh` (one model on the node).

## Method

1. `export_co.py` starts vLLM on a clean checkout of the set's vLLM commit (`vllm/` first on
   `PYTHONPATH`, vLLM nightly image, `common/scripts/vrun`) with the engine arguments of the
   models' `bench/vllm_e2e.py` and `load_format="dummy"`. The DecodeSpec binds the caches,
   so `kernel.cfgs[M]` is the `Cfg` that the vLLM runtime builds for that deployment. In
   each TP rank (`llm.apply_model`) the script builds the checkout's
   `kernels/kernel.build(cfg)`, calls it with `COMPILE_ONLY=1`, and takes the ELF from the
   lowered module (`gpu.binary`, spike slice A). `src` =
   `kernels/mkcore/key.compile_key(cfg.compiled(), <kernels dir>)` of the checkout (`cfg`
   where the checkout has no `compiled()`, rt0). Every TP rank must give the same bytes.
2. Each deployment is exported twice (runs `a` and `b`, two vLLM processes each);
   `logs/determinism.log` compares the manifests and every file byte for byte.
3. `h1_co.py` starts vLLM with the real weights on the same checkout and deployment. Per
   rank and width it: selects the manifest entry by the runtime `Cfg`'s `src` (a missing
   entry fails); compares the key fields; checks the file sha256; runs 16 steps from zeroed
   cache pages (seeded tokens, positions 0-15, `kernel.stage` under a forward context with
   disjoint block tables, slot mappings and state indices) with the FlyDSL launch, then
   with the code object (`hipModuleLoadData`, `hipModuleLaunchKernel`, grid `cus`, block
   `block_threads`, `lds_bytes`, the runtime's `tab` and `glob`, `nlayers` = the model's
   layer count), then FlyDSL again; and checks that the ELF that the runtime launcher
   compiled has the manifest sha256. PASS: equal output hashes, finite outputs, equal
   fields, equal sha256, on every rank.

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

## Sets

| Set | Qwen3.8 (TP1, TP2) | Qwen3.6 | Gemma4 |
|---|---|---|---|
| `rt0` | `fd8a77a5f` (rt0, rt0) | `70f57a0ae` (rt0) | `7f586cd2a` (rt0) |
| `rt2` | `9fc7bc243` (rt2, rt0) | `6657a8352` (rt2) | `5a25d336a` (rt0) |
