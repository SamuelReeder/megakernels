# Spike A: standalone code object for Qwen3.8-27B TP1 decode (disposable)

Base: megakernels 2dd8dc3, FlyDSL 0.4.0, torch 2.14.1+rocm7.2 (HIP 7.2.53211), MI300X
gfx942, 304 CUs, node banff-cyxtera-s70-4, job 68209737. Logs: `run2.log`,
`co_only.log`, `export_c.log`; numbers: `run2/results.json`.

## Result

| M | code object | symbol | FlyDSL | hipModuleLaunchKernel | same, in HIP graph | fresh process, .co only |
|---|---|---|---|---|---|---|
| 1 | `run2/qwen38_decode_m1.co` (237312 B) | `qwen38_decode_m1_8612ae82` | 4db972e0ec306d4f | 4db972e0ec306d4f | 4db972e0ec306d4f | 4db972e0ec306d4f |
| 8 | `run2/qwen38_decode_m8.co` (279424 B) | `qwen38_decode_m8_c6efe94d` | cf305425fce9de61 | cf305425fce9de61 | cf305425fce9de61 | cf305425fce9de61 |

Hashes: tests/logits_hash.py procedure (8 steps, 64 layers, seed 7, lm_head on).

Host time of one launch (median us, 200 samples, sync before each):
FlyDSL `JitFunction.__call__` 75.0 (M=1) / 70.8 (M=8); FlyDSL `flyc.compile` fast path
4.6 / 3.9; ctypes `hipModuleLaunchKernel` 4.2 / 4.0; `CUDAGraph.replay` (tok copy +
launch + pos += 1) 8.4 / 7.2. Device time per step: 8.19 / 8.20 ms (M=1, FlyDSL / hip),
11.90 / 11.88 ms (M=8). `hipModuleLoad`: 2.0 ms.

## How the code object is produced

1. Construct `mk.runner.Engine` (only defines the FlyDSL launcher).
2. Set `COMPILE_ONLY=1` (FlyDSL env var, no prefix) and call `eng.launch(tab, glob, L,
   stream=...)`. FlyDSL compiles and returns `None`; nothing runs on the GPU.
3. Read `eng.launch._last_compiled[1]._ir_text`. Parse it. Take
   `gpu.ObjectAttr(gpu.binary.objects[0]).object`. FlyDSL `max_blocks_per_cu` uses the
   same path. The bytes are a plain AMDGPU ELF (`\x7fELF`), not an offload bundle, though
   the pipeline says `format=fatbin`. No unbundling is necessary.
4. Alternative, also verified: `flyc.compile(...).export_to_c(dir, name)` writes a 670952 B
   host `.o` and `.h`. The extracted `.co` is byte-equal inside that `.o`
   (`export_c.log`). The compile is deterministic: two compiles gave the same sha256
   (0c7400e5... for M=1).

## Code object facts (`run2/m1_readelf_*.txt`)

- ELF64 DYN, EM_AMDGPU, ABI version 4 (code object v6), flags 0x54c = gfx942, xnack any,
  sramecc any. Metadata `amdhsa.version 1.2`.
- Kernel `qwen38_decode_m1_8612ae82` (`.kd` descriptor in .rodata). 233040 B .text.
- kernarg segment 20 B, align 8, **no hidden args**: offset 0 `tab` i64, offset 8 `glob`
  i64, offset 16 `nlayers` i32 (all `by_value`).
- LDS (group segment) 57408 B static, no dynamic LDS. Private segment 8 B (scratch on).
- VGPR 256, AGPR 0, SGPR 104, SGPR spills 425 (M=8: 442), VGPR spills 1. Wave64.
- `reqd_workgroup_size` 512x1x1, `uniform_work_group_size` 1.
- The host side of the FlyDSL launcher calls nothing except one `gpu.launch_func`
  (no memset, no post-load processors, no link libs).

## Interface for slice C (hipDNN engine)

- Code object: `spike/hipdnn/a/run2/qwen38_decode_m1.co`, symbol
  `qwen38_decode_m1_8612ae82`. Use `hipModuleLoad` (or `hipModuleLoadData`) and
  `hipModuleGetFunction`.
- Launch: `hipModuleLaunchKernel(fn, 304,1,1, 512,1,1, sharedMem=0, stream,
  kernelParams={&tab, &glob, &nlayers}, extra=nullptr)`. `tab` and `glob` are device
  addresses (i64), `nlayers` = 64 (i32). Grid = CU count; all 304 CTAs must be
  co-resident (grid barriers). 1 CTA per CU fits (LDS 57 KB, 256 VGPR x 8 waves).
- The code object is specialized for the full `mk.kernel.Cfg`: M, cus = 304, maxlen = 128,
  block = 208, maxblocks, KV strides, xpack = False, page strides, lm_head = True, TP1,
  chunks. The kernel name carries a crc of that key. Another maxlen or another CU count
  needs another code object [INFERENCE: maxlen and block enter the key through Cfg].
- Tables: `mk.runner.Engine.__init__` (mk/runner.py:47-187) builds both.
  `glob` = int64[18] on the device: G_EMBED, G_LM, G_FNORM (weights from `Packed`), then
  engine buffers G_ROPE, G_HB, G_SSQ, G_Y, G_Y2, G_HSSQ, G_PART, G_CNT, G_FLAGS, G_REC,
  G_TOK, G_POS, G_OUT (logits fp32 [M, vocab]); G_PEERS/G_RANK unused at TP1.
  `tab` = int64[64][18] on the device: per-layer weight and scale pointers, norms, KV
  block (attention) or conv/SSM state (GDN), and cache index pointers.
  Weights: `R.Weights(path, "mxfp4", keep_raw=True, dequant=False, layers=64)` then
  `TP.packed(ctx, W, layers=64)`. Every pointed-to tensor must stay alive while the
  tables are in use.
- Per step, on the launch stream, in order: (1) copy the M int32 tokens into `eng.tok`,
  (2) launch, (3) `eng.pos += 1` (device add). The kernel reads tok/pos through `glob`.
  Logits are read from `eng.logits` after the launch.
- State that persists between launches, all on the device, no per-step host argument:
  barrier epoch (flag buffer word at line `cus + 8`; CTA 0 advances it at the end of each
  step), GEMV counters `cnt` (the last arriver resets each to 0), KV cache, GDN
  conv/SSM state, `pos`. Initial state = zeros (Engine allocates with `torch.zeros`).
  `Engine.reset()` zeroes state/kv/pos only, not flags/cnt.
- One engine (one set of buffers) must not have two launches in flight on different
  streams. Launches on one stream in order are fine; a HIP graph capture of
  tok-copy + launch + pos += 1 replays bit-exact.
- Suggested hipDNN mapping: custom_op inputs = two int64 device tensors (`tab` [64*18],
  `glob` [18]) by UID; JSON payload: `{"symbol", "grid":304, "block":512, "nlayers":64,
  "M":1}`. Python keeps the `Engine` and does steps (1) and (3) around `Graph.execute`.
