# Common FlyDSL megakernel package (`common/mkcore/`): design

Status: design only (2026-10-29). No model code moves until the prerequisites in
"Migration plan" are true. This document is the decision record. Update it at each
milestone.

## Fixed point

The three gfx942 FlyDSL decode megakernels (`qwen3.8-27b-gfx942/`, `qwen3.6-35b-a3b-gfx942/`,
`gemma4-26b-a4b-gfx942/`) get their shared infrastructure from one package in `common/`.
The refactor stays true to these properties, even if the mechanism changes:

- F1. For every model, the kernel output bits do not change. The logits hash before and
  after each step is equal (M = 1, 2, 4, 8; `lm_head` on and off; TP1, and TP2 for Qwen3.8).
- F2. A model depends only on `common/` and on itself. `common/` depends on no model.
- F3. Every compile-time setting is a `Cfg` field or a listed debug global, and the
  compile key holds all of them plus the source of every module the kernel traces
  (model and common).
- F4. A vLLM tree gets a self-contained copy of the kernel code of each model.
- F5. Each model's hard gates stay the correctness oracle: `tests/test_decode.py`
  (M = 1, 2, 4, 8, no non-finite logits), `tests/test_determinism.py`, `tests/eval_text.py`
  (KL against BF16 within the model's limit), batch invariance.

## Scope

- In: the FlyDSL gfx942 kernels of the three models above, their `mk/` packages, their
  `scripts/`, and `scripts/export_vllm.py`.
- Out: `qwen3.5-0.8b/` (CUDA, RTX 5080) and `common/bw.cu` (they stay as they are). The
  model-specific stages (attention, GDN mixer, MoE), `reference.py`, `runner.Engine`,
  `pack.Packed` and the gate tests stay in the models in phase 1 (see "Phase 2
  candidates").

## Inventory

Copies compared: `main` at `13b25e7` (g = Gemma 4, q6 = Qwen3.6, q8 = Qwen3.8) and
`origin/users/sareeder/qwen38-tp2` at `57d6a48` (t2 = Qwen3.8 TP2). The TP2 branch forks
from `c3c07f9`; `main` has no change under `qwen3.8-27b-gfx942/mk/` or `scripts/` since
then, so every q8 / t2 difference is TP2 work.

Method (repeat it in milestone M0): for each file that exists in two or more copies,
compare whole files, then compare every top-level `def`, `class` and assignment by the
hash of its source text (Python `ast`, `git show <ref>:<path>`). Labels A, B, C name
distinct versions of one definition; `.` = absent.

Classes:

- identical: the same bytes in every copy that has the file.
- superset: one copy adds definitions or adds code behind a trace-time flag or a
  defaulted parameter. The other copies are the superset with the flag off. A merge is
  expected to give the same ISA.
- divergent: the same name has incompatible behavior, layout, numerics or `Cfg` fields.
  A merge changes the bits or the ISA of at least one model.

### M0 re-inventory (`main` at `cceba50`, 2026-10-29)

P1-P3 are merged. On `main`, q8 is the TP2 code: the q8 / t2 columns of the tables below
collapse into one copy (q8 = t2), except `kernel.py`, where the merge adds the compile
key. Same method (`git show`, `ast` source hash per top-level name), run against
`13b25e7` (g, q6) and the pre-merge TP2 head (q8). Every definition not listed here has
the label and the class of the tables below. Lines g / q6 / q8 on `main`: `dev.py` 409 /
398 / 375, `gemv.py` 440 / 534 / 421, `kernel.py` 507 / 570 / 607, `moe.py` 728 / 515 / -,
`runner.py` 231 / 203 / 204, `scripts/mkrun` 24 / 21 / 16.

| Name | g | q6 | q8 | Change since the first inventory | Class | Owner |
|---|---|---|---|---|---|---|
| `dev.grid_barrier` | A | . | A | q6 deletes it (no caller). g and q8 have no caller either. | superset (unused) | M4: not moved, dropped |
| `dev.grid_wait` | A | B | C | q6 only changes the docstring (fixed-stage-cost). g = q6 code; q8 has no `fence`. | superset (q6 / g with `fence`) | M4 |
| `gemv.Gemv` | A | B | C | q6 adds the fixed-stage-cost `cta` mode (`cta=False` default): a CTA holds whole K groups (`C = 1`), waves reduce through LDS (`cta_reduce`, `cta_groups`, `NPWC`, `lds_floats`), no global split-K hand-off. `small` becomes `not cta and U < NW`. | superset (`cta` off = g code path) | M5: `mkcore.Gemv` = q6 (with `cta`) + q8 `mx` + t2 deferred protocol (D8) |
| `kernel.compile_key` | A | A | A | Hotfix in all three (D5). Same text. Extras: g, q6 `PROFILE, D.PROBE`; q8 `PROFILE, D.PROBE, D.ARPROBE`. | identical | M2 (`mkcore.key`) |
| `kernel.kernel_name` | A | B | . | g, q6: docstring now says the cache keys on `compile_key`. q8 builds the name inline in `build` (`qwen38_decode_m{M}_{crc32(repr(cfg))}`), no function. | model-specific prefix, same format | M2 (`mkcore.key.kernel_name`) |
| `kernel.Cfg` | A | B | C | g adds `e1_cta`, `e2_rows` (MoE batch); q6 adds `cta_red`, `c_out` (fixed-stage-cost); q8 adds `tp`, `ar_dtype`, chunks (TP2). | model-specific | stays; M2 adds `fence` / `chaos` to q8, `chaos` to g |
| `moe.MoeGemv`, `routing`, `run_stage_cta`, `reduce_row`, `_lds_wait` | A | B / . | . | g MoE batch: CTA mode of the gate-up GEMV (`Cfg.e1_cta`), row-split combine (`e2_rows`). | model-specific (MoE) | phase 2 |
| `runner.Engine`, `kernel.build` | A | B | C | per-model features above | model-specific | stays |
| `scripts/mkrun` | A | B | C | q6 adds `MK_SRC` (sync folder per branch). | divergent | M3 (D12) |
| q6 `scripts/ab_bench.sh`, `fsc_*.sh` | | | | new, fixed-stage-cost measurement scripts | single model | stays |

No shared definition is without a class and an owner. The fixed-stage-cost barrier work
changes no barrier code (only the `grid_wait` docstring and the dead `grid_barrier`); its
speed comes from the `Gemv.cta` mode and the kernel schedule. The `vendor/buffer_ops.py`
copies, `stages.py`, `pack.py`, `mixer.py`, `fp4.py`, `setup_env.sh`, `isa_stats` did not
change. Script: `common/results/m0/inventory.py`; output: `common/results/m0/inventory.txt`.

### Files

| File | Lines g / q6 / q8 / t2 | Class | Differences |
|---|---|---|---|
| `mk/vendor/buffer_ops.py` | 491 / 491 / 491 / 491 | identical | FlyDSL `kernels/common/buffer_ops.py` @ `8bcefbc`. vLLM exports swap it for `aiter.ops.flydsl.kernels.buffer_ops`. |
| `mk/vendor/__init__.py`, `mk/__init__.py` | 1, 0 | identical | |
| `mk/dev.py` | 409 / 401 / 323 / 375 | superset + 3 divergent bodies | 64 top-level names; 42 identical in all four copies. Table "dev.py" below. |
| `mk/gemv.py` | 440 / 436 / 399 / 421 | divergent | `Gemv` class: g = q6, q8, t2 differ. fp8 scale layout g vs q6. Table "gemv.py". |
| `mk/fp4.py` | - / - / 102 / 102 | single model | MXFP4 host repack + device decode helpers for the `mx` GEMV kind. `unpack_ref` imports `reference`. |
| `mk/stages.py` | 568 / 383 / 420 / 603 | mostly model-specific | 48 names; 2 identical in all. Shared helpers in table "stages.py". |
| `mk/pack.py` | 180 / 190 / 123 / 180 | divergent | Tile-layout functions shared; `Packed` per model; t2 adds TP sharding. |
| `mk/kernel.py` | 425 / 521 / 531 / 589 | model-specific | Identical constants: `WAVES`, `NT`, `_STREAM`, `CNT_STRIDE`, `EPOCH_WORD_LINE`, `PROFILE`; g = q6 for `LDS_MAX`, `STAGES_PER_LAYER`. `Cfg`, `build`, `kernel_name` differ. |
| `mk/runner.py` | 230 / 203 / 183 / 204 | model-specific | `Engine` differs in all four. |
| `mk/dims.py`, `mk/reference.py` | | model-specific | Identical in all four copies: `dims.EPS` and `reference.pow2_scale` only. |
| `mk/mixer.py` | - / 582 / 580 / 579 | Qwen-family, divergent | q6 vs q8: 11 of 24 names identical (`gdn_task`, `attn_task`, `_attn_chunk`, `rms_rope` differ). |
| `mk/moe.py` | 549 / 515 / - / - | divergent | g vs q6: 7 of 15 names identical (`Route`, `slot_of`, `route_hash`, `i2f`, `f2i`, `NEG`, `NSUB`). |
| `mk/sync.py`, `mk/peer.py`, `mk/tp.py` | t2 only: 177, 157, 130 | single model | `sync.py`, `peer.py`: copies of vLLM DSv4.1 `mono/common/sync.py` and `peer_memory.py` (vLLM `0b5a1a0ae`); the export checks `sync.py` against vLLM from `POLL_MAX` on. |
| `scripts/setup_env.sh` | - / 20 / 20 / 20 | identical (header comment differs) | Gemma has none: `suite_austin.sh` runs `../qwen3.6-35b-a3b-gfx942/scripts/setup_env.sh`; `README.md`, `tests/run_suite.sh`, `tests/run_m2.sh` point to the Qwen3.8 one. |
| `scripts/mkrun` | 24 / 19 / 16 / 16 | divergent | q8: rsync to `~/mk/src`; q6: rsync to `~/mk/src36`, copies `bench/results/` back; g: tar to node `/tmp/g4src-*`, no shared sync folder, `$G4`, `$OUT`, `REVISION`. Job names `mkdev`, `mk36`, `g4dev`. |
| `scripts/isa_stats` | - / 12 / 12 / 12 | parameterized copy | Dump directory and checkpoint path differ. |
| `scripts/isa_blocks.py` | q8 only | single model | |
| `scripts/export_vllm.py` | 83 / 79 / 44 / 69 | divergent | Module list, destination, `pack.py` cut (`PACK_KEEP`), t2 `sync` rewrite, ruff rules. |
| `scripts/suite_austin.sh` | 41 / 39 / - / - | parameterized copy | Checkpoints, `REV`, work directory. |
| `scripts/finish_m5.sh`, `scripts/repro_sbatch.sh`, `scripts/stage_models.sh` | q6; g; g | single model | |
| `bench/stock/*.sh`, `summarize.py` | 1-7 lines differ | parameterized copies | Model names, job names, node list, TP. |
| `bench/bench_decode.py`, `bench/profile_decode.py`, `tools/timeline.py`, `tools/plot_results.py` | | parameterized copies | Model `main`; stage color table. |
| `tests/test_decode.py`, `test_determinism.py`, `eval_text.py`, `run_suite.sh` | | model-specific | The gates. Only `eval_text.nll` is identical in all four. |

### dev.py

| Names | g | q6 | q8 | t2 | Class |
|---|---|---|---|---|---|
| 42 helpers: `CM_DEV`, `CM_NT`, `LINE`, `traced`, `rsrc`, `gptr`, `gld32`, `gst_bf16`, `ld`, `ldf`, `st`, `opaque_*`, `uniform`, `gather*`, `_DPP`, `_dpp`, `_reduce`, `xor_sum`, `xor_max`, `absf`, `fma`, `fma4`, `rsq`, `exp2`, `log2`, `logf1p_exp`, `expf`, `sigmoid`, `silu`, `bf_lo`, `bf_hi`, `pow2_scale`, `fp8_pack4`, `quant8`, `memrealtime`, `poll_ge`, `grid_barrier`, `PROBE`, `probe`, `_probe` | A | A | A | A | identical |
| `_dpp_rows`, `_wave_uniform`, `wave_max`, `wave_sum`, `_asm`, `FENCE` | A | A | . | . | superset (q6, copied to g) |
| `CHAOS`, `chaos_sleep` | . | A | . | . | superset (q6; g removed it) |
| `tanh`, `gelu_tanh`, `bf16_rne`, `pack_bf2` | A | . | . | . | superset (g) |
| `CM_SYS`, `bf16_pair`, `bf16r`, `drain_cta`, `st_stamps` | . | . | . | A | superset (t2) |
| `TRACE` (rank `Dims`, all-reduce config), `ARPROBE` | . | . | . | A | model state in a shared module (t2) |
| `grid_arrive` | A | B | C | C | divergent body: q6 = `fence` (`buffer_wbl2 sc1`) + `CHAOS`; g = `fence` only; q8 = neither |
| `grid_wait` | A | A | B | B | divergent body: `fence` (`buffer_inv sc1`) or not |
| `atomic_add_dev` | A | B | C | C | divergent body: `FENCE` (g, q6) + `CHAOS` (q6) or neither |

The three divergent bodies are superset-compatible: the extra code sits behind
`const_expr(fence)`, `const_expr(CHAOS["on"])` or a trace-time `if FENCE["on"]`, and every
default is off.

### gemv.py

| Name | g | q6 | q8 | t2 | Notes |
|---|---|---|---|---|---|
| `PAD_TILES` | A | A | A | A | 16 |
| `mfma_fp8`, `_imin` | A | A | . | . | |
| `fp8_load`, `fp8_compute` | A | B | . | . | **Divergent layout, same kind name `fp8`.** q6: 2 fp32 scales per unit (rows 0-7, rows 8-15; 128 x 128 block scales). g: 16 fp32 scales per unit (one per output row; per-channel scales). |
| `Gemv` | A | A | B | C | g = q6: kinds `fp8`, `bf16`; `small` mode (U < NW: router, `in_proj` b/a); `np_` select for empty waves. q8: kind `mx` (MXFP4 via `fp4.device_helpers()`), no `small`, no select. t2 = q8 + deferred epilogue protocol (`push`, `finish`, `la_finish`, `la_reduce(push_only)`). |
| `H4`, `CHUNKS` | . | . | A | A | q8 module-level K-chunk table (Qwen3.6 I8 rejects such knobs). t2 moves it into `Cfg.chunks` via `chunks(tp, override)`. |

### stages.py (shared helpers only)

| Name | g | q6 | q8 | t2 | Class |
|---|---|---|---|---|---|
| `zero_xs_tail` | A | A | A | A | identical; reads the module global `NT` that `build` sets |
| `store_quant8`, `store_quant8_at` | A | A | B | B | superset: g and q6 factor the body into `store_quant8_at`; q8 inlines the same code |
| `epi_raw` | A | A | B | B | superset: g and q6 add `off=0` |
| `rstd_from_ssq` | A | B | C | C | **divergent numerics**: g and q6 sum with DPP `wave_sum`; q8 with the `xor_sum` butterfly. g also changes the output slot and takes `EPS` from `dims`. |
| `prologue_scaled` | A | B | B | B | divergent: q6 adds the per-head gated norm; g adds probes |
| `embed_stage` | A | B | B | C | divergent: g scales and rounds; t2 exchanges ranks |
| `prologue_norm`, `EpiRes`, `_epi_res`, `EpiConv`, `prologue_norm_bf16`, `final_norm_stage`, `prologue_act`, `epi_act` | . | A | A or B | B or C | Qwen-family only; read `H`, `VOCAB`, `CONV_DIM` from `dims`; t2 adds the all-reduce |

Other numerically divergent helpers with one role: bf16 rounding is the integer
round-to-nearest-even `bf16_rne` / `pack_bf2` in `g/mk/dev.py`, and `.to(fx.BFloat16)` in
`mixer.bf16r` / `mixer.pack_bf2` (q6, q8, t2) and `t2/mk/dev.py` `bf16_pair` / `bf16r`. The
two `pack_bf2` functions have the same name and different code.

### pack.py (host side)

| Name | g | q6 | q8 | t2 | Class |
|---|---|---|---|---|---|
| `unit_order` | A | A | B | B | superset: g and q6 add leading dims (experts) and `nrest` |
| `gu_order` | A | A | B | B | superset: g and q6 accept batched input |
| `pack_bf16` | A | A | B | B | superset: g and q6 cast to bf16 first (a no-op for bf16 input) |
| `fp8_tiles` | A | B | . | . | divergent: scale layout of the GEMV kind |
| `row_scales` (q6), `gu_order_rows` (g), `shard_*` (t2) | | | | | single model |

## The hardest divergences

1. **One kind name, two fp8 layouts.** `Gemv(kind="fp8")` loads 2 scales per unit in
   Qwen3.6 and 16 in Gemma. `pack.fp8_tiles` and the vLLM `runtime/weights.py` of each
   model follow the layout. A shared `Gemv` cannot keep one name for both. Resolution: two
   kinds, `fp8_b128` (Qwen3.6) and `fp8_row` (Gemma), with packers `fp8_tiles_b128` and
   `fp8_tiles_row`. The kind string is model source, so the compile key covers it.
2. **Barriers, debug globals and the compile key.** `grid_arrive`, `grid_wait` and
   `atomic_add_dev` have three versions, and the fixed-stage-cost branch changes the
   Qwen3.6 barrier code now. The flags live in module dicts (`FENCE`, `CHAOS`, `PROBE`,
   `PROFILE`, t2 `TRACE`, `ARPROBE`) that `build` sets. FlyDSL 0.4.0 does not put these
   modules or the kernel name in its disk-cache key (see D5), so a helper in another
   directory is invisible to the cache. Resolution: superset barrier code with
   `const_expr` flags from common-owned `Cfg` fields; one `configure(cfg)` entry; an
   explicit compile key (D5); a re-inventory after fixed-stage-cost merges (M0).
3. **Same role, different numerics.** `rstd_from_ssq` reduces with DPP (`wave_sum`) in
   Qwen3.6 and Gemma and with `xor_sum` in Qwen3.8; the order of the float adds differs,
   so the rstd bits differ. bf16 rounding has an integer and a convert version with the
   same name `pack_bf2`. One shared version changes the bits of at least one model (F1).
   Resolution: do not merge these. Keep `rstd_from_ssq` in each model. Keep both bf16
   paths in `mkcore.dev` under different names (D7).

Next in difficulty: the `Gemv` `small` mode adds a runtime select (`np_`) to every GEMV
of Qwen3.6 and Gemma, which Qwen3.8 does not have (D8); the t2 deferred epilogue protocol
is a superset (`getattr(epilogue, "deferred", False)` is a trace-time constant).

## Design

### Layout

```
common/
  DESIGN.md              this file
  bw.cu                  unchanged (CUDA roofline)
  mkcore/                FlyDSL gfx942 megakernel infrastructure; imports no model
    __init__.py          SRC_HASH: sha256 of the package's .py files
    key.py               compile key and kernel name (D5), Cfg field checks (D4)
    dev.py               device helpers: memory ops, DPP/xor reductions, math, quant8,
                         grid barrier, device atomics, probes, PROFILE; grid constants
                         (WAVES, NT, CNT_STRIDE, EPOCH_WORD_LINE, LDS_MAX)
    gemv.py              Gemv: unit stream, split-K, fixed-order last arriver, epilogue
                         protocol (call, pre, deferred push/finish); kinds bf16, mx,
                         fp8_b128, fp8_row
    fp4.py               MXFP4 repack and device decode (the mx kind), without unpack_ref
    pack.py              host tile layouts: unit_order, gu_order, pack_bf16,
                         fp8_tiles_b128, fp8_tiles_row, row_scales, PAD_TILES
    stagelib.py          zero_xs_tail, store_quant8, store_quant8_at, epi_raw
    vendor/buffer_ops.py FlyDSL buffer_ops @ 8bcefbc (vLLM: aiter's copy)
  scripts/
    mkrun                ship <model>/ + common/ to the held allocation, run a command
    setup_env.sh         the pinned venv (torch 2.14.1+rocm7.2, flydsl 0.4.0)
    isa_stats            compile with IR dumps, print VGPR / spill counts
    isa_dump             normalized final ISA + sha256 per kernel (oracle, M1)
    gates.sh             all models: logits hash, ISA compare, hard gates (M1)
    export_lib.py        vLLM export helpers shared by each model's export_vllm.py
    suite_austin.sh      second-node suite, parameterized by model directory
  tests/                 CPU tests of mkcore host code against recorded goldens
  results/<milestone>/   logs of gates.sh runs for common changes
```

Stays in each model: `mk/dims.py`, `mk/kernel.py` (`Cfg`, `build`, the stage schedule,
`CNT_REGION`), `mk/stages.py` (model prologues and epilogues, `rstd_from_ssq`),
`mk/mixer.py`, `mk/moe.py`, `mk/attention.py`, `mk/reference.py`, `mk/runner.py`,
`mk/pack.py` (`Packed` and model-specific packers), Qwen3.8 `mk/sync.py`, `mk/peer.py`,
`mk/tp.py`, the gate tests, `bench/`, `tools/`, `scripts/export_vllm.py` (model rules),
model-only scripts.

Rules:

- R-dep: `mkcore` imports no model module. A model imports `mkcore` and itself only.
  `gates.sh` checks both with a grep.
- R-share: a definition moves to `mkcore` when two or more models use it and one version
  serves all of them with an unchanged ISA (D8 lists the one exception). The existing
  `dev.py` copies come from one ancestor, so `mkcore.dev` is their union (D7).
- R-state: model state never lives in `mkcore` modules. t2 `TRACE` (rank `Dims`) moves to
  the Qwen3.8 package. `ARPROBE`, `drain_cta`, `st_stamps` stay with Qwen3.8 TP until a
  second model uses TP.

### API boundaries

- Imports. Model code: `from mkcore import dev as D`, `from mkcore.gemv import Gemv`.
  Inside `mkcore`: relative imports only (`from . import dev as D`). The export depends on
  this (D6).
- `Gemv(name, N, K, M, cus, waves, C, kind)`: unchanged constructor. `C` (K chunks) is
  always an argument; chunk tables (`c_*` fields, t2 `Cfg.chunks`) stay in the model.
- Per-unit kind functions are public: model `moe.py` (Qwen3.6, Gemma) imports
  `fp8_load` and `fp8_compute` by name for `MoeGemv`. They become `fp8_b128_load`,
  `fp8_b128_compute`, `fp8_row_load`, `fp8_row_compute` (plus `mfma_fp8`), and each
  `moe.py` imports the pair of its model's kind.
- Epilogue protocol (duck-typed, as today): `__call__(cg, tots)`; optional `pre(cg)`;
  optional `deferred = True` with `push(cg, tots)` and `finish(cg)`.
- `NT`: a constant in `mkcore.dev` (512 in all models). No module may assign it.

### Cfg ownership (D4)

Each model keeps one flat frozen `Cfg` dataclass. `mkcore.key.COMMON_FIELDS` lists the
fields that `mkcore` code reads:

| Field | Type | Read by |
|---|---|---|
| `M` | int | `Gemv` geometry, `key.kernel_name` |
| `cus` | int | `Gemv`, `grid_wait` |
| `fence` | bool | `grid_arrive`, `grid_wait`, `atomic_add_dev` |
| `chaos` | bool | `grid_arrive`, `atomic_add_dev` |

`mkcore.key.check_cfg(cfg)` asserts that each field exists with its type. Qwen3.8 adds
`fence` and `chaos` (default `False`); Gemma adds `chaos`. Every other field belongs to the
model. Debug globals that `mkcore` reads: `dev.PROBE`, `dev.PROFILE`, `dev.FENCE`,
`dev.CHAOS`. `mkcore.dev.configure(cfg)` sets `FENCE` and `CHAOS` from `cfg` at the start
of `build`; nothing else writes them.

### Compile key and kernel name (D5)

Facts from the flydsl 0.4.0 wheel (`flydsl/compiler/jit_function.py`):

- The disk-cache key of a `@flyc.jit` function holds its source, the source of helpers
  that resolve by direct global name and lie in the same directory as the root file
  (`_is_user_function`, lines 416-421; `_collect_dependency_sources`, 551-565), and the
  closure values of type int, float, bool, str, None, tuple, Enum (`_CLOSURE_SCALARS`,
  line 151; line 490).
- Helpers reached as module attributes (`D.ld`, `S.epi_raw`) are not followed. A
  `mkcore` file is in another directory, so it is never followed. Measured: see "Compile
  cache test" (cases A, O).
- Values read from a non-scalar closure object (`cfg.lm_head`, `cfg.fence`, `SP[...]`,
  `GEO`) are not in the key. Measured: case C.
- The `@flyc.kernel(name=...)` argument is not in the key: no kernel body references
  `name` (checked in all three `mk/kernel.py`). Measured: case N. The comments in each
  `kernel_name` ("FlyDSL's disk cache keys on the name") are wrong.
- The cache is on by default (`flydsl/utils/env.py` line 296). The default directory is
  `$HOME/.flydsl/cache`, the same shared-home directory for the venv and for the vLLM
  nightly image. Every `mkrun` sets `FLYDSL_RUNTIME_ENABLE_CACHE=0`; the vLLM trees set
  nothing.

Design:

1. `mkcore.SRC_HASH` = sha256 over the sorted relative paths and bytes of the package's
   `.py` files, computed at import.
2. `mkcore.key.compile_key(cfg, pkg_dir, *extra)` = sha256 of `repr((cfg, *extra))`, then
   `SRC_HASH`, then `source_hash(pkg_dir)` (sorted relative paths and bytes of the `.py`
   files under the model's kernel package: `mk/` standalone, `kernels/` in vLLM, which
   then holds `kernels/mkcore/` too). `extra` = the debug globals that change the traced
   code: `PROFILE`, `D.PROBE` (all three models), `D.ARPROBE` (Qwen3.8). `dev.FENCE` and
   `dev.CHAOS` need no entry: `build` sets them from `cfg.fence` / `cfg.chaos`, which
   `repr(cfg)` holds. (Fixed at M1: this is the hotfix key plus `SRC_HASH`.)
3. `mkcore.key.kernel_name(prefix, cfg, key)` = `f"{prefix}_m{cfg.M}_{crc32(key):08x}"`
   (the current format; the crc is now of the key, not of `repr(cfg)`).
4. `build` binds `key` (a `str`) in the closure of the `@flyc.jit` launcher with the
   statement `key` as the first line of the launcher body, so FlyDSL 0.4.0 puts it into
   the cache key. Measured: case X (a helper edit plus a new key string compiles a new
   binary with the correct result).

Rejected: rely on `FLYDSL_EXTRA_SOURCE_DIRS` / `jit_function.EXTRA_SOURCE_DIRS` alone. It
hashes source files only, not `Cfg`; it is a process-wide setting with an `lru_cache`.
Rejected: a cache directory per source hash alone. It covers source edits, not `Cfg`
fields read through `cfg.*` (case C, S2). Rejected: keep the current name-only hash. The
name is not in the key (case N).

### How a model finds `mkcore` (D3)

`mk/__init__.py` (empty today) puts `<repo>/common` at the front of `sys.path` and asserts
that `mkcore.__file__` lies under that directory. Every entry point already imports `mk`
first, so no script changes. `common/scripts/mkrun` and the suite scripts ship
`<model>/` and `common/` together and keep them siblings.

Rejected: `pip install -e common` into the shared venv `~/mk/venv`. Several source trees
(worktrees, node `/tmp` copies) share that venv, so an editable install points every
tree to one `mkcore`, and a kernel silently builds with another tree's code. Rejected:
`PYTHONPATH` set by `mkrun` only. A direct `python tests/...` run then fails or picks up a
stale copy.

### Versioning (D2)

Models track `common/` at the same commit (monorepo HEAD). There is one `mkcore` at a
time.

- A change under `common/mkcore/` merges only with a `common/scripts/gates.sh all` run on
  that commit: every model's logits hash equals its recorded baseline (or the change is a
  logged, gated numerics change of named models), every model's hard gates pass, and the
  logs are committed under `common/results/<date>-<topic>/`.
- A change that must alter one model's bits goes behind a `Cfg` field whose default keeps
  the old code, or it is gated as a numerics change of that model (all its gates, a new
  logits baseline, a DESIGN log entry).
- Each model README names the megakernels commit of its last full gate run.
- vLLM trees pin by copy: the export writes the megakernels commit and `SRC_HASH` into the
  header of `kernels/mkcore/__init__.py`.

Rejected: each model pins a `common` commit. Two pinned versions need two importable
copies of `mkcore` (copies again, or a submodule per model).

### vLLM export (D6)

What is on the vLLM branches (`SamuelReeder/vllm`): `vllm/models/common/specialized_decode/`
holds `spec.py` (`Feature`, `DecodeUnsupported`, `Step`, `DecodeImpl`, `DecodeSpec`),
`dispatch.py` (`SpecializedDecode`, `DecodeWarmup`), `guards.py` (`tp_group`, `tp_agree`,
`select_decode_spec`): model-agnostic runtime protocol, used by DSv4.1, Kimi K3 and others.
`dispatch.py` already differs across our branches (gemma4 `4f72009af` = qwen36
`0ad8a7c56`; qwen38 `0b5a1a0ae`; qwen38-tp2 `6220b7379`). AMD kernel code shared across
models does not live there: vLLM commit `2b4eaee84` (2026-10-16) moved DSv4.1's
`mono/common` into `common/specialized_decode/amd/mono/`, and `0d7f3e718` (2026-10-18)
moved it back into the model directory. `kimi_k3` and `deepseek_v4` each keep their own
`mono/common/sync.py`. Our kernels live in `vllm/models/{qwen3_5,qwen3_5_moe,gemma4}/amd/
specialized_decode/kernels/` (Qwen3.8, Qwen3.6, Gemma).

Design:

1. Each model's export copies `common/mkcore/` (without `vendor/`) into
   `kernels/mkcore/` of that model, so each vLLM model directory stays self-contained
   (F4).
2. Rewrites: in model modules, `from mkcore` becomes `from .mkcore`; in `mkcore/dev.py`,
   `from .vendor import buffer_ops as bo` becomes `from aiter.ops.flydsl.kernels import
   buffer_ops as bo` (the current rule). `mkcore` internals need no rewrite (relative).
3. `common/scripts/export_lib.py` holds the shared steps: copy `mkcore`, rewrite imports,
   write the SPDX header and the commit / `SRC_HASH` line, assert that no excluded module
   is imported, run ruff (`I`, format, full check) from the vLLM root. Each model's
   `scripts/export_vllm.py` keeps its module list, destination, `pack.py` cut and the t2
   `sync` rule, and calls `export_lib`.
4. `mkcore.pack` has no `reference` import, so it ships whole; the model `PACK_KEEP` cut
   shrinks to model packers. vLLM `runtime/weights.py` of each model imports the layout
   functions from `..kernels.mkcore.pack`; change it in the same vLLM commit as the
   re-export.
5. In vLLM, the compile key hashes `kernels/mkcore/` and `kernels/` (D5).

Rejected: one shared copy under `vllm/models/common/specialized_decode/` (or `.../amd/`).
It couples three vLLM branches that merge separately, every `mkcore` change then needs
all three models re-validated in vLLM at once, and the vLLM authors reverted the same move
for DSv4.1 two days after making it.

### Test strategy

Oracles, cheapest first:

1. CPU goldens (`common/tests/`, no GPU): for fixed seeds, the byte hash of every
   `mkcore.pack` output; for every model's `Cfg.specs()` at M = 1..8 and `cus` = 304, the
   host-side `Gemv` geometry (`U`, `NW`, `XR`, `XSR`, `slots`, `wave_lo`, `cta_range`,
   `nseg_host`, `part_floats`). Record the goldens from the per-model code at M1; compare
   after every move.
2. ISA identity (`common/scripts/isa_dump`): build each model at M = 1 and 8, `lm_head`
   on and off (TP2 fp32 and bf16 all-reduce for Qwen3.8) with `FLYDSL_DUMP_IR=1`; take
   `22_final_isa.s`, replace the kernel symbol with a fixed name, drop comment lines; the
   sha256 must equal the baseline.
3. Logits hash (new `tests/logits_hash.py` per model, about 40 lines, added at M1 before
   any move): fixed seed and real-text prompt, prefill with the model's reference, N
   steps with fixed tokens; sha256 of the fp32 logits (and of the bf16 hidden states with
   `lm_head=False`) at M = 1, 2, 4, 8. Gemma prefills past the 1024 window. Qwen3.8 TP2:
   per rank, fp32 and bf16 all-reduce.
4. The hard gates of each model (F5), unchanged: `tests/run_suite.sh` subsets.
5. Export: run every model's `export_vllm.py` into a scratch vLLM worktree; ruff passes;
   in the vLLM container the exported `kernels` package imports and `build`s the M = 1
   kernel with the same ISA hash as the standalone build. Full `bench/vllm_e2e.py` at M8.

`common/scripts/gates.sh [all | model ...]` runs 2-4 for each model on one held
allocation (one GPU; two GPUs for the Qwen3.8 TP2 part), writes logs to
`common/results/<run>/<model>/`, records node and job, and prints one PASS / FAIL line per
model and oracle. Oracle 1 runs on the login host.

The gate tests are not refactored in phase 1: the oracle does not change in the same
change as the code it checks.

## Compile cache test (2026-10-29)

Run on banff-cyxtera-s70-4, job 68206732 (one MI300X), sources at `1c46059`. Scripts and
logs: `common/results/cache/` (`cache_min.py`, `cache_model.py`, `run_node.sh`, `logs/`).
FlyDSL 0.4.0 in both environments: the hipdnn image with `~/mk/venv`
(`logs/env_venv.txt`) and the vLLM nightly image `vllm-rocm-nightly-00b6c4c5.sqsh`
(`logs/env_vllm.txt`). Neither sets a `FLYDSL_*` variable; both default to the cache on
in `/home/AMD/sareeder/.flydsl/cache`.

Minimal kernel (`cache_min.py`; `logs/cache_min_venv.txt`, `logs/cache_min_vllm.txt`).
Each case: base build with the cache on in a fresh directory, then the changed build
with the cache on in the same directory, then the changed build with the cache off.

| Case | Change | Cache on | Cache off | Result |
|---|---|---|---|---|
| A | helper in another file, same directory, called as `H.f` (as `D.silu`) | 11111 | 11112 | **stale binary** |
| O | helper in another directory, called as `O.f` (as a future `mkcore`) | 11111 | 11121 | **stale binary** |
| D | helper in another file, imported by name | 11211 | 11211 | new binary |
| K | int closure constant of the kernel | 12111 | 12111 | new binary |
| C | field of a frozen dataclass in the closure (as `cfg.lm_head`) | 11111 | 21111 | **stale binary** |
| N | `@flyc.kernel(name=...)` only | 11111 | 11111 | no new cache entry: the name is not in the key |
| X | case A plus a new `str` constant referenced in the launcher (D5 key) | 11112 | 11112 | new binary (venv run only) |

Qwen3.6 kernel, M = 1, 4 layers, 4 fixed-token steps, sha256 of the outputs
(`cache_model.py`; `logs/cache_model_qwen36_m1_s1.txt`, `logs/cache_model_qwen36_m1_s2s3.txt`):

| Scenario | Base (cache on) | Changed, same cache | Changed, cache off | Result |
|---|---|---|---|---|
| S1: `mk/dev.py` `silu` x (1 + 2^-10) | `88292b7377a71a1a` | `88292b7377a71a1a` | `d9fc0ac21e9623a0` | **old binary used; the edit has no effect** |
| S2: `lm_head` True -> False | `88292b7377a71a1a` | no result: the step did not finish in 600 s (1800 s in the first attempt), process killed | `49e205655f16e5dc` | **old binary used; the kernel hangs** |
| S3: `MK_CHUNKS=lm=8` (`Cfg.c_lm`, the packed LM head follows) | `88292b7377a71a1a` | `6003cb5daf85deb1` (finite) | `98e0150e11a66632` | **old binary used; wrong logits, no error** |

Conditions for a stale binary. All must hold:

1. The cache is on (`FLYDSL_RUNTIME_ENABLE_CACHE` unset or 1) and the cache directory
   already has an entry from an earlier process. The default directory
   `$HOME/.flydsl/cache` is on the shared home: every node, the venv and the vLLM image
   use the same one. It holds 45 `launch_*` entries from 2026-10-22 (origin unknown).
2. The text of the `@flyc.jit` launcher and the `@flyc.kernel` function is the same,
   with the same scalar closure values (`M`, `CUS` and the other ints), the same
   directly named same-directory helpers, the same argument types, and the same FlyDSL.
3. The change is only in code or values the key does not see: a helper called as a module
   attribute (`D.*`, `S.*`, `X.*`, `MO.*`, `A.*`: almost every device function of all
   three kernels), a `Cfg` field read through `cfg.*`, `SP[...]` or `GEO` (in all three
   models: `lm_head`, `fence`, the GEMV chunks, the KV block size and strides,
   `maxblocks`, `kv_xpack`, `conv_s0`, `ssm_s0`), or the kernel name.

Exposure (no model code or vLLM branch was changed):

- Standalone: every `scripts/mkrun`, `isa_stats` and `suite_austin.sh`, Gemma's
  `repro_sbatch.sh`, and the TP2 `scratch/m3/task.sh` set `FLYDSL_RUNTIME_ENABLE_CACHE=0`.
  `tests/run_suite.sh` (all three), Qwen3.8 `bench/repro.sh`, and direct `python
  tests/...` runs from the READMEs do not, so they can use a stale binary.
- vLLM branches: the runtime builds through `KK.build(cfg)` with the cache on. In the vLLM
  nightly image `import aiter` sets `FLYDSL_RUNTIME_CACHE_DIR` to aiter's bundled
  `aiter/jit/flydsl_cache` when the variable is unset (`aiter/__init__.py` lines 94-97), so
  the kernels go into the container's site-packages, not `$HOME/.flydsl/cache` (seen in
  the hotfix runs). A fresh container starts without our entries; a container that lives
  on (a named or writable container, a long-running server image) keeps them. In such a
  container, a re-export that changes only helper modules (`dev`, `gemv`, `stages`,
  `mixer`, `moe`, `attention`) keeps the old binary after a restart, and a restart with
  another KV geometry (block size, attention backend layout, `max_model_len`) can load a
  binary built for the old geometry. [INFERENCE for the geometry fields: the same
  mechanism as case C and S2 / S3; not run in vLLM.] Our own vLLM measurement scripts
  (`~/mk/m6run.sh`, `~/mk/g4m6run.sh`) set `FLYDSL_RUNTIME_ENABLE_CACHE=0`, so the
  published vLLM numbers are not affected.

Mitigation:

1. Now, without code changes: run with `FLYDSL_RUNTIME_ENABLE_CACHE=0` (each process
   compiles once per `Cfg`), or point `FLYDSL_RUNTIME_CACHE_DIR` at an empty directory
   for each deployment, and delete the cache directory after each re-export
   (`$HOME/.flydsl/cache`; in a vLLM container, aiter's `aiter/jit/flydsl_cache`).
2. Fix (D5): `build` puts a `str` key in the launcher closure that holds `repr(cfg)`, the
   debug globals and the source hashes of the model package (and of `mkcore` after M2).
   Case X shows that a `str` closure value gives a new binary. A cache directory per
   source hash alone is not enough: it misses `Cfg` fields (case C, S2, S3). Applied
   ahead of M2 as the hotfix below.
3. `$HOME/.flydsl/cache` on Alola was cleared after the hotfix tests (46 entries, 2.2 MB,
   2026-10-29).

buffer_ops (D13.5): aiter's `aiter/ops/flydsl/kernels/buffer_ops.py` in the vLLM nightly
image is not the vendored FlyDSL `8bcefbc` copy (`logs/buffer_ops.diff`, 417 changed lines).
Apart from docstrings, the three functions the kernels call differ:
`create_buffer_resource_from_addr`, `buffer_load`, `buffer_store`. aiter adds a
compatibility layer for the cache-policy operand (enum attribute or i32). Whether the
exported kernels give the same ISA is open; oracle 5 answers it at M3.

### Hotfix: compile key ahead of M2 (M2-prefix, 2026-10-29)

Main pulled D5 forward. Each model gets the same few lines in `mk/kernel.py` (copies until
`mkcore` exists; M2 moves them into `mkcore/key.py`):

- `compile_key(cfg, *extra)`: sha256 over `repr((cfg, *extra))` and every `.py` file of
  the kernel package (`pathlib.Path(__file__).parent.rglob("*.py")`, sorted relative
  paths, each path and its bytes).
- `build`: `key = compile_key(cfg)` (Qwen3.6 and Gemma pass `PROFILE, D.PROBE`, as their
  `kernel_name` does); the first line of the `@flyc.jit` launcher is `key  # noqa: B018`.
- The kernel code and the kernel name do not change.

Where:

| Tree | Branch | Commit | Owner |
|---|---|---|---|
| megakernels `qwen3.8-27b-gfx942/` | `users/sareeder/flydsl-cache-key` (from `main` `13b25e7`) | `57b6c6e`, docstring rewrap `9134175` | this task |
| vLLM `qwen3_5/.../kernels/kernel.py` (Qwen3.8) | `users/sareeder/qwen38-gfx942-specialized-decode` | `cfc4c8c5b` (re-export of `9134175`) | this task |
| vLLM `qwen3_5_moe/.../kernels/kernel.py` (Qwen3.6) | `users/sareeder/qwen36-gfx942-specialized-decode` | `88e48072d` (hand edit) | this task |
| megakernels `qwen3.6-35b-a3b-gfx942/`, `gemma4-26b-a4b-gfx942/` | their open branches | - | FixedStageCost, GemmaMoeBatch |

Evidence, Qwen3.8 TP1 (banff-cyxtera-s70-4, job 68206986; `cache_model_q38.py`;
`logs/hotfix/q38_branch.txt`, `logs/hotfix/q38_main.txt`). 4 layers, M = 1, 4 steps, cache
on; the reference value is the changed build with the cache off:

| Scenario | `main` (no key), changed + same cache | Hotfix, changed + same cache | Changed, cache off |
|---|---|---|---|
| S0 second identical build | - | `27756f3b`, no new entry: the cache still hits | - |
| S1 `dev.silu` x (1 + 2^-10) | `27756f3b`: **stale** | `d7d8323c`, new entry | `d7d8323c` |
| S2 KV block 208 -> 64 | new entry (the block reaches a scalar closure value) | new entry | same hash `27756f3b` (the block size does not change the math) |
| S3 `gemv.CHUNKS` lm 8 -> 4 | no result in 1500 s: **stale, hangs** | `787f83f1`, new entry | `787f83f1` |

Logits before and after (64 layers, 8 fixed-token steps, cache off): M = 1 `4db972e0ec306d4f`
and M = 8 `cf305425fce9de61` on both `main` and the hotfix. Gates on the hotfix
(`logs/hotfix/gate_*.txt`, cache off as `mkrun` runs them): `test_decode` M = 1 and 8
(64 layers, 16 steps) PASS, no non-finite logits; `test_determinism` M = 1 and 8 (64 steps)
PASS. The gates ran on `57b6c6e`; `9134175` changes only the docstring.

vLLM smoke (same job, vLLM nightly image, branch `vllm` package on `PYTHONPATH`, FlyDSL
cache on, which here means aiter's directory): `bench/vllm_e2e.py` kernel on, greedy, 8
prompts (`logs/hotfix/v38_*`, `v36_*`, `b38_*`, `b36_*`). Each run compiled four new
entries (widths 1, 2, 4, 8) and passed.

- Qwen3.6 (`88e48072d`, `MK_ATTN=TRITON_ATTN`, 160 tokens): tokens and logprobs equal to the
  recorded kernel-on run (`qwen3.6-35b-a3b-gfx942/bench/results/m6/m4/raw/on_TRITON_ATTN_r1.json`)
  and to a same-day run of the pre-hotfix head `0ad8a7c56`.
- Qwen3.8 (`cfc4c8c5b`, default backend, 128 tokens): two hotfix runs, both with the
  same tokens as the recorded run `~/mk/vout/on.json` of 2026-10-27
  (`logs/hotfix/ref_v38_on_20261027.json`). The first run (fresh container) also has the
  same logprobs. The second run (the same container, after other vLLM runs) has the
  same tokens and 149 logprobs that differ by up to 0.33. The same-day run of the
  pre-hotfix head `0b5a1a0ae` in that container has the same logprob drift and flips one
  near-tie token (prompt 3, token 1, logprob -1.46 vs -1.59). The kernel code is the
  same in all three runs, so vLLM's own path is not bitwise reproducible from a fresh to
  a used container on Qwen3.8; this is not a hotfix effect.
- TPOT (one run each, not a speed claim): hotfix 8.19 / 9.13 / 10.23 / 12.65 ms and
  pre-hotfix 8.44 / 9.12 / 10.26 / 12.77 ms at batch 1 / 2 / 4 / 8 (Qwen3.8); hotfix 3.76 /
  4.43 / 5.69 / 7.58 ms and pre-hotfix 3.79 / 4.42 / 5.71 / 7.53 ms (Qwen3.6). Both are
  slower than the 2026-10-27 numbers by the same amount, so the cause is the day and node
  load, not the hotfix.

Not covered by this hotfix: the Gemma and Qwen3.8 TP2 vLLM branches, and the standalone
Qwen3.6 and Gemma directories (their owners apply the same lines).

## Milestone log

### M1: oracles and baselines (2026-10-29, `main` `cceba50` + M0, sources `1a64394`)

Oracles added: `<model>/tests/logits_hash.py` (oracle 3), `common/scripts/isa_dump`
(oracle 2), `common/scripts/gates.sh` + `gates_job.sh` (sbatch wrapper), CPU goldens
`common/tests/goldens.py` / `goldens.txt` (oracle 1), cache test `common/tests/cache_test.py`.
Jobs on banff-cyxtera-s70-4, cache off: Qwen3.8 68207960 (with the cache test), Qwen3.6
68207986, Gemma 68207987, Qwen3.8 TP2 68207988 (2 GPUs). Logs: `common/results/m1/`
(`summary.txt`, `<part>/hashes.txt`, `<part>/isa_renorm.txt`, gate logs). `RUNS=2`: the
logits hash ran twice per model; both runs are equal (`hash_run2=run1 PASS` for all four).

First check: `main`'s Qwen3.8 `kernel.py` (TP2 merge with the compile-key conflict
resolved) gives the pre-merge TP1 hashes: M = 1 `4db972e0ec306d4f`, M = 8 `cf305425fce9de61`
(`q8/hashes.txt`; recorded in "Hotfix" above). The merge changed no bits.

Logits hash inputs (deviation from "Test strategy": no reference prefill; the kernel
writes its own KV, which covers the same code, and a reference prefill adds minutes per
M): tokens `randint(0, 200000)`, CUDA generator seed 7, maxlen = steps + 64 (Qwen3.8: 128,
as the hotfix). Qwen3.8 has no Engine path for `lm_head=False` (vLLM only): that build is
not in its logits or ISA oracle. Gemma runs 1040 steps, past the 1024-position window.

| Model | Steps, layers | lm | M = 1 | M = 2 | M = 4 | M = 8 |
|---|---|---|---|---|---|---|
| Qwen3.8 TP1 | 8, 64 | on | `4db972e0ec306d4f` | `c33baa53118edc27` | `412eec22e6dbe80b` | `cf305425fce9de61` |
| Qwen3.8 TP2 fp32 (gathered) | 8, 64 | on | `1e8ad0ba3f15077a` | `ec3d5ed217c5c0c6` | `bf44cbcb806eaa8d` | `32d7b498e590659b` |
| Qwen3.8 TP2 bf16 (gathered) | 8, 64 | on | `9489275e99206138` | `ffaf9971d594de69` | `d4e6389ed1a2448e` | `cc9e3b01889e1a4d` |
| Qwen3.6 | 16, 40 | on | `d5604309e6d480d9` | `b7ca25b1820fd3c6` | `450529bb14a9acd7` | `cc940f489fa21d01` |
| Qwen3.6 | 16, 40 | off | `416bf85f760cfce9` | `330e6a953f46de98` | `9c2ab9c777ba6878` | `7197290b3d310bc8` |
| Gemma 4 | 1040, 30 | on | `b24507ff6293f91a` | `8036b849244aaac4` | `82c285a00cfcfb8b` | `fa2d2856bf90a8e4` |
| Gemma 4 | 1040, 30 | off | `00e033b221ce103f` | `bfc9706774984895` | `50ef2de028a1eaf8` | `7122e51d62dd5c6a` |

TP2 per-rank vocab-slice hashes are in `q8tp2/hashes.txt`.

ISA (normalized `22_final_isa.s`, sha256 16; M = 1 / 8): Qwen3.8 TP1 `7a34625c72591084` /
`32bbd1d67f3e014e`; TP2 fp32 `52e87a3c06736983` / `18040b8e5ecfa3c2`, bf16
`b077d7ae35c914dc` / `78d9c7682e922348` (both ranks equal); Qwen3.6 on `8f9ca5691d1e1b07` /
`987b6be70b787442`, off `3088f771e4013fcf` / `25df478d656b90b7`; Gemma on `c911f702945b840a`
/ `10f9098f4a402300`, off `f74fdd1dd40e690b` / `8616a1d230537c9c`. The first `isa_dump`
missed the symbol inside `.L<name>.num_vgpr` labels (`\b` before the name); fixed in
`isa_dump`, and `isa_renorm.txt` re-applies the fixed rule to the kept dumps (the `isa.txt`
files of the run hold the first rule).

Hard gates (`summary.txt`): all PASS for Qwen3.8 TP1 (`test_decode` M = 1, 2, 4, 8;
determinism M = 1, 8; `eval_text`), Qwen3.6 (plus `test_chaos` M = 1, 8), Gemma
(`test_decode` M = 1, 2, 4, 8 at CTX 512 and M = 8 at CTX 2048; determinism 1100 steps M = 1,
8; `eval_text` 1536; `test_attention`), Qwen3.8 TP2 (determinism fp32 / bf16 M = 1, 8; ranks
bitwise equal in every `test_decode` step; `eval_text` fp32). `summary.txt` lists
`q8tp2 test_decode_fp32_m1 FAIL`: that is the old `test_decode` criterion (c1), report only
since TP2 milestone 4 (`qwen3.8-27b-gfx942/docs/DESIGN-TP2.md`, "Gates from milestone 4
on"); it fails at M = 1 with 1 - worst cos 0.000897, the value recorded there. Not a red
baseline. `gates.sh` now gates TP2 `test_decode` on "ranks bitwise equal" only.

CPU goldens: 298 lines (pack outputs per model, `Gemv` / `MoeGemv` geometry at M = 1..8,
cus 304, Qwen3.8 at TP1 and TP2), recorded on the login node, re-run equal.

Cache test (`cache_test.txt`, Qwen3.6, fresh cache directory, cache on, all PASS):
(a) two `Cfg` that differ in a field read as `cfg.*`: `lm_head` and `cta_red` each give a
new entry and the cache-off result. (b) a comment line in `mk/dev.py` gives a new entry
(the key hashes the bytes: a comment edit costs one recompile) and equal bits. (c) the
`str` key in the launcher closure: same sources and `Cfg` twice hit the cache (no new
entry); a `silu` edit gives a new entry with the cache-off result.

## Decisions

- D1. Package `common/mkcore/` (FlyDSL gfx942 only), scripts in `common/scripts/`. Evidence:
  the inventory; `docs/adding-a-model.md` ("Shared code goes in `common/` only once a
  second model needs it"). Rejected: package name `mk` (every model package is `mk`).
- D2. Track HEAD with all-model gates per common change (see "Versioning").
- D3. `mk/__init__.py` path insert (see "How a model finds `mkcore`").
- D4. Flat model `Cfg` plus `COMMON_FIELDS` check. Rejected: a `CommonCfg` base dataclass.
  A defaulted base field (`fence`) cannot precede the model's non-default fields without
  `kw_only`, which changes every positional `Cfg(...)` call.
- D5. Explicit compile key in the launcher closure (see "Compile key").
- D6. vLLM: per-model `kernels/mkcore/` copy (see "vLLM export").
- D7. `mkcore.dev` is the union of the four `dev.py` copies. Collisions get names that
  state the method: `bf16_rne` / `pack_bf2_rne` (Gemma, integer RNE) and `bf16r` /
  `pack_bf2` (convert, from Qwen `mixer.py` and t2). Barrier and atomic functions take the
  q6 superset form. Evidence: an uncalled helper emits no code; the extra barrier code
  sits behind `const_expr` flags. Rejected: model-local `dev` extension modules (two
  modules for one concern, and the export must merge them).
- D8. `Gemv` takes the g / q6 class plus t2's deferred protocol plus q8's `mx` kind. The
  `np_` empty-wave select runs only when `const_expr(self.small)`. Evidence: with U >= NW
  every wave of a GEMV owns at least one unit (`wave_lo` = gw U // NW), so the select is
  a no-op there. Cost: Qwen3.6 and Gemma lose one select per piece in non-small GEMVs, so
  their ISA changes (tier B, D9). Rejected: the select in every GEMV (adds runtime work to
  Qwen3.8 and changes its ISA).
- D9. Acceptance tiers per model and step. Tier A (default): ISA hash equal. Tier B: ISA
  differs, logits hashes equal, every hard gate passes, and `bench/bench_decode.py` TPOT
  (median of 3, same node and day as a re-measured baseline) is within 1 % at M = 1, 2,
  4, 8; log it in this file. Anything else is a numerics change and is not part of this
  refactor. (Approved by Main, 2026-10-29.)
- D10. Keep `sync.py`, `peer.py`, `tp.py`, `TRACE`, `ARPROBE` in Qwen3.8 until a second
  model uses TP. Evidence: one user; the vLLM file is the source of truth and the export
  checks the copy against it.
- D11. `rstd_from_ssq` and `prologue_scaled` stay per model (divergence 3).
- D12. `common/scripts/mkrun` uses the Gemma method (tar to node `/tmp/<model>-src-*`, no
  shared sync folder, `REVISION` file) and copies `<model>/bench/results/` back like
  Qwen3.6. Usage: `common/scripts/mkrun <model-dir> [-j JOB] 'cmd'`; job name from
  `$MK_JOB` or `-j`. The per-model `mkrun`, `setup_env.sh`, `isa_stats` are deleted (no
  wrappers).
- D13. Decisions by Main on the open questions (2026-10-29):
  1. The cache test runs now, before any migration (results in "Compile cache test").
  2. Tier B is accepted under D9 (logits hash, all gates, TPOT within 1 %, median of 3).
     D8 stays as written.
  3. No shorter checks: every milestone runs the full gates of every model.
  4. `rstd_from_ssq` copies stay separate (D11). This refactor changes no bits.
  5. M0 compares the vendored `buffer_ops.py` with aiter's copy in the nightly vLLM image
     and records the diff (done early, see "Compile cache test").
  6. The Qwen3.8 TP2 vLLM kernels keep importing the mailbox from DSv4.1
     (`deepseek_v41/amd/mono/common/sync.py`, the source of truth). `export_vllm.py`
     already fails when the copy in `mk/sync.py` differs.
  Migration milestones M0+ start only after the Gemma MoE, fixed-stage-cost and TP2
  branches merge; Main starts them.

## Rejected alternatives

- Keep the copies (status quo, Gemma's "copy and diverge", D2 in its DESIGN.md). Cost
  measured above: about 2.5 k lines in the second and third copies of `dev.py`, `gemv.py`
  and `buffer_ops.py` on `main` alone (724 + 835 + 982);
  fixes do not propagate (Qwen3.8 has no `fence`, Gemma dropped `CHAOS`); Gemma
  already reaches into the Qwen directories for `setup_env.sh`, so "models never depend on
  each other" is false today. Copying stays right for a new model's first milestones;
  after that, the model moves its shared parts to `mkcore`.
- Git submodule for `common`. The code is in the same repository and the same commits;
  a submodule adds a second history, a pin per model and a manual bump, and the cluster
  scripts ship trees by tar.
- pip package (`mkcore` on an index or `pip install -e`). Same pin problem as a
  submodule, plus the shared-venv aliasing of D3. vLLM must not depend on a megakernels
  package: it gets copies (F4).
- Shared code in `vllm/models/common/specialized_decode/`: see D6.
- A generic megakernel framework (stage graph, declarative schedule). Not needed to remove
  the duplication; it changes every kernel and so every ISA.

## Invariants of the refactor

- I-bits: per model, logits hash after each milestone equals the M1 baseline (F1).
- I-isa: tier A or a logged tier B per model and milestone (D9).
- I-gates: every hard gate of every model passes after each milestone (F5).
- I-key: editing any `mkcore` or model `mk/` file, or changing any `Cfg` field, gives a new
  compile key; checked by the M1 cache test, re-run at M2 and M8 (F3).
- I-dep: R-dep holds (`gates.sh` grep).
- I-export: each vLLM export builds an ISA-identical kernel (oracle 5) (F4).
- I-host: CPU goldens of `mkcore.pack` and `Gemv` geometry unchanged.

## Stop conditions

Stop with `STOP: <violated invariant>` and evidence when:

- a move breaks I-bits for a model and the function cannot stay per model;
- a merged definition needs a per-model switch with more than two variants, or an
  `if model == ...` test (keep it per model instead, and record it);
- the M1 cache test shows that no closure statement makes FlyDSL recompile after a
  `mkcore` edit (I-key): no code moves until it is solved;
- a gate fails on the baseline commit before any change (the baseline is red);
- a branch that changes a shared file (any file in the inventory) is open when a
  milestone starts: wait for it to merge, then redo M0 for that file;
- no gfx942 allocation for 24 hours.

## Migration plan

Prerequisites, in order: (P1) the Gemma MoE batch branch is merged; (P2) the
fixed-stage-cost branch (Qwen3.6 barrier code, `mkrun`) is merged; (P3)
`users/sareeder/qwen38-tp2` is merged into `main` (no conflict expected in
`qwen3.8-27b-gfx942/mk/`: `main` has not changed it since the fork). Each milestone is one
branch and one PR. After each milestone run `common/scripts/gates.sh all` on
banff-cyxtera-s70-4 (record node and job) and commit the logs.

| # | Outcome | Out of scope | Acceptance evidence |
|---|---|---|---|
| M0 | Re-run the inventory on merged `main`; update the tables in this file; classify every new difference (expect the fixed-stage-cost barrier change). | Any code change. | Updated tables committed; no shared definition without a class and an owner milestone. |
| M1 | Oracles and baselines, no kernel change: `tests/logits_hash.py` per model; `common/scripts/isa_dump`, `gates.sh`; CPU goldens; the FlyDSL cache test (fresh cache dir, cache on: (a) two `Cfg` that differ in one non-scalar-derived field, (b) a comment edit in `dev.py`, (c) a `str` key in the launcher closure). | Moving code. | Baseline hashes (2 runs equal) and ISA hashes per model under `common/results/m1/`; cache-test log that answers (a)-(c); D5 statement fixed. |
| M2 | `common/mkcore/__init__.py` (`SRC_HASH`), `key.py`; each model's `mk/__init__.py` path insert; the per-model `compile_key` copies of the hotfix (see "Hotfix: compile key ahead of M2") move into `mkcore/key.py` and also hash `mkcore`; Qwen3.8 and Gemma get the missing `fence` / `chaos` fields. | Moving helpers. | Tier A for every model (only the symbol name changes); logits hashes equal; cache test passes (I-key); full gates. |
| M3 | Move `vendor/buffer_ops.py`; `common/scripts/{mkrun, setup_env.sh, isa_stats, suite_austin.sh}`; delete the per-model copies; fix Gemma's `../qwen3.6...` and Qwen3.8 references; `export_lib.py` and the three `export_vllm.py`. | `dev.py`. | Tier A; hashes equal; full gates via the new `mkrun`; export into scratch vLLM trees diff only in import lines; oracle 5. |
| M4 | `mkcore/dev.py` (D7); `configure(cfg)`; Qwen `mixer.bf16r` / `mixer.pack_bf2` and t2 `bf16_pair` move in; Gemma `pack_bf2` becomes `pack_bf2_rne`; t2 `TRACE` and TP probes move into Qwen3.8 modules; Qwen3.8 `sync.py` imports from `mkcore` (its vLLM check starts at `POLL_MAX`, so the import lines may change); grid constants and `PROFILE` move from `kernel.py`. | `gemv.py`. | Tier A for all; hashes equal; full gates; oracle 5. |
| M5 | `mkcore/gemv.py` (D8) and `mkcore/fp4.py` (`unpack_ref` moves to Qwen3.8 `reference.py`); kinds `fp8_b128`, `fp8_row`, `mx`, `bf16`. | `pack.py`. | Tier A for Qwen3.8 TP1 / TP2; tier B logged for Qwen3.6 and Gemma (D8); hashes equal; geometry goldens equal; full gates; bench median of 3 for tier B. |
| M6 | `mkcore/pack.py`; model `pack.py` keeps `Packed` and model packers; export `PACK_KEEP` shrinks; vLLM `runtime/weights.py` imports change on each vLLM branch. | Device code. | Pack goldens byte-equal; hashes equal; full gates; vLLM weight loading smoke (oracle 5 with real weights at M = 1). |
| M7 | `mkcore/stagelib.py` (`zero_xs_tail`, `store_quant8`, `store_quant8_at`, `epi_raw`). | `rstd_from_ssq`, `prologue_scaled` (D11). | Tier A for all; hashes equal; full gates. |
| M8 | Re-export all three models (and TP2) into their vLLM branches; `bench/vllm_e2e.py` per model. | New features. | vLLM e2e logs and same-day TPOT against the pre-refactor vLLM numbers, under each model's `bench/results/`. |
| M9 | Docs: top-level `README.md` ("each model depends only on `common/`"), `docs/adding-a-model.md` (start from `mkcore`, then copy what diverges), each model's DESIGN "Reuse" section, this file's status. | Code. | Docs PR merged. |

Each milestone repeats the cache test when it changes `key.py` or the export.

## Phase 2 candidates (no decision yet)

Re-inventory after M9. Candidates, each with the R-share rule:

- Qwen-family stages shared by Qwen3.6 and Qwen3.8: `prologue_norm`, `EpiRes`,
  `EpiConv`, `prologue_norm_bf16`, `final_norm_stage`, and `mixer.py` (GDN and attention:
  11 of 24 definitions identical). They read `H`, `VOCAB`, `CONV_DIM` from `dims`; a move
  makes these parameters.
- MoE routing helpers shared by Gemma and Qwen3.6 (`Route`, `slot_of`, `route_hash`,
  `i2f`, `f2i`). Wait for the Gemma MoE work to settle.
- Harness: `bench/stock/*`, `bench/bench_decode.py`, `bench/profile_decode.py`,
  `tools/timeline.py` (stage color table per model), `eval_text.nll`.
- TP: `sync.py`, `peer.py` into `mkcore/tp/` when a second model uses TP (D10).

## Open questions

1. Answered by the compile cache test: the FlyDSL 0.4.0 disk cache can serve a stale
   binary (see "Compile cache test"). The hotfix covers Qwen3.8 TP1 and the Qwen3.8 and
   Qwen3.6 vLLM branches. Open: the Gemma and TP2 vLLM branches.
2. to 6.: decided, see D13.

## Disposable

- `tests/logits_hash.py` baselines and `common/results/m1/` stay until M9, then the
  logits hash stays as a gate of `gates.sh` with the M9 baseline.
- The inventory procedure is a command sequence in this file, not a tool. Make it a
  script only if M0 needs it more than once.
