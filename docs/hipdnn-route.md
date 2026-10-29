# Shipping the megakernels through hipDNN to vLLM: design

Status: design open, 2026-10-29. This document records decisions and evidence. It is not a plan.
Each decision has a recommendation. A recommendation is not a decision until the owner signs it
off. Do not start implementation milestones from this document until the decisions in section 5
are closed.

## 0. Evidence references

Every citation is `<tag>:<path>:<line>` against the ref in this table, or a URL.

| Tag | Repository, ref | Path prefix |
|---|---|---|
| P | ROCm/rocm-libraries PR [#14255](https://github.com/ROCm/rocm-libraries/pull/14255) head `cb84aeb` | `projects/hipdnn/tools/hipdnn_vllm/` |
| D | ROCm/rocm-libraries `develop` `478baf73e66` | repository root |
| V | SamuelReeder/vllm `users/sareeder/qwen38-gfx942-specialized-decode` `cfc4c8c5b` (stacked on vLLM PR [#61311](https://github.com/vllm-project/vllm/pull/61311)) | `vllm/` |
| V2 | SamuelReeder/vllm `users/sareeder/qwen38-gfx942-tp2` `732f2867` | `vllm/` |
| S | megakernels `users/sareeder/hipdnn-spike` `dd51e2c` | `spike/hipdnn/` |
| G | megakernels `users/sareeder/runtime-geom` `4d41b3a` | `spike/rtgeom/` |
| A | proaditya/aiter `aitermk-qwen38` `29711d1` | `aitermk/engine_adapters/vllm/` |

The spike branches in rocm-libraries (`users/sareeder/megakernel-spike`, `102d947a342`) and vLLM
(`users/sareeder/qwen38-hipdnn-spike`, `6e0656478`) are local only. Their diffs are committed as
`S:c/rocm-libraries-spike-c.diff`, `S:engine/rocm-libraries-spike.diff` and `S:c/vllm-spike.diff`.

## 1. Fixed point

One decode step of a supported model runs as one persistent launch of a code object that FlyDSL
built. The launch gives the same bits as the FlyDSL launch of the same kernel source and `Cfg`.
hipDNN selects and launches the code object. vLLM keeps prefill, the LM head, sampling and every
step that the kernel does not serve. A deployment that does not ask for the kernel sees no change.

This stays true if every mechanism below is replaced.

## 2. Facts from the investigation

### 2.1 What the spike proved (Qwen3.8-27B TP1, MI300X)

- The FlyDSL code object is a plain AMDGPU ELF with a deterministic sha256. It is 237 KB at M = 1
  and 279 KB at M = 8 (`S:FINDINGS.md:26`). The kernel ABI is 20 B of kernarg: `tab` i64, `glob`
  i64, `nlayers` i32. It has no hidden arguments and no dynamic LDS (`S:FINDINGS.md:27`,
  `S:a/INTERFACE.md:41-43`).
- The path `Graph.custom_op` -> hip-kernel-provider ingestor pack `megakernel` -> kpack gives the
  same logits hash as the FlyDSL launch at M = 1 and M = 8. This is true eager and inside
  `torch.cuda.graph`. 96 greedy steps are bitwise equal (`S:FINDINGS.md:28-29`).
- In vLLM, widths 1, 2, 4 and 8 ran through hipDNN, eager and under graph capture. Greedy
  outputs are equal to the FlyDSL path for 8 prompts x 128 tokens in 3 of 3 runs
  (`S:FINDINGS.md:32`).
- Host cost per step: hipDNN execute 4.4 / 7.0 us (M = 1 / 8), the FlyDSL JIT call 70 us, the
  FlyDSL compiled fast path 3.9 us. TPOT is equal within noise (`S:FINDINGS.md:31`).
- An unknown key declines with "No engine configurations available" (`S:FINDINGS.md:30`).
- hipDNN does not remove the vLLM side: step checks, cache binding, weight repacking and the
  `tab`/`glob` tables stay in the DecodeSpec (`S:FINDINGS.md:59-63`).

Problems found (`S:FINDINGS.md:35-57`): the geometry is in the binary; torch ships a second
`libhipdnn_backend.so` and the spike needed `LD_PRELOAD`; a build from the Ubuntu 24.04 image does
not load in the Ubuntu 22.04 vLLM image; the feature flag needs two places; the custom op has
friction (unused `compute_data_type`, one int per byte in Python `set_data`). The spike did not
cover TP2, a co-residency check, concurrent execute from two threads, or kpack size.

### 2.2 Run-time geometry

`Cfg.rt_geom` (`G:RESULTS.md`, `S:FINDINGS.md:72-90`):

- 0: today's kernel. max_model_len, block, all strides and the CU count are compiled in.
- 1: max_model_len, maxblocks, block and all strides are run-time values. +1.1 % TPOT at M = 8
  (attention stage).
- 2: only max_model_len and maxblocks are run-time values. Within -0.8 % to +0.63 % TPOT on
  MI325X. Logits hashes are equal for rt 0, 1 and 2 at M = 1, 2, 4, 8.

rt2 builds the same ELF at max_model_len 4096 and 8192, but rt2 did not run through vLLM's
exported-binary path. rt2 was not measured on MI300X. Only Qwen3.8 TP1 has `rt_geom` at the time of
writing.

### 2.3 hipdnn_vllm (#14255)

- State: draft, author dpeabody-amd, last update 2026-10-23, 96 new files under
  `projects/hipdnn/tools/hipdnn_vllm/`. `pre-commit` fails. The superbuild CI does not run the
  plugin; GPU suites are author-run. Its paged-attention engines depend on #14245, #14247, #14254.
- Hook: a vLLM general plugin, `hipdnn = "hipdnn_vllm:register"` (`P:pyproject.toml:23-24`).
  `register()` (`P:hipdnn_vllm/__init__.py:127`) registers the `HIPDNN` attention backend
  (`P:hipdnn_vllm/__init__.py:95-104`), IR providers for `rms_norm` and `fused_add_rms_norm`, out-of-tree
  CustomOps (`P:hipdnn_vllm/__init__.py:166-171`), and an opt-in monkeypatch of
  `dispatch_unquantized_gemm` (`P:hipdnn_vllm/ops/linear.py:282`). The linear patch checks the exact
  vLLM version first (`P:hipdnn_vllm/ops/linear.py:215`).
- The `HIPDNN` attention backend subclasses `TritonAttentionBackend` and uses Triton's KV layout
  (`P:hipdnn_vllm/attention/backend.py:336`). The PR body states that its paged kernel is slower
  per decode step than vLLM's Triton kernel on CDNA.
- Runtime: one hipDNN handle per device, a process mutex (`P:csrc/runtime.cpp:36`,
  `P:csrc/runtime.cpp:261`). Plan build runs validate, build_operation_graph,
  create_execution_plans, check_support and build_plans (`P:csrc/runtime.cpp:321-408`). The plan
  cache is an LRU that also caches declines (`P:csrc/plan_cache.h:56-72`).
- Decline taxonomy: `NO_BUILD`, `NO_ENGINE`, `CHECK_SUPPORT`, `BUILD_PLANS`,
  `UNSUPPORTED_FEATURE`, `DISABLED`, `CAPTURE_INVARIANT`, `EXECUTE` (`P:csrc/decline.h:16-25`).
  A decline runs vLLM's own op. A HIP error inside an execute becomes an `EXECUTE` decline, and the
  caller's fallback runs (`P:csrc/runtime.cpp:474-491`). Ops have a `strict` argument that raises
  instead (`P:csrc/ops.cpp:73-74`).
- Graph capture: a plan hit under capture is pinned for the life of the process. A miss under
  capture is a `CAPTURE_INVARIANT` decline. No plan is built while a stream captures
  (`P:csrc/runtime.cpp:143-164`, `P:csrc/runtime.cpp:416-429`, `P:csrc/plan_cache.h:9`).
- Private hipDNN: the wheel bundles `backend/<family>/libhipdnn_backend.so` and
  `providers/<family>/`. `_C` loads them by explicit path with `RTLD_LOCAL`. torch's copy stays
  loaded and idle (`P:hipdnn_vllm/_bootstrap.py:6-25`, `P:hipdnn_vllm/_bootstrap.py:242`). The
  build fails if a provider has a `NEEDED` entry for `libhipdnn_backend`
  (`P:scripts/build_wheel.sh:169`). Extra provider directories load through
  `HIPDNN_VLLM_PROVIDER_DIRS`, additive or absolute (`P:hipdnn_vllm/_bootstrap.py:154-176`).
- Packaging: one wheel per GPU family; `gfx94X = ["gfx942"]` (`P:BUILD_PINS.toml:34-39`).
  Exact pins: Python 3.12, vLLM `0.32.1rc1.dev117+gdf3bd094a.rocm100`, torch `2.13.0+rocm10.0.0`,
  ROCm 10.0.0 (`P:BUILD_PINS.toml:9-15`). The provider build turns on
  `HIPDNN_ENABLE_KERNEL_INGESTOR` and requires DVC descriptors (`P:BUILD_PINS.toml:22-31`).
  `build_wheel.sh` runs `dvc pull` on the ingestor descriptor folder (`P:scripts/build_wheel.sh:91`)
  and builds `hipdnn;hip-kernel-provider;miopen-provider;hipblaslt-provider`
  (`P:scripts/build_wheel.sh:97-98`). The hipDNN commit is not pinned; it is the `HIPDNN_SRC`
  checkout.
- Absent: no model-runner, whole-model or decode-step hook; no TP group, peer memory or IPC (the
  rank only labels the stats file, `P:hipdnn_vllm/stats.py:143-153`); no third-party op. The torch
  op list is fixed (`P:csrc/ops.cpp:70-86`).
- The PR body plans to remove the bundled backend when vLLM pins a torch whose ROCm ships hipDNN
  with the needed changes and TheRock enables rocKE in hip-kernel-provider.

Answer to "can it host a whole-step decode hook": not today. It can host the launch of an opaque
custom-op graph with one small addition (a generic custom-op execute op). It has no code that
decides when a decode step runs.

### 2.4 vLLM #61311 and our DecodeSpecs

- #61311 is open, author simondanielsson, with merge conflicts on 2026-10-29. It refactors DSv4.1
  onto `DecodeSpec`. Qwen and Gemma specs are on our branches only.
- No registry. The package exports `DecodeSpec`, `SpecializedDecode`, `Step`,
  `select_decode_spec`, `tp_agree` (`V:models/common/specialized_decode/__init__.py:5-15`). Each model
  class returns its specs from a static method. The base returns `()`, the ROCm subclass imports
  `DECODE_SPEC` (`V:model_executor/models/qwen3_5.py:366-374`, `V:model_executor/models/qwen3_5.py:502-509`).
  An out-of-tree plugin can add a spec only by patching that static method.
- `DecodeSpec` fields: `name`, `supports`, `widths`, `refuse_config`, `refuse_model`,
  `refuse_caches`, `skip_step`, `build`, `pad_steps`, `set_capture_sizes`
  (`V:models/common/specialized_decode/spec.py:107-145`). The kernel object has `run` and
  `bind_caches`, and optionally `warmup_keys` and `compile` (`spec.py:65-98`).
- Refusal policy: with `--specialized-decode` set, a config refusal raises `DecodeUnsupported` at
  startup (`V:models/common/specialized_decode/guards.py:145-165`). A model refusal is agreed over the
  TP group with `tp_agree` (a gloo MIN all-reduce) and raises (`guards.py:44-56`,
  `dispatch.py:68-84`). A cache refusal raises (`dispatch.py:94-107`). Per step, `plan` returns
  `None` for an unserved width, a PIECEWISE graph, unbound caches or a `skip_step` reason, and the
  model runs its own forward (`dispatch.py:116-143`).
- Model Runner V2 and FULL graphs are required (`guards.py:97-102`). The spec's widths become the
  capture sizes (`guards.py:131-143`).
- Launch today: an in-process FlyDSL `@flyc.jit` launcher (`V2:models/qwen3_5/amd/specialized_decode/kernels/kernel.py:449-454`),
  called from `runtime/decode.py:276-288`. Widths are 1, 2, 4, 8 capped by `max_num_seqs`
  (`V2:models/qwen3_5/amd/specialized_decode/spec.py:131-133`). Qwen3.6 and Gemma4 import the same
  `_widths`.
- KV layouts the Qwen3.8 spec accepts: two TRITON_ATTN view forms and ROCM_ATTN (x-packed), by
  backend name (`V2:models/qwen3_5/amd/specialized_decode/spec.py:76-110`). Any other backend name
  refuses, which includes `HIPDNN`.
- The spike changed only the launch: `VLLM_MK_HIPDNN=1` builds one hipDNN graph per width and
  replaces the FlyDSL call (`S:c/vllm-spike.diff`, `runtime/hipdnn_launch.py`). The spike keyed
  geometry as `crc32(repr(Cfg with M = 0))` (`hipdnn_launch.py:38-39` in that diff).
- TP2 peer memory: the DecodeSpec builds a `PeerBuffer` from aiter's `UncachedIpcHeap`, exchanges
  IPC handles over the TP gloo group, agrees with `tp_agree`, and puts the peer table address and
  the rank into `glob` (`V2:models/qwen3_5/amd/specialized_decode/runtime/decode.py:95-109`,
  `decode.py:234-235`; `V2:models/deepseek_v41/amd/mono/common/peer_memory.py:10-56`). The per-step
  path decision has no collective; it depends only on the scheduler output
  (`decode.py:19-23`; megakernels `qwen3.8-27b-gfx942/docs/DESIGN-TP2.md` section 8).

### 2.5 aitermk (AMD, gfx950)

aitermk is a general plugin that needs no vLLM source change (`A:README.md:3-4`). `register`
wraps Model Runner V2's `GPUModelRunner`: `load_model`, `execute_model`, `prepare_inputs`,
`prepare_attn`, the cudagraph replays, `forward`, `compute_logits` and the speculator's `propose`
(`A:README.md:47-54`). It loads AOT `*.hsaco` from a directory named by an environment variable
(`A:README.md:27`). It fails closed at model load unless the deployment matches the kernel
(`A:README.md:35-38`). It is TP1 only.

### 2.6 Ingestor contract on develop

- The hipDNN custom op is on develop: `Graph::custom_op` (`D:projects/hipdnn/frontend/include/hipdnn_frontend/Graph.hpp:6232`).
- RFC 0017 status "In progress": the single-kernel path ships; UMD, UDD and native UHD resolve to
  C++ symbols; declarative criteria, UDD formulas, runtime drop-in, multi-launch and composition are
  not built (`D:projects/hipdnn/docs/rfcs/0017_UniversalKernelDescriptor.md:4-5`, `:114-136`).
  Below, `0017:<line>` means a line of this file.
- A prebuilt code object ships as build-time kind `hsaco`, packed as-is into a per-arch kpack
  (`0017:129-131`, `0017:296-300`, `0017:1884-1897`).
- KMD rule: the KMD field tuple is the kernel's unique catalog key, and every quantity a kernel
  bakes must be a KMD field that the matcher pins against the graph (`0017:229`, `0017:845-860`).
- A Python DSL is a build-time adapter, not a runtime dependency. A DSL that needs Python at run
  time ships as its own extension wheel with a provider (`0017:183-216`).
- Descriptor families: `HKP_DESCRIPTOR_FAMILIES` maps each folder to its enabling option; develop
  has only `rocKE` (`D:dnn-providers/hip-kernel-provider/descriptor-packaging/cmake/HkpPackaging.cmake:32`).
- The matcher context carries `deviceProperties`, with `multiProcessorCount`
  (`D:projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/ingestor/MatchContext.hpp:178`,
  `DeviceProperties.hpp:34`).
- Launch is plain `hipModuleLaunchKernel` (`D:dnn-providers/hip-kernel-provider/src/compilation/Kernel.cpp:96`).
  No code in hip-kernel-provider or the plugin SDK uses a cooperative launch or an occupancy query.
- Spike-only: the `megakernel` family and its option, `MegakernelNative.cpp` with symbols
  `megakernel.match_custom_op`, `megakernel.kernel_matches`, `megakernel.dispatch`, the KMD
  fields `model`, `M`, `geom` (`S:engine/rocm-libraries-spike.diff`, `S:c/rocm-libraries-spike-c.diff`).

### 2.7 Code-object precedent and CI

| Precedent | Artifact | Storage | Integrity | Evidence |
|---|---|---|---|---|
| asm_sdpa (AITER FMHA) | 294 prebuilt `.co`, gfx942 and gfx950 | plain git | sha256 manifest per arch in `SOURCE.md` | `D:dnn-providers/hip-kernel-provider/src/engines/asm_sdpa_engine/CMakeLists.txt:17`; `asm/asm_kernels/README.md:25` |
| ingestor rocKE bundles | KDPs, `*.co`, `*.hsaco` | DVC remote `ingestor`, `s3://therock-dvc/rocm-libraries/hipdnn/ingestor`, anonymous read | DVC md5, size and file count at configure | `D:.dvc/config`; `D:dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/README.md:73-87` |
| rocKE | Python DSL compiled at build time | private wheels in a build venv | wheel content stamp | `D:dnn-providers/hip-kernel-provider/descriptor-packaging/cmake/HkpPackaging.cmake:341-346,895-896` |
| superbuild CI | Python 3.12, ROCm from pip wheels, `dvc pull` of ingestor descriptors | | | `D:.github/workflows/hipdnn-superbuild-ci.yml:246-253` |

FlyDSL 0.4.1 is on PyPI as a 77 MB `manylinux_2_27` wheel per CPython version, with its own MLIR
(https://pypi.org/project/flydsl/, fetched 2026-10-29). The kernel sources are not in the wheel;
they are in this repository. CI has Python 3.12, so the wheel installs. Nobody has checked that a
FlyDSL compile runs without a GPU, or that two hosts give the same ELF.

## 3. Required properties (invariants)

| Id | Property | Check |
|---|---|---|
| H1 | Bitwise equal. For each shipped key, the launch through hipDNN gives the same output bits as the FlyDSL launch of the same kernel source and `Cfg`. | Logits hashes at every shipped M (as `S:c/logs/c1_final.log`); ELF sha256 equals the export manifest. |
| H2 | Decline goes to vLLM's path. If hipDNN has no plan for a width, that width runs vLLM's own forward. hipDNN never runs a different binary. | Remove one binary from the pack; that width logs a decline and runs vLLM's path; the other widths run the kernel. |
| H3 | All TP ranks agree. For each step, every rank takes the kernel path, or every rank takes vLLM's path. | `tp_agree` on the plan-build result per width; `VLLM_SPECIALIZED_DECODE_CHECK_PATHS` in validation runs. |
| H4 | Graph-capturable. The launch is captured in vLLM's FULL graph with fixed addresses. No plan is built during capture. | vLLM e2e under FULL graphs; zero `CAPTURE_INVARIANT` declines in the hipdnn_vllm report. |
| H5 | Co-residency is guaranteed. hipDNN launches a persistent kernel only when every CTA can be resident at the same time. Else it declines at plan build. | Matcher test with a wrong CU count; occupancy query result in the plan log. |
| H6 | Off by default. Without explicit opt-in, no code path, import or library load changes. | Default vLLM run with hipdnn_vllm installed: no megakernel plan, no FlyDSL import. |
| H7 | A failed launch never falls back after state changes. The kernel writes the KV cache and the GDN state in place. An execute failure raises; it does not re-run the step on vLLM's path. | Inject a HIP error after a launch; the engine stops. |

Preferences, not invariants: no FlyDSL or LLVM at serve time; no `LD_PRELOAD`; one binary list per
model that does not grow with max_model_len.

## 4. Binary key

A code object is selected by this tuple. Each item is a KMD field (RFC 0017 rule, `0017:845-860`),
except the arch, which the kpack arch table checks.

| Field | Source | Values today |
|---|---|---|
| `model` | DecodeSpec name | `qwen38-27b-mxfp4-gfx942`, `qwen36-35b-a3b-fp8-gfx942`, `gemma4-26b-a4b-fp8-gfx942` |
| `src` | sha256 of the kernel sources plus `repr(Cfg.compiled())`, the FlyDSL disk-cache key of vLLM `cfc4c8c5b` | one value per kernel release |
| arch | kpack arch table | gfx942 |
| `cus` | `Cfg`; pinned against `deviceProperties.multiProcessorCount` | 304 (MI300X, MI325X) |
| `kv` | KV layout class from the backend and view | Qwen3.8: TRITON form 1, TRITON form 2, ROCM x-packed; Qwen3.6: same set; Gemma4: TRITON only |
| `block` | KV block size (compiled at rt 0 and rt 2) | Qwen3.8 in vLLM: 784 |
| `M` | width | 1, 2, 4, 8 |
| `tp` | `Cfg.tp` | 1; 2 for Qwen3.8 |
| `rt` | `Cfg.rt_geom` | 0 today for Qwen3.6, Gemma4, Qwen3.8 TP2; 2 proposed |
| `lm_head` | `Cfg` | false in vLLM |

`src` alone identifies the bits, because the compile is deterministic for one FlyDSL version
(`S:FINDINGS.md:26`). The other fields exist so that the matcher can pin them against the graph
and the device, and so that a decline message names the field that differs.

Count per model, with rt2 and one KV layout class per model:

| Model | TP | Widths | Objects | Size each |
|---|---|---|---|---|
| Qwen3.8-27B | 1 | 1, 2, 4, 8 | 4 | 107-140 KB at rt0 (`S:c/export/`), 109-142 KB at rt1 (`G:logs/aus_runs/co_rt1_manifest.jsonl`); vLLM objects have no LM head |
| Qwen3.8-27B | 2 | 1, 2, 4, 8 | 4 (both ranks use the same binary) | not measured |
| Qwen3.6-35B-A3B | 1 | 1, 2, 4, 8 | 4 | not measured |
| Gemma4-26B-A4B | 1 | 1, 2, 4, 8 | 4 | not measured |

Total: 16 objects for one KV class per model (about 2 MB before kpack compression), 40 if every
accepted Qwen layout ships. At rt0, the count multiplies by the number of supported max_model_len
values. [INFERENCE] MI300X and MI325X share one binary, because both are gfx942 with 304 CUs and
the compile input is the same; the TP2 kernel gave the same bits on both (`qwen3.8-27b-gfx942/README.md`).
Another gfx942 part with another CU count needs its own set.

## 5. Decisions

Each decision lists the candidates, their tradeoffs, the evidence that would reject each, and a
recommendation.

### D1. Where vLLM enters the megakernel path

| Candidate | For | Against | Rejected if |
|---|---|---|---|
| (a) hipdnn_vllm wraps the model runner, as aitermk does | No vLLM source change (`A:README.md:3-4`). | Patches about eight runner internals. Duplicates #61311's guards, capture sizes and TP agreement. hipdnn_vllm pins an exact vLLM build (`P:BUILD_PINS.toml:11`), so every vLLM bump re-checks the patches. | #61311 merges: the runner hooks then duplicate an upstream mechanism. |
| (b) An out-of-tree plugin registers a DecodeSpec | Host code and binaries ship and version together. Per-model code stays out of vLLM. | #61311 has no registry (`V:model_executor/models/qwen3_5.py:366-374`). Without a hook, the plugin patches a static method. | #61311 maintainers refuse a registry hook, or the patch target moves with each vLLM bump. |
| (c) In-tree DecodeSpecs call hipdnn_vllm only for the launch | The spike proved it (`S:FINDINGS.md:9-20`). The smallest change: one launch backend. hipdnn_vllm gets one generic op, not model code. | Host code (vLLM) and binaries (hipdnn_vllm wheel) ship separately and must match. The per-model specs must go upstream in vLLM. | vLLM does not take the per-model specs, or hipdnn_vllm owners refuse a generic custom-op execute op. |

Recommendation: (c). The `src` field (section 4) turns any mismatch between the host code and the
binary into a decline (H2), not wrong bits. Keep (b) as the fallback if vLLM does not take the
per-model specs; then the specs move into a plugin that this repository owns, not into
hipdnn_vllm. Reject (a) while #61311 is open.

### D2. Launch interface in hipdnn_vllm

| Candidate | For | Against | Rejected if |
|---|---|---|---|
| (a) A generic op `custom_op_exec(op_id, payload, inputs, strict)` in `_C` | One op serves every custom op. Uses the plan cache, capture pinning and stats as they are. | The plan key must include the payload bytes. | The plan-key cost per step shows in TPOT. |
| (b) A megakernel-specific op | A typed schema. | Model knowledge enters hipdnn_vllm. | n/a |
| (c) The DecodeSpec calls `hipdnn_frontend` directly, as the spike did | No hipdnn_vllm change. | Loads a second hipDNN backend; needs `LD_PRELOAD` (`S:FINDINGS.md:42-45`). | Any run with torch 2.13 (fails without `LD_PRELOAD`). |

Recommendation: (a), with `strict` always true for this op (H7). Build the plan for each width in
the DecodeSpec's warmup (`compile`), before capture, so H4 holds.

### D3. hipDNN graph form

| Candidate | For | Against |
|---|---|---|
| (a) `Graph.custom_op`, opaque payload, native pack symbols | On develop today. The spike used it. | The payload schema is ours to version. Two friction items need small frontend fixes. |
| (b) A declarative UKD with UDD formulas | Data only, no native code. | UDD formulas are not built (`0017:4-5`). |
| (c) A new first-class hipDNN operation | Typed. | No other user. Large API change. |

Recommendation: (a). Put a schema version in the payload. Ask the hipDNN team to fix the two custom
op friction items (`S:FINDINGS.md:52-54`).

### D4. Geometry at compile time or at run time

| Candidate | For | Against | Rejected if |
|---|---|---|---|
| rt0 and a fixed list of geometries | No kernel change. | One set per max_model_len. | n/a: the list grows with each deployment. |
| rt1 | One set per (model, M, CU count). | +1.1 % TPOT at M = 8 (`G:RESULTS.md`). | The block size never varies in practice (then rt2 is enough). |
| rt2 | 4 per model per (layout, block). Within +0.63 %. | Block and strides stay compiled. | MI300X TPOT for rt2 exceeds +1 %, or rt2 fails the vLLM exported-binary run. |

Recommendation: rt2 for every model, with explicit KMD fields (section 4) in place of the spike's
`geom` crc.

### D5. Where code objects are built and stored

| Candidate | For | Against | Rejected if |
|---|---|---|---|
| (a) Commit `.co` to rocm-libraries git | Precedent: 294 asm_sdpa `.co` with sha256 manifests. | The ingestor descriptor policy puts code objects in DVC, not git (`D:.../descriptors/README.md:73-87`). Each kernel release adds 2-5 MB to history. | The hipDNN maintainers apply the DVC policy (likely). |
| (b) Export in this repository, store in the DVC `ingestor` remote with a committed sha256 lock | Follows develop's policy. `build_wheel.sh` already pulls that folder (`P:scripts/build_wheel.sh:91`). The validated bytes are the shipped bytes. | Needs write access to the DVC remote. | The DVC remote does not accept non-rocKE bundles. |
| (c) rocm-libraries CI compiles with a pinned FlyDSL wheel, like rocKE | No binary in storage. Source and binary cannot drift. | Adds a 77 MB build dependency and cross-repository kernel sources. FlyDSL brings its own MLIR and LLVM, separate from the CI's ROCm toolchain. Output equality across hosts is not checked. | Two hosts give different ELF sha256, or the compile needs a GPU. |

Recommendation: (b) now. Add (c) later as a check job that rebuilds and compares sha256 with the
lock. Do not ship from (c).

### D6. Delivery inside hipdnn_vllm's bundled backend

| Candidate | For | Against |
|---|---|---|
| (a) `megakernel` descriptor family in hip-kernel-provider, option default OFF; hipdnn_vllm adds the option to `BUILD_PINS.toml` `cmake_flags` | Rides on the private backend load (no `LD_PRELOAD`). The KDP declares gfx942 only, so only the `gfx94X` wheel holds the objects. | The family lives in rocm-libraries; the hipDNN team reviews it. When the bundled backend goes away, TheRock must enable the family. |
| (b) A separate provider in an extension wheel, loaded with `HIPDNN_VLLM_PROVIDER_DIRS` | Independent release. | Must match the bundled backend version (`P:hipdnn_vllm/_bootstrap.py` version guard). A second provider build to maintain. |
| (c) Runtime drop-in of descriptors | No rebuild. | Not built (`0017:131`). |

Recommendation: (a). One option name for both the CMake option and the family entry, to remove
the two-place flag (`S:FINDINGS.md:48-51`).

### D7. Co-residency guarantee (H5)

| Candidate | For | Against | Rejected if |
|---|---|---|---|
| (a) Matcher pins `cus` to `multiProcessorCount`; the dispatch `prepare` calls `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor` and declines below 1 | Uses existing match context. A decline, not a hang. | Does not protect against other work on the same CUs at run time. | n/a |
| (b) Cooperative launch (`hipModuleLaunchCooperativeKernel`) | The runtime refuses a grid that cannot be resident. | Not used anywhere in hip-kernel-provider. Capture support in HIP graphs is not checked. Possible launch cost. | HIP graph capture rejects it, or TPOT rises. |
| (c) Trust grid = CU count (today) | No work. | A hang on a partitioned or shared GPU. | Any deployment with GPU partitioning. |

Recommendation: (a) now; measure (b) and switch if capture accepts it at no TPOT cost. State in
the pack that vLLM runs the decode step on one stream and that other processes on the GPU are not
supported.

### D8. TP2 and peer memory

| Candidate | For | Against |
|---|---|---|
| (a) The caller owns peer memory. The DecodeSpec builds the `PeerBuffer` and puts addresses into `glob`, as today (`V2:.../runtime/decode.py:95-109,234-235`). hipDNN sees opaque pointers. | No hipDNN API change. The TP2 code in vLLM stays as it is. | The DecodeSpec keeps the aiter `UncachedIpcHeap` dependency. |
| (b) hipDNN gets a peer-memory or communicator API | hipDNN owns the whole step. | No precedent in hipDNN or hipdnn_vllm (no TP code, section 2.3). Large API. |

Recommendation: (a). Add two rules: agree the plan-build result per width with `tp_agree` at
warmup (H3); treat any execute failure under TP2 as fatal (H7), because one rank on vLLM's path
leaves the peer spinning on a mailbox.

### D9. Persistent device state

The barrier epoch, the GEMV counters and the staging buffers persist across steps.

| Candidate | For | Against |
|---|---|---|
| (a) The caller owns them, reached through `tab`/`glob` (today) | Works; the spike used it. | hipDNN cannot check them. |
| (b) hipDNN workspace | Allocated by hipDNN. | The workspace is per-execute scratch from the caching allocator (`P:csrc/graph_api.cpp:130-159`); its contents do not persist. |

Recommendation: (a). Report workspace size 0.

### D10. Decline policy

| Case | Recommended behavior |
|---|---|
| No binary or matcher refusal for a width | `skip_step` returns the reason for that width; vLLM's path runs it; one log line and a hipdnn_vllm stats row (H2). |
| No binary for any width, `--specialized-decode` set | Follow #61311: raise `DecodeUnsupported` at startup (`guards.py:160-165`). Do not fall back to the FlyDSL JIT silently. |
| Execute failure | Raise (H7). |
| Miss under capture | Cannot occur if warmup built every width; if it occurs, raise. |

### D11. Opt-in (H6)

Recommendation: three gates, all off by default: vLLM `--specialized-decode`; a launch backend
setting in the DecodeSpec (`flydsl` default, `hipdnn` opt-in); and the hipdnn_vllm op name in
`HIPDNN_VLLM_OPS`, outside the default list, as `conv` and `linear` are today
(`P:hipdnn_vllm/config.py:27-37`).

## 6. Packaging path, end to end (recommended candidates)

1. This repository exports each code object (`COMPILE_ONLY=1`, the ELF from `gpu.binary`) with a
   manifest line: key fields, sha256, FlyDSL version, megakernels commit.
2. A GPU job checks H1 for each object and commits the manifest as the lock.
3. The objects go to the DVC `ingestor` remote under
   `kernel_ingestor_engine/descriptors/megakernel/`, with the KDP, UKDs, KMD, UMD, UDD and UED.
4. hip-kernel-provider packs them with kind `hsaco` into the gfx942 kpack when the option is ON.
5. `build_wheel.sh gfx94X` pulls DVC, builds the provider with the option, and bundles it.
6. At run time, hipdnn_vllm loads the private backend; the DecodeSpec builds one plan per width at
   warmup through the generic op; vLLM captures the launches.

## 7. Ownership and upstream sequence

| Step | Owner | Depends on |
|---|---|---|
| 1. rt2 for Qwen3.6, Gemma4, Qwen3.8 TP2; exporter and lock | megakernels | none |
| 2. RFC 0017 note: persistent-grid kernels (CU pin, occupancy decline, caller-owned state, strict execute) | hipDNN team, with us | RFC 0017 |
| 3. Custom-op friction fixes in the frontend | hipDNN team | none |
| 4. `megakernel` descriptor family, native pack, DVC bundle | us, reviewed by the hipDNN team | 1, 2 |
| 5. #14255 merges | dpeabody-amd | #14245, #14247, #14254 for attention only |
| 6. Generic custom-op execute op, opt-in op name, `BUILD_PINS.toml` flag | us, PR to hipdnn_vllm | 4, 5 |
| 7. #61311 merges | simondanielsson | vLLM review |
| 8. Per-model DecodeSpecs upstream with a launch backend setting | us | 6, 7 |

If step 8 fails, use D1 (b): a plugin in this repository registers the specs; then a registry
hook in #61311 replaces the static-method patch.

## 8. Risks

1. Hang from missing co-residency. The grid barrier waits forever if one CTA is not resident.
   Nothing in hipDNN checks this today (section 2.6). D7 reduces it; other processes on the GPU
   still can cause it.
2. Version lock across three repositories. Kernel sources in vLLM, binaries in the hipdnn_vllm
   wheel, and hipdnn_vllm's exact `vllm==` pin (`P:BUILD_PINS.toml:11`) must agree. The `src` field
   makes drift a decline, so the cost is lost speed with no error. Our vLLM fork is based after
   `df3bd094`, so pip refuses hipdnn_vllm's pin unless both move together.
3. Fallback after in-place writes. hipdnn_vllm's default is to fall back after an `EXECUTE`
   decline (`P:csrc/runtime.cpp:474-491`). For this kernel that re-runs a step on changed KV and
   GDN state, or splits TP ranks. H7 and the `strict` flag remove it; a review must keep it.
4. #14255 is a draft with no activity since 2026-10-23 and failing pre-commit. #61311 has merge
   conflicts. Each delay moves step 6 or step 8.
5. The `HIPDNN` attention backend name refuses in our `refuse_caches` (`V2:.../spec.py:93-103`).
   A user who enables hipdnn_vllm attention loses the kernel until the spec accepts `HIPDNN` as
   the Triton layout.
6. The bundled backend is temporary (#14255 body). The `megakernel` family must then be ON in
   TheRock's hip-kernel-provider, or the delivery changes to D6 (b).
7. rt2 is measured only on MI325X and only for Qwen3.8 TP1.

## 9. Open questions

| Question | Evidence that settles it |
|---|---|
| Does HIP graph capture accept a cooperative launch on ROCm 10, and at what cost? | Capture a cooperative launch of the M = 1 binary in vLLM; compare TPOT. |
| Does FlyDSL compile without a GPU, and do two hosts give the same ELF? | Export on a CPU-only host and on a gfx942 node; compare sha256. |
| Is rt2 within +1 % on MI300X, and does it pass the vLLM exported-binary path? | `bench_ab.py` on MI300X; `vllm_e2e.py` with rt2 code objects. |
| Does vLLM take the per-model DecodeSpecs upstream? | Maintainer answer on the first spec PR after #61311. |
| Does the DVC `ingestor` remote accept non-rocKE bundles, and who has write access? | Ask the hip-kernel-provider owners. |
| Which KV layout class does each model use by default on gfx942 in the pinned vLLM? | Read the backend selection at the pinned commit; log `attention_backend` in an e2e run. |
| Can two threads call execute on one device at the same time? | hipdnn_vllm holds a per-device mutex (`P:csrc/runtime.cpp:36`); confirm vLLM never calls the decode op from two threads. |

## 10. Disposable

The spike code (`S:a/`, `S:engine/`, `S:c/`), its `geom` crc key, the hard-coded `nlayers`, grid
and block in the payload, and `LD_PRELOAD` are disposable. Keep their logs as evidence.
