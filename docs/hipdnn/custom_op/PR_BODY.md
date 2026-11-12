**Stacked on #14255** (`users/dpeabody/hipdnn-vllm`, head `cb84aebee6f`). Review only the 3 commits after that head. The megakernel engine (`users/sareeder/megakernel-engine`) is a separate change and is not in this branch.

## Summary
This change adds an opaque CUSTOM_OP route to `hipdnn_vllm`. A caller (the vLLM megakernel decode launch) gives an op ID, a JSON payload and input tensors. `_C` builds a one-node CUSTOM_OP graph, and hipDNN selects an engine that accepts it (for example the megakernel ingestor engine). The route is opt-in. With the default `HIPDNN_VLLM_OPS`, no CUSTOM_OP node goes to hipDNN.

JIRA ID : AICK-1847

## Risk Assessment
Risk 2. The new code is a new op that is off by default: when `custom_op` is not in `HIPDNN_VLLM_OPS`, prepare returns `DISABLED` and exec declines without a graph build. No default op changes its route or engine. The `build_wheel.sh` changes add family-scoped CMake flags and one more runtime library directory for the dependency check. The gfx94X wheel sets `-DENABLE_MEGAKERNEL_ENGINE=ON`. Without the engine sources, CMake reports the variable as unused and the build does not change.

## ASIC Coverage
No PR CI job builds or tests `hipdnn_vllm`: no file outside `projects/hipdnn/tools/hipdnn_vllm/` refers to it, other than one document. Passing PR CI therefore does not cover this change. The only coverage is the manual gfx942 (MI300X) job below.
- gfx94X: wheel built; unit tests and `test_custom_op.py` passed on gfx942, with and without the engine.
- gfx950: wheel built (build only). Not tested on a GPU.
- gfx110X-all, gfx1151, gfx120X-all: not built and not tested.
- The `build_wheel.sh` runtime library change (`rocm_sysdeps/lib`) applies to every family. It is verified only for the gfx94X and gfx950 builds.

The C++ code is host-side and does not depend on the GPU architecture. Before merge, build the other family wheels, or accept the gap explicitly.

## Testing Summary
- Unit (no GPU): the `hipdnn_vllm` `tests/unit` suite on the gfx94X wheel of this branch.
- GPU, this branch: `tests/gpu/test_custom_op.py`. The engine tests skip because the engine is not in the build.
- GPU, this branch with the engine merged locally (not pushed): `tests/gpu/test_custom_op.py` with the production catalog (incl. the H6 opt-in check), and again with the engine's stand-in catalog for capture/replay and key mismatch.
- Regression: the plan-key and `torch.compile` tests fail on the previous head's wheel (`89fcfed57f6`) and pass on this head.
- All wheels were built in the vLLM ROCm image (Ubuntu 22.04) that ran the tests, on one gfx942 node (SLURM job 68378225).

## Testing Checklist
- [x] gfx94X wheel build - `scripts/build_wheel.sh gfx94X` - Status: Passed ([build_A.log](https://github.com/SamuelReeder/megakernels/blob/users/sareeder/custom-op-evidence/docs/hipdnn/custom_op/rebuild_68378225/build_A.log))
- [x] gfx950 wheel build (build only) - `scripts/build_wheel.sh gfx950` - Status: Passed ([build_C.log](https://github.com/SamuelReeder/megakernels/blob/users/sareeder/custom-op-evidence/docs/hipdnn/custom_op/rebuild_68378225/build_C.log))
- [x] Unit tests - `pytest tests/unit` - 205 passed - Status: Passed ([unit_A.log](https://github.com/SamuelReeder/megakernels/blob/users/sareeder/custom-op-evidence/docs/hipdnn/custom_op/rebuild_68378225/unit_A.log))
- [x] custom_op GPU tests, gfx942, no engine - `pytest tests/gpu/test_custom_op.py` - 7 passed, 3 skipped (engine not in build) - Status: Passed ([custom_op_A.log](https://github.com/SamuelReeder/megakernels/blob/users/sareeder/custom-op-evidence/docs/hipdnn/custom_op/rebuild_68378225/custom_op_A.log))
- [x] custom_op GPU tests, gfx942, engine merged locally - `pytest tests/gpu/test_custom_op.py` - 8 passed, 2 skipped (stand-in not loaded) - Status: Passed ([build_B.log](https://github.com/SamuelReeder/megakernels/blob/users/sareeder/custom-op-evidence/docs/hipdnn/custom_op/rebuild_68378225/build_B.log), [custom_op_B.log](https://github.com/SamuelReeder/megakernels/blob/users/sareeder/custom-op-evidence/docs/hipdnn/custom_op/rebuild_68378225/custom_op_B.log))
- [x] Stand-in capture/replay and key mismatch, gfx942, engine merged locally - `HIPDNN_DESCRIPTOR_DIR=<stand-in> pytest tests/gpu/test_custom_op.py -k megakernel` - 2 passed - Status: Passed ([custom_op_B_standin.log](https://github.com/SamuelReeder/megakernels/blob/users/sareeder/custom-op-evidence/docs/hipdnn/custom_op/rebuild_68378225/custom_op_B_standin.log))
- [x] Regression tests fail on the previous head's wheel - `pytest tests/gpu/test_custom_op.py -k "nul or compile"` - 2 failed, as expected - Status: Passed ([custom_op_old.log](https://github.com/SamuelReeder/megakernels/blob/users/sareeder/custom-op-evidence/docs/hipdnn/custom_op/rebuild_68378225/custom_op_old.log))
- [ ] gfx110X-all, gfx1151, gfx120X-all wheel builds - `scripts/build_wheel.sh <family>` - Status: Not run
- [ ] gfx950 GPU tests - Status: Not run
- [ ] Full `tests/gpu` on this branch - not run here. On the earlier branch, its 74 paged-SDPA failures also occur on the #14255 head ([NOTES.md](https://github.com/SamuelReeder/megakernels/blob/users/sareeder/custom-op-evidence/docs/hipdnn/custom_op/NOTES.md)) - Status: Pending
- [ ] PR CI - GitHub PR checks (they do not build `hipdnn_vllm`) - Status: Pending

Job script and summary: [run.sh](https://github.com/SamuelReeder/megakernels/blob/users/sareeder/custom-op-evidence/docs/hipdnn/custom_op/rebuild_68378225/run.sh), [run.log](https://github.com/SamuelReeder/megakernels/blob/users/sareeder/custom-op-evidence/docs/hipdnn/custom_op/rebuild_68378225/run.log). Design: [hipdnn-route.md](https://github.com/SamuelReeder/megakernels/blob/main/docs/hipdnn-route.md).

## Technical Changes
- `_C` gets `custom_op_prepare(op_id, payload, inputs) -> str` and `custom_op_exec(op_id, payload, Tensor(a!)[] inputs, strict)` (`csrc/graphs/custom_op.cpp`). Prepare returns the decline reason, or an empty string when a plan is ready, and never raises. Exec raises only when `strict`; otherwise it counts a `fallback` for op `custom_op`.
- The exec schema marks the inputs as mutated, because the engine writes through them. Without this annotation, `torch.compile` removes the call (exec returns nothing).
- Graph: one CUSTOM_OP node. Inputs bind as UIDs 1..n in call order. The node data is the payload bytes. The plan key is the op ID, the payload, and the dims, strides and dtype of each input; it never includes addresses. No plan builds during graph capture (`CAPTURE_INVARIANT` decline).
- `PlanKey::add(std::string_view)` appends the exact bytes. The op ID and the payload use it, so bytes after an embedded NUL are part of the key.
- Opt-in: `custom_op` is a valid name in `HIPDNN_VLLM_OPS` but is not in the defaults. The bootstrap calls `custom_op_set_enabled(op_enabled("custom_op"))`, which returns the previous state. While off, the op declines `DISABLED` without a graph build.
- `BUILD_PINS.toml`: `[provider.family_cmake_flags]`, with `gfx94X = ["-DENABLE_MEGAKERNEL_ENGINE=ON"]`.
- `fake_impls.py` registers a fake implementation of `custom_op_exec`, for `torch.compile`. `setup.py` compiles `csrc/graphs/custom_op.cpp`. The README lists the new op.
- `build_wheel.sh`: appends the family flags to the provider flags and to `_build_info.json`. The dependency check also searches `_rocm_sdk_core/lib/rocm_sysdeps/lib`, so the wheel builds in the vLLM ROCm image.
- `tests/gpu/test_custom_op.py`: decline contract, plan-cache keying (incl. payload bytes after a NUL), exec under `torch.compile`, disabled path, capture miss, the H6 opt-in check, and stand-in capture/replay and key mismatch. The fixtures restore the previous enable state. The engine tests skip when the engine or the stand-in is not loaded.
