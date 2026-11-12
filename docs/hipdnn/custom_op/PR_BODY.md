**Stacked on #14255** (`users/dpeabody/hipdnn-vllm`, head `cb84aebee6f`). Review only the 3 commits after that head. The megakernel engine (`users/sareeder/megakernel-engine`) is a separate change and is not in this branch.

## Summary
This change adds an opaque CUSTOM_OP route to `hipdnn_vllm`. A caller (the vLLM megakernel decode launch) gives an op ID, a JSON payload and input tensors. `_C` builds a one-node CUSTOM_OP graph, and hipDNN selects an engine that accepts it (for example the megakernel ingestor engine). The route is opt-in. With the default `HIPDNN_VLLM_OPS`, no CUSTOM_OP node goes to hipDNN.

JIRA ID : AICK-1847

## Risk Assessment
Risk 2. The new code is a new op that is off by default: when `custom_op` is not in `HIPDNN_VLLM_OPS`, prepare returns `DISABLED` and exec declines without a graph build. No default op changes its route or engine. The `build_wheel.sh` changes add family-scoped CMake flags and one more runtime library directory for the dependency check. The gfx94X wheel sets `-DENABLE_MEGAKERNEL_ENGINE=ON`. Without the engine sources, CMake reports the variable as unused and the build does not change.

## ASIC Coverage
Host-side code and a gfx94X-only build flag. No kernel code. The engine-backed tests run only on gfx942 (the megakernel catalog is gfx942-only). This change was tested on gfx942 (MI300X). Other families get the new op but no engine for it, so the op declines; PR CI is sufficient for them.

## Testing Summary
- Unit (no GPU): the `hipdnn_vllm` `tests/unit` suite on the gfx94X wheel of this branch.
- GPU, this branch: `tests/gpu/test_custom_op.py`. The engine tests skip because the engine is not in the build.
- GPU, this branch with the engine merged locally (not pushed): `tests/gpu/test_custom_op.py` with the production catalog (incl. the H6 opt-in check), and again with the engine's stand-in catalog for capture/replay and key mismatch.
- The wheels were built in the vLLM ROCm image (Ubuntu 22.04) that ran the tests, on one gfx942 node (SLURM job 68377779).

## Testing Checklist
- [x] gfx94X wheel build, this branch - `scripts/build_wheel.sh gfx94X` - Status: Passed ([build_A.log](https://github.com/SamuelReeder/megakernels/blob/users/sareeder/custom-op-evidence/docs/hipdnn/custom_op/rebuild_68377779/build_A.log))
- [x] Unit tests - `pytest tests/unit` - 205 passed - Status: Passed ([unit_A.log](https://github.com/SamuelReeder/megakernels/blob/users/sareeder/custom-op-evidence/docs/hipdnn/custom_op/rebuild_68377779/unit_A.log))
- [x] custom_op GPU tests, gfx942, no engine - `pytest tests/gpu/test_custom_op.py` - 5 passed, 3 skipped (engine not in build) - Status: Passed ([custom_op_A.log](https://github.com/SamuelReeder/megakernels/blob/users/sareeder/custom-op-evidence/docs/hipdnn/custom_op/rebuild_68377779/custom_op_A.log))
- [x] custom_op GPU tests, gfx942, engine merged locally - `pytest tests/gpu/test_custom_op.py` - 6 passed, 2 skipped (stand-in not loaded) - Status: Passed ([build_B.log](https://github.com/SamuelReeder/megakernels/blob/users/sareeder/custom-op-evidence/docs/hipdnn/custom_op/rebuild_68377779/build_B.log), [custom_op_B.log](https://github.com/SamuelReeder/megakernels/blob/users/sareeder/custom-op-evidence/docs/hipdnn/custom_op/rebuild_68377779/custom_op_B.log))
- [x] Stand-in capture/replay and key mismatch, gfx942, engine merged locally - `HIPDNN_DESCRIPTOR_DIR=<stand-in> pytest tests/gpu/test_custom_op.py -k megakernel` - 2 passed - Status: Passed ([custom_op_B_standin.log](https://github.com/SamuelReeder/megakernels/blob/users/sareeder/custom-op-evidence/docs/hipdnn/custom_op/rebuild_68377779/custom_op_B_standin.log))
- [ ] Full `tests/gpu` on this branch - not run here. On the earlier branch, its 74 paged-SDPA failures also occur on the #14255 head ([NOTES.md](https://github.com/SamuelReeder/megakernels/blob/users/sareeder/custom-op-evidence/docs/hipdnn/custom_op/NOTES.md)) - Status: Pending
- [ ] PR CI - GitHub PR checks - Status: Pending

Job script and summary: [run.sh](https://github.com/SamuelReeder/megakernels/blob/users/sareeder/custom-op-evidence/docs/hipdnn/custom_op/rebuild_68377779/run.sh), [run.log](https://github.com/SamuelReeder/megakernels/blob/users/sareeder/custom-op-evidence/docs/hipdnn/custom_op/rebuild_68377779/run.log). Design: [hipdnn-route.md](https://github.com/SamuelReeder/megakernels/blob/main/docs/hipdnn-route.md).

## Technical Changes
- `_C` gets `custom_op_prepare(op_id, payload, inputs) -> str` and `custom_op_exec(op_id, payload, inputs, strict)` (`csrc/graphs/custom_op.cpp`). Prepare returns the decline reason, or an empty string when a plan is ready, and never raises. Exec raises only when `strict`; otherwise it counts a `fallback` for op `custom_op`.
- Graph: one CUSTOM_OP node. Inputs bind as UIDs 1..n in call order. The node data is the payload bytes. The plan key is the op ID, the payload, and the dims, strides and dtype of each input; it never includes addresses. No plan builds during graph capture (`CAPTURE_INVARIANT` decline).
- Opt-in: `custom_op` is a valid name in `HIPDNN_VLLM_OPS` but is not in the defaults. The bootstrap calls `custom_op_set_enabled(op_enabled("custom_op"))`. While off, the op declines `DISABLED` without a graph build.
- `BUILD_PINS.toml`: `[provider.family_cmake_flags]`, with `gfx94X = ["-DENABLE_MEGAKERNEL_ENGINE=ON"]`.
- `fake_impls.py` registers a fake implementation of `custom_op_exec`, for `torch.compile`. `setup.py` compiles `csrc/graphs/custom_op.cpp`. The README lists the new op.
- `build_wheel.sh`: appends the family flags to the provider flags and to `_build_info.json`. The dependency check also searches `_rocm_sdk_core/lib/rocm_sysdeps/lib`, so the wheel builds in the vLLM ROCm image.
- `tests/gpu/test_custom_op.py`: decline contract, plan-cache keying, disabled path, capture miss, the H6 opt-in check, and stand-in capture/replay and key mismatch. The engine tests skip when the engine or the stand-in is not loaded.
