## Summary

This change adds a `megakernel` descriptor family to the kernel ingestor engine of hip-kernel-provider. The family lets hipDNN run precompiled persistent decode megakernels through a CUSTOM_OP node. The `ENABLE_MEGAKERNEL_ENGINE` CMake option controls the family. The default is OFF.

Tracking: no Jira issue has been identified for this change.

## Risk Assessment

Risk 2 of 5. With the option OFF (the default), these parts of the product change:

- `HKP_DESCRIPTOR_FAMILIES` gets the entry `megakernel=ENABLE_MEGAKERNEL_ENGINE`. The pack excludes the `megakernel/` folder when the option is OFF.
- `HkpPackaging.cmake` checks the megakernel lock before packing, unless the family is excluded. Separate helpers build the folder filters and lock commands. The rebase keeps both upstream staging inputs and lock dependencies.
- `Kernel::function()`, a new accessor on `compilation::Kernel`. Only the megakernel pack calls it.
- `MEGAKERNEL_ENGINE_NAME`, a new constant in `IngestorPacks.hpp`.
- `TestKernelIngestorEngine` keeps the upstream dispatch symbols and adds one megakernel symbol under `#ifdef ENABLE_MEGAKERNEL_ENGINE`.

All other new C++ compiles only with the option ON. With the option ON, the pack step checks the sha256 and size of each code object against the committed lock. A mismatch stops the build.

DVC: the code objects are not in the `ingestor` remote. An anonymous GET of the pointer's `.dir` object returns 404. We do not have write access to the bucket. Thus this PR does not contain the pointer `descriptors/megakernel/artifacts.dvc`. The pointer and the configure-time DVC check for the folder are on `users/sareeder/megakernel-engine-dvc`, one commit on this branch. That commit must land together with the upload, not before. These lanes would fail if the pointer landed before the upload:

- `hipdnn-superbuild-ci.yml`, both lanes: `dvc pull --recursive dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors` (Windows lines 230-231, Linux lines 518-519).
- Bare `dvc pull` of the whole repository: `component-ci-miopen.yml:172`, `hipblaslt-asan-ci.yml:176`, `miopen-wrapper-flag-on.yml:64`, and TheRock `fetch_sources`.

Anonymous `dvc pull --recursive` of the descriptors folder (DVC 3.62.0, as in CI) fetched all 106 tracked outputs on `3bb8d27b9db`. On the `-dvc` branch (`62482b627f2`), it fetched those same outputs but failed to check out `megakernel/artifacts`. See the [PR branch pull](https://github.com/SamuelReeder/megakernels/blob/24f93bf4509b0ee8dab9148e326f3691103a17aa/docs/hipdnn/engine/logs/rebase-3bb8d27b9db/dvc-pull-anonymous.log) and [pointer branch pull](https://github.com/SamuelReeder/megakernels/blob/24f93bf4509b0ee8dab9148e326f3691103a17aa/docs/hipdnn/engine/logs/rebase-3bb8d27b9db/dvc-pull-pointer-branch.log).

## ASIC Coverage

The change is arch-scoped: the code objects and the lock are for gfx942 only. The matcher declines a payload whose `arch` is not the device arch.

CI jobs that build hip-kernel-provider (`.github/workflows/hipdnn-superbuild-ci.yml`, preset `hipdnn-dev-all`, `HIPDNN_ENABLE_KERNEL_INGESTOR=ON`):

| Job | GPU_TARGETS | Megakernel family |
|---|---|---|
| HipDNN Linux Superbuild | gfx942 | Not built. The job does not set `ENABLE_MEGAKERNEL_ENGINE` (default OFF). |
| HipDNN Windows Superbuild | gfx1151 | Not built, for the same reason. |

PR CI covers only the OFF build, on gfx942 and gfx1151. CI ASAN and clang-tidy do not see the megakernel code. Manual ON checks use gfx942 only. `therock-multi-arch-ci.yml` also runs on PRs (default families gfx94X, gfx950, gfx125X). This work did not confirm that it builds hip-kernel-provider.

Before the `-dvc` commit lands, one gfx942 ON run with the objects pulled from the `ingestor` remote is necessary.

## Testing Summary

- Unit tests (`hip_kernel_provider_tests`) with the option OFF, ON, and ON with no code objects.
- Megakernel cases: 15 with the option ON. 12 are matcher cases: payload, key fields, device CU count, `nlayers` bounds (0, -1, INT32_MAX + 1; INT32_MAX passes), input count (0, 1, 3). 3 are co-residency cases: another CU count, no workgroup per CU, accepted. With the option OFF, none of them compile.
- Lock tests (`test_hkp_megakernel_lock.py`, 11): sha mismatch, extra object, missing object, stale KDP, malformed lock, a bundle that is not pulled, and a sub-build whose pack step fails on a corrupt object with the family ON and packs with it OFF.
- GPU integration test on gfx942: a stand-in object with the same ABI goes through `Graph.custom_op` and execute. A real megakernel needs model weights, thus the test uses a stand-in.
- ASAN: an ON build, unit tests and GPU integration passed on head `6285fe01629` (develop `9e91949f8f8`). ASAN was not rerun after this rebase.
- clang-format 18.1.3 and clang-tidy 20.1.2 (provider `.clang-tidy`, warnings as errors) on the changed C++.
- Anonymous `dvc pull` of the descriptors folder with DVC 3.62.0, as in CI.
- Local pre-commit passed on all changed files on both branches. The 11 lock tests also passed locally with CMake 4.4.4.

Build: `cmake --preset hip-kernel-provider -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON -DHIPDNN_ENABLE_SDPA=ON -DGPU_TARGETS=gfx942 -DENABLE_MEGAKERNEL_ENGINE=OFF|ON`. The image is `hipdnn_latest_gfx942.sqsh`. `HIPDNN_ENABLE_SDPA=ON` matches CI because develop does not build the frontend tests without it. Code objects came from a local copy of the DVC output. The PR head is `3bb8d27b9db` on develop `8e0433e4829`. Script: [run-review.sh](https://github.com/SamuelReeder/megakernels/blob/6e3b333327eaf8eabd73c81cae9d1630b20a966d/docs/hipdnn/engine/scripts/run-review.sh).

The [gfx942 run log][run] lists the node, tool versions, and exit codes. All [rebase logs](https://github.com/SamuelReeder/megakernels/tree/0b5abcb6b1081753396e9bca3ae979240d5a11f8/docs/hipdnn/engine/logs/rebase-3bb8d27b9db) are pinned to the evidence commit.

## Testing Checklist

- [x] Build, option OFF - Passed ([build-OFF.log][build-off]).
- [x] Build, option ON - Passed. The pack verified the lock and packed the gfx942 KDP ([build-ON.log][build-on], [run log][run]).
- [x] Build, option ON with no code objects - Passed. The lock check passed and the family packed no kernels ([build-ON-noobj.log][build-noobj], [run log][run]).
- [x] Provider unit tests, option OFF - 2530 passed, 13 skipped ([unit-OFF.log][unit-off]); no megakernel tests ran.
- [x] Provider unit tests, option ON - 2545 passed, 13 skipped ([unit-ON.log][unit-on]); [15 megakernel cases passed][mk-on].
- [x] Provider unit tests, option ON with no code objects - 2545 passed, 13 skipped ([unit-ON-noobj.log][unit-noobj]); [15 megakernel cases passed][mk-noobj].
- [x] Lock tests - 11 passed on [Alola][lock-gpu] and [locally][lock-local].
- [x] GPU integration, gfx942 - 2 `IntegrationGpuMegakernel.*` tests passed ([integration log][integration]).
- [x] ASAN, option ON, gfx942, previous head 6285fe01629 (`hipdnn_latest_gfx942_asan.sqsh`, `BUILD_ADDRESS_SANITIZER=ON`, `HSA_XNACK=1` as ctest sets it) - `hip_kernel_provider_tests` and `IntegrationGpuMegakernel.*` - Status: Passed, 1884 passed, 11 skipped; 2 passed; no sanitizer report ([run.log](https://github.com/SamuelReeder/megakernels/blob/6e3b333327eaf8eabd73c81cae9d1630b20a966d/docs/hipdnn/engine/logs/review-6285fe0/asan/run.log), [unit-ON.log](https://github.com/SamuelReeder/megakernels/blob/6e3b333327eaf8eabd73c81cae9d1630b20a966d/docs/hipdnn/engine/logs/review-6285fe0/asan/unit-ON.log), [integration-gpu-megakernel-ON.log](https://github.com/SamuelReeder/megakernels/blob/6e3b333327eaf8eabd73c81cae9d1630b20a966d/docs/hipdnn/engine/logs/review-6285fe0/asan/integration-gpu-megakernel-ON.log))
- [x] clang-format 18.1.3 on nine changed C++ files - Passed ([format log][format]).
- [x] clang-tidy 20.1.2 with warnings as errors on five changed translation units, option ON - Passed ([tidy log][tidy]).
- [x] Changed-file pre-commit - Status: Passed on the [PR branch](https://github.com/SamuelReeder/megakernels/blob/24f93bf4509b0ee8dab9148e326f3691103a17aa/docs/hipdnn/engine/logs/rebase-3bb8d27b9db/pre-commit.log) and [DVC branch](https://github.com/SamuelReeder/megakernels/blob/24f93bf4509b0ee8dab9148e326f3691103a17aa/docs/hipdnn/engine/logs/rebase-3bb8d27b9db/pre-commit-dvc.log).
- [ ] DVC upload of `megakernel/artifacts` to the `ingestor` remote - Status: Pending (needs write access)
- [ ] gfx942 ON build with the objects pulled from the `ingestor` remote, then land the `-dvc` commit - Status: Pending (after the upload)
- [ ] PR CI - GitHub PR checks - Status: Pending

[run]: https://github.com/SamuelReeder/megakernels/blob/0b5abcb6b1081753396e9bca3ae979240d5a11f8/docs/hipdnn/engine/logs/rebase-3bb8d27b9db/run.log
[build-off]: https://github.com/SamuelReeder/megakernels/blob/0b5abcb6b1081753396e9bca3ae979240d5a11f8/docs/hipdnn/engine/logs/rebase-3bb8d27b9db/build-OFF.log
[build-on]: https://github.com/SamuelReeder/megakernels/blob/0b5abcb6b1081753396e9bca3ae979240d5a11f8/docs/hipdnn/engine/logs/rebase-3bb8d27b9db/build-ON.log
[build-noobj]: https://github.com/SamuelReeder/megakernels/blob/0b5abcb6b1081753396e9bca3ae979240d5a11f8/docs/hipdnn/engine/logs/rebase-3bb8d27b9db/build-ON-noobj.log
[unit-off]: https://github.com/SamuelReeder/megakernels/blob/0b5abcb6b1081753396e9bca3ae979240d5a11f8/docs/hipdnn/engine/logs/rebase-3bb8d27b9db/unit-OFF.log
[unit-on]: https://github.com/SamuelReeder/megakernels/blob/0b5abcb6b1081753396e9bca3ae979240d5a11f8/docs/hipdnn/engine/logs/rebase-3bb8d27b9db/unit-ON.log
[unit-noobj]: https://github.com/SamuelReeder/megakernels/blob/0b5abcb6b1081753396e9bca3ae979240d5a11f8/docs/hipdnn/engine/logs/rebase-3bb8d27b9db/unit-ON-noobj.log
[mk-on]: https://github.com/SamuelReeder/megakernels/blob/0b5abcb6b1081753396e9bca3ae979240d5a11f8/docs/hipdnn/engine/logs/rebase-3bb8d27b9db/unit-megakernel-ON.log
[mk-noobj]: https://github.com/SamuelReeder/megakernels/blob/0b5abcb6b1081753396e9bca3ae979240d5a11f8/docs/hipdnn/engine/logs/rebase-3bb8d27b9db/unit-megakernel-ON-noobj.log
[lock-gpu]: https://github.com/SamuelReeder/megakernels/blob/0b5abcb6b1081753396e9bca3ae979240d5a11f8/docs/hipdnn/engine/logs/rebase-3bb8d27b9db/pytest-lock.log
[lock-local]: https://github.com/SamuelReeder/megakernels/blob/0b5abcb6b1081753396e9bca3ae979240d5a11f8/docs/hipdnn/engine/logs/rebase-3bb8d27b9db/pytest-lock-local.log
[integration]: https://github.com/SamuelReeder/megakernels/blob/0b5abcb6b1081753396e9bca3ae979240d5a11f8/docs/hipdnn/engine/logs/rebase-3bb8d27b9db/integration-gpu-megakernel-ON.log
[format]: https://github.com/SamuelReeder/megakernels/blob/0b5abcb6b1081753396e9bca3ae979240d5a11f8/docs/hipdnn/engine/logs/rebase-3bb8d27b9db/clang-format.log
[tidy]: https://github.com/SamuelReeder/megakernels/blob/0b5abcb6b1081753396e9bca3ae979240d5a11f8/docs/hipdnn/engine/logs/rebase-3bb8d27b9db/clang-tidy-ON.log

## Technical Changes

- `ENABLE_MEGAKERNEL_ENGINE` option (default OFF; requires `HIPDNN_ENABLE_KERNEL_INGESTOR=ON`). The same name is the `HKP_DESCRIPTOR_FAMILIES` entry and the compile definition that builds the native pack.
- `packs/MegakernelNative.cpp` registers `hipkernel:Megakernel`.
  - Matcher: a CUSTOM_OP node with an op id prefix of `megakernel.` and a JSON payload (schema 1). The key fields `model`, `src`, `arch`, `cus`, `kv`, `block`, `M`, `tp`, `rt` and `lm_head` must equal the descriptor. A decline names the field that differs. `cus` is also checked against the device. `nlayers` must be a positive i32, and the node must have exactly two inputs `[tab, glob]`.
  - Dispatcher: `hipModuleLaunchKernel` with grid = `cus` and the arguments `tab`, `glob`, `nlayers` on the handle stream. No workspace.
  - Co-residency: the kernel spins on a grid barrier, so every workgroup must be resident. `prepare()` declines (NOT_APPLICABLE) when the device CU count is not `cus` or no workgroup fits per CU. `launch()` does the same check for the device of the handle's stream, once per device and plan, before it launches. Thus a plan does not launch on a same-arch device with fewer CUs (for example a CPX partition). A failed HIP query is INTERNAL_ERROR, not NOT_APPLICABLE. The rule is in `packs/MegakernelCoResidency.hpp`, which has no HIP dependency, so unit tests check it without a device.
- `Kernel::function()` accessor, so that the pack can query the occupancy of the loaded function.
- `descriptors/megakernel/`: KMD, UED, UDD, UMD and the lock `manifest.json` (16 gfx942 code objects, megakernels `common/results/co/`, manifest sha256 5b83b31430d7). The DVC pointer is on the `-dvc` branch.
- `descriptor-packaging/tools/hkp_megakernel_lock.py`: the pack step runs `verify --lock` before it packs. The step is skipped when the family is excluded. With no `artifacts/`, `verify` passes and the family packs no kernel.
- Tests: `TestMegakernelMatchers.cpp`, `test_hkp_megakernel_lock.py`, `IntegrationGpuMegakernel.cpp` with its stand-in descriptors, and the megakernel dispatch symbol in `TestKernelIngestorEngine`.

### DVC upload (with the `-dvc` commit)

The objects were pushed and pulled only through a local DVC remote. The upload to the shared `ingestor` remote needs write access to its bucket. With write credentials in the environment (not in a config file), on `users/sareeder/megakernel-engine-dvc`, from the repository root:

```
dvc remote modify --local ingestor allow_anonymous_login false
dvc push -r ingestor dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/megakernel/artifacts.dvc
```

The first command is necessary: without it, the push fails with AccessDenied (see `descriptors/README.md`). Then, with no credentials, do `dvc pull -r ingestor` of the same pointer and run `python3 dnn-providers/hip-kernel-provider/descriptor-packaging/tools/hkp_megakernel_lock.py verify --lock dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/megakernel/manifest.json`.

### FUTURE WORK

- After the upload, set `-DENABLE_MEGAKERNEL_ENGINE=ON` in the HipDNN Linux Superbuild lane (gfx942). Then CI ASAN, CI clang-tidy and `IntegrationGpuMegakernel` cover the family.
