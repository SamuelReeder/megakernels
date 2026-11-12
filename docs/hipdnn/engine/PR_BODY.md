## Summary

This change adds a `megakernel` descriptor family to the kernel ingestor engine of hip-kernel-provider. The family lets hipDNN run precompiled persistent decode megakernels through a CUSTOM_OP node. The `ENABLE_MEGAKERNEL_ENGINE` CMake option controls the family. The default is OFF.

JIRA ID : AICK-1847

> PLACEHOLDER. This is not the ticket for this work. No Jira ticket for this work was found (search `text ~ "megakernel"`, 2026-11-12). AICK-1847 comes from #14255. The owner will supply the correct ticket.

## Risk Assessment

Risk 2 of 5. With the option OFF (the default), these parts of the product change:

- `HKP_DESCRIPTOR_FAMILIES` gets the entry `megakernel=ENABLE_MEGAKERNEL_ENGINE`. The pack excludes the `megakernel/` folder when the option is OFF.
- `HkpPackaging.cmake` gets the lock wiring: `HKP_MEGAKERNEL_LOCK_TOOL` and a `verify` command before the pack step. The command is not added when the family is excluded.
- `Kernel::function()`, a new accessor on `compilation::Kernel`. Only the megakernel pack calls it.
- `MEGAKERNEL_ENGINE_NAME`, a new constant in `IngestorPacks.hpp`.
- `TestKernelIngestorEngine` gets one symbol under `#ifdef ENABLE_MEGAKERNEL_ENGINE`.

All other new C++ compiles only with the option ON. With the option ON, the pack step checks the sha256 and size of each code object against the committed lock. A mismatch stops the build.

DVC: the code objects are not in the `ingestor` remote. An anonymous GET of the pointer's `.dir` object returns 404. We do not have write access to the bucket. Thus this PR does not contain the pointer `descriptors/megakernel/artifacts.dvc`. The pointer and the configure-time DVC check for the folder are on `users/sareeder/megakernel-engine-dvc`, one commit on this branch. That commit must land together with the upload, not before. These lanes would fail if the pointer landed before the upload:

- `hipdnn-superbuild-ci.yml`, both lanes: `dvc pull --recursive dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors` (Windows lines 230-231, Linux lines 518-519).
- Bare `dvc pull` of the whole repository: `component-ci-miopen.yml:172`, `hipblaslt-asan-ci.yml:176`, `miopen-wrapper-flag-on.yml:64`, and TheRock `fetch_sources`.

On this branch, an anonymous `dvc pull --recursive` of the descriptors folder (DVC 3.62.0, as in CI) passes. On the `-dvc` branch it fails on the megakernel pointer, as expected (`logs/review-6285fe0/dvc-pull-anonymous.log`).

## ASIC Coverage

The change is arch-scoped: the code objects and the lock are for gfx942 only. The matcher declines a payload whose `arch` is not the device arch.

CI jobs that build hip-kernel-provider (`.github/workflows/hipdnn-superbuild-ci.yml`, preset `hipdnn-dev-all`, `HIPDNN_ENABLE_KERNEL_INGESTOR=ON`):

| Job | GPU_TARGETS | Megakernel family |
|---|---|---|
| HipDNN Linux Superbuild | gfx942 | Not built. The job does not set `ENABLE_MEGAKERNEL_ENGINE` (default OFF). |
| HipDNN Windows Superbuild | gfx1151 | Not built, for the same reason. |

Thus PR CI covers only the OFF build, on gfx942 and gfx1151. CI ASAN and CI clang-tidy do not see the megakernel code. The ON build, the ASAN ON build and the GPU test ran only on Alola gfx942 (MI300X node `banff-cyxtera-s70-4`). No other family was built or tested with the option ON. `therock-multi-arch-ci.yml` (default families gfx94X, gfx950, gfx125X) also runs on PRs. This work did not confirm that it builds hip-kernel-provider.

Before the `-dvc` commit lands, one gfx942 ON run with the objects pulled from the `ingestor` remote is necessary.

## Testing Summary

- Unit tests (`hip_kernel_provider_tests`) with the option OFF, ON, and ON with no code objects.
- Megakernel cases: 15 with the option ON. 12 are matcher cases: payload, key fields, device CU count, `nlayers` bounds (0, -1, INT32_MAX + 1; INT32_MAX passes), input count (0, 1, 3). 3 are co-residency cases: another CU count, no workgroup per CU, accepted. With the option OFF, none of them compile.
- Lock tests (`test_hkp_megakernel_lock.py`, 11): sha mismatch, extra object, missing object, stale KDP, malformed lock, a bundle that is not pulled, and a sub-build whose pack step fails on a corrupt object with the family ON and packs with it OFF.
- GPU integration test on gfx942: a stand-in object with the same ABI goes through `Graph.custom_op` and execute. A real megakernel needs model weights, thus the test uses a stand-in.
- ASAN: an ON build in the ASAN image, with the unit tests and the GPU test.
- clang-format 18.1.3 and clang-tidy 20.1.2 (provider `.clang-tidy`, warnings as errors) on the changed C++.
- Anonymous `dvc pull` of the descriptors folder with DVC 3.62.0, as in CI.

Build: `cmake --preset hip-kernel-provider -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON -DHIPDNN_ENABLE_SDPA=ON -DGPU_TARGETS=gfx942 -DENABLE_MEGAKERNEL_ENGINE=OFF|ON`, image `hipdnn_latest_gfx942.sqsh` (ASAN: `hipdnn_latest_gfx942_asan.sqsh`, `-DBUILD_ADDRESS_SANITIZER=ON`). `HIPDNN_ENABLE_SDPA=ON` is set as in CI: develop 9e91949f8f8 does not build the frontend tests without it. The code objects came from a local copy of the DVC output. Script: [`scripts/run-review.sh`](scripts/run-review.sh). Logs: [`logs/review-6285fe0/`](logs/review-6285fe0/). Each log starts with the head it ran on.

## Testing Checklist

- [x] Build, option OFF - `cmake --build` - Status: Passed ([build-OFF.log](logs/review-6285fe0/main/build-OFF.log))
- [x] Build, option ON, pack step checks the lock and packs the gfx942 KDP - `cmake --build` - Status: Passed ([build-ON.log](logs/review-6285fe0/main/build-ON.log), [run.log](logs/review-6285fe0/main/run.log))
- [x] Build, option ON with no code objects: lock check passes, family packs nothing - `cmake --build` - Status: Passed ([build-ON-noobj.log](logs/review-6285fe0/main/build-ON-noobj.log), [run.log](logs/review-6285fe0/main/run.log))
- [x] Unit tests, option OFF - `hip_kernel_provider_tests` - Status: Passed, 1869 passed, 11 skipped ([unit-OFF.log](logs/review-6285fe0/main/unit-OFF.log))
- [x] Unit tests, option ON - `hip_kernel_provider_tests` - Status: Passed, 1884 passed, 11 skipped ([unit-ON.log](logs/review-6285fe0/main/unit-ON.log); megakernel cases 15 passed: [unit-megakernel-ON.log](logs/review-6285fe0/main/unit-megakernel-ON.log))
- [x] Unit tests, option ON, no code objects - `hip_kernel_provider_tests` - Status: Passed, 1884 passed, 11 skipped ([unit-ON-noobj.log](logs/review-6285fe0/main/unit-ON-noobj.log))
- [x] Lock tests - `python3 -m pytest tests/test_hkp_megakernel_lock.py` - Status: Passed, 11 passed ([pytest-lock.log](logs/review-6285fe0/main/pytest-lock.log))
- [x] GPU integration, gfx942 - `hip_kernel_provider_integration_tests --gtest_filter='IntegrationGpuMegakernel.*'` - Status: Passed, 2 passed ([integration-gpu-megakernel-ON.log](logs/review-6285fe0/main/integration-gpu-megakernel-ON.log))
- [x] ASAN, option ON, gfx942 (`hipdnn_latest_gfx942_asan.sqsh`, `BUILD_ADDRESS_SANITIZER=ON`, `HSA_XNACK=1` as ctest sets it) - `hip_kernel_provider_tests` and `IntegrationGpuMegakernel.*` - Status: Passed, 1884 passed, 11 skipped; 2 passed; no sanitizer report ([run.log](logs/review-6285fe0/asan/run.log), [unit-ON.log](logs/review-6285fe0/asan/unit-ON.log), [integration-gpu-megakernel-ON.log](logs/review-6285fe0/asan/integration-gpu-megakernel-ON.log))
- [x] clang-format 18 - `clang-format --dry-run --Werror` on 9 changed C++ files - Status: Passed ([clang-format.log](logs/review-6285fe0/main/clang-format.log))
- [x] clang-tidy 20 - `clang-tidy --warnings-as-errors='*'` on 5 changed TUs, option ON - Status: Passed ([clang-tidy-ON.log](logs/review-6285fe0/main/clang-tidy-ON.log))
- [x] Anonymous `dvc pull --recursive` of the descriptors folder, DVC 3.62.0 - Status: Passed on this branch; fails on the `-dvc` branch as expected ([dvc-pull-anonymous.log](logs/review-6285fe0/dvc-pull-anonymous.log))
- [ ] DVC upload of `megakernel/artifacts` to the `ingestor` remote - Status: Pending (needs write access)
- [ ] gfx942 ON build with the objects pulled from the `ingestor` remote, then land the `-dvc` commit - Status: Pending (after the upload)
- [ ] PR CI - GitHub PR checks - Status: Pending

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
