## Summary

This change adds a `megakernel` descriptor family to the kernel ingestor engine of hip-kernel-provider. The family lets hipDNN run precompiled persistent decode megakernels through a CUSTOM_OP node. The `ENABLE_MEGAKERNEL_ENGINE` CMake option controls the family. The default is OFF.

JIRA ID : AICK-1847

> Placeholder. No Jira ticket for this work was found (search `text ~ "megakernel"`, 2026-11-12). AICK-1847 comes from #14255. Replace it with the correct ticket before you open the PR.

## Risk Assessment

Risk 2 of 5. With the option OFF (the default), the product changes only in three places: the `HKP_DESCRIPTOR_FAMILIES` table, the configure-time DVC check (one more producer in the loop), and one test list under `#ifdef ENABLE_MEGAKERNEL_ENGINE`. All other new code compiles only with the option ON. With the option ON, the pack step checks the sha256 and size of each code object against the committed lock. A mismatch stops the build.

One CI risk is known: both lanes of `hipdnn-superbuild-ci.yml` run `dvc pull --recursive dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors`. That command also reads the new `megakernel/artifacts.dvc` pointer. The objects are not in the `ingestor` remote yet (see "DVC upload"). Thus PR CI can fail at that step until the upload is done. [INFERENCE: not run in CI.]

## ASIC Coverage

The change is arch-scoped: the code objects and the lock are for gfx942 only. The matcher declines a payload whose `arch` is not the device arch.

CI jobs that build hip-kernel-provider (`.github/workflows/hipdnn-superbuild-ci.yml`, preset `hipdnn-dev-all`, `HIPDNN_ENABLE_KERNEL_INGESTOR=ON`):

| Job | GPU_TARGETS | Megakernel family |
|---|---|---|
| HipDNN Linux Superbuild | gfx942 | Not built. The job does not set `ENABLE_MEGAKERNEL_ENGINE` (default OFF). |
| HipDNN Windows Superbuild | gfx1151 | Not built, for the same reason. |

Thus PR CI covers only the OFF build on gfx942 and gfx1151. The ON build and the GPU test ran only on the Alola gfx942 run below (MI300X node `banff-cyxtera-s70-4`). No other family was built or tested with the option ON. `therock-multi-arch-ci.yml` (default families gfx94X, gfx950, gfx125X) also runs on PRs. This work did not confirm that it builds hip-kernel-provider.

Before merge, one gfx942 run of the ON build with the objects from the `ingestor` remote (not from a local copy) is necessary.

## Testing Summary

- Unit tests (`hip_kernel_provider_tests`) with the option OFF and ON, on develop 9e91949f8f8 plus this branch.
- Lock tests (`test_hkp_megakernel_lock.py`): sha mismatch, extra object, missing object, stale KDP, malformed lock, a bundle that is not pulled, and a sub-build whose pack step fails on a corrupt object with the family ON and packs with it OFF.
- GPU integration test on gfx942: a stand-in object with the same ABI goes through `Graph.custom_op` and execute. A real megakernel needs model weights, thus the test uses a stand-in.
- clang-format 18.1.3 and clang-tidy 20.1.2 (provider `.clang-tidy`, warnings as errors) on the changed C++.

Build: `cmake --preset hip-kernel-provider -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON -DHIPDNN_ENABLE_SDPA=ON -DGPU_TARGETS=gfx942 -DENABLE_MEGAKERNEL_ENGINE=OFF|ON`, image `hipdnn_latest_gfx942.sqsh`. `HIPDNN_ENABLE_SDPA=ON` is set as in CI: develop 9e91949f8f8 does not build the frontend tests without it. The code objects came from a local copy of the DVC output, because the `ingestor` remote does not have them. Script: [`scripts/run-rebase.sh`](scripts/run-rebase.sh). Logs: [`logs/rebase-ee1c33b/`](logs/rebase-ee1c33b/).

## Testing Checklist

- [x] Build, option OFF - `cmake --build` - Status: Passed ([build-OFF.log](logs/rebase-ee1c33b/build-OFF.log))
- [x] Build, option ON, pack step checks the lock and packs the gfx942 KDP - `cmake --build` - Status: Passed ([build-ON.log](logs/rebase-ee1c33b/build-ON.log), [run.log](logs/rebase-ee1c33b/run.log))
- [x] Unit tests, option OFF - `hip_kernel_provider_tests` - Status: Passed, 1869 passed, 11 skipped ([unit-OFF.log](logs/rebase-ee1c33b/unit-OFF.log))
- [x] Unit tests, option ON - `hip_kernel_provider_tests` - Status: Passed, 1879 passed, 11 skipped ([unit-ON.log](logs/rebase-ee1c33b/unit-ON.log), megakernel cases: [unit-megakernel-ON.log](logs/rebase-ee1c33b/unit-megakernel-ON.log))
- [x] Lock tests - `python3 -m pytest tests/test_hkp_megakernel_lock.py` - Status: Passed, 11 passed ([pytest-lock.log](logs/rebase-ee1c33b/pytest-lock.log))
- [x] GPU integration, gfx942 - `hip_kernel_provider_integration_tests --gtest_filter='IntegrationGpuMegakernel.*'` - Status: Passed, 2 passed ([integration-gpu-megakernel-ON.log](logs/rebase-ee1c33b/integration-gpu-megakernel-ON.log))
- [x] clang-format 18 - `clang-format --dry-run --Werror` on 8 changed C++ files - Status: Passed ([clang-format.log](logs/rebase-ee1c33b/clang-format.log), empty)
- [x] clang-tidy 20 - `clang-tidy --warnings-as-errors='*'` on 5 changed TUs, option ON - Status: Passed ([clang-tidy-ON.log](logs/rebase-ee1c33b/clang-tidy-ON.log))
- [ ] DVC upload of `megakernel/artifacts` to the `ingestor` remote - Status: Pending (needs write access)
- [ ] gfx942 ON build with the objects from the `ingestor` remote - Status: Pending (after the upload)
- [ ] PR CI - GitHub PR checks - Status: Pending

## Technical Changes

- `ENABLE_MEGAKERNEL_ENGINE` option (default OFF; requires `HIPDNN_ENABLE_KERNEL_INGESTOR=ON`). The same name is the `HKP_DESCRIPTOR_FAMILIES` entry and the compile definition that builds the native pack.
- `packs/MegakernelNative.cpp` registers `hipkernel:Megakernel`.
  - Matcher: a CUSTOM_OP node with an op id prefix of `megakernel.` and a JSON payload (schema 1). The key fields `model`, `src`, `arch`, `cus`, `kv`, `block`, `M`, `tp`, `rt` and `lm_head` must equal the descriptor. A decline names the field that differs. `cus` is also checked against the device.
  - Dispatcher: an occupancy check at prepare (decline NOT_APPLICABLE when no block fits). The launch is `hipModuleLaunchKernel` with grid = `cus` and the arguments `tab`, `glob`, `nlayers` on the handle stream. No workspace.
- `Kernel::function()` accessor, so that the pack can query the occupancy of the loaded function.
- `descriptors/megakernel/`: KMD, UED, UDD, UMD, the lock `manifest.json` (16 gfx942 code objects, megakernels `common/results/co/`, manifest sha256 5b83b31430d7), and the DVC pointer `artifacts.dvc`.
- `descriptor-packaging/tools/hkp_megakernel_lock.py`: the pack step runs `verify --lock` before it packs. The step is skipped when the family is excluded.
- The configure-time DVC check also covers the megakernel folder. When an output is missing, the megakernel bundle is left out of the product pack (`HKP_PRODUCT_EXCLUDED_BUNDLES`), as for rocKE bundles.
- Tests: `TestMegakernelMatchers.cpp`, `test_hkp_megakernel_lock.py`, `IntegrationGpuMegakernel.cpp` with its stand-in descriptors, and the megakernel dispatch symbol in `TestKernelIngestorEngine`.

### DVC upload

The objects were pushed and pulled only through a local DVC remote. The upload to the shared `ingestor` remote needs write access to its bucket. With write credentials in the environment (not in a config file), from the repository root:

```
dvc push -r ingestor dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/megakernel/artifacts.dvc
```

Then do an anonymous `dvc pull -r ingestor` of the same pointer and run `python3 dnn-providers/hip-kernel-provider/descriptor-packaging/tools/hkp_megakernel_lock.py verify --lock dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/megakernel/manifest.json`.
