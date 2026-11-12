# Megakernel engine (ENABLE_MEGAKERNEL_ENGINE): build and test notes

Design: megakernels `docs/hipdnn-route.md` (sections 3-5, decisions D3, D5, D6, D7).
PR branch `users/sareeder/megakernel-engine` on `origin/develop` 9e91949f8f8 (3 commits, head
in "Current results" at the end). The DVC pointer is on `users/sareeder/megakernel-engine-dvc`,
one commit on top, to land with the upload. Branch history: 6a27c440ff9 base, old head
18c6ef46d60 (backup `-archive`), first rebase ee1c33bed13 (backup `-archive2`).

## What is here

| Part | Where |
|---|---|
| Option (default OFF), family entry, compile definition: one name `ENABLE_MEGAKERNEL_ENGINE` | `CMakeLists.txt`, `descriptor-packaging/cmake/HkpPackaging.cmake`, `src/engines/kernel_ingestor_engine/CMakeLists.txt` |
| Native UMD (graph + kernel scope) and UDD | `src/engines/kernel_ingestor_engine/packs/MegakernelNative.cpp` |
| Descriptors, lock (DVC pointer on the `-dvc` branch) | `src/engines/kernel_ingestor_engine/descriptors/megakernel/` |
| Lock check in the pack step | `descriptor-packaging/tools/hkp_megakernel_lock.py`, wired in `hkp_wire_pack_target` |
| Configure DVC check for the megakernel folder (`-dvc` branch only) | `CMakeLists.txt` |
| Co-residency rule (prepare and per launch device) | `packs/MegakernelCoResidency.hpp`, `packs/MegakernelNative.cpp` |
| Unit tests: matcher declines | `src/tests/engines/kernel_ingestor_engine/packs/TestMegakernelMatchers.cpp` |
| Tests: lock mismatch fails the tool and the build | `descriptor-packaging/tests/test_hkp_megakernel_lock.py` |
| GPU integration test (stand-in object, Graph.custom_op + execute) | `src/integration_tests/kernel_ingestor_engine/IntegrationGpuMegakernel.cpp`, `test_descriptors/integration/megakernel/` |

Payload (schema 1): `{schema, model, src, arch, cus, kv, block, M, tp, rt, lm_head, nlayers}`;
op id `megakernel.<model>`. Graph scope declines (INFO line
`hipkernel:Megakernel declined custom op '<id>' [<field>]: ...`) on: no/!object JSON
`[payload]`, schema != 1 `[schema]`, missing or mistyped key `[<key>]`, op id model differs
`[model]`, arch != device `[arch]`, cus != device multiProcessorCount `[cus]`, nlayers not a
positive i32 `[nlayers]`. Kernel scope compares model, cus, kv, block, M, tp, rt, lm_head, src
with the KMD record and logs the first differing field (`src` last; kernels of other models
are skipped silently). `prepare()` declines (NOT_APPLICABLE, `[occupancy]`) when
`hipModuleOccupancyMaxActiveBlocksPerMultiprocessor` < 1 for block_threads / lds_bytes. Launch:
grid (cus), block (block_threads), dynamic LDS lds_bytes (0 for FlyDSL objects; agreed with
CoExporter), args (tab addr i64, glob addr i64, nlayers i32) on the handle stream, workspace 0.

## Decisions taken here

- Lock: `manifest.json` is CoExporter's final set per design D4 (megakernels
  `users/sareeder/co-export` 0cafc63, `common/results/co/rt2/final/`, manifest sha256
  5b83b31430d7...): 16 objects, Qwen3.8 TP1 rt2, Qwen3.8 TP2 rt0, Qwen3.6 rt2, Gemma4 rt0,
  M 1, 2, 4, 8; `src` is vLLM's `compile_key`. Before: the rt0 set (1f9108f, e689d054dfca,
  built and tested below), and earlier the spike's four Qwen3.8 objects.
- The pack step skips the lock check when `artifacts/` is absent (no `dvc pull`): configure
  already warns, as for rocKE, and nothing is packed.
- `lm_head`, `cus`, ... are KMD fields of type bool/int, all `matcher_only_fields`.
- The GPU test uses a stand-in object with the same ABI, because a real megakernel needs
  model weights behind `tab`. Its engine is `hipkernel:megakernel_standin` (integration set
  only, packed only with the option ON).
- `Kernel::function()` accessor added so the pack can query occupancy of the loaded function.

## Results before the rebase (history: base 6a27c440ff9, Alola CPU node ctr-smc-s22-17)

These numbers are from the old base and are not current. See "Current results".

Log: `logs/run-off-on.log` (job 68218157, head a90fe00804e).

| Build | Build | Unit tests (`hip_kernel_provider_tests`) | Megakernel cases |
|---|---|---|---|
| OFF | rc 0 | 1109 passed, 68 skipped (no device), 0 failed (`logs/unit-OFF.log`) | 0 (not compiled) |
| ON | rc 0 | 1119 passed, 68 skipped, 0 failed (`logs/unit-ON.log`) | 10 passed (`logs/unit-megakernel-ON.log`) |

- OFF configure: `megakernel` listed as a disabled family folder (`logs/configure-OFF-hkp.log`);
  the OFF build differs from develop only in code under `#ifdef ENABLE_MEGAKERNEL_ENGINE`, the
  family table and the per-family DVC loop. Same pass/skip set as the ON build minus the 10
  megakernel cases.
- ON: product root packs `megakernel/artifacts/megakernel.kdp.json` for gfx942, integration
  root packs the stand-in (`logs/run-off-on.log`, `logs/build-ON-megakernel.log`).
- clang-tidy-20 (repo `.clang-tidy`, warnings as errors) on the 5 touched TUs: rc 0
  (`logs/tidy-ON.log`). A full-tree tidy build found 11 findings in earlier heads, all fixed
  (jobs 68216113, 68216970). clang-format 18 applied to every touched C++ file.
- Lock tests: 11 passed (`logs/pytest-lock.log`): sha mismatch, extra object, missing
  object, stale KDP, malformed lock, unpulled bundle passes, and a sub-build whose pack step
  fails on a corrupt object with the family ON and packs with it OFF.
- DVC: `artifacts.dvc` round trip through a LOCAL remote (`/tmp/mk-dvc-remote`, scratch DVC
  repo): push 6 files, delete artifacts and cache, pull, `hkp_megakernel_lock.py verify` OK;
  the pointer is byte-identical after regenerating the KDP. Nothing was pushed to `ingestor`.

With the exporter's 16-object rt0 lock (head af5a3559c2e, job 68219792,
`logs/run-on-rt0-set.log`): ON build rc 0, lock verify passes in the pack step, product KDP
packed for gfx942, 1119 passed / 68 skipped / 0 failed, clang-tidy rc 0, lock pytest 11 passed.

With the rt2/rt0 final lock (0cafc63): `hkp_megakernel_lock.py verify` passes and the KDP
was regenerated; the lock has the same entry shape and count as the rt0 set that built and
passed above, so the ON build was not rerun for it.

## Not done

- GPU integration test: done after the rebase (2 passed,
  `logs/rebase-ee1c33b/integration-gpu-megakernel-ON.log`).

## DVC upload, once write access to the ingestor bucket exists

From the repository root, with credentials in the environment (never in a config file):

```
dvc push -r ingestor dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/megakernel/artifacts.dvc
```

Then check anonymously that `dvc pull -r ingestor <same pointer>` restores the files and that
`python3 dnn-providers/hip-kernel-provider/descriptor-packaging/tools/hkp_megakernel_lock.py verify --lock <...>/megakernel/manifest.json` passes.

## Rebase onto develop 9e91949f8f8 (2026-11-12)

The branch was rebuilt as three linear commits on `origin/develop` 9e91949f8f8. Backup of the
old head 18c6ef46d60: `users/sareeder/megakernel-engine-archive`. This folder
(`megakernel-notes/`) moved here. The product diff is the same as the old head, except for the
conflict resolutions below (`logs/rebase-ee1c33b/diff-check.log`).

| File | Develop change since 6a27c440ff9 | Resolution |
|---|---|---|
| `CMakeLists.txt` (option text) | FlyDSL added to the `REQUIRE_DVC_DESCRIPTORS` text | Text names rocKE, FlyDSL and megakernel. |
| `CMakeLists.txt` (DVC check) | Loop over producers (rocKE, FlyDSL); a missing rocKE output adds its bundle to `HKP_PRODUCT_EXCLUDED_BUNDLES` | Keep develop's loop. Add `megakernel` as a producer. A missing megakernel output adds `megakernel` to `HKP_PRODUCT_EXCLUDED_BUNDLES`, so the pack step skips the bundle and its lock check. The old branch had its own per-family loop; it is gone. |
| `HkpPackaging.cmake` | `FlyDSL` family entry | Both entries. |
| engine `CMakeLists.txt` | FlyDSL pack sources | Both blocks. |
| `IngestorPacks.cpp/.hpp` | `hipkernel:Gfx950ConvDgradAot` pack | Both. |
| tests `CMakeLists.txt` | gfx942 unified 3D table generation | Both. |
| `TestKernelIngestorEngine.cpp` | Symbol list became a `std::vector` with an `#ifdef HIPDNN_ENGINE_FLYDSL` block | Keep develop's vector. Add `megakernel.dispatch` under `#ifdef ENABLE_MEGAKERNEL_ENGINE`. The old `std::array` with `sv` literals is gone. |

Results on head ee1c33bed13 (gfx942 node banff-cyxtera-s70-4, job 68378885, `logs/rebase-ee1c33b/run.log`):
OFF and ON builds rc 0; unit OFF 1869 passed / 11 skipped, ON 1879 passed / 11 skipped;
megakernel cases ON 11 passed; `IntegrationGpuMegakernel.*` 2 passed; lock pytest 11 passed;
clang-format and clang-tidy rc 0. The first try (job 68378708, `logs-try1` on Alola only)
failed in `hipdnn_frontend_tests` (`TestSdpaFwdNode.cpp`: no `Graph::sdpa`) because
`HIPDNN_ENABLE_SDPA` was OFF; develop does not build without it. CI sets it ON.

## Review changes (2026-11-12, PR head 6285fe01629, `-dvc` head caf017e6d2e)

Backup of the first rebase head ee1c33bed13: `users/sareeder/megakernel-engine-archive2`.

| Finding | Change | Test |
|---|---|---|
| BLOCKING: pointer 404 on `ingestor`, every `dvc pull` of the folder fails | `artifacts.dvc` and the megakernel DVC configure check moved to `users/sareeder/megakernel-engine-dvc` (one commit on the PR head). README and lock-tool text say so. | Anonymous `dvc pull --recursive` of the descriptors folder, DVC 3.62.0: PR branch rc 0, `-dvc` branch rc 1 on the megakernel pointer (`logs/review-6285fe0/dvc-pull-anonymous.log`). ON build with no objects: rc 0, no megakernel file packed, lock verify passes (`main/run.log`, `main/build-ON-noobj.log`). |
| IMPORTANT: launch device not re-checked | `launch()` resolves the device from the stream (`launchDeviceOrdinal`), then checks its CU count and occupancy once per device and plan before it launches. The rule is `packs/MegakernelCoResidency.hpp` (no HIP). | `TestMegakernelCoResidency.*` (3 cases). A second device was not available, so the launch path on another device is not exercised on hardware. |
| SUGGESTION: occupancy query error read as NOT_APPLICABLE | A failed HIP query is INTERNAL_ERROR. | Code review only (no fault injection). |
| SUGGESTION: tests | `nlayers` 0, -1, INT32_MAX + 1 decline, INT32_MAX passes; 0, 1, 3 inputs decline. | `TestMegakernelMatchers.*` (12 cases). |
| Coverage: ASAN | ON build in `hipdnn_latest_gfx942_asan.sqsh`. | See below. |

Current results (gfx942, `scripts/run-review.sh`, every log starts with the head):

- main image (job 68379300, `logs/review-6285fe0/main/`): OFF, ON and ON with no objects build rc 0.
  Unit OFF 1869 passed / 11 skipped; ON 1884 / 11; ON no objects 1884 / 11. Megakernel cases
  (`TestMegakernel*`) ON 15 passed: 12 matcher, 3 co-residency. `IntegrationGpuMegakernel.*`
  2 passed. Lock pytest 11 passed. clang-format 18 and clang-tidy 20 rc 0.
- ASAN image (job 68380455, `logs/review-6285fe0/asan/`): ON build rc 0; unit 1884 / 11;
  `IntegrationGpuMegakernel.*` 2 passed; no sanitizer report. The first ASAN run (job 68379807,
  `logs/review-6285fe0/asan-try1/run.log`) ran the binaries without `HSA_XNACK=1`, which ctest
  sets from `Sanitizers.cmake`; then `hipStreamCreate` failed and some HIP launches crashed in
  `libamdhip64`, in megakernel and other tests alike.

The `logs/rebase-ee1c33b/` logs do not record the head in the file; the folder name gives it.
`scripts/run-rebase.sh` is the version that ran (with `HIPDNN_ENABLE_SDPA=ON`).

## Rebase onto develop 11ae0945ecb (2026-11-12, PR head bf3255d71eb, `-dvc` head 4f484aef7d6)

Backup of 6285fe01629: `users/sareeder/megakernel-engine-archive3`. Develop added the
`hipkernel:Gfx1250ConvDgradAot` engine. One conflict: the pack table in `IngestorPacks.cpp`;
both entries kept (Gfx1250ConvDgradAot, then the megakernel entry under its `#ifdef`). The
other shared files (`IngestorPacks.hpp`, engine and test `CMakeLists.txt`,
`TestKernelIngestorEngine.cpp`, integration `CMakeLists.txt`, `descriptors/README.md`) merged
without conflict. The product +/- lines are the same as on 6285fe01629
(`logs/rebase-bf3255d/diff-check.log`, empty diff). The `-dvc` commit rebased without conflict.

Results (job 68380805, `logs/rebase-bf3255d/`): OFF, ON and ON with no objects build rc 0; unit
OFF 1879 passed / 13 skipped, ON 1894 / 13, ON no objects 1894 / 13; `TestMegakernel*` 15
passed; `IntegrationGpuMegakernel.*` 2 passed; lock pytest 11 passed; clang-format and
clang-tidy rc 0. Anonymous `dvc pull --recursive` (DVC 3.62.0): PR branch rc 0, `-dvc` branch
rc 1 on the megakernel pointer. ASAN not rerun: no megakernel code changed in the rebase.
