#!/bin/bash
# Megakernel engine rebuild (head ee1c33bed13 on develop 9e91949f8f8): build hipDNN +
# hip-kernel-provider (ingestor ON) with ENABLE_MEGAKERNEL_ENGINE OFF and ON, unit tests,
# lock pytest, clang-format and clang-tidy on the changed C++, GPU integration test.
# HIPDNN_ENABLE_SDPA=ON as in hipdnn-superbuild-ci.yml: develop 9e91949f8f8 does not build the
# frontend tests without it (TestSdpaFwdNode.cpp calls Graph::sdpa).
# Runs inside hipdnn_latest_gfx942.sqsh on a gfx942 node; /tmp is node-local.
set -uo pipefail
export PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
LOG=~/work/mkeng2/logs; rm -rf $LOG; mkdir -p $LOG
exec > >(tee $LOG/run.log) 2>&1
echo "host=$(hostname) nproc=$(nproc) date=$(date -Is)"
echo "gpu arch=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+')"
python3 -c "import zstandard, msgpack, pytest; print('py deps ok')"
W=/tmp/mkeng2; rm -rf $W; mkdir -p $W/src; cd $W
tar xzf ~/work/mkeng2/mk-src2.tgz -C src
P=src/dnn-providers/hip-kernel-provider
D=$P/src/engines/kernel_ingestor_engine/descriptors/megakernel
tar xzf ~/work/mkeng2/mk-art2.tgz -C $D   # stands in for `dvc pull` (no ingestor bucket access)
cmake --version | head -1; hipcc --version 2>/dev/null | grep -i "HIP version"
for MODE in OFF ON; do
  B=$W/build-$MODE
  t0=$(date +%s)
  cmake --fresh --preset hip-kernel-provider -S src -B $B -G Ninja \
    -DROCM_LIBS_ENABLE_COMPONENTS="hipdnn;hipdnn-integration-tests;hip-kernel-provider" \
    -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON -DHIPDNN_ENABLE_SDPA=ON -DGPU_TARGETS=gfx942 \
    -DENABLE_CLANG_FORMAT=ON -DENABLE_CLANG_TIDY=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DENABLE_MEGAKERNEL_ENGINE=$MODE -DPython3_EXECUTABLE=/usr/bin/python3 \
    -DCMAKE_INSTALL_PREFIX=$W/install-$MODE > $LOG/configure-$MODE.log 2>&1
  echo "MODE=$MODE configure_rc=$? s=$(( $(date +%s)-t0 ))"
  grep -iE "megakernel|DVC output" $LOG/configure-$MODE.log | head
  t0=$(date +%s)
  cmake --build $B -j $(nproc) > $LOG/build-$MODE.log 2>&1; rc=$?
  echo "MODE=$MODE build_rc=$rc s=$(( $(date +%s)-t0 ))"
  grep -nE "hkp megakernel lock|megakernel" $LOG/build-$MODE.log | head -10
  if [ $rc -ne 0 ]; then grep -nE "error|FAILED" $LOG/build-$MODE.log | head -40; continue; fi
  find $B -path "*megakernel*" -name "*.kdp.json" | head
  ( cd $B/bin && ./hip_kernel_provider_tests --gtest_brief=1 > $LOG/unit-$MODE.log 2>&1; echo "MODE=$MODE unit_rc=$?" )
  grep -E "^\[  (PASSED|FAILED|SKIPPED) |tests ran" $LOG/unit-$MODE.log | head
  ( cd $B/bin && ./hip_kernel_provider_tests --gtest_filter='TestMegakernel*:*EveryHandler*' > $LOG/unit-megakernel-$MODE.log 2>&1; echo "MODE=$MODE unit_megakernel_rc=$?" )
  grep -E "^\[  (PASSED|FAILED|SKIPPED) |tests ran" $LOG/unit-megakernel-$MODE.log | head
done
CXX_CHANGED="src/engines/kernel_ingestor_engine/packs/MegakernelNative.cpp
src/engines/kernel_ingestor_engine/IngestorPacks.cpp
src/engines/kernel_ingestor_engine/IngestorPacks.hpp
src/compilation/Kernel.hpp
src/tests/engines/kernel_ingestor_engine/packs/TestMegakernelMatchers.cpp
src/tests/engines/kernel_ingestor_engine/TestKernelIngestorEngine.cpp
src/integration_tests/kernel_ingestor_engine/IntegrationGpuMegakernel.cpp
src/engines/kernel_ingestor_engine/test_descriptors/integration/megakernel/kernels/MegakernelStandIn.cpp"
CF=$(command -v clang-format-18 || command -v clang-format); echo "clang-format: $($CF --version)"
( cd $P && $CF --style=file --dry-run --Werror $CXX_CHANGED ) > $LOG/clang-format.log 2>&1
echo "clang_format_rc=$?"; head -20 $LOG/clang-format.log
CT=$(command -v clang-tidy-20 || command -v clang-tidy); echo "clang-tidy: $($CT --version | grep version)"
( cd $P && $CT -p $W/build-ON --config-file=.clang-tidy --warnings-as-errors='*' \
    src/engines/kernel_ingestor_engine/packs/MegakernelNative.cpp \
    src/engines/kernel_ingestor_engine/IngestorPacks.cpp \
    src/tests/engines/kernel_ingestor_engine/packs/TestMegakernelMatchers.cpp \
    src/tests/engines/kernel_ingestor_engine/TestKernelIngestorEngine.cpp \
    src/integration_tests/kernel_ingestor_engine/IntegrationGpuMegakernel.cpp ) > $LOG/clang-tidy-ON.log 2>&1
echo "clang_tidy_rc=$?"; grep -E "error:|warning:" $LOG/clang-tidy-ON.log | head -20
( cd $W/build-ON/bin && HIP_VISIBLE_DEVICES=0 ./hip_kernel_provider_integration_tests --gtest_filter='IntegrationGpuMegakernel.*' > $LOG/integration-gpu-megakernel-ON.log 2>&1; echo "integration_rc=$?" )
grep -E "^\[ *(OK|PASSED|FAILED|SKIPPED|RUN) |tests ran" $LOG/integration-gpu-megakernel-ON.log | head -20
( cd $P/descriptor-packaging && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -v -p no:cacheprovider tests/test_hkp_megakernel_lock.py > $LOG/pytest-lock.log 2>&1; echo "pytest_rc=$?" )
tail -3 $LOG/pytest-lock.log
cd /; rm -rf $W
echo "done $(date -Is)"
