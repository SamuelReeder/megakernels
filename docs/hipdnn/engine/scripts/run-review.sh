#!/bin/bash
# Megakernel engine, PR branch users/sareeder/megakernel-engine on develop 9e91949f8f8.
# Builds hipDNN + hip-kernel-provider (ingestor ON, HIPDNN_ENABLE_SDPA=ON as in
# hipdnn-superbuild-ci.yml: develop does not build the frontend tests without it).
#   KIND=main (hipdnn_latest_gfx942.sqsh):
#     OFF, ON (objects from the local copy), ON-noobj (no artifacts/: the family packs no
#     kernel); hip_kernel_provider_tests for each; clang-format and clang-tidy on the
#     changed C++; IntegrationGpuMegakernel.* on the ON build; lock pytest.
#   KIND=asan (hipdnn_latest_gfx942_asan.sqsh): ON with BUILD_ADDRESS_SANITIZER=ON;
#     hip_kernel_provider_tests and IntegrationGpuMegakernel.*.
# /tmp is node-local. The source tarball carries HEAD_SHA.
set -uo pipefail
KIND=${KIND:-main}
export PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
LOG=~/work/mkeng3/logs-$KIND; rm -rf $LOG; mkdir -p $LOG
exec > >(tee $LOG/run.log) 2>&1
W=/tmp/mkeng3-$KIND; rm -rf $W; mkdir -p $W/src; cd $W
tar xzf ~/work/mkeng3/mk-src3.tgz -C src
echo "head=$(cat src/HEAD_SHA) kind=$KIND host=$(hostname) nproc=$(nproc) date=$(date -Is)"
echo "gpu arch=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+')"
python3 -c "import zstandard, msgpack, pytest; print('py deps ok')"
P=src/dnn-providers/hip-kernel-provider
D=$P/src/engines/kernel_ingestor_engine/descriptors/megakernel
cmake --version | head -1; hipcc --version 2>/dev/null | grep -i "HIP version"
stamp() { echo "head=$(cat $W/src/HEAD_SHA) kind=$KIND mode=$1 $(date -Is)"; }

if [ $KIND = asan ]; then MODES="ON"; EXTRA="-DBUILD_ADDRESS_SANITIZER=ON"; else MODES="OFF ON ON-noobj"; EXTRA=""; fi
for MODE in $MODES; do
  B=$W/build-$MODE
  rm -rf $D/artifacts
  [ $MODE = ON ] && tar xzf ~/work/mkeng3/mk-art3.tgz -C $D   # local copy; not in the ingestor remote
  OPT=${MODE%%-*}
  t0=$(date +%s)
  { stamp $MODE; cmake --fresh --preset hip-kernel-provider -S src -B $B -G Ninja \
    -DROCM_LIBS_ENABLE_COMPONENTS="hipdnn;hipdnn-integration-tests;hip-kernel-provider" \
    -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON -DHIPDNN_ENABLE_SDPA=ON -DGPU_TARGETS=gfx942 \
    -DENABLE_CLANG_FORMAT=ON -DENABLE_CLANG_TIDY=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DENABLE_MEGAKERNEL_ENGINE=$OPT -DPython3_EXECUTABLE=/usr/bin/python3 $EXTRA \
    -DCMAKE_INSTALL_PREFIX=$W/install-$MODE; } > $LOG/configure-$MODE.log 2>&1
  echo "MODE=$MODE configure_rc=$? s=$(( $(date +%s)-t0 ))"
  t0=$(date +%s)
  { stamp $MODE; cmake --build $B -j $(nproc); } > $LOG/build-$MODE.log 2>&1; rc=$?
  echo "MODE=$MODE build_rc=$rc s=$(( $(date +%s)-t0 ))"
  if [ $rc -ne 0 ]; then grep -nE "error|FAILED|hkp megakernel lock" $LOG/build-$MODE.log | head -40; continue; fi
  echo "MODE=$MODE megakernel files in the packed product root:"
  find $B/lib/hipdnn_plugins/engines/arch_content/hip-kernel-provider -path '*megakernel*' -type f | sed "s|$B/||" | sort | head -40
  ( cd $B/bin && { stamp $MODE; ./hip_kernel_provider_tests --gtest_brief=1; } > $LOG/unit-$MODE.log 2>&1; echo "MODE=$MODE unit_rc=$?" )
  grep -E "^\[  (PASSED|FAILED|SKIPPED) |tests ran" $LOG/unit-$MODE.log | head
  ( cd $B/bin && { stamp $MODE; ./hip_kernel_provider_tests --gtest_filter='TestMegakernel*'; } > $LOG/unit-megakernel-$MODE.log 2>&1; echo "MODE=$MODE unit_megakernel_rc=$?" )
  grep -E "^\[  (PASSED|FAILED|SKIPPED) |tests ran" $LOG/unit-megakernel-$MODE.log | head
  if [ $MODE = ON ]; then
    ( cd $B/bin && { stamp $MODE; HIP_VISIBLE_DEVICES=0 ./hip_kernel_provider_integration_tests --gtest_filter='IntegrationGpuMegakernel.*'; } > $LOG/integration-gpu-megakernel-$MODE.log 2>&1; echo "MODE=$MODE integration_rc=$?" )
    grep -E "^\[ *(OK|PASSED|FAILED|SKIPPED) |tests ran" $LOG/integration-gpu-megakernel-$MODE.log | head
  fi
done

if [ $KIND = main ]; then
  CXX_CHANGED="src/engines/kernel_ingestor_engine/packs/MegakernelNative.cpp
src/engines/kernel_ingestor_engine/packs/MegakernelCoResidency.hpp
src/engines/kernel_ingestor_engine/IngestorPacks.cpp
src/engines/kernel_ingestor_engine/IngestorPacks.hpp
src/compilation/Kernel.hpp
src/tests/engines/kernel_ingestor_engine/packs/TestMegakernelMatchers.cpp
src/tests/engines/kernel_ingestor_engine/TestKernelIngestorEngine.cpp
src/integration_tests/kernel_ingestor_engine/IntegrationGpuMegakernel.cpp
src/engines/kernel_ingestor_engine/test_descriptors/integration/megakernel/kernels/MegakernelStandIn.cpp"
  CF=$(command -v clang-format-18 || command -v clang-format)
  ( cd $P && stamp format && $CF --version && $CF --style=file --dry-run --Werror $CXX_CHANGED ) > $LOG/clang-format.log 2>&1
  echo "clang_format_rc=$?"; tail -5 $LOG/clang-format.log
  CT=$(command -v clang-tidy-20 || command -v clang-tidy)
  ( cd $P && stamp tidy && $CT --version | grep version && $CT -p $W/build-ON --config-file=.clang-tidy --warnings-as-errors='*' \
      src/engines/kernel_ingestor_engine/packs/MegakernelNative.cpp \
      src/engines/kernel_ingestor_engine/IngestorPacks.cpp \
      src/tests/engines/kernel_ingestor_engine/packs/TestMegakernelMatchers.cpp \
      src/tests/engines/kernel_ingestor_engine/TestKernelIngestorEngine.cpp \
      src/integration_tests/kernel_ingestor_engine/IntegrationGpuMegakernel.cpp ) > $LOG/clang-tidy-ON.log 2>&1
  echo "clang_tidy_rc=$?"; grep -E "error:|warning:" $LOG/clang-tidy-ON.log | grep -v "warnings generated" | head -20
  ( cd $P/descriptor-packaging && stamp pytest && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -v -p no:cacheprovider tests/test_hkp_megakernel_lock.py ) > $LOG/pytest-lock.log 2>&1
  echo "pytest_rc=$?"; tail -2 $LOG/pytest-lock.log
fi
cd /; rm -rf $W
echo "done $(date -Is)"
