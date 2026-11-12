#!/bin/bash
# MkEngine: build hipDNN + hip-kernel-provider (ingestor ON) with ENABLE_MEGAKERNEL_ENGINE OFF and ON,
# run unit tests of each. Runs inside hipdnn_latest_gfx942.sqsh on a CPU node; /tmp is node-local.
set -uo pipefail
MODES=ON
export PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; export PYTHONNOUSERSITE=1
python3 -c "import zstandard, msgpack, pytest; print(\"py deps ok\")"
W=/tmp/mkeng-gpu; [ "${REUSE:-0}" = 1 ] || rm -rf $W; mkdir -p $W/src; cd $W
echo "host=$(hostname) nproc=$(nproc) date=$(date -Is)"
rm -rf src.new; mkdir src.new; tar xzf ~/work/mkeng/mk-src.tgz -C src.new
python3 - <<'PY'
import filecmp, os, shutil
n=0
for root, dirs, files in os.walk('src.new'):
    for f in files:
        a=os.path.join(root,f); b='src'+a[len('src.new'):]
        if not os.path.exists(b) or not filecmp.cmp(a,b,shallow=False):
            os.makedirs(os.path.dirname(b), exist_ok=True); shutil.copyfile(a,b); n+=1
print("source files updated:", n)
PY
D=src/dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/megakernel
tar xzf ~/work/mkeng/mk-art.tgz -C $D   # stands in for `dvc pull` (bucket not writable)
cmake --version | head -1; hipcc --version 2>/dev/null | grep -i "HIP version"
LOG=~/work/mkeng/logs-gpu; mkdir -p $LOG
for MODE in ${MODES:-OFF ON}; do
  B=$W/build-$MODE
  TIDY=OFF
  t0=$(date +%s)
  [ "${REUSE:-0}" = 1 ] && [ -f $B/build.ninja ] && FRESH= || FRESH=--fresh
  cmake $FRESH --preset hip-kernel-provider -S src -B $B -G Ninja \
    -DROCM_LIBS_ENABLE_COMPONENTS="hipdnn;hipdnn-integration-tests;hip-kernel-provider" \
    -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON -DGPU_TARGETS=gfx942 \
    -DENABLE_CLANG_FORMAT=ON -DENABLE_CLANG_TIDY=$TIDY -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DENABLE_MEGAKERNEL_ENGINE=$MODE -DPython3_EXECUTABLE=/usr/bin/python3 -DCMAKE_INSTALL_PREFIX=$W/install-$MODE > $LOG/configure-$MODE.log 2>&1
  echo "MODE=$MODE configure_rc=$? s=$(( $(date +%s)-t0 ))"
  grep -iE "megakernel|DVC output" $LOG/configure-$MODE.log | head
  t0=$(date +%s)
  cmake --build $B -j $(nproc) > $LOG/build-$MODE.log 2>&1; rc=$?
  echo "MODE=$MODE build_rc=$rc s=$(( $(date +%s)-t0 ))"
  if [ $rc -ne 0 ]; then grep -nE "error|FAILED|hkp megakernel lock" $LOG/build-$MODE.log | head -40; continue; fi
  if [ $MODE = ON ]; then
    P=src/dnn-providers/hip-kernel-provider
    clang-tidy-20 -p $B --config-file=$P/.clang-tidy --warnings-as-errors='*' \
      $P/src/engines/kernel_ingestor_engine/packs/MegakernelNative.cpp \
      $P/src/engines/kernel_ingestor_engine/IngestorPacks.cpp \
      $P/src/tests/engines/kernel_ingestor_engine/packs/TestMegakernelMatchers.cpp \
      $P/src/integration_tests/kernel_ingestor_engine/IntegrationGpuMegakernel.cpp \
      $P/src/tests/engines/kernel_ingestor_engine/TestKernelIngestorEngine.cpp > $LOG/tidy-ON.log 2>&1
    echo "MODE=ON clang_tidy_rc=$?"; grep -E "error:|warning:" $LOG/tidy-ON.log | head -20
  fi
  BIN=$B/bin
  ls $BIN | grep -E "hip_kernel_provider.*tests" 
  find $B -path "*megakernel*" -name "*.kdp.json" -o -path "*megakernel*" -name "*.kpack" | head
  ( cd $BIN && ./hip_kernel_provider_tests --gtest_brief=1 > $LOG/unit-$MODE.log 2>&1; echo "MODE=$MODE unit_rc=$?" )
  grep -E "^\[  (PASSED|FAILED|SKIPPED) |tests ran" $LOG/unit-$MODE.log | head
  ( cd $BIN && ./hip_kernel_provider_tests --gtest_filter='TestMegakernel*' > $LOG/unit-megakernel-$MODE.log 2>&1; echo "MODE=$MODE megakernel_unit_rc=$?"; tail -3 $LOG/unit-megakernel-$MODE.log )
done
arch=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+'); echo "gpu arch=$arch"
( cd $W/build-ON/bin && HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0} ./hip_kernel_provider_integration_tests --gtest_filter='IntegrationGpuMegakernel.*' > $LOG/integration-megakernel-ON.log 2>&1; echo "integration_rc=$?"; tail -6 $LOG/integration-megakernel-ON.log )
echo "packaging pytest:"
( cd src/dnn-providers/hip-kernel-provider/descriptor-packaging && python3 -m pytest -q -p no:cacheprovider tests/test_hkp_megakernel_lock.py > $LOG/pytest-lock.log 2>&1; echo "pytest_rc=$?"; tail -2 $LOG/pytest-lock.log )
echo "done $(date -Is)"
