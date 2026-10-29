#!/bin/bash
# Slice B build: rocm-libraries develop@470e78bcfb0 (+ spike diff when EXTRA has
# -DENABLE_MEGAKERNEL_ENGINE=ON), hipdnn + hip-kernel-provider + hipdnn-python, kernel
# ingestor ON, gfx942. Runs inside the hipDNN image; /work = node-local /tmp dir.
# Source arrives as `git archive` of the needed paths (GitHub fetch from AUSTIN ran ~80 KB/s).
set -uo pipefail
EXTRA="${EXTRA:-}"
cd /work
echo "host=$(hostname) nproc=$(nproc) date=$(date -Is) EXTRA=$EXTRA"
test -f src/CMakeLists.txt || { echo "no source in /work/src"; exit 1; }
python3 -c "import zstandard, msgpack; print('zstandard+msgpack ok')" 2>&1 | tail -1
ls /opt/rocm/lib/cmake | grep -i kpack || echo "NOTE: no rocm-kpack cmake package in /opt/rocm"
t0=$(date +%s)
cmake --fresh --preset hip-kernel-provider -S src -B build -G Ninja \
  -DROCM_LIBS_ENABLE_COMPONENTS="hipdnn;hip-kernel-provider;hipdnn-python" \
  -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON -DGPU_TARGETS=gfx942 \
  -DENABLE_CLANG_TIDY=OFF -DENABLE_CLANG_FORMAT=OFF \
  -DHIPDNN_SKIP_TESTS=ON -DHIPKERNELPROVIDER_ENABLE_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX=/work/install $EXTRA > configure.log 2>&1
rc=$?
echo "configure_rc=$rc configure_s=$(( $(date +%s) - t0 ))"
if [ $rc -ne 0 ]; then grep -n "Error" -A8 configure.log | head -60; tail -20 configure.log; exit 1; fi
grep -iE "kpack|ingestor|hkp|megakernel" configure.log | head -20
t0=$(date +%s)
cmake --build build -j 64 > build.log 2>&1
rc=$?
echo "build_rc=$rc build_s=$(( $(date +%s) - t0 ))"
if [ $rc -ne 0 ]; then grep -nE "error|FAILED" build.log | head -40; tail -20 build.log; exit 1; fi
t0=$(date +%s)
cmake --install build > install.log 2>&1; echo "install_rc=$? install_s=$(( $(date +%s) - t0 ))"
find /work/install -name "*.kpack" -o -name "*.kdp.json" | head -20
find build -name "*.so" -path "*python*" | head
du -sh /work/install
