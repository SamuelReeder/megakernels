#!/bin/bash
# Slice C build of hipdnn + hip-kernel-provider (ENABLE_MEGAKERNEL_ENGINE=ON) + hipdnn-python, gfx942.
# Runs inside a container; /work = node-local dir with src/ (git archive of the spike branch).
# TAG=hipdnnimg: ROCm in /opt/rocm (hipDNN image). TAG=vllmimg: ROCm from the _rocm_sdk_devel wheel
# that torch in the vLLM image uses. The rocm_kpack python source is copied from the hipDNN image
# (/opt/rocm-kpack/python) to /work/kpack-python by the hipdnnimg run.
set -uo pipefail
TAG=${TAG:?}
cd /work
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "$HOME" | paste -sd:)
PY=$(command -v python3)
echo "host=$(hostname) nproc=$(nproc) date=$(date -Is) TAG=$TAG"
if [ -d /opt/rocm-kpack/python ] && [ ! -d /work/kpack-python ]; then cp -r /opt/rocm-kpack/python /work/kpack-python; fi
if [ -d /opt/rocm/lib/llvm ]; then R=/opt/rocm; else
  R=$(python3 -c "import _rocm_sdk_devel, pathlib; print(pathlib.Path(_rocm_sdk_devel.__file__).parent)"); fi
echo "ROCM_PATH=$R"; ls "$R/lib/llvm/bin/clang++" && "$R/lib/llvm/bin/clang++" --version | head -1
echo "cmake $(cmake --version | head -1) ninja $(ninja --version) python $(python3 --version)"
python3 -c "import zstandard, msgpack; print('zstandard+msgpack ok')" 2>&1 | tail -1
ls -d "$R"/lib/cmake/* | tr '\n' ' '; echo
B=/work/build-$TAG; I=/work/install-$TAG
t0=$(date +%s)
cmake --fresh --preset hip-kernel-provider -S src -B $B -G Ninja -DROCM_PATH=$R \
  -DROCM_LIBS_ENABLE_COMPONENTS="hipdnn;hip-kernel-provider;hipdnn-python" \
  -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON -DGPU_TARGETS=gfx942 \
  -DENABLE_CLANG_TIDY=OFF -DENABLE_CLANG_FORMAT=OFF \
  -DHIPDNN_SKIP_TESTS=ON -DHIPKERNELPROVIDER_ENABLE_TESTS=OFF \
  -DHIPKERNELPROVIDER_KPACK_PYTHON_DIR=/work/kpack-python \
  -DPython_EXECUTABLE=$PY -DPython3_EXECUTABLE=$PY \
  -DCMAKE_INSTALL_PREFIX=$I -DENABLE_MEGAKERNEL_ENGINE=ON > configure-$TAG.log 2>&1
rc=$?
echo "configure_rc=$rc configure_s=$(( $(date +%s) - t0 ))"
if [ $rc -ne 0 ]; then grep -n "Error" -A12 configure-$TAG.log | head -80; tail -30 configure-$TAG.log; exit 1; fi
t0=$(date +%s)
cmake --build $B -j "$(nproc)" > build-$TAG.log 2>&1
rc=$?
echo "build_rc=$rc build_s=$(( $(date +%s) - t0 ))"
if [ $rc -ne 0 ]; then grep -nE "error|FAILED" build-$TAG.log | head -40; tail -20 build-$TAG.log; exit 1; fi
cmake --install $B > install-$TAG.log 2>&1; echo "install_rc=$?"
python3 src/projects/hipdnn/python/frontend_wheel_package/pack_frontend_wheel.py --build-dir $B \
  --wheel-dir /work/wheel-$TAG > wheel-$TAG.log 2>&1; echo "wheel_rc=$? (staged dir is used)"
ls /work/wheel-$TAG/hipdnn_frontend | head
find $I -name "*.kpack" | head
ldd $I/lib/libhipdnn_backend.so | grep -E "amdhip|hiprtc|comgr|stdc"
