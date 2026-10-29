#!/bin/bash
# Slice C run inside the vLLM image; /work = node-local dir with mk/ (megakernels tree: model dir,
# common/, spike/), install-$TAG/ and wheel-$TAG/ from c0_build.sh. TAG selects the hipDNN build.
set -uo pipefail
TAG=${TAG:-vllmimg}
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "$HOME" | paste -sd:)
export PIP_CACHE_DIR=/tmp/pipcache FLYDSL_RUNTIME_ENABLE_CACHE=0 PYTHONPYCACHEPREFIX=/tmp/pycache
export PYTHONPATH=/work/wheel-$TAG LD_LIBRARY_PATH=/work/install-$TAG/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export HIPDNN_LOG_LEVEL=${HIPDNN_LOG_LEVEL:-warn}
# torch 2.13+rocm10 (vLLM image) loads _rocm_sdk_libraries/lib/libhipdnn_backend.so at import; the
# spike extension then binds to that older backend (undefined hipdnnGetEngineName...). Preload ours.
[ "${PRELOAD:-1}" = 1 ] && export LD_PRELOAD=/work/install-$TAG/lib/libhipdnn_backend.so
echo "host=$(hostname) date=$(date -Is) TAG=$TAG arch=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx9[0-9a-f]+')"
cd /work/mk/qwen3.8-27b-gfx942
python3 "$@"
