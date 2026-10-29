#!/bin/bash
# Slice C step 4: bench/vllm_e2e.py (kernel on) in the vLLM nightly image, TP1, with the overlay
# /work/vllm-src = copy of ~/mk/vllm-src (image build + specialized decode) plus the
# users/sareeder/qwen38-hipdnn-spike specialized_decode package (shipped to /work/vllm-overlay).
# Usage: c4_vllm.sh NAME [ENV=VALUE...]   e.g. VLLM_MK_HIPDNN_EXPORT=/work/export or VLLM_MK_HIPDNN=1
set -uo pipefail
name=$1; shift
for kv in "$@"; do export "$kv"; done
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "$HOME" | paste -sd:)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=1 TOKENIZERS_PARALLELISM=false
export FLYDSL_RUNTIME_ENABLE_CACHE=0 PIP_CACHE_DIR=/tmp/pipcache PYTHONDONTWRITEBYTECODE=1
if [ ! -d /work/vllm-src ]; then cp -a /home/AMD/sareeder/mk/vllm-src /work/vllm-src; fi
tar -C /work/vllm-overlay -cf - . | tar -C /work/vllm-src -xf -
export PYTHONPATH=/work/vllm-src
if [ "${VLLM_MK_HIPDNN:-0}" = 1 ]; then
  export PYTHONPATH=/work/vllm-src:/work/wheel-vllmimg
  export LD_LIBRARY_PATH=/work/install-vllmimg/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  export LD_PRELOAD=/work/install-vllmimg/lib/libhipdnn_backend.so HIPDNN_LOG_LEVEL=${HIPDNN_LOG_LEVEL:-warn}
fi
echo "# host=$(hostname) date=$(date -Is) name=$name env: $*"
mkdir -p /work/vout
cd /work/mk/qwen3.8-27b-gfx942
timeout 3000 python3 bench/vllm_e2e.py /home/AMD/sareeder/mk/models/Qwen3.8-27B-Quark-AWQ-MXFP4 on \
  /work/vout/$name.json ${MAXTOK:-128}
echo "exit $?"
