#!/bin/bash
# Spike rtgeom (disposable): bench/vllm_e2e.py (kernel on, TP1) at one max_model_len, run by
# rtrun vllm. /work/vllm-src = copy of ~/mk/vllm-src (vLLM image build) with the files of
# the vLLM spike branch (users/sareeder/qwen38-rtgeom-spike) from /work/vllm-overlay.
# Usage: bash ../spike/rtgeom/vllm_run.sh NAME MAXLEN [ENV=VALUE...]
#   VLLM_MK_RTGEOM=1|2: run-time-geometry kernel; VLLM_MK_HIPDNN_EXPORT=/work/out/co_X: write
#   the code objects; VLLM_MK_RTGEOM_CO=/work/out/co_X: launch those code objects.
# Output: /work/out/vllm/NAME.{json,log}.
set -uo pipefail
name=$1 L=$2
shift 2
for kv in "$@"; do export "$kv"; done
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "$HOME" | paste -sd:)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=1 TOKENIZERS_PARALLELISM=false
if [ ! -d /work/vllm-src ]; then cp -a /home/AMD/sareeder/mk/vllm-src /work/vllm-src; fi
tar -C /work/vllm-overlay -cf - vllm | tar -C /work/vllm-src -xf -
export PYTHONPATH=/work/vllm-src
mkdir -p /work/out/vllm
echo "# host=$(hostname) date=$(date -Is) name=$name maxlen=$L env: $* vllm overlay $(cat /work/vllm-overlay/REVISION) mk $(cat /work/mk/REVISION)" \
  > /work/out/vllm/$name.log
MK_MAXLEN=$L timeout 3000 python3 bench/vllm_e2e.py /home/AMD/sareeder/mk/models/Qwen3.8-27B-Quark-AWQ-MXFP4 on \
  /work/out/vllm/$name.json ${MAXTOK:-128} >> /work/out/vllm/$name.log 2>&1
rc=$?
echo "exit $rc" >> /work/out/vllm/$name.log
echo "$name exit $rc"
grep -hE "SPIKE rtgeom|KV block|batch [1248]: TPOT" /work/out/vllm/$name.log | sed -E 's/^.*\] //' | cut -c1-400
