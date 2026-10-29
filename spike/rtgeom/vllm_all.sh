#!/bin/bash
# Spike rtgeom (disposable): the vLLM runs of step 3, run by rtrun vllm. Per max_model_len:
# the specialized kernel (rt_geom 0, compiled for that geometry) and the rt_geom=1 code
# objects exported once, at 4096 (VLLM_MK_HIPDNN_EXPORT), launched with
# hipModuleLaunchKernel (VLLM_MK_RTGEOM_CO).
# Usage: bash ../spike/rtgeom/vllm_all.sh [MAXLENS=4096,8192,16384]
set -uo pipefail
R=../spike/rtgeom/vllm_run.sh
CO=/work/out/co_rt1
bash $R spec4096 4096
bash $R rt4096 4096 VLLM_MK_RTGEOM=1 VLLM_MK_HIPDNN_EXPORT=$CO
for L in $(echo "${1:-4096,8192,16384}" | tr , ' '); do
  [ "$L" = 4096 ] || bash $R spec$L "$L"
  bash $R rtco$L "$L" VLLM_MK_RTGEOM=1 VLLM_MK_RTGEOM_CO=$CO
done
sha256sum $CO/*.co
cat $CO/manifest.jsonl
