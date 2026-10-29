#!/bin/bash
# rt2 evidence, Qwen3.8 TP1: spike/rtgeom/vllm_all.sh (users/sareeder/runtime-geom) with
# rt_geom 2: per max_model_len the specialized kernel (rt_geom 0, compiled for that
# geometry) and the rt_geom 2 code objects exported once at 4096 (VLLM_MK_HIPDNN_EXPORT),
# launched with hipModuleLaunchKernel (VLLM_MK_RTGEOM_CO). vLLM: users/sareeder/
# qwen38-rtgeom-spike (its kernel copy is the spike's rt_geom code). Run by
# `rt2run vllm qwen3.8-27b-gfx942 <qwen38-rtgeom-spike checkout>` in the model dir.
# Usage: bash ../common/results/rt2/vllm_q38_co.sh [MAXLENS=4096,8192,16384]
# Output: /work/out/q38_vllm_co/NAME.{json,log}, co_rt2/.
set -uo pipefail
W=$M/Qwen3.8-27B-Quark-AWQ-MXFP4
O=/work/out/q38_vllm_co
CO=$O/co_rt2
mkdir -p "$O"
run() {  # run NAME MAXLEN [ENV=VALUE...]
  local name=$1 L=$2
  shift 2
  (
    for kv in "$@"; do export "$kv"; done
    echo "# $(date -Is) $(hostname) name=$name maxlen=$L env: $* $(cat ../REVISION 2>/dev/null)" > "$O/$name.log"
    MK_MAXLEN=$L timeout 3000 python3 bench/vllm_e2e.py "$W" on "$O/$name.json" 128 >> "$O/$name.log" 2>&1
    echo "exit $?" >> "$O/$name.log"
  )
  echo "$name: $(grep -hE 'KV block|batch [1248]: TPOT|^exit' "$O/$name.log" | sed -E 's/^.*\] //' | cut -c1-160 | tr '\n' ';')"
}
run spec4096 4096
run rt4096 4096 VLLM_MK_RTGEOM=2 VLLM_MK_HIPDNN_EXPORT=$CO
for L in $(echo "${1:-4096,8192,16384}" | tr , ' '); do
  [ "$L" = 4096 ] || run spec$L "$L"
  run rtco$L "$L" VLLM_MK_RTGEOM=2 VLLM_MK_RTGEOM_CO=$CO
done
sha256sum "$CO"/*.co
cat "$CO"/manifest.jsonl
python3 ../common/results/rt2/vcompare.py "$O/spec4096.json" "$O/rt4096.json" | tail -1
for L in $(echo "${1:-4096,8192,16384}" | tr , ' '); do
  python3 ../common/results/rt2/vcompare.py "$O/spec$L.json" "$O/rtco$L.json" | tail -1
done
