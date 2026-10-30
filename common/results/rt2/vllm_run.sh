#!/bin/bash
# rt2 evidence: the model's bench/vllm_e2e.py (kernel on, TP1, greedy) at each MAXLEN, run
# by `rt2run vllm MODEL_DIR VLLM_DIR` in the model dir. NAME tags the vLLM tree (base: the
# branch head, rt_geom 0; rt2: the commit with the rt_geom 2 kernels).
# Usage: bash ../common/results/rt2/vllm_run.sh NAME MAXLEN...
# Output: /work/out/vllm_<model>/NAME_<maxlen>.{json,log}.
set -uo pipefail
name=$1
shift
model=$(basename "$PWD")
case $model in
  qwen3.8*) W=$M/Qwen3.8-27B-Quark-AWQ-MXFP4 ;;
  qwen3.6*) W=$M/Qwen3.6-35B-A3B-FP8 ;;
  gemma4*) W=$M/gemma-4-26B-A4B-it-FP8-dynamic ;;
esac
O=/work/out/vllm_${model%%-*}
mkdir -p "$O"
for L in "$@"; do
  f=$O/${name}_$L
  echo "# $(date -Is) $(hostname) $name maxlen=$L $(cat ../REVISION)" > "$f.log"
  case $model in
    qwen3.8*) MK_MAXLEN=$L timeout 3000 python3 bench/vllm_e2e.py "$W" on "$f.json" 128 >> "$f.log" 2>&1 ;;
    *) MK_MAXLEN=$L timeout 3000 python3 bench/vllm_e2e.py run "$W" on "$f.json" >> "$f.log" 2>&1 ;;
  esac
  echo "exit $?" >> "$f.log"
  echo "${name}_$L: $(grep -hE 'block-table columns|batch [1248]: TPOT|^exit|Error' "$f.log" | sed -E 's/^.*\] //' | cut -c1-200 | tr '\n' ';')"
done
