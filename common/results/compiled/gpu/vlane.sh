#!/bin/bash
# One vLLM lane of gpu.sh, inside the vLLM nightly image on the node. For each SPEC
# NAME:TREE:MODEL:MAXLEN:TP:DUMP it runs MODEL's bench/vllm_e2e.py from mk_new (kernel on,
# greedy, as common/results/rt2/vllm_run.sh) with W/v_TREE's vllm/ first on PYTHONPATH (plus
# the image's compiled .so files, as common/scripts/vrun). DUMP=1: FlyDSL IR dumps; writes
# NAME_isa.txt = per built kernel its name and the sha256 of its final ISA, so two runs at
# different max_model_len show whether they built the same code objects.
# Usage: vlane.sh W O SPEC...   (HIP_VISIBLE_DEVICES set by the caller)
set -u
W=$1 O=$2
shift 2
so=$(find /usr /opt -maxdepth 7 -name _C.abi3.so -path '*packages/vllm/*' | head -1)
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false FLYDSL_RUNTIME_ENABLE_CACHE=0 \
  PYTHONPYCACHEPREFIX=/tmp/pycache_$$ PIP_CACHE_DIR=/tmp/pipcache
M=/home/AMD/sareeder/mk/models
mkdir -p "$O/vllm"
for spec; do
  IFS=: read -r name tree model L tp dump <<< "$spec"
  v=$W/v_$tree
  cp -n "$(dirname "$so")"/*.so "$(dirname "$so")/_version.py" "$v/vllm/" 2>/dev/null
  case $model in
    q8) d=qwen3.8-27b-gfx942 w=$M/Qwen3.8-27B-Quark-AWQ-MXFP4 ;;
    q6) d=qwen3.6-35b-a3b-gfx942 w=$M/Qwen3.6-35B-A3B-FP8 ;;
    g) d=gemma4-26b-a4b-gfx942 w=$M/gemma-4-26B-A4B-it-FP8-dynamic ;;
  esac
  f=$O/vllm/$name
  echo "# $(date -Is) $(hostname) $name: vllm v_$tree $(cat "$v/REVISION"), megakernels $(cat "$W/mk_new/REVISION") bench, maxlen $L, tp $tp, HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES, vllm .so from $so" > "$f.log"
  env=(MK_MAXLEN="$L" PYTHONPATH="$v")
  [ "$tp" = 2 ] && env+=(MK_TP=2)
  dd=/tmp/cc_dump_$name
  if [ "$dump" = 1 ]; then rm -rf "$dd"; env+=(FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR="$dd"); fi
  cd "$W/mk_new/$d" || exit 1
  if [ "$model" = q8 ]; then args=(bench/vllm_e2e.py "$w" on "$f.json" 128)
  else args=(bench/vllm_e2e.py run "$w" on "$f.json"); fi
  env "${env[@]}" timeout 3000 python3 "${args[@]}" >> "$f.log" 2>&1
  echo "exit $?" >> "$f.log"
  if [ "$dump" = 1 ]; then
    for x in "$dd"/*/22_final_isa.s; do
      [ -f "$x" ] && echo "$(basename "$(dirname "$x")") $(sha256sum < "$x" | cut -c1-16)"
    done | sort > "${f}_isa.txt"
    rm -rf "$dd"
  fi
  echo "$name: $(grep -hE 'block-table columns|batch [1248]: TPOT|^exit|Error' "$f.log" | sed -E 's/^.*\] //' | cut -c1-160 | tr '\n' ';')"
done
