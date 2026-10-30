#!/bin/bash
# Produce one code-object set (common/results/co/NOTES.md) on a held gfx942 allocation:
# per model, node_model.sh through common/scripts/vrun on a clean detached checkout of the
# set's vLLM commit, results in ~/coexport/sets/SET on the Alola home (MARKHAM nodes share
# it); then fetch them here and merge run a of every model into SET/final.
# Usage: common/results/co/run_set.sh JOBID SET [MODEL_KEY...]   (keys: q8 q6 g; default all)
# Env: VLLM_WT (dir of the detached vLLM worktrees co-<sha>, default
#      ~/ROCm-workspace/worktrees/vllm).
set -uo pipefail
job=$1 set=$2
shift 2
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
vwt=${VLLM_WT:-/home/sareeder/ROCm-workspace/worktrees/vllm}
S=/home/AMD/sareeder/coexport/sets/$set
D4="block=784,maxlen=4096"
case $set in
  rt0) q8=fd8a77a5f q6=70f57a0ae g=7f586cd2a q8l=4096 q6l=4096 ;;
  rt2) q8=9fc7bc243 q6=6657a8352 g=5a25d336a q8l=4096:8192 q6l=4096:8192 ;;
  *) echo "unknown set $set" >&2; exit 2 ;;
esac
[ -z "$(git -C "$root" status --porcelain common)" ] || { echo "commit common/ first" >&2; exit 2; }
mc=$(git -C "$root" rev-parse HEAD)
mkdir -p "$here/$set"
for k in "${@:-q8 q6 g}"; do
  for kk in $k; do
    case $kk in
      q8) dir=qwen3.8-27b-gfx942 model=qwen38-27b-mxfp4-gfx942 w=Qwen3.8-27B-Quark-AWQ-MXFP4 c=$q8
          deps=("kv=ROCM_ATTN,block=784,maxlen=$q8l,tp=1,M=1:2:4:8,cus=304"
                "kv=ROCM_ATTN,block=784,maxlen=4096,tp=2,M=1:2:4:8,cus=304") ;;
      q6) dir=qwen3.6-35b-a3b-gfx942 model=qwen36-35b-a3b-fp8-gfx942 w=Qwen3.6-35B-A3B-FP8 c=$q6
          deps=("kv=TRITON_ATTN:1,block=1056,maxlen=$q6l,tp=1,M=1:2:4:8,cus=304") ;;
      g) dir=gemma4-26b-a4b-gfx942 model=gemma4-26b-a4b-fp8-gfx942 w=gemma-4-26B-A4B-it-FP8-dynamic c=$g
         deps=("kv=TRITON_ATTN:1,block=16,maxlen=4096,tp=1,M=1:2:4:8,cus=304") ;;
    esac
    v=$vwt/co-$c
    vc=$(git -C "$v" rev-parse HEAD)
    [ -z "$(git -C "$v" status --porcelain vllm)" ] || { echo "$v is dirty" >&2; exit 2; }
    echo "# $(date -Is) set $set $model vllm $vc megakernels $mc"
    MK_TIME=${MK_TIME:-150} VLLM_REV=$vc "$root/common/scripts/vrun" "$root/$dir" "$v" -j "$job" \
      "bash ../common/results/co/node_model.sh $S $model \$M/$w $vc $mc ${deps[*]}" \
      2>&1 | tee "$here/$set/run_$model.log" | grep -E '^export|EQUAL|DIFFERENT|^H1|^# H1|rror' | cut -c1-400
  done
done
ssh -o BatchMode=yes alola "tar -C $S -cf - --exclude=b ." | tar -C "$here/$set" -xf -
echo "fetched $S -> $here/$set"
