#!/bin/bash
# Step 1a (docs/hipdnn-route.md) GPU evidence on one held 2-GPU gfx942 allocation on a
# MARKHAM node (shared home: ~/mk/venv, ~/mk/models, ~/mk/images). Run on the workstation.
#   gpu.sh stage JOB   ship the trees (git archive) to the node's /tmp/cc_JOB:
#                      mk_new (this branch's HEAD), mk_main (megakernels main 10bd202), and
#                      the vLLM trees v_<name> of the TREES below
#   gpu.sh run JOB     start plan.sh in tmux on the login node (log ~/mk/out/cc_JOB/plan.log)
#   gpu.sh fetch JOB   copy ~/mk/out/cc_JOB to common/results/compiled/gpu/
# plan.sh (on the login node): two vLLM lanes (GPU 0, GPU 1: vlane.sh, the vLLM nightly
# image), the TP2 vLLM runs (both GPUs), then gates.sh q8 q6 g for mk_new (GPU 0) and
# mk_main (GPU 1), then gates.sh q8tp2 (srun -n2) for mk_new and mk_main (hipdnn image).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
VR=${VR:-/home/sareeder/ROCm-workspace/repos/vllm}
MAIN=${MAIN:-10bd202}
# vLLM trees: name=commit (new heads of step 1a and the previous fork heads)
TREES="q8new=9fc7bc243 q8tp1new=0332fdc96 q8prev=aa9c7ae58 q8tp1prev=cfc4c8c5b
q6new=6657a8352 q6prev=0801f7971 gnew=5a25d336a gprev=7f586cd2a"
mode=$1 job=$2
W=/tmp/cc_$job
O=/home/AMD/sareeder/mk/out/cc_$job
on_node() { ssh -o BatchMode=yes alola "srun --jobid=$job --overlap -n1 --time=${2:-10} bash -c '$1'"; }
case $mode in
  stage)
    on_node "mkdir -p $W && hostname && rocminfo | grep -m1 -oE gfx[0-9a-f]+ && df -h /tmp | tail -1"
    git -C "$root" archive --prefix=mk_new/ HEAD | gzip -1 | on_node "gzip -d | tar -xf - -C $W && echo \"$(git -C "$root" rev-parse HEAD)\" > $W/mk_new/REVISION"
    git -C "$root" archive --prefix=mk_main/ "$MAIN" | gzip -1 | on_node "gzip -d | tar -xf - -C $W && echo \"$(git -C "$root" rev-parse "$MAIN")\" > $W/mk_main/REVISION"
    for t in $TREES; do
      n=${t%%=*} c=${t#*=}
      git -C "$VR" archive --prefix="v_$n/" "$c" vllm tests pyproject.toml | gzip -1 \
        | on_node "gzip -d | tar -xf - -C $W && echo $(git -C "$VR" rev-parse "$c") > $W/v_$n/REVISION"
    done
    on_node "ls $W; du -sh $W"
    ;;
  run)
    for f in plan.sh vlane.sh; do ssh -o BatchMode=yes alola "mkdir -p $O && cat > $O/$f" < "$here/$f"; done
    ssh -o BatchMode=yes alola "tmux new-session -d -s cc_$job 'JOB=$job W=$W O=$O bash $O/plan.sh > $O/plan.log 2>&1'"
    echo "tmux cc_$job on the login node; log $O/plan.log"
    ;;
  fetch)
    mkdir -p "$here/gpu"
    ssh -o BatchMode=yes alola "tar -C $O -czf - --exclude='*.s' ." | tar -C "$here/gpu" -xzf -
    ;;
esac
