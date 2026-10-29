#!/bin/bash
# Spike only (disposable). Sync this worktree to Alola /cluster and run one step of the
# spike in the held 2-GPU allocation (job name mktp2). Logs: /cluster/logs/qwen38-tp2/RUN/,
# copied back to bench/results/tp2/logs/RUN/. The home quota is full: nothing is written there.
# Usage: bench/results/tp2/ar_spike/alola.sh RUN STEP [ARGS]  (STEP: env | spike | coll-torch | coll-vllm)
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
run=$1 step=$2
shift 2
R=/cluster/logs/qwen38-tp2
job=${MK_JOB:-$(ssh -o BatchMode=yes alola 'squeue -h -u $USER -n mktp2 -t R -o %i | head -1')}
[ -n "$job" ] || { echo "no mktp2 allocation" >&2; exit 2; }
rsync -a --delete --exclude .git --exclude __pycache__ --exclude bench/results/tp2/logs "$here/" alola:$R/src/
HIPDNN=/cluster/images/hipdnn/hipdnn_20261024_gfx942.sqsh
VLLM=/home/AMD/sareeder/mk/images/vllm-rocm-nightly-00b6c4c5.sqsh
case $step in
  env) img=$HIPDNN n=1 ;;
  spike|coll-torch) img=$HIPDNN n=2 ;;
  coll-vllm|coll-vllm-noaiter) img=$VLLM n=2 ;;
  *) echo "bad step" >&2; exit 2 ;;
esac
ssh -o BatchMode=yes alola "mkdir -p $R/$run && srun --jobid=$job --overlap -n$n --ntasks-per-node=$n --time=30 \
  --container-image=$img --container-mounts=/cluster:/cluster,/home/AMD/sareeder:/home/AMD/sareeder \
  bash $R/src/bench/results/tp2/ar_spike/task.sh $R/$run $step $*" 2>&1 || echo "step $step exit $?"
mkdir -p "$here/bench/results/tp2/logs/$run"
rsync -a alola:$R/$run/ "$here/bench/results/tp2/logs/$run/"
