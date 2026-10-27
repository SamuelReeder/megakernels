#!/bin/bash
# Disposable: run a milestone 3 plan in the held 2-GPU allocation (job mktp2).
# Usage (this host): scratch/m3/run.sh RUN PLAN   (PLAN: a file of scratch/m3/plans/)
# Syncs this worktree to /cluster/logs/qwen38-tp2/src, runs the plan on the login node
# in tmux session m3-RUN; logs in /cluster/logs/qwen38-tp2/RUN (copy back: fetch.sh).
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
run=$1 plan=$2
R=/cluster/logs/qwen38-tp2
job=${MK_JOB:-$(ssh -o BatchMode=yes alola 'squeue -h -u $USER -n mktp2 -t R -o %i | head -1')}
[ -n "$job" ] || { echo "no mktp2 allocation" >&2; exit 2; }
rsync -a --delete --exclude .git --exclude __pycache__ --exclude scratch/logs "$here/" alola:$R/src/
ssh -o BatchMode=yes alola "mkdir -p $R/$run && cp $R/src/scratch/m3/plans/$plan $R/$run/plan.sh && \
  tmux new-session -d -s m3-$run 'JOB=$job OUT=$R/$run SRC=$R/src bash $R/src/scratch/m3/plan_lib.sh $R/$run/plan.sh > $R/$run/driver.log 2>&1'"
echo "started m3-$run in job $job"
