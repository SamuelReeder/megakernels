#!/bin/bash
# sbatch wrapper of gates.sh: runs it in the hipdnn gfx942 image with ~/mk/venv on the
# allocation (one task per GPU). Usage (login node):
#   sbatch -p miopen -A miopen --constraint="GFX942&MARKHAM" --gres=gpu:gfx942-mi300x:N \
#     -n N --cpus-per-task=16 --mem=64G -t T --exclude=ctr-cx63-mi300x-12 \
#     -o LOG common/scripts/gates_job.sh SRC OUT "PARTS" [EXTRA_CMD]
# SRC: a megakernels tree on the shared home; EXTRA_CMD runs after the gates (task 0).
set -u
src=$1 out=$2 parts=$3 extra=${4:-}
mkdir -p "$out"
srun --container-image=/cluster/images/hipdnn/hipdnn_latest_gfx942.sqsh \
  --container-mounts=/cluster:/cluster,/home/AMD/sareeder:/home/AMD/sareeder \
  bash -c "export PIP_CACHE_DIR=/tmp/pipcache FLYDSL_RUNTIME_ENABLE_CACHE=0 PYTHONDONTWRITEBYTECODE=1; \
    . \$HOME/mk/venv/bin/activate; cd $src && common/scripts/gates.sh $out $parts; \
    if [ \"\$SLURM_PROCID\" = 0 ] && [ -n \"$extra\" ]; then $extra; fi"
echo "job $SLURM_JOB_ID done $(date -Is)"
