#!/bin/bash
#SBATCH -p miopen
#SBATCH -A miopen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH -t 06:00:00
#SBATCH --constraint=GFX942&MARKHAM
# Reproducibility run (docs/DESIGN.md "Milestone 5"): the full tests/run_suite.sh on the node
# Slurm picks, from a source tarball of this directory, in the hipdnn gfx942 image with the
# venv of scripts/setup_env.sh. MARKHAM nodes share the login home: the checkpoints come
# from ~/mk/models and the logs go to OUT_DIR.
# Usage (from the login node):
#   tar -C gemma4-26b-a4b-gfx942 --exclude=__pycache__ -czf ~/mk/g4src_REV.tgz .
#   sbatch [--exclude=NODE] [--begin=YYYY-MM-DDTHH:MM] scripts/repro_sbatch.sh \
#     ~/mk/g4src_REV.tgz OUT_DIR
set -u
tgz=$1
out=$2
img=$(readlink -f /cluster/images/hipdnn/hipdnn_latest_gfx942.sqsh)
mkdir -p "$out"
echo "image $img" > "$out/image.txt"
srun --container-image="$img" --container-mounts="/cluster:/cluster,$HOME:$HOME,/tmp:/tmp" \
  bash -lc "export PIP_CACHE_DIR=/tmp/pipcache FLYDSL_RUNTIME_ENABLE_CACHE=0 \
PYTHONPYCACHEPREFIX=/tmp/pycache; src=/tmp/g4repro-\$SLURM_JOB_ID; mkdir -p \$src && \
tar -xzf $tgz -C \$src && . \$HOME/mk/venv/bin/activate && cd \$src && \
bash tests/run_suite.sh \$HOME/mk/models $out; rc=\$?; rm -rf \$src; exit \$rc"
