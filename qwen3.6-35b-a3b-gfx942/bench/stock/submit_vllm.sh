#!/bin/bash
# Run on the alola login node (inside tmux). Allocates one MARKHAM gfx942 GPU and
# runs vllm_all.sh inside the vLLM container. Args are passed to vllm_all.sh.
# Environment (e.g. VARIANT=aiter VLLM_ROCM_USE_AITER=1) is exported into the container by srun.
# Do not edit the scripts on the shared home while a job runs: bash reads them lazily over NFS.
IMG=/home/AMD/sareeder/mk/images/vllm-rocm-nightly-00b6c4c5.sqsh
S=/home/AMD/sareeder/mk/stock36
srun -p defq,miopen -A miopen -J mkstock --constraint="GFX942&MARKHAM" --gres=gpu:gfx942-mi300x:1 \
  --cpus-per-task=16 --mem=192G -t 02:00:00 \
  --container-image=$IMG --container-mounts=/cluster:/cluster,/home/AMD/sareeder:/home/AMD/sareeder \
  --container-writable bash $S/vllm_all.sh "$@"
