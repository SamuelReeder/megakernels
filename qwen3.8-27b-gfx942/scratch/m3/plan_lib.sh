#!/bin/bash
# Disposable: runs on the Alola login node. Defines the step helpers, then sources PLAN.
#   one NAME [GPU] ENV... -- CMD...   one process (GPU 0 or 1), backgrounded with &
#   two NAME ENV... -- CMD...         one process per GPU (MK_TP=2), foreground
# A plan waits for its background steps with `wait`.
IMG=/cluster/images/hipdnn/hipdnn_20261024_gfx942.sqsh
MOUNTS=/cluster:/cluster,/home/AMD/sareeder:/home/AMD/sareeder
MX=/home/AMD/sareeder/mk/models/Qwen3.8-27B-Quark-AWQ-MXFP4
BF=/home/AMD/sareeder/mk/models/Qwen3.8-27B
PORT=29600
one() {
  local name=$1 gpu=$2
  shift 2
  srun --jobid="$JOB" --overlap -n1 --time=180 --container-image=$IMG --container-mounts=$MOUNTS \
    bash "$SRC/scratch/m3/task.sh" "$OUT" "$name" HIP_VISIBLE_DEVICES=$gpu "$@" &
}
two() {
  local name=$1
  shift
  PORT=$((PORT + 1))
  srun --jobid="$JOB" --overlap -n2 --ntasks-per-node=2 --time=180 --container-image=$IMG \
    --container-mounts=$MOUNTS bash "$SRC/scratch/m3/task.sh" "$OUT" "$name" MK_TP=2 MK_PORT=$PORT "$@"
}
VLLM_IMG=/home/AMD/sareeder/mk/images/vllm-rocm-nightly-00b6c4c5.sqsh
vllm_step() {  # one process on both GPUs in the vLLM nightly image, foreground
  local name=$1
  shift
  srun --jobid="$JOB" --overlap -n1 --time=180 --container-image=$VLLM_IMG --container-writable \
    --container-mounts=$MOUNTS bash "$SRC/scratch/m3/task.sh" "$OUT" "$name" NOVENV=1 "$@"
}
echo "plan $1 start $(date -u +%T) job $JOB"
. "$1"
wait
echo "plan done $(date -u +%T)"
