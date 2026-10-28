#!/bin/bash
# Spike only (disposable). One srun task of a spike step (see alola.sh).
# Usage: task.sh OUT_DIR STEP [ARGS]
out=$1 step=$2
shift 2
r=${SLURM_PROCID:-0}
export FLYDSL_RUNTIME_ENABLE_CACHE=0 PIP_CACHE_DIR=/tmp/pipcache PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")/../.."
case $step in
  env)
    { hostname; date; rocm-smi --showtopotype; rocm-smi --showtopoweight; df -h /dev/shm
      env | grep -E '^(NCCL|RCCL|HSA|HIP|ROCR|SLURM_JOB_GPUS|SLURM_STEP_GPUS|GPU_DEVICE)'
      ulimit -l; ls /dev/dri | wc -l; } > "$out/env.log" 2>&1 ;;
  spike)
    . "$HOME/mk/venv/bin/activate"
    [ $# -eq 0 ] && set -- 1 8
    timeout 1200 python scratch/ar_spike/run.py "$out" "$@" > "$out/spike_r$r.log" 2>&1 ;;
  coll-torch)
    . "$HOME/mk/venv/bin/activate"
    timeout 900 python scratch/ar_spike/coll.py "$out" torch > "$out/coll_torch_r$r.log" 2>&1 ;;
  coll-vllm|coll-vllm-noaiter)
    [ "$step" = coll-vllm ] && export VLLM_ROCM_USE_AITER=1 || export VLLM_ROCM_USE_AITER=0
    timeout 900 python3 scratch/ar_spike/coll.py "$out" "${step#coll-}" > "$out/${step//-/_}_r$r.log" 2>&1 ;;
esac
echo "rank $r step $step exit $?" >> "$out/steps.log"
