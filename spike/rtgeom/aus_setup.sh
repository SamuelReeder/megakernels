#!/bin/bash
# Spike rtgeom (disposable): prepare an AUSTIN gfx942 node (no shared home) for rtrun AUS=1.
# Runs on the node outside a container: ssh alola "srun --jobid=J --overlap bash -s" < aus_setup.sh
# Copies the staged files (/cluster/images/sareeder/rtgeom: the vLLM image and a tar of
# ~/mk/{venv, vllm-src, models/Qwen3.8-27B-Quark-AWQ-MXFP4} from the MARKHAM home) to
# /tmp/rtgeom_<job>: home/mk/... and vllm.sqsh.
set -uo pipefail
S=/cluster/images/sareeder/rtgeom
W=/tmp/rtgeom_$SLURM_JOB_ID
echo "node $(hostname -s) job $SLURM_JOB_ID $(date -Is)"
rocminfo | grep -m1 -oE 'gfx[0-9a-f]+'
rocm-smi --showproductname 2>/dev/null | grep -m2 -iE "card series|card model"
df -h /tmp | tail -1
mkdir -p "$W/home/mk" "$W/out"
t0=$(date +%s)
cp "$S/vllm-rocm-nightly-00b6c4c5.sqsh" "$W/vllm.sqsh"
echo "image copied $(( $(date +%s) - t0 ))s"
t0=$(date +%s)
tar -xf "$S/mk_bundle.tar" -C "$W/home/mk"
echo "bundle extracted $(( $(date +%s) - t0 ))s"
du -sh "$W/home/mk"/* "$W/vllm.sqsh"
df -h /tmp | tail -1
