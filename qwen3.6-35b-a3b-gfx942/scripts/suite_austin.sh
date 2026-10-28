#!/bin/bash
# Milestone 5 suite on an AUSTIN gfx942 node (MI325X: no shared home). Runs inside the
# hipdnn gfx942 image, reads nothing from the home directory, and prints the results as a
# base64 tar between RESULTS-BEGIN / RESULTS-END (scripts/finish_m5.sh unpacks them).
# Start it from the Alola login node (docs/gpu-access.md, "Run a Job"):
#   tmux new-session -d -s mk36s3 "srun -p defq,miopen -A miopen -J mk36s3 \
#     --gres=gpu:gfx942-mi325x:1 --constraint='GFX942&AUSTIN' --cpus-per-task=16 \
#     --mem=400G -t 03:00:00 --container-image=/cluster/images/hipdnn/hipdnn_latest_gfx942.sqsh \
#     bash -s < ~/mk/suite36/suite_austin.sh > ~/mk/suite36/mk36s3.log 2>&1"
# The body is one function, so bash parses it fully before a command reads stdin.
set -uo pipefail
main() {
REV=${REV:-41443a0}
echo "SLURM_JOB_ID=$SLURM_JOB_ID node $(hostname -s) $(date -Is)"
df -h /tmp | tail -1
arch=$(rocminfo | grep -m1 -oE 'gfx[0-9a-f]+')
echo "arch $arch"
[ "$arch" = gfx942 ] || { echo "no gfx942 GPU"; exit 1; }
W=/tmp/mk36s3_$SLURM_JOB_ID
mkdir -p "$W" && cd "$W" || exit 1
trap 'rm -rf "$W"' EXIT
# the home directory is not mounted: MIOpen (HF's conv1d in tests/ref_vs_hf.py) needs a
# writable user database and cache, else miopenStatusUnknownError
export HOME=$W PIP_CACHE_DIR=$W/pipcache HF_HOME=$W/hf FLYDSL_RUNTIME_ENABLE_CACHE=0
git clone -q https://github.com/SamuelReeder/megakernels.git src && git -C src checkout -q "$REV" || exit 1
cd src/qwen3.6-35b-a3b-gfx942 || exit 1
MK_ENV=$W/venv bash scripts/setup_env.sh || exit 1
. "$W/venv/bin/activate"
for r in Qwen3.6-35B-A3B-FP8 Qwen3.6-35B-A3B; do
  hf download "Qwen/$r" --local-dir "$W/models/$r" --exclude '*.pth' > /dev/null || exit 1
done
O=bench/results/m5/$(hostname -s)_$SLURM_JOB_ID
bash tests/run_suite.sh "$W/models" "$O"
echo RESULTS-BEGIN
tar -C bench/results/m5 -czf - "$(basename "$O")" | base64
echo RESULTS-END
}
main < /dev/null
exit
