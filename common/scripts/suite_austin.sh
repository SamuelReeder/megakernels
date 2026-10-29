#!/bin/bash
# A model's tests/run_suite.sh on an AUSTIN gfx942 node (no shared home). Runs inside the
# hipdnn gfx942 image, reads nothing from the home directory, downloads the model's
# checkpoints to node /tmp, and prints the results as a base64 tar between RESULTS-BEGIN /
# RESULTS-END. MODEL (a model directory name) selects the checkpoints; REV is the
# megakernels commit or branch (default main).
# Start it from the Alola login node (docs/gpu-access.md, "Run a Job"):
#   MODEL=gemma4-26b-a4b-gfx942 tmux new-session -d -s mkaus "srun -p defq,miopen -A miopen \
#     -J mkaus --gres=gpu:1 --constraint='GFX942&AUSTIN' --cpus-per-task=16 --mem=400G \
#     -t 04:00:00 --container-image=/cluster/images/hipdnn/hipdnn_latest_gfx942.sqsh \
#     bash -s < ~/mk/aus/suite_austin.sh > ~/mk/aus/mkaus.log 2>&1"
# Unpack: sed -n '/RESULTS-BEGIN/,/RESULTS-END/p' LOG | sed '1d;$d' | base64 -d | tar -xz
# (qwen3.6-35b-a3b-gfx942/scripts/finish_m5.sh does this for Qwen3.6).
# The body is one function, so bash parses it fully before a command reads stdin.
set -uo pipefail
main() {
REV=${REV:-main}
case ${MODEL:?set MODEL to a model directory name} in
  qwen3.6-35b-a3b-gfx942) CKPTS="Qwen/Qwen3.6-35B-A3B-FP8 Qwen/Qwen3.6-35B-A3B" ;;
  gemma4-26b-a4b-gfx942) CKPTS="RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic google/gemma-4-26B-A4B-it" ;;
  qwen3.8-27b-gfx942) CKPTS="amd/Qwen3.8-27B-Quark-AWQ-MXFP4 Qwen/Qwen3.8-27B" ;;
  *) echo "unknown MODEL $MODEL"; exit 1 ;;
esac
echo "SLURM_JOB_ID=$SLURM_JOB_ID node $(hostname -s) $(date -Is) model $MODEL rev $REV"
df -h /tmp | tail -1
arch=$(rocminfo | grep -m1 -oE 'gfx[0-9a-f]+')
echo "arch $arch"
[ "$arch" = gfx942 ] || { echo "no gfx942 GPU"; exit 1; }
W=/tmp/mkaus_$SLURM_JOB_ID
mkdir -p "$W" && cd "$W" || exit 1
trap 'rm -rf "$W"' EXIT
# the home directory is not mounted: MIOpen (HF's conv1d in tests/ref_vs_hf.py) needs a
# writable user database and cache, else miopenStatusUnknownError
export HOME=$W PIP_CACHE_DIR=$W/pipcache HF_HOME=$W/hf FLYDSL_RUNTIME_ENABLE_CACHE=0
git clone -q https://github.com/SamuelReeder/megakernels.git src && git -C src checkout -q "$REV" || exit 1
git -C src rev-parse HEAD > "src/$MODEL/REVISION"
cd "src/$MODEL" || exit 1
MK_ENV=$W/venv bash ../common/scripts/setup_env.sh || exit 1
. "$W/venv/bin/activate"
for r in $CKPTS; do
  hf download "$r" --local-dir "$W/models/${r#*/}" --exclude '*.pth' > /dev/null || exit 1
done
O=$W/out/$(hostname -s)_$SLURM_JOB_ID
bash tests/run_suite.sh "$W/models" "$O"
echo RESULTS-BEGIN
tar -C "$W/out" -czf - "$(basename "$O")" | base64
echo RESULTS-END
}
main < /dev/null
exit
