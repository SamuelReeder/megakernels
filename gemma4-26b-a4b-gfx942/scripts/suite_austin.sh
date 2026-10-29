#!/bin/bash
# Milestone 5 suite on an AUSTIN gfx942 node (no shared home). From
# qwen3.6-35b-a3b-gfx942/scripts/suite_austin.sh. Runs inside the hipdnn gfx942 image,
# reads nothing from the home directory, downloads both checkpoints to node /tmp, and
# prints the results as a base64 tar between RESULTS-BEGIN / RESULTS-END.
# Start it from the Alola login node:
#   tmux new-session -d -s g4aus "srun -p defq,miopen -A miopen -J g4aus \
#     --gres=gpu:1 --constraint='GFX942&AUSTIN' --cpus-per-task=16 --mem=400G -t 04:00:00 \
#     --container-image=/cluster/images/hipdnn/hipdnn_latest_gfx942.sqsh \
#     bash -s < ~/mk/g4aus/suite_austin.sh > ~/mk/g4aus/g4aus.log 2>&1"
# Unpack: sed -n '/RESULTS-BEGIN/,/RESULTS-END/p' LOG | sed '1d;$d' | base64 -d | tar -xz
# The body is one function, so bash parses it fully before a command reads stdin.
set -uo pipefail
main() {
REV=${REV:-users/sareeder/gemma4-26b-a4b}
echo "SLURM_JOB_ID=$SLURM_JOB_ID node $(hostname -s) $(date -Is)"
df -h /tmp | tail -1
arch=$(rocminfo | grep -m1 -oE 'gfx[0-9a-f]+')
echo "arch $arch"
[ "$arch" = gfx942 ] || { echo "no gfx942 GPU"; exit 1; }
W=/tmp/g4aus_$SLURM_JOB_ID
mkdir -p "$W" && cd "$W" || exit 1
trap 'rm -rf "$W"' EXIT
# the home directory is not mounted: MIOpen needs a writable user database and cache
export HOME=$W PIP_CACHE_DIR=$W/pipcache HF_HOME=$W/hf FLYDSL_RUNTIME_ENABLE_CACHE=0
git clone -q https://github.com/SamuelReeder/megakernels.git src && git -C src checkout -q "$REV" || exit 1
git -C src rev-parse HEAD > src/gemma4-26b-a4b-gfx942/REVISION
cd src/gemma4-26b-a4b-gfx942 || exit 1
MK_ENV=$W/venv bash ../qwen3.6-35b-a3b-gfx942/scripts/setup_env.sh || exit 1
. "$W/venv/bin/activate"
hf download RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic \
  --local-dir "$W/models/gemma-4-26B-A4B-it-FP8-dynamic" > /dev/null || exit 1
hf download google/gemma-4-26B-A4B-it --local-dir "$W/models/gemma-4-26B-A4B-it" > /dev/null || exit 1
O=$W/out/$(hostname -s)_$SLURM_JOB_ID
bash tests/run_suite.sh "$W/models" "$O"
echo RESULTS-BEGIN
tar -C "$W/out" -czf - "$(basename "$O")" | base64
echo RESULTS-END
}
main < /dev/null
exit
