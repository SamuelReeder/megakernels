#!/bin/bash
# Download the Gemma 4 26B-A4B checkpoints to the compute node's local disk.
# The Alola home is at its quota, so the models live in node-local /tmp for the allocation.
# Usage (inside the allocation): bash scripts/stage_models.sh [DEST=/tmp/g4models]
set -euo pipefail
DEST=${1:-/tmp/g4models}
HF=${HF:-$HOME/mk/venv/bin/hf}
mkdir -p "$DEST"
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HOME=/tmp/hfhome
for repo in google/gemma-4-26B-A4B-it RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic; do
  "$HF" download "$repo" --local-dir "$DEST/${repo#*/}" > /dev/null
  echo "$repo -> $DEST/${repo#*/} ($(du -sh "$DEST/${repo#*/}" | cut -f1))"
done
