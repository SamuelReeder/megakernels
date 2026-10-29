#!/bin/bash
# Spike rtgeom (disposable): on an AUSTIN node, inside the hipdnn gfx942 image with
# /tmp/rtgeom_<job>/home mounted as the home and the sources in /work/mk (rtrun ships
# them): create ~/mk/venv (common/scripts/setup_env.sh) and download the checkpoint.
set -uo pipefail
export PIP_CACHE_DIR=/tmp/pipcache HF_HOME=/tmp/rtg_hf
echo "home $HOME $(date -Is)"
bash /work/mk/common/scripts/setup_env.sh || exit 1
echo "env done $(date -Is)"
. "$HOME/mk/venv/bin/activate"
hf download amd/Qwen3.8-27B-Quark-AWQ-MXFP4 --local-dir "$HOME/mk/models/Qwen3.8-27B-Quark-AWQ-MXFP4" \
  --exclude "*.pth" > /dev/null || exit 1
du -sh "$HOME/mk/models/Qwen3.8-27B-Quark-AWQ-MXFP4"
echo "model done $(date -Is)"
