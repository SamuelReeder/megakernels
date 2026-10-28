#!/bin/bash
# Disposable: copy the vLLM TP2 worktree (python sources and tests) to the cluster and add
# the nightly image's compiled extensions (taken from the qwen38 overlay ~/mk/vllm-src).
set -euo pipefail
V=/home/sareeder/ROCm-workspace/worktrees/vllm/qwen38-gfx942-tp2
D=/cluster/logs/qwen38-tp2/vllm-src
rsync -a --delete --exclude .git --exclude __pycache__ --exclude '*.so' "$V/vllm" "$V/tests" "$V/pyproject.toml" alola:$D/
ssh -o BatchMode=yes alola "cp ~/mk/vllm-src/vllm/*.so $D/vllm/ && cd $D && echo synced \$(ls vllm/*.so | wc -l) so"
echo "vllm commit $(git -C $V rev-parse --short HEAD)"
