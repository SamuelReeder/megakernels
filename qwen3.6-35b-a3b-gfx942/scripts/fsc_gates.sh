#!/bin/bash
# Hard gates of the branch head (run from the branch's sync folder by scripts/mkrun):
# kernel vs reference at M = 1, 2, 4, 8, determinism and batch invariance, real-text
# KL against BF16, and the timing-perturbation run (Cfg.chaos) bitwise against no chaos.
# Usage: scripts/fsc_gates.sh [OUT_DIR]  (default bench/results/fsc/final/gates)
set -u
O=${1:-bench/results/fsc/final/gates}
F=$HOME/mk/models/Qwen3.6-35B-A3B-FP8
BF=$HOME/mk/models/Qwen3.6-35B-A3B
mkdir -p "$O"
{
  date -Is
  hostname
  rocminfo | grep -m1 -oE 'gfx[0-9a-f]+'
  echo "branch head: $(cat REVISION)  FLYDSL_RUNTIME_ENABLE_CACHE=$FLYDSL_RUNTIME_ENABLE_CACHE"
} > "$O/env.txt" 2>&1
for m in 1 2 4 8; do
  python tests/test_decode.py "$F" $m 40 32 > "$O/test_decode_m$m.txt" 2>&1
done
for m in 1 8; do
  python tests/test_determinism.py "$F" $m 32 > "$O/determinism_m$m.txt" 2>&1
  python tests/test_chaos.py "$F" $m 16 > "$O/chaos_m$m.txt" 2>&1
done
python tests/eval_text.py "$F" "$BF" > "$O/eval_text.txt" 2>&1
date -Is >> "$O/env.txt"
for f in "$O"/*.txt; do
  echo "== $f"
  grep -v -e "Loading weights" -e "warn" "$f" | tail -n 3
done
