#!/bin/bash
# Before / after TPOT and stage profiles on one node in one allocation (run from the
# branch's sync folder by scripts/mkrun). Before: main 13b25e7, exported to
# ~/mk/fixed-stage-cost-main. Profiles of "before" use this branch with the old
# settings (cta_red=0, skip_idle=0, out_proj K chunks 2): the same code path as main,
# plus PSTEPS.
set -u
A=$PWD
B=$HOME/mk/fixed-stage-cost-main
O=$A/bench/results/fsc/final
F=$HOME/mk/models/Qwen3.6-35B-A3B-FP8
mkdir -p "$O"
{
  date -Is
  hostname
  rocminfo | grep -m1 -oE 'gfx[0-9a-f]+'
  python -c "import torch, flydsl; print('torch', torch.__version__, 'flydsl', flydsl.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).multi_processor_count, 'CUs')"
  echo "after: $(cat REVISION)  before: 13b25e7 (main)"
  echo "FLYDSL_RUNTIME_ENABLE_CACHE=$FLYDSL_RUNTIME_ENABLE_CACHE"
} > "$O/env.txt" 2>&1
for m in 1 2 4 8; do
  (cd "$B" && python bench/bench_decode.py "$F" $m 512 256 3) > "$O/bench_before_m$m.txt" 2>&1
  python bench/bench_decode.py "$F" $m 512 256 3 > "$O/bench_after_m$m.txt" 2>&1
done
for m in 1 8; do
  MK_CFG=cta_red=0,skip_idle=0 MK_CHUNKS=out=2 PSTEPS=3 python bench/profile_decode.py "$F" $m 512 \
    "$O/tl_before_m$m.json" > "$O/profile_before_m$m.txt" 2>&1
  PSTEPS=3 python bench/profile_decode.py "$F" $m 512 "$O/tl_after_m$m.json" > "$O/profile_after_m$m.txt" 2>&1
done
date -Is >> "$O/env.txt"
for f in "$O"/bench_*.txt; do echo "$(basename "$f") $(grep -o 'TPOT [0-9.]*' "$f" | awk '{print $2}' | sort -n | sed -n 2p)"; done
