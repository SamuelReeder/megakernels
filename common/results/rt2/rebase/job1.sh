#!/bin/bash
# rt2 after the rebase onto main e72a981, on one 2-GPU allocation (run from the tree root in
# the hipdnn gfx942 image with the venv; /work/base: M1 hashes and M7 ISA per gate part).
# GPU 0: common/scripts/gates.sh q8 q6 g (rt_geom defaults), then the rt_geom 0 ISA dumps
# against the same base. GPU 1: vgeom.py (Qwen3.8 rt0 vs rt2 at vLLM's geometry), then
# checks.sh hash (logits hash rt_geom 0 and 2, M = 1, 2, 4, 8) for the three models.
# Output: /work/out/rb1/.
set -u
O=/work/out/rb1
R=$PWD
mkdir -p "$O"
{ date -Is; hostname; uptime; rocm-smi --showuse --showproductname 2>/dev/null | grep -E "^GPU"; cat REVISION
  python -c "import torch, flydsl; print('torch', torch.__version__, 'flydsl', flydsl.__version__)"; } > "$O/env.txt" 2>&1
(
  export HIP_VISIBLE_DEVICES=1
  cd "$R/qwen3.8-27b-gfx942" && python -B ../common/results/rt2/vgeom.py "$M/Qwen3.8-27B-Quark-AWQ-MXFP4" > "$O/vgeom_q38.txt" 2>&1
  echo "vgeom rc $?" >> "$O/progress.txt"
  for m in qwen3.8-27b-gfx942:Qwen3.8-27B-Quark-AWQ-MXFP4:q8 qwen3.6-35b-a3b-gfx942:Qwen3.6-35B-A3B-FP8:q6 gemma4-26b-a4b-gfx942:gemma-4-26B-A4B-it-FP8-dynamic:g; do
    IFS=: read -r d w t <<< "$m"
    cd "$R/$d" && bash ../common/results/rt2/checks.sh "$M/$w" "$O/hash_$t" hash > "$O/hash_$t.log" 2>&1
    echo "hash $t rc $?" >> "$O/progress.txt"
  done
) &
(
  export HIP_VISIBLE_DEVICES=0
  BASE=/work/base MODELS=$M bash common/scripts/gates.sh "$O/gates" q8 q6 g > "$O/gates.log" 2>&1
  echo "gates rc $?" >> "$O/progress.txt"
  for m in qwen3.8-27b-gfx942:Qwen3.8-27B-Quark-AWQ-MXFP4:q8 qwen3.6-35b-a3b-gfx942:Qwen3.6-35B-A3B-FP8:q6 gemma4-26b-a4b-gfx942:gemma-4-26B-A4B-it-FP8-dynamic:g; do
    IFS=: read -r d w t <<< "$m"
    mkdir -p "$O/isa_rt0/$t"
    cd "$R/$d" && MK_RTGEOM=0 MK_CFG=rt_geom=0 bash ../common/scripts/isa_dump "$M/$w" "$O/isa_rt0/$t/isa" | sort > "$O/isa_rt0/$t/isa.txt"
    if diff /work/base/$t/isa.txt "$O/isa_rt0/$t/isa.txt" > "$O/isa_rt0/$t/isa.txt.diff"; then r=PASS; else r=FAIL; fi
    echo "GATE $t isa_rt0=base $r" | tee -a "$O/isa_rt0/summary.txt"
  done
  echo "isa_rt0 done" >> "$O/progress.txt"
) &
wait
date -Is >> "$O/env.txt"
echo JOB1-DONE
