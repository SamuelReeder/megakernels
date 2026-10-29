#!/bin/bash
# Milestone 3 suite (correctness) and the standalone benchmark. Run from this directory
# inside the hipdnn gfx942 image with the venv of common/scripts/setup_env.sh:
#   bash tests/run_suite.sh MODELS_DIR OUT_DIR [bench]
# MODELS_DIR holds gemma-4-26B-A4B-it (BF16) and gemma-4-26B-A4B-it-FP8-dynamic.
set -uo pipefail
M=$1
O=$2
F8=$M/gemma-4-26B-A4B-it-FP8-dynamic
mkdir -p "$O"
{
  date -Is
  hostname
  echo "slurm job ${SLURM_JOB_ID:-?}"
  rocminfo | grep -m1 -oE 'gfx[0-9a-f]+'
  cat /sys/class/drm/card*/device/vbios_version 2>/dev/null | sort -u
  python -c "import torch, flydsl; print('torch', torch.__version__, 'hip', torch.version.hip, 'flydsl', flydsl.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).multi_processor_count, 'CUs')"
  cat REVISION 2>/dev/null
} > "$O/env.txt" 2>&1
if [ "${3:-}" != bench ]; then
  for ctx in 512 2048 4096; do
    for m in 1 2 4 8; do
      python tests/test_decode.py "$F8" "$m" 30 16 "$ctx" > "$O/test_decode_m${m}_ctx$ctx.txt" 2>&1
    done
  done
  for m in 1 8; do
    python tests/test_determinism.py "$F8" "$m" 1100 > "$O/determinism_m$m.txt" 2>&1
  done
  python tests/eval_text.py "$F8" "$M/gemma-4-26B-A4B-it" 1536 > "$O/eval_text.txt" 2>&1
  python tests/test_attention.py > "$O/test_attention.txt" 2>&1
fi
for m in 1 2 4 8; do
  python bench/bench_decode.py "$F8" "$m" 512 256 3 > "$O/bench_m$m.txt" 2>&1
done
for ctx in 2048 4096; do
  python bench/bench_decode.py "$F8" 1 "$ctx" 256 2 > "$O/bench_m1_ctx$ctx.txt" 2>&1
done
for f in "$O"/*.txt; do
  echo "== $f"
  grep -v -e "Loading weights" -e "warn" "$f" | tail -n 4
done
