#!/bin/bash
# Reproducibility run: correctness smoke + decode benchmark at M = 1, 2, 4, 8 on the
# node this runs on. Run inside the hipdnn gfx942 image with the venv of
# scripts/setup_env.sh, from the repository root:
#   bash bench/repro.sh MODEL_DIR OUT_DIR
set -uo pipefail
MODEL=$1
OUT=$2
mkdir -p "$OUT"
{
  date -Is
  hostname
  rocminfo | grep -m1 -oE 'gfx[0-9a-f]+'
  rocm-smi --showproductname | grep -m1 -E 'Card (series|SKU)|GPU' || true
  python -c "import torch, flydsl; print('torch', torch.__version__, 'hip', torch.version.hip, 'flydsl', flydsl.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).multi_processor_count, 'CUs')"
  git rev-parse HEAD 2>/dev/null || cat REVISION 2>/dev/null
} > "$OUT/env.txt" 2>&1
python tests/test_decode.py "$MODEL" 1 64 8 > "$OUT/test_decode_m1.txt" 2>&1
python tests/test_determinism.py "$MODEL" 1 32 > "$OUT/determinism_m1.txt" 2>&1
for m in 1 2 4 8; do
  python bench/bench_decode.py "$MODEL" "$m" 512 64 3 > "$OUT/bench_m$m.txt" 2>&1
done
tail -n 2 "$OUT"/*.txt
