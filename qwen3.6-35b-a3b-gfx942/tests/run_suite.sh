#!/bin/bash
# Full verification + benchmark suite (about 1.5 h on one MI300X). Run from the
# qwen3.6-35b-a3b-gfx942 directory inside the hipdnn gfx942 image with the venv of
# scripts/setup_env.sh:
#   bash tests/run_suite.sh MODELS_DIR OUT_DIR
# MODELS_DIR holds Qwen3.6-35B-A3B (BF16) and Qwen3.6-35B-A3B-FP8.
set -uo pipefail
M=$1
O=$2
F8=$M/Qwen3.6-35B-A3B-FP8
mkdir -p "$O"
{
  date -Is
  hostname
  rocminfo | grep -m1 -oE 'gfx[0-9a-f]+'
  cat /sys/class/drm/card*/device/vbios_version 2>/dev/null | sort -u
  python -c "import torch, flydsl; print('torch', torch.__version__, 'hip', torch.version.hip, 'flydsl', flydsl.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).multi_processor_count, 'CUs')"
  git rev-parse HEAD 2>/dev/null || cat REVISION 2>/dev/null
} > "$O/env.txt" 2>&1
python tests/ref_vs_hf.py "$M/Qwen3.6-35B-A3B" "$F8" > "$O/ref_vs_hf.txt" 2>&1
for m in 1 2 4 8; do
  python tests/test_decode.py "$F8" "$m" 40 32 > "$O/test_decode_m$m.txt" 2>&1
done
for m in 1 8; do
  python tests/test_determinism.py "$F8" "$m" 32 > "$O/determinism_m$m.txt" 2>&1
done
python tests/eval_text.py "$F8" "$M/Qwen3.6-35B-A3B" > "$O/eval_text.txt" 2>&1
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
