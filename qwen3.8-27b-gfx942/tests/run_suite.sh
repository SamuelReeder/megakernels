#!/bin/bash
# Full verification + benchmark suite (about 1 h on one MI300X). Run from the repo root
# inside the hipdnn gfx942 image with the venv of scripts/setup_env.sh:
#   bash tests/run_suite.sh MODELS_DIR OUT_DIR
# MODELS_DIR holds Qwen3.8-27B (BF16) and Qwen3.8-27B-Quark-AWQ-MXFP4.
set -uo pipefail
M=$1
O=$2
MX=$M/Qwen3.8-27B-Quark-AWQ-MXFP4
mkdir -p "$O"
python tests/ref_vs_hf.py "$M/Qwen3.8-27B" > "$O/ref_vs_hf.txt" 2>&1
for m in 1 2 4 8; do
  python tests/test_decode.py "$MX" "$m" 64 16 > "$O/test_decode_m$m.txt" 2>&1
done
python tests/eval_text.py "$MX" "$M/Qwen3.8-27B" > "$O/eval_text.txt" 2>&1
for m in 1 8; do
  python tests/test_determinism.py "$MX" "$m" 64 > "$O/determinism_m$m.txt" 2>&1
done
for m in 1 2 4 8; do
  python bench/bench_decode.py "$MX" "$m" 512 64 3 > "$O/bench_m$m.txt" 2>&1
done
for ctx in 2048 4096; do
  python bench/bench_decode.py "$MX" 1 "$ctx" 64 2 > "$O/bench_m1_ctx$ctx.txt" 2>&1
done
for f in "$O"/*.txt; do
  echo "== $f"
  grep -v -e "Loading weights" -e "warn" "$f" | tail -n 4
done
