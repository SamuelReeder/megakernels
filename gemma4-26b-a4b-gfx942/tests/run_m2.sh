#!/bin/bash
# Milestone 2 (reference) and attention-stage checks. Run from this directory inside the
# hipdnn gfx942 image with the venv of common/scripts/setup_env.sh:
#   bash tests/run_m2.sh MODELS_DIR OUT_DIR
# MODELS_DIR holds gemma-4-26B-A4B-it (BF16) and gemma-4-26B-A4B-it-FP8-dynamic.
set -uo pipefail
M=$1
O=$2
mkdir -p "$O"
clean() { grep -v -i -e "warn" -e "Loading weights" -e pyxis; }
python tests/ref_vs_hf.py "$M/gemma-4-26B-A4B-it" 1300 fp32 2>&1 | clean > "$O/ref_vs_hf_fp32.txt"
python tests/hf_bf16_vs_fp32.py "$M/gemma-4-26B-A4B-it" 1300 2>&1 | clean > "$O/hf_bf16_vs_fp32.txt"
python tests/ref_vs_hf.py "$M/gemma-4-26B-A4B-it" 1300 bf16 2>&1 | clean > "$O/ref_vs_hf_bf16.txt"
python tests/ref_modes.py "$M/gemma-4-26B-A4B-it-FP8-dynamic" 400 2>&1 | clean > "$O/ref_modes.txt"
python tests/test_attention.py 2>&1 | clean > "$O/test_attention.txt"
for f in "$O"/*.txt; do
  echo "== $f"
  tail -n 3 "$f"
done
