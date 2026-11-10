#!/bin/bash
# Step 1a (docs/hipdnn-route.md) CPU checks of each vLLM branch, on the workstation:
# ruff (the version in vLLM's pre-commit config) over the model's specialized_decode
# package, and the #61311 conformance kit plus the model's spec tests (VLLM_TARGET_DEVICE=cpu,
# the vLLM test venv). Usage: local_checks.sh VLLM_WORKTREES_DIR OUT_DIR
# Env: RUFF (ruff 0.14.0), PY (python with vLLM's test deps).
V=$1
mkdir -p "$2"
O=$(realpath "$2")
mkdir -p "$O"
RUFF=${RUFF:-/tmp/ruff014/bin/ruff} PY=${PY:-/tmp/vllmtest/bin/python}
for e in qwen38-gfx942-specialized-decode:qwen3_5:test_qwen3_5_specialized_decode.py \
         qwen38-gfx942-tp2:qwen3_5:test_qwen3_5_specialized_decode.py \
         qwen36-gfx942-specialized-decode:qwen3_5_moe:test_qwen3_5_moe_specialized_decode.py \
         gemma4-moe-batch:gemma4:test_gemma4_specialized_decode.py; do
  IFS=: read -r b pkg t <<< "$e"
  tdir=tests/models/${pkg%_moe}
  [ "$pkg" = gemma4 ] && tdir=tests/models/gemma4
  cd "$V/$b" || exit 1
  d=vllm/models/$pkg/amd/specialized_decode
  {
    echo "# $(date -Is) $b $(git rev-parse HEAD) dirty $(git status --porcelain vllm tests | wc -l); $($RUFF --version); $($PY -c 'import torch; print("torch", torch.__version__)')"
    echo "## ruff check $d"; $RUFF check --config pyproject.toml "$d"; echo "rc $?"
    echo "## ruff format --check $d"; $RUFF format --check --config pyproject.toml "$d"; echo "rc $?"
  } > "$O/ruff_$b.txt" 2>&1
  {
    echo "# $(date -Is) $b $(git rev-parse HEAD) dirty $(git status --porcelain vllm tests | wc -l)"
    VLLM_TARGET_DEVICE=cpu PYTHONPATH=$PWD timeout 1200 "$PY" -m pytest -q -p no:cacheprovider \
      "$tdir/$t" tests/models/common/specialized_decode
    echo "rc $?"
  } > "$O/kit_$b.txt" 2>&1
  echo "$b ruff: $(grep -c '^rc 0' "$O/ruff_$b.txt")/2 rc0; kit: $(grep -E 'passed|failed|error' "$O/kit_$b.txt" | tail -1)"
done | tee "$O/summary.txt"
