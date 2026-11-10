#!/bin/bash
# Step 8 (docs/hipdnn-route.md) CPU checks of each vLLM branch, on the workstation:
# ruff (the version in vLLM's pre-commit config) over the files that the hipDNN launch
# backend commit touches, and the #61311 conformance kit plus the model's spec tests
# (VLLM_TARGET_DEVICE=cpu, the vLLM test venv with flydsl 0.4.1 added).
# Usage: local_checks.sh VLLM_WORKTREES_DIR OUT_DIR
# Env: RUFF (ruff 0.14.0), PY (python with vLLM's test deps).
V=$1
mkdir -p "$2"
O=$(realpath "$2")
RUFF=${RUFF:-/tmp/ruff014/bin/ruff} PY=${PY:-/tmp/vllmtest/bin/python}
for e in qwen38-gfx942-specialized-decode:qwen3_5:tests/models/qwen3_5/test_qwen3_5_specialized_decode.py \
         qwen38-gfx942-tp2:qwen3_5:tests/models/qwen3_5/test_qwen3_5_specialized_decode.py \
         qwen36-gfx942-specialized-decode:qwen3_5_moe:tests/models/qwen3_5/test_qwen3_5_moe_specialized_decode.py \
         gemma4-moe-batch:gemma4:tests/models/gemma4/test_gemma4_specialized_decode.py; do
  IFS=: read -r b pkg t <<< "$e"
  cd "$V/$b" || exit 1
  files="vllm/models/$pkg/amd/specialized_decode vllm/models/common/specialized_decode $t vllm/envs.py"
  {
    echo "# $(date -Is) $b $(git rev-parse HEAD) dirty $(git status --porcelain vllm tests | wc -l); $($RUFF --version)"
    echo "## ruff check"; $RUFF check --config pyproject.toml $files; echo "rc $?"
    echo "## ruff format --check"; $RUFF format --check --config pyproject.toml $files; echo "rc $?"
  } > "$O/ruff_$b.txt" 2>&1
  {
    echo "# $(date -Is) $b $(git rev-parse HEAD) dirty $(git status --porcelain vllm tests | wc -l); $($PY -c 'import torch; print("torch", torch.__version__)')"
    VLLM_TARGET_DEVICE=cpu PYTHONPATH=$PWD timeout 1200 "$PY" -m pytest -v -p no:cacheprovider \
      "$t" tests/models/common/specialized_decode
    echo "rc $?"
  } > "$O/kit_$b.txt" 2>&1
  echo "$b ruff: $(grep -c '^rc 0' "$O/ruff_$b.txt")/2 rc0; hipdnn tests: $(grep -c 'test_hipdnn.*PASSED\|test_default_launch.*PASSED' "$O/kit_$b.txt") passed; kit: $(grep -E 'passed|failed|error' "$O/kit_$b.txt" | tail -1)"
done | tee "$O/summary.txt"
