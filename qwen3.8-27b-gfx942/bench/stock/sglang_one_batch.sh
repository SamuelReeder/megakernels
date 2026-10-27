#!/bin/bash
# Runs INSIDE the SGLang container on one gfx942 GPU.
# usage: sglang_one_batch.sh <model_dir_name> [extra sglang server args...]
# sglang.bench_one_batch: static batch, input 512, output 256, batch 1,2,4,8; then batch 1 twice more.
set -u
M=/home/AMD/sareeder/mk/models/$1
OUT=/home/AMD/sareeder/mk/stock/results/sglang_$1
shift
mkdir -p "$OUT"
cd "$OUT"
{
  echo "date: $(date -Is)"; echo "node: $(hostname)"
  echo "arch: $(rocminfo | grep -m1 -oE 'gfx[0-9a-f]+')"
  rocm-smi --showproductname 2>/dev/null | grep -E 'Card (Series|Model)|GFX' | head -4
  echo "rocm: $(cat /opt/rocm/.info/version 2>/dev/null)"
  python3 -c 'import sglang,torch;print("sglang",sglang.__version__);print("torch",torch.__version__,"hip",torch.version.hip)'
  env | grep -E '^(SGLANG|HIP|ROCR|HSA|AITER)' | sort
  echo "extra args: $*"
} > env.txt 2>&1
cat env.txt
run() { # <name> <batch sizes...>
  local n=$1; shift
  python3 -m sglang.bench_one_batch --model-path "$M" --tp-size 1 --batch-size "$@" \
    --input-len 512 --output-len 256 --run-name "$n" --result-filename "$OUT/$n.jsonl" \
    "${EXTRA[@]}" > "$OUT/$n.log" 2>&1
  echo "[$n] exit $?"; grep -E 'Decode\.|Prefill\.|Benchmark|Error|error' "$OUT/$n.log" | tail -20
}
EXTRA=("$@")
run sweep 1 2 4 8
run b1_rep2 1
run b1_rep3 1
