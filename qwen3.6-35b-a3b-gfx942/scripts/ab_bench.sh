#!/bin/bash
# A/B TPOT of kernel variants in one allocation (bench/bench_decode.py, 3 repeats each).
# Usage: bash scripts/ab_bench.sh OUT_DIR "1 8" base:: bar8:bar_ng=8: lds:lds_red=1:out=1
# A variant is TAG:MK_CFG:MK_CHUNKS. Writes OUT_DIR/bench_TAG_mM.txt and prints the
# median TPOT of every variant.
set -uo pipefail
O=$1
MS=$2
shift 2
F8=${F8:-$HOME/mk/models/Qwen3.6-35B-A3B-FP8}
mkdir -p "$O"
for v in "$@"; do
  IFS=: read -r tag cfg ch <<< "$v"
  for m in $MS; do
    MK_CFG=$cfg MK_CHUNKS=$ch python bench/bench_decode.py "$F8" "$m" 512 256 3 > "$O/bench_${tag}_m$m.txt" 2>&1
    med=$(grep -o 'TPOT [0-9.]*' "$O/bench_${tag}_m$m.txt" | awk '{print $2}' | sort -n | sed -n 2p)
    echo "$tag M=$m median TPOT ${med:-FAILED} ms (MK_CFG=$cfg MK_CHUNKS=$ch)"
  done
done
