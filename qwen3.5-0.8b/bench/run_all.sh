#!/usr/bin/env bash
# Final comparison: megakernel (bf16, int8) vs llama.cpp (BF16, Q8_0) decode throughput.
#
#   LLAMA_BENCH=~/src/llama.cpp-latest/build/bin/llama-bench GGUF_DIR=~/models/qwen35 bench/run_all.sh
#
# Other GPU clients (including Windows-side apps under WSL) can steal bandwidth mid-run,
# so every segment is bracketed by a raw DRAM bandwidth check (../common/bw.cu) and redone if
# either check falls below MIN_GBPS.
set -euo pipefail
cd "$(dirname "$0")/.."
LLAMA_BENCH=${LLAMA_BENCH:-$HOME/src/llama.cpp-latest/build/bin/llama-bench}
GGUF_DIR=${GGUF_DIR:-$HOME/models/qwen35}
MIN_GBPS=${MIN_GBPS:-880}
TRIES=${TRIES:-6}
mkdir -p build results
[ -x build/bw ] || nvcc -O3 -std=c++17 -gencode=arch=compute_120a,code=sm_120a ../common/bw.cu -o build/bw

bw() { ./build/bw | awk '/tma ring  stages=4 chunk=16384B/ {print int($(NF-1))}'; }

segment() {  # segment <name> <command...>
    local name=$1; shift
    for try in $(seq 1 "$TRIES"); do
        local before after
        before=$(bw)
        "$@"
        after=$(bw)
        if [ "$before" -ge "$MIN_GBPS" ] && [ "$after" -ge "$MIN_GBPS" ]; then
            echo "[$name] ok (raw bandwidth $before / $after GB/s)"
            return 0
        fi
        echo "[$name] GPU contended ($before / $after GB/s), retrying in 30 s ($try/$TRIES)"
        sleep 30
    done
    echo "[$name] gave up: GPU never quiet" >&2
    return 1
}

for q in bf16 int8; do
    segment "megakernel $q" python bench/bench_decode.py --quant "$q" --json "results/decode_$q.json"
done
for m in BF16 Q8_0; do
    segment "llama.cpp $m" bash -c "'$LLAMA_BENCH' -m '$GGUF_DIR/Qwen3.5-0.8B-$m.gguf' -p 0 -n 256 \
        -d 0,512,2048,4096 -fa on -ngl 99 -r 3 -o json 2>/dev/null > results/llamacpp_$m.json"
done
