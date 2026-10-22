#!/usr/bin/env bash
# Final comparison: megakernel (bf16, int8; plain and speculative) vs llama.cpp (BF16, Q8_0) and vLLM (bf16)
# decode throughput.
#
#   LLAMA_BENCH=~/src/llama.cpp-latest/build/bin/llama-bench GGUF_DIR=~/models/qwen35 bench/run_all.sh
#
# VLLM_PY points at a separate environment with vLLM installed (skipped if missing); ONLY="vllm" runs
# just that engine (megakernel, spec, llamacpp, vllm).
#
# Other GPU clients (including Windows-side apps under WSL) can steal bandwidth mid-run,
# so every segment is bracketed by a raw DRAM bandwidth check (../common/bw.cu) and redone if
# either check falls below MIN_GBPS.
set -euo pipefail
cd "$(dirname "$0")/.."
LLAMA_BENCH=${LLAMA_BENCH:-$HOME/src/llama.cpp-latest/build/bin/llama-bench}
GGUF_DIR=${GGUF_DIR:-$HOME/models/qwen35}
VLLM_PY=${VLLM_PY:-$HOME/venvs/vllm/bin/python}
ONLY=${ONLY:-megakernel spec llamacpp vllm}
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

want() { [[ " $ONLY " == *" $1 "* ]]; }

if want megakernel; then for q in bf16 int8; do
    segment "megakernel $q" python bench/bench_decode.py --quant "$q" --json "results/decode_$q.json"
done; fi
if want spec; then for q in bf16 int8; do
    segment "megakernel spec $q" python bench/bench_spec.py --quant "$q" --json "results/spec_$q.json"
done; fi
if want llamacpp; then for m in BF16 Q8_0; do
    segment "llama.cpp $m" bash -c "'$LLAMA_BENCH' -m '$GGUF_DIR/Qwen3.5-0.8B-$m.gguf' -p 0 -n 256 \
        -d 0,512,1024,2048,4096,8192,16384,32768 -fa on -ngl 99 -r 3 -o json 2>/dev/null > results/llamacpp_$m.json"
done; fi
if want vllm && [ -x "$VLLM_PY" ]; then
    # vLLM JIT-compiles kernels, so it needs its environment's ninja on PATH
    segment "vLLM bf16" env PATH="$(dirname "$VLLM_PY"):$PATH" "$VLLM_PY" bench/bench_vllm.py --json results/vllm_bf16.json
fi
