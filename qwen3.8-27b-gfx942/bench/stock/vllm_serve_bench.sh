#!/bin/bash
# Runs INSIDE the vLLM container on TP gfx942 GPUs (env TP, default 1).
# usage: vllm_serve_bench.sh <model_dir_name> <out_tag> [extra vllm serve args...]
# OUT_ROOT (default ~/mk/stock/results) holds the results; caches go to /tmp.
# Starts `vllm serve` (TP $TP, graphs on = default, prefix caching off), then runs
# `vllm bench serve` (random 512 in / 256 out, ignore-eos) at max-concurrency 1,2,4,8,
# plus 2 extra concurrency-1 repeats (3 total) for run-to-run variation.
set -u
MODELS=/home/AMD/sareeder/mk/models
M=$MODELS/$1
TP=${TP:-1}
OUT=${OUT_ROOT:-/home/AMD/sareeder/mk/stock/results}/$2
export XDG_CACHE_HOME=/tmp/xdg VLLM_CACHE_ROOT=/tmp/vllm_cache TRITON_CACHE_DIR=/tmp/triton
shift 2
mkdir -p "$OUT"
PORT=$((20000 + ${SLURM_JOB_ID:-0} % 10000)) # per-job port: jobs on one node share the host network
# Slurm exports ROCR_VISIBLE_DEVICES; vLLM's ray import rejects it. Translate to HIP_VISIBLE_DEVICES
# (index 0 if the cgroup already exposes only one GPU).
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
  r=$ROCR_VISIBLE_DEVICES; unset ROCR_VISIBLE_DEVICES
  if [ "$(rocminfo | grep -c 'Name: *gfx')" = 1 ]; then export HIP_VISIBLE_DEVICES=0; else export HIP_VISIBLE_DEVICES=$r; fi
fi

{
  echo "date: $(date -Is)"; echo "node: $(hostname)"
  echo "arch: $(rocminfo | grep -m1 -oE 'gfx[0-9a-f]+')"
  python3 -c 'import torch;p=torch.cuda.get_device_properties(0);print("gpu",torch.cuda.get_device_name(0),p.gcnArchName,round(p.total_memory/2**30),"GiB","count",torch.cuda.device_count())'
  echo "rocm sdk: $(pip list 2>/dev/null | grep -iE '^(rocm|_rocm|amd-aiter|triton)' | tr -s ' ' | paste -sd';')"
  python3 -c 'import vllm,torch;print("vllm",vllm.__version__);print("torch",torch.__version__,"hip",torch.version.hip)'
  python3 -c 'import aiter;print("aiter",getattr(aiter,"__version__","?"))' 2>&1 | tail -1
  env | grep -E '^(VLLM|HIP|ROCR|HSA|AITER)' | sort
  echo "serve args: $*"
} > "$OUT/env.txt" 2>&1
cat "$OUT/env.txt"

vllm serve "$M" --served-model-name m --tensor-parallel-size $TP --max-model-len 4096 \
  --no-enable-prefix-caching --language-model-only --port $PORT "$@" > "$OUT/server.log" 2>&1 &
PID=$!
ready=0
for _ in $(seq 480); do
  kill -0 $PID 2> /dev/null || break
  if curl -sf localhost:$PORT/v1/models | grep -q "\"root\": *\"$M\""; then ready=1; break; fi
  sleep 5
done
if [ $ready != 1 ]; then
  echo "SERVER FAILED ($M)"; tail -60 "$OUT/server.log"; kill $PID 2> /dev/null; wait $PID; exit 1
fi
echo "server ready $(date -Is)"

bench() { # <concurrency> <num_prompts> <name>
  vllm bench serve --backend vllm --model m --tokenizer "$M" --port $PORT \
    --dataset-name random --random-input-len 512 --random-output-len 256 --ignore-eos \
    --max-concurrency $1 --num-prompts $2 --seed $RANDOM \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir "$OUT" --result-filename "$3.json" > "$OUT/$3.log" 2>&1
  grep -E 'Median (TPOT|ITL|TTFT)|Output token throughput' "$OUT/$3.log" | sed "s/^/[$3] /"
}

bench 8 16 warmup
for B in 1 2 4 8; do bench $B $((B * 8)) c$B; done
bench 1 8 c1_rep2
bench 1 8 c1_rep3

kill $PID; wait $PID 2> /dev/null
echo "done $M $(date -Is)"
