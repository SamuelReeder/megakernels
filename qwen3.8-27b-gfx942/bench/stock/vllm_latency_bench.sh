#!/bin/bash
# Runs INSIDE the vLLM container on TP gfx942 GPUs (env TP, default 1).
# usage: vllm_latency_bench.sh <model_dir_name> <out_tag> [extra vllm bench latency args...]
# Engine-level decode time per step, the same quantity bench/bench_decode.py reports for
# the megakernel (no HTTP server, no scheduler queueing): `vllm bench latency` (offline
# LLM engine, CUDA graphs, prefix caching off) with a 512-token prompt, once with 1 output
# token and once with 1 + STEPS (default 64); TPOT = (latency difference) / STEPS, at
# batch 1, 2, 4, 8. OUT_ROOT (default ~/mk/stock/results) holds the results.
set -u
MODELS=/home/AMD/sareeder/mk/models
M=$MODELS/$1
TP=${TP:-1}
STEPS=${STEPS:-64}
OUT=${OUT_ROOT:-/home/AMD/sareeder/mk/stock/results}/$2
export XDG_CACHE_HOME=/tmp/xdg VLLM_CACHE_ROOT=/tmp/vllm_cache TRITON_CACHE_DIR=/tmp/triton
shift 2
mkdir -p "$OUT"
# Slurm exports ROCR_VISIBLE_DEVICES; vLLM's ray import rejects it (see vllm_serve_bench.sh).
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
  r=$ROCR_VISIBLE_DEVICES; unset ROCR_VISIBLE_DEVICES
  if [ "$(rocminfo | grep -c 'Name: *gfx')" = 1 ]; then export HIP_VISIBLE_DEVICES=0; else export HIP_VISIBLE_DEVICES=$r; fi
fi
{
  echo "date: $(date -Is)"; echo "node: $(hostname)"
  python3 -c 'import vllm,torch;print("vllm",vllm.__version__);print("torch",torch.__version__,"hip",torch.version.hip)'
  env | grep -E '^(VLLM|HIP|ROCR|HSA|AITER)' | sort
  echo "TP $TP STEPS $STEPS args: $*"
} > "$OUT/env.txt" 2>&1
cat "$OUT/env.txt"

lat() { # <batch> <output_len> <name> [extra args...]
  local b=$1 n=$2 name=$3
  shift 3
  vllm bench latency --model "$M" --tensor-parallel-size $TP --max-model-len 4096 \
    --no-enable-prefix-caching --input-len 512 --output-len $n --batch-size $b \
    --num-iters-warmup 3 --num-iters 10 --output-json "$OUT/$name.json" "$@" > "$OUT/$name.log" 2>&1
}
for B in 1 2 4 8; do
  lat $B 1 b${B}_out1 "$@"
  lat $B $((STEPS + 1)) b${B}_out$((STEPS + 1)) "$@"
  python3 - "$OUT" $B $STEPS << 'PY'
import json, sys
out, b, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
r1 = json.load(open(f"{out}/b{b}_out1.json"))
rn = json.load(open(f"{out}/b{b}_out{n + 1}.json"))
med = lambda r: sorted(r["latencies"])[len(r["latencies"]) // 2]
tpot = (med(rn) - med(r1)) / n * 1e3
print(f"[b{b}] engine TPOT {tpot:.3f} ms (median latency {med(rn):.4f} s with {n + 1} outputs, "
      f"{med(r1):.4f} s with 1)")
json.dump(dict(batch=b, steps=n, tpot_ms=tpot, lat_out1_s=med(r1), lat_outn_s=med(rn)),
          open(f"{out}/b{b}_tpot.json", "w"))
PY
done
