#!/bin/bash
# Runs INSIDE the vLLM container. Benchmarks each checkpoint once its download is complete,
# preferring the order given (default: FP8, BF16).
# usage: vllm_all.sh [ckpt ...]
set -u
D=$(dirname "$(readlink -f "$0")")
export HF_HUB_OFFLINE=1
pending=("$@")
[ ${#pending[@]} = 0 ] && pending=(gemma-4-26B-A4B-it-FP8-dynamic gemma-4-26B-A4B-it)

ready() { # complete = config + every shard named in the index (or model.safetensors) present, no partial downloads
  python3 - "/home/AMD/sareeder/mk/models/$1" << 'EOF'
import json, os, sys, glob
d = sys.argv[1]
idx = os.path.join(d, "model.safetensors.index.json")
if glob.glob(os.path.join(d, ".cache/huggingface/download/**/*.incomplete"), recursive=True): sys.exit(1)
if not os.path.exists(os.path.join(d, "config.json")): sys.exit(1)
shards = set(json.load(open(idx))["weight_map"].values()) if os.path.exists(idx) else {"model.safetensors"}
sys.exit(0 if all(os.path.exists(os.path.join(d, s)) for s in shards) else 1)
EOF
}

idle=0
while [ ${#pending[@]} -gt 0 ]; do
  ran=0
  for i in "${!pending[@]}"; do
    c=${pending[$i]}
    ready "$c" || continue
    echo "===== $c $(date -Is)"
    bash "$D/vllm_serve_bench.sh" "$c" "vllm${VARIANT:+_$VARIANT}_$c"
    unset "pending[$i]"; pending=("${pending[@]}"); ran=1; idle=0; break
  done
  if [ $ran = 0 ]; then # do not hold the GPU idle for more than 30 min waiting for downloads
    idle=$((idle + 1)); [ $idle -gt 30 ] && { echo "still not downloaded: ${pending[*]}"; exit 3; }
    sleep 60
  fi
done
