#!/bin/bash
# Integration runs of the fixed-stage-cost branch in one 1-GPU allocation (run on the
# Alola login node, inside tmux): bash scripts/fsc_finish.sh JOBID [PHASES]
# PHASES: any of "hash vllm headline" (default: all three, in this order).
#  1. logits hashes of the kernel before the cache-key commit (OLD, same tests) and the
#     full gates of this head (they print the same hashes)
#  2. vLLM branch (~/mk/vllm36-src on PYTHONPATH; it needs the image's vllm/*.so copied
#     in): kernel on / off, TRITON_ATTN and
#     ROCM_ATTN, greedy tokens and TPOT at batch 1, 2, 4, 8 (bench/vllm_e2e.py)
#  3. headline: kernel (bench_decode, median of 3) and stock vLLM FP8 with and without
#     AITER (bench/stock/vllm_serve_bench.sh, as in the README) at batch 1, 2, 4, 8.
#     GPU use of all 8 GPUs before and after each measurement; up to 3 attempts until
#     the other GPUs are idle.
set -u
J=$1
P=" ${2:-hash vllm headline} "
A=$HOME/mk/fixed-stage-cost-src
OLD=$HOME/mk/fixed-stage-cost-old
O=$A/bench/results/fsc/finish
F=$HOME/mk/models/Qwen3.6-35B-A3B-FP8
MKIMG=/cluster/images/hipdnn/hipdnn_latest_gfx942.sqsh
VLIMG=$HOME/mk/images/vllm-rocm-nightly-00b6c4c5.sqsh
MNT=/cluster:/cluster,/home/AMD/sareeder:/home/AMD/sareeder
mkdir -p "$O/hash_old" "$O/vllm" "$O/headline"
node=$(squeue -h -j "$J" -o %N)
# own GPU: its PCI address (in the job's cgroup amd-smi shows only this GPU)
own=$(srun --jobid="$J" --overlap -n1 amd-smi list --csv | awk -F, 'NR == 2 {print $2}')
echo "job $J node $node own GPU $own head $(cat "$A/REVISION")" | tee "$O/env.txt"

mk() { # <dir> <command>
  srun --jobid="$J" --overlap -n1 --container-image=$MKIMG --container-mounts=$MNT \
    --container-name=fscmk bash -lc "export PIP_CACHE_DIR=/tmp/pipcache \
    FLYDSL_RUNTIME_ENABLE_CACHE=0; . \$HOME/mk/venv/bin/activate; cd $1 && $2" 2>&1
}
vl() { # <command>, in the vLLM image
  srun --jobid="$J" --overlap -n1 --container-image="$VLIMG" --container-mounts=$MNT \
    --container-writable --container-name=fscvl bash -lc "export \
    PIP_CACHE_DIR=/tmp/pipcache FLYDSL_RUNTIME_ENABLE_CACHE=0 HF_HUB_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false; cd $A && $1" 2>&1
}
smi() { # <log>: busy % of every GPU on the node (sysfs: the cgroup hides the others from
  # amd-smi), 5 samples 1 s apart; prints the highest busy % of the other GPUs
  srun --jobid="$J" --overlap -n1 bash -c 'date -Is; for s in 1 2 3 4 5; do
    for c in /sys/class/drm/card*/device; do [ -f $c/gpu_busy_percent ] &&
      echo "$(basename $(readlink -f $c)) busy $(cat $c/gpu_busy_percent) vram_used" \
        "$(cat $c/mem_info_vram_used)"; done; sleep 1; done' > "$1.tmp" 2>&1
  cat "$1.tmp" >> "$1"
  awk -v own="$own" '$2 == "busy" && $1 != own && $3 > m {m = $3} END {print m + 0}' "$1.tmp"
  rm -f "$1.tmp"
}

# 1. hashes before the cache-key commit, then the gates of this head
if [[ $P == *" hash "* ]]; then
{ echo "OLD tree: $(cat "$OLD/REVISION") mk/ with this head's tests/"; date -Is; } > "$O/hash_old/env.txt"
for m in 1 8; do
  mk "$OLD" "python tests/test_decode.py $F $m 40 32" > "$O/hash_old/test_decode_m$m.txt"
  mk "$OLD" "python tests/test_determinism.py $F $m 32" > "$O/hash_old/determinism_m$m.txt"
done
mk "$A" "bash scripts/fsc_gates.sh bench/results/fsc/finish/gates"
grep -H "sha256" "$O"/hash_old/*.txt "$O"/gates/test_decode_m[18].txt "$O"/gates/determinism_m*.txt
fi

# 2. vLLM branch, kernel on / off
if [[ $P == *" vllm "* ]]; then
for attn in TRITON_ATTN ROCM_ATTN; do
  for mode in off on; do
    vl "PYTHONPATH=\$HOME/mk/vllm36-src MK_ATTN=$attn python bench/vllm_e2e.py run $F $mode \
      $O/vllm/${mode}_$attn.json" > "$O/vllm/${mode}_$attn.log"
    grep -E "TPOT|Error|error:" "$O/vllm/${mode}_$attn.log" | tail -5
  done
  vl "python bench/vllm_e2e.py compare $O/vllm/off_$attn.json $O/vllm/on_$attn.json" \
    > "$O/vllm/compare_$attn.txt"
  tail -5 "$O/vllm/compare_$attn.txt"
done
fi

# 3. headline on a quiet node
if [[ $P == *" headline "* ]]; then
for a in 1 2 3; do
  H=$O/headline/a$a
  mkdir -p "$H"
  L=$H/smi.txt
  worst=0
  note() { echo "== $1" >> "$L"; u=$(smi "$L"); [ "$u" -gt "$worst" ] && worst=$u; }
  note "before kernel"
  for m in 1 2 4 8; do
    mk "$A" "python bench/bench_decode.py $F $m 512 256 3" > "$H/bench_m$m.txt"
  done
  note "after kernel"
  for v in aiter plain; do
    note "before vllm $v"
    tag=fsc_a${a}_vllm_$v
    env_aiter=""; [ $v = aiter ] && env_aiter="VLLM_ROCM_USE_AITER=1"
    vl "$env_aiter bash bench/stock/vllm_serve_bench.sh Qwen3.6-35B-A3B-FP8 $tag" > "$H/vllm_$v.txt"
    cp -r "$HOME/mk/stock36/results/$tag" "$H/vllm_$v"
    rm -rf "$HOME/mk/stock36/results/$tag"
    note "after vllm $v"
  done
  echo "attempt $a: highest GPU use of the other GPUs $worst %" | tee -a "$O/headline/attempts.txt"
  for m in 1 2 4 8; do
    echo "kernel m$m $(grep -o 'TPOT [0-9.]*' "$H/bench_m$m.txt" | awk '{print $2}' | sort -n | sed -n 2p)"
  done | tee -a "$O/headline/attempts.txt"
  [ "$worst" -le 5 ] && break
  [ $a -lt 3 ] && sleep 600
done
fi
echo "finished $(date -Is)"
