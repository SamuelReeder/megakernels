#!/bin/bash
# M8 open item: vLLM TPOT of the pre-refactor exports (base) vs main's exports (new), 3
# alternating rounds, kernel on, greedy, B = 1, 2, 4, 8, per model (Qwen3.6 TRITON_ATTN,
# Gemma, Qwen3.8 TP1 and TP2), plus Qwen3.8 TP1 on the rt2 vLLM export (users/sareeder/rt2,
# for its output modes). Runs from this host: NRUN ships this megakernels tree (its
# bench/vllm_e2e.py) and one vLLM tree per run to the held 2-GPU allocation; before each run
# it records the node load (the node's jobs and GPUs, uptime, rocm-smi of the allocation).
# Usage: drive.sh OUT_DIR   (env NRUN, JOB, NODE, VB (pre-refactor vLLM trees), V, VRT2)
set -u
O=$1
mkdir -p "$O"
T=$(cd "$(dirname "$0")/../../.." && pwd)
load() {
  echo "LOAD $1 $(date -u +%FT%TZ)"
  ssh -o BatchMode=yes alola "squeue -h -w $NODE -o '%i %j %u %b %C'; srun --jobid=$JOB --overlap -n1 --time=2 bash -c 'uptime; rocm-smi --showuse --showmemuse 2>/dev/null | grep -E \"^GPU\"'" 2>&1
}
run() {  # run TAG VDIR VREV CMD
  local tag=$1 vd=$2 vr=$3; shift 3
  { load "$tag"; VLLM_REV=$vr MK_TIME=40 $NRUN vllm "$T" "$vd" "mkdir -p /work/out/m8q && $*"; echo "rc $?"; } > "$O/$tag.log" 2>&1
  grep -hE "^LOAD|load average|^GPU\[|^[0-9]+ |TPOT|^rc|Error|error:" "$O/$tag.log" | grep -v "^\s*$" | cut -c1-200
}
Q6='cd qwen3.6-35b-a3b-gfx942 && MK_ATTN=TRITON_ATTN python bench/vllm_e2e.py run $M/Qwen3.6-35B-A3B-FP8 on'
G='cd gemma4-26b-a4b-gfx942 && python bench/vllm_e2e.py run $M/gemma-4-26B-A4B-it-FP8-dynamic on'
Q8='cd qwen3.8-27b-gfx942 && python bench/vllm_e2e.py $M/Qwen3.8-27B-Quark-AWQ-MXFP4 on'
for r in 1 2 3; do
  if [ $((r % 2)) = 1 ]; then order="base new"; else order="new base"; fi
  for k in $order; do
    if [ $k = base ]; then q6=$VB/q6 g=$VB/g q8=$VB/q8 r6=bdf23a462 rg=6d0ec44e8 r8=6220b7379
    else q6=$V/qwen36-gfx942-specialized-decode g=$V/gemma4-moe-batch q8=$V/qwen38-gfx942-tp2 r6=70f57a0ae rg=7f586cd2a r8=fd8a77a5f; fi
    run "q6_${k}_r$r" $q6 $r6 "$Q6 /work/out/m8q/q6_${k}_r$r.json"
    run "g_${k}_r$r" $g $rg "$G /work/out/m8q/g_${k}_r$r.json"
    run "q8tp1_${k}_r$r" $q8 $r8 "HIP_VISIBLE_DEVICES=0 $Q8 /work/out/m8q/q8tp1_${k}_r$r.json 128"
    [ $k = new ] && run "q8tp1_rt2_r$r" $VRT2 aa9c7ae58 "HIP_VISIBLE_DEVICES=0 $Q8 /work/out/m8q/q8tp1_rt2_r$r.json 128"
    run "q8tp2_${k}_r$r" $q8 $r8 "MK_TP=2 $Q8 /work/out/m8q/q8tp2_${k}_r$r.json 128"
  done
done
ssh -o BatchMode=yes alola "srun --jobid=$JOB --overlap -n1 --time=3 bash -c 'tar -C /tmp/rt2_$JOB/out -cf - m8q'" | tar -C "$O" -xf -
echo DRIVE-DONE
