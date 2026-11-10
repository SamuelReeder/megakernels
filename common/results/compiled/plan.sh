#!/bin/bash
# gpu.sh's plan, on the Alola login node (in tmux): srun steps into the held allocation JOB.
# Env: JOB, W (node /tmp tree dir), O (~/mk/out/cc_JOB, shared home).
set -u
VIMG=/home/AMD/sareeder/mk/images/vllm-rocm-nightly-00b6c4c5.sqsh
HIMG=/cluster/images/hipdnn/hipdnn_latest_gfx942.sqsh
MNT=/cluster:/cluster,/home/AMD/sareeder:/home/AMD/sareeder,/tmp:/tmp
HENV="export PIP_CACHE_DIR=/tmp/pipcache FLYDSL_RUNTIME_ENABLE_CACHE=0 PYTHONDONTWRITEBYTECODE=1; . \$HOME/mk/venv/bin/activate"
vl() {  # vl GPUS SPEC...: one vLLM lane
  local g=$1; shift
  srun --jobid="$JOB" --overlap -n1 --time=170 --container-image=$VIMG --container-writable \
    --container-mounts=$MNT bash -lc "export HIP_VISIBLE_DEVICES=$g; bash $O/vlane.sh $W $O $*"
}
gl() {  # gl N GPUS TREE PARTS: gates.sh in the hipdnn image with the venv
  srun --jobid="$JOB" --overlap -n"$1" --time=150 --container-image=$HIMG --container-mounts=$MNT \
    bash -lc "$HENV; export HIP_VISIBLE_DEVICES=$2; cd $W/$3 && common/scripts/gates.sh $O/gates_$3 $4"
}
echo "start $(date -Is) job $JOB"
srun --jobid="$JOB" --overlap -n1 --time=2 bash -c 'hostname; rocminfo | grep -m1 -oE "gfx[0-9a-f]+"; rocm-smi --showproductname | grep -m1 -i "card series"; uptime'
# 1. greedy kernel-on smoke: new heads vs previous heads; rt2 trees at 4096 and 8192 with ISA dumps
vl 0 q8new_4096:q8new:q8:4096:1:1 q8prev_4096:q8prev:q8:4096:1:0 q8tp1prev_4096:q8tp1prev:q8:4096:1:0 \
  q6new_4096:q6new:q6:4096:1:1 q6prev_4096:q6prev:q6:4096:1:0 gnew_4096:gnew:g:4096:1:1 > "$O/lane0.log" 2>&1 &
vl 1 q8new_8192:q8new:q8:8192:1:1 q8prev_8192:q8prev:q8:8192:1:0 q8tp1prev_8192:q8tp1prev:q8:8192:1:0 \
  q6new_8192:q6new:q6:8192:1:1 q6prev_8192:q6prev:q6:8192:1:0 gprev_4096:gprev:g:4096:1:0 > "$O/lane1.log" 2>&1 &
wait
echo "vllm tp1 done $(date -Is)"
# 2. Qwen3.8 TP2 (rt_geom 0 at TP2): new vs previous head
vl 0,1 q8new_tp2_4096:q8new:q8:4096:2:0 q8prev_tp2_4096:q8prev:q8:4096:2:0 > "$O/lane_tp2.log" 2>&1
echo "vllm tp2 done $(date -Is)"
# 3. megakernels gates: this branch (GPU 0) and main (GPU 1)
gl 1 0 mk_new "q8 q6 g" > "$O/gates_mk_new.log" 2>&1 &
gl 1 1 mk_main "q8 q6 g" > "$O/gates_mk_main.log" 2>&1 &
wait
echo "gates done $(date -Is)"
for t in mk_new mk_main; do gl 2 0,1 $t q8tp2 > "$O/gates_${t}_q8tp2.log" 2>&1; done
echo "gates q8tp2 done $(date -Is)"
for p in q8 q6 g q8tp2; do
  for f in hashes.txt isa.txt; do
    if diff "$O/gates_mk_main/$p/$f" "$O/gates_mk_new/$p/$f" > /dev/null 2>&1; then r=PASS; else r=FAIL; fi
    echo "GATE $p $f new=main $r"
  done
done | tee "$O/compare.txt"
echo "PLAN-DONE $(date -Is)"
