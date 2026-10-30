#!/bin/bash
# Prepare the unattended GPU pipeline (driver.py) on Alola, from this workstation. The login
# node and the compute nodes have no checkouts, so this ships node-ready bundles of the pinned
# commits (git archive, as gpu.sh stage and common/scripts/vrun do), the scripts and pins.env
# to ~/mk/pipeline, and makes the hvco job hold its allocation for the pipeline after its own
# tests (~/work/hvco_run.sh, edited in place: srun reads it from stdin when the job starts).
# Idempotent. Then start the driver on Alola: `bash ~/mk/pipeline/run.sh start`.
# Usage: common/results/pipeline/prep.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
WS=/home/sareeder/ROCm-workspace
VR=$WS/repos/vllm
A="ssh -o BatchMode=yes alola"
# pinned commits
MK_CC=$(git -C "$root" rev-parse 9a195e4)    # users/sareeder/cfg-compiled (VllmRt2, gpu.sh)
MK_MAIN=$(git -C "$root" rev-parse 10bd202)  # megakernels main for the gates (gpu.sh MAIN)
MK_CO=$(git -C "$root" rev-parse f9e41ae)    # users/sareeder/co-export (CoExporter)
MK_E2E=$(git -C "$root" rev-parse HEAD)      # users/sareeder/hipdnn-design (e2e_plan.sh, this)
RL_SHA=$(git -C "$WS/worktrees/rocm-libraries/hipdnn-vllm-custom-op" rev-parse 66d0838cd8f)
KP_SHA=a022846cf553c2b135410a5168f97705f1b9c6ac  # rocm-systems shared/kpack (notes/custom_op build)
VIMG=/home/AMD/sareeder/mk/images/vllm-rocm-nightly-00b6c4c5.sqsh
HIMG=/cluster/images/hipdnn/hipdnn_20261024_gfx942.sqsh  # pinned: hipdnn_latest moves on Saturdays
T=$(mktemp -d /tmp/mkpipe_prep.XXXX)
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bundles" "$T/cc" "$T/co"

# mk_tree COMMIT DIR DEST: model DIR + common/ of megakernels COMMIT as vrun ships them
# (no bench/results, no common/results except co/node_model.sh)
mk_tree() {
  mkdir -p "$3"
  git -C "$root" archive "$1" "$2" common | tar -xf - -C "$3"
  rm -rf "$3/$2/bench/results"
  find "$3/common/results" -mindepth 1 -maxdepth 1 ! -name co -exec rm -rf {} +
  [ -d "$3/common/results/co" ] && find "$3/common/results/co" -mindepth 1 ! -name node_model.sh -exec rm -rf {} +
  true
}
vsrc() {  # vsrc COMMIT DEST: the vLLM checkout's vllm/ as vsrc/vllm
  mkdir -p "$2/vsrc" && git -C "$VR" archive "$1" vllm | tar -xf - -C "$2/vsrc"
}

# a_cc: gpu.sh stage (mk_new, mk_main, v_<name> of TREES, each with REVISION)
eval "$(git -C "$root" show "$MK_CC:common/results/compiled/gpu.sh" | sed -n '/^TREES="/,/"$/p')"
d=$T/cc_tree
for x in "mk_new=$MK_CC" "mk_main=$MK_MAIN"; do
  mkdir -p "$d/${x%%=*}" && git -C "$root" archive "${x#*=}" | tar -xf - -C "$d/${x%%=*}"
  echo "${x#*=}" > "$d/${x%%=*}/REVISION"
done
for t in $TREES; do
  n=${t%%=*} c=${t#*=}
  mkdir -p "$d/v_$n" && git -C "$VR" archive "$c" vllm tests pyproject.toml | tar -xf - -C "$d/v_$n"
  git -C "$VR" rev-parse "$c" > "$d/v_$n/REVISION"
done
tar -C "$d" -czf "$T/bundles/cc.tgz" .
for f in plan.sh vlane.sh; do git -C "$root" show "$MK_CC:common/results/compiled/$f" > "$T/cc/$f"; done
git -C "$root" show "$MK_CC:common/results/rt2/vcompare.py" > "$T/cc/vcompare.py"

# c_<set>_<k>: run_set.sh's per-model node_model.sh call (deployments as in run_set.sh)
git -C "$root" show "$MK_CO:common/scripts/export_co.py" > "$T/co/export_co.py"
co() {  # co SET K DIR MODEL WEIGHTS VLLM_COMMIT DEPLOY...
  local set=$1 k=$2 dir=$3 model=$4 w=$5 vc
  vc=$(git -C "$VR" rev-parse "$6"); shift 6
  local b=$T/co_${set}_$k
  mk_tree "$MK_CO" "$dir" "$b"
  vsrc "$vc" "$b"
  echo "megakernels $MK_CO dirty 0; vllm $vc dirty 0" > "$b/REVISION"
  tar -C "$b" -czf "$T/bundles/co_${set}_$k.tgz" . && rm -rf "$b"
  echo "$dir $model $w $vc $MK_CO $*" > "$T/co/${set}_$k.args"
}
for set in rt0 rt2; do
  case $set in
    rt0) q8=fd8a77a5f q6=70f57a0ae g=7f586cd2a q8l=4096 q6l=4096 ;;
    rt2) q8=9fc7bc243 q6=6657a8352 g=5a25d336a q8l=4096:8192 q6l=4096:8192 ;;
  esac
  co $set q8 qwen3.8-27b-gfx942 qwen38-27b-mxfp4-gfx942 Qwen3.8-27B-Quark-AWQ-MXFP4 $q8 \
    "kv=ROCM_ATTN,block=784,maxlen=$q8l,tp=1,M=1:2:4:8,cus=304" "kv=ROCM_ATTN,block=784,maxlen=4096,tp=2,M=1:2:4:8,cus=304"
  co $set q6 qwen3.6-35b-a3b-gfx942 qwen36-35b-a3b-fp8-gfx942 Qwen3.6-35B-A3B-FP8 $q6 \
    "kv=TRITON_ATTN:1,block=1056,maxlen=$q6l,tp=1,M=1:2:4:8,cus=304"
  co $set g gemma4-26b-a4b-gfx942 gemma4-26b-a4b-fp8-gfx942 gemma-4-26B-A4B-it-FP8-dynamic $g \
    "kv=TRITON_ATTN:1,block=16,maxlen=4096,tp=1,M=1:2:4:8,cus=304"
done

# e_<k>, f_drop_<k>: e2e_plan.sh runs/drop trees; first its export precondition (the heads'
# kernels/ equal the rt2 set's commits, so the set's src keys are the heads')
e2e=$root/common/results/step8/e2e_plan.sh
for p in "e745c7910 0332fdc96 qwen3_5" "b9a1943fb 9fc7bc243 qwen3_5" "81b2b128d 6657a8352 qwen3_5_moe" "3cd36e151 5a25d336a gemma4"; do
  set -- $p
  git -C "$VR" diff --quiet "$2" "$1" -- "vllm/models/$3/amd/specialized_decode/kernels" \
    || { echo "$1: kernels/ differ from $2: the rt2 set does not serve it" >&2; exit 1; }
done
for k in q8 q8tp2 q6 g; do
  read -r dir head < <(bash "$e2e" info $k)
  vc=$(git -C "$VR" rev-parse "$head")
  b=$T/e2e_$k
  mk_tree "$MK_E2E" "$dir" "$b"
  vsrc "$vc" "$b"
  echo "megakernels $MK_E2E dirty 0; vllm $vc dirty 0" > "$b/REVISION"
  tar -C "$b" -czf "$T/bundles/e2e_$k.tgz" . && rm -rf "$b"
done

cp "$here"/{driver.py,stage.sh,wheel_node.sh,run.sh,dryrun.py} "$e2e" "$T/"
cat > "$T/pins.env" <<EOF
MK_CC=$MK_CC
MK_MAIN=$MK_MAIN
MK_CO=$MK_CO
MK_E2E=$MK_E2E
PIPE_REV="hipdnn-design $(git -C "$root" rev-parse --short HEAD) dirty $(git -C "$root" status --porcelain common/results/pipeline common/results/step8 | wc -l)"
RL_SHA=$RL_SHA
KP_SHA=$KP_SHA
VIMG=$VIMG
HIMG=$HIMG
EOF
rm -rf "$T/cc_tree"
du -sh "$T"/bundles/* | sort -h | tail -4
$A "test -f $VIMG && test -f $HIMG && test -f ~/work/mkeng/gpu.sh && test -f ~/work/mkeng/mk-src.tgz" \
  || { echo "an image or ~/work/mkeng input is missing on Alola" >&2; exit 1; }
tar -C "$T" -czf - . | $A "mkdir -p ~/mk/pipeline && rm -rf ~/mk/pipeline/bundles && tar -xzf - -C ~/mk/pipeline && ls ~/mk/pipeline ~/mk/pipeline/bundles"

# the hvco job: after its tests, hold the allocation while the driver lives (driver.alive
# touched within 15 min) and has not released it (state/release_<job>); mark its tests done
if $A 'grep -q pipeline_hold ~/work/hvco_run.sh'; then
  echo "hvco_run.sh: hold present"
else
  $A 'cat ~/work/hvco_run.sh' > "$T/hvco_run.sh"
  python3 - "$T/hvco_run.sh" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
hold = """pipeline_hold() {  # GPU pipeline (megakernels common/results/pipeline): keep the allocation
  local P=$HOME/mk/out/pipeline/state; mkdir -p $P; touch $P/hvco_done_${SLURM_JOB_ID}
  echo "pipeline hold $(date -Is)"
  while [ ! -e $P/release_${SLURM_JOB_ID} ] && [ -n "$(find $P/driver.alive -mmin -15 2>/dev/null)" ]; do sleep 60; done
  echo "pipeline hold end $(date -Is)"
}
"""
old = """trap 'cd /; rm -rf $W; echo "cleaned $(date -Is)"' EXIT"""
assert s.count(old) == 1 and s.count("set -uo pipefail\n") == 1, "hvco_run.sh changed: patch by hand"
s = s.replace("set -uo pipefail\n", "set -uo pipefail\n" + hold)
s = s.replace(old, old[:-len("' EXIT")] + "; pipeline_hold' EXIT")
open(p, "w").write(s)
PY
  bash -n "$T/hvco_run.sh"
  # cat > keeps the inode: the pending srun of job hvco reads this file (its stdin) at job start
  $A 'cat > ~/work/hvco_run.sh' < "$T/hvco_run.sh"
  $A 'grep -n "pipeline_hold\|trap" ~/work/hvco_run.sh'
fi
echo "prepared ~/mk/pipeline on Alola ($(git -C "$root" rev-parse --short HEAD)); start: $A 'bash ~/mk/pipeline/run.sh start'"
