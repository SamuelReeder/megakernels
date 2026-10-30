#!/bin/bash
# One stage of the pipeline (driver.py), on the Alola login node: srun steps into job JOB.
# Usage: stage.sh NAME JOB GPUS TLEFT   (GPUS: GPU indices "0", "1" or "0,1", "-" for none;
#        TLEFT: minutes until the job ends). Output: the driver appends it to
#        ~/mk/out/pipeline/NAME.log; one PASS or FAIL line per check, then RESULT.
# Exit: 0 PASS (all checks pass), 1 FAIL (stage ran, a check failed), 2 ERROR (no result).
# Inputs (prep.sh): $PIPE_DIR/pins.env, bundles/, cc/, co/, e2e_plan.sh, wheel_node.sh.
set -uo pipefail
name=$1 job=$2 gpus=$3 tleft=$4
P=${PIPE_DIR:-$HOME/mk/pipeline}
O=${PIPE_OUT:-$HOME/mk/out/pipeline}
ST=$O/state
E=$O/e2e
# shellcheck source=/dev/null
. "$P/pins.env"
RT2=${RT2_FINAL:-${E2E_SET:-$HOME/coexport/sets/rt2/final}}  # the e2e code-object set (lock)
MNT=/cluster:/cluster,$HOME:$HOME,/tmp:/tmp
trap 'rc=$?; case $rc in 0) r=PASS;; 1) r=FAIL;; *) r=ERROR;; esac
      echo "RESULT $name $r rc=$rc $(date -Is)"; echo $rc > "$ST/$name.rc"' EXIT
echo "# $(date -Is) $name job $job gpus $gpus tleft $tleft pins $(tr '\n' ' ' < "$P/pins.env" | cut -c1-400)"

if [ "${PIPE_DRY:-0}" = 1 ]; then  # stub: sleep 80 % of the estimate; outcome per attempt from
  # state/dry_<name> (words, e.g. "ERROR PASS"), default PASS
  sleep "$(python3 -c "print(${STAGE_EST:-1} * 0.8 * ${PIPE_SCALE:-60})")"
  r=$(awk -v a="${STAGE_ATTEMPT:-1}" '{print $a}' "$ST/dry_$name" 2>/dev/null)
  r=${r:-PASS}
  echo "$r dry stub $name on job $job gpus $gpus tleft $tleft (attempt ${STAGE_ATTEMPT:-1})"
  case $r in PASS) exit 0 ;; FAIL) exit 1 ;; *) exit 2 ;; esac
fi

tcap() { if [ "$1" -gt "$tleft" ]; then echo "$tleft"; else echo "$1"; fi; }
step() { local t=$1; shift; srun --jobid="$job" --overlap -n1 --time="$t" "$@"; }
verdict() { if [ "$1" = 0 ]; then echo "PASS ${*:2}"; else echo "FAIL ${*:2}"; fail=1; fi; }
fail=0

# vstep BUNDLE MODELDIR MINUTES CMD: common/scripts/vrun on the node. The bundle (model
# directory, common/, vsrc/vllm, REVISION) goes to node /tmp; CMD runs in the model directory
# in the vLLM image with vsrc/ first on PYTHONPATH (plus the image's compiled vllm .so files).
vstep() {
  local b=$1 d=$2 t=$3 cmd=$4 src=/tmp/pipe_${name}_$$ rc
  step 10 bash -c "rm -rf $src && mkdir -p $src && tar -xzf $b -C $src" || return 2
  step "$(tcap "$t")" --container-image="$VIMG" --container-mounts="$MNT" --container-writable \
    bash -lc "so=\$(find /usr /opt -maxdepth 7 -name _C.abi3.so -path '*packages/vllm/*' | head -1)
      cp -n \$(dirname \$so)/*.so \$(dirname \$so)/_version.py $src/vsrc/vllm/
      export PYTHONPATH=$src/vsrc PIP_CACHE_DIR=/tmp/pipcache FLYDSL_RUNTIME_ENABLE_CACHE=0 \
        HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPYCACHEPREFIX=/tmp/pycache_$$ \
        M=\$HOME/mk/models HIP_VISIBLE_DEVICES=$gpus
      cd $src/$d && echo node \$(hostname) job $job gpus $gpus \$(cat $src/REVISION) vllm .so from \$so
      $cmd"
  rc=$?
  step 5 rm -rf "$src"
  return $rc
}

case $name in
a_cc)  # VllmRt2 common/results/compiled/gpu.sh stage + run (plan.sh) on both GPUs
  W=/tmp/cc_$job X=$HOME/mk/out/cc_$job
  step 15 bash -c "rm -rf $W && mkdir -p $W && tar -xzf $P/bundles/cc.tgz -C $W && hostname && ls $W && du -sh $W" || exit 2
  mkdir -p "$X" && cp "$P/cc/plan.sh" "$P/cc/vlane.sh" "$X/"
  JOB=$job W=$W O=$X bash "$X/plan.sh" > "$X/plan.log" 2>&1
  step 5 rm -rf "$W"
  grep -q PLAN-DONE "$X/plan.log" || { echo "no PLAN-DONE in $X/plan.log"; tail -5 "$X/plan.log"; exit 2; }
  while read -r l; do  # "GATE q8 hashes.txt new=main PASS" -> "PASS GATE q8 hashes.txt new=main"
    echo "${l##* } ${l% *}"; case $l in *FAIL) fail=1 ;; esac
  done < "$X/compare.txt"
  for f in "$X"/vllm/*.log; do
    grep -q '^exit 0$' "$f"; verdict $? "vllm $(basename "$f" .log) exit 0"
  done
  for p in q8new_4096:q8prev_4096 q8new_8192:q8prev_8192 q8new_4096:q8tp1prev_4096 \
           q8new_8192:q8tp1prev_8192 q6new_4096:q6prev_4096 q6new_8192:q6prev_8192 \
           gnew_4096:gprev_4096 q8new_tp2_4096:q8prev_tp2_4096; do
    a=$X/vllm/${p%:*}.json b=$X/vllm/${p#*:}.json
    r=$(python3 "$P/cc/vcompare.py" "$a" "$b" 2>&1 | grep '^RESULT' | cut -c1-300)
    case $r in *"tokens equal True"*) verdict 0 "greedy $p: $r" ;; *) verdict 1 "greedy $p: ${r:-no result}" ;; esac
  done
  for m in q8new q6new; do  # rt2: one code object per width serves 4096 and 8192
    a=$X/vllm/${m}_4096_isa.txt b=$X/vllm/${m}_8192_isa.txt
    [ -s "$a" ] && cmp -s "$a" "$b"; verdict $? "rt2 one binary $m: ISA 4096 == 8192 ($(wc -l < "$a" 2>/dev/null) kernels)"
  done
  exit $fail
  ;;
b_mkeng)  # MkEngine ~/work/mkeng/gpu.sh in the hipdnn image: engine ON build, unit, GPU integration
  step "$(tcap 150)" --container-image="$HIMG" --container-mounts="/cluster:/cluster,$HOME:$HOME" \
    bash -c "export HIP_VISIBLE_DEVICES=$gpus; bash $HOME/work/mkeng/gpu.sh" > "$O/b_mkeng.out" 2>&1
  rc=$?
  cat "$O/b_mkeng.out"
  [ $rc = 0 ] || [ -s "$HOME/work/mkeng/logs-gpu/integration-megakernel-ON.log" ] || exit 2
  for k in "MODE=ON build_rc=0" "MODE=ON unit_rc=0" "MODE=ON megakernel_unit_rc=0" "gpu arch=gfx942" \
           "integration_rc=0" "pytest_rc=0"; do
    grep -q "^$k" "$O/b_mkeng.out"; verdict $? "mkeng $k"
  done
  exit $fail
  ;;
c_rt0_merge|c_rt2_merge)  # merge run a of each model into the set's final/ (login node, stdlib)
  set_=${name#c_}; set_=${set_%_merge}
  S=$HOME/coexport/sets/$set_
  dirs=()
  for m in qwen38-27b-mxfp4-gfx942 qwen36-35b-a3b-fp8-gfx942 gemma4-26b-a4b-fp8-gfx942; do
    [ -f "$S/$m/a/manifest.json" ] || { echo "FAIL $S/$m/a/manifest.json missing"; exit 2; }
    dirs+=("$S/$m/a")
  done
  rm -rf "$S/final"
  python3 -B "$P/co/export_co.py" --out "$S/final" --merge "${dirs[@]}"; verdict $? "merge $set_ -> $S/final"
  exit $fail
  ;;
c_rt*)  # CoExporter run_set.sh, one model: node_model.sh (export a, b; determinism; H1)
  set_=${name#c_}; set_=${set_%_*}
  read -r dir model w vc mc deps < "$P/co/${name#c_}.args"
  S=$HOME/coexport/sets/$set_
  mkdir -p "$S"
  echo "# set $set_ $model vllm $vc megakernels $mc deploy $deps"
  vstep "$P/bundles/co_${name#c_}.tgz" "$dir" 170 \
    "bash ../common/results/co/node_model.sh $S $model \$M/$w $vc $mc $deps" > "$O/$name.out" 2>&1
  rc=$?
  grep -E '^export|EQUAL|DIFFERENT|^H1|^# H1|rror' "$O/$name.out" | cut -c1-400
  [ -f "$S/$model/a/manifest.json" ] || { echo "no $S/$model/a/manifest.json (rc $rc)"; tail -20 "$O/$name.out"; exit 2; }
  while read -r l; do
    case $l in *": EQUAL") echo "PASS determinism $model: $l" ;; *": DIFFERENT") echo "FAIL determinism $model: $l"; fail=1 ;; esac
  done < "$S/$model/logs/determinism.log"
  grep -hE '^H1 (PASS|FAIL)|^H1-SUMMARY' "$S/$model/logs/h1.log" | sed -E 's/^H1 (PASS|FAIL) /\1 H1 /'
  grep -qE '^H1 FAIL' "$S/$model/logs/h1.log" && fail=1
  [ $rc = 0 ] || fail=1
  exit $fail
  ;;
d_wheel)  # e2e_plan.sh stage + wheel (+ drop's wheel), in the hipdnn image on any node
  [ -f "$RT2/manifest.json" ] || { echo "no $RT2/manifest.json"; exit 2; }
  mkdir -p "$E"
  # the node reads a private copy: prep.sh replacing $P/wheel_node.sh on the NFS home while
  # the node's bash reads it ends the build with "Stale file handle"
  wn=$E/wheel_node.$$.sh
  cp "$P/wheel_node.sh" "$wn"
  step "$(tcap 220)" --container-image="$HIMG" --container-mounts="/cluster:/cluster,$HOME:$HOME" \
    bash -c "RL_SHA=$RL_SHA KP_SHA=$KP_SHA bash $wn $RT2 $E" > "$E/wheel_node.log" 2>&1
  rc=$?
  rm -f "$wn"
  grep -E '^(PASS|FAIL) ' "$E/wheel_node.log"
  ls "$E"/wheel/*.whl "$E"/wheel-drop4/*.whl 2>/dev/null
  [ -n "$(ls "$E"/wheel/*.whl 2>/dev/null)" ] || { tail -30 "$E/wheel_node.log"; exit 2; }
  [ $rc = 0 ] || exit 1
  exit 0
  ;;
e_*|f_drop_*)  # e2e_plan.sh runs (e_K: four modes) or drop (f_drop_K: hipdnn on the drop4 wheel)
  k=${name#e_}; k=${k#f_drop_}
  read -r dir _ < <(bash "$P/e2e_plan.sh" info "$k")
  if [ "${name%%_*}" = e ]; then tag=runs whl=wheel modes=$(bash "$P/e2e_plan.sh" modes "$k")
  else tag=drop4 whl=wheel-drop4 modes=hipdnn; fi
  [ -n "$(ls "$E/$whl"/*.whl 2>/dev/null)" ] || { echo "no wheel in $E/$whl"; exit 2; }
  res=$E/$tag/$k
  rm -rf "$res" && mkdir -p "$res"
  cmd=$(MODES=$modes bash "$P/e2e_plan.sh" cmd "$k" "$res" "$E/$whl/hipdnn_vllm-*.whl")
  vstep "$P/bundles/e2e_$k.tgz" "$dir" 240 "$cmd" > "$E/${tag}_$k.log" 2>&1
  rc=$?
  grep -E "launches through hipDNN|keeps vLLM's path|DecodeUnsupported|TPOT|Error" "$E/${tag}_$k.log" | cut -c1-300 | head -40
  n=0
  for m in $modes; do [ -f "$res/$m.json" ] && n=$((n + 1)); verdict $([ -f "$res/$m.json" ]; echo $?) "$tag $k $m.json"; done
  [ $n = 0 ] && exit 2
  [ $rc = 0 ] || fail=1
  exit $fail
  ;;
g_check)  # e2e_plan.sh check: H1 H2 H3 (q8tp2 ran with CHECK_PATHS) H4 H6, TPOT, payloads
  OUT=$E FINAL=$RT2 bash "$P/e2e_plan.sh" check
  grep -q '^FAIL' "$E/verdicts.txt" && exit 1
  grep -q '^PASS' "$E/verdicts.txt" || exit 2
  exit 0
  ;;
*) echo "unknown stage $name"; exit 2 ;;
esac
