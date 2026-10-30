#!/bin/bash
# Step 8 (docs/hipdnn-route.md) GPU validation plan of the vLLM hipDNN launch backend
# (VLLM_SPECIALIZED_DECODE_LAUNCH=hipdnn). NOT RUN YET: it needs a gfx942 allocation
# (pending jobs 68215720, 68215745, 68215802). Run each stage on its own, in order:
#
#   e2e_plan.sh export   JOBID   1. code-object set from the vLLM heads (H1 inputs)
#   e2e_plan.sh stage            2. descriptors: lock + artifacts in the engine tree (DVC-local)
#   e2e_plan.sh wheel    JOBID   3. hipdnn_vllm gfx94X wheel with the megakernel engine
#   e2e_plan.sh runs     JOBID   4. vLLM kernel on: flydsl vs hipdnn, default, no-wheel
#   e2e_plan.sh drop     JOBID   5. H2: wheel without the M=4 objects; that width on vLLM
#   e2e_plan.sh check            6. verdicts H1 H2 H3 H4 H6, TPOT table, payload = manifest
#
# Pass criteria (docs/hipdnn-route.md section 3):
#   H1  per model: greedy tokens and logprobs of hipdnn equal flydsl, bit for bit.
#   H2  drop: the startup log has "width 4 keeps vLLM's path: hipDNN declined"; widths
#       1, 2, 8 log "launches through hipDNN"; no DecodeUnsupported; the run completes.
#   H3  Qwen3.8 TP2 runs with VLLM_SPECIALIZED_DECODE_CHECK_PATHS=1 and do not raise.
#   H4  FULL graphs (vllm_e2e.py default mode); the hipdnn_vllm reports have zero
#       CAPTURE_INVARIANT declines and zero custom_op fallback rows.
#   H6  wheel installed, VLLM_SPECIALIZED_DECODE_LAUNCH and HIPDNN_VLLM_OPS unset: tokens
#       equal the no-wheel run, the report has no custom_op row, no megakernel decision.
#   TPOT hipdnn vs flydsl at B = 1, 2, 4, 8 (vllm_e2e.py tpot_ms), reported, no bar.
#   Payloads: every "launches through hipDNN ... payload" line selects exactly one
#       manifest entry (all key fields equal).
#
# Env: MK_CO (megakernels worktree of users/sareeder/co-export, run_set.sh), FINAL (the
#   set's merged directory, default MK_CO's common/results/co/rt2/final), RL (rocm-
#   libraries checkout that merges users/sareeder/hipdnn-vllm-custom-op 0398903c801 and
#   the megakernel engine branch 85a5c480493), VLLM_WT (vLLM worktrees), OUT (results,
#   default this directory/gpu), M (models dir on the node), WHEEL_ENV (build_wheel.sh
#   VENV/WORK/KPACK_PYTHON_DIR settings: a venv with the vLLM image's torch, so that _C
#   loads in the image).
# common/results/pipeline/ runs these stages unattended on the Alola login node: it calls
#   `e2e_plan.sh info K` and `e2e_plan.sh cmd K RES WHEEL` for the node commands, and
#   `e2e_plan.sh check` with OUT and FINAL on the Alola home.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
MK_CO=${MK_CO:-/home/sareeder/ROCm-workspace/worktrees/megakernels/co-export}
RL=${RL:-/home/sareeder/ROCm-workspace/worktrees/rocm-libraries/megakernel-engine}
VLLM_WT=${VLLM_WT:-/home/sareeder/ROCm-workspace/worktrees/vllm}
OUT=${OUT:-$here/gpu}
SET=rt2
FINAL=${FINAL:-$MK_CO/common/results/co/$SET/final}
DESC=$RL/dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/megakernel
# vLLM heads with the hipDNN launch backend, and the rt2 set's commits they sit on:
# the commit does not touch kernels/, so src (the kernels/ hash) is the set's.
declare -A HEAD=([q8]=e745c7910 [q8tp2]=b9a1943fb [q6]=81b2b128d [g]=3cd36e151)
declare -A BASE=([q8]=0332fdc96 [q8tp2]=9fc7bc243 [q6]=6657a8352 [g]=5a25d336a)
declare -A WT=([q8]=qwen38-gfx942-specialized-decode [q8tp2]=qwen38-gfx942-tp2
               [q6]=qwen36-gfx942-specialized-decode [g]=gemma4-moe-batch)
declare -A PKG=([q8]=qwen3_5 [q8tp2]=qwen3_5 [q6]=qwen3_5_moe [g]=gemma4)
declare -A DIR=([q8]=qwen3.8-27b-gfx942 [q8tp2]=qwen3.8-27b-gfx942
                [q6]=qwen3.6-35b-a3b-gfx942 [g]=gemma4-26b-a4b-gfx942)
declare -A W=([q8]=Qwen3.8-27B-Quark-AWQ-MXFP4 [q8tp2]=Qwen3.8-27B-Quark-AWQ-MXFP4
              [q6]=Qwen3.6-35B-A3B-FP8 [g]=gemma-4-26B-A4B-it-FP8-dynamic)
declare -A ENVS=([q8]="MK_ATTN=ROCM_ATTN" [q8tp2]="MK_ATTN=ROCM_ATTN MK_TP=2 VLLM_SPECIALIZED_DECODE_CHECK_PATHS=1"
                 [q6]="MK_ATTN=TRITON_ATTN" [g]="MK_ATTN=TRITON_ATTN")
OPS_ON=attention,encoder_attention,rms_norm,fused_add_rms_norm,layer_norm,custom_op
# bench/vllm_e2e.py CLI: Qwen3.8 takes WEIGHTS MODE OUT; Qwen3.6 and Gemma take run WEIGHTS MODE OUT.
declare -A E2E=([q8]="bench/vllm_e2e.py" [q8tp2]="bench/vllm_e2e.py"
                [q6]="bench/vllm_e2e.py run" [g]="bench/vllm_e2e.py run")

# node_cmd K RES WHEEL: the command of stage 4 for model K, run in the model directory in
# the vLLM image; results in RES, the wheel file (glob) WHEEL. MODES selects a subset; each
# mode runs even when an earlier one fails. flydsl2 repeats flydsl: the run-to-run spread
# of vLLM at TP2 (qwen3.8 docs/DESIGN-TP2.md: TP2 is not repeatable inside vLLM).
node_cmd() {
  local k=$1 res=$2 whl=$3 mode
  local rep="HIPDNN_VLLM_REPORT_DIR=$res/rep"
  local e2e="${ENVS[$k]} python ${E2E[$k]} \$M/${W[$k]} on"
  local cmd="pip install -q --no-deps $whl && mkdir -p $res/rep"
  for mode in ${MODES:-$(modes "$k")}; do
    case $mode in
      flydsl|flydsl2|hipdnn) cmd+="; VLLM_SPECIALIZED_DECODE_LAUNCH=${mode%2} HIPDNN_VLLM_OPS=$OPS_ON $rep/$mode $e2e $res/$mode.json" ;;
      default) cmd+="; $rep/default $e2e $res/default.json" ;;
      nowheel) cmd+="; pip uninstall -q -y hipdnn_vllm && $e2e $res/nowheel.json" ;;
    esac
  done
  echo "$cmd"
}
modes() {  # the stage 4 modes of model K
  if [ "$1" = q8tp2 ]; then echo "flydsl flydsl2 hipdnn default nowheel"; else echo "flydsl hipdnn default nowheel"; fi
}

stage=${1:?usage: e2e_plan.sh export|stage|wheel|runs|drop|check [JOBID] | info K | cmd K RES WHEEL}
job=${2:-}
case $stage in info|cmd|modes) ;; *) mkdir -p "$OUT" ;; esac

case $stage in
export)  # 1. the set; then prove that the heads' kernels/ equal the set's
  for k in q8 q8tp2 q6 g; do
    v=$VLLM_WT/${WT[$k]}
    git -C "$v" diff --quiet "${BASE[$k]}" "${HEAD[$k]}" -- "vllm/models/${PKG[$k]}/amd/specialized_decode/kernels" \
      || { echo "$k: kernels/ changed since ${BASE[$k]}: re-export from ${HEAD[$k]}" >&2; exit 1; }
  done
  "$MK_CO/common/results/co/run_set.sh" "$job" "$SET"  # -> $MK_CO/common/results/co/rt2/final
  ;;
stage)  # 2. DVC-local: the set's bytes in artifacts/, the lock, the generated KDP
  final=$FINAL
  mkdir -p "$DESC/artifacts"
  cp "$final/manifest.json" "$DESC/manifest.json"
  cp "$final"/*.co "$DESC/artifacts/"
  tool=$RL/dnn-providers/hip-kernel-provider/descriptor-packaging/tools/hkp_megakernel_lock.py
  python3 "$tool" generate --lock "$DESC/manifest.json"
  python3 "$tool" verify --lock "$DESC/manifest.json" | tee "$OUT/lock_verify.txt"
  # configure compares the folder's file count with artifacts.dvc: record the new folder
  (cd "$RL" && dvc add "$DESC/artifacts")
  ;;
wheel)  # 3. on a compute node (build_wheel.sh compiles); BUILD_PINS gfx94X has the engine ON.
  # RL_ALOLA: the same rocm-libraries commit as RL, checked out on the Alola home, with
  # stage 2's descriptors/megakernel/ copied in (rsync before this stage).
  wd=${WHEEL:-step8/wheel} rla=${RL_ALOLA:?set RL_ALOLA}
  ssh -o BatchMode=yes alola "srun --jobid=$job --overlap -n1 bash -lc '${WHEEL_ENV:?set WHEEL_ENV} HIPDNN_SRC=$rla \
    OUT=\$HOME/$wd $rla/projects/hipdnn/tools/hipdnn_vllm/scripts/build_wheel.sh gfx94X'" \
    2>&1 | tee "$OUT/wheel_build_$(basename "$wd").log"
  ;;
runs)  # 4. per model: flydsl, hipdnn, default (wheel, nothing opted in), no wheel.
  # MODES and WHEEL select a subset (stage 5); results on the Alola home, then fetched.
  wd=${WHEEL:-step8/wheel} tag=${TAG:-runs}
  for k in ${MODELS:-q8 q8tp2 q6 g}; do
    cmd=$(node_cmd "$k" "\$HOME/step8/$tag/$k" "\$HOME/$wd/hipdnn_vllm-*.whl")
    MK_TIME=240 VLLM_REV=${HEAD[$k]} "$root/common/scripts/vrun" "$root/${DIR[$k]}" "$VLLM_WT/${WT[$k]}" \
      -j "$job" "$cmd" > "$OUT/${tag}_$k.log" 2>&1 || echo "$k: run failed (see ${tag}_$k.log)"
    mkdir -p "$OUT/$tag/$k"
    ssh -o BatchMode=yes alola "tar -C step8/$tag/$k -cf - ." | tar -C "$OUT/$tag/$k" -xf -
  done
  ;;
drop)  # 5. H2: a wheel whose lock has no M=4 entries; hipdnn runs of Qwen3.8 and Qwen3.6
  python3 - "$DESC/manifest.json" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]); a = p.parent / "artifacts"
entries = json.loads(p.read_text())
for e in entries:
    if e["M"] == 4:
        (a / e["file"]).unlink()
p.write_text(json.dumps([e for e in entries if e["M"] != 4], indent=1, sort_keys=True) + "\n")
PY
  tool=$RL/dnn-providers/hip-kernel-provider/descriptor-packaging/tools/hkp_megakernel_lock.py
  python3 "$tool" generate --lock "$DESC/manifest.json"
  python3 "$tool" verify --lock "$DESC/manifest.json" | tee "$OUT/lock_verify_drop4.txt"
  WHEEL=step8/wheel-drop4 "$0" wheel "$job"
  WHEEL=step8/wheel-drop4 TAG=drop4 MODELS="q8 q6" MODES=hipdnn "$0" runs "$job"
  "$0" stage  # restore the full lock
  ;;
check)  # 6. verdicts from the fetched results
  { for k in q8 q8tp2 q6 g; do
    echo "== $k (vLLM ${HEAD[$k]})"
    python3 - "$OUT/runs/$k" "$FINAL/manifest.json" "$OUT/runs_$k.log" <<'PY' || echo "FAIL $k: check error (missing results?)"
import glob, json, re, sys
d, man, log = sys.argv[1:]
r = {m: json.load(open(f"{d}/{m}.json")) for m in ("flydsl", "hipdnn", "default", "nowheel")}
try:
    r["flydsl2"] = json.load(open(f"{d}/flydsl2.json"))
except OSError:
    pass
v = lambda ok: "PASS" if ok else "FAIL"
# Qwen3.8 saves the generated tokens' logprobs, Qwen3.6 and Gemma the top-5 per position
lp = lambda o: o["logprobs"] if "logprobs" in o else o["top"]
same = lambda a, b: all(x["tokens"] == y["tokens"] and lp(x) == lp(y)
                        for x, y in zip(r[a]["outputs"], r[b]["outputs"]))
print(v(same("hipdnn", "flydsl")), "H1 tokens+logprobs hipdnn == flydsl")
if "flydsl2" in r:  # TP2: the run-to-run spread of vLLM itself, for reading H1
    eq = lambda a, b: sum(x["tokens"] == y["tokens"] and lp(x) == lp(y)
                          for x, y in zip(r[a]["outputs"], r[b]["outputs"]))
    print(f"H1 spread: prompts bitwise equal hipdnn/flydsl {eq('hipdnn', 'flydsl')}, hipdnn/flydsl2 "
          f"{eq('hipdnn', 'flydsl2')}, flydsl/flydsl2 {eq('flydsl', 'flydsl2')} of {len(r['flydsl']['outputs'])}")
print(v(all(x["tokens"] == y["tokens"] for x, y in zip(r["default"]["outputs"], r["nowheel"]["outputs"]))),
      "H6 tokens default == no wheel")
for B in ("1", "2", "4", "8"):
    f, h = r["flydsl"]["tpot_ms"][B], r["hipdnn"]["tpot_ms"][B]
    print(f"TPOT B={B}: flydsl {f:.3f} ms, hipdnn {h:.3f} ms ({(h / f - 1) * 100:+.2f} %)")
def rows(sub):
    out = []
    for f in glob.glob(f"{d}/rep/{sub}/report-*.json"):
        out += json.load(open(f)).get("counters", [])
    return out
cap = [x for x in rows("hipdnn") if "CAPTURE_INVARIANT" in json.dumps(x)]
fb = [x for x in rows("hipdnn") if x.get("op") == "custom_op" and x.get("route") == "fallback"]
print(v(not cap and not fb), "H4 CAPTURE_INVARIANT rows:", len(cap), "custom_op fallback rows:", len(fb))
n6 = len([x for x in rows("default") if x.get("op") == "custom_op"])
print(v(n6 == 0), "H6 default custom_op rows:", n6)
if d.rstrip("/").endswith("q8tp2"):  # the four runs loaded above completed
    print("PASS H3 TP2 runs with VLLM_SPECIALIZED_DECODE_CHECK_PATHS=1 completed (no path disagreement raised)")
keys = ("model", "src", "arch", "cus", "kv", "block", "M", "tp", "rt", "lm_head")
entries = json.load(open(man))
n = 0
for m in re.finditer(r"launches through hipDNN \S+; payload (\{.*\})", open(log).read()):
    p = json.loads(m.group(1))
    hits = [e for e in entries if all(e[k] == p[k] for k in keys)]
    n += 1
    print(v(len(hits) == 1), "payload M", p["M"], "tp", p["tp"], "manifest entries:", len(hits))
print(v(n > 0), "payload lines:", n)
PY
  done
  for k in q8 q6; do  # H2
    nd=$(grep -c "width 4 keeps vLLM's path: hipDNN declined" "$OUT/drop4_$k.log" 2>/dev/null || true)
    nl=$(grep -c 'launches through hipDNN' "$OUT/drop4_$k.log" 2>/dev/null || true)
    nu=$(grep -c DecodeUnsupported "$OUT/drop4_$k.log" 2>/dev/null || true)
    [ "${nd:-0}" -gt 0 ] && [ "${nl:-0}" -gt 0 ] && [ "${nu:-0}" = 0 ] && [ -f "$OUT/drop4/$k/hipdnn.json" ] && h2=PASS || h2=FAIL
    echo "$h2 H2 $k drop4: ${nd:-0} decline line(s) for width 4; ${nl:-0} launch line(s) (3 per cache bind);" \
      "DecodeUnsupported: ${nu:-0}; hipdnn.json: $(ls "$OUT/drop4/$k/hipdnn.json" 2>/dev/null || echo missing)"
  done; } | tee "$OUT/verdicts.txt"
  ;;
modes)  # model K's stage 4 modes
  modes "$job"
  ;;
info)  # model K: its directory and vLLM head
  echo "${DIR[$job]} ${HEAD[$job]}"
  ;;
cmd)  # model K's stage 4 command (RES results dir, WHEEL wheel file) for the pipeline
  node_cmd "$job" "${3:?RES}" "${4:?WHEEL}"
  ;;
*) echo "unknown stage $stage" >&2; exit 2 ;;
esac
