#!/bin/bash
# All-model gates of the common/ refactor (common/DESIGN.md "Test strategy", oracles 2-4).
# Runs on a held gfx942 allocation in the hipdnn image with the venv, from the repo root.
#   common/scripts/gates.sh OUT [part ...]     parts: q8 q6 g (one GPU), q8tp2 (two GPUs:
#                                              run the whole script under `srun -n2`)
# Env: MODELS (default ~/mk/models); RUNS=2 runs the logits hash twice (baselines);
#      BASE=<dir with <part>/hashes.txt and isa.txt> compares hashes and ISA with it.
# Writes OUT/<part>/*.txt, OUT/<part>/hashes.txt, isa.txt, and prints one line per gate:
#   GATE <part> <name> PASS|FAIL
# FLYDSL_RUNTIME_ENABLE_CACHE=0 for every run (common/DESIGN.md "Compile cache test").
set -u
export FLYDSL_RUNTIME_ENABLE_CACHE=0
O=$(realpath -m "$1"); shift
MODELS=${MODELS:-$HOME/mk/models}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUNS=${RUNS:-1}
rank=${SLURM_PROCID:-0}
parts=${*:-q8 q6 g}
mkdir -p "$O"

gate() {  # gate PART NAME LOG cmd...: run, record exit code
  local p=$1 n=$2 log=$3; shift 3
  "$@" > "$log" 2>&1
  local rc=$?
  [ "$rank" = 0 ] && echo "GATE $p $n $([ $rc = 0 ] && echo PASS || echo "FAIL rc=$rc")" | tee -a "$O/summary.txt"
}

check() {  # check PART FILE KIND: compare OUT/PART/FILE with BASE/PART/FILE
  local p=$1 f=$2
  [ -n "${BASE:-}" ] && [ "$rank" = 0 ] || return 0
  if diff "$BASE/$p/$f" "$O/$p/$f" > "$O/$p/$f.diff"; then r=PASS; else r=FAIL; fi
  echo "GATE $p $f=base $r" | tee -a "$O/summary.txt"
}

hashes() {  # hashes PART WEIGHTS: logits hash RUNS times, then the ISA dump
  local p=$1 w=$2 d=$O/$1 sfx=""
  [ "$p" = q8tp2 ] && sfx=_r$rank
  for r in $(seq 1 "$RUNS"); do
    for ar in ${ARS:-x}; do
      local env=()
      [ "$ar" = x ] || env=(MK_TP=2 MK_AR=$ar)
      gate "$p" "logits_hash_run${r}_$ar" "$d/logits_hash_run${r}_$ar$sfx.txt" \
        env "${env[@]}" python -B tests/logits_hash.py "$w"
    done
  done
  [ "$rank" = 0 ] || return 0
  sleep 5
  grep -h '^HASH' "$d"/logits_hash_run1_*.txt | sort > "$d/hashes.txt"
  for r in $(seq 2 "$RUNS"); do
    if diff <(grep -h '^HASH' "$d"/logits_hash_run${r}_*.txt | sort) "$d/hashes.txt" > /dev/null; then x=PASS; else x=FAIL; fi
    echo "GATE $p hash_run$r=run1 $x" | tee -a "$O/summary.txt"
  done
  check "$p" hashes.txt
}

isa() {  # isa PART WEIGHTS
  local p=$1 w=$2 d=$O/$1
  for ar in ${ARS:-x}; do
    if [ "$ar" = x ]; then common_isa "$w" "$d"; else MK_TP=2 MK_AR=$ar common_isa "$w" "$d"; fi
  done > "$d/isa_r$rank.txt"
  [ "$rank" = 0 ] || return 0
  sleep 5
  sort "$d"/isa_r*.txt > "$d/isa.txt"
  ! grep -q NO-DUMP "$d/isa.txt" && echo "GATE $p isa_dump PASS" | tee -a "$O/summary.txt" \
    || echo "GATE $p isa_dump FAIL" | tee -a "$O/summary.txt"
  check "$p" isa.txt
}
common_isa() { MK_TP=${MK_TP:-1} "$ROOT/common/scripts/isa_dump" "$1" "$2/isa"; }

{
  date -Is; hostname; echo "job ${SLURM_JOB_ID:-?} rank $rank parts $parts"
  rocminfo | grep -m1 -oE 'gfx[0-9a-f]+'
  python -c "import torch, flydsl; print('torch', torch.__version__, 'flydsl', flydsl.__version__, torch.cuda.device_count(), 'GPUs')"
  cat "$ROOT/REVISION" 2>/dev/null || git -C "$ROOT" rev-parse HEAD
} > "$O/env_${parts// /_}_r$rank.txt" 2>&1

for p in $parts; do
  mkdir -p "$O/$p"
  case $p in
    q8)
      cd "$ROOT/qwen3.8-27b-gfx942" || exit 1
      MX=$MODELS/Qwen3.8-27B-Quark-AWQ-MXFP4
      hashes q8 "$MX"; isa q8 "$MX"
      for m in 1 2 4 8; do gate q8 test_decode_m$m "$O/q8/test_decode_m$m.txt" python -B tests/test_decode.py "$MX" $m 64 16; done
      for m in 1 8; do gate q8 determinism_m$m "$O/q8/determinism_m$m.txt" python -B tests/test_determinism.py "$MX" $m 64; done
      gate q8 eval_text "$O/q8/eval_text.txt" python -B tests/eval_text.py "$MX" "$MODELS/Qwen3.8-27B"
      ;;
    q8tp2)
      cd "$ROOT/qwen3.8-27b-gfx942" || exit 1
      MX=$MODELS/Qwen3.8-27B-Quark-AWQ-MXFP4
      export ARS="fp32 bf16"
      hashes q8tp2 "$MX"; isa q8tp2 "$MX"
      # TP2 hard gates (qwen3.8 docs/DESIGN-TP2.md, "Gates from milestone 4 on"): ranks
      # bitwise equal every step, determinism, eval_text (KL, batch invariance). The
      # test_decode PASS line (c1) is report only: it fails at M = 1 seed 0 on the
      # baseline as documented there (1 - worst cos 0.000897). G1 (median 1 - cos against
      # TP1) follows from equal logits hashes.
      for ar in fp32 bf16; do
        for m in 1 8; do
          log=$O/q8tp2/test_decode_${ar}_m${m}_r$rank.txt
          gate q8tp2 ranks_equal_${ar}_m$m "$log" sh -c "MK_TP=2 MK_AR=$ar python -B tests/test_decode.py $MX $m 64 16; [ $rank != 0 ] || grep -q 'ranks bitwise equal every step: True' $log"
          gate q8tp2 determinism_${ar}_m$m "$O/q8tp2/determinism_${ar}_m${m}_r$rank.txt" env MK_TP=2 MK_AR=$ar python -B tests/test_determinism.py "$MX" $m 64
        done
      done
      gate q8tp2 eval_text_fp32 "$O/q8tp2/eval_text_fp32_r$rank.txt" env MK_TP=2 MK_AR=fp32 python -B tests/eval_text.py "$MX" "$MODELS/Qwen3.8-27B"
      unset ARS
      ;;
    q6)
      cd "$ROOT/qwen3.6-35b-a3b-gfx942" || exit 1
      F8=$MODELS/Qwen3.6-35B-A3B-FP8
      hashes q6 "$F8"; isa q6 "$F8"
      for m in 1 2 4 8; do gate q6 test_decode_m$m "$O/q6/test_decode_m$m.txt" python -B tests/test_decode.py "$F8" $m 40 32; done
      for m in 1 8; do
        gate q6 determinism_m$m "$O/q6/determinism_m$m.txt" python -B tests/test_determinism.py "$F8" $m 32
        gate q6 chaos_m$m "$O/q6/chaos_m$m.txt" python -B tests/test_chaos.py "$F8" $m 16
      done
      gate q6 eval_text "$O/q6/eval_text.txt" python -B tests/eval_text.py "$F8" "$MODELS/Qwen3.6-35B-A3B"
      ;;
    g)
      cd "$ROOT/gemma4-26b-a4b-gfx942" || exit 1
      F8=$MODELS/gemma-4-26B-A4B-it-FP8-dynamic
      hashes g "$F8"; isa g "$F8"
      for m in 1 2 4 8; do gate g test_decode_m${m}_ctx512 "$O/g/test_decode_m${m}_ctx512.txt" python -B tests/test_decode.py "$F8" $m 30 16 512; done
      gate g test_decode_m8_ctx2048 "$O/g/test_decode_m8_ctx2048.txt" python -B tests/test_decode.py "$F8" 8 30 16 2048
      for m in 1 8; do gate g determinism_m$m "$O/g/determinism_m$m.txt" python -B tests/test_determinism.py "$F8" $m 1100; done
      gate g eval_text "$O/g/eval_text.txt" python -B tests/eval_text.py "$F8" "$MODELS/gemma-4-26B-A4B-it" 1536
      gate g test_attention "$O/g/test_attention.txt" python -B tests/test_attention.py
      ;;
  esac
done
[ "$rank" = 0 ] && date -Is >> "$O/env_${parts// /_}_r$rank.txt"
exit 0
