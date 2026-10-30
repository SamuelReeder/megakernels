#!/bin/bash
# rt2 evidence: one model's standalone checks of kernel.Cfg.rt_geom 2 (the default) against
# 0, on one GPU, from the model directory, inside the hipdnn gfx942 image with the venv.
# Usage: bash ../common/results/rt2/checks.sh WEIGHTS OUT_DIR PART...
#   PART: hash (tests/logits_hash.py at rt_geom 0 and 2, M = 1, 2, 4, 8), decode
#   (tests/test_decode.py, M = 1 and 8), determinism (M = 1 and 8), ab (../common/results/
#   rt2/ab.py: TPOT rt_geom 0 vs 2, ctx 512, 128 steps, 3 rounds, M = 1 and 8).
set -uo pipefail
W=$1 O=$2
shift 2
mkdir -p "$O"
model=$(basename "$PWD")
rt() {  # rt LEVEL CMD...: run CMD with the model's rt_geom selector
  local v=$1
  shift
  case $model in
    qwen3.8*) MK_RTGEOM=$v "$@" ;;
    *) MK_CFG=rt_geom=$v "$@" ;;
  esac
}
{ date -Is; hostname; rocminfo | grep -m1 -oE 'gfx[0-9a-f]+'; rocm-smi --showproductname 2>/dev/null | grep -m1 -i "card series"
  cat REVISION 2>/dev/null || cat ../REVISION 2>/dev/null; } > "$O/env.txt" 2>&1
cat "$O/env.txt"
for part in "$@"; do
  case $part in
    hash)
      for v in 0 2; do
        rt $v python -B tests/logits_hash.py "$W" 1,2,4,8 > "$O/hash_rt$v.txt" 2>&1
        echo "rt_geom=$v:"; grep '^HASH' "$O/hash_rt$v.txt"
      done
      if diff <(grep '^HASH' "$O/hash_rt0.txt") <(grep '^HASH' "$O/hash_rt2.txt") > /dev/null \
        && [ "$(grep -c '^HASH' "$O/hash_rt2.txt")" -gt 0 ]; then
        echo "hash rt0 == rt2: PASS"
      else
        echo "hash rt0 == rt2: FAIL"
      fi ;;
    decode)
      for m in 1 8; do
        case $model in
          qwen3.8*) a=(64 16) ;;
          qwen3.6*) a=(40 32) ;;
          gemma4*) a=(30 16 2048) ;;
        esac
        python -B tests/test_decode.py "$W" $m "${a[@]}" > "$O/test_decode_m$m.txt" 2>&1
        echo "test_decode M=$m: $(grep -E 'PASS|FAIL' "$O/test_decode_m$m.txt" | tail -1)"
      done ;;
    determinism)
      for m in 1 8; do
        case $model in
          qwen3.8*) s=64 ;;
          qwen3.6*) s=32 ;;
          gemma4*) s=1100 ;;
        esac
        python -B tests/test_determinism.py "$W" $m $s > "$O/determinism_m$m.txt" 2>&1
        echo "determinism M=$m: $(grep -iE 'bitwise|PASS|FAIL' "$O/determinism_m$m.txt" | tail -2 | tr '\n' ' ')"
      done ;;
    ab)
      for m in 1 8; do
        python -B ../common/results/rt2/ab.py "$W" $m 512 128 3 "$O/ab_m$m.json" > "$O/ab_m$m.txt" 2>&1
        grep -E '^SUMMARY|Error' "$O/ab_m$m.txt"
      done ;;
  esac
done
