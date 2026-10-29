#!/bin/bash
# Spike rtgeom (disposable): the standalone oracles, run by rtrun hipdnn in the model dir.
# Usage: bash ../spike/rtgeom/gpu_checks.sh PART...   PART: hash | decode | determinism
# Logs: /work/out/checks/.
set -uo pipefail
MX=$HOME/mk/models/Qwen3.8-27B-Quark-AWQ-MXFP4
O=/work/out/checks
mkdir -p "$O"
rocminfo | grep -m1 -oE 'gfx[0-9a-f]+'
for part in "$@"; do
  case $part in
    hash)  # baseline M=1 4db972e0ec306d4f, M=8 cf305425fce9de61 (spike/hipdnn/c/logs/c1_final.log)
      for rt in 0 1 2; do
        MK_RTGEOM=$rt python -B tests/logits_hash.py "$MX" 1,2,4,8 > "$O/hash_rt$rt.log" 2>&1
        echo "rt_geom=$rt:"; grep HASH "$O/hash_rt$rt.log"
      done ;;
    decode)  # vs the torch reference (64 layers, 16 steps), plain and x-packed KV
      for m in 1 8; do
        MK_RTGEOM=1 python -B tests/test_decode.py "$MX" $m 64 16 > "$O/test_decode_rt1_m$m.log" 2>&1
        MK_RTGEOM=1 MK_XPACK=1 MK_BLOCK=784 python -B tests/test_decode.py "$MX" $m 64 16 \
          > "$O/test_decode_rt1_xpack784_m$m.log" 2>&1
        for f in "$O"/test_decode_rt1_m$m.log "$O"/test_decode_rt1_xpack784_m$m.log; do
          echo "$(basename "$f"): $(grep -E 'worst cos|PASS|FAIL' "$f" | tr '\n' ' ')"
        done
      done ;;
    determinism)
      for m in 1 8; do
        MK_RTGEOM=1 python -B tests/test_determinism.py "$MX" $m 64 > "$O/determinism_rt1_m$m.log" 2>&1
        echo "determinism rt1 M=$m: $(grep -E 'bitwise' "$O/determinism_rt1_m$m.log")"
      done ;;
  esac
done
