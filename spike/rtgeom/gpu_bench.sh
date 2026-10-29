#!/bin/bash
# Spike rtgeom (disposable): A/B TPOT (bench_ab.py) of rt_geom 0 (specialized), 1 and 2 on
# one node, run by rtrun hipdnn in the model dir.
# Usage: bash ../spike/rtgeom/gpu_bench.sh short|long|xpack|profile [Ms=1,2,4,8]
# Logs: /work/out/bench/.
set -uo pipefail
MX=$HOME/mk/models/Qwen3.8-27B-Quark-AWQ-MXFP4
O=/work/out/bench
mkdir -p "$O"
what=$1 Ms=${2:-1,2,4,8}
for m in ${Ms//,/ }; do
  case $what in
    short) args=(512 64 3 640) ;;   # bench_decode's maxlen for ctx 512: ((512+64+63)//64+1)*64
    long) args=(3500 64 3 4096) ;;
    xpack) args=(3500 64 3 4096); export MK_XPACK=1 MK_BLOCK=784 ;;
    profile)
      for rt in 0 1 2; do
        MK_RTGEOM=$rt python -B bench/profile_decode.py "$MX" $m 3500 > "$O/profile_long_rt${rt}_m$m.log" 2>&1
        echo "== profile M=$m rt_geom=$rt"; tail -n 12 "$O/profile_long_rt${rt}_m$m.log"
      done
      continue ;;
  esac
  f=$O/ab_${what}_m$m
  python -B ../spike/rtgeom/bench_ab.py "$MX" $m "${args[0]}" "${args[1]}" "${args[2]}" "${args[3]}" 0,1,2 \
    "$f.json" > "$f.log" 2>&1
  grep -E "SUMMARY|Error|error" "$f.log" | cut -c1-330
done
