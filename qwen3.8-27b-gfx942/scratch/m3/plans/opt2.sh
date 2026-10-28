# Optimization round 2 (deferred poll and the single grid_wait are now the code):
# new baseline, the SGPR candidate (MK_AR_OPAQUE=1, measure-only) with ISA spill counts,
# and K-chunk variants at TP2 (MK_CHUNKS). bench_decode ctx 512, 64 steps, 3 reps.
for m in 1 2 4 8; do
  two base_bench_m$m MK_AR=fp32 -- python bench/bench_decode.py $MX $m 512 64 3
done
for m in 1 8; do
  two opq_bench_m$m MK_AR=fp32 MK_AR_OPAQUE=1 -- python bench/bench_decode.py $MX $m 512 64 3
done
two base_det_m1 MK_AR=fp32 -- python tests/test_determinism.py $MX 1 64
two opq_det_m1 MK_AR=fp32 MK_AR_OPAQUE=1 -- python tests/test_determinism.py $MX 1 64
for v in base opq; do
  o=0; [ $v = opq ] && o=1
  two isa_$v MK_AR=fp32 MK_AR_OPAQUE=$o FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=$OUT/isa_$v -- \
    python bench/bench_decode.py $MX 1 8 4 1
done
for ch in "out=1,o=1,down=2" "down=2" "lm=4"; do
  t=$(echo $ch | tr ',=' '__')
  for m in 1 8; do
    two ch_${t}_bench_m$m MK_AR=fp32 MK_CHUNKS=$ch -- python bench/bench_decode.py $MX $m 512 64 3
  done
done
for m in 1 8; do
  two v16_bench_m$m MK_AR=fp32 MK_AR_V16=1 -- python bench/bench_decode.py $MX $m 512 64 3
done
two v16_det_m1 MK_AR=fp32 MK_AR_V16=1 -- python tests/test_determinism.py $MX 1 64
two v16_det_m8 MK_AR=fp32 MK_AR_V16=1 -- python tests/test_determinism.py $MX 8 64
two base_det_m8 MK_AR=fp32 -- python tests/test_determinism.py $MX 8 64
