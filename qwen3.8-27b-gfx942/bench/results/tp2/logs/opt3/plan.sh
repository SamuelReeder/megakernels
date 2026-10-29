# Optimization round 3: 16-byte pushes are now the fp32 code. K chunks: base (lm=8),
# lm=4, lm=4 + out=1 + o=1, interleaved per M. Then the all-reduce timing split of the
# base (with and without ar_local, M = 1 and 8) to decide on the prologue merge.
for m in 1 2 4 8; do
  for rep in a b; do
    two base_${rep}_bench_m$m MK_AR=fp32 -- python bench/bench_decode.py $MX $m 512 64 3
    two lm4_${rep}_bench_m$m MK_AR=fp32 MK_CHUNKS=lm=4 -- python bench/bench_decode.py $MX $m 512 64 3
    two lm4oo_${rep}_bench_m$m MK_AR=fp32 MK_CHUNKS=lm=4,out=1,o=1 -- python bench/bench_decode.py $MX $m 512 64 3
  done
done
for m in 1 8; do
  two ars_m$m MK_AR=fp32 MK_TIMEOUT=900 -- python scratch/m3/arsplit.py $MX $m 512 5 $OUT/ar_fp32_m$m
  two ars_local_m$m MK_AR=fp32 MK_AR_LOCAL=1 MK_TIMEOUT=900 -- \
    python scratch/m3/arsplit.py $MX $m 512 5 $OUT/ar_fp32_local_m$m
done
