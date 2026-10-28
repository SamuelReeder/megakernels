# Optimization round 1 (same node, same day): deferred poll (MK_AR_DEFER) and the second
# grid_wait (MK_ONE_WAIT, measure-only toggle). Speed: bench_decode ctx 512, 64 steps, 3 reps.
# Bits: test_determinism logits sha256 per build (must match where the change is bit-neutral).
for m in 1 8; do
  one tp1_bench_m$m 0 -- python bench/bench_decode.py $MX $m 512 64 3
  wait
  one tp1w1_bench_m$m 0 MK_ONE_WAIT=1 -- python bench/bench_decode.py $MX $m 512 64 3
  wait
done
for m in 1 2 4 8; do
  two tp2_d0_bench_m$m MK_AR=fp32 MK_AR_DEFER=0 -- python bench/bench_decode.py $MX $m 512 64 3
  two tp2_d1_bench_m$m MK_AR=fp32 MK_AR_DEFER=1 -- python bench/bench_decode.py $MX $m 512 64 3
done
for m in 1 8; do
  two tp2w1_bench_m$m MK_AR=fp32 MK_ONE_WAIT=1 -- python bench/bench_decode.py $MX $m 512 64 3
done
for m in 1 8; do
  one tp1_det_m$m 0 -- python tests/test_determinism.py $MX $m 64
  wait
  one tp1w1_det_m$m 0 MK_ONE_WAIT=1 -- python tests/test_determinism.py $MX $m 64
  wait
  two tp2_d0_det_m$m MK_AR=fp32 MK_AR_DEFER=0 -- python tests/test_determinism.py $MX $m 64
  two tp2_d1_det_m$m MK_AR=fp32 MK_AR_DEFER=1 -- python tests/test_determinism.py $MX $m 64
  two tp2w1_det_m$m MK_AR=fp32 MK_ONE_WAIT=1 -- python tests/test_determinism.py $MX $m 64
done
