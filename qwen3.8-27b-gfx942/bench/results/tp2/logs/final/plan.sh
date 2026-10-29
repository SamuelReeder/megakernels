# Final TP2 kernel on this node: hard gates (TP2 fp32 and TP1 at the same M and seed,
# determinism with logits hash, eval_text) and speed (TP1 on GPU 0, TP2), M = 1, 2, 4, 8.
for m in 1 2 4 8; do
  one tp1_dec_m${m}_s0 0 MK_SEED=0 -- python tests/test_decode.py $MX $m 64 16
  two tp2_dec_m${m}_s0 MK_AR=fp32 MK_SEED=0 -- python tests/test_decode.py $MX $m 64 16
  wait
done
for m in 1 8; do
  for s in 1 2; do
    one tp1_dec_m${m}_s$s 0 MK_SEED=$s -- python tests/test_decode.py $MX $m 64 16
    two tp2_dec_m${m}_s$s MK_AR=fp32 MK_SEED=$s -- python tests/test_decode.py $MX $m 64 16
    wait
  done
  two tp2_det_m$m MK_AR=fp32 -- python tests/test_determinism.py $MX $m 64
done
two tp2_eval MK_AR=fp32 -- python tests/eval_text.py $MX $BF
one tp1_eval 0 -- python tests/eval_text.py $MX $BF
wait
for m in 1 2 4 8; do
  one tp1_bench_m$m 0 -- python bench/bench_decode.py $MX $m 512 64 3
  wait
  two tp2_bench_m$m MK_AR=fp32 -- python bench/bench_decode.py $MX $m 512 64 3
done
