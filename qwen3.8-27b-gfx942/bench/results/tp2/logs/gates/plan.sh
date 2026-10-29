# Milestone 3 hard gates after the ARPROBE change (probes off): TP2 fp32 and TP1 at the same
# M and seed, determinism, eval_text.
for m in 1 2 4 8; do
  one tp1_dec_m${m}_s0 0 MK_SEED=0 -- python tests/test_decode.py $MX $m 64 16
  one tp1_dec_m${m}_s0b 1 MK_SEED=0 -- python tests/test_decode.py $MX $m 64 16
  wait
  two tp2fp32_dec_m${m}_s0 MK_AR=fp32 MK_SEED=0 -- python tests/test_decode.py $MX $m 64 16
done
for m in 1 8; do
  two tp2fp32_det_m$m MK_AR=fp32 -- python tests/test_determinism.py $MX $m 64
done
two tp2fp32_eval MK_AR=fp32 -- python tests/eval_text.py $MX $BF
one tp1_eval 0 -- python tests/eval_text.py $MX $BF
wait
