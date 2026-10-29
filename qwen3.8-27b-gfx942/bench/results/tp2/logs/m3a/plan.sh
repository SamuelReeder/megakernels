# TP1 on this node: kernel vs reference at 3 seeds (M = 1, 8), M = 2, 4 (seed 0),
# determinism, eval_text. GPU 0 and GPU 1 run one process each in parallel.
for s in 0 1 2; do
  one tp1_dec_m1_s$s 0 MK_SEED=$s -- python tests/test_decode.py $MX 1 64 16
  one tp1_dec_m8_s$s 1 MK_SEED=$s -- python tests/test_decode.py $MX 8 64 16
  wait
done
one tp1_dec_m2_s0 0 MK_SEED=0 -- python tests/test_decode.py $MX 2 64 16
one tp1_dec_m4_s0 1 MK_SEED=0 -- python tests/test_decode.py $MX 4 64 16
wait
one tp1_det_m1 0 -- python tests/test_determinism.py $MX 1 64
one tp1_det_m8 1 -- python tests/test_determinism.py $MX 8 64
wait
one tp1_eval 0 -- python tests/eval_text.py $MX $BF
wait
# TP2, both all-reduce formats: kernel vs reference (M = 1, 2, 4, 8, seed 0; M = 1, 8 at
# seeds 1, 2), determinism (M = 1, 8), eval_text.
for ar in fp32 bf16; do
  for m in 1 2 4 8; do
    two tp2${ar}_dec_m${m}_s0 MK_AR=$ar MK_SEED=0 -- python tests/test_decode.py $MX $m 64 16
  done
  for s in 1 2; do
    for m in 1 8; do
      two tp2${ar}_dec_m${m}_s$s MK_AR=$ar MK_SEED=$s -- python tests/test_decode.py $MX $m 64 16
    done
  done
  for m in 1 8; do
    two tp2${ar}_det_m$m MK_AR=$ar -- python tests/test_determinism.py $MX $m 64
  done
  two tp2${ar}_eval MK_AR=$ar -- python tests/eval_text.py $MX $BF
done
