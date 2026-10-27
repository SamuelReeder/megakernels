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
