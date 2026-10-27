# Failure rate of the test_decode criterion at more seeds (3..7), M = 1 and 2:
# TP1 (two processes in parallel), then TP2 fp32 and bf16.
for s in 3 4 5 6 7; do
  one tp1_dec_m1_s$s 0 MK_SEED=$s -- python tests/test_decode.py $MX 1 64 16
  one tp1_dec_m2_s$s 1 MK_SEED=$s -- python tests/test_decode.py $MX 2 64 16
  wait
done
for ar in fp32 bf16; do
  for s in 3 4 5 6 7; do
    for m in 1 2; do
      two tp2${ar}_dec_m${m}_s$s MK_AR=$ar MK_SEED=$s -- python tests/test_decode.py $MX $m 64 16
    done
  done
done
