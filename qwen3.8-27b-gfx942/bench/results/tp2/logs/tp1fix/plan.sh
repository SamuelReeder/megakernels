# TP1 suite on master + 5e151e9 (second grid_wait removed), from a copy of the TP1 repo at
# $R/src-tp1: test_decode M = 1, 2, 4, 8 (seed 0, 64 layers, 16 steps), determinism M = 1, 8,
# eval_text. GPU 0, one at a time.
T=/cluster/logs/qwen38-tp2/src-tp1
for m in 1 2 4 8; do
  one tp1fix_dec_m$m 0 MK_SEED=0 -- bash -c "cd $T && python tests/test_decode.py $MX $m 64 16"
  wait
done
for m in 1 8; do
  one tp1fix_det_m$m 0 -- bash -c "cd $T && python tests/test_determinism.py $MX $m 64"
  wait
done
one tp1fix_eval 0 -- bash -c "cd $T && python tests/eval_text.py $MX $BF"
wait
