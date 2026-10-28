# 1-layer kernel-vs-reference smoke tests (TP1 regression, TP2 both formats), then one
# full 64-layer TP2 run.
one tp1_l1_m2 0 -- python tests/test_decode.py $MX 2 1 4
wait
two tp2fp32_l1_m2 MK_AR=fp32 -- python tests/test_decode.py $MX 2 1 4
two tp2bf16_l1_m2 MK_AR=bf16 -- python tests/test_decode.py $MX 2 1 4
two tp2fp32_l64_m1 MK_AR=fp32 -- python tests/test_decode.py $MX 1 64 16
