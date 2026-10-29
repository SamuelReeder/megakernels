two pdiag_tp2fp32_m1 MK_AR=fp32 MK_TIMEOUT=400 -- python bench/profile_decode.py $MX 1 8
two pdiag_tp2fp32_m1_nopro MK_AR=fp32 MK_TIMEOUT=400 -- python bench/bench_decode.py $MX 1 8 8 1
