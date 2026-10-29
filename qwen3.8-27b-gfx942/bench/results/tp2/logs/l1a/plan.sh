# 1-layer check: kernel residual against the emulating reference in TP1's and TP2's sum
# order, M = 8, 3 seeds; TP1 for comparison.
two l1_tp2fp32 MK_AR=fp32 -- python scratch/m3/dbg.py $MX 8 1 3
one l1_tp1 0 -- python scratch/m3/dbg.py $MX 8 1 3
wait
