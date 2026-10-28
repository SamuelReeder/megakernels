# All-reduce timing split (DESIGN-TP2 section 11): stamped steps, fp32 / bf16, with the
# all-reduce and with ar_local, M = 1 and 8. 512 warm-up steps, then 5 stamped steps.
for m in 1 8; do
  for ar in fp32 bf16; do
    two ars_${ar}_m$m MK_AR=$ar MK_TIMEOUT=900 -- python scratch/m3/arsplit.py $MX $m 512 5 $OUT/ar_${ar}_m$m
    two ars_${ar}_local_m$m MK_AR=$ar MK_AR_LOCAL=1 MK_TIMEOUT=900 -- \
      python scratch/m3/arsplit.py $MX $m 512 5 $OUT/ar_${ar}_local_m$m
  done
done
