# Down-stage per-wave probes (last down layer): why bf16 is slower than fp32 at M = 1.
for ar in fp32 bf16; do
  for loc in 0 1; do
    two pdown_${ar}_l$loc MK_AR=$ar MK_AR_LOCAL=$loc PROBE=down DETAIL=1 MK_TIMEOUT=900 -- \
      python bench/profile_decode.py $MX 1 512
  done
done
