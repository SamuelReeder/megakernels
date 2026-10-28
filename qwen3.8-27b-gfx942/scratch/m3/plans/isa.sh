# ISA of the TP2 kernels (register and spill counts): one short real run each, IR dumps on.
for ar in fp32 bf16; do
  for loc in 0 1; do
    two isa_${ar}_l$loc MK_AR=$ar MK_AR_LOCAL=$loc FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=$OUT/isa_${ar}_l$loc -- \
      python bench/bench_decode.py $MX 1 8 4 1
  done
done
