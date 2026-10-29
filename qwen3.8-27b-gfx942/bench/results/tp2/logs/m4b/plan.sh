# Milestone 4, rest of m4 after the ar_local hang (the embedding exchange of ar_local
# builds, fixed): bf16 benches, ar_local benches, profiles, stock vLLM TP2.
for m in 1 2 4 8; do
  two tp2bf16_bench_m$m MK_AR=bf16 -- python bench/bench_decode.py $MX $m 512 64 3
done
for ar in fp32 bf16; do
  for m in 1 8; do
    two tp2${ar}local_bench_m$m MK_AR=$ar MK_AR_LOCAL=1 -- python bench/bench_decode.py $MX $m 512 64 3
  done
done
for m in 1 8; do
  one tp1_prof_m$m 0 -- python bench/profile_decode.py $MX $m 512 $OUT/tl_tp1_m$m.json
  wait
  for ar in fp32 bf16; do
    for loc in 0 1; do
      two tp2${ar}l${loc}_prof_m$m MK_AR=$ar MK_AR_LOCAL=$loc -- \
        python bench/profile_decode.py $MX $m 512 $OUT/tl_tp2${ar}l${loc}_m$m.json
    done
  done
done
for c in Qwen3.8-27B-FP8 Qwen3.8-27B-Quark-AWQ-MXFP4; do
  vllm_step vllm_tp2_$c TP=2 VLLM_ROCM_USE_AITER=1 OUT_ROOT=$OUT -- \
    bash bench/stock/vllm_serve_bench.sh $c vllm_aiter_tp2_$c
done
