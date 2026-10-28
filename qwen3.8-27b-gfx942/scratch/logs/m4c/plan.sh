# Milestone 4, rest: per-stage profiles (after the profiler fix: its buffers were on GPU 0
# in both ranks, so rank 1's stamps went to the other GPU and its first step hung), then
# stock vLLM TP2.
two pdiag2_tp2fp32_m1 MK_AR=fp32 MK_TIMEOUT=400 -- python bench/profile_decode.py $MX 1 8
for m in 1 8; do
  [ $m = 8 ] && { one tp1_prof_m$m 0 -- python bench/profile_decode.py $MX $m 512 $OUT/tl_tp1_m$m.json; wait; }
  for ar in fp32 bf16; do
    for loc in 0 1; do
      two tp2${ar}l${loc}_prof_m$m MK_AR=$ar MK_AR_LOCAL=$loc MK_TIMEOUT=900 -- \
        python bench/profile_decode.py $MX $m 512 $OUT/tl_tp2${ar}l${loc}_m$m.json
    done
  done
done
for c in Qwen3.8-27B-FP8 Qwen3.8-27B-Quark-AWQ-MXFP4; do
  vllm_step vllm_tp2_$c TP=2 VLLM_ROCM_USE_AITER=1 OUT_ROOT=$OUT MK_TIMEOUT=5400 -- \
    bash bench/stock/vllm_serve_bench.sh $c vllm_aiter_tp2_$c
done
