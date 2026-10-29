# Stock vLLM TP2 FP8 (nightly, AITER), engine-level TPOT, 3 independent runs (median).
for r in 1 2 3; do
  vllm_step vllm_lat_tp2_fp8_r$r TP=2 VLLM_ROCM_USE_AITER=1 OUT_ROOT=$OUT MK_TIMEOUT=5400 -- \
    bash bench/stock/vllm_latency_bench.sh Qwen3.8-27B-FP8 vllm_lat_tp2_fp8_r$r
done
