# FP8 baseline on this node: the kernel (TP1 on GPU 0, TP2 fp32) and stock vLLM TP2
# (nightly, AITER): engine-level TPOT (vllm bench latency) for FP8, BF16, MXFP4, and the
# server TPOT (vllm bench serve) for FP8.
for m in 1 2 4 8; do
  one tp1_bench_m$m 0 -- python bench/bench_decode.py $MX $m 512 64 3
  wait
done
for m in 1 2 4 8; do
  two tp2fp32_bench_m$m MK_AR=fp32 -- python bench/bench_decode.py $MX $m 512 64 3
done
for c in Qwen3.8-27B-FP8 Qwen3.8-27B Qwen3.8-27B-Quark-AWQ-MXFP4; do
  vllm_step vllm_lat_tp2_$c TP=2 VLLM_ROCM_USE_AITER=1 OUT_ROOT=$OUT MK_TIMEOUT=5400 -- \
    bash bench/stock/vllm_latency_bench.sh $c vllm_lat_tp2_$c
done
vllm_step vllm_tp2_Qwen3.8-27B-FP8 TP=2 VLLM_ROCM_USE_AITER=1 OUT_ROOT=$OUT MK_TIMEOUT=5400 -- \
  bash bench/stock/vllm_serve_bench.sh Qwen3.8-27B-FP8 vllm_aiter_tp2_Qwen3.8-27B-FP8
