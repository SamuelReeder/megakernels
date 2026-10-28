# Stock vLLM TP2 on the BF16 checkpoint (the FP8 checkpoint is gone from ~/mk/models).
vllm_step vllm_tp2_Qwen3.8-27B TP=2 VLLM_ROCM_USE_AITER=1 OUT_ROOT=$OUT MK_TIMEOUT=5400 -- \
  bash bench/stock/vllm_serve_bench.sh Qwen3.8-27B vllm_aiter_tp2_Qwen3.8-27B
