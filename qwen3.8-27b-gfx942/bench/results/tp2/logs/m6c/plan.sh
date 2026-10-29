# M6 repeatability diagnosis: vLLM's TP2 path is not run-to-run repeatable with the
# kernel off (m6b). Same runs with vLLM's custom all-reduce disabled (RCCL only).
VS=/cluster/logs/qwen38-tp2/vllm-src
E="VLLM_USE_V2_MODEL_RUNNER=1 TOKENIZERS_PARALLELISM=false MK_TIMEOUT=5400 PYTHONPATH=$VS MK_TP=2 MK_NO_CUSTOM_AR=1"
for r in 1 2; do
  vllm_step m6_off_noca_r$r $E -- python bench/vllm_e2e.py $MX off $OUT/off_noca_r$r.json 128
  vllm_step m6_on_noca_r$r $E -- python bench/vllm_e2e.py $MX on $OUT/on_noca_r$r.json 128
done
