# M6 diagnosis: rank equality of the kernel's output inside vLLM at TP2 (every step,
# graph replays included) and the epoch words of both ranks.
VS=/cluster/logs/qwen38-tp2/vllm-src
E="VLLM_USE_V2_MODEL_RUNNER=1 TOKENIZERS_PARALLELISM=false MK_TIMEOUT=5400 PYTHONPATH=$VS MK_TP=2 RANKCHECK_DIR=$OUT"
vllm_step m6_rankcheck $E -- python scratch/m6/rankcheck_e2e.py $MX on $OUT/on_rankcheck.json 128
