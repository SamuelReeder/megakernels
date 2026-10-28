# M6 diagnosis: rank equality of the kernel's output inside vLLM at TP2 (every step,
# graph replays included) and the epoch words of both ranks (scratch/m6/rankcheck).
VS=/cluster/logs/qwen38-tp2/vllm-src
E="VLLM_USE_V2_MODEL_RUNNER=1 TOKENIZERS_PARALLELISM=false MK_TIMEOUT=5400 MK_TP=2 RANKCHECK_DIR=$OUT"
vllm_step m6_rankcheck $E PYTHONPATH=$SRC/scratch/m6/rankcheck:$VS -- python bench/vllm_e2e.py $MX on $OUT/on_rankcheck.json 128
