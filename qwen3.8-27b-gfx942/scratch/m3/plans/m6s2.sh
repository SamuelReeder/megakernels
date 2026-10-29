# Re-run of the kernel-on run 2 of m6s.sh (its container failed to start: no space left on
# /var/tmp on the node, filled by other jobs at that time).
VS=/cluster/logs/qwen38-tp2/vllm-src
E="VLLM_USE_V2_MODEL_RUNNER=1 TOKENIZERS_PARALLELISM=false MK_TIMEOUT=5400 MK_TP=2"
mkdir -p $OUT/on_r2
vllm_step m6s_on_r2 $E PYTHONPATH=$SRC/scratch/m6/statecheck:$VS STATECHECK_DIR=$OUT/on_r2 -- python bench/vllm_e2e.py $MX on $OUT/on_r2.json 128
