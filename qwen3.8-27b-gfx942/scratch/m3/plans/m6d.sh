# M6 diagnosis: rank equality of the kernel's output inside vLLM at TP2 (every step,
# graph replays included) and the epoch words of both ranks (scratch/m6/rankcheck), and checksums of
# every GDN state before the first 64 launches, in two runs.
VS=/cluster/logs/qwen38-tp2/vllm-src
mkdir -p $OUT/r1 $OUT/r2
E="VLLM_USE_V2_MODEL_RUNNER=1 TOKENIZERS_PARALLELISM=false MK_TIMEOUT=5400 MK_TP=2"
for r in 1 2; do
  vllm_step m6_rankcheck_r$r $E PYTHONPATH=$SRC/scratch/m6/rankcheck:$VS RANKCHECK_DIR=$OUT/r$r -- python bench/vllm_e2e.py $MX on $OUT/on_rankcheck_r$r.json 128
done
