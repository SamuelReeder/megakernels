# Milestone 5: the final run (gates and speed, TP1 and TP2) on another node or day, then
# the vLLM rank-equality check of scratch/m6/rankcheck (two runs).
. $SRC/scratch/m3/plans/final.sh
mkdir -p $OUT/r1 $OUT/r2
VS=/cluster/logs/qwen38-tp2/vllm-src
E="VLLM_USE_V2_MODEL_RUNNER=1 TOKENIZERS_PARALLELISM=false MK_TIMEOUT=5400 MK_TP=2"
for r in 1 2; do
  vllm_step m6_rankcheck_r$r $E PYTHONPATH=$SRC/scratch/m6/rankcheck:$VS RANKCHECK_DIR=$OUT/r$r -- python bench/vllm_e2e.py $MX on $OUT/on_rankcheck_r$r.json 128
done
