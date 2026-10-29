# GDN-state check (DESIGN-TP2 section 11): two vLLM TP2 runs with the kernel off, two with
# it on, each with the statecheck hook (scratch/m6/statecheck); compare with statecmp.py.
VS=/cluster/logs/qwen38-tp2/vllm-src
E="VLLM_USE_V2_MODEL_RUNNER=1 TOKENIZERS_PARALLELISM=false MK_TIMEOUT=5400 MK_TP=2"
for mode in off on; do
  for r in 1 2; do
    mkdir -p $OUT/${mode}_r$r
    vllm_step m6s_${mode}_r$r $E PYTHONPATH=$SRC/scratch/m6/statecheck:$VS STATECHECK_DIR=$OUT/${mode}_r$r -- python bench/vllm_e2e.py $MX $mode $OUT/${mode}_r$r.json 128
  done
done
