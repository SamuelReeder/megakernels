# M6 follow-up: run-to-run repeatability in vLLM (kernel on twice, off twice at TP2;
# kernel on twice at TP1 with the TP1 branch overlay ~/mk/vllm-src on GPU 0), then the
# conformance kit with tblib installed.
VS=/cluster/logs/qwen38-tp2/vllm-src
E="VLLM_USE_V2_MODEL_RUNNER=1 TOKENIZERS_PARALLELISM=false MK_TIMEOUT=5400"
for r in 1 2; do
  vllm_step m6_on_r$r $E PYTHONPATH=$VS MK_TP=2 -- python bench/vllm_e2e.py $MX on $OUT/on_r$r.json 128
  vllm_step m6_off_r$r $E PYTHONPATH=$VS MK_TP=2 -- python bench/vllm_e2e.py $MX off $OUT/off_r$r.json 128
done
for r in 1 2; do
  vllm_step m6_tp1on_r$r $E PYTHONPATH=/home/AMD/sareeder/mk/vllm-src HIP_VISIBLE_DEVICES=0 -- python bench/vllm_e2e.py $MX on $OUT/tp1on_r$r.json 128
done
vllm_step m6_kit PYTHONPATH=$VS MK_TIMEOUT=1800 -- bash -c "pip install -q tblib && cd $VS && python -m pytest -q tests/models/qwen3_5/test_qwen3_5_specialized_decode.py tests/models/common/specialized_decode"
