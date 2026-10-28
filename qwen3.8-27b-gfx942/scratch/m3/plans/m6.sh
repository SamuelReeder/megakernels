# Milestone 6: vLLM TP2 (overlay /cluster/logs/qwen38-tp2/vllm-src on the nightly image).
# Kernel on with the path check, kernel on (speed), kernel off; greedy outputs + TPOT.
VS=/cluster/logs/qwen38-tp2/vllm-src
E="PYTHONPATH=$VS VLLM_USE_V2_MODEL_RUNNER=1 MK_TP=2 TOKENIZERS_PARALLELISM=false MK_TIMEOUT=5400"
vllm_step m6_on_check $E VLLM_SPECIALIZED_DECODE_CHECK_PATHS=1 -- python bench/vllm_e2e.py $MX on $OUT/on_check.json 128
vllm_step m6_on $E -- python bench/vllm_e2e.py $MX on $OUT/on.json 128
vllm_step m6_off $E -- python bench/vllm_e2e.py $MX off $OUT/off.json 128
vllm_step m6_kit PYTHONPATH=$VS MK_TIMEOUT=1800 -- bash -c "cd $VS && python -m pytest -q tests/models/qwen3_5/test_qwen3_5_specialized_decode.py tests/models/common/specialized_decode"
