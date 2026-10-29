#!/bin/bash
# Runs inside a container on the held gfx942 node (srun --jobid --overlap). Usage:
#   run_node.sh venv SRC OUT   hipdnn image + ~/mk/venv: minimal test, Qwen3.6 model test
#   run_node.sh vllm SRC OUT   vLLM nightly image: environment, minimal test, buffer_ops copy
# SRC holds qwen3.6-35b-a3b-gfx942/ and common/results/cache/. OUT gets the logs.
# Logs in logs/ came from: run 1 = this script in both modes (cache_min_venv_run1.txt,
# cache_min_vllm.txt, env_*.txt, buffer_ops files, cache_model_qwen36_m1_s1.txt: S1, then S2
# hit the old 1800 s timeout); run 2 = `python -B cache_min.py WORK MK_DIR` with case X added
# (cache_min_venv.txt); run 3 = `python -B cache_model.py MK_DIR FP8_DIR WORK S2,S3`
# (cache_model_qwen36_m1_s2s3.txt). Container: hipdnn_latest_gfx942.sqsh + ~/mk/venv.
set -uo pipefail
mode=$1 src=$2 out=$3
mkdir -p "$out"
w=/tmp/mkcache_work_${mode}_$$
rm -rf "$w"; mkdir -p "$w"
q36=$src/qwen3.6-35b-a3b-gfx942
t=$src/common/results/cache
unset FLYDSL_RUNTIME_ENABLE_CACHE FLYDSL_RUNTIME_CACHE_DIR PYTHONPYCACHEPREFIX
if [ "$mode" = venv ]; then . "$HOME/mk/venv/bin/activate"; fi
{
  date -Is; hostname; echo "job ${SLURM_JOB_ID:-?} mode $mode"
  rocminfo | grep -m1 -oE 'gfx[0-9a-f]+'
  python -c "import torch, flydsl, os; print('torch', torch.__version__, 'flydsl', flydsl.__version__, flydsl.__file__); print('FLYDSL env', {k: v for k, v in os.environ.items() if k.startswith('FLYDSL')})"
  python - <<'PY'
from flydsl.utils import env
print("enable_cache default", env.runtime.enable_cache, "cache_dir", env.runtime.cache_dir)
PY
  cat "$src/REVISION" 2>/dev/null
} > "$out/env_$mode.txt" 2>&1
python -B "$t/cache_min.py" "$w/min" "$q36" > "$out/cache_min_$mode.txt" 2>&1
if [ "$mode" = venv ]; then
  python -B "$t/cache_model.py" "$q36" "$HOME/mk/models/Qwen3.6-35B-A3B-FP8" "$w/model" \
    > "$out/cache_model_qwen36_m1.txt" 2>&1
else
  python - > "$out/aiter_buffer_ops.txt" 2>&1 <<PY
import shutil, aiter, aiter.ops.flydsl.kernels.buffer_ops as b
print("aiter", getattr(aiter, "__version__", "?"), b.__file__)
shutil.copy(b.__file__, "$out/aiter_buffer_ops.py")
PY
  diff "$q36/mk/vendor/buffer_ops.py" "$out/aiter_buffer_ops.py" > "$out/buffer_ops.diff"
  echo "diff exit $? (0 = identical)" >> "$out/aiter_buffer_ops.txt"
fi
rm -rf "$w"
