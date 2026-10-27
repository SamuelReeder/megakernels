#!/bin/bash
# Create the pinned Python env for the gfx942 megakernels (shared with qwen3.8-27b-gfx942).
# Run inside hipdnn_latest_gfx942.sqsh on a MARKHAM node (shared home).
set -euo pipefail
ENV=${MK_ENV:-$HOME/mk/venv}
export PIP_CACHE_DIR=/tmp/pipcache
if [ ! -x "$ENV/bin/python" ]; then python3 -m venv "$ENV"; fi
. "$ENV/bin/activate"
pip install -q --upgrade pip
pip install -q --index-url https://download.pytorch.org/whl/rocm7.2 torch==2.14.1+rocm7.2
pip install -q flydsl==0.4.0 numpy==2.5.4 safetensors==0.8.0 huggingface_hub==2.2.0
# tests/eval_text.py and tests/ref_vs_hf.py only
pip install -q transformers==5.20.2 tokenizers==0.23.3 accelerate==1.15.0
python - <<'PY'
import torch, flydsl, numpy
print("torch", torch.__version__, "hip", torch.version.hip, "dev", torch.cuda.get_device_name(0),
      "cus", torch.cuda.get_device_properties(0).multi_processor_count,
      "arch", torch.cuda.get_device_properties(0).gcnArchName)
print("flydsl", flydsl.__version__ if hasattr(flydsl, "__version__") else "?", "numpy", numpy.__version__)
PY
