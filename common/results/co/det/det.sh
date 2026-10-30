#!/bin/bash
# node, in the vLLM image; $1 = extracted bundle dir
B=$1; K=$B/vsrc/vllm/models/qwen3_5_moe/amd/specialized_decode/kernels
export CO_SCRIPTS=$B/common/scripts DET_OUT=$B/det FLYDSL_RUNTIME_ENABLE_CACHE=0 FLYDSL_COMPILE_ARCH=gfx942 FLYDSL_GPU_ARCH=gfx942 HIP_VISIBLE_DEVICES=
mkdir -p $DET_OUT
python -c 'import importlib.metadata as m; print("flydsl", m.version("flydsl"))'
for r in 1 2 3; do python -B /home/AMD/sareeder/coexport/det/det.py bare $K 2 1 2 2>&1 | grep -E "^DET|Error|error" | head -8; done
for r in 1 2; do python -B /home/AMD/sareeder/coexport/det/det.py vllm $K 1 1 2 2>&1 | grep -E "^DET|Error|error" | head -8; done
