#!/bin/bash
B=$1; K=$B/vsrc/vllm/models/qwen3_5_moe/amd/specialized_decode/kernels
export CO_SCRIPTS=$B/common/scripts DET_OUT=$B/det FLYDSL_RUNTIME_ENABLE_CACHE=0 FLYDSL_COMPILE_ARCH=gfx942 FLYDSL_GPU_ARCH=gfx942 HIP_VISIBLE_DEVICES= FLYDSL_DUMP_IR=1
mkdir -p $DET_OUT
for r in 1 2 3 4; do FLYDSL_DUMP_DIR=$B/dump$r python -B /home/AMD/sareeder/coexport/det/det.py bare $K 1 2 2>&1 | grep -E "^DET|Error" | head -3; done
cd $B; for f in $(cd dump1/*/ && ls); do echo "$f $(for r in 1 2 3 4; do md5sum < dump$r/*/$f | cut -c1-8; done | tr '\n' ' ')"; done
tar -czf /home/AMD/sareeder/coexport/det/dumps.tgz dump1 dump2 dump3 dump4
