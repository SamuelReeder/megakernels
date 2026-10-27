#!/bin/bash
# Run on the alola login node (inside tmux). One MARKHAM gfx942 GPU, SGLang mi30x image,
# sglang.bench_one_batch for each checkpoint given (default BF16, FP8).
IMG=/cluster/images/lmsysorg+sglang-rocm+v0.5.10.post1-rocm700-mi30x-20260428.sqsh
S=/home/AMD/sareeder/mk/stock36
ckpts="${*:-Qwen3.6-35B-A3B Qwen3.6-35B-A3B-FP8}"
srun -p defq,miopen -A miopen -J mkstock-sgl --constraint="GFX942&MARKHAM" --gres=gpu:gfx942-mi300x:1 \
  --cpus-per-task=16 --mem=192G -t 02:00:00 \
  --container-image=$IMG --container-mounts=/cluster:/cluster,/home/AMD/sareeder:/home/AMD/sareeder \
  --container-writable bash -c "export HF_HUB_OFFLINE=1; for c in $ckpts; do echo \"===== \$c \$(date -Is)\"; bash $S/sglang_one_batch.sh \$c; done"
