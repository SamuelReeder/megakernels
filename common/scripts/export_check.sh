#!/bin/bash
# Oracle 5 (common/DESIGN.md "Test strategy"): export every model into scratch clones of
# its vLLM branch, from a base megakernels tree and from a new one, diff the two exports,
# and build the M = 1 kernel from the new export (common/tests/export_isa.py): its
# normalized ISA hash must equal the standalone build's. Also builds the standalone kernel
# in the same image (tag "standalone") to separate the image from the export.
# Runs on a GPU node in the vLLM nightly image (has aiter, FlyDSL 0.4.0), node /tmp only.
# Usage: export_check.sh OUT BASE_TREE NEW_TREE [model ...]   (models: q8 q6 g)
# Prints "EXPORT <model> diff-lines <n> files <list>" and "ISA5 <model> <tag> <sha256 16>".
set -u
O=$(realpath -m "$1") BASE=$2 NEW=$3
shift 3
mkdir -p "$O"
W=/tmp/export_check_$$
mkdir -p "$W"
trap 'rm -rf "$W"' EXIT
export FLYDSL_RUNTIME_ENABLE_CACHE=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PIP_CACHE_DIR=$W/pip
MODELS=${MODELS:-$HOME/mk/models}
VLLM=https://github.com/SamuelReeder/vllm.git
norm() { sed -E 's/(qwen38|qwen36moe|gemma4)_decode_m[0-9]+_[0-9a-f]{8}/KERNEL/g' "$1" | grep -vE '^\s*(;|//)'; }
for m in ${*:-q8 q6 g}; do
  case $m in
    q8) dir=qwen3.8-27b-gfx942 br=qwen38-gfx942-tp2 pkg=qwen3_5 w=$MODELS/Qwen3.8-27B-Quark-AWQ-MXFP4 ;;
    q6) dir=qwen3.6-35b-a3b-gfx942 br=qwen36-gfx942-specialized-decode pkg=qwen3_5_moe w=$MODELS/Qwen3.6-35B-A3B-FP8 ;;
    g) dir=gemma4-26b-a4b-gfx942 br=gemma4-moe-batch pkg=gemma4 w=$MODELS/gemma-4-26B-A4B-it-FP8-dynamic ;;
  esac
  k=vllm/models/$pkg/amd/specialized_decode/kernels
  git clone -q --depth 1 -b "users/sareeder/$br" "$VLLM" "$W/$m-base" || { echo "EXPORT $m clone-failed"; continue; }
  git -C "$W/$m-base" log --oneline -1 > "$O/vllm_$m.txt"
  if [ ! -x "$W/ruff/bin/ruff" ]; then
    v=$(grep -A2 'ruff-pre-commit' "$W/$m-base/.pre-commit-config.yaml" | grep -m1 -oE 'v[0-9.]+' | tr -d v)
    python -m pip install -q --prefix "$W/ruff" "ruff==$v" > "$O/ruff_install.txt" 2>&1
    echo "ruff $v" >> "$O/ruff_install.txt"
  fi
  export RUFF=$W/ruff/bin/ruff
  cp -a "$W/$m-base" "$W/$m-new"
  (cd "$BASE/$dir" && python -B scripts/export_vllm.py "$W/$m-base"; echo "rc $?") > "$O/export_${m}_base.txt" 2>&1
  (cd "$NEW/$dir" && python -B scripts/export_vllm.py "$W/$m-new"; echo "rc $?") > "$O/export_${m}_new.txt" 2>&1
  diff -ru "$W/$m-base/$k" "$W/$m-new/$k" > "$O/export_$m.diff"
  git -C "$W/$m-new" status --porcelain > "$O/export_${m}_vs_branch.txt"
  git -C "$W/$m-new" diff --stat >> "$O/export_${m}_vs_branch.txt"
  echo "EXPORT $m diff-lines $(grep -cE '^[-+][^-+]' "$O/export_$m.diff") files $(grep -E '^(diff|Only)' "$O/export_$m.diff" | awk '{print $NF}' | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
  for tag in export standalone; do
    mod=${pkg:+vllm.models.$pkg.amd.specialized_decode.kernels}
    [ $tag = standalone ] && mod=mk
    d=$W/dump_${m}_$tag
    PYTHONPATH=$W/$m-new FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=$d \
      python -B "$NEW/common/tests/export_isa.py" "$NEW/$dir" "$mod" "$w" > "$O/isa5_${m}_$tag.txt" 2>&1
    f=$(ls "$d"/*/22_final_isa.s 2>/dev/null | head -1)
    if [ -z "$f" ]; then echo "ISA5 $m $tag NO-DUMP"; continue; fi
    norm "$f" > "$O/isa5_${m}_$tag.s"
    echo "ISA5 $m $tag $(sha256sum < "$O/isa5_${m}_$tag.s" | cut -c1-16)"
  done
done | tee "$O/summary.txt"
