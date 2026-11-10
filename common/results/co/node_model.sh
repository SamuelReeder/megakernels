#!/bin/bash
# One model of a code-object set, on the node, in the model directory (common/scripts/vrun:
# vLLM image, the checkout's vllm/ in ../vsrc): export twice (a, b), compare the two runs
# byte for byte (determinism), then check H1 on run a.
# Usage: bash ../common/results/co/node_model.sh SETDIR MODEL WEIGHTS VLLM_COMMIT MK_COMMIT DEPLOY...
# Output: SETDIR/MODEL/{a,b}/ (manifest.json, *.co, export.log), SETDIR/MODEL/logs/.
set -uo pipefail
S=$1 model=$2 W=$3 vc=$4 mc=$5
shift 5
O=$S/$model
mkdir -p "$O/logs"
dep=()
for d in "$@"; do dep+=(--deploy "$d"); done
{ echo "# $(date -Is) $(hostname) $(cat ../REVISION)"; rocminfo | grep -m1 -oE 'gfx[0-9a-f]+'
  rocm-smi --showproductname 2>/dev/null | grep -m1 -i "card series"
  python -c 'import importlib.metadata as m, torch; print("flydsl", m.version("flydsl"), "torch", torch.__version__, "dev", torch.cuda.get_device_name(0), "cus", torch.cuda.get_device_properties(0).multi_processor_count)'
} > "$O/logs/env.txt" 2>&1
cat "$O/logs/env.txt"
rc=0
for r in a b; do
  rm -rf "${O:?}/$r"
  python -B ../common/scripts/export_co.py --model "$model" --weights "$W" --vllm-dir ../vsrc \
    --vllm-commit "$vc" --mk-commit "$mc" --out "$O/$r" "${dep[@]}" > "$O/logs/export_$r.log" 2>&1
  echo "export $r: exit $? $(grep -h '^MANIFEST' "$O/logs/export_$r.log")"
done
{ echo "# determinism: export runs a and b, $(date -Is) $(hostname)"
  if cmp "$O/a/manifest.json" "$O/b/manifest.json"; then echo "manifest.json a == b: EQUAL"
  else echo "manifest.json a == b: DIFFERENT"; rc=1; fi
  (cd "$O/a" && sha256sum manifest.json *.co) > "$O/logs/sha_a.txt"
  (cd "$O/b" && sha256sum manifest.json *.co) > "$O/logs/sha_b.txt"
  if cmp "$O/logs/sha_a.txt" "$O/logs/sha_b.txt"; then echo "files and sha256 a == b: EQUAL"
  else echo "files and sha256 a == b: DIFFERENT"; rc=1; fi
  cat "$O/logs/sha_a.txt"; } > "$O/logs/determinism.log" 2>&1
cat "$O/logs/determinism.log"
python -B ../common/scripts/h1_co.py --model "$model" --weights "$W" --vllm-dir ../vsrc \
  --co-dir "$O/a" "${dep[@]}" > "$O/logs/h1.log" 2>&1 || rc=1
grep -hE '^H1|^# H1' "$O/logs/h1.log"
exit $rc
