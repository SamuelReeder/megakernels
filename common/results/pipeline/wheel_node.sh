#!/bin/bash
# Pipeline stage d_wheel, on a compute node inside the hipdnn image (Ubuntu 24, as the
# hipdnn_env docker of hipdnn_vllm/notes/custom_op/NOTES.md): the hipdnn_vllm gfx94X wheel
# of rocm-libraries RL_SHA with the code-object set SETDIR as the megakernel descriptor
# folder (DVC-local, as step8/e2e_plan.sh stage: lock, artifacts, generated KDP, verify,
# `dvc add` so configure's file count matches), then the drop4 wheel (the set without its
# M = 4 objects, for H2) from the same build tree. Nothing is pushed.
# Usage: wheel_node.sh SETDIR OUTDIR   (env RL_SHA, KP_SHA: rocm-systems commit of kpack)
# Output: OUTDIR/wheel/*.whl, OUTDIR/wheel-drop4/*.whl, OUTDIR/{lock_verify,
#   lock_verify_drop4,wheel_build,wheel_build_drop4}.txt|log; PASS/FAIL lines on stdout.
set -uo pipefail
final=$1 out=$2
W=/tmp/pipe_wheel_${SLURM_JOB_ID:-local}_$$
trap 'cd /; rm -rf "$W"' EXIT
export PIP_CACHE_DIR=$W/pipcache XDG_CACHE_HOME=$W/cache PYTHONNOUSERSITE=1
mkdir -p "$W" "$out" && cd "$W" || exit 2
echo "start $(date -Is) $(hostname) nproc=$(nproc) rocm-libraries $RL_SHA set $final"
git init -q src && git -C src remote add origin https://github.com/ROCm/rocm-libraries.git \
  && git -C src fetch -q --depth 1 origin "$RL_SHA" && git -C src checkout -q FETCH_HEAD || exit 2
echo "src $(git -C src log --oneline -1)"
D=dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/megakernel
TOOL=src/dnn-providers/hip-kernel-provider/descriptor-packaging/tools/hkp_megakernel_lock.py
T=src/projects/hipdnn/tools/hipdnn_vllm

# build tools as build_wheel.sh installs them (it reuses this TOOLS_VENV)
python3.12 -m venv tools && tools/bin/pip install -q cmake ninja pytest build numpy \
  "zstandard>=0.20.0" msgpack "dvc[s3]==3.62.0" "pathspec<1" || exit 2

# desc DROP_M: the set (without width DROP_M if given) as the descriptor folder; the full
# set keeps the lock file byte for byte
desc() {
  rm -rf "src/$D/artifacts" && mkdir -p "src/$D/artifacts"
  python3.12 - "$final" "src/$D" "$1" <<'PY' || return 1
import json, pathlib, shutil, sys
src, d, drop = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
entries = json.loads((src / "manifest.json").read_text())
keep = [e for e in entries if not (drop and e["M"] == int(drop))]
for e in keep:
    shutil.copyfile(src / e["file"], d / "artifacts" / e["file"])
if drop:
    (d / "manifest.json").write_text(json.dumps(keep, indent=1) + "\n")
else:
    shutil.copyfile(src / "manifest.json", d / "manifest.json")
print(f"descriptor folder: {len(keep)} of {len(entries)} objects (drop M={drop or '-'})")
PY
  python3.12 "$TOOL" generate --lock "src/$D/manifest.json" || return 1
  python3.12 "$TOOL" verify --lock "src/$D/manifest.json" || return 1
  # configure compares the folder's file count (and size) with artifacts.dvc: record the
  # local folder in the pointer (no DVC cache or remote is involved)
  python3.12 - "src/$D/artifacts" <<'PY' || return 1
import pathlib, re, sys
a = pathlib.Path(sys.argv[1]); p = a.with_name("artifacts.dvc")
files = [f for f in a.rglob("*") if f.is_file()]
s = re.sub(r"(?m)^  nfiles: \d+$", f"  nfiles: {len(files)}", p.read_text())
s = re.sub(r"(?m)^  size: \d+$", f"  size: {sum(f.stat().st_size for f in files)}", s)
p.write_text(s)
PY
  grep -E "nfiles|size" "src/$D/artifacts.dvc"
}

# the pinned vLLM venv (BUILD_PINS.toml), as notes/custom_op's build
set -- $(python3.12 -c "import tomllib;p=tomllib.load(open('$T/BUILD_PINS.toml','rb'))['pins'];print(p['vllm'],p['rocm'],p['vllm_index'])")
# torchvision pinned: whl-next also serves 0.29.0+rocm10.0.0, which pip prefers and which
# does not load with torch 2.13.0 (hipdnn_vllm notes/custom_op/NOTES.md)
python3.12 -m venv venv && venv/bin/pip install -q --upgrade pip \
  && venv/bin/pip install -q "vllm==$1" "rocm[devel]==$2" "torchvision==0.28.0+rocm10.0.0" \
       --extra-index-url "$3" --extra-index-url https://stable.repo.amd.com/rocm/whl-next/ || exit 2
venv/bin/python -c "import torch, vllm; print('torch', torch.__version__, 'vllm', vllm.__version__)" || exit 2
mkdir kp && git -C kp init -q . && git -C kp remote add origin https://github.com/ROCm/rocm-systems.git \
  && git -C kp sparse-checkout set shared/kpack && git -C kp fetch -q --depth 1 --filter=blob:none origin "$KP_SHA" \
  && git -C kp checkout -q FETCH_HEAD || exit 2

build() {  # build OUT_SUBDIR LOGNAME
  rm -rf "${out:?}/$1"
  VENV=$W/venv WORK=$W/work OUT=$out/$1 KPACK_PYTHON_DIR=$W/kp/shared/kpack/python TOOLS_VENV=$W/tools \
    JOBS=$(nproc) bash "$T/scripts/build_wheel.sh" gfx94X > "$out/$2.log" 2>&1
  grep -q "^WHEEL_OK" "$out/$2.log" && ls "$out/$1"/*.whl > /dev/null 2>&1
}

rc=0
desc "" > "$out/lock_verify.txt" 2>&1; r=$?; cat "$out/lock_verify.txt"
if [ $r = 0 ]; then echo "PASS lock verify (full set)"; else echo "FAIL lock verify (full set)"; exit 1; fi
t0=$(date +%s)
if build wheel wheel_build; then echo "PASS wheel $(ls "$out"/wheel/*.whl) ($(( $(date +%s) - t0 )) s)"
else echo "FAIL wheel (see $out/wheel_build.log)"; grep -nE "error:|FAILED|Error" "$out/wheel_build.log" | head -20; exit 1; fi
desc 4 > "$out/lock_verify_drop4.txt" 2>&1; r=$?; cat "$out/lock_verify_drop4.txt"
if [ $r = 0 ]; then echo "PASS lock verify (drop M=4)"; else echo "FAIL lock verify (drop M=4)"; exit 1; fi
t0=$(date +%s)
if build wheel-drop4 wheel_build_drop4; then echo "PASS wheel-drop4 $(ls "$out"/wheel-drop4/*.whl) ($(( $(date +%s) - t0 )) s)"
else echo "FAIL wheel-drop4 (see $out/wheel_build_drop4.log)"; rc=1; fi
echo "done $(date -Is)"
exit $rc
