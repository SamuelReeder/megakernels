#!/bin/bash
# Pipeline stage d_wheel, on a compute node inside the vLLM image the e2e runs use
# (stage.sh): the hipdnn_vllm gfx94X wheel of rocm-libraries RL_SHA with the code-object
# set SETDIR as the megakernel descriptor folder (DVC-local, as step8/e2e_plan.sh stage:
# lock, artifacts, generated KDP, verify, `dvc add`), then the drop4 wheel (the set
# without its M = 4 objects, for H2) from the same build tree. Nothing is pushed.
# The wheel must load in that image (Ubuntu 22.04: glibc 2.35, libstdc++ 6.0.30): a wheel
# built in the Ubuntu 24 hipdnn image needs GLIBC_2.38 and GLIBCXX_3.4.32, its bundled
# backend fails to load and the frontend falls back to rocm-sdk's 0.3.0 backend. So it
# builds against the image's own python, torch 2.13.0+rocm10.0.0 and rocm-sdk 10.0.0
# (the BUILD_PINS versions; e2e_plan.sh WHEEL_ENV).
# Usage: wheel_node.sh SETDIR OUTDIR   (env RL_SHA, KP_SHA: rocm-systems commit of kpack)
# Output: OUTDIR/wheel/*.whl, OUTDIR/wheel-drop4/*.whl, OUTDIR/{lock_verify,
#   lock_verify_drop4,wheel_build,wheel_build_drop4}.txt|log; PASS/FAIL lines on stdout.
set -uo pipefail
final=$1 out=$2
W=/tmp/pipe_wheel_${SLURM_JOB_ID:-local}_$$
trap 'cd /; rm -rf "$W"' EXIT
export PIP_CACHE_DIR=$W/pipcache XDG_CACHE_HOME=$W/cache PYTHONNOUSERSITE=1 DVC_NO_ANALYTICS=1
mkdir -p "$W" "$out" && cd "$W" || exit 2
echo "start $(date -Is) $(hostname) nproc=$(nproc) rocm-libraries $RL_SHA set $final"
# the build reads the root CMake files, cmake/, dnn-providers/, projects/hipdnn/ and shared/
# (about 0.3 GB of the 11 GB tree): a sparse, blob-filtered fetch, retried (a full depth-1
# fetch broke off once with "unexpected disconnect while reading sideband packet")
# (cone mode keeps the root files; git 2.34 of Ubuntu 22.04 does not default to it)
git init -q src && git -C src remote add origin https://github.com/ROCm/rocm-libraries.git \
  && git -C src sparse-checkout init --cone \
  && git -C src sparse-checkout set cmake dnn-providers projects/hipdnn shared .dvc || exit 2
for i in 1 2 3 4; do
  git -C src fetch -q --depth 1 --filter=blob:none origin "$RL_SHA" && git -C src checkout -q FETCH_HEAD && break
  echo "fetch attempt $i failed $(date -Is)"; sleep 60
done
[ "$(git -C src rev-parse HEAD 2>/dev/null)" = "$RL_SHA" ] || exit 2
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
  # configure compares the folder's file count with artifacts.dvc, and build_wheel.sh runs
  # `dvc pull --recursive` on the descriptors: record the folder in the local DVC cache and
  # the pointer (nothing is pushed). DVC writes its site cache under /var/tmp: the
  # container must be writable (stage.sh --container-writable).
  (cd src && ../tools/bin/dvc add "$D/artifacts") || return 1
  grep -E "nfiles|size" "src/$D/artifacts.dvc"
}

# VENV for build_wheel.sh: the image's python (its site-packages hold torch, rocm-sdk and
# vLLM) and rocm-sdk
# (wrappers, not symlinks: a symlinked python in venv/bin takes venv/ as its prefix;
# absolute paths: the mounted home's ~/.local/bin on PATH has another python3.12)
mkdir -p venv/bin || exit 2
printf '#!/bin/sh\nexec /usr/bin/python3.12 "$@"\n' > venv/bin/python
printf '#!/bin/sh\nexec /usr/local/bin/rocm-sdk "$@"\n' > venv/bin/rocm-sdk
chmod +x venv/bin/python venv/bin/rocm-sdk
venv/bin/python -c "import torch, vllm; print('torch', torch.__version__, 'vllm', vllm.__version__)" || exit 2
venv/bin/rocm-sdk init > /dev/null 2>&1
cat /etc/os-release | grep -m1 PRETTY_NAME; ldd --version | head -1
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
