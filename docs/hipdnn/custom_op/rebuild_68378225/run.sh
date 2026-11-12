#!/bin/bash
# hipdnn_vllm custom_op, rebuilt branch: wheels built in the vLLM image, then tests.
#  old: the previous head's wheel (OLDWHL) with REBUILT's tests: the NUL-key and torch.compile
#       regression tests must fail on it.
#  A: rocm-libraries REBUILT (the PR branch, no engine): tests/unit (no GPU) + test_custom_op.py
#  C: REBUILT, gfx950 family wheel (build only; the node is gfx942).
#  B: ENGINE + REBUILT's hipdnn_vllm/ (= the local merge, not pushed): test_custom_op.py,
#     production catalog and stand-in catalog.
# Runs inside the container (srun --container-image=VIMG --container-writable).
set -uo pipefail
: "${REBUILT:?}" "${ENGINE:?}" "${MERGE_TREE:?}" "${SETDIR:?}" "${KP_SHA:?}" "${OUT:?}" "${STANDIN:?}" "${OLDWHL:?}"
W=/tmp/hvco_rb_${SLURM_JOB_ID:-local}_$$
trap 'cd /; rm -rf "$W"' EXIT
export PIP_CACHE_DIR=$W/pipcache XDG_CACHE_HOME=$W/cache PYTHONNOUSERSITE=1 DVC_NO_ANALYTICS=1 FLYDSL_RUNTIME_ENABLE_CACHE=0
mkdir -p "$W" "$OUT" && cd "$W" || exit 2
echo "start $(date -Is) $(hostname) nproc=$(nproc) arch=$(rocminfo | grep -m1 -oE 'gfx[0-9a-f]+')"
grep -m1 PRETTY_NAME /etc/os-release; ldd --version | head -1
git init -q src && git -C src remote add origin https://github.com/ROCm/rocm-libraries.git \
  && git -C src sparse-checkout init --cone \
  && git -C src sparse-checkout set cmake dnn-providers projects/hipdnn shared .dvc || exit 2
for i in 1 2 3 4; do
  git -C src fetch -q --depth 1 --filter=blob:none origin "$REBUILT" "$ENGINE" && break
  echo "fetch attempt $i failed $(date -Is)"; sleep 60
done
git -C src checkout -q "$REBUILT" || exit 2
echo "src A $(git -C src log --oneline -1)"
T=src/projects/hipdnn/tools/hipdnn_vllm
D=dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/megakernel

python3.12 -m venv tools && tools/bin/pip install -q cmake ninja pytest build numpy \
  "zstandard>=0.20.0" msgpack "dvc[s3]==3.62.0" "pathspec<1" || exit 2
mkdir -p venv/bin
printf '#!/bin/sh\nexec /usr/bin/python3.12 "$@"\n' > venv/bin/python
printf '#!/bin/sh\nexec /usr/local/bin/rocm-sdk "$@"\n' > venv/bin/rocm-sdk
chmod +x venv/bin/python venv/bin/rocm-sdk
PY=$W/venv/bin/python
$PY -c "import torch, torchvision, vllm; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'vllm', vllm.__version__)" || exit 2
$PY -m pip install -q pytest 2>&1 | tail -1
venv/bin/rocm-sdk init > /dev/null 2>&1
mkdir kp && git -C kp init -q . && git -C kp remote add origin https://github.com/ROCm/rocm-systems.git \
  && git -C kp sparse-checkout set shared/kpack && git -C kp fetch -q --depth 1 --filter=blob:none origin "$KP_SHA" \
  && git -C kp checkout -q FETCH_HEAD || exit 2

build() {  # build NAME [FAMILY]
  rm -rf "${OUT:?}/$1"; local t0; t0=$(date +%s)
  VENV=$W/venv WORK=$W/work OUT=$OUT/$1 KPACK_PYTHON_DIR=$W/kp/shared/kpack/python TOOLS_VENV=$W/tools \
    JOBS=$(nproc) bash "$T/scripts/build_wheel.sh" "${2:-gfx94X}" > "$OUT/build_$1.log" 2>&1
  if grep -q "^WHEEL_OK" "$OUT/build_$1.log" && ls "$OUT/$1"/*.whl > /dev/null 2>&1; then
    echo "PASS build $1 $(ls "$OUT/$1"/*.whl) ($(( $(date +%s) - t0 )) s)"
  else echo "FAIL build $1 (build_$1.log)"; grep -nE "error:|FAILED" "$OUT/build_$1.log" | head -20; return 1; fi
}
install() {  # install NAME
  $PY -m pip install -q --no-deps --force-reinstall "$OUT/$1"/*.whl || return 1
  $PY -c "import json,os,hipdnn_vllm;d=json.load(open(os.path.join(os.path.dirname(hipdnn_vllm.__file__),'_build_info.json')));print('installed', d['plugin_commit'], 'dirty', d['plugin_tree_dirty_files'], d['provider_cmake_flags'][-1])"
}
P="$PY -m pytest -p no:cacheprovider -rfs"
rm -rf tA && mkdir tA && cp -r $T/tests tA/

# ---- old: previous head's wheel, this head's tests (failing before)
$PY -m pip install -q --no-deps --force-reinstall "$OLDWHL" || exit 1
echo "== old wheel $(basename "$(dirname "$OLDWHL")"): NUL-key and torch.compile tests (expected to fail)"
(cd tA && $P -v tests/gpu/test_custom_op.py -k "nul or compile" > "$OUT/custom_op_old.log" 2>&1); echo "custom_op old rc=$?"
grep -E "PASS|FAIL|SKIP|passed|failed" "$OUT/custom_op_old.log" | cut -c1-240

# ---- A: the PR branch
build A || exit 1
install A || exit 1
echo "== A unit (source tree, no GPU)"
(cd $T && HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= $P -q tests/unit > "$OUT/unit_A.log" 2>&1); echo "unit rc=$?"
grep -E "^(FAILED|ERROR)|passed|failed" "$OUT/unit_A.log" | cut -c1-240
# ---- C: the PR branch, gfx950 family (build only)
build C gfx950 || echo "gfx950 build failed"
grep -E "^provider flags|^gpu targets" "$OUT/build_C.log"
echo "== A test_custom_op (installed wheel, GPU)"
(cd tA && $P -v tests/gpu/test_custom_op.py > "$OUT/custom_op_A.log" 2>&1); echo "custom_op A rc=$?"
grep -E "PASS|FAIL|SKIP|passed|failed" "$OUT/custom_op_A.log"

# ---- B: engine + the PR branch's hipdnn_vllm/ (the local merge's tree)
git -C src checkout -q "$ENGINE" && git -C src checkout "$REBUILT" -- projects/hipdnn/tools/hipdnn_vllm || exit 2
git -C src -c user.name=node -c user.email=node@local commit -qm "node: engine $ENGINE + hipdnn_vllm of $REBUILT" || exit 2
echo "src B $(git -C src log --oneline -1) tree $(git -C src rev-parse HEAD^{tree}) expect $MERGE_TREE"
[ "$(git -C src rev-parse HEAD^{tree})" = "$MERGE_TREE" ] && echo "PASS tree B = local merge tree" || echo "INFO sparse tree differs from full merge tree (sparse paths only)"
# megakernel descriptor folder from SETDIR, recorded in the local DVC cache (nothing pushed)
rm -rf "src/$D/artifacts" && mkdir -p "src/$D/artifacts"
python3.12 - "$SETDIR" "src/$D" <<'PY' || exit 1
import json, pathlib, shutil, sys
src, d = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
for e in json.loads((src / "manifest.json").read_text()):
    shutil.copyfile(src / e["file"], d / "artifacts" / e["file"])
PY
cmp "$SETDIR/manifest.json" "src/$D/manifest.json" && echo "PASS set manifest = engine lock"
python3.12 src/dnn-providers/hip-kernel-provider/descriptor-packaging/tools/hkp_megakernel_lock.py generate --lock "src/$D/manifest.json" || exit 1
python3.12 src/dnn-providers/hip-kernel-provider/descriptor-packaging/tools/hkp_megakernel_lock.py verify --lock "src/$D/manifest.json" || exit 1
(cd src && ../tools/bin/dvc add "$D/artifacts" > /dev/null) || exit 1
build B || exit 1
install B || exit 1
rm -rf tB && mkdir tB && cp -r $T/tests tB/ && mkdir standin && tar xzf "$STANDIN" -C standin
echo "== B test_custom_op (production catalog)"
(cd tB && $P -v tests/gpu/test_custom_op.py > "$OUT/custom_op_B.log" 2>&1); echo "custom_op B rc=$?"
grep -E "PASS|FAIL|SKIP|passed|failed" "$OUT/custom_op_B.log"
echo "== B test_custom_op -k megakernel (stand-in catalog)"
(cd tB && HIPDNN_DESCRIPTOR_DIR=$W/standin/integration $P -v tests/gpu/test_custom_op.py -k megakernel > "$OUT/custom_op_B_standin.log" 2>&1); echo "custom_op B standin rc=$?"
grep -E "PASS|FAIL|SKIP|passed|failed" "$OUT/custom_op_B_standin.log"
echo "done $(date -Is)"
