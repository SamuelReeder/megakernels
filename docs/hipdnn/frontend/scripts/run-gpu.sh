#!/bin/bash
# Build the archived merge of the frontend and engine on gfx942 and run the Python oracle.
set -euo pipefail
cd "$(dirname "$0")"
work=$PWD
mkdir -p src logs
exec >&2
trap 'echo "GPU run exit=$? $(date -Is)"' EXIT
export PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HIP_VISIBLE_DEVICES=0
{
    cat source-revisions
    echo "host=$(hostname) job=${SLURM_JOB_ID:-none} cpus=$(nproc) date=$(date -Is)"
    rocminfo > logs/rocminfo.log
    grep -m1 -E 'Name:[[:space:]]*gfx942' logs/rocminfo.log
    cmake --version
    python3 --version
} | tee logs/platform.log
tar xzf source.tgz -C src
tar xzf mk-art3.tgz -C src/dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/megakernel
export LD_LIBRARY_PATH="$work/build/lib:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
{
    cat source-revisions
    set -x
    cmake --preset hip-kernel-provider -S src -B build -G Ninja \
        '-DROCM_LIBS_ENABLE_COMPONENTS=hipdnn;hipdnn-integration-tests;hip-kernel-provider;hipdnn-python' \
        -DROCM_PATH=/opt/rocm -DGPU_TARGETS=gfx942 -DHIPDNN_ENABLE_SDPA=ON \
        -DENABLE_CLANG_TIDY=OFF -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON -DENABLE_MEGAKERNEL_ENGINE=ON \
        -DPython_EXECUTABLE="$(command -v python3)" -DPython3_EXECUTABLE="$(command -v python3)"
} > logs/configure-gpu.log 2>&1
{
    cat source-revisions
    set -x
    cmake --build build -j "${SLURM_CPUS_PER_TASK:-$(nproc)}" --target \
        hipdnn_frontend_bindings hip_kernel_provider hkp_packaging_integration hkp_packaging_product
} > logs/build-gpu.log 2>&1
set +x
mkdir -p package/hipdnn_frontend
cp src/projects/hipdnn/python/frontend_wheel_package/src/hipdnn_frontend/__init__.py package/hipdnn_frontend/
find build -name 'hipdnn_frontend_python*.so' -exec cp {} package/hipdnn_frontend/ \;
plugin="$work/build/lib/hipdnn_plugins/engines/libhip_kernel_provider.so"
export HIPDNN_DESCRIPTOR_DIR="${plugin%/*}/test_arch_content/hip-kernel-provider/integration"
export HIPDNN_DESCRIPTOR_RUNTIME_DIR="${plugin%/*}/arch_content/hip-kernel-provider"
{
    cat source-revisions
    printf 'descriptor root: %s\n' "$HIPDNN_DESCRIPTOR_DIR"
    test -f "$plugin"
    find "$HIPDNN_DESCRIPTOR_DIR" -iname '*megakernel*'
    PYTHONPATH="$work/package" python3 "$work/run_standin.py" "$plugin"
} | tee logs/stand-in-oracle.log
