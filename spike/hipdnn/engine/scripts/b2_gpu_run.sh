#!/bin/bash
# Slice B GPU smoke on a gfx942 node, inside the hipDNN image; /work = node-local /tmp dir
# holding install/ and wheel/ copied from the build node, plus mk_custom_op.py.
set -uo pipefail
cd /work
echo "host=$(hostname) date=$(date -Is)"
arch=$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | sort -u | tr '\n' ' '); echo "arch=$arch"
[ -n "$arch" ] || { echo "no GPU visible"; exit 1; }
export PIP_CACHE_DIR=/work/pipcache
[ -x venv/bin/python ] || { python3 -m venv --system-site-packages venv && venv/bin/pip install -q numpy; }
export LD_LIBRARY_PATH=/work/install/lib:/opt/rocm/lib
export PYTHONPATH=/work/wheel
export HIPDNN_LOG_LEVEL=${HIPDNN_LOG_LEVEL:-warn}
find install -path "*megakernel*" | sort
venv/bin/python mk_custom_op.py ${ARGS:-}
