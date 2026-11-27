#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
# Stream source and evidence because Austin workers cannot see this login-node home.
worker=$(cat <<'WORKER'
set -euo pipefail
stage=$(mktemp -d /tmp/mk-frontend.XXXXXX)
trap 'rm -rf "$stage"' EXIT
tar xf - -C "$stage"
rc=0
bash "$stage/run-gpu.sh" >&2 || rc=$?
tar cf - -C "$stage/logs" .
exit "$rc"
WORKER
)
set +e
tar cf - source.tgz source-revisions mk-art3.tgz run-gpu.sh run_standin.py | \
    srun -p miopen -A miopen -J mk-frontend-390d38cee6 --constraint=GFX942 \
        --exclude=ctr-cx63-mi300x-12 --gres=gpu:1 --cpus-per-task=32 -t 00:35:00 \
        --container-image=/cluster/images/hipdnn/hipdnn_latest_gfx942.sqsh \
        --container-mounts=/cluster:/cluster --container-workdir=/tmp \
        bash -c "$worker" > logs.tar 2> srun.log
rc=$?
set -e
mkdir -p logs
if [ -s logs.tar ]; then tar xf logs.tar -C logs; fi
printf 'srun_exit=%d %s\n' "$rc" "$(date -Is)" > status.log
exit "$rc"
