#!/bin/bash
# Disposable: one srun task of a milestone 3 run (see run.sh). Usage:
#   task.sh OUT_DIR NAME ENV... -- COMMAND...
# Writes OUT_DIR/NAME.r<rank>.log and appends the exit code to OUT_DIR/status.log.
out=$1 name=$2
shift 2
while [ "$1" != "--" ]; do export "$1"; shift; done
shift
r=${SLURM_PROCID:-0}
export FLYDSL_RUNTIME_ENABLE_CACHE=0 PIP_CACHE_DIR=/tmp/pipcache PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$(dirname "$0")/../.."
[ -n "${NOVENV:-}" ] || . "$HOME/mk/venv/bin/activate"
{ echo "# $(hostname) $(date -u +%FT%TZ) rank $r env MK_TP=${MK_TP:-1} MK_AR=${MK_AR:-} MK_SEED=${MK_SEED:-}"
  echo "# $*"; } > "$out/$name.r$r.log"
timeout "${MK_TIMEOUT:-3600}" "$@" >> "$out/$name.r$r.log" 2>&1
rc=$?
echo "$name rank $r exit $rc $(date -u +%T)" >> "$out/status.log"
