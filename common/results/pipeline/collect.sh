#!/bin/bash
# After the pipeline (~/mk/out/pipeline/state/ALL_DONE on Alola; partial results also work):
# fetch the results into each branch's result folder and commit them there. No push.
#   hipdnn-design  common/results/pipeline/out/  driver log, summary, stage logs, state
#                  common/results/step8/gpu/     e2e runs, drop4, verdicts (wheels: sha256 only)
#   cfg-compiled   common/results/compiled/gpu/  ~/mk/out/cc_<job> (as gpu.sh fetch)
#   co-export      common/results/co/<set>/      ~/coexport/sets/<set> without run b (as
#                  run_set.sh); .co files only if all are under 5 MB, else their sha256 only
#   megakernel-engine (rocm-libraries) dnn-providers/hip-kernel-provider/megakernel-notes/logs-gpu/
# Usage: common/results/pipeline/collect.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
WS=/home/sareeder/ROCm-workspace
MKW=$WS/worktrees/megakernels
A="ssh -o BatchMode=yes alola"
commit() {  # commit REPO PATH MESSAGE
  git -C "$1" add -A -- "$2"
  if git -C "$1" diff --cached --quiet -- "$2"; then echo "nothing new in $1/$2"; return; fi
  git -C "$1" -c user.name="Samuel Reeder" -c user.email=samuel.reeder@amd.com commit -q -m "$3" -- "$2"
  echo "committed $(git -C "$1" log --oneline -1) in $1"
}
fetch() {  # fetch REMOTE_DIR LOCAL_DIR [tar options...]
  local r=$1 l=$2; shift 2
  $A "test -d $r" || { echo "no $r on Alola"; return 1; }
  mkdir -p "$l" && $A "tar -C $r -czf - $* ." | tar -C "$l" -xzf -
}
stages=$($A 'cat ~/mk/out/pipeline/state/stages.json')
job_of() { python3 -c "import json,sys; print(json.loads(sys.argv[1])['$1'].get('job', ''))" "$stages"; }

# hipdnn-design: pipeline logs and the e2e results
H=$MKW/hipdnn-design
fetch '~/mk/out/pipeline' "$H/common/results/pipeline/out" --exclude=./e2e --exclude=driver.lock --exclude='*.tmp'
if fetch '~/mk/out/pipeline/e2e' "$H/common/results/step8/gpu" --exclude='*.whl'; then
  $A 'cd ~/mk/out/pipeline/e2e && sha256sum wheel*/*.whl' > "$H/common/results/step8/gpu/wheels.sha256" || true
fi
commit "$H" common/results/pipeline/out "pipeline: GPU run logs and summary (driver $(date -I))"
commit "$H" common/results/step8/gpu "step 8: GPU e2e results of the pipeline (runs, drop4, verdicts)"

# cfg-compiled: gpu.sh's plan output of the a_cc job
j=$(job_of a_cc)
if [ -n "$j" ] && fetch "~/mk/out/cc_$j" "$MKW/cfg-compiled/common/results/compiled/gpu" "--exclude='*.s'"; then
  commit "$MKW/cfg-compiled" common/results/compiled/gpu "compiled: GPU evidence of gpu.sh plan (pipeline, job $j)"
fi

# co-export: the rt0 and rt2 sets
for set in rt0 rt2; do
  big=$($A "find ~/coexport/sets/$set -name '*.co' -size +5M 2>/dev/null | wc -l" || echo 1)
  ex="--exclude=./*/b"
  [ "$big" = 0 ] || ex="$ex --exclude='*.co'"
  if fetch "~/coexport/sets/$set" "$WS/worktrees/megakernels/co-export/common/results/co/$set" $ex; then
    [ "$big" = 0 ] || $A "cd ~/coexport/sets/$set && sha256sum */a/*.co final/*.co" \
      > "$MKW/co-export/common/results/co/$set/co_on_alola.sha256"
    commit "$MKW/co-export" "common/results/co/$set" \
      "co-export: $set set from the pipeline (manifests, determinism, H1; .co $([ "$big" = 0 ] && echo included || echo "on Alola ~/coexport/sets/$set"))"
  fi
done

# megakernel-engine: the GPU integration run
R=$WS/worktrees/rocm-libraries/megakernel-engine
L=dnn-providers/hip-kernel-provider/megakernel-notes/logs-gpu
if fetch '~/work/mkeng/logs-gpu' "$R/$L"; then
  $A 'cat ~/mk/out/pipeline/b_mkeng.out' > "$R/$L/gpu-run.log" 2>/dev/null || true
  commit "$R" "$L" "docs(hip-kernel-provider): megakernel engine gfx942 GPU run logs (pipeline)"
fi
$A 'cat ~/mk/out/pipeline/summary.txt' | head -40
