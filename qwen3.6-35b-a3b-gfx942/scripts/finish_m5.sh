#!/bin/bash
# Finish milestone 5 (second node) from a finished SLURM suite job (suite36.sbatch or
# suite36_austin.sbatch on Alola):
#   1. copy its results from Alola to bench/results/m5/<node>_<job>/,
#   2. check them: every gate prints PASS, the node is not the day-1 node, and the
#      median TPOT at batch 1, 2, 4, 8 is within 3 % of day 1 (bench/results/m4/bench_v5_m*.txt),
#   3. replace the README line "Second node: pending (job N)." with the result,
#   4. commit the results and the README.
# Usage (from anywhere on the controller host):
#   qwen3.6-35b-a3b-gfx942/scripts/finish_m5.sh [--dry-run] [--allow-same-node] JOBID
# --dry-run: steps 1 and 2 only. --allow-same-node: skip the "not the day-1 node" gate
# (for a test on a same-node run only). Exits 1 with the reasons when a gate fails; the
# README and git are then unchanged.
set -euo pipefail
DRY=0
SAME=0
while [ $# -gt 1 ]; do
  case $1 in
    --dry-run) DRY=1 ;;
    --allow-same-node) SAME=1 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
JOB=${1:?usage: finish_m5.sh [--dry-run] [--allow-same-node] JOBID}
cd "$(dirname "$0")/.."

# Result directory: the shared home (suite36.sbatch), else /cluster (suite36_austin.sbatch).
REMOTE=$(ssh -o BatchMode=yes alola "ls -d ~/mk/suite36/*/qwen3.6-35b-a3b-gfx942/bench/results/m5/*_$JOB 2>/dev/null || ls -d /cluster/sareeder/mkhome/mk/suite36/*/qwen3.6-35b-a3b-gfx942/bench/results/m5/*_$JOB 2>/dev/null | head -1")
[ -n "$REMOTE" ] || { echo "no results for job $JOB on Alola" >&2; exit 1; }
STATE=$(ssh -o BatchMode=yes alola "sacct -j $JOB -X -n -o State | head -1" | tr -d ' ')
NAME=$(basename "$REMOTE")
OUT=bench/results/m5/$NAME
mkdir -p "$OUT"
rsync -a "alola:$REMOTE/" "$OUT/"
LOG=$(ssh -o BatchMode=yes alola "ls ~/mk/suite36/*/*_$JOB.out 2>/dev/null || ls /cluster/sareeder/mkhome/*_$JOB.out 2>/dev/null")
[ -z "$LOG" ] || scp -q "alola:$LOG" "$OUT/slurm.out"
echo "job $JOB ($STATE): $REMOTE -> $OUT"

LINE=$(python3 - "$OUT" "$JOB" "$STATE" "$SAME" <<'PY'
import json, math, os, statistics, sys

out, job, state, same = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"
bad = []


def lines(name):
    p = os.path.join(out, name)
    if not os.path.exists(p):
        bad.append(f"{name}: missing")
        return []
    return [x for x in open(p).read().splitlines() if x.strip()]


if state != "COMPLETED":
    bad.append(f"job state {state}")
for name in ["ref_vs_hf.txt", "eval_text.txt"] + [f"test_decode_m{m}.txt" for m in (1, 2, 4, 8)] \
        + [f"determinism_m{m}.txt" for m in (1, 8)]:
    ls = lines(name)
    if ls and ls[-1] != "PASS":
        bad.append(f"{name}: last line {ls[-1]!r}")


def bench(path):
    d = json.loads(open(path).read().strip().splitlines()[-1])
    return d, statistics.median(r["tpot_ms"] for r in d["runs"])


tpot, ratio, node, gpu, date = {}, {}, None, None, None
for b in (1, 2, 4, 8):
    p = os.path.join(out, f"bench_m{b}.txt")
    try:
        d, t = bench(p)
    except Exception as e:  # missing file or a crashed run: no JSON line
        bad.append(f"bench_m{b}.txt: {type(e).__name__}")
        continue
    ref, t1 = bench(f"bench/results/m4/bench_v5_m{b}.txt")
    tpot[b], ratio[b] = t, t / t1 - 1
    node, gpu, date = d["node"].split(".")[0], d["gpu"], d["date"][:10]
    if abs(ratio[b]) > 0.03:
        bad.append(f"batch {b}: TPOT {t:.3f} ms vs day 1 {t1:.3f} ms ({100 * ratio[b]:+.1f} %)")
    day1 = ref["node"].split(".")[0]
if node is not None and node == day1 and not same:
    bad.append(f"ran on the day-1 node {node}, not a second node")
ppl = [x for x in lines("eval_text.txt") if x.startswith("all:")]
rev = (lines("env.txt") or ["?"])[-1]
if bad:
    print("FAIL: " + "; ".join(bad), file=sys.stderr)
    sys.exit(1)
worst = max(abs(r) for r in ratio.values())
print(
    f"Second node (`bench/results/m5/{os.path.basename(out)}/`, {date}, {gpu} on {node},\n"
    f"commit `{rev}`): every test passes, batch-invariant and bitwise deterministic,\n"
    f"perplexity {ppl[0].split()[3]}, TPOT "
    + " / ".join(f"{tpot[b]:.2f}" for b in (1, 2, 4, 8))
    + f" ms at batch 1 / 2 / 4 / 8 (within {max(1, math.ceil(100 * worst))} % of day 1)."
)
PY
)
echo "README line:"
echo "$LINE"
[ "$DRY" = 0 ] || exit 0

grep -qE '^Second node: pending \(job [0-9]+\)\.$' README.md || {
  echo "README has no 'Second node: pending (job N).' line" >&2
  exit 1
}
python3 - "$LINE" <<'PY'
import re, sys

t = open("README.md").read()
t = re.sub(r"^Second node: pending \(job \d+\)\.$", lambda _: sys.argv[1], t, count=1, flags=re.M)
open("README.md", "w").write(t)
PY
git add "$OUT" README.md
git commit -q -m "qwen3.6-35b-a3b-gfx942: second-node suite run ($NAME)"
git log --oneline -1
