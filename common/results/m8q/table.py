"""M8 quiet-node re-measure: TPOT table of drive.sh's results (median of the rounds per
variant and batch, min-max, delta vs base), output identity against base_r1, and the node
load recorded before each run (1-minute load average, GPUs allocated on the node by all jobs,
ours included). Usage: python common/results/m8q/table.py OUT_DIR"""
import glob
import json
import os
import re
import statistics as st
import sys

O = sys.argv[1]
J = f"{O}/m8q"
ROWS = [("q6", "Qwen3.6 (TRITON_ATTN)"), ("g", "Gemma"), ("q8tp1", "Qwen3.8 TP1"),
        ("q8tp2", "Qwen3.8 TP2")]


def runs(p, k):
    return {re.search(r"_r(\d)\.json", f).group(1): json.load(open(f))
            for f in sorted(glob.glob(f"{J}/{p}_{k}_r*.json"))}


def load(tag):
    f = f"{O}/{tag}.log"
    if not os.path.exists(f):
        return "?"
    s = open(f).read()
    la = re.search(r"load average: ([\d.]+)", s)
    other = sum(int(g) for g in re.findall(r"gres/gpu(?::[\w-]+)?:(\d+)", s.split("load average")[0]))
    return f"{la.group(1) if la else '?'}/{other}"


print("TPOT ms, median of rounds [min, max]; delta = new / base - 1 (and rt2 / base)")
for p, name in ROWS:
    R = {k: runs(p, k) for k in ("base", "new", "rt2")}
    if not R["base"] or not R["new"]:
        print(f"{name}: missing {({k: len(v) for k, v in R.items()})}")
        continue
    bs = sorted(next(iter(R["base"].values()))["tpot_ms"], key=int)
    print(f"\n{name}: rounds base {sorted(R['base'])} new {sorted(R['new'])}"
          + (f" rt2 {sorted(R['rt2'])}" if R["rt2"] else ""))
    for b in bs:
        cells = []
        med = {}
        for k in ("base", "new", "rt2"):
            v = [r["tpot_ms"][b] for r in R[k].values()]
            if not v:
                continue
            med[k] = st.median(v)
            cells.append(f"{k} {med[k]:.3f} [{min(v):.3f}, {max(v):.3f}]")
        d = "  ".join(f"{k}/base {100 * (med[k] / med['base'] - 1):+.2f}%" for k in med if k != "base")
        print(f"  B={b}: " + " | ".join(cells) + f"  {d}")
    ref = R["base"].get("1") or next(iter(R["base"].values()))
    ident = []
    for k in ("base", "new", "rt2"):
        for r, d in sorted(R[k].items()):
            same = sum(a == b for a, b in zip(ref["outputs"], d["outputs"]))
            ident.append(f"{k}_r{r} {same}/{len(ref['outputs'])} (load {load(f'{p}_{k}_r{r}')})")
    print("  outputs identical to base_r1 (load: 1-min load average / GPUs allocated on the "
          "node incl. our 2): " + ", ".join(ident))
