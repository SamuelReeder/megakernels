"""Disposable: per-stage comparison of profile timelines (bench/profile_decode.py json):
TP1, TP2 with the all-reduce (l0) and with ar_local (l1), per format, rank 0.
Median over layers of the stage span (first release -> last arrival) and of the barrier
(last arrival -> last release). For a GEMV kind also the span the section 10 fit predicts
for TP2 (a + bytes / 2 / B, the M = 1 / M = 8 fits of the TP1 timelines).
Usage: python stages.py RUN_DIR M"""
import json
import os
import statistics as st
import sys

d, M = sys.argv[1], int(sys.argv[2])
B = 0.53125
BYTES = {"inp": 16480 * 5120 * B, "out": 5120 * 6144 * B, "gu": 2 * 17408 * 5120 * B,
         "down": 17408 * 5120 * B, "qkv": 14336 * 5120 * B, "o": 5120 * 6144 * B}
FIT = {1: (6.52, 3.66), 8: (11.93, 3.41)}  # us, TB/s (scratch/logs/stage_fit.txt)


def load(f):
    if not os.path.exists(f):
        return None
    t = json.load(open(f))
    us, names = t["us"], t["stages"]
    rows = {}
    for b in range(len(us) - 1):
        span = max(c[0] for c in us[b + 1]) - min(c[1] for c in us[b])
        bar = max(c[1] for c in us[b + 1]) - max(c[0] for c in us[b + 1]) if b + 2 < len(us) else 0.0
        rows.setdefault(names[b], []).append((span, bar))
    total = max(c[0] for c in us[-1]) - min(c[1] for c in us[0])
    return {k: (st.median(x for x, _ in v), st.median(y for _, y in v), len(v)) for k, v in rows.items()}, total


runs = {"TP1": f"tl_tp1_m{M}.json"}
for ar in ("fp32", "bf16"):
    for loc in (0, 1):
        runs[f"{ar}{' local' if loc else ''}"] = f"tl_tp2{ar}l{loc}_m{M}.json.r0"
data = {k: load(os.path.join(d, f)) for k, f in runs.items()}
data = {k: v for k, v in data.items() if v}
kinds = ["inp", "gdn", "out", "qkv", "attn", "o", "gu", "down", "lm"]
print(f"M = {M}: median span / barrier per stage, us (rank 0)")
print(f"{'stage':6s} " + " ".join(f"{k:>16s}" for k in data) + f" {'fit TP2':>8s}")
for kd in kinds:
    cells = []
    for k, (rows, _) in data.items():
        s, b, n = rows.get(kd, (0, 0, 0))
        cells.append(f"{s:8.2f}/{b:6.2f} ")
    fit = ""
    if kd in BYTES:
        a, bw = FIT[M]
        fit = f"{a + BYTES[kd] / 2 / (bw * 1e6):8.2f}"
    print(f"{kd:6s} " + " ".join(f"{c:>16s}" for c in cells) + f" {fit}")
print("total stamped step, ms: " + ", ".join(f"{k} {v[1] / 1e3:.3f}" for k, v in data.items()))
for ar in ("fp32", "bf16"):
    if ar in data and f"{ar} local" in data:
        r, l = data[ar][0], data[f"{ar} local"][0]
        ex = {kd: r[kd][0] - l[kd][0] for kd in ("out", "o", "down") if kd in r}
        print(f"{ar}: all-reduce stages, span with minus span without the all-reduce: "
              + ", ".join(f"{k} {v:+.2f} us" for k, v in ex.items())
              + f" | whole step {data[ar][1] - data[ar + ' local'][1]:+.1f} us")
