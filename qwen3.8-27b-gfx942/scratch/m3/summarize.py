"""Disposable: summarize milestone 3 logs (rank-0 logs of a run directory).
Usage: python scratch/m3/summarize.py RUN_DIR [RUN_DIR ...]"""
import glob
import os
import re
import sys

rows = {}
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "*.r0.log"))):
        name = os.path.basename(f)[: -len(".r0.log")]
        s = open(f).read()
        r = {"file": f, "pass": "PASS" in s.splitlines()[-1:] or s.rstrip().endswith("PASS")}
        m = re.search(r"kernel~exact:\s+worst cos ([\d.]+) top1 ([\d.]+)", s)
        if m:
            r["kx_cos"], r["kx_top"] = float(m[1]), float(m[2])
        m = re.search(r"emulate~exact:\s+worst cos ([\d.]+) top1 ([\d.]+)", s)
        if m:
            r["ex_cos"], r["ex_top"] = float(m[1]), float(m[2])
        m = re.search(r"kernel~emulate: worst cos ([\d.]+)", s)
        if m:
            r["ke_cos"] = float(m[1])
        m = re.search(r"ranks bitwise equal every step: (\w+)", s)
        if m:
            r["ranks"] = m[1]
        m = re.search(r"3 runs bitwise equal: (\w+)", s)
        if m:
            r["det"] = m[1]
        m = re.search(r"all: ppl kernel ([\d.]+) w4 ([\d.]+) bf16 ([\d.]+)", s)
        if m:
            r["ppl"] = (float(m[1]), float(m[2]), float(m[3]))
        m = re.search(r"all: KL\(bf16\|\|kernel\) ([\d.]+)", s)
        if m:
            r["kl"] = float(m[1])
        r["inv"] = re.findall(r"batch-invariant (\w+)", s)
        rows[name] = r

for name, r in rows.items():
    parts = [f"{name:22s}", "PASS" if r["pass"] else "FAIL"]
    if "kx_cos" in r:
        parts.append(
            f"1-cos(k,x) {1 - r['kx_cos']:.6f} top1 {r['kx_top']:.3f} | 1-cos(e,x) {1 - r['ex_cos']:.6f}"
            f" top1 {r['ex_top']:.3f} | cos(k,e) {r['ke_cos']:.6f}"
        )
    for k in ("ranks", "det"):
        if k in r:
            parts.append(f"{k} {r[k]}")
    if "ppl" in r:
        parts.append(f"ppl k/w4/bf16 {r['ppl']} KL {r.get('kl')} batch-inv {set(r['inv'])}")
    print(" ".join(parts))

# TP1 seed spread: largest ratio of (1 - worst cos(kernel, exact)) between TP1 seeds at one M
for M in (1, 8):
    v = [1 - rows[f"tp1_dec_m{M}_s{s}"]["kx_cos"] for s in range(3) if f"tp1_dec_m{M}_s{s}" in rows]
    if len(v) > 1:
        print(f"TP1 M={M}: 1 - worst cos over seeds {['%.6f' % x for x in v]}, max/min {max(v) / min(v):.3f}")
