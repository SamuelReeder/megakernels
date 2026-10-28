"""Disposable: hard gates and speed of a final.sh run directory (DESIGN-TP2 section 9 gates).
Usage: python scratch/m3/final_summary.py RUN_DIR"""
import glob
import os
import re
import statistics as st
import sys

d = sys.argv[1]


def log(name):
    f = os.path.join(d, name + ".r0.log")
    return open(f).read() if os.path.exists(f) else None


def kx(name):
    s = log(name)
    return [1 - float(m[1]) for m in re.finditer(r"kernel~exact cos ([\d.]+)", s)] if s else []


runs = [(1, s) for s in (0, 1, 2)] + [(2, 0), (4, 0)] + [(8, s) for s in (0, 1, 2)]
e1 = [x for M, s in runs for x in kx(f"tp1_dec_m{M}_s{s}")]
e2 = [x for M, s in runs for x in kx(f"tp2_dec_m{M}_s{s}")]
r = st.median(e2) / st.median(e1)
hdr = log("tp2_dec_m1_s0")
node = re.search(r"# (\S+)", hdr)[1]
day = re.search(r" (\d{4}-\d\d-\d\d)T", hdr)[1]
print(f"node {node}  date {day}")
print(f"G1 median per-step 1-cos to exact: TP2 {st.median(e2):.6f} TP1 {st.median(e1):.6f} "
      f"ratio {r:.3f} (<= 1.2) {'PASS' if r <= 1.2 else 'FAIL'} (n {len(e2)} / {len(e1)})")
k1 = float(re.search(r"all: KL\(bf16\|\|kernel\) ([\d.]+)", log("tp1_eval"))[1])
k2 = float(re.search(r"all: KL\(bf16\|\|kernel\) ([\d.]+)", log("tp2_eval"))[1])
print(f"G2 eval_text KL: TP2 {k2:.5f} TP1 {k1:.5f} ({100 * (k2 / k1 - 1):+.1f} %, <= +10 %) "
      f"{'PASS' if k2 <= 1.1 * k1 else 'FAIL'}")
ranks = all("ranks bitwise equal every step: True" in (log(n) or "") for n in
            [f"tp2_dec_m{M}_s{s}" for M, s in runs] + ["tp2_det_m1", "tp2_det_m8", "tp2_eval"])
print(f"G3 ranks bitwise equal in every TP2 run: {'PASS' if ranks else 'FAIL'}")
det = all("3 runs bitwise equal: True" in (log(f"tp2_det_m{m}") or "") for m in (1, 8))
sha = [re.search(r"logits sha256 (\w+)", log(f"tp2_det_m{m}") or "") for m in (1, 8)]
print(f"G4 determinism M = 1, 8: {'PASS' if det else 'FAIL'}; logits sha256 "
      f"{[s[1] if s else None for s in sha]}")
bi = re.findall(r"batch-invariant (\w+)", log("tp2_eval") or "")
print(f"G5 batch-invariant rows (eval_text): {'PASS' if bi and all(b == 'True' for b in bi) else 'FAIL'}")
nf = [int(m) for n in [f"{v}_dec_m{M}_s{s}" for v in ("tp1", "tp2") for M, s in runs]
      for m in re.findall(r"non-finite logits \(kernel or reference\): (\d+)", log(n) or "")]
print(f"non-finite steps in test_decode runs: {sum(nf)} ({len(nf)} runs report it)")
print("speed (bench_decode ctx 512, 64 steps, median of 3 fixed-token TPOT, ms):")
for M in (1, 2, 4, 8):
    t = {}
    for v in ("tp1", "tp2"):
        s = log(f"{v}_bench_m{M}")
        xs = sorted(float(x) for x in re.findall(r"fixed-token TPOT ([\d.]+)", s or ""))
        t[v] = xs[len(xs) // 2] if xs else float("nan")
    print(f"  M = {M}: TP1 {t['tp1']:.3f}  TP2 {t['tp2']:.3f}  TP2 faster by "
          f"{100 * (1 - t['tp2'] / t['tp1']):.1f} %")
print("report only (old test_decode criterion):",
      " ".join(f"{os.path.basename(f)[:-7]}={'PASS' if 'PASS' in open(f).read()[-200:] else 'FAIL'}"
               for f in sorted(glob.glob(os.path.join(d, "tp2_dec_*.r0.log")))))
