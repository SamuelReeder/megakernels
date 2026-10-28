"""Disposable: the section 9 checks per (M, seed) for TP2 fp32 and bf16 against TP1:
  c1: the test_decode criterion (PASS line);
  c2: 1 - worst cos(TP2, exact) <= F (1 - worst cos(TP1, exact)) + 1e-5 and
      top-1(TP2, exact) >= top-1(TP1, exact) - 0.05,
F = the largest TP1-to-TP1 ratio of 1 - worst cos over seeds (M = 1 and 8, seeds 0..2),
or 1.5 if that is smaller. Usage: python bound.py LOG_DIR..."""
import glob
import os
import re
import sys

R = {}
for d in sys.argv[1:]:
    for f in glob.glob(os.path.join(d, "*_dec_m*_s*.r0.log")):
        n = os.path.basename(f)[: -len(".r0.log")]
        v, m, s = re.match(r"(\w+?)_dec_m(\d+)_s(\d+)", n).groups()
        t = open(f).read()
        kx = re.search(r"kernel~exact:\s+worst cos ([\d.]+) top1 ([\d.]+)", t)
        if not kx:
            continue
        R[(v, int(m), int(s))] = (1 - float(kx[1]), float(kx[2]), t.rstrip().endswith("PASS"))

ratios = []
for M in (1, 8):
    v = [R[("tp1", M, s)][0] for s in range(3) if ("tp1", M, s) in R]
    ratios.append(max(v) / min(v))
F = max(1.5, max(ratios))
print(f"TP1 seed ratios (M=1, M=8, seeds 0..2): {[round(r, 3) for r in ratios]} -> factor F = {F:.3f}")
for v in ("tp1", "tp2fp32", "tp2bf16"):
    n1 = n2 = tot = 0
    for (vv, M, s), (e, top, ok1) in sorted(R.items()):
        if vv != v or ("tp1", M, s) not in R:
            continue
        e1, t1, _ = R[("tp1", M, s)]
        ok2 = e <= F * e1 + 1e-5 and top >= t1 - 0.05
        tot += 1
        n1 += ok1
        n2 += ok2
        if v != "tp1":
            print(f"  {v} M={M} s={s}: 1-cos {e:.6f} (TP1 {e1:.6f}, ratio {e / e1:6.2f}) top1 {top:.3f}"
                  f" (TP1 {t1:.3f}) | c1 {'PASS' if ok1 else 'FAIL'} c2 {'PASS' if ok2 else 'FAIL'}")
    print(f"{v}: test_decode criterion (c1) {n1}/{tot}, TP1 bound (c2) {n2}/{tot}")
