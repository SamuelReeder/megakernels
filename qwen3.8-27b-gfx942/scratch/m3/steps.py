"""Disposable: per-step (1 - cos) kernel~exact and emulate~exact from test_decode logs:
median and mean over the 16 steps, for TP1 and TP2 at the same M and seed."""
import re, statistics as st, sys, os
d = sys.argv[1]
def steps(name):
    f = os.path.join(d, name + ".r0.log")
    if not os.path.exists(f): return None
    kx, ex, ke = [], [], []
    for m in re.finditer(r"kernel~emulate cos ([\d.]+) .*?kernel~exact cos ([\d.]+) .*?emulate~exact cos ([\d.]+)", open(f).read()):
        ke.append(1 - float(m[1])); kx.append(1 - float(m[2])); ex.append(1 - float(m[3]))
    return kx, ex, ke
print(f"{'run':22s} {'kx med':>9s} {'kx mean':>9s} {'ex med':>9s} {'ex mean':>9s} {'ke med':>9s}")
agg = {}
for M, seeds in ((1, (0, 1, 2)), (2, (0,)), (4, (0,)), (8, (0, 1, 2))):
    for s in seeds:
        for v in ("tp1", "tp2", "tp2fp32", "tp2bf16"):
            r = steps(f"{v}_dec_m{M}_s{s}")
            if r is None: continue
            kx, ex, ke = r
            agg.setdefault(v, []).extend(kx)
            print(f"{v + '_m%d_s%d' % (M, s):22s} {st.median(kx):9.6f} {st.mean(kx):9.6f} {st.median(ex):9.6f} {st.mean(ex):9.6f} {st.median(ke):9.6f}")
for v, xs in agg.items():
    print(f"all steps {v:8s}: n {len(xs)} median {st.median(xs):.6f} mean {st.mean(xs):.6f}")
