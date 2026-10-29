"""Split a profiled decode step (bench/profile_decode.py TIMELINE.json) into parts.

Barrier b releases the CTAs that start stage b; they arrive at barrier b + 1. Each
stage interval (median release of barrier b -> median release of barrier b + 1; for the
last stage, -> its last arrival) splits exactly into:
  work      median arrival - median release (a typical CTA: prologue, stream, hand-off)
  skew      last arrival - median arrival (the late CTAs)
  barrier   median release of b + 1 - last arrival (the barrier protocol itself)
The GEMV stage work is then fitted against the stage's weight bytes:
work = a + bytes / BW (least squares over the stage kinds, medians over layers).
Several timelines (steps of one run: profile_decode.py PSTEPS) are pooled: medians over
layers and steps, the "sum" column and the step parts per step (mean).
Usage: python tools/stage_breakdown.py TIMELINE.json [TIMELINE_s0.json ...]
"""
import json
import os
import statistics as st
import sys

H, FI, VOCAB = 2048, 512, 248320
# weight bytes per stage at M rows (fp8 tiles + 8 B of scales per 2 KiB unit; bf16 2 B/elt)
FP8 = lambda n, k: n * k + (n // 16) * (k // 128) * 8  # noqa: E731
BYTES = {
    "inp": FP8(12288, H) + 64 * H * 2,
    "qkv": FP8(9216, H),
    "out": FP8(H, 4096),
    "o": FP8(H, 4096),
    "router": 272 * H * 2,
    "lm": VOCAB * H * 2,
}
MOE_M1 = {"e1": 9 * FP8(2 * FI, H), "e2": 9 * FP8(H, FI)}  # U = 8 + shared at M = 1


def breakdown(paths):
    rows = {}
    tot = dict(work=0.0, skew=0.0, barrier=0.0)
    steps = []
    for path in paths:
        d = json.load(open(path))
        us, names, M = d["us"], d["stages"], d["M"]
        for b, nm in enumerate(names):
            rel = [c[1] for c in us[b]]
            arr = [c[0] for c in us[b + 1]]
            nxt = st.median(c[1] for c in us[b + 1]) if b + 2 < len(us) else max(arr)
            work = st.median(arr) - st.median(rel)
            skew = max(arr) - st.median(arr)
            bar = nxt - max(arr)
            rows.setdefault(nm, []).append((work, skew, bar))
            tot["work"] += work / len(paths)
            tot["skew"] += skew / len(paths)
            tot["barrier"] += bar / len(paths)
        steps.append(max(c[0] for c in us[-1]))
    print(f"== {paths[0]} ({len(paths)} steps): M={M}, step median {st.median(steps):.1f} us "
          f"(first barrier arrival -> last arrival at the end)")
    print(f"{'stage':7}{'n':>4}{'work':>8}{'skew':>8}{'barrier':>9}{'interval':>10}{'sum us':>9}")
    for nm, v in rows.items():
        w = st.median(r[0] for r in v)
        s = st.median(r[1] for r in v)
        b = st.median(r[2] for r in v)
        print(f"{nm:7}{len(v) // len(paths):4d}{w:8.2f}{s:8.2f}{b:9.2f}{w + s + b:10.2f}"
              f"{sum(sum(r) for r in v) / len(paths):9.1f}")
    T = sum(tot.values())
    print("step parts: " + ", ".join(f"{k} {v:.0f} us ({100 * v / T:.1f} %)" for k, v in tot.items()))
    fit = dict(BYTES)
    if M == 1:
        fit.update(MOE_M1)
    ks = [k for k in fit if k in rows and k != "lm"]
    xs = [fit[k] / 1e6 for k in ks]  # MB; slope in us / MB, so 1 / slope is in TB/s
    out = {}
    for col, f in (("work", lambda r: r[0]), ("interval", sum)):
        ys = [st.median(f(r) for r in rows[k]) for k in ks]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        a = my - slope * mx
        res = [y - (a + slope * x) for x, y in zip(xs, ys)]
        nst = sum(len(rows[k]) for k in ks) // len(paths)
        stream = sum(len(rows[k]) * x * slope for k, x in zip(ks, xs)) / len(paths)
        print(f"fit {col:8} = {a:5.2f} us + bytes / {1 / slope:.2f} TB/s over {', '.join(ks)} "
              f"(max |residual| {max(map(abs, res)):.2f} us); {nst} stages: bytes part "
              f"{stream:.0f} us, fixed part {nst * a:.0f} us")
        out[col] = (a, 1 / slope)
    # step split: GEMV weight bytes (LM head included) at the LM head's stream rate (a
    # stage that is all stream; STREAM_TBS=<TB/s> fixes it, to compare runs), the rest of
    # the GEMV work (prologue, hand-off), late CTAs, mixer tasks, barriers
    per = lambda k, i: sum(r[i] for r in rows[k]) / len(paths)  # noqa: E731
    bw = BYTES["lm"] / 1e6 / st.median(r[0] for r in rows["lm"])  # MB / us = TB/s
    bw = float(os.environ.get("STREAM_TBS", bw))
    cat = dict(stream=0.0, gemv_fixed=0.0, gemv_skew=0.0, mixer=0.0, moe_unknown=0.0,
               barrier=tot["barrier"])
    for k in rows:
        if k in fit:
            s = len(rows[k]) / len(paths) * fit[k] / 1e6 / bw
            cat["stream"] += s
            cat["gemv_fixed"] += per(k, 0) - s
            cat["gemv_skew"] += per(k, 1)
        elif k in ("gdn", "attn"):
            cat["mixer"] += per(k, 0) + per(k, 1)
        else:
            cat["moe_unknown"] += per(k, 0) + per(k, 1)
    T = sum(cat.values())
    print(f"step split (GEMV bytes at {bw:.2f} TB/s): " + ", ".join(
        f"{k} {v:.0f} us ({100 * v / T:.1f} %)" for k, v in cat.items() if v))
    return rows, tot, out


if __name__ == "__main__":
    breakdown(sys.argv[1:])
