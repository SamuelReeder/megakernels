"""Disposable: the three-part split of the added cost per all-reduce (DESIGN-TP2 section
11) from arsplit_an.py's quantities, weighted by the layer count of each stage kind.
  push/poll    = xfer + the own push/poll share of the GEMV-segment growth
  slower GPU   = wait + the peer-late share of the GEMV-segment growth
  drain        = drain segment growth (remaining stores + s_waitcnt vmcnt(0) at the barrier)
  barrier      = barrier segment growth
The GEMV-segment growth (epilogues block their wave inside the stream) is split by the
per-epilogue ratio of peer-late wait to own push/poll.
Usage: python scratch/m3/arsplit_parts.py DIR [FORMAT ...]"""
import statistics as st
import sys

import torch  # noqa: F401  (used by the exec'd arsplit_an.py code)

src = open(__file__.replace("arsplit_parts.py", "arsplit_an.py")).read()
exec(src[src.index("AR_STAGES ="):src.index("for fmt in (")].replace("D, M = sys.argv[1], int(sys.argv[2])", ""))
D = sys.argv[1]
for M in (1, 8):
    for fmt in sys.argv[2:] or ("fp32", "bf16"):
        globals()["M"] = M
        a, _, sa, _, grp = run(fmt, False)
        l, _, sl, _, _ = run(fmt, True)
        parts = dict(pushpoll=0.0, slower=0.0, drain=0.0, barrier=0.0)
        for s in AR_STAGES:
            n = n_layers(s)
            dm = {k: st.mean(a[s][k]) - st.mean(l[s][k]) for k in SEGS}
            late = st.mean(x[0] for x in grp[s])
            own = st.mean(x[1] for x in grp[s])
            fl = late / (late + own)
            parts["pushpoll"] += n * (dm["xfer"] + (1 - fl) * dm["gemv"])
            parts["slower"] += n * (dm["wait"] + fl * dm["gemv"])
            parts["drain"] += n * dm["drain"]
            parts["barrier"] += n * dm["bar"]
        tot = sum(parts.values())
        step = (st.median(sum(sa, [])) - st.median(sum(sl, []))) / 126
        print(f"{fmt} M={M}: per all-reduce (stamped) {tot / 126:.2f} us (whole-step difference {step:.2f}): "
              + ", ".join(f"{k} {v / 126:+.2f} us ({100 * v / tot:.0f} %)" for k, v in parts.items()))
