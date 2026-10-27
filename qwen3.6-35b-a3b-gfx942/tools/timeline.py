"""Render a per-CU timeline of one decode step from a dump of bench/profile_decode.py.

Each CTA stamps the time it arrives at every grid barrier and the time the barrier releases it
(thread 0, 100 MHz clock). Stage b of a CTA runs from its release from barrier b to its
arrival at barrier b + 1; the time between arrival and release is spent waiting for the slowest CTA.
Usage: python tools/timeline.py TIMELINE.json OUT.png [--zoom T0 T1]
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

COLORS = {  # stage kind -> (legend label, color)
    "inp": ("GDN in_proj", "#2a78d6"),
    "gdn": ("DeltaNet", "#eb6834"),
    "out": ("out_proj", "#1baf7a"),
    "qkv": ("attn qkv", "#7d4fc9"),
    "attn": ("attention", "#c23b5a"),
    "o": ("attn o_proj", "#1baf7a"),
    "router": ("router", "#6b6a64"),
    "e1": ("experts gate/up", "#eda100"),
    "e2": ("experts down", "#e87fa3"),
    "lm": ("LM head", "#0b7d32"),
}
WAIT = "#e1e0d9"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("png")
    ap.add_argument("--zoom", type=float, nargs=2)
    a = ap.parse_args()
    d = json.load(open(a.json))
    us, stages, cus = d["us"], d["stages"], d["cus"]
    lo, hi = a.zoom or (0, max(c[0] for c in us[-1]))
    polys = {k: [] for k in COLORS}
    waits = []
    rect = lambda x0, x1, y: [(x0, y + 0.1), (x1, y + 0.1), (x1, y + 0.9), (x0, y + 0.9)]
    for b, kind in enumerate(stages):
        for c in range(cus):
            rel, arr = us[b][c][1], us[b + 1][c][0]
            nxt = us[b + 1][c][1]
            if arr >= lo and rel <= hi:
                polys[kind].append(rect(rel, arr, c))
            if nxt >= lo and arr <= hi:
                waits.append(rect(arr, nxt, c))
    fig, ax = plt.subplots(figsize=(14, 6.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.add_collection(PolyCollection(waits, facecolors=WAIT, linewidths=0))
    for k, p in polys.items():
        if p:
            ax.add_collection(PolyCollection(p, facecolors=COLORS[k][1], linewidths=0))
    ax.set_xlim(lo, hi)
    ax.set_ylim(cus, 0)
    unit = "µs"
    ax.set_xlabel(f"time since step start ({unit})", color="#52514e")
    ax.set_ylabel("CU (one persistent CTA each)", color="#52514e")
    total = max(c[0] for c in us[-1])
    title = f"One decode step (FP8, batch {d['M']}, ctx={d['ctx']}): {total / 1000:.2f} ms across {cus} CUs"
    if a.zoom:
        title += f" — {lo:.0f}–{hi:.0f} µs"
    ax.set_title(title, loc="left", color="#0b0b0b", fontsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors="#898781")
    shown = [k for k in COLORS if polys[k]]
    labels = dict.fromkeys(COLORS[k][0] if k != "o" else "out_proj / o_proj" for k in shown)
    labels.pop("out_proj", None) if "o" in shown else None
    col = {COLORS[k][0] if k != "o" else "out_proj / o_proj": COLORS[k][1] for k in shown}
    handles = [Patch(color=col[l], label=l) for l in labels]
    handles.append(Patch(color=WAIT, label="waiting at the grid barrier"))
    ax.legend(handles=handles, ncol=9, loc="upper left", bbox_to_anchor=(0, -0.09), frameon=False,
              labelcolor="#52514e", fontsize=9)
    fig.tight_layout()
    fig.savefig(a.png, facecolor=fig.get_facecolor())
    print("wrote", a.png)


if __name__ == "__main__":
    main()
