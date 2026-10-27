"""Render docs/throughput.png from the raw results in bench/results/ and bench/stock/raw/.

All numbers come from one MI300X (banff-cyxtera-s70-4): the megakernel suite run and the
same-GPU stock rerun. Per-user decode speed = 1000 / TPOT.
"""

import json
import os
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MK = os.path.join(ROOT, "bench", "results", "banff-cyxtera-s70-4_suite_41b0956")
STOCK = os.path.join(ROOT, "bench", "stock", "raw")
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
ENGINE = {"megakernel": "#2a78d6", "SGLang": "#eb6834", "vLLM": "#1baf7a"}  # categorical slots 1-3
BATCHES = (1, 2, 4, 8)


def megakernel():
    """batch -> median TPOT (ms) over the repetitions of bench/bench_decode.py."""
    out = {}
    for b in BATCHES:
        last = open(os.path.join(MK, f"bench_m{b}.txt")).read().strip().splitlines()[-1]
        out[b] = statistics.median(r["tpot_ms"] for r in json.loads(last)["runs"])
    return out


def vllm(d):
    """batch -> median TPOT of `vllm bench serve` (batch 1: median of the 3 runs)."""
    j = lambda n: json.load(open(os.path.join(STOCK, d, f"{n}.json")))["median_tpot_ms"]
    out = {b: j(f"c{b}") for b in BATCHES}
    out[1] = statistics.median([j("c1"), j("c1_rep2"), j("c1_rep3")])
    return out


def sglang(d):
    """batch -> median decode latency of `sglang.bench_one_batch` (batch 1: median of the 3 runs)."""
    rows = [json.loads(l) for f in ("sweep", "b1_rep2", "b1_rep3")
            for l in open(os.path.join(STOCK, d, f"{f}.jsonl")) if l.strip()]
    out = {b: 1000 * [r for r in rows if r["run_name"] == "sweep" and r["batch_size"] == b][-1]["median_decode_latency"]
           for b in BATCHES}
    out[1] = statistics.median(1000 * r["median_decode_latency"] for r in rows if r["batch_size"] == 1)
    return out


def rounded_hbar(ax, y, w, h, color):
    r = min(h / 2, w * 0.02)
    ax.add_patch(FancyBboxPatch((0, y - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={r}", lw=0, color=color,
                                mutation_aspect=1))
    ax.add_patch(plt.Rectangle((0, y - h / 2), min(w, 2 * r), h, lw=0, color=color))


def main():
    series = [  # (engine, label, line style, batch -> TPOT ms)
        ("megakernel", "megakernel MXFP4", "-", megakernel()),
        ("vLLM", "vLLM FP8 + AITER", "-", vllm("vllm_aiter_Qwen3.8-27B-FP8_samegpu_202610222307")),
        ("vLLM", "vLLM BF16", (0, (5, 3)), vllm("vllm_Qwen3.8-27B")),
        ("SGLang", "SGLang BF16", (0, (5, 3)), sglang("sglang_Qwen3.8-27B")),
    ]
    tps = [(e, lab, ls, {b: 1000 / t for b, t in d.items()}) for e, lab, ls, d in series]

    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 4.6), dpi=160, gridspec_kw=dict(width_ratios=[1, 1.3], wspace=0.28))
    fig.patch.set_facecolor(SURFACE)
    for ax in (a, b):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)

    # Left: batch 1
    ys = [0, 1.2, 2.2, 3.2]
    vmax = max(d[1] for *_, d in tps)
    for y, (e, lab, _, d) in zip(ys, tps):
        rounded_hbar(a, y, d[1], 0.5, ENGINE[e])
        a.text(d[1] + vmax * 0.015, y, f"{d[1]:.0f}", va="center", ha="left", color=INK, fontsize=10)
    a.text(vmax * 1.25, (ys[0] + ys[1]) / 2, f"{tps[0][3][1] / tps[1][3][1]:.2f}×", va="center", ha="right", color=INK2,
           fontsize=11, fontweight="bold")
    a.set_yticks(ys, [t[1] for t in tps], color=INK2, fontsize=10)
    a.set_ylim(ys[-1] + 0.6, -0.6)
    a.set_xlim(0, vmax * 1.27)
    a.set_xlabel("tokens / s per user", color=INK2)
    a.tick_params(axis="y", length=0)
    a.xaxis.grid(True, color=GRID, lw=0.8)
    a.set_axisbelow(True)
    a.set_title("Decode speed, batch 1, 512-token context", loc="left", color=INK, fontsize=12)

    # Right: per-user speed vs batch size, speedup over vLLM FP8 + AITER annotated on the megakernel line
    for e, lab, ls, d in tps:
        b.plot(BATCHES, [d[x] for x in BATCHES], color=ENGINE[e], lw=2, ls=ls, marker="o", ms=5,
               solid_capstyle="round")
    mk, ref = tps[0][3], tps[1][3]
    for x in BATCHES:
        b.annotate(f"{mk[x] / ref[x]:.2f}×", (x, mk[x]), xytext=(0, 9), textcoords="offset points", ha="center",
                   color=INK2, fontsize=9, fontweight="bold")
    ends = sorted([d[8], lab] for _, lab, _, d in tps)
    ys2, gap = [v for v, _ in ends], 150 * 0.05
    for _ in range(50):
        for i in range(1, len(ys2)):
            if ys2[i] - ys2[i - 1] < gap:
                mid = (ys2[i] + ys2[i - 1]) / 2
                ys2[i - 1], ys2[i] = mid - gap / 2, mid + gap / 2
    for (v, text), y in zip(ends, ys2):
        b.annotate(text, (8, v), xytext=(8 * 1.18, y), textcoords="data", va="center", color=INK2, fontsize=9,
                   arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=0, shrinkB=3)
                   if abs(y - v) > 0.5 else None)
    b.set_xscale("log", base=2)
    b.set_xticks(BATCHES, [str(x) for x in BATCHES])
    b.minorticks_off()
    b.set_xlim(0.85, 8 * 2.3)
    b.set_ylim(0, 150)
    b.yaxis.grid(True, color=GRID, lw=0.8)
    b.set_axisbelow(True)
    b.set_xlabel("batch size (concurrent sequences)", color=INK2)
    b.set_ylabel("tokens / s per user", color=INK2)
    b.set_title("Per-user speed vs. batch size", loc="left", color=INK, fontsize=12)
    handles = [plt.Line2D([], [], color=ENGINE[k], lw=2, label=k) for k in ENGINE]
    handles += [plt.Line2D([], [], color=MUTED, lw=2, label="MXFP4 / FP8"),
                plt.Line2D([], [], color=MUTED, lw=2, ls=(0, (5, 3)), label="BF16")]
    b.legend(handles=handles, loc="lower left", frameon=False, ncol=2, fontsize=9, labelcolor=INK2)

    fig.text(0.01, -0.02, "Qwen3.8-27B, one MI300X (gfx942, banff-cyxtera-s70-4), TP1, plain decoding. Megakernel: median of 3 "
             "runs of 64 timed decode steps. vLLM 0.29.1: `vllm bench serve` median TPOT, HIP graphs on. SGLang 0.5.10: "
             "`bench_one_batch` median decode step.", color=MUTED, fontsize=8.5)
    out = os.path.join(ROOT, "docs", "throughput.png")
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
