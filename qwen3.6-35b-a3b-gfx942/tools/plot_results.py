"""Render docs/throughput.png from raw results: per-user and total decode speed.

Usage: python tools/plot_results.py MK_DIR [OUT.png]
MK_DIR holds bench_m{1,2,4,8}.txt of bench/bench_decode.py (a suite run). The stock
baselines come from bench/results/stock/ (same GPU model; see docs/DESIGN.md, "Stock
baselines"). Per-user speed = 1000 / TPOT; total speed = batch x per-user speed.
"""

import json
import os
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOCK = os.path.join(ROOT, "bench", "results", "stock")
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
COLOR = {"megakernel": "#2a78d6", "vLLM": "#1baf7a", "SGLang": "#eb6834"}
BATCHES = (1, 2, 4, 8)


def megakernel(d):
    """batch -> median TPOT (ms) over the repetitions of bench/bench_decode.py."""
    out = {}
    for b in BATCHES:
        last = open(os.path.join(d, f"bench_m{b}.txt")).read().strip().splitlines()[-1]
        out[b] = statistics.median(r["tpot_ms"] for r in json.loads(last)["runs"])
    return out


def vllm(name):
    """batch -> median TPOT of `vllm bench serve` (batch 1: median of 3 runs)."""
    j = lambda n: json.load(open(os.path.join(STOCK, name, f"{n}.json")))["median_tpot_ms"]  # noqa: E731
    out = {b: j(f"c{b}") for b in BATCHES}
    out[1] = statistics.median([j("c1"), j("c1_rep2"), j("c1_rep3")])
    return out


def sglang(name):
    """batch -> median decode step of `sglang.bench_one_batch` (batch 1: median of 3)."""
    rows = [json.loads(line) for f in ("sweep", "b1_rep2", "b1_rep3")
            for line in open(os.path.join(STOCK, name, f"{f}.jsonl")) if line.strip()]
    out = {b: 1000 * [r for r in rows if r["run_name"] == "sweep" and r["batch_size"] == b][-1]
           ["median_decode_latency"] for b in BATCHES}
    out[1] = statistics.median(1000 * r["median_decode_latency"] for r in rows
                               if r["batch_size"] == 1)
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(BATCHES, [str(x) for x in BATCHES])
    ax.minorticks_off()
    ax.set_xlim(0.85, 9.5)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("batch size (concurrent sequences)", color=INK2)


def main():
    mk_dir = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "docs", "throughput.png")
    series = [  # (engine, label, line style, batch -> TPOT ms)
        ("megakernel", "megakernel FP8", "-", megakernel(mk_dir)),
        ("vLLM", "vLLM FP8 + AITER", "-", vllm("vllm_aiter_Qwen3.6-35B-A3B-FP8")),
        ("vLLM", "vLLM FP8", (0, (5, 3)), vllm("vllm_Qwen3.6-35B-A3B-FP8")),
        ("SGLang", "SGLang FP8", "-", sglang("sglang_Qwen3.6-35B-A3B-FP8")),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=160, gridspec_kw=dict(wspace=0.25))
    fig.patch.set_facecolor(SURFACE)
    ref, mk = series[1][3], series[0][3]
    for k, ax in enumerate(axes):
        style(ax)
        f = lambda d, x: (x if k else 1) * 1000 / d[x]  # noqa: E731
        for e, lab, ls, d in series:
            ax.plot(BATCHES, [f(d, x) for x in BATCHES], color=COLOR[e], lw=2, ls=ls,
                    marker="o", ms=5, label=lab)
        for x in BATCHES:
            ax.annotate(f"{ref[x] / mk[x]:.2f}x", (x, f(mk, x)), xytext=(0, 9),
                        textcoords="offset points", ha="center", color=INK2, fontsize=9,
                        fontweight="bold")
        ax.set_ylim(0, 1.15 * max(f(d, x) for *_, d in series for x in BATCHES))
    axes[0].set_ylabel("tokens / s per user", color=INK2)
    axes[0].set_title("Per-user decode speed", loc="left", color=INK, fontsize=12)
    axes[1].set_ylabel("tokens / s, all users", color=INK2)
    axes[1].set_title("Total decode throughput", loc="left", color=INK, fontsize=12)
    axes[0].legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK2)
    fig.text(0.01, -0.08, "Qwen3.6-35B-A3B FP8 checkpoint, one MI300X (gfx942), TP1, 512-token "
             "prompt, 256 output tokens. Megakernel: bench/bench_decode.py, median of 3 runs.\n"
             "vLLM 0.32.1rc1 nightly: `vllm bench serve` median TPOT. SGLang 0.5.10: "
             "`bench_one_batch`. Labels: speedup over vLLM FP8 + AITER.", color=MUTED,
             fontsize=8.2)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
