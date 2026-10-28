"""Render docs/throughput_tp2.png: TP2 decode speed against TP1 and stock vLLM TP2.

All numbers from one MI300X node (banff-cyxtera-s70-4, 2026-10-28), engine level (no
HTTP server): the kernels from bench/bench_decode.py (median of 3 runs of 64 steps), vLLM
from bench/stock/vllm_latency_bench.sh (FP8: median of 3 runs; BF16: 1 run).
Logs: scratch/logs/final/ (kernels), scratch/logs/fp8x3/ and scratch/logs/fp8/ (vLLM).
Per-user decode speed = 1000 / TPOT. docs/DESIGN-TP2.md section 11 has the tables.
"""

import json
import os
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "scratch", "logs")
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BATCHES = (1, 2, 4, 8)


def kernel(v):
    """batch -> median fixed-token TPOT (ms) of the 3 runs of the final run (rank 0)."""
    out = {}
    for b in BATCHES:
        s = open(os.path.join(LOGS, "final", f"{v}_bench_m{b}.r0.log")).read()
        out[b] = statistics.median(float(x) for x in re.findall(r"fixed-token TPOT ([\d.]+)", s))
    return out


def vllm(dirs):
    """batch -> median engine-level TPOT (ms) over the runs in dirs."""
    t = lambda d, b: json.load(open(os.path.join(LOGS, d, f"b{b}_tpot.json")))["tpot_ms"]  # noqa: E731
    return {b: statistics.median(t(d, b) for d in dirs) for b in BATCHES}


def main():
    series = [  # (label, color, line style, batch -> TPOT ms)
        ("megakernel TP2 (2 GPUs)", "#2a78d6", "-", kernel("tp2")),
        ("megakernel TP1 (1 GPU)", "#2a78d6", (0, (2, 2)), kernel("tp1")),
        ("vLLM TP2 FP8 + AITER", "#1baf7a", "-",
         vllm([f"fp8x3/vllm_lat_tp2_fp8_r{r}" for r in (1, 2, 3)])),
        ("vLLM TP2 BF16 + AITER", "#1baf7a", (0, (5, 3)), vllm(["fp8/vllm_lat_tp2_Qwen3.8-27B"])),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    for lab, col, ls, d in series:
        ys = [1000 / d[b] for b in BATCHES]
        ax.plot(BATCHES, ys, color=col, lw=2, ls=ls, marker="o", ms=5, label=lab)
    mk, ref = series[0][3], series[2][3]
    for b in BATCHES:
        ax.annotate(f"{ref[b] / mk[b]:.2f}×", (b, 1000 / mk[b]), xytext=(0, 9), textcoords="offset points",
                    ha="center", color=INK2, fontsize=9, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks(BATCHES, [str(x) for x in BATCHES])
    ax.minorticks_off()
    ax.set_ylim(0, 210)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("batch size (concurrent sequences)", color=INK2)
    ax.set_ylabel("tokens / s per user", color=INK2)
    ax.set_title("Qwen3.8-27B decode, tensor parallel 2 on MI300X (XGMI)", loc="left", color=INK, fontsize=12)
    ax.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK2)
    fig.text(0.01, -0.03, "banff-cyxtera-s70-4, 2026-10-28, ctx 512. Engine level, no HTTP server: megakernel "
             "bench_decode.py (median of 3 x 64 steps); vLLM nightly 0.32.1rc1 `bench latency` (65 - 1 output "
             "tokens) / 64, includes its scheduler and sampler. Labels: vLLM TP2 FP8 TPOT / megakernel TP2 TPOT.",
             color=MUTED, fontsize=7.5, wrap=True)
    out = os.path.join(ROOT, "docs", "throughput_tp2.png")
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", out)
    for lab, _, _, d in series:
        print(f"{lab:26s}", " ".join(f"{d[b]:.3f}" for b in BATCHES))


if __name__ == "__main__":
    main()
