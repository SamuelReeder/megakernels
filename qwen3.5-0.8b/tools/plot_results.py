"""Render docs/throughput.png from results/{decode,llamacpp}_*.json (written by bench/run_all.sh).

Every engine is compared on the same footing: best of 3 timed repetitions of wall-clock
generation, so megakernel numbers include launch gaps and host round trips.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
ENGINE = {"megakernel": "#2a78d6", "llama.cpp": "#eb6834", "vLLM": "#1baf7a"}  # categorical slots 1-3


def load():
    r = lambda f: json.load(open(os.path.join(ROOT, "results", f)))
    series = {}
    for q in ("int8", "bf16"):
        rows = r(f"decode_{q}.json")["rows"]
        series[("megakernel", q)] = {row["ctx"] if row["ctx"] > 1 else 0: row["wall_tok_s"] for row in rows}
    for m, q in (("Q8_0", "int8"), ("BF16", "bf16")):
        series[("llama.cpp", q)] = {x["n_depth"]: max(x["samples_ts"]) for x in r(f"llamacpp_{m}.json")}
    if os.path.exists(os.path.join(ROOT, "results", "vllm_bf16.json")):
        series[("vLLM", "bf16")] = {x["ctx"] if x["ctx"] > 1 else 0: x["tok_s"]
                                    for x in r("vllm_bf16.json")["rows"]}
    return series


def rounded_hbar(ax, y, w, h, color):
    """Bar from the baseline with a 4px-ish rounded data end, square at the baseline."""
    r = min(h / 2, w * 0.02)
    ax.add_patch(FancyBboxPatch((0, y - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={r}", lw=0, color=color,
                                mutation_aspect=1))
    ax.add_patch(plt.Rectangle((0, y - h / 2), min(w, 2 * r), h, lw=0, color=color))


def main():
    s = load()
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6), dpi=160, gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.28))
    fig.patch.set_facecolor(SURFACE)
    for ax in (a, b):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)

    # Left: bars at a 512-token context
    rows = [("megakernel", "int8", "megakernel int8"), ("llama.cpp", "int8", "llama.cpp Q8_0"),
            ("megakernel", "bf16", "megakernel bf16"), ("llama.cpp", "bf16", "llama.cpp BF16")]
    if ("vLLM", "bf16") in s:
        rows.append(("vLLM", "bf16", "vLLM bf16"))
    ys = [0, 1, 2.6, 3.6, 4.6][:len(rows)]
    vmax = max(s[(e, q)][512] for e, q, _ in rows)
    for y, (e, q, label) in zip(ys, rows):
        v = s[(e, q)][512]
        rounded_hbar(a, y, v, 0.5, ENGINE[e])
        a.text(v + vmax * 0.015, y, f"{v:.0f}", va="center", ha="left", color=INK, fontsize=10)
    # speedup label of the megakernel bar over another engine's bar: (megakernel row, other row, label y)
    pairs = [(0, 1, ys[0] + 0.5), (2, 3, ys[2] + 0.5)] + ([(2, 4, ys[4])] if len(rows) > 4 else [])
    for i, j, y in pairs:
        sp = s[rows[i][:2]][512] / s[rows[j][:2]][512]
        a.text(vmax * 1.2, y, f"{sp:.2f}×", va="center", ha="right", color=INK2, fontsize=11, fontweight="bold")
    a.set_yticks(ys, [r[2] for r in rows], color=INK2, fontsize=10)
    a.set_ylim(ys[-1] + 0.6, -0.6)
    a.set_xlim(0, vmax * 1.22)
    a.set_xlabel("tokens / s", color=INK2)
    a.tick_params(axis="y", length=0)
    a.xaxis.grid(True, color=GRID, lw=0.8)
    a.set_axisbelow(True)
    a.set_title("Decode throughput, 512-token context", loc="left", color=INK, fontsize=12)

    # Right: throughput vs context length
    for (e, q), d in s.items():
        xs = sorted(d)
        b.plot(xs, [d[x] for x in xs], color=ENGINE[e], lw=2, ls="-" if q == "int8" else (0, (5, 3)),
               marker="o", ms=5, solid_capstyle="round")
        label = {"int8": "int8" if e == "megakernel" else "Q8_0", "bf16": "BF16" if e == "llama.cpp" else "bf16"}[q]
        # these two end close together: pull their labels apart with leader lines
        dy = {("megakernel", "bf16"): 18, ("llama.cpp", "int8"): -2, ("vLLM", "bf16"): 4,
              ("llama.cpp", "bf16"): -10}.get((e, q), 0)
        b.annotate(f"{e} {label}", (xs[-1], d[xs[-1]]), xytext=(14, dy), textcoords="offset points", va="center",
                   color=INK2, fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8) if dy else None)
    b.set_xticks([0, 512, 2048, 4096], ["1", "512", "2048", "4096"])
    b.set_xlim(-150, 5400)
    b.set_ylim(0, 1000)
    b.yaxis.grid(True, color=GRID, lw=0.8)
    b.set_axisbelow(True)
    b.set_xlabel("context length (tokens already in the cache)", color=INK2)
    b.set_ylabel("tokens / s", color=INK2)
    b.set_title("Throughput vs. context length", loc="left", color=INK, fontsize=12)
    handles = [plt.Line2D([], [], color=ENGINE[k], lw=2, label=k) for k in ENGINE if any(e == k for e, _ in s)]
    handles += [plt.Line2D([], [], color=MUTED, lw=2, label="int8 / Q8_0"),
                plt.Line2D([], [], color=MUTED, lw=2, ls=(0, (5, 3)), label="bf16")]
    b.legend(handles=handles, loc="lower left", frameon=False, ncol=2, fontsize=9, labelcolor=INK2)

    fig.text(0.01, -0.02, "Qwen3.5-0.8B, batch 1, RTX 5080. Best of 3 runs of wall-clock generation for every engine; "
             f"llama.cpp build {json.load(open(os.path.join(ROOT, 'results', 'llamacpp_BF16.json')))[0]['build_commit']}"
             ", flash attention on, all layers on GPU.", color=MUTED, fontsize=8.5)
    out = os.path.join(ROOT, "docs", "throughput.png")
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
