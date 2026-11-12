"""Speedup of the FP8 gfx942 megakernels over stock vLLM on the same FP8 weights and MI300X.

Only like-for-like comparisons (same checkpoint on both sides). Qwen3.8 (MXFP4) is in the
report's supplementary section, not here.

Sources (time per output token, one MI300X banff-cyxtera-s70-4, 512-token prompts):
- Qwen3.6-35B-A3B FP8: qwen3.6-35b-a3b-gfx942/README.md "Results" (dot: vs vLLM FP8+AITER of
  2026-10-27; bar top: same session 2026-10-29, job 68207371).
- Gemma 4 26B-A4B FP8: gemma4-26b-a4b-gfx942/README.md "Results" (dot: vs vLLM FP8 of
  2026-10-28; bar top: same job 2026-10-29, 68207195).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = [1, 2, 4, 8]
SERIES = [
    # label, dot (best stock-vLLM day), bar top (same-session baseline), color
    ("Qwen3.6-35B-A3B (MoE)", [1.64, 1.46, 1.24, 1.13], [1.72, 1.54, 1.33, 1.25], "tab:blue"),
    ("Gemma 4 26B-A4B (MoE)", [1.18, 1.14, 1.09, 1.06], [1.64, 1.54, 1.49, 1.37], "tab:green"),
]

fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150)
for i, (label, low, high, color) in enumerate(SERIES):
    x = [b * (1 + 0.035 * (i - 0.5)) for b in B]  # small offset so points do not overlap
    ax.plot(x, low, "o", color=color, label=label, ms=7)
    ax.plot(x, low, ":", color=color, lw=1, alpha=0.5)
    ax.vlines(x, low, high, color=color, lw=2.5, alpha=0.35)
    ax.plot(x, high, "_", color=color, ms=10, mew=2, alpha=0.6)
    for xi, yi in zip(x, low):
        ax.annotate(f"{yi:.2f}x", (xi, yi), textcoords="offset points", xytext=(8, -4), fontsize=8,
                    color=color)
ax.axvspan(9.2, 17, color="0.88")
ax.text(12.4, 1.45, "above 8\nsequences:\nnot tested", ha="center",
        va="center", fontsize=9, color="0.25")
ax.axhline(1.0, color="0.4", lw=1, ls="--")
ax.set_xscale("log", base=2)
ax.set_xticks([1, 2, 4, 8, 16])
ax.set_xticklabels(["1", "2", "4", "8", "16"])
ax.set_xlim(0.85, 17)
ax.set_ylim(0.9, 1.9)
ax.set_xlabel("concurrent sequences (batch)")
ax.set_ylabel("decode speedup per user vs stock vLLM")
ax.set_title("FP8 megakernels vs stock vLLM, same FP8 weights, one MI300X", fontsize=11)
ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(0.62, 1.0), frameon=False)
fig.text(0.01, 0.01, "Dot (labelled): against the best stock-vLLM day. Bar: up to the same-session baseline. "
         "Measured at batch 1, 2, 4, 8 only.", fontsize=7.5, color="0.3")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(__file__.replace("plot_speedup.py", "speedup.png"))
