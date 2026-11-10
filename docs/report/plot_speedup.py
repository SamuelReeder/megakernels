"""Speedup of each gfx942 megakernel over stock vLLM on the same MI300X, per batch size.

Sources (time per output token, one MI300X banff-cyxtera-s70-4, ctx 512):
- Qwen3.6-35B-A3B FP8: qwen3.6-35b-a3b-gfx942/README.md "Results" (low: vs vLLM+AITER of
  2026-10-27; high: same session 2026-10-29, job 68207371).
- Gemma 4 26B-A4B FP8: gemma4-26b-a4b-gfx942/README.md "Results" (low: vs vLLM of 2026-10-28;
  high: same job 2026-10-29, 68207195).
- Qwen3.8-27B MXFP4 TP1: qwen3.8-27b-gfx942/README.md "Results" (vs vLLM 0.29.1 FP8+AITER,
  2026-10-22).
- Qwen3.8-27B MXFP4 TP2: qwen3.8-27b-gfx942/README.md "Tensor parallel 2" (vs vLLM TP2
  FP8+AITER, 2026-10-28).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = [1, 2, 4, 8]
SERIES = [
    # label, low (conservative), high (same session) or None, color, note
    ("Qwen3.6-35B-A3B, FP8", [1.64, 1.46, 1.24, 1.13], [1.72, 1.54, 1.33, 1.25], "tab:blue"),
    ("Gemma 4 26B-A4B, FP8", [1.18, 1.14, 1.09, 1.06], [1.64, 1.54, 1.49, 1.37], "tab:green"),
    ("Qwen3.8-27B, MXFP4 vs FP8 *", [2.06, 2.07, 1.98, 1.68], None, "tab:red"),
    ("Qwen3.8-27B TP2, MXFP4 vs FP8 *", [1.76, 1.79, 1.58, 1.41], None, "tab:orange"),
]

fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
for i, (label, low, high, color) in enumerate(SERIES):
    x = [b * (1 + 0.04 * (i - 1.5)) for b in B]  # small offset so points do not overlap
    ax.plot(x, low, "o", color=color, label=label, ms=7)
    ax.plot(x, low, ":", color=color, lw=1, alpha=0.5)
    if high:
        ax.vlines(x, low, high, color=color, lw=2.5, alpha=0.35)
        ax.plot(x, high, "_", color=color, ms=10, mew=2, alpha=0.6)
ax.axvspan(9.2, 17, color="0.88")
ax.text(12.4, 1.45, "above 8 sequences:\nvLLM's own path\n(not measured)", ha="center",
        va="center", fontsize=9, color="0.25")
ax.axhline(1.0, color="0.4", lw=1, ls="--")
ax.set_xscale("log", base=2)
ax.set_xticks([1, 2, 4, 8, 16])
ax.set_xticklabels(["1", "2", "4", "8", "16"])
ax.set_xlim(0.85, 17)
ax.set_ylim(0.9, 2.6)
ax.set_xlabel("concurrent sequences (batch)")
ax.set_ylabel("decode speedup per user vs stock vLLM")
ax.set_title("gfx942 megakernels vs stock vLLM, one MI300X (TP2: two)", fontsize=11)
ax.legend(fontsize=8.5, loc="upper left", ncol=2, frameon=False)
fig.text(0.01, 0.01, "Dot: against the best stock-vLLM day. Bar: up to the same-session baseline. Measured at batch 1, 2, 4, 8 only.\n"
         "* MXFP4 weights (half the bytes of FP8, 4.9 % higher perplexity than BF16) against vLLM FP8; vLLM has no fast MXFP4 path on gfx942.",
         fontsize=7.5, color="0.3")
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(__file__.replace("plot_speedup.py", "speedup.png"))
