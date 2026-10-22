"""Per-SM timeline of one decode step: grid barriers (first commit) vs. flag-in-data dataflow (now).

The first version of the kernel separated its 121 ops per token with grid-wide barriers. This
profiles that version from git history (in a throwaway worktree, in its own process because both
kernels register the same extension name) and the current kernel, both in bf16 at the same context,
and draws them on one time axis.

    python tools/before_after.py --png docs/before_after.png
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V1_COMMIT = "247df97"
OP_KINDS = ["in_proj", "mixer", "out_proj", "mlp_gate_up", "mlp_down"]

V1_SNIPPET = r"""
import sys, numpy as np, torch
from qwen_mk.engine import Megakernel
ctx, reps, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
mk = Megakernel(max_seq=ctx + 64)
best = None
for _ in range(reps):
    mk.reset()
    last = mk.run([(i * 7919) % 50000 + 1000 for i in range(ctx)], n_generate=1)[-1]
    buf = mk.prof_buffer()
    mk.run([last], n_generate=8, prof=buf, prof_step=4)
    torch.cuda.synchronize()
    g, n, F = mk.grid, mk.ext.PROF_MAX_OPS, mk.ext.PROF_FIELDS
    ops = buf.cpu().numpy().astype(np.int64)[: g * n * F].reshape(g, n, F)
    span = ops[:, n - 2, 1].max() - ops[:, n - 1, 3].min()
    if best is None or span < best[0]:
        best = (span, ops)
np.save(out, best[1])
"""


def profile_v1(ctx, reps):
    with tempfile.TemporaryDirectory() as tmp:
        tree, out = os.path.join(tmp, "v1"), os.path.join(tmp, "ops.npy")
        subprocess.run(["git", "worktree", "add", "--detach", tree, V1_COMMIT], cwd=ROOT, check=True,
                       capture_output=True)
        try:
            # The first commit lives at the repository root; later ones moved it under the model directory.
            src = tree if os.path.isdir(os.path.join(tree, "qwen_mk")) else os.path.join(tree, "qwen3.5-0.8b")
            subprocess.run([sys.executable, "-I", "-c", V1_SNIPPET, str(ctx), str(reps), out], cwd=src,
                           env={**os.environ, "PYTHONPATH": src}, check=True)
            ops = np.load(out)
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", tree], cwd=ROOT, capture_output=True)
    n = ops.shape[1]
    t0 = ops[:, n - 1, 3].min()
    # the first kernel's profile has no slot for the LM head, so both sides are cut at the end of layer 23
    arrive, release = (ops[:, :n - 1, 0] - t0) / 1e3, (ops[:, :n - 1, 1] - t0) / 1e3
    prev = np.concatenate([(ops[:, n - 1, 3:4] - t0) / 1e3, release[:, :-1]], axis=1)
    # op i computes between release[i-1] and arrive[i], then waits at the barrier until release[i]
    return dict(wait_start=arrive, start=release, end=release, comp_start=prev, comp_end=arrive)


def profile_now(ctx, reps):
    from qwen_mk.engine import Megakernel
    from timeline import collect, summarize

    mk = Megakernel(max_seq=ctx + 64)
    d = min((collect(mk, ctx) for _ in range(reps)), key=lambda d: summarize(mk, d)["step_us"])
    return mk, {k: v[:, :120] for k, v in dict(wait_start=d["wait_start"], start=d["start"], end=d["end"],
                                               comp_start=d["start"], comp_end=d["end"]).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--zoom", type=float, nargs=2, default=(100, 180), metavar=("T0", "T1"))
    ap.add_argument("--png", default=os.path.join(ROOT, "docs", "before_after.png"))
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from timeline import COLORS, WAIT_COLOR, op_names

    old = profile_v1(args.ctx, args.reps)
    mk, new = profile_now(args.ctx, args.reps)
    names = op_names(mk.cfg)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.6), dpi=150, sharey=True,
                             gridspec_kw=dict(width_ratios=[2.2, 1], wspace=0.04, hspace=0.28))
    fig.patch.set_facecolor("#fcfcfb")
    full = max(old["comp_end"][:, -1].max(), new["end"][:, -1].max()) * 1.01
    for row, (d, title) in enumerate(((old, "Before: a grid-wide barrier after every op"),
                                      (new, "After: ops hand results over through flags in the data"))):
        g = d["start"].shape[0]
        step = d["comp_end"][:, -1].max()
        idle = float(np.mean(d["start"] - d["wait_start"], axis=0).sum())
        for col, ax in enumerate(axes[row]):
            ax.set_facecolor("#fcfcfb")
            lo, hi = (0, full) if col == 0 else args.zoom
            for i, (_, k, _) in enumerate(names[:120]):
                for b in range(g):
                    ws, st = d["wait_start"][b, i], d["start"][b, i]
                    cs, ce = (d["comp_start"][b, i], d["comp_end"][b, i])
                    if max(ws, cs) > hi or min(st, ce) < lo:
                        continue
                    ax.add_patch(plt.Rectangle((ws, b + 0.1), st - ws, 0.8, color=WAIT_COLOR, lw=0))
                    ax.add_patch(plt.Rectangle((cs, b + 0.1), ce - cs, 0.8, color=COLORS[k], lw=0))
            ax.set_xlim(lo, hi)
            ax.set_ylim(g, 0)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            for s_ in ("left", "bottom"):
                ax.spines[s_].set_color("#c3c2b7")
            ax.tick_params(colors=MUTED)
            if col == 0:
                ax.set_ylabel("SM", color=INK2)
                ax.set_title(f"{title}: layers 0-23 in {step:.0f} µs, {idle:.0f} µs of it waiting", loc="left",
                             color=INK, fontsize=11.5)
            else:
                ax.set_title(f"zoom: {lo:.0f}-{hi:.0f} µs", loc="left", color=INK2, fontsize=10)
                ax.tick_params(axis="y", length=0)
    for ax in axes[1]:
        ax.set_xlabel("time since step start (µs)", color=INK2)
    handles = [Patch(color=COLORS[k], label=k.replace("_", " ")) for k in COLORS if k != "lm_head"]
    handles.append(Patch(color=WAIT_COLOR, label="waiting (barrier / inputs)"))
    fig.legend(handles=handles, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 0.025), frameon=False, labelcolor=INK2, fontsize=9)
    fig.text(0.01, 0.005, f"Qwen3.5-0.8B bf16, ctx={args.ctx}, RTX 5080. Fastest of {args.reps} profiled steps each; "
             f"'before' is commit {V1_COMMIT}. The LM head is omitted: the first kernel's profiler has no slot for it.",
             color=MUTED, fontsize=8.5)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.15)
    fig.savefig(args.png, facecolor=fig.get_facecolor())
    print("wrote", args.png)


if __name__ == "__main__":
    main()
