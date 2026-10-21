"""Per-SM timeline of one decode step inside the megakernel.

Each CTA's consumer thread 0 records, for every op, when its inputs became ready
(all LL words it reads carried the expected epoch) and when it published its outputs. Warp 0 also accumulates how long it
waited for weights to land in the ring, and the producer records how long it stalled
on a full ring. Nsight sees the megakernel as one opaque kernel; this shows the inside.

    python tools/timeline.py --ctx 512 --png docs/timeline.png --json results/profile.json
"""

import argparse
import json

import numpy as np
import torch

from qwen_mk.engine import Megakernel

OP_KINDS = ["in_proj", "mixer", "out_proj", "mlp_gate_up", "mlp_down"]
# Reference categorical palette, slots 1-6 in fixed order (validated adjacent-pair order).
COLORS = {
    "in_proj": "#2a78d6",
    "mixer": "#eb6834",
    "out_proj": "#1baf7a",
    "mlp_gate_up": "#eda100",
    "mlp_down": "#e87ba4",
    "lm_head": "#008300",
}
WAIT_COLOR = "#e1e0d9"


def op_names(cfg):
    names = []
    for l, t in enumerate(cfg.layer_types):
        mixer = "gdn" if t == "linear_attention" else "attn"
        for k in OP_KINDS:
            names.append((l, k, mixer))
    names.append((-1, "lm_head", ""))
    return names


def collect(mk, ctx, step=4):
    mk.reset()
    prompt = [(i * 7919) % 50000 + 1000 for i in range(ctx)]
    last = mk.run(prompt, n_generate=1)[-1]
    buf = mk.prof_buffer()
    mk.run([last], n_generate=step + 4, prof=buf, prof_step=step)
    torch.cuda.synchronize()
    g, n_ops, F = mk.grid, mk.ext.PROF_MAX_OPS, mk.ext.PROF_FIELDS
    raw = buf.cpu().numpy().astype(np.int64)
    ops = raw[: g * n_ops * F].reshape(g, n_ops, F)
    producer_stall = raw[g * n_ops * F:].reshape(g, F)[:, 0]
    t0 = ops[:, n_ops - 1, 3].min()
    ready = (ops[:, :, 0] - t0) / 1e3  # inputs observed, compute starts (us)
    done = (ops[:, :, 1] - t0) / 1e3   # outputs published
    full_wait = ops[:, :, 2] / 1e3
    prev_done = np.concatenate([(ops[:, n_ops - 1, 3:4] - t0) / 1e3, done[:, :-1]], axis=1)
    return dict(wait_start=prev_done, start=ready, end=done, full_wait=full_wait,
                producer_stall=producer_stall / 1e3)


def summarize(mk, d):
    names = op_names(mk.cfg)
    step_us = d["end"][:, -1].max()
    busy = d["end"] - d["start"]
    wait = d["start"] - d["wait_start"]
    by_kind = {}
    for i, (l, k, mixer) in enumerate(names):
        key = f"{k}:{mixer}" if k == "mixer" else k
        e = by_kind.setdefault(key, dict(busy_us=0.0, dep_wait_us=0.0, weight_wait_us=0.0, count=0))
        e["busy_us"] += busy[:, i].mean()
        e["dep_wait_us"] += wait[:, i].mean()
        e["weight_wait_us"] += d["full_wait"][:, i].mean()
        e["count"] += 1
    return dict(
        step_us=float(step_us),
        ops=len(names),
        mean_dep_wait_us=float(wait.mean()),
        total_dep_wait_us=float(wait.mean(axis=0).sum()),
        total_weight_wait_us=float(d["full_wait"].mean(axis=0).sum()),
        by_kind={k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv for kk, vv in v.items()}
                 for k, v in by_kind.items()},
    )


def plot(mk, d, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    names = op_names(mk.cfg)
    g = d["start"].shape[0]
    fig, ax = plt.subplots(figsize=(14, 6.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for i, (l, k, _) in enumerate(names):
        c = COLORS[k]
        for b in range(g):
            ax.add_patch(plt.Rectangle((d["wait_start"][b, i], b + 0.08), d["start"][b, i] - d["wait_start"][b, i],
                                       0.84, color=WAIT_COLOR, lw=0))
            ax.add_patch(plt.Rectangle((d["start"][b, i], b + 0.08), d["end"][b, i] - d["start"][b, i], 0.84,
                                       color=c, lw=0))
    ax.set_xlim(0, d["end"].max() * 1.005)
    ax.set_ylim(g, 0)
    ax.set_xlabel("time since step start (µs)", color="#52514e")
    ax.set_ylabel("SM (one persistent CTA each)", color="#52514e")
    ax.set_title(title, loc="left", color="#0b0b0b", fontsize=12)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors="#898781")
    handles = [Patch(color=COLORS[k], label=k.replace("_", " ")) for k in COLORS]
    handles.append(Patch(color=WAIT_COLOR, label="waiting on inputs"))
    ax.legend(handles=handles, ncol=7, loc="upper left", bbox_to_anchor=(0, -0.09), frameon=False,
              labelcolor="#52514e", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--png", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--quant", choices=["bf16", "int8"], default="bf16")
    args = ap.parse_args()
    mk = Megakernel(max_seq=args.ctx + 64, quant=args.quant)
    d = collect(mk, args.ctx)
    s = summarize(mk, d)
    print(f"step {s['step_us']:.1f} us over {s['ops']} ops; dependency wait {s['total_dep_wait_us']:.1f} us/step "
          f"(mean {s['mean_dep_wait_us']:.2f} us/op); weight wait {s['total_weight_wait_us']:.1f} us/step")
    print(f"{'op':16s} {'n':>3s} {'busy us':>9s} {'dep-wait us':>12s} {'weight-wait us':>15s}")
    for k, v in s["by_kind"].items():
        print(f"{k:16s} {v['count']:3d} {v['busy_us']:9.1f} {v['dep_wait_us']:12.1f} {v['weight_wait_us']:15.1f}")
    if args.png:
        plot(mk, d, args.png, f"One decode step, ctx={args.ctx}: {s['step_us']:.0f} µs across {mk.grid} SMs")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(s, f, indent=1)


if __name__ == "__main__":
    main()
