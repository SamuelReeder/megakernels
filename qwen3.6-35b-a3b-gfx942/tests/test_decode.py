"""End-to-end check: megakernel decode steps vs the torch reference, in both its
emulating (the kernel's rounding) and exact (fp32) modes.

Decodes greedily from random tokens (all three follow the exact reference's tokens),
comparing the logits of every step, and the last layer's router logits and expert
selection against the emulating reference.
Usage: python tests/test_decode.py FP8_DIR [M] [LAYERS] [STEPS]
Env: MK_XPACK=1 (ROCM_ATTN KV layout), MK_BLOCK (KV block), MK_SEED.
"""
import os
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
from mk import reference as R  # noqa: E402
from mk.pack import Packed  # noqa: E402
from mk.runner import Engine, default_chunks  # noqa: E402

path = sys.argv[1]
M = int(sys.argv[2]) if len(sys.argv) > 2 else 1
LAYERS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 8
TIE = 1e-3  # router logits closer than this to the 8th/9th boundary count as near-ties


def cmp(a, b):
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).min().item()
    top = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    return cos, top


def routing_check(klg, tr):
    """Kernel router logits [M, LGN] vs the emulating reference's last-layer trace:
    (cosine, rows with a different expert set, of which near-ties)."""
    rl = tr["router"]
    cos = torch.nn.functional.cosine_similarity(klg[:, : rl.shape[1]], rl, dim=-1).min().item()
    kidx, _ = R.route(klg[:, :256])
    diff = tie = 0
    for m in range(klg.shape[0]):
        if set(kidx[m].tolist()) != set(tr["idx"][m].tolist()):
            diff += 1
            s = rl[m, :256].sort(descending=True).values
            tie += int(float(s[7] - s[8]) < TIE)
    return cos, diff, tie


def main():
    t0 = time.time()
    W = R.Weights(path, keep_raw=True, layers=LAYERS)
    PK = Packed(W, default_chunks(), layers=LAYERS)
    eng = Engine(
        PK,
        M,
        maxlen=512,
        layers=LAYERS,
        xpack=bool(int(os.environ.get("MK_XPACK", "0"))),
        block=int(os.environ.get("MK_BLOCK", "208")),
    )
    print(f"load+pack+build {time.time() - t0:.0f}s  kernel cfg {eng.cfg}")
    st = R.State.new(M, 512)
    st_x = R.State.new(M, 512)
    torch.manual_seed(int(os.environ.get("MK_SEED", "0")))
    tok = torch.randint(0, 200000, (M,), device="cuda")
    rows, bad_route = [], 0
    for s in range(STEPS):
        tr = []
        emu = R.step(W, tok, st, layers=LAYERS, emulate=True, trace=tr)
        ex = R.step(W, tok, st_x, layers=LAYERS, emulate=False)
        out = eng.step(tok).clone()
        torch.cuda.synchronize()
        rc, rdiff, rtie = routing_check(eng.router.clone(), tr[-1])
        bad_route += rdiff - rtie
        k_e, k_x, e_x = cmp(out, emu), cmp(out, ex), cmp(emu, ex)
        rows.append((k_e, k_x, e_x))
        print(
            f"step {s}: kernel~emulate cos {k_e[0]:.6f} top1 {k_e[1]:.2f} | kernel~exact "
            f"cos {k_x[0]:.6f} top1 {k_x[1]:.2f} | emulate~exact cos {e_x[0]:.6f} top1 "
            f"{e_x[1]:.2f} | router cos {rc:.6f} expert-set diffs {rdiff} (near-ties {rtie})"
        )
        tok = ex.argmax(-1)
    mean = lambda k, j: sum(r[k][j] for r in rows) / len(rows)  # noqa: E731
    worst = lambda k: min(r[k][0] for r in rows)  # noqa: E731
    print(f"kernel~emulate: worst cos {worst(0):.6f} top1 {mean(0, 1):.3f}")
    print(f"kernel~exact:   worst cos {worst(1):.6f} top1 {mean(1, 1):.3f}")
    print(f"emulate~exact:  worst cos {worst(2):.6f} top1 {mean(2, 1):.3f}")
    print(f"routing: {bad_route} expert-set differences that are not near-ties")
    # fp8 activation rounding is chaotic over 40 layers: two correct fp8 implementations
    # (kernel, emulating reference) drift apart as far as each drifts from the exact
    # model. Pass: the kernel is as close to the exact model as the emulating reference
    # is. With <= 4 layers the drift is small, so the last layer must also select the
    # same experts as the emulating reference except at near-ties.
    ok = (
        worst(0) > 0.95
        and (1 - worst(1)) <= 2 * (1 - worst(2)) + 1e-5
        and mean(1, 1) >= mean(2, 1) - 0.1
        and (bad_route == 0 or LAYERS > 4)
    )
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
