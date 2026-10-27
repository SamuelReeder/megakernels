"""End-to-end check: megakernel decode steps vs the torch reference, in both its
emulating (fp8 activations, as the kernel) and exact (bf16 activations) modes.

Feeds a prompt token by token (both sides), then decodes greedily, comparing the
logits of every step. Usage: python tests/test_decode.py MODEL_DIR [M] [LAYERS] [STEPS]
TP2: MK_TP=2 MK_AR=fp32|bf16 srun -n2 python tests/test_decode.py ... (mk/tp.py); rank 0
runs the reference, and the residual must be bitwise equal on both ranks every step.
"""
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
from mk import reference as R  # noqa: E402
from mk import tp as TP  # noqa: E402

path = sys.argv[1]
M = int(sys.argv[2]) if len(sys.argv) > 2 else 1
LAYERS = int(sys.argv[3]) if len(sys.argv) > 3 else 64
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 8


def main():
    t0 = time.time()
    ctx = TP.init()
    W = R.Weights(path, "mxfp4", keep_raw=True, layers=LAYERS, dequant=ctx.main)
    PK = TP.packed(ctx, W, layers=LAYERS)
    eng = TP.engine(ctx, PK, M, maxlen=512, layers=LAYERS, xpack=bool(int(__import__("os").environ.get("MK_XPACK", "0"))), block=int(__import__("os").environ.get("MK_BLOCK", "208")))
    if not ctx.main:  # the other ranks only run their shard: rank 0 sends the tokens
        for s in range(STEPS):
            eng.step(torch.zeros(M, dtype=torch.long))
        sys.exit(0 if TP.agree(ctx, eng.ranks_equal) else 1)
    # emulate the TP kernel's sum order: K-slice partials per rank, added in rank order
    # (rounded to bf16 for MK_AR=bf16); MK_EMU_SPLIT=0 emulates TP1's order instead
    split = ctx.tp if int(__import__("os").environ.get("MK_EMU_SPLIT", "1")) else 1
    ar_bf16 = ctx.tp > 1 and ctx.ar_dtype == "bf16"
    print(f"load+pack+build {time.time() - t0:.0f}s tp {ctx.tp} ar {ctx.ar_dtype if ctx.tp > 1 else '-'}")
    st = R.State.new(M, 512)
    st_x = R.State.new(M, 512)
    torch.manual_seed(int(__import__("os").environ.get("MK_SEED", "0")))
    tok = torch.randint(0, 200000, (M,), device="cuda")

    def cmp(a, b):
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).min().item()
        top = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
        return cos, top

    rows = []
    for s in range(STEPS):
        emu = R.step(W, tok, st, layers=LAYERS, emulate=True, ar_tp=split, ar_bf16=ar_bf16)
        ex = R.step(W, tok, st_x, layers=LAYERS, emulate=False)
        out = eng.step(tok).clone()
        torch.cuda.synchronize()
        k_e, k_x, e_x = cmp(out, emu), cmp(out, ex), cmp(emu, ex)
        rows.append((k_e, k_x, e_x))
        print(f"step {s}: kernel~emulate cos {k_e[0]:.6f} top1 {k_e[1]:.2f} | kernel~exact cos {k_x[0]:.6f} "
              f"top1 {k_x[1]:.2f} | emulate~exact cos {e_x[0]:.6f} top1 {e_x[1]:.2f}")
        tok = ex.argmax(-1)           # all three follow the exact reference's tokens
    mean = lambda k, j: sum(r[k][j] for r in rows) / len(rows)  # noqa: E731
    worst = lambda k: min(r[k][0] for r in rows)  # noqa: E731
    print(f"kernel~emulate: worst cos {worst(0):.6f} top1 {mean(0, 1):.3f}")
    print(f"kernel~exact:   worst cos {worst(1):.6f} top1 {mean(1, 1):.3f}")
    print(f"emulate~exact:  worst cos {worst(2):.6f} top1 {mean(2, 1):.3f}")
    # fp8 activation rounding is chaotic over 64 layers: two correct fp8 implementations
    # (kernel, emulating reference) drift apart as far as each drifts from the exact
    # (bf16-activation) model (with random tokens, worst-row cosines near 0.97 occur).
    # Pass: the kernel is as close to the exact model as the emulating reference is.
    ok = (worst(0) > 0.95 and (1 - worst(1)) <= 2 * (1 - worst(2)) + 1e-5
          and mean(1, 1) >= mean(2, 1) - 0.1)
    if ctx.tp > 1:
        print(f"ranks bitwise equal every step: {eng.ranks_equal}")
        ok = TP.agree(ctx, ok and eng.ranks_equal)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
