"""End-to-end check: megakernel decode steps vs the torch reference, in both its
emulating (the kernel's rounding) and exact modes, after a CTX-token context.

The reference prefills CTX tokens of real text per row (exact mode, rows at different
offsets of the text); the kernel's paged caches and the emulating reference start from
that cache. Then STEPS greedy steps (all three follow the exact reference's tokens),
comparing the logits of every step. CTX > 1024 exercises the sliding window.
Routing (docs/DESIGN.md I5): with debug_route the kernel keeps every layer's router
logits and CTA 0's routing tables; per layer its tables must equal reference.route_logits
of its own logits, and all CTAs must build the same tables.
Usage: python tests/test_decode.py FP8_DIR [M] [LAYERS] [STEPS] [CTX]
"""
import os
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
from transformers import AutoTokenizer  # noqa: E402

from mk import reference as R  # noqa: E402
from mk.dims import NL  # noqa: E402
from mk.pack import Packed  # noqa: E402
from mk.runner import Engine, default_chunks  # noqa: E402
from tests.texts import long_ids  # noqa: E402

path = sys.argv[1]
M = int(sys.argv[2]) if len(sys.argv) > 2 else 1
LAYERS = int(sys.argv[3]) if len(sys.argv) > 3 else NL
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 16
CTX = int(sys.argv[5]) if len(sys.argv) > 5 else 512
MARGIN = 0.5  # a top-1 change with an exact logit gap above this counts as "hard"


def cmp(a, b):
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).min().item()
    top = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    return cos, top


def main():
    t0 = time.time()
    W = R.Weights(path, "fp8", layers=LAYERS)
    PK = Packed(W, default_chunks(), layers=LAYERS)
    maxlen = CTX + STEPS + 8
    eng = Engine(PK, M, maxlen=maxlen, layers=LAYERS, debug_route=True)
    print(f"node {os.uname().nodename} load+pack+build {time.time() - t0:.0f}s cfg {eng.cfg}")
    tok = AutoTokenizer.from_pretrained(path)
    text = torch.tensor(long_ids(tok, CTX + 64 * M + 1), device="cuda")
    ids = torch.stack([text[64 * m : 64 * m + CTX + 1] for m in range(M)])
    st_x = R.State.new(M, maxlen)
    t1 = time.time()
    R.prefill(W, ids[:, :CTX], st_x, layers=LAYERS)
    print(f"reference prefill {CTX} x {M} rows: {time.time() - t1:.0f}s")
    st_e = R.State(st_x.ks.clone(), st_x.vs.clone(), st_x.kf.clone(), st_x.vf.clone(),
                   st_x.pos.clone())
    eng.load_kv(st_x, CTX)
    cur = ids[:, CTX]
    rows, hard, hard_e, i5_bad, code_bad, rdiff, nonfinite = [], 0, 0, 0, 0, 0, 0
    pes = [W.L(i, "router.per_expert_scale") for i in range(LAYERS)]
    for s in range(STEPS):
        dump = {}
        emu = R.step(W, cur, st_e, layers=LAYERS, mode="emulate", dump=dump)
        ex = R.step(W, cur, st_x, layers=LAYERS, mode="exact")
        out = eng.step(cur).clone()
        torch.cuda.synchronize()
        # min() over the cosines can skip a NaN: count non-finite logits on their own
        # (as qwen3.8-27b-gfx942/tests/test_decode.py since c77297d)
        nonfinite += int(not bool(torch.isfinite(out).all() and torch.isfinite(emu).all()
                                  and torch.isfinite(ex).all()))
        rh = eng.route_hash[:LAYERS]
        i5_bad += int(not (bool((rh == rh[:, :1]).all()) and bool((rh != 0).all())))
        # per layer: the kernel's tables = reference.route_logits of its own logits
        bad_l, cos_l = 0, []
        for i in range(LAYERS):
            klg = eng.router[i]
            ridx, rw = R.route_logits(klg, pes[i])
            kp, kw = eng.route_pick[i].long(), eng.route_w[i][:, :8]
            bad_l += int(not torch.equal(kp, ridx) or not torch.allclose(kw, rw, rtol=1e-5,
                                                                           atol=1e-7))
            cos_l.append(torch.nn.functional.cosine_similarity(
                klg, dump[i]["logits"], dim=-1).min().item())
            rdiff += sum(set(kp[m].tolist()) != set(dump[i]["ids"][m].tolist())
                         for m in range(M))
        code_bad += int(bad_l > 0)
        k_e, k_x, e_x = cmp(out, emu), cmp(out, ex), cmp(emu, ex)
        rows.append((k_e, k_x, e_x))
        gap = (ex.max(-1).values - ex.gather(1, out.argmax(-1, keepdim=True))[:, 0]).max()
        gap_e = (ex.max(-1).values - ex.gather(1, emu.argmax(-1, keepdim=True))[:, 0]).max()
        hard += int(float(gap) > MARGIN)
        hard_e += int(float(gap_e) > MARGIN)
        print(
            f"step {s}: kernel~emulate cos {k_e[0]:.6f} top1 {k_e[1]:.2f} | kernel~exact "
            f"cos {k_x[0]:.6f} top1 {k_x[1]:.2f} (gap {float(gap):.3f}) | emulate~exact cos "
            f"{e_x[0]:.6f} top1 {e_x[1]:.2f} | router cos min over layers {min(cos_l):.6f} "
            f"(layer 0 {cos_l[0]:.6f}) | routing-code layers wrong {bad_l}"
        )
        cur = ex.argmax(-1)
    mean = lambda k, j: sum(r[k][j] for r in rows) / len(rows)  # noqa: E731
    worst = lambda k: min(r[k][0] for r in rows)  # noqa: E731
    print(f"kernel~emulate: worst cos {worst(0):.6f} top1 {mean(0, 1):.3f}")
    print(f"kernel~exact:   worst cos {worst(1):.6f} top1 {mean(1, 1):.3f}")
    print(f"emulate~exact:  worst cos {worst(2):.6f} top1 {mean(2, 1):.3f}")
    print(f"routing vs emulate (drifted inputs; information): {rdiff} (row, layer, step) "
          f"expert-set differences of {M * LAYERS * STEPS}")
    print(f"steps with a top-1 change at an exact logit gap > {MARGIN} (information): "
          f"kernel {hard}, emulating reference {hard_e}")
    print(f"I5: steps where two CTAs built different routing tables: {i5_bad}")
    print(f"routing code: steps where a layer's tables differ from route_logits of the "
          f"kernel's logits: {code_bad}")
    print(f"steps with non-finite logits (kernel or reference): {nonfinite}")
    # fp8 activation rounding is chaotic over 30 layers (routing flips): two correct fp8
    # implementations (kernel, emulating reference) drift apart as far as each drifts
    # from the exact model, and the emulating reference itself changes top-1 tokens at
    # large exact logit gaps (docs/DESIGN.md "Correctness oracle"). Pass (P3): the kernel
    # is as close to the exact model as the emulating reference is, in worst-step cosine
    # and in top-1 agreement (factor 2 plus one row-step of slack); its routing tables are
    # route_logits of its own logits and equal in every CTA. kernel~emulate is printed
    # only: no absolute floor (the emulating reference itself drifts to cosine 0.89 from
    # the exact model at M = 2), and per step the angles obey the triangle inequality, so
    # a relative bound adds nothing (docs/DESIGN.md "Correctness oracle").
    n = len(rows) * M
    ok = (
        (1 - worst(1)) <= 2 * (1 - worst(2)) + 1e-5
        and (1 - mean(1, 1)) <= 2 * (1 - mean(2, 1)) + 1 / n
        and i5_bad == 0
        and code_bad == 0
        and nonfinite == 0
    )
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
