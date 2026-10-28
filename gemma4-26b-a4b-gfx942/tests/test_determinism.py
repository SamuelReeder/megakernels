"""Determinism and batch invariance (docs/DESIGN.md invariant I4); from
qwen3.6-35b-a3b-gfx942/tests/test_determinism.py.

1. Run to run: three runs decode the same tokens at M; the logits of every step must be
   bitwise equal (no float atomics; every sum runs in a fixed order).
2. Batch invariance: row 0 decodes the same tokens alone (M = 1) and inside batches of
   M = 2, 4, 8 whose other rows decode other random tokens (other experts, another
   number of selected experts). Row 0's logits must be bitwise equal in all of them.
3. Padding rows: at M = 8 with rows 4..7 marked padding (valid = 0, as vLLM pads a
   batch), row 0 must again be bitwise equal to the M = 1 run.
STEPS > 1024 crosses the sliding window.
Usage: python tests/test_determinism.py FP8_DIR [M] [STEPS]
"""
import os
import sys

import torch

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
from mk import reference as R  # noqa: E402
from mk.pack import Packed  # noqa: E402
from mk.runner import Engine, default_chunks  # noqa: E402


def run(eng, toks, valid=None):
    eng.reset()
    out = []
    for t in range(toks.shape[0]):
        out.append(eng.step(toks[t], valid).clone())
        if t % 256 == 0:
            torch.cuda.synchronize()
            print(f"  M={eng.M} step {t}", flush=True)
    return torch.stack(out)


def main():
    path = sys.argv[1]
    M = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    print(f"node {os.uname().nodename}")
    PK = Packed(R.Weights(path, "fp8"), default_chunks())
    gen = torch.Generator("cuda").manual_seed(7)
    toks = torch.randint(0, 200000, (steps, 8), device="cuda", generator=gen)

    e = Engine(PK, M, maxlen=steps + 64)
    a, b, c = run(e, toks[:, :M]), run(e, toks[:, :M]), run(e, toks[:, :M])
    same = torch.equal(a, b) and torch.equal(a, c)
    print(f"M={M} steps={steps}: 3 runs bitwise equal: {same}")
    del e, a, b, c

    ok = same
    alone = padded = None
    for m in (1, 2, 4, 8):
        e = Engine(PK, m, maxlen=steps + 64)
        out = run(e, toks[:, :m])[:, 0]
        if m == 8:
            valid = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], device="cuda")
            padded = run(e, toks[:, :m], valid)[:, 0]
        del e
        torch.cuda.empty_cache()
        if alone is None:
            alone = out
            continue
        eq = torch.equal(out, alone)
        ndiff = int((out != alone).any(-1).sum())
        print(f"row 0 alone vs inside M={m}: bitwise equal {eq} (steps that differ: {ndiff})")
        ok &= eq
    eq = torch.equal(padded, alone)
    print(f"row 0 alone vs inside M=8 with rows 4..7 padding: bitwise equal {eq}")
    ok &= eq
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
