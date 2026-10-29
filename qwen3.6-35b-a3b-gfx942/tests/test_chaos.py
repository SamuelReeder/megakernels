"""Timing perturbation (P1, I2): the logits must not depend on the arrival order.

Decodes STEPS steps of random tokens at M rows twice: with the kernel of MK_CFG, and
with the same kernel plus `chaos=1` (dev.chaos_sleep: random 0..7 us delays before every
grid-barrier arrival and every counter atomic). The logits of every step must be
bitwise equal. A race in a barrier, counter or flag protocol shows up as a difference.
Usage: [MK_CFG=...] python tests/test_chaos.py FP8_DIR [M] [STEPS]
"""
import os
import sys

import torch

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
from mk import reference as R  # noqa: E402
from mk.pack import Packed  # noqa: E402
from mk.runner import Engine, default_chunks  # noqa: E402


def run(PK, M, toks):
    e = Engine(PK, M, maxlen=toks.shape[0] + 64)
    print(f"cfg {e.cfg}", flush=True)
    out = torch.stack([e.step(toks[t]).clone() for t in range(toks.shape[0])])
    del e
    torch.cuda.empty_cache()
    return out


def main():
    path = sys.argv[1]
    M = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    PK = Packed(R.Weights(path, keep_raw=True), default_chunks())
    gen = torch.Generator("cuda").manual_seed(11)
    toks = torch.randint(0, 200000, (steps, M), device="cuda", generator=gen)
    base = os.environ.get("MK_CFG", "")
    a = run(PK, M, toks)
    os.environ["MK_CFG"] = ",".join(filter(None, [base, "chaos=1"]))
    b = run(PK, M, toks)
    os.environ["MK_CFG"] = base
    nan = not torch.isfinite(a).all().item()
    ndiff = int((a != b).any(-1).any(-1).sum())
    ok = torch.equal(a, b) and not nan
    print(f"M={M} steps={steps}: chaos vs no chaos bitwise equal {torch.equal(a, b)} "
          f"(steps that differ: {ndiff}); non-finite logits: {nan}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
