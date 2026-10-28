"""A/B check of two kernel configurations that must give the same bits (an optimization
that only moves work): decode the same random tokens with Cfg flags A and B (MK_CFG
syntax, e.g. "router_publish=0") and compare the logits of every step bitwise.
Usage: python tests/ab_bitwise.py FP8_DIR M STEPS "FLAGS_A" "FLAGS_B"
"""
import os
import sys

import torch

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
from mk import reference as R  # noqa: E402
from mk.pack import Packed  # noqa: E402
from mk.runner import Engine, default_chunks  # noqa: E402


def run(PK, M, toks, flags):
    os.environ["MK_CFG"] = flags
    e = Engine(PK, M, maxlen=toks.shape[0] + 64)
    out = []
    for t in range(toks.shape[0]):
        out.append(e.step(toks[t]).clone())
        if t % 128 == 127:
            torch.cuda.synchronize()
    print(f"{flags or '(defaults)'}: {e.cfg}", flush=True)
    return torch.stack(out)


def main():
    path, M, steps, fa, fb = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
    print(f"node {os.uname().nodename}")
    PK = Packed(R.Weights(path, "fp8"), default_chunks())
    g = torch.Generator("cuda").manual_seed(3)
    toks = torch.randint(0, 262144, (steps, M), device="cuda", generator=g)
    a, b = run(PK, M, toks, fa), run(PK, M, toks, fb)
    same = torch.equal(a, b)
    nd = int((a != b).any(-1).any(-1).sum())
    print(f"M={M} steps={steps}: bitwise equal {same} (steps that differ: {nd})")
    print("PASS" if same else "FAIL")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
