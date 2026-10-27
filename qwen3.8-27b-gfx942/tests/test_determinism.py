"""Run-to-run determinism: two engines decode the same tokens; the logits of every step
must be bitwise equal (the kernel has no float atomics; split-K sums run in a fixed
order). Usage: python tests/test_determinism.py MODEL_DIR [M] [STEPS]
"""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
from mk import reference as R  # noqa: E402
from mk.pack import Packed  # noqa: E402
from mk.runner import Engine  # noqa: E402


def run(eng, toks):
    eng.reset()
    out = []
    for t in range(toks.shape[0]):
        out.append(eng.step(toks[t]).clone())
    return torch.stack(out)


def main():
    path = sys.argv[1]
    M = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    PK = Packed(R.Weights(path, "mxfp4", keep_raw=True, dequant=False))
    toks = torch.randint(0, 200000, (steps, M), device="cuda", generator=torch.Generator("cuda").manual_seed(7))
    a = run(Engine(PK, M, maxlen=steps + 64), toks)
    b = run(Engine(PK, M, maxlen=steps + 64), toks)
    c = run(Engine(PK, M, maxlen=steps + 64), toks)
    same = torch.equal(a, b) and torch.equal(a, c)
    print(f"M={M} steps={steps}: 3 runs bitwise equal: {same}")
    print("PASS" if same else "FAIL")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
