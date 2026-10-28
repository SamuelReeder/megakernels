"""Run-to-run determinism: two engines decode the same tokens; the logits of every step
must be bitwise equal (the kernel has no float atomics; split-K sums run in a fixed
order). Usage: python tests/test_determinism.py MODEL_DIR [M] [STEPS]
TP2: MK_TP=2 MK_AR=fp32|bf16 srun -n2 ... (mk/tp.py): the gathered logits of the 3 runs
must be bitwise equal, and the residual bitwise equal on both ranks every step.
"""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
from mk import reference as R  # noqa: E402
from mk import tp as TP  # noqa: E402


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
    ctx = TP.init()
    PK = TP.packed(ctx, R.Weights(path, "mxfp4", keep_raw=True, dequant=False))
    toks = torch.randint(0, 200000, (steps, M), device="cuda", generator=torch.Generator("cuda").manual_seed(7))
    engs = [TP.engine(ctx, PK, M, maxlen=steps + 64) for _ in range(3)]
    a, b, c = (run(e, toks) for e in engs)
    same = torch.equal(a, b) and torch.equal(a, c)
    print(f"M={M} steps={steps}: 3 runs bitwise equal: {same}")
    # a build change that must not change any bit: compare this hash across builds
    sha = __import__("hashlib").sha256(a.float().cpu().numpy().tobytes()).hexdigest()[:16]
    print(f"logits sha256 {sha}")
    if ctx.tp > 1:
        req = all(e.ranks_equal for e in engs)
        print(f"tp {ctx.tp} ar {ctx.ar_dtype}: ranks bitwise equal every step: {req}")
        same = TP.agree(ctx, same and req)
    print("PASS" if same else "FAIL")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
