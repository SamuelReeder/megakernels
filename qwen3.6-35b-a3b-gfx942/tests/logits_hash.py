"""Bit oracle of the common/ refactor (common/DESIGN.md oracle 3): sha256 of the output
bytes of STEPS decode steps with fixed tokens (seed 7), per M, with lm_head on (fp32
logits) and off (bf16 normed hidden states). Any change of the kernel's output bits
changes the hash.
Usage: python tests/logits_hash.py MODEL_DIR [Ms=1,2,4,8] [STEPS=16] [LAYERS=40] [LM=on,off]
"""
import hashlib
import sys

import torch

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
from mk import reference as R  # noqa: E402
from mk.pack import Packed  # noqa: E402
from mk.runner import Engine, default_chunks  # noqa: E402


def main():
    path = sys.argv[1]
    Ms = [int(m) for m in (sys.argv[2] if len(sys.argv) > 2 else "1,2,4,8").split(",")]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    L = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    lms = (sys.argv[5] if len(sys.argv) > 5 else "on,off").split(",")
    W = R.Weights(path, keep_raw=True, layers=L)
    PK = Packed(W, default_chunks(), layers=L)
    for lm in lms:
        for M in Ms:
            eng = Engine(PK, M, maxlen=steps + 64, layers=L, lm_head=lm == "on")
            toks = torch.randint(0, 200000, (steps, M), device="cuda",
                                 generator=torch.Generator("cuda").manual_seed(7))
            h, fin = hashlib.sha256(), True
            for t in range(steps):
                o = eng.step(toks[t]).clone()
                torch.cuda.synchronize()
                h.update(o.contiguous().view(torch.uint8).cpu().numpy().tobytes())
                fin &= bool(torch.isfinite(o).all())
            print(f"HASH tp1 M={M} lm={lm} steps={steps} layers={L} "
                  f"{'logits' if lm == 'on' else 'hidden'} {h.hexdigest()[:16]} finite {fin}",
                  flush=True)
            del eng
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
