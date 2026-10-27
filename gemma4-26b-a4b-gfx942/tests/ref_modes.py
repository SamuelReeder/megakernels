"""The reference's emulating mode against its exact mode on the FP8 checkpoint, real text.

Both modes read the same FP8 weights; emulate quantizes every GEMV input as the kernel does
(e4m3fnuz, one power-of-two scale per (row, 64 K)), exact rounds it to bf16. Reports:
  - per-position logits cosine and top-1 agreement (emulate vs exact), next-token NLL of each;
  - routing: per (layer, token), whether the two modes pick the same 8 experts, and the gap
    between the 8th and 9th router logit (exact mode) where they differ: how close to a tie
    the routing decisions are. The kernel must reproduce the emulating reference's routing
    except at near-ties (docs/DESIGN.md I5).
Usage: python tests/ref_modes.py <fp8 model dir> [N_TOKENS=400]
Pass: emulate perplexity within 2 % of exact, mean logits cosine > 0.99. (Top-1 agreement is
reported, not gated: the instruction-tuned model on raw text has perplexity ~450 (HF bf16:
459 on these 400 tokens), so many positions are near-ties and argmax flips are noise.)
"""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
from mk import reference as R  # noqa: E402
from tests.texts import long_ids  # noqa: E402

path = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 400
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(path)
ids = torch.tensor(long_ids(tok, N), device="cuda")
W = R.Weights(path, "fp8")
out = {}
for mode in ("exact", "emulate"):
    st = R.State.new(1, N + 8)
    lg, routes, gaps = [], [], []
    for t in range(N):
        d = {}
        lg.append(R.step(W, ids[t : t + 1], st, dump=d, mode=mode)[0])
        routes.append(torch.stack([d[i]["ids"][0].sort().values for i in range(R.NL)]))
        top9 = torch.stack([d[i]["logits"][0].topk(9).values for i in range(R.NL)])
        gaps.append(top9[:, 7] - top9[:, 8])
    out[mode] = (torch.stack(lg), torch.stack(routes), torch.stack(gaps))

(lx, rx, gx), (le, re_, _) = out["exact"], out["emulate"]
cos = torch.nn.functional.cosine_similarity(lx, le, dim=-1)
top = (lx.argmax(-1) == le.argmax(-1)).float()


def nll(lg):
    lp = torch.log_softmax(lg[:-1].double(), -1)
    return -lp.gather(1, ids[1:, None]).squeeze(1)


same = (rx == re_).all(-1)  # [N, NL]
print(f"{N} tokens: emulate~exact logits cos min {cos.min():.6f} mean {cos.mean():.6f} "
      f"top1 {top.mean():.3f}")
print(f"ppl exact {nll(lx).mean().exp():.4f} emulate {nll(le).mean().exp():.4f}")
print(f"routing: same top-8 set in {same.float().mean():.4f} of (token, layer) pairs; "
      f"differs in {(~same).sum().item()} of {same.numel()}")
print(f"exact-mode 8th-9th router logit gap: median {gx.median():.4f}, "
      f"min {gx.min():.2e}, < 1e-3 in {(gx < 1e-3).float().mean():.4f}, "
      f"< 1e-2 in {(gx < 1e-2).float().mean():.4f}")
if (~same).any():
    print(f"gap where the sets differ: median {gx[~same].median():.4f}, max {gx[~same].max():.4f}")
ratio = (nll(le).mean().exp() / nll(lx).mean().exp()).item()
print(f"emulate / exact perplexity {ratio:.4f}")
ok = abs(ratio - 1) < 0.02 and cos.mean() > 0.99
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
