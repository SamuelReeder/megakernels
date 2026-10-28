"""Teacher-forced quality check on real text: megakernel vs the PyTorch reference.

The same token sequences go through three systems:
  kernel  the megakernel (FP8 checkpoint, fp8 activations), one decode step per token
  fp8     mk/reference.py exact mode (prefill): the same FP8 weights, bf16 GEMV inputs
  bf16    mk/reference.py exact mode on the BF16 checkpoint (checked against HF
          transformers by tests/ref_vs_hf.py)
Sequences: the five passages of tests/texts.py, and one long text of LONG tokens
(> the 1024-token sliding window). For each: perplexity, top-1 agreement, KL against
bf16. Batch invariance: the passages decoded alone (M = 1) and together (M = 5, first
min-length tokens) must give bitwise-equal kernel logits.
Pass: kernel perplexity <= 1.02 x the fp8 reference's on every sequence group, and
batch invariance.
Usage: python tests/eval_text.py FP8_DIR BF16_DIR [LONG=1536]
"""
import os
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
from transformers import AutoTokenizer  # noqa: E402

from mk import reference as R  # noqa: E402
from mk.pack import Packed  # noqa: E402
from mk.runner import Engine, default_chunks  # noqa: E402
from tests.texts import TEXTS, long_ids  # noqa: E402


def ref_logits(W, ids):
    """Reference logits [L, V] (cpu) for every position of ids [L] (exact prefill)."""
    st = R.State.new(1, len(ids) + 8)
    hid = R.prefill(W, ids[None].cuda(), st)[0]
    return torch.cat([R.logits_of(W, hid[i:i + 256]).cpu() for i in range(0, len(ids), 256)])


def kernel_logits(eng, rows):
    """Kernel logits [M][L][V] (cpu) for M equal-length token rows, one step per position."""
    eng.reset()
    out = [eng.step(rows[:, t].cuda()).clone().cpu() for t in range(rows.shape[1])]
    return torch.stack(out, 1)


def nll(logits, ids):
    lp = torch.log_softmax(logits[:-1].double(), -1)
    return -lp.gather(1, ids[1:, None]).squeeze(1)


def kl(a, b):
    return torch.nn.functional.kl_div(torch.log_softmax(a.double(), -1),
                                      torch.log_softmax(b.double(), -1),
                                      log_target=True, reduction="none").sum(-1).mean().item()


def main():
    fp8_dir, bf_dir = sys.argv[1], sys.argv[2]
    LONG = int(sys.argv[3]) if len(sys.argv) > 3 else 1536
    tok = AutoTokenizer.from_pretrained(bf_dir)
    seqs = [torch.tensor(tok(t).input_ids) for t in TEXTS]
    longs = torch.tensor(long_ids(tok, LONG))
    L = min(len(s) for s in seqs)
    print(f"node {os.uname().nodename}; {len(seqs)} passages {[len(s) for s in seqs]} tokens, "
          f"long text {LONG} tokens")
    t0 = time.time()
    W = R.Weights(fp8_dir, "fp8")
    allseq = seqs + [longs]
    ref_f8 = [ref_logits(W, s) for s in allseq]
    PK = Packed(W, default_chunks())
    del W
    torch.cuda.empty_cache()
    eng1 = Engine(PK, 1, maxlen=LONG + 64)
    ker1 = [kernel_logits(eng1, s[None])[0] for s in allseq]
    del eng1
    engM = Engine(PK, len(seqs), maxlen=max(len(s) for s in seqs) + 64)
    kerM = kernel_logits(engM, torch.stack([s[:L] for s in seqs]))
    del engM, PK
    torch.cuda.empty_cache()
    print(f"fp8 runs {time.time() - t0:.0f}s")
    W = R.Weights(bf_dir, "bf16")
    ref_bf = [ref_logits(W, s) for s in allseq]
    del W
    print(f"bf16 runs {time.time() - t0:.0f}s")

    ok = True
    groups = {"passages": [], "long": []}
    for i, s in enumerate(allseq):
        n = {"kernel": nll(ker1[i], s), "fp8": nll(ref_f8[i], s), "bf16": nll(ref_bf[i], s)}
        groups["long" if i == len(seqs) else "passages"].append(n)
        a_f8 = (ker1[i].argmax(-1) == ref_f8[i].argmax(-1)).float().mean().item()
        a_bf = (ker1[i].argmax(-1) == ref_bf[i].argmax(-1)).float().mean().item()
        a_f8bf = (ref_f8[i].argmax(-1) == ref_bf[i].argmax(-1)).float().mean().item()
        name = "long" if i == len(seqs) else f"passage {i}"
        msg = (f"{name}: ppl kernel {n['kernel'].mean().exp():.3f} fp8 {n['fp8'].mean().exp():.3f} "
               f"bf16 {n['bf16'].mean().exp():.3f} | top1 kernel=fp8 {a_f8:.3f} kernel=bf16 "
               f"{a_bf:.3f} fp8=bf16 {a_f8bf:.3f} | KL(bf16||kernel) {kl(ker1[i], ref_bf[i]):.4f} "
               f"KL(bf16||fp8) {kl(ref_f8[i], ref_bf[i]):.4f}")
        if i < len(seqs):
            same = torch.equal(kerM[i], ker1[i][:L])
            ok &= same
            msg += f" | batch-invariant {same}"
        if i == len(seqs):
            for lo, hi in ((0, 1024), (1024, LONG)):
                pk = n["kernel"][lo:hi - 1].mean().exp()
                pf = n["fp8"][lo:hi - 1].mean().exp()
                pb = n["bf16"][lo:hi - 1].mean().exp()
                msg += f"\n  positions {lo}..{hi - 1}: ppl kernel {pk:.3f} fp8 {pf:.3f} bf16 {pb:.3f}"
        print(msg)
    for gname, ns in groups.items():
        ppl = {k: torch.cat([n[k] for n in ns]).mean().exp().item() for k in ns[0]}
        print(f"{gname}: ppl kernel {ppl['kernel']:.3f} fp8 {ppl['fp8']:.3f} bf16 {ppl['bf16']:.3f}")
        # the kernel's fp8 activations may cost a little: allow 2 % over the reference
        # on the same FP8 weights with bf16 GEMV inputs
        ok &= ppl["kernel"] <= 1.02 * ppl["fp8"]
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
