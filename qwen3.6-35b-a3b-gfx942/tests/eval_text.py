"""Teacher-forced quality check on real text: megakernel vs the PyTorch reference.

The same token sequence is fed one token per decode step to three systems:
  kernel  the megakernel (FP8 checkpoint, fp8 activations)
  fp8     mk/reference.py, exact mode: the same FP8 weights, fp32 activations
  bf16    mk/reference.py on the BF16 checkpoint (validated against HF transformers by
          tests/ref_vs_hf.py)
For each, the next-token NLL gives a perplexity; the kernel is also compared with
fp8 and bf16 per position (top-1 agreement, KL against bf16). A second pass checks
batch invariance: each passage decoded alone (M = 1) and inside a batch (M =
len(TEXTS), first min-length tokens) must give bitwise-equal kernel logits.

Usage: python tests/eval_text.py FP8_DIR BF16_DIR
"""
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
from mk import reference as R  # noqa: E402
from mk.pack import Packed  # noqa: E402
from mk.runner import Engine, default_chunks  # noqa: E402

TEXTS = [
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the "
    "age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the "
    "season of Light, it was the season of Darkness, it was the spring of hope, it was the winter "
    "of despair, we had everything before us, we had nothing before us, we were all going direct to "
    "Heaven, we were all going direct the other way - in short, the period was so far like the "
    "present period, that some of its noisiest authorities insisted on its being received, for good "
    "or for evil, in the superlative degree of comparison only.",
    "A hash table stores key-value pairs in an array of buckets. A hash function maps each key to a "
    "bucket index. When two keys map to the same bucket, a collision occurs; separate chaining keeps "
    "a linked list per bucket, while open addressing probes other buckets until it finds a free one. "
    "With a good hash function and a load factor below one, lookups, insertions and deletions take "
    "constant time on average. Resizing the table when the load factor grows keeps that guarantee.",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    a, b = 0, 1\n    for _ in "
    "range(n):\n        a, b = b, a + b\n    return a\n\n\nif __name__ == \"__main__\":\n    for i in "
    "range(10):\n        print(i, fibonacci(i))\n",
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must "
    "be in want of a wife. However little known the feelings or views of such a man may be on his first "
    "entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that "
    "he is considered the rightful property of some one or other of their daughters. \"My dear Mr. "
    "Bennet,\" said his lady to him one day, \"have you heard that Netherfield Park is let at last?\" Mr. "
    "Bennet replied that he had not. \"But it is,\" returned she; \"for Mrs. Long has just been here, and "
    "she told me all about it.\" Mr. Bennet made no answer. \"Do you not want to know who has taken it?\" "
    "cried his wife impatiently. \"You want to tell me, and I have no objection to hearing it.\" This was "
    "invitation enough.",
    "The mitochondrion is the site of aerobic respiration in eukaryotic cells. Pyruvate from "
    "glycolysis enters the matrix, where the citric acid cycle oxidizes it to carbon dioxide and "
    "transfers electrons to NADH and FADH2. The electron transport chain in the inner membrane then "
    "pumps protons across the membrane, and ATP synthase uses the resulting gradient to make ATP.",
]


def ref_logits(W, ids):
    """Reference logits for each position of ids [L] (one decode step per token)."""
    st = R.State.new(1, len(ids) + 8)
    out = []
    for t in range(len(ids)):
        out.append(R.step(W, ids[t:t + 1].cuda(), st, emulate=False)[0].float().cpu())
    return torch.stack(out)


def kernel_logits(eng, rows):
    """Kernel logits [M][L][V] for M equal-length token rows (one step per position)."""
    eng.reset()
    L = rows.shape[1]
    out = []
    for t in range(L):
        out.append(eng.step(rows[:, t].cuda()).clone().cpu())
    return torch.stack(out, 1)


def nll(logits, ids):
    lp = torch.log_softmax(logits[:-1].double(), -1)
    return -lp.gather(1, ids[1:, None]).squeeze(1)


def main():
    fp8_dir, bf_dir = sys.argv[1], sys.argv[2]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(bf_dir)
    seqs = [torch.tensor(tok(t).input_ids) for t in TEXTS]
    L = min(len(s) for s in seqs)                     # the batch-invariance run's length
    Lmax = max(len(s) for s in seqs)
    print(f"{len(seqs)} passages, {[len(s) for s in seqs]} tokens")

    t0 = time.time()
    W = R.Weights(fp8_dir, keep_raw=True)
    ref_f8 = [ref_logits(W, s) for s in seqs]
    PK = Packed(W, default_chunks())
    eng1 = Engine(PK, 1, maxlen=Lmax + 64)
    ker1 = [kernel_logits(eng1, s[None])[0] for s in seqs]
    del eng1
    engM = Engine(PK, len(seqs), maxlen=Lmax + 64)
    kerM = kernel_logits(engM, torch.stack([s[:L] for s in seqs]))
    del engM, PK, W
    torch.cuda.empty_cache()
    print(f"fp8 runs {time.time() - t0:.0f}s")
    W = R.Weights(bf_dir)
    ref_bf = [ref_logits(W, s) for s in seqs]
    del W
    print(f"bf16 runs {time.time() - t0:.0f}s")

    def kl(a, b):
        return torch.nn.functional.kl_div(torch.log_softmax(a.double(), -1),
                                          torch.log_softmax(b.double(), -1),
                                          log_target=True, reduction="none").sum(-1).mean().item()

    ok = True
    tot = {"kernel": [], "fp8": [], "bf16": []}
    for i, s in enumerate(seqs):
        n = {"kernel": nll(ker1[i], s), "fp8": nll(ref_f8[i], s), "bf16": nll(ref_bf[i], s)}
        for k in tot:
            tot[k].append(n[k])
        agree_f8 = (ker1[i].argmax(-1) == ref_f8[i].argmax(-1)).float().mean().item()
        agree_bf = (ker1[i].argmax(-1) == ref_bf[i].argmax(-1)).float().mean().item()
        agree_f8bf = (ref_f8[i].argmax(-1) == ref_bf[i].argmax(-1)).float().mean().item()
        same = torch.equal(kerM[i], ker1[i][:L])
        ok &= same
        print(f"passage {i}: ppl kernel {n['kernel'].mean().exp():.3f} fp8 {n['fp8'].mean().exp():.3f} "
              f"bf16 {n['bf16'].mean().exp():.3f} | top1 kernel=fp8 {agree_f8:.3f} kernel=bf16 "
              f"{agree_bf:.3f} fp8=bf16 {agree_f8bf:.3f} | KL(bf16||kernel) {kl(ker1[i], ref_bf[i]):.4f} "
              f"KL(bf16||fp8) {kl(ref_f8[i], ref_bf[i]):.4f} | batch-invariant {same}")
    ppl = {k: torch.cat(v).mean().exp().item() for k, v in tot.items()}
    print(f"all: ppl kernel {ppl['kernel']:.3f} fp8 {ppl['fp8']:.3f} bf16 {ppl['bf16']:.3f}")
    # the kernel's fp8 activations may cost a little: allow 2 % perplexity over the
    # reference on the same FP8 weights with fp32 activations
    ok &= ppl["kernel"] <= 1.02 * ppl["fp8"]
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
