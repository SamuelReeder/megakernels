"""Check mk/reference.py against HF transformers.

Usage: python tests/ref_vs_hf.py BF16_DIR [FP8_DIR]

1. HF transformers runs the BF16 checkpoint in fp32 over a prompt (logits of every
   position); the exact reference (fp32) decodes the same prompt token by token.
   Gate: per-position cosine >= 0.9999 and the same top-1 token at every position.
2. Information only: the emulating reference on the same weights, and (with FP8_DIR) the
   exact and emulating references on the FP8 checkpoint, against the HF logits: cosine,
   top-1 agreement, KL(HF || x).
"""

import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
from mk import reference as R  # noqa: E402

TEXT = (
    "The quick brown fox jumps over the lazy dog. In 1969, the first humans landed on the "
    "Moon. A hash table maps keys to buckets with a hash function; collisions are resolved "
    "by chaining or by open addressing. The capital of France is"
)


def compare(name, x, ref):
    cos = torch.nn.functional.cosine_similarity(x, ref, dim=-1)
    top = (x.argmax(-1) == ref.argmax(-1)).float()
    kl = torch.nn.functional.kl_div(
        x.log_softmax(-1), ref.log_softmax(-1), log_target=True, reduction="none"
    ).sum(-1)
    print(
        f"{name:28s} cos min {cos.min():.6f} mean {cos.mean():.6f} | top1 {top.mean():.4f} "
        f"| KL mean {kl.mean():.2e} max {kl.max():.2e}"
    )
    return cos, top


def ref_logits(path, ids, emulate):
    W = R.Weights(path)
    st = R.State.new(1, len(ids) + 8)
    out = R.prefill(W, [ids], st, emulate=emulate)[:, 0]
    del W, st
    torch.cuda.empty_cache()
    return out


def main():
    bf_dir = sys.argv[1]
    fp8_dir = sys.argv[2] if len(sys.argv) > 2 else None
    from transformers import AutoTokenizer, Qwen3_5MoeForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(bf_dir)
    ids = tok(TEXT).input_ids
    print(f"{len(ids)} tokens")
    t0 = time.time()
    hf = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        bf_dir, dtype=torch.float32, device_map="cuda"
    ).eval()
    with torch.no_grad():
        hf_logits = hf(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
        hf_logits = hf_logits.logits[0].float()
    del hf
    torch.cuda.empty_cache()
    print(f"hf fp32 done {time.time() - t0:.0f}s")

    ex = ref_logits(bf_dir, ids, False)
    print(f"reference done {time.time() - t0:.0f}s")
    cos, top = compare("exact ref (BF16) vs HF", ex, hf_logits)
    bad = (top < 1).nonzero().flatten().tolist()
    for p in bad:
        t2 = hf_logits[p].topk(2).values
        print(f"  pos {p}: top-1 differs; HF top-2 margin {float(t2[0] - t2[1]):.4f}")
    print("HF next:", repr(tok.decode(hf_logits[-1].argmax())), " ref next:",
          repr(tok.decode(ex[-1].argmax())))
    ok = bool(cos.min() >= 0.9999) and not bad

    compare("emulate ref (BF16) vs HF", ref_logits(bf_dir, ids, True), hf_logits)
    if fp8_dir:
        compare("exact ref (FP8) vs HF BF16", ref_logits(fp8_dir, ids, False), hf_logits)
        compare("emulate ref (FP8) vs HF BF16", ref_logits(fp8_dir, ids, True), hf_logits)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
