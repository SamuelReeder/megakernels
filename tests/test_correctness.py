"""Megakernel vs. the fp32 reference decoder (which matches HF transformers)."""

import pytest
import torch

from qwen_mk import weights
from qwen_mk.engine import Megakernel
from qwen_mk.reference import ReferenceModel

PROMPT = "The quick brown fox jumps over the lazy dog. In 1969, Neil Armstrong"


@pytest.fixture(scope="module")
def models():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(weights.snapshot_path())
    mk = Megakernel(max_seq=1024)
    cfg, w = weights.load()
    ref = ReferenceModel(cfg, w, max_seq=1024)
    return tok, mk, ref


def test_prompt_logits(models):
    tok, mk, ref = models
    ids = tok(PROMPT).input_ids
    mk.reset()
    ref.reset()
    worst = 0.0
    for i, t in enumerate(ids):
        out, lg = mk.run([t], return_logits=True)
        r = ref.step(t)
        err = (lg[0] - r).abs().max().item()
        worst = max(worst, err)
        assert lg[0].argmax().item() == r.argmax().item(), f"argmax mismatch at {i}"
    print(f"max |logit diff| over prompt: {worst:.3e}")
    assert worst < 0.1


def test_batched_prompt_matches_stepwise(models):
    """Feeding the whole prompt in one launch must match feeding it token by token."""
    tok, mk, _ = models
    ids = tok(PROMPT).input_ids
    mk.reset()
    a = mk.run(ids, n_generate=24)
    mk.reset()
    b = []
    for t in ids[:-1]:
        mk.run([t])
    b = mk.run([ids[-1]], n_generate=24)
    assert a == b


def test_greedy_continuation(models):
    tok, mk, ref = models
    ids = tok(PROMPT).input_ids
    mk.reset()
    got = mk.run(ids, n_generate=48)
    ref.reset()
    for t in ids:
        lg = ref.step(t)
    want = []
    for _ in range(48):
        nxt = lg.argmax().item()
        want.append(nxt)
        lg = ref.step(nxt)
    print("megakernel:", repr(tok.decode(got)))
    print("reference: ", repr(tok.decode(want)))
    match = next((i for i, (x, y) in enumerate(zip(got, want)) if x != y), len(want))
    assert match >= 40, f"diverged at token {match}"
