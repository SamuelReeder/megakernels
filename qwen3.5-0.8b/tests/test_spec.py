"""Speculative decoding with the MTP head must reproduce plain decoding bit for bit.

A speculative cycle runs the pending token and the draft through the same per-token arithmetic
as two plain steps, so its ids (greedy or sampled with the same seed) and even its logits are
identical, whatever the drafts are. The drafts themselves are checked against the PyTorch MTP.
"""

import pytest
import torch

from qwen_mk import weights
from qwen_mk.engine import Megakernel

PROMPTS = [
    "Explain how a CPU cache works to a high-school student.",
    "Write a Python function that checks whether a string is a palindrome, with tests.",
    "Write a short poem about autumn in a city.",
]


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(weights.snapshot_path())


@pytest.fixture(scope="module", params=["bf16", "int8"])
def mk(request):
    return Megakernel(max_seq=1024, quant=request.param)


def chat_ids(tok, prompt):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    return tok(text, add_special_tokens=False).input_ids


def stops(tok):
    return [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]


@pytest.mark.parametrize("temperature,chunk", [(0.0, 256), (0.0, 16), (0.8, 16)])
def test_spec_matches_plain(mk, tok, temperature, chunk):
    for seed, prompt in enumerate(PROMPTS):
        ids = chat_ids(tok, prompt)
        kw = dict(max_new_tokens=96, temperature=temperature, seed=seed, stop_ids=stops(tok), chunk=chunk)
        mk.reset()
        want = list(mk.generate(ids, **kw))
        mk.reset()
        got = list(mk.generate(ids, spec=True, **kw))
        assert got == want, prompt


def test_verify_logits_bitwise(mk, tok):
    ids = chat_ids(tok, PROMPTS[0])
    mk.reset()
    last = mk.prefill(ids)
    p0 = mk.pos
    want, logits = mk.run([last], n_generate=48, return_logits=True)
    mk.reset()
    mk.prefill(ids)
    got, _ = mk.run_spec(last, last, 48, return_logits=True)
    assert got == want
    p = p0
    for c, (n_new, _) in enumerate(mk.last_cycles):
        assert torch.equal(mk.last_logits[c, 0], logits[p - p0])
        if n_new == 2:
            assert torch.equal(mk.last_logits[c, 1], logits[p + 1 - p0])
        p += n_new
    assert p == mk.pos


@pytest.mark.parametrize("n", [1, 2, 7])
def test_limit_and_continuation(mk, tok, n):
    """Odd limits cut an accepted cycle short; the state must still continue exactly."""
    ids = chat_ids(tok, PROMPTS[1])
    mk.reset()
    last = mk.prefill(ids)
    a = mk.run([last], n_generate=n)
    a += mk.run([a[-1]], n_generate=40)
    mk.reset()
    mk.prefill(ids)
    b, draft = mk.run_spec(last, last, n)
    assert len(b) == n
    more, _ = mk.run_spec(b[-1], draft, 40)
    assert b + more == a


def test_stop_id(mk, tok):
    ids = chat_ids(tok, PROMPTS[0])
    mk.reset()
    last = mk.prefill(ids)
    free = mk.run([last], n_generate=64)
    for stop in free[5:20]:  # stop on tokens that land on either slot of a cycle
        mk.reset()
        mk.prefill(ids)
        want = mk.run([last], n_generate=64, stop_ids=[stop])
        want_pos = mk.pos
        want += mk.run([want[-1]], n_generate=16)
        mk.reset()
        mk.prefill(ids)
        got, draft = mk.run_spec(last, last, 64, stop_ids=[stop])
        assert got[-1] == stop and mk.pos == want_pos
        got += mk.run_spec(got[-1], draft, 16)[0]
        assert got == want


def test_drafts_match_reference_mtp(mk, tok):
    """The kernel's drafts equal the PyTorch MTP head's argmax over HF hidden states, up to near-ties."""
    if mk.quant != "bf16":
        pytest.skip("reference comparison is meaningful for bf16 weights")
    from transformers import AutoModelForCausalLM

    from qwen_mk.mtp import MTP, load_mtp

    ref = MTP(mk.cfg, load_mtp(), mk.embed, mk.max_seq)  # its own cache: mk.mtp shares the kernel's
    hf = AutoModelForCausalLM.from_pretrained(weights.snapshot_path(), dtype=torch.bfloat16).cuda().eval()
    agree = total = 0
    for prompt in PROMPTS:
        ids = chat_ids(tok, prompt)
        mk.reset()
        last = mk.prefill(ids)
        out, _ = mk.run_spec(last, last, 128, stop_ids=stops(tok))
        seq = ids + out
        with torch.no_grad():
            h = hf(torch.tensor([seq], device="cuda"), output_hidden_states=True, logits_to_keep=1).hidden_states[-1]
        logits = ref.forward(h[0, :-1].float(), torch.tensor(seq[1:], device="cuda"), 0)
        if mk.draft_ids is not None:  # the draft head only scores the draft vocabulary
            guess = mk.draft_ids[logits[:, mk.draft_ids.long()].argmax(-1)].tolist()
        else:
            guess = logits.argmax(-1).tolist()
        p = len(ids) - 1
        for c, (n_new, draft) in enumerate(mk.last_cycles):
            if c > 0:  # cycle c's draft came from the MTP at position p - 1, guessing x_{p+1}
                agree += draft == guess[p - 1]
                total += 1
            p += n_new
    assert agree / total > 0.97, (agree, total)
