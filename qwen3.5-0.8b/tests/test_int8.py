"""int8 megakernel vs. the fp32 reference decoder running on the same dequantized weights.

This isolates kernel bugs from quantization error: both sides use identical weights, so
they should agree as closely as the bf16 kernel agrees with the bf16 reference.
"""

import pytest
import torch

from qwen_mk import weights
from qwen_mk.engine import Megakernel, dequantize_q8, quantize_q8
from qwen_mk.reference import ReferenceModel

PROMPT = "The quick brown fox jumps over the lazy dog. In 1969, Neil Armstrong"
GEMV_WEIGHTS = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
                "linear_attn.in_proj_b", "linear_attn.in_proj_a", "linear_attn.out_proj", "self_attn.q_proj",
                "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")


def roundtrip(w):
    return dequantize_q8(quantize_q8(w), w.shape[1])


@pytest.fixture(scope="module")
def models():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(weights.snapshot_path())
    mk = Megakernel(max_seq=1024, quant="int8")
    cfg, w = weights.load()
    for k in list(w):
        if k.startswith("layers.") and k.endswith(".weight") and any(n in k for n in GEMV_WEIGHTS):
            w[k] = roundtrip(w[k])
    w["lm_head.weight"] = roundtrip(w["embed_tokens.weight"])  # tied: lookup stays bf16
    ref = ReferenceModel(cfg, w, max_seq=1024)
    return tok, mk, ref


def test_quantize_roundtrip_error():
    w = torch.randn(64, 1024, device="cuda").to(torch.bfloat16)
    err = (roundtrip(w) - w.float()).abs()
    scale = w.float().reshape(64, -1, 32).abs().amax(-1, keepdim=True) / 127
    assert (err.reshape(64, -1, 32) <= scale * 0.5 + 1e-2 * scale).all()


def test_int8_prompt_logits_match_dequantized_reference(models):
    tok, mk, ref = models
    ids = tok(PROMPT).input_ids
    mk.reset()
    ref.reset()
    _, lg = mk.run(ids, all_logits=True)
    worst = 0.0
    for i, t in enumerate(ids):
        r = ref.step(t)
        worst = max(worst, (lg[i] - r).abs().max().item())
        assert lg[i].argmax().item() == r.argmax().item(), f"argmax mismatch at {i}"
    print(f"int8 max |logit diff| vs dequantized reference: {worst:.3e}")
    assert worst < 0.1


def test_all_logits_matches_stepwise(models):
    tok, mk, _ = models
    ids = tok(PROMPT).input_ids
    mk.reset()
    _, batched = mk.run(ids, all_logits=True)
    mk.reset()
    step = torch.cat([mk.run([t], return_logits=True)[1] for t in ids])
    assert torch.equal(batched, step)  # the kernel is deterministic: no atomics in any reduction
