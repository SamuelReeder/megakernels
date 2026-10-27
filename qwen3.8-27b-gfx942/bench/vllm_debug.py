"""Debug: at each decode step inside vLLM, run the first L layers (+ final norm) through
vLLM's own modules and through the kernel on the same caches, and compare.

Engine in-process, eager (no CUDA graphs), a small cache so it can be snapshotted.
Usage: VLLM_ENABLE_V1_MULTIPROCESSING=0 python bench/vllm_debug.py MODEL_DIR [LAYERS...]
"""
import os
import sys

import torch

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.models.qwen3_5.amd.specialized_decode.runtime import decode as RD  # noqa: E402

LAYERS = [int(x) for x in sys.argv[2:]] or [1, 3, 4, 64]
orig_run = RD.Qwen35Gfx942Decode.run
seen = {"n": 0}


def snapshot(dec):
    return {n: (tuple(t.clone() for t in c) if isinstance(c, tuple) else c.clone())
            for n, c in dec.bound.items()}


def restore(dec, snap):
    for n, c in dec.bound.items():
        if isinstance(c, tuple):
            for t, s in zip(c, snap[n]):
                t.copy_(s)
        else:
            c.copy_(snap[n])


def vllm_layers(dec, input_ids, positions, nl):
    m = dec.model
    h = m.embed_input_ids(input_ids)
    res = None
    for layer in m.layers[:nl]:
        h, res = layer(positions=positions, hidden_states=h, residual=res)
    h, _ = m.norm(h, res)
    return h


def kernel_run(self, step, input_ids, nl):
    """Decode.run with the first nl layers only (the launch's layer count)."""
    return self.launch(step.kernel_width, self.stage(step, input_ids, None), nl)


def run(self, step, input_ids, inputs_embeds, **kw):
    seen["n"] += 1
    if seen["n"] <= 3:
        W = step.kernel_width
        snap = snapshot(self)
        for nl in LAYERS:
            restore(self, snap)
            ref = vllm_layers(self, input_ids[:W], step.positions[..., :W], nl).float()
            restore(self, snap)
            out = kernel_run(self, step, input_ids, nl).float()
            cos = torch.nn.functional.cosine_similarity(out, ref, dim=-1)
            print(f"step {seen['n']} layers {nl}: cos per row {[round(c, 5) for c in cos.tolist()]} "
                  f"max abs {(out - ref).abs().max().item():.3g} ref norm {ref.norm(dim=-1).tolist()}",
                  flush=True)
        restore(self, snap)
    return orig_run(self, step, input_ids, inputs_embeds, **kw)


RD.Qwen35Gfx942Decode.run = run
_bind = RD.Qwen35Gfx942Decode.bind_caches


def bind(self, caches):
    self.bound = dict(caches)
    return _bind(self, caches)


RD.Qwen35Gfx942Decode.bind_caches = bind

llm = LLM(model=sys.argv[1], max_model_len=2048, language_model_only=True, enable_prefix_caching=False,
          enforce_eager=True, num_gpu_blocks_override=16, max_num_seqs=2, specialized_decode="monokernel",
          attention_backend=os.environ.get("MK_ATTN") or None)
out = llm.generate(["The quick brown fox jumps over the lazy dog. Explain why this sentence is famous."],
                   SamplingParams(temperature=0.0, max_tokens=4))
print(repr(out[0].outputs[0].text))
