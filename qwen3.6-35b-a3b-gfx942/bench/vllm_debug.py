"""Debug the kernel inside vLLM (engine in-process, eager, a small cache).

1. Weights: vLLM's loaded FP8 tensors against the checkpoint's bits (the repack assumes
   vLLM's e4m3fnuz form: the e4m3fn bits with 0x80 -> 0, and the block scales x 2).
2. At the first decode steps: the first L layers (+ final norm) through vLLM's own
   modules and through the kernel on the same caches (restored in between); cosine per
   row of the normed hidden states.
Usage: python bench/vllm_debug.py MODEL_DIR [LAYERS...]   (MK_ATTN picks the backend)
"""
import json
import os
import sys

import torch
from safetensors import safe_open

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.models.qwen3_5_moe.amd.specialized_decode.runtime import decode as RD  # noqa: E402

MODEL = sys.argv[1]
LAYERS = [int(x) for x in sys.argv[2:]] or [1, 3, 4, 8, 40]
P = "model.language_model.layers."
orig_run = RD.Qwen35MoeGfx942Decode.run
seen = {"n": 0}


def ckpt(name):
    ix = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    with safe_open(f"{MODEL}/{ix[name]}", "pt") as f:
        return f.get_tensor(name)


def fnuz(w):
    u8 = w.view(torch.uint8)
    return torch.where(u8 == 0x80, torch.zeros_like(u8), u8)


def check_weights(dec):
    """vLLM's tensors (slices of fused ones) against the checkpoint."""
    L = dec.model.layers
    e = L[0].mlp.experts.routed_experts
    cases = [
        ("layer 0 in_proj_qkvz[:8192] = in_proj_qkv", L[0].linear_attn.in_proj_qkvz, 0,
         P + "0.linear_attn.in_proj_qkv"),
        ("layer 0 in_proj_qkvz[8192:] = in_proj_z", L[0].linear_attn.in_proj_qkvz, 8192,
         P + "0.linear_attn.in_proj_z"),
        ("layer 3 qkv_proj[8192:8704] = k_proj", L[3].self_attn.qkv_proj, 8192,
         P + "3.self_attn.k_proj"),
        ("layer 0 shared gate_up[512:] = up_proj", L[0].mlp.shared_expert.gate_up_proj, 512,
         P + "0.mlp.shared_expert.up_proj"),
    ]
    for what, mod, r0, name in cases:
        w, s = ckpt(name + ".weight"), ckpt(name + ".weight_scale_inv")
        vw = mod.weight[r0: r0 + w.shape[0]].view(torch.uint8).cpu()
        vs = mod.weight_scale_inv[r0 // 128: (r0 + w.shape[0]) // 128].cpu()
        print(f"{what}: dtype {mod.weight.dtype}, bits equal {torch.equal(vw, fnuz(w))}, "
              f"scale == 2 x ckpt {torch.equal(vs, 2 * s.float())}", flush=True)
    for x in (0, 77, 255):
        for pj, half in (("gate_proj", 0), ("up_proj", 512)):
            w, s = ckpt(f"{P}0.mlp.experts.{x}.{pj}.weight"), ckpt(f"{P}0.mlp.experts.{x}.{pj}.weight_scale_inv")
            vw = e.w13_weight[x, half: half + 512].view(torch.uint8).cpu()
            vs = e.w13_weight_scale_inv[x, half // 128: half // 128 + 4].cpu()
            print(f"layer 0 expert {x} w13 {pj}: bits equal {torch.equal(vw, fnuz(w))}, "
                  f"scale == 2 x ckpt {torch.equal(vs, 2 * s.float())}", flush=True)
        w, s = ckpt(f"{P}0.mlp.experts.{x}.down_proj.weight"), ckpt(f"{P}0.mlp.experts.{x}.down_proj.weight_scale_inv")
        print(f"layer 0 expert {x} w2: bits equal {torch.equal(e.w2_weight[x].view(torch.uint8).cpu(), fnuz(w))}, "
              f"scale == 2 x ckpt {torch.equal(e.w2_weight_scale_inv[x].cpu(), 2 * s.float())}", flush=True)
    print("router", L[0].mlp.gate.weight.dtype, "equal",
          torch.equal(L[0].mlp.gate.weight.cpu(), ckpt(P + "0.mlp.gate.weight")),
          "| in_proj_ba", L[0].linear_attn.in_proj_ba.weight.dtype, "b equal",
          torch.equal(L[0].linear_attn.in_proj_ba.weight[:32].cpu(), ckpt(P + "0.linear_attn.in_proj_b.weight")),
          flush=True)


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


def run(self, step, input_ids, inputs_embeds, **kw):
    seen["n"] += 1
    if seen["n"] == 1:
        sfc = self.vllm_config.compilation_config.static_forward_context
        layer = sfc[self.attn_names[0]]
        kv = self.bound[self.attn_names[0]]
        print("attention:", layer.attn_backend.get_name(), "KV view", tuple(kv.shape),
              kv.stride(), flush=True)
        check_weights(self)
    if seen["n"] <= 3:
        W = step.kernel_width
        snap = snapshot(self)
        for nl in LAYERS:
            restore(self, snap)
            ref = vllm_layers(self, input_ids[:W], step.positions[..., :W], nl).float()
            restore(self, snap)
            out = self.launch(W, self.stage(step, input_ids, None), nl).float()
            cos = torch.nn.functional.cosine_similarity(out, ref, dim=-1)
            print(f"step {seen['n']} layers {nl}: cos per row {[round(c, 5) for c in cos.tolist()]} "
                  f"max abs {(out - ref).abs().max().item():.3g}", flush=True)
        restore(self, snap)
    return orig_run(self, step, input_ids, inputs_embeds, **kw)


RD.Qwen35MoeGfx942Decode.run = run
_bind = RD.Qwen35MoeGfx942Decode.bind_caches


def bind(self, caches):
    self.bound = dict(caches)
    return _bind(self, caches)


RD.Qwen35MoeGfx942Decode.bind_caches = bind

llm = LLM(model=MODEL, max_model_len=2048, language_model_only=True, enable_prefix_caching=False,
          enforce_eager=True, num_gpu_blocks_override=16, max_num_seqs=2, specialized_decode="monokernel",
          attention_backend=os.environ.get("MK_ATTN") or None)
out = llm.generate(["The quick brown fox jumps over the lazy dog. Explain why this sentence is famous."],
                   SamplingParams(temperature=0.0, max_tokens=4))
print(repr(out[0].outputs[0].text))
