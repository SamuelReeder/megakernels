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
    if seen["n"] == 1:
        sfc = self.vllm_config.compilation_config.static_forward_context
        layer = sfc[self.attn_names[0]]
        print("attention:", layer.attn_backend.get_name(), "interleaved V:",
              getattr(layer.impl, "_use_interleaved_v_cache", None), flush=True)
    if seen["n"] <= 3:
        W = step.kernel_width
        snap = snapshot(self)
        # the kernel's KV addressing against vLLM's own split of layer 3's cache
        from vllm.models.qwen3_5.amd.specialized_decode.spec import attention_backend, kv_geometry
        from vllm.v1.attention.ops.paged_attn import PagedAttention
        name = self.attn_names[0]
        kv = self.bound[name]
        sfc = self.vllm_config.compilation_config.static_forward_context
        geo = kv_geometry(kv, attention_backend(sfc[name]))
        print("kv", tuple(kv.shape), kv.stride(), "geometry", geo, flush=True)
        if geo[5]:
            B, s0, s1, s2, sv, _ = geo
            flat = torch.as_strided(kv, (kv.numel() * 2,), (1,))
            kc, vc = PagedAttention.split_kv_cache(kv.transpose(0, 1), 4, 256)
            md = __import__("vllm.forward_context", fromlist=["x"]).get_forward_context().attn_metadata
            blk = int(md[self.attn_groups[0]].block_table[0, 0])
            ok_k = ok_v = True
            for h, p, d in ((0, 0, 0), (1, 3, 9), (3, 10, 255), (2, 5, 130)):
                k1 = flat[blk * s0 + h * s1 + (d // 8) * 8 * B + p * 8 + d % 8]
                v1 = flat[blk * s0 + sv + h * s1 + d * B + p]
                ok_k &= bool(k1 == kc[blk, h, d // 8, p, d % 8])
                ok_v &= bool(v1 == vc[blk, h, d, p])
            print("block", blk, "K formula ok", ok_k, "V formula ok", ok_v,
                  "K[0,0,:4]", kc[blk, 0, 0, 0, :4].tolist(), flush=True)
        # layer 3 (the first attention layer): o_proj's input, from vLLM and the kernel
        restore(self, snap)
        cap = {}
        hk = self.model.layers[3].self_attn.o_proj.register_forward_pre_hook(
            lambda mod, args: cap.update(x=args[0].detach().float().clone()))
        at = self.model.layers[3].self_attn.attn

        def grab(mod, args, kwargs, out):
            a = list(args) + [kwargs.get(k) for k in ("query", "key", "value") if k in kwargs]
            cap.update(q=a[0].detach().float().clone(), o=out.detach().float().clone())
        ha = at.register_forward_hook(grab, with_kwargs=True)
        vllm_layers(self, input_ids[:W], step.positions[..., :W], 4)
        hk.remove()
        ha.remove()
        # torch oracle for row 0 over vLLM's cache (which now holds this step's k, v)
        kc, vc = PagedAttention.split_kv_cache(kv.transpose(0, 1), 4, 256) if geo[5] else (None, None)
        if geo[5]:
            md = __import__("vllm.forward_context", fromlist=["x"]).get_forward_context().attn_metadata
            btr = md[self.attn_groups[0]].block_table[0]
            pos = int(step.positions[..., 0].reshape(-1)[0])
            B = geo[0]
            ps = torch.arange(pos + 1, device=kv.device)
            bl = btr[ps // B].long()
            K = kc[bl, :, :, ps % B, :].float().reshape(pos + 1, 4, 256)  # [p, h, 32, 8]
            V = vc[bl, :, :, ps % B].float()  # [p, h, 256]
            q = cap["q"][0].view(24, 256)
            sc = torch.einsum("hd,phd->hp", q, K.repeat_interleave(6, 1)) * at.impl.scale
            o = torch.einsum("hp,phd->hd", sc.softmax(-1), V.repeat_interleave(6, 1))
            vo = cap["o"][0].view(24, 256)
            oc = torch.nn.functional.cosine_similarity(o, vo, dim=-1)
            print(f"step {seen['n']} pos {pos} vLLM attn vs torch oracle cos min {oc.min().item():.4f}",
                  flush=True)
        kvA = kv.clone()
        restore(self, snap)
        kernel_run(self, step, input_ids, 4)
        bl0 = int(btr[0])
        kA, vA = PagedAttention.split_kv_cache(kvA.transpose(0, 1), 4, 256)
        kB, vB = PagedAttention.split_kv_cache(kv.transpose(0, 1), 4, 256)
        kS, vS = PagedAttention.split_kv_cache(snap[name].transpose(0, 1), 4, 256)
        for p in (0, pos - 1, pos):
            print(f"step {seen['n']} p{p}: K snap==vLLM {torch.equal(kS[bl0, :, :, p], kA[bl0, :, :, p])}"
                  f" kernel==vLLM K {torch.equal(kB[bl0, :, :, p], kA[bl0, :, :, p])}"
                  f" V {torch.equal(vB[bl0, :, :, p], vA[bl0, :, :, p])}"
                  f" K maxdiff {(kB[bl0, :, :, p].float() - kA[bl0, :, :, p].float()).abs().max().item():.3g}",
                  flush=True)
        def eqp(a, b):  # [h, ..., p] -> per position: equal (NaN == NaN)
            a, b = a.float(), b.float()
            e = (a == b) | (a.isnan() & b.isnan())
            return e.flatten(0, -2).all(0) if a.dim() == 3 else e.permute(2, 0, 1, 3).flatten(1).all(1)
        for tag, x, y in (("K", kB, kS), ("V", vB, vS)):
            ch = (~eqp(x[bl0], y[bl0])).nonzero().flatten().tolist()
            print(f"step {seen['n']} kernel changed {tag} positions {ch[:12]}..{ch[-4:]} (n={len(ch)})", flush=True)
            if tag == "K" and ch:
                e = (x[bl0].float() == y[bl0].float()) | (x[bl0].float().isnan() & y[bl0].float().isnan())
                hg = (~e).any(-1).any(-1).nonzero().tolist()  # (h, dgroup)
                print("   changed (head, dgroup):", hg[:40], "values", x[bl0, hg[0][0], hg[0][1], ch[0]].tolist(),
                      "was", y[bl0, hg[0][0], hg[0][1], ch[0]].tolist(), flush=True)
        for tag, x, y in (("K", kA, kS), ("V", vA, vS)):
            ch = (~eqp(x[bl0], y[bl0])).nonzero().flatten().tolist()
            print(f"step {seen['n']} vLLM changed {tag} positions {ch[:12]} (n={len(ch)})", flush=True)
        print("vLLM K h0 p pos d0..7", kA[bl0, 0, 0, pos].tolist(), "kernel", kB[bl0, 0, 0, pos].tolist(),
              "snap", kS[bl0, 0, 0, pos].tolist(), flush=True)
        ky = self.y2[:W, : 24 * 256].float()
        rx = cap["x"][:W]
        per = torch.nn.functional.cosine_similarity(ky.view(W, 24, 256), rx.view(W, 24, 256), dim=-1)
        print(f"step {seen['n']} attn out cos per q head {[round(c, 3) for c in per[0].tolist()]}", flush=True)
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
