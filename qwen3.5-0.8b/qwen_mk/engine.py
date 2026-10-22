"""Python driver for the decode megakernel: packs weights, owns state, launches steps."""

import torch

from . import _ext, weights
from .reference import rope_tables

LAYER_FIELDS = 17
# Steps per launch. Windows' GPU watchdog (TDR) resets kernels that run ~2 s; at ~2 ms/step
# this keeps each launch well under it.
MAX_STEPS_PER_LAUNCH = 256
# How far (bytes per SM) the producer prefetches weights into L2 ahead of its smem ring.
PF_BYTES = 32 * 1024
Q8_GROUP = 32


def quantize_q8(w, group=Q8_GROUP):
    """[R, K] weights -> [R, K + 2K/group] bytes: (q + 128) as uint8, then bf16 absmax scales.

    Same scheme as llama.cpp's Q8_0 (symmetric int8, one scale per 32 weights), laid out so
    each row stays one contiguous span the kernel can bulk-copy.
    """
    R, K = w.shape
    out = torch.empty(R, K + 2 * K // group, dtype=torch.uint8, device=w.device)
    for r in range(0, R, 16384):  # in slices: the LM head would need ~3 GB of fp32 temporaries
        g = w[r: r + 16384].float().reshape(-1, K // group, group)
        scale = (g.abs().amax(-1) / 127).clamp_min(1e-12).to(torch.bfloat16)
        q = torch.round(g / scale.float()[..., None]).clamp(-128, 127) + 128
        out[r: r + 16384, :K] = q.reshape(-1, K).to(torch.uint8)
        out[r: r + 16384, K:] = scale.view(torch.uint8).reshape(-1, 2 * K // group)
    return out


def dequantize_q8(packed, K, group=Q8_GROUP):
    """Inverse of quantize_q8, in fp32 (used to check the kernel against the reference)."""
    q = packed[:, :K].to(torch.int16) - 128
    scale = packed[:, K:].contiguous().view(torch.bfloat16).float()
    return (q.float().reshape(len(packed), -1, group) * scale[..., None]).reshape(-1, K)


class Megakernel:
    def __init__(self, model=weights.DEFAULT_MODEL, max_seq=8192, device="cuda", pf_bytes=PF_BYTES, quant="bf16"):
        assert quant in ("bf16", "int8"), quant
        self.quant = quant
        self.pf_bytes = pf_bytes
        device = torch.device(device)
        if device.index is None:
            device = torch.device(device.type, torch.cuda.current_device())
        torch.cuda.set_device(device)
        self.ext = _ext.load()
        self.cfg, w = weights.load(model, device)
        cfg = self.cfg
        self.max_seq = max_seq
        self.device = device
        self.grid = torch.cuda.get_device_properties(device).multi_processor_count
        self._keep = []  # every tensor whose pointer the kernel holds

        def keep(t):
            t = t.contiguous()
            self._keep.append(t)
            return t.data_ptr()

        def mat(t):  # a GEMV weight in the kernel's format
            return keep(quantize_q8(t) if quant == "int8" else t)

        f32 = dict(device=device, dtype=torch.float32)
        self.conv_hist, self.state, self.k_cache, self.v_cache = [], [], [], []
        layer_ptrs = []
        dummy = keep(torch.zeros(16, **f32))
        for i, kind in enumerate(cfg.layer_types):
            m = f"layers.{i}."
            gu = torch.stack([w[m + "mlp.gate_proj.weight"], w[m + "mlp.up_proj.weight"]], dim=1).reshape(-1, cfg.hidden)
            common = [
                None,
                None,
                mat(gu),
                mat(w[m + "mlp.down_proj.weight"]),
                keep(w[m + "input_layernorm.weight"]),
                keep(w[m + "post_attention_layernorm.weight"]),
            ]
            if kind == "linear_attention":
                p = m + "linear_attn."
                w_in = torch.cat([w[p + "in_proj_qkv.weight"], w[p + "in_proj_z.weight"],
                                  w[p + "in_proj_b.weight"], w[p + "in_proj_a.weight"]])
                hist = torch.zeros(w[p + "conv1d.weight"].shape[0], 4, **f32)
                st = torch.zeros(cfg.lin_heads, cfg.lin_v_dim, cfg.lin_k_dim, **f32)
                self.conv_hist.append(hist)
                self.state.append(st)
                common[0], common[1] = mat(w_in), mat(w[p + "out_proj.weight"])
                extra = [keep(w[p + "conv1d.weight"].reshape(-1, 4)), keep(w[p + "A_log"].float()),
                         keep(w[p + "dt_bias"]), keep(w[p + "norm.weight"].float()), keep(hist), keep(st),
                         dummy, dummy, dummy, dummy, 0]
            else:
                p = m + "self_attn."
                w_in = torch.cat([w[p + "q_proj.weight"], w[p + "k_proj.weight"], w[p + "v_proj.weight"]])
                kc = torch.zeros(cfg.n_kv_heads, max_seq, cfg.head_dim, device=device, dtype=torch.bfloat16)
                vc = torch.zeros_like(kc)
                self.k_cache.append(kc)
                self.v_cache.append(vc)
                common[0], common[1] = mat(w_in), mat(w[p + "o_proj.weight"])
                extra = [dummy, dummy, dummy, dummy, dummy, dummy,
                         keep(w[p + "q_norm.weight"]), keep(w[p + "k_norm.weight"]), keep(kc), keep(vc), 1]
            layer_ptrs += common + extra
        assert len(layer_ptrs) == LAYER_FIELDS * cfg.n_layers

        cos, sin = rope_tables(cfg, max_seq, device)
        i64 = dict(device=device, dtype=torch.int64)
        # LL activation buffers (see csrc/megakernel.cu): (epoch << 32) | fp32 bits
        self.ll = [torch.zeros(n, **i64) for n in (cfg.hidden, 8224, 2048, 2048, cfg.intermediate, 2 * self.grid)]
        self.attn_cnt = torch.zeros(cfg.n_kv_heads, device=device, dtype=torch.int32)
        part = torch.zeros(cfg.n_kv_heads * self.ext.MAX_SPLITS * 4 * self.ext.PART_STRIDE, **f32)
        g = [keep(w["embed_tokens.weight"]), keep(w["norm.weight"]), keep(cos), keep(sin),
             *[keep(t) for t in self.ll], keep(part), keep(self.attn_cnt), mat(w["embed_tokens.weight"])]
        self.embed = w["embed_tokens.weight"]
        del w
        self.ext.set_params(layer_ptrs, g, max_seq)
        self.weight_bytes = self._weight_bytes()
        self.reset()

    def _weight_bytes(self):
        """Bytes of weights streamed per decode step: (layers, LM head)."""
        c = self.cfg
        per_layer = {
            "linear_attention": 8224 * 1024 + 1024 * 2048,
            "full_attention": 5120 * 1024 + 1024 * 2048,
        }
        mlp = 3 * c.intermediate * c.hidden
        per_weight = 1 + 2 / Q8_GROUP if self.quant == "int8" else 2
        body = sum(per_layer[t] + mlp for t in c.layer_types) * per_weight
        return int(body), int(c.vocab * c.hidden * per_weight)

    def reset(self):
        """Start a new sequence (the KV cache does not need clearing: positions are rewritten)."""
        for t in self.conv_hist + self.state + self.ll:
            t.zero_()
        self.attn_cnt.zero_()
        self.epoch = 1  # LL buffers are zeroed, so epoch 0 never looks ready
        self.pos = 0

    @torch.no_grad()
    def run(self, tokens, n_generate=1, temperature=0.0, seed=0, stop_ids=(), return_logits=False, prof=None,
            prof_step=0, timing=False, all_logits=False):
        """Feed `tokens`, then let the model continue for up to `n_generate` predicted tokens.

        One kernel launch: the prompt tokens are fed one per step (no LM head except on the
        last), and each predicted token is fed back in on the GPU. Returns the predicted ids
        (ending early with a stop id, which is *not* fed into the state), plus their logits if
        `return_logits`. With `all_logits`, logits for every fed token are returned instead
        (for scoring text). With `timing`, `self.last_step_ns` holds GPU time per step.
        """
        tokens = list(tokens)
        assert tokens, "need at least one input token"
        assert n_generate >= 1
        n_forced = len(tokens)
        n_steps = n_forced + n_generate - 1
        if self.pos + n_steps > self.max_seq:
            raise ValueError(f"sequence would exceed max_seq={self.max_seq}")
        dev = self.device
        forced = torch.tensor(tokens, dtype=torch.int32, device=dev)
        stops = torch.tensor(list(stop_ids) or [-1], dtype=torch.int32, device=dev)
        out = torch.zeros(n_steps, dtype=torch.int64, device=dev)
        info = torch.zeros(2, dtype=torch.int32, device=dev)
        ts = torch.zeros(n_steps + 1, dtype=torch.int64, device=dev) if timing else None
        lm_from = 0 if all_logits else n_forced - 1
        logits = torch.empty(n_steps, self.cfg.vocab, device=dev) if return_logits or all_logits else None
        ptr = lambda t: t.data_ptr() if t is not None else 0
        self.ext.launch(forced.data_ptr(), n_forced, n_steps, self.pos, self.epoch, out.data_ptr(), ptr(logits),
                        ptr(prof), prof_step, 1.0 / temperature if temperature > 0 else 0.0, seed, self.grid,
                        self.pf_bytes, stops.data_ptr(), len(stop_ids), info.data_ptr(), ptr(ts),
                        int(self.quant == "int8"), lm_from)
        done, epoch = info.tolist()
        self.pos += done
        self.epoch = epoch & 0xFFFFFFFF
        if self.epoch > 0xFFFF0000:
            raise RuntimeError("epoch counter exhausted; call reset()")
        if timing:
            t = ts[: done + 1].cpu()
            self.last_step_ns = (t[1:] - t[:-1]).tolist()
        ids = (0xFFFFFFFF - (out[n_forced - 1: done] & 0xFFFFFFFF)).tolist()
        if all_logits:
            return ids, logits[:done]
        if return_logits:
            return ids, logits[n_forced - 1: done]
        return ids

    def prefill(self, tokens, chunk=512):
        """Feed all but the last token (in launches short enough for the GPU watchdog)."""
        tokens = list(tokens)
        for i in range(0, len(tokens) - 1, chunk):
            self.run(tokens[i: min(i + chunk, len(tokens) - 1)])
        return tokens[-1]

    def prof_buffer(self):
        n = self.grid * self.ext.PROF_MAX_OPS * self.ext.PROF_FIELDS + self.grid * self.ext.PROF_FIELDS
        return torch.zeros(n, dtype=torch.int64, device=self.device)

    def generate(self, prompt_ids, max_new_tokens=256, temperature=0.0, seed=0, stop_ids=(), chunk=16):
        """Yield generated ids. The state ends exactly after the last fed token, so a
        conversation can continue by passing the next prompt (starting with the stop id)."""
        last = self.prefill(prompt_ids)
        produced = 0
        while produced < max_new_tokens:
            n = min(chunk, max_new_tokens - produced, MAX_STEPS_PER_LAUNCH)
            ids = self.run([last], n_generate=n, temperature=temperature, seed=seed, stop_ids=stop_ids)
            for t in ids:
                yield t
                produced += 1
            if ids and ids[-1] in stop_ids:
                return
            last = ids[-1]
        # max_new_tokens reached: the final predicted id has not been fed yet
        self.pending = last
