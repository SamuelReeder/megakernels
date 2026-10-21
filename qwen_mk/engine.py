"""Python driver for the decode megakernel: packs weights, owns state, launches steps."""

import torch

from . import _ext, weights
from .reference import rope_tables

LAYER_FIELDS = 17
# Steps per launch. Windows' GPU watchdog (TDR) resets kernels that run ~2 s; at ~2 ms/step
# this keeps each launch well under it.
MAX_STEPS_PER_LAUNCH = 256
# How far (bytes per SM) the producer prefetches weights into L2 ahead of its smem ring.
PF_BYTES = 256 * 1024


class Megakernel:
    def __init__(self, model=weights.DEFAULT_MODEL, max_seq=8192, device="cuda", pf_bytes=PF_BYTES):
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
                keep(gu),
                keep(w[m + "mlp.down_proj.weight"]),
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
                common[0], common[1] = keep(w_in), keep(w[p + "out_proj.weight"])
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
                common[0], common[1] = keep(w_in), keep(w[p + "o_proj.weight"])
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
             *[keep(t) for t in self.ll], keep(part), keep(self.attn_cnt)]
        self.embed = w["embed_tokens.weight"]
        del w
        self.ext.set_params(layer_ptrs, g, max_seq)
        self.weight_bytes = self._weight_bytes()
        self.reset()

    def _weight_bytes(self):
        """Bytes of weights streamed per decode step (excluding the LM head)."""
        c = self.cfg
        per_layer = {
            "linear_attention": 8224 * 1024 + 1024 * 2048,
            "full_attention": 5120 * 1024 + 1024 * 2048,
        }
        mlp = 3 * c.intermediate * c.hidden
        body = sum(per_layer[t] + mlp for t in c.layer_types) * 2
        return body, c.vocab * c.hidden * 2

    def reset(self):
        """Start a new sequence (the KV cache does not need clearing: positions are rewritten)."""
        for t in self.conv_hist + self.state + self.ll:
            t.zero_()
        self.attn_cnt.zero_()
        self.epoch = 1  # LL buffers are zeroed, so epoch 0 never looks ready
        self.pos = 0

    @torch.no_grad()
    def run(self, tokens, n_generate=0, temperature=0.0, seed=0, return_logits=False, prof=None, prof_step=0):
        """Feed `tokens` (prompt) then generate `n_generate` more tokens in one launch.

        Returns the generated token ids (the first one predicted after the last prompt token,
        so `n_generate` ids), and optionally the logits for each produced token.
        """
        tokens = list(tokens)
        assert tokens, "need at least one input token"
        n_forced = len(tokens)
        n_steps = n_forced + max(n_generate, 1) - 1
        if self.pos + n_steps > self.max_seq:
            raise ValueError(f"sequence would exceed max_seq={self.max_seq}")
        forced = torch.tensor(tokens, dtype=torch.int32, device=self.device)
        out = torch.zeros(n_steps, dtype=torch.int64, device=self.device)
        logits = torch.empty(n_steps, self.cfg.vocab, device=self.device) if return_logits else None
        self.ext.launch(forced.data_ptr(), n_forced, n_steps, self.pos, self.epoch, out.data_ptr(),
                        logits.data_ptr() if logits is not None else 0, prof.data_ptr() if prof is not None else 0,
                        prof_step, 1.0 / temperature if temperature > 0 else 0.0, seed, self.grid, self.pf_bytes)
        n_lm = n_steps - (n_forced - 1)
        self.epoch += n_steps * self.cfg.n_layers * self.ext.OPS_PER_LAYER + n_lm
        if self.epoch > 0xFFFF0000:
            raise RuntimeError("epoch counter exhausted; call reset()")
        self.pos += n_steps
        ids = (0xFFFFFFFF - (out[n_forced - 1:] & 0xFFFFFFFF)).tolist()
        if return_logits:
            return ids, logits[n_forced - 1:]
        return ids

    def prof_buffer(self):
        n = self.grid * self.ext.PROF_MAX_OPS * self.ext.PROF_FIELDS + self.grid * self.ext.PROF_FIELDS
        return torch.zeros(n, dtype=torch.int64, device=self.device)

    def generate(self, prompt_ids, max_new_tokens=256, temperature=0.0, seed=0, stop_ids=(), chunk=16):
        """Yield generated token ids, launching `chunk` decode steps at a time."""
        produced = 0
        ids = self.run(prompt_ids, n_generate=min(chunk, max_new_tokens), temperature=temperature, seed=seed)
        while True:
            for t in ids:
                yield t
                produced += 1
                if t in stop_ids or produced >= max_new_tokens:
                    return
            n = min(chunk, max_new_tokens - produced, MAX_STEPS_PER_LAUNCH)
            # The last id was predicted but not yet fed; feed it and keep going.
            ids = self.run([ids[-1]], n_generate=n, temperature=temperature, seed=seed)
