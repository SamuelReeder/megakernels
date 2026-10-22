"""Qwen3.5's multi-token-prediction (MTP) head, the draft model for speculative decoding.

The checkpoint ships one extra decoder layer that predicts the token after next. At position i
it takes the main model's final (normed) hidden state h_i and the embedding of the token x_{i+1}
that h_i predicted:

    u = fc([rmsnorm(embed(x_{i+1})), rmsnorm(h_i)])          # 2048 -> 1024
    u = gated full-attention decoder layer(u), RoPE position i
    logits = lm_head(rmsnorm(u))                             # tied embeddings -> guess for x_{i+2}

The layer attends over its own KV cache, one entry per position. This module is the batched
PyTorch version: the oracle for the kernel's MTP ops, and the way the draft's KV cache gets
filled for prompt tokens (the kernel only runs the MTP during speculative generation).
"""

import os

import torch
import torch.nn.functional as F
from safetensors import safe_open

from . import weights
from .reference import apply_rope, rmsnorm, rope_tables


def load_mtp(model=weights.DEFAULT_MODEL, device="cuda"):
    """MTP weights keyed without the `mtp.` prefix, e.g. `layers.0.self_attn.q_proj.weight`."""
    path = weights.snapshot_path(model)
    out = {}
    for name in sorted(os.listdir(path)):
        if name.endswith(".safetensors"):
            with safe_open(os.path.join(path, name), "pt", device="cpu") as f:
                for k in f.keys():
                    if k.startswith("mtp."):
                        out[k[4:]] = f.get_tensor(k).to(device)
    return out


class MTP:
    """Batched fp32 MTP head with its own KV cache (bf16, like the kernel's)."""

    def __init__(self, cfg, w, embed, max_seq, device="cuda", lin=None):
        self.cfg, self.device, self.max_seq = cfg, device, max_seq
        self.w = {k: v.float() for k, v in w.items()}
        self.embed = embed
        # `lin` maps a weight name to the matrix actually used, e.g. dequantized int8 weights
        self.lin = lin or (lambda name: self.w[name])
        self.cos, self.sin = rope_tables(cfg, max_seq, device)
        self.k_cache = torch.zeros(cfg.n_kv_heads, max_seq, cfg.head_dim, device=device, dtype=torch.bfloat16)
        self.v_cache = torch.zeros_like(self.k_cache)

    def _proj(self, name, x):
        return x @ self.lin(name).float().t()

    def layer_input(self, h, next_tokens):
        """u_i = fc([norm(embed(x_{i+1})), norm(h_i)]) for hidden states [n, 1024] (post final norm)."""
        c = self.cfg
        e = rmsnorm(self.embed[next_tokens].float(), self.w["pre_fc_norm_embedding.weight"], c.eps)
        hh = rmsnorm(h.float(), self.w["pre_fc_norm_hidden.weight"], c.eps)
        return self._proj("fc.weight", torch.cat([e, hh], dim=-1))

    def qkv(self, u, positions):
        c, p = self.cfg, "layers.0.self_attn."
        x = rmsnorm(u, self.w["layers.0.input_layernorm.weight"], c.eps)
        n = len(u)
        qg = self._proj(p + "q_proj.weight", x).view(n, c.n_heads, 2 * c.head_dim)
        q, gate = qg[..., :c.head_dim], qg[..., c.head_dim:]
        k = self._proj(p + "k_proj.weight", x).view(n, c.n_kv_heads, c.head_dim)
        v = self._proj(p + "v_proj.weight", x).view(n, c.n_kv_heads, c.head_dim)
        q = rmsnorm(q, self.w[p + "q_norm.weight"], c.eps)
        k = rmsnorm(k, self.w[p + "k_norm.weight"], c.eps)
        cos, sin = self.cos[positions][:, None], self.sin[positions][:, None]
        return apply_rope(q, cos, sin, c.rotary_dim), apply_rope(k, cos, sin, c.rotary_dim), v, gate

    @torch.no_grad()
    def fill_kv(self, h, next_tokens, start):
        """Write cache entries for positions start .. start+n-1 (prompt prefill: K/V only)."""
        positions = torch.arange(start, start + len(h), device=self.device)
        _, k, v, _ = self.qkv(self.layer_input(h, next_tokens), positions)
        self.k_cache[:, positions] = k.transpose(0, 1).to(torch.bfloat16)
        self.v_cache[:, positions] = v.transpose(0, 1).to(torch.bfloat16)

    @torch.no_grad()
    def forward(self, h, next_tokens, start):
        """Logits for x_{i+2} at positions start .. start+n-1, attending causally to the cache."""
        c, p = self.cfg, "layers.0.self_attn."
        n = len(h)
        positions = torch.arange(start, start + n, device=self.device)
        u = self.layer_input(h, next_tokens)
        q, k, v, gate = self.qkv(u, positions)
        self.k_cache[:, positions] = k.transpose(0, 1).to(torch.bfloat16)
        self.v_cache[:, positions] = v.transpose(0, 1).to(torch.bfloat16)
        T = start + n
        K = self.k_cache[:, :T].float().repeat_interleave(c.n_heads // c.n_kv_heads, dim=0)  # [Hq, T, D]
        V = self.v_cache[:, :T].float().repeat_interleave(c.n_heads // c.n_kv_heads, dim=0)
        scores = torch.einsum("nhd,htd->nht", q, K) * c.head_dim ** -0.5
        mask = torch.arange(T, device=self.device)[None, :] > positions[:, None]  # [n, T]
        scores = scores.masked_fill(mask[:, None, :], float("-inf"))
        o = torch.einsum("nht,htd->nhd", scores.softmax(-1), V) * torch.sigmoid(gate)
        u = u + self._proj(p + "o_proj.weight", o.reshape(n, -1))
        x = rmsnorm(u, self.w["layers.0.post_attention_layernorm.weight"], c.eps)
        m = "layers.0.mlp."
        u = u + self._proj(m + "down_proj.weight",
                           F.silu(self._proj(m + "gate_proj.weight", x)) * self._proj(m + "up_proj.weight", x))
        return lm_logits(rmsnorm(u, self.w["norm.weight"], c.eps), self.embed)


def lm_logits(x, embed, chunk=32768):
    """x [n, 1024] @ embed.T in fp32, a vocab chunk at a time (a full fp32 copy would be 1 GB)."""
    return torch.cat([x @ embed[i: i + chunk].float().t() for i in range(0, len(embed), chunk)], dim=-1)
