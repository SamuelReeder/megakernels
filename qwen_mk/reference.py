"""Token-by-token reference decoder in plain PyTorch.

This mirrors the math the megakernel performs, one op at a time, with explicit
recurrent / KV state. It is the oracle for kernel correctness tests and is itself
checked against Hugging Face transformers in tests/test_reference.py.
"""

import torch
import torch.nn.functional as F


def rmsnorm(x, w, eps):
    """Qwen3.5 RMSNorm: weights are stored as offsets from 1."""
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())


def rope_tables(cfg, max_seq, device):
    half = cfg.rotary_dim // 2
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.rotary_dim, 2, dtype=torch.int64).float() / cfg.rotary_dim))
    freqs = torch.arange(max_seq, dtype=torch.float32)[:, None] * inv_freq[None, :]
    assert freqs.shape[1] == half
    return freqs.cos().to(device), freqs.sin().to(device)


def apply_rope(x, cos, sin, rotary_dim):
    """Half-split (NeoX-style) rotation of the first `rotary_dim` dims of each head."""
    half = rotary_dim // 2
    x1, x2, rest = x[..., :half], x[..., half:rotary_dim], x[..., rotary_dim:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest], dim=-1)


class ReferenceModel:
    def __init__(self, cfg, weights, max_seq=4096, device="cuda", compute_dtype=torch.float32):
        self.cfg = cfg
        self.dt = compute_dtype
        self.w = {k: v.to(device=device, dtype=compute_dtype) for k, v in weights.items()}
        self.device = device
        self.max_seq = max_seq
        self.cos, self.sin = rope_tables(cfg, max_seq, device)
        self.reset()

    def reset(self):
        c, d = self.cfg, self.device
        self.pos = 0
        self.conv_hist = {}
        self.state = {}
        self.k_cache = {}
        self.v_cache = {}
        conv_dim = 2 * c.lin_heads * c.lin_k_dim + c.lin_heads * c.lin_v_dim
        for i, t in enumerate(c.layer_types):
            if t == "linear_attention":
                self.conv_hist[i] = torch.zeros(conv_dim, c.conv_k - 1, device=d)
                self.state[i] = torch.zeros(c.lin_heads, c.lin_k_dim, c.lin_v_dim, device=d)
            else:
                self.k_cache[i] = torch.zeros(c.n_kv_heads, self.max_seq, c.head_dim, device=d)
                self.v_cache[i] = torch.zeros(c.n_kv_heads, self.max_seq, c.head_dim, device=d)

    def _lin(self, name, x):
        return self.w[name].float() @ x

    def _gdn(self, i, h):
        c, p = self.cfg, f"layers.{i}.linear_attn."
        H, Dk, Dv = c.lin_heads, c.lin_k_dim, c.lin_v_dim
        qkv = self._lin(p + "in_proj_qkv.weight", h)
        z = self._lin(p + "in_proj_z.weight", h).view(H, Dv)
        b = self._lin(p + "in_proj_b.weight", h)
        a = self._lin(p + "in_proj_a.weight", h)

        win = torch.cat([self.conv_hist[i], qkv[:, None]], dim=1)
        self.conv_hist[i] = win[:, 1:].clone()
        conv = F.silu((win * self.w[p + "conv1d.weight"].float()[:, 0, :]).sum(-1))
        q, k, v = torch.split(conv, [H * Dk, H * Dk, H * Dv])
        q = F.normalize(q.view(H, Dk), dim=-1, eps=1e-6) * Dk**-0.5
        k = F.normalize(k.view(H, Dk), dim=-1, eps=1e-6)
        v = v.view(H, Dv)

        beta = torch.sigmoid(b)
        g = -self.w[p + "A_log"].float().exp() * F.softplus(a + self.w[p + "dt_bias"].float())

        S = self.state[i] * g.exp()[:, None, None]
        kv_mem = torch.einsum("hkv,hk->hv", S, k)
        delta = (v - kv_mem) * beta[:, None]
        S = S + k[:, :, None] * delta[:, None, :]
        self.state[i] = S
        o = torch.einsum("hkv,hk->hv", S, q)

        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + c.eps) * self.w[p + "norm.weight"].float() * F.silu(z)
        return self._lin(p + "out_proj.weight", o.flatten())

    def _attn(self, i, h):
        c, p = self.cfg, f"layers.{i}.self_attn."
        Hq, Hkv, D = c.n_heads, c.n_kv_heads, c.head_dim
        qg = self._lin(p + "q_proj.weight", h).view(Hq, 2 * D)
        q, gate = qg[:, :D], qg[:, D:]
        k = self._lin(p + "k_proj.weight", h).view(Hkv, D)
        v = self._lin(p + "v_proj.weight", h).view(Hkv, D)
        q = rmsnorm(q, self.w[p + "q_norm.weight"], c.eps)
        k = rmsnorm(k, self.w[p + "k_norm.weight"], c.eps)
        cos, sin = self.cos[self.pos], self.sin[self.pos]
        q = apply_rope(q, cos, sin, c.rotary_dim)
        k = apply_rope(k, cos, sin, c.rotary_dim)
        self.k_cache[i][:, self.pos] = k
        self.v_cache[i][:, self.pos] = v
        n = self.pos + 1
        K = self.k_cache[i][:, :n].repeat_interleave(Hq // Hkv, dim=0)
        V = self.v_cache[i][:, :n].repeat_interleave(Hq // Hkv, dim=0)
        scores = torch.einsum("hd,htd->ht", q, K) * D**-0.5
        o = torch.einsum("ht,htd->hd", scores.softmax(-1), V)
        o = o * torch.sigmoid(gate)
        return self._lin(p + "o_proj.weight", o.flatten())

    @torch.no_grad()
    def step(self, token):
        """Run one token and return float32 logits for the next position."""
        c = self.cfg
        x = self.w["embed_tokens.weight"][token].float()
        for i, t in enumerate(c.layer_types):
            h = rmsnorm(x, self.w[f"layers.{i}.input_layernorm.weight"], c.eps)
            x = x + (self._gdn(i, h) if t == "linear_attention" else self._attn(i, h))
            h = rmsnorm(x, self.w[f"layers.{i}.post_attention_layernorm.weight"], c.eps)
            m = f"layers.{i}.mlp."
            act = F.silu(self._lin(m + "gate_proj.weight", h)) * self._lin(m + "up_proj.weight", h)
            x = x + self._lin(m + "down_proj.weight", act)
        self.pos += 1
        h = rmsnorm(x, self.w["norm.weight"], c.eps)
        return self.w["embed_tokens.weight"].float() @ h
