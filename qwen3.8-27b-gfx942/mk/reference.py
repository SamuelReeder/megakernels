"""PyTorch reference of one Qwen3.8-27B decode step (text model).

The numerics follow docs/DESIGN.md:
- residual stream h: fp32
- RMSNorm in fp32, output rounded to bf16 (GEMV input)
- GEMV: bf16 input x weight (exact bf16 for MXFP4), fp32 accumulation, fp32 output
- GDN conv state: bf16 (the model dtype, as vLLM and HF keep it); SSM state: fp32
- KV cache: bf16
The same module runs prefill by calling `step` once per prompt token.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
from safetensors import safe_open

from .dims import (CONV_DIM, CONV_K, DK, DV, EPS, HD, KEY_DIM, NKH, NKV, NL, NQ, NVH, ROT, THETA,
                   VAL_DIM, is_attn)

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def dequant_mxfp4(w_u8: torch.Tensor, s_u8: torch.Tensor) -> torch.Tensor:
    """[N, K/2] packed e2m1 (low nibble = even k) + [N, K/32] e8m0 -> bf16 [N, K] (exact)."""
    lut = E2M1.to(w_u8.device)
    lo = lut[(w_u8 & 0xF).long()]
    hi = lut[(w_u8 >> 4).long()]
    w = torch.stack([lo, hi], dim=-1).reshape(w_u8.shape[0], -1)
    scale = torch.exp2(s_u8.float() - 127.0).repeat_interleave(32, dim=1)
    return (w * scale).to(torch.bfloat16)


class Weights:
    """Loads text-model weights as bf16 tensors on `device`.

    fmt="mxfp4": amd/Qwen3.8-27B-Quark-AWQ-MXFP4 (dequantized exactly to bf16).
    fmt="bf16":  Qwen/Qwen3.8-27B.
    Raw MXFP4 bytes are kept in `self.q` (name -> (w_u8, s_u8)) for the kernel.
    """

    LIN = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
           "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    def __init__(self, path: str, fmt: str, device="cuda", keep_raw=False, layers=NL, dequant=True):
        self.fmt, self.device, self.t, self.q = fmt, device, {}, {}
        idx = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(idx):
            files = sorted(set(json.load(open(idx))["weight_map"].values()))
        else:
            files = ["model.safetensors"]
        want = lambda k: ("visual" not in k and "mtp" not in k and
                          (".layers." not in k or int(k.split(".layers.")[1].split(".")[0]) < layers))
        for f in files:
            with safe_open(os.path.join(path, f), framework="pt", device="cpu") as sf:
                for k in sf.keys():
                    if want(k):
                        self.t[k] = sf.get_tensor(k)
        if fmt == "mxfp4":
            for k in [k for k in self.t if k.endswith(".weight_scale")]:
                base = k[: -len(".weight_scale")]
                w, s = self.t.pop(base + ".weight"), self.t.pop(k)
                if keep_raw:
                    self.q[base] = (w, s)
                if dequant:
                    self.t[base + ".weight"] = dequant_mxfp4(w.to(device), s.to(device)).cpu()
        for k in self.t:
            self.t[k] = self.t[k].to(device)

    def __getitem__(self, k):
        return self.t[k]

    def L(self, i, name):
        return self.t[f"model.language_model.layers.{i}.{name}"]


def rms0(x32: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Zero-centred RMSNorm in fp32: x * rsqrt(mean(x^2)+eps) * (1+w)."""
    r = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS)
    return x32 * r * (1.0 + w.float())


def norm_in(h32: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The RMSNorm in front of a GEMV. Emulate mode: the megakernel scales the bf16 copy
    of the residual (rstd from the fp32 residual)."""
    if not EMULATE:
        return rms0(h32, w)
    r = torch.rsqrt(h32.pow(2).mean(-1, keepdim=True) + EPS)
    return h32.to(torch.bfloat16).float() * r * (1.0 + w.float())


EMULATE = False   # set by step(): mirror the megakernel's fp8 activation quantization


def pow2_scale(amax):
    """(qinv, scale) powers of two with amax * qinv in [120, 240] (e4m3fnuz max 240)."""
    m, e = torch.frexp(amax)
    e = e - 1 + (m * 2 > 1.875).int()
    return torch.exp2(7.0 - e), torch.exp2(e - 7.0)


def quant_fp8(x32: torch.Tensor) -> torch.Tensor:
    """Megakernel GEMV input: e4m3fnuz with one pow2 scale per (row, 128 K block)."""
    M, K = x32.shape
    xb = x32.reshape(M, K // 128, 128)
    qi, sc = pow2_scale(xb.abs().amax(-1, keepdim=True).clamp_min(1e-30))
    return ((xb * qi).to(torch.float8_e4m3fnuz).float() * sc).reshape(M, K)


def lin(x32: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """GEMV input x (fp32) -> fp32 output with fp32 accumulation.

    Exact mode rounds x to bf16 (the model's activation dtype); emulate mode applies
    the megakernel's fp8 quantization.
    """
    xq = quant_fp8(x32.float()) if EMULATE else x32.to(torch.bfloat16).float()
    return xq @ w.float().t()


def lin_bf16(x32: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """LM head: bf16 activations in both modes."""
    return x32.to(torch.bfloat16).float() @ w.float().t()


def rope_cos_sin(pos: torch.Tensor):
    inv = THETA ** (-torch.arange(0, ROT, 2, device=pos.device, dtype=torch.float64) / ROT)
    ang = pos.double()[:, None] * inv[None, :]
    ang = torch.cat([ang, ang], -1)
    return ang.cos().float(), ang.sin().float()


def apply_rope(x: torch.Tensor, cos, sin):
    """x [M, heads, 256] fp32; NeoX half-split rotation on the first 64 dims."""
    xr, xp = x[..., :ROT], x[..., ROT:]
    rot = torch.cat([-xr[..., ROT // 2:], xr[..., : ROT // 2]], -1)
    xr = xr * cos[:, None, :] + rot * sin[:, None, :]
    return torch.cat([xr, xp], -1)


@dataclass
class State:
    """Per-sequence caches for M rows."""
    conv: torch.Tensor  # [48 gdn layers, M, 10240, 3] fp32 holding bf16 values (oldest first)
    ssm: torch.Tensor   # [48, M, 48 vh, 128 v, 128 k] fp32
    k: torch.Tensor     # [16 attn layers, M, 4, maxlen, 256] bf16
    v: torch.Tensor     # [16, M, 4, maxlen, 256] bf16
    pos: torch.Tensor   # [M] int64: number of tokens already in the cache

    @staticmethod
    def new(M: int, maxlen: int, device="cuda"):
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=device)
        return State(z(48, M, CONV_DIM, CONV_K - 1), z(48, M, NVH, DV, DK),
                     z(16, M, NKV, maxlen, HD, dt=torch.bfloat16),
                     z(16, M, NKV, maxlen, HD, dt=torch.bfloat16),
                     torch.zeros(M, dtype=torch.long, device=device))

    def clone(self):
        return State(*(t.clone() for t in (self.conv, self.ssm, self.k, self.v, self.pos)))


def gdn(W: Weights, i: int, x: torch.Tensor, st: State, g_i: int) -> torch.Tensor:
    M = x.shape[0]
    pre = "linear_attn."
    qkv = lin(x, W.L(i, pre + "in_proj_qkv.weight"))      # [M, 10240]
    z = lin(x, W.L(i, pre + "in_proj_z.weight"))          # [M, 6144]
    b = lin(x, W.L(i, pre + "in_proj_b.weight"))          # [M, 48]
    a = lin(x, W.L(i, pre + "in_proj_a.weight"))          # [M, 48]
    cw = W.L(i, pre + "conv1d.weight").float()[:, 0, :]  # [10240, 4]
    win = torch.cat([st.conv[g_i], qkv[:, :, None]], -1)  # [M, 10240, 4]
    st.conv[g_i] = win[:, :, 1:].to(torch.bfloat16).float()
    y = (win * cw[None]).sum(-1)
    y = y * torch.sigmoid(y)                               # silu
    q = y[:, :KEY_DIM].reshape(M, NKH, DK)
    k = y[:, KEY_DIM: 2 * KEY_DIM].reshape(M, NKH, DK)
    v = y[:, 2 * KEY_DIM:].reshape(M, NVH, DV)
    q = q / torch.sqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)
    q = q * DK ** -0.5
    rep = NVH // NKH
    q = q.repeat_interleave(rep, 1)                        # [M, 48, 128]
    k = k.repeat_interleave(rep, 1)
    beta = torch.sigmoid(b)
    A = W.L(i, pre + "A_log").float()
    dtb = W.L(i, pre + "dt_bias").float()
    g = -torch.exp(A)[None] * torch.nn.functional.softplus(a + dtb[None])  # [M, 48]
    S = st.ssm[g_i]                                        # [M, 48, v, k]
    S = S * torch.exp(g)[:, :, None, None]
    kv = torch.einsum("mhvk,mhk->mhv", S, k)
    delta = (v - kv) * beta[:, :, None]
    S = S + delta[:, :, :, None] * k[:, :, None, :]
    st.ssm[g_i] = S
    o = torch.einsum("mhvk,mhk->mhv", S, q)                # [M, 48, 128]
    o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS)
    o = o * W.L(i, pre + "norm.weight").float()
    zz = z.reshape(M, NVH, DV)
    o = o * (zz * torch.sigmoid(zz))
    return lin(o.reshape(M, VAL_DIM), W.L(i, pre + "out_proj.weight"))


def attn(W: Weights, i: int, x: torch.Tensor, st: State, a_i: int) -> torch.Tensor:
    M = x.shape[0]
    pre = "self_attn."
    qg = lin(x, W.L(i, pre + "q_proj.weight")).reshape(M, NQ, 2 * HD)
    q, gate = qg[..., :HD], qg[..., HD:].reshape(M, NQ * HD)
    k = lin(x, W.L(i, pre + "k_proj.weight")).reshape(M, NKV, HD)
    v = lin(x, W.L(i, pre + "v_proj.weight")).reshape(M, NKV, HD)
    q = rms0(q, W.L(i, pre + "q_norm.weight"))
    k = rms0(k, W.L(i, pre + "k_norm.weight"))
    cos, sin = rope_cos_sin(st.pos)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    q = q.to(torch.bfloat16).float()                       # q enters QK^T in bf16 (model dtype)
    out = torch.empty(M, NQ, HD, device=x.device)
    for m in range(M):
        p = int(st.pos[m])
        st.k[a_i, m, :, p] = k[m].to(torch.bfloat16)
        st.v[a_i, m, :, p] = v[m].to(torch.bfloat16)
        K = st.k[a_i, m, :, : p + 1].float()               # [4, L, 256]
        V = st.v[a_i, m, :, : p + 1].float()
        qh = q[m].reshape(NKV, NQ // NKV, HD)              # q head h uses kv head h // 6
        s = torch.einsum("gqd,gld->gql", qh, K) * HD ** -0.5
        pr = torch.softmax(s, -1)
        out[m] = torch.einsum("gql,gld->gqd", pr, V).reshape(NQ, HD)
    o = out.reshape(M, NQ * HD) * torch.sigmoid(gate)
    return lin(o, W.L(i, pre + "o_proj.weight"))


@torch.no_grad()
def step(W: Weights, tok: torch.Tensor, st: State, layers: int = NL, dump=None, emulate=False):
    """One decode step for M rows. tok [M] int64. Returns fp32 logits [M, VOCAB].

    Advances st.pos by one. `dump`, if a dict, receives per-layer h (fp32).
    emulate: quantize GEMV inputs as the megakernel does (else bf16, the model dtype).
    """
    global EMULATE
    EMULATE = emulate
    h = W["model.language_model.embed_tokens.weight"][tok].float()
    g_i = a_i = 0
    for i in range(layers):
        x = norm_in(h, W.L(i, "input_layernorm.weight"))
        if is_attn(i):
            h = h + attn(W, i, x, st, a_i)
            a_i += 1
        else:
            h = h + gdn(W, i, x, st, g_i)
            g_i += 1
        x = norm_in(h, W.L(i, "post_attention_layernorm.weight"))
        gt = lin(x, W.L(i, "mlp.gate_proj.weight"))
        up = lin(x, W.L(i, "mlp.up_proj.weight"))
        act = gt * torch.sigmoid(gt) * up
        h = h + lin(act, W.L(i, "mlp.down_proj.weight"))
        if dump is not None:
            dump[i] = h.clone()
    st.pos += 1
    x = norm_in(h, W["model.language_model.norm.weight"])
    return lin_bf16(x, W["lm_head.weight"])


def prefill(W: Weights, prompts: list[list[int]], st: State):
    """Feed prompts token by token (all rows the same length). Returns last logits."""
    L = len(prompts[0])
    assert all(len(p) == L for p in prompts)
    logits = None
    for t in range(L):
        tok = torch.tensor([p[t] for p in prompts], device=st.pos.device)
        logits = step(W, tok, st)
    return logits
