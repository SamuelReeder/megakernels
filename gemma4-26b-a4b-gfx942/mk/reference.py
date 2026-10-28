"""PyTorch reference of one Gemma 4 26B-A4B decode step (text model), M rows.

Numerics (docs/DESIGN.md):
- residual stream, norms, raw projection outputs, router logits, softmax: fp32
- exact mode: every GEMV input rounded to bf16 (the model dtype), fp32 accumulation
- emulate mode: GEMV inputs quantized as the megakernel does (e4m3fnuz, one power-of-two
  scale per (row, 128 K)); the router input stays bf16 in both modes
- q enters QK^T in bf16; the KV cache holds bf16
- fp32 mode: no bf16 rounding anywhere (fp32 KV cache, fp32 sqrt(H) embedding scale), to
  compare with HF transformers run in fp32
- FP8 checkpoint: y = (x @ W_fp8^T) * row_scale, exact in fp32
Prefill runs by calling `step` once per prompt token.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import torch
from safetensors import safe_open

from .dims import (
    CONFIG,
    EFF,
    EMBED_SCALE,
    EPS,
    HD_F,
    HD_S,
    LAYER_TYPES,
    NKV_F,
    NKV_S,
    NDS,
    NL,
    NQ,
    ROPE,
    ROT_F,
    SOFTCAP,
    THETA_F,
    THETA_S,
    TOPK,
    WINDOW,
    H,
    is_full,
)

L = "model.language_model.layers.{}."
QBLK = 128  # K per activation scale of the fp8 GEMV


def check_config(path: str):
    """Assert the checkpoint's text config matches dims.py."""
    cfg = json.load(open(os.path.join(path, "config.json")))
    tc = cfg.get("text_config", cfg)
    bad = {k: (tc.get(k), v) for k, v in CONFIG.items() if tc.get(k) != v}
    assert not bad, f"config differs from dims.py: {bad}"
    assert tc["layer_types"] == LAYER_TYPES, "layer_types differ from dims.is_full"
    for t, want in ROPE.items():
        got = tc["rope_parameters"][t]
        assert all(got.get(k) == v for k, v in want.items()), (t, got)
    assert not tc.get("attn_logit_softcapping"), "attention soft-cap not supported"


class Weights:
    """Text-model weights on `device`, as stored in the checkpoint.

    fmt="bf16": google/gemma-4-26B-A4B-it (experts fused: gate_up_proj [E, 2 EFF, H],
    down_proj [E, H, EFF]). fmt="fp8": RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic (E4M3FN
    weights, one bf16 scale per output row, experts stored one by one).
    lin(i, name) -> (weight, row scale or None); expert(i, e) -> (gate, up, down) pairs.
    """

    def __init__(self, path: str, fmt: str, device="cuda", layers: int = NL):
        check_config(path)
        self.fmt, self.t = fmt, {}
        idx = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(idx):
            files = sorted(set(json.load(open(idx))["weight_map"].values()))
        else:
            files = ["model.safetensors"]

        def want(k):
            if not k.startswith("model.language_model."):
                return False
            return ".layers." not in k or int(k.split(".layers.")[1].split(".")[0]) < layers

        for f in files:
            with safe_open(os.path.join(path, f), framework="pt", device="cpu") as sf:
                for k in sf.keys():
                    if want(k):
                        self.t[k] = sf.get_tensor(k).to(device)

    def __getitem__(self, k):
        return self.t[k]

    def L(self, i, name):
        return self.t[L.format(i) + name]

    def lin(self, i, name):
        """name: e.g. 'self_attn.q_proj', 'mlp.down_proj'."""
        w = self.L(i, name + ".weight")
        s = self.t.get(L.format(i) + name + ".weight_scale")
        return w, s

    def expert(self, i, e):
        if self.fmt == "bf16":
            gu = self.L(i, "experts.gate_up_proj")[e]
            return (gu[:EFF], None), (gu[EFF:], None), (self.L(i, "experts.down_proj")[e], None)
        return tuple(self.lin(i, f"experts.{e}.{n}") for n in ("gate_proj", "up_proj", "down_proj"))

    @property
    def embed(self):
        return self.t["model.language_model.embed_tokens.weight"]


MODE = "exact"  # set by step(): "exact" | "emulate" | "fp32"


def bf(x: torch.Tensor) -> torch.Tensor:
    """Round to bf16 (the model dtype), except in fp32 mode (the HF fp32 comparison)."""
    return x if MODE == "fp32" else x.to(torch.bfloat16).float()


def rms(x32: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
    """Gemma RMSNorm in fp32: x * rsqrt(mean(x^2) + eps) * w (not zero-centred)."""
    r = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS)
    return x32 * r if w is None else x32 * r * w.float()


def pow2_scale(amax):
    """(qinv, scale) powers of two with amax * qinv in [120, 240] (e4m3fnuz max 240)."""
    m, e = torch.frexp(amax)
    e = e - 1 + (m * 2 > 1.875).int()
    return torch.exp2(7.0 - e), torch.exp2(e - 7.0)


def quant_fp8(x32: torch.Tensor, blk: int = QBLK) -> torch.Tensor:
    """Megakernel GEMV input: e4m3fnuz with one pow2 scale per (row, blk K). K that is not
    a multiple of blk is zero-padded first (the kernel's padded expert down input)."""
    M, K = x32.shape
    Kp = -(-K // blk) * blk
    xp = torch.nn.functional.pad(x32, (0, Kp - K))
    xb = xp.reshape(M, Kp // blk, blk)
    qi, sc = pow2_scale(xb.abs().amax(-1, keepdim=True).clamp_min(1e-30))
    return ((xb * qi).to(torch.float8_e4m3fnuz).float() * sc).reshape(M, Kp)[:, :K]


def lin(x32: torch.Tensor, ws) -> torch.Tensor:
    """GEMV: x (fp32) -> fp32, fp32 accumulation. Exact mode rounds x to bf16; emulate
    mode applies the megakernel's fp8 quantization; fp32 mode keeps x. ws = (weight, row
    scale or None)."""
    w, s = ws
    xq = quant_fp8(x32.float()) if MODE == "emulate" else bf(x32.float())
    y = xq @ w.float().t()
    return y if s is None else y * s.float().reshape(1, -1)


def lin_cat(x32, parts):
    return torch.cat([lin(x32, p) for p in parts], -1)


def gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def rope_inv_freq(full: bool, device="cuda") -> torch.Tensor:
    """Per-pair inverse frequencies (fp64): HD/2 pairs, NeoX pairs (d, d + HD/2)."""
    if full:
        i = torch.arange(HD_F // 2, device=device, dtype=torch.float64)
        f = THETA_F ** (-2 * i / HD_F)
        return torch.where(i < ROT_F, f, torch.zeros_like(f))  # proportional: identity rest
    i = torch.arange(HD_S // 2, device=device, dtype=torch.float64)
    return THETA_S ** (-2 * i / HD_S)


def rope_cos_sin(pos: torch.Tensor, full: bool):
    """cos, sin [len(pos), HD/2] fp32 of each angle pair."""
    ang = pos.double()[:, None] * rope_inv_freq(full, pos.device)[None, :]
    return ang.cos().float(), ang.sin().float()


def apply_rope(x: torch.Tensor, cos, sin):
    """x [M, heads, HD] fp32; NeoX half-split rotation (pairs d, d + HD/2)."""
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    c, s = cos[:, None, :], sin[:, None, :]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)


@dataclass
class State:
    """Per-sequence KV caches for M rows (contiguous; the kernel's runner pages them)."""

    ks: torch.Tensor  # [25 sliding layers, M, NKV_S, maxlen, HD_S] bf16 (fp32 mode: fp32)
    vs: torch.Tensor
    kf: torch.Tensor  # [5 full layers, M, NKV_F, maxlen, HD_F]
    vf: torch.Tensor
    pos: torch.Tensor  # [M] int64: tokens already in the cache

    @staticmethod
    def new(M: int, maxlen: int, device="cuda", dtype=torch.bfloat16):
        z = lambda *s: torch.zeros(*s, dtype=dtype, device=device)  # noqa: E731
        ns = sum(not is_full(i) for i in range(NL))
        return State(
            z(ns, M, NKV_S, maxlen, HD_S),
            z(ns, M, NKV_S, maxlen, HD_S),
            z(NL - ns, M, NKV_F, maxlen, HD_F),
            z(NL - ns, M, NKV_F, maxlen, HD_F),
            torch.zeros(M, dtype=torch.long, device=device),
        )


def attn(W: Weights, i: int, x: torch.Tensor, st: State, li: int) -> torch.Tensor:
    """Attention block of layer i (li: index among the layers of its type)."""
    M = x.shape[0]
    full = is_full(i)
    hd, nkv = (HD_F, NKV_F) if full else (HD_S, NKV_S)
    pre = "self_attn."
    q = lin(x, W.lin(i, pre + "q_proj")).reshape(M, NQ, hd)
    k = lin(x, W.lin(i, pre + "k_proj")).reshape(M, nkv, hd)
    v = k if full else lin(x, W.lin(i, pre + "v_proj")).reshape(M, nkv, hd)  # k_eq_v
    q = rms(q, W.L(i, pre + "q_norm.weight"))
    k = rms(k, W.L(i, pre + "k_norm.weight"))
    v = rms(v)
    cos, sin = rope_cos_sin(st.pos, full)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    q = bf(q)  # q enters QK^T in bf16 (model dtype)
    kc, vc = (st.kf, st.vf) if full else (st.ks, st.vs)
    out = torch.empty(M, NQ, hd, device=x.device)
    g = NQ // nkv
    for m in range(M):
        p = int(st.pos[m])
        kc[li, m, :, p] = k[m].to(kc.dtype)
        vc[li, m, :, p] = v[m].to(vc.dtype)
        lo = 0 if full else max(0, p - WINDOW + 1)
        K = kc[li, m, :, lo : p + 1].float()  # [nkv, len, hd]
        V = vc[li, m, :, lo : p + 1].float()
        qh = q[m].reshape(nkv, g, hd)  # q head h uses kv head h // g
        s = torch.einsum("gqd,gld->gql", qh, K)  # scaling 1.0
        out[m] = torch.einsum("gql,gld->gqd", torch.softmax(s, -1), V).reshape(NQ, hd)
    return lin(out.reshape(M, NQ * hd), W.lin(i, pre + "o_proj"))


def route_logits(logits: torch.Tensor, pes: torch.Tensor):
    """Routing from router logits [N, NE]: softmax, top-8 (rank order), renormalized,
    times per_expert_scale[expert]. -> (ids [N, TOPK] int64, weights fp32)."""
    probs = torch.softmax(logits.float(), -1)
    top, ids = torch.topk(probs, TOPK, dim=-1)
    return ids, top / top.sum(-1, keepdim=True) * pes.float()[ids]


def route(W: Weights, i: int, h1: torch.Tensor):
    """Router of layer i on h1 [N, H]: (ids, weights, logits fp32 [N, NE])."""
    xr = rms(h1) * (H**-0.5) * W.L(i, "router.scale").float()
    logits = bf(xr) @ W.L(i, "router.proj.weight").float().t()
    ids, wts = route_logits(logits, W.L(i, "router.per_expert_scale"))
    return ids, wts, logits


def moe(W: Weights, i: int, xe: torch.Tensor, ids, wts) -> torch.Tensor:
    """sum over k = 0..7 in rank order of w_k expert_k(xe) (the kernel's slot order),
    fp32. Rows are computed per expert, all rows that selected it at once."""
    N = xe.shape[0]
    ys = torch.zeros(N, TOPK, H, device=xe.device)
    for e in ids.unique().tolist():
        r, k = (ids == e).nonzero(as_tuple=True)
        g, u, d = W.expert(i, e)
        x = xe[r]
        ys[r, k] = lin(gelu_tanh(lin(x, g)) * lin(x, u), d)
    out = torch.zeros(N, H, device=xe.device)
    for k in range(TOPK):
        out = out + wts[:, k : k + 1] * ys[:, k]
    return out


def dense_mlp(W: Weights, i: int, xd: torch.Tensor) -> torch.Tensor:
    """The dense MLP (2112) as the kernel runs it: NDS pseudo-experts of EFF columns
    (their down inputs quantized separately in emulate mode), summed in order."""
    (gw, gs), (uw, us), (dw, ds) = (W.lin(i, f"mlp.{n}") for n in ("gate_proj", "up_proj",
                                                                       "down_proj"))
    out = torch.zeros(xd.shape[0], H, device=xd.device)
    for k in range(NDS):
        sl = slice(k * EFF, (k + 1) * EFF)
        sc = lambda s: None if s is None else s.reshape(-1)[sl]  # noqa: E731
        a = gelu_tanh(lin(xd, (gw[sl], sc(gs)))) * lin(xd, (uw[sl], sc(us)))
        out = out + lin(a, (dw[:, sl], ds))
    return out


def ffn(W: Weights, i: int, h: torch.Tensor, a: torch.Tensor, dump=None):
    """Everything after the attention output a of layer i, for any batch of rows."""
    h1 = h + rms(a, W.L(i, "post_attention_layernorm.weight"))
    d = dense_mlp(W, i, rms(h1, W.L(i, "pre_feedforward_layernorm.weight")))
    ids, wts, logits = route(W, i, h1)
    e = moe(W, i, rms(h1, W.L(i, "pre_feedforward_layernorm_2.weight")), ids, wts)
    y = rms(d, W.L(i, "post_feedforward_layernorm_1.weight")) + rms(
        e, W.L(i, "post_feedforward_layernorm_2.weight")
    )
    h = (h1 + rms(y, W.L(i, "post_feedforward_layernorm.weight"))) * W.L(
        i, "layer_scalar"
    ).float()
    if dump is not None:
        dump[i] = dict(h=h.clone(), h1=h1.clone(), ids=ids.clone(), wts=wts.clone(),
                       logits=logits.clone())
    return h


def embed(W: Weights, tok: torch.Tensor) -> torch.Tensor:
    if MODE == "fp32":  # HF in fp32 scales by the fp32 sqrt(H)
        return W.embed[tok].float() * math.sqrt(H)
    # bf16 x bf16(sqrt H) = 53.0, rounded to bf16 as vLLM and HF bf16 do
    return (W.embed[tok].float() * EMBED_SCALE).to(torch.bfloat16).float()


@torch.no_grad()
def hidden_step(W: Weights, tok: torch.Tensor, st: State, layers: int = NL, dump=None,
                mode="exact") -> torch.Tensor:
    """One decode step for M rows. tok [M] int64. Returns the final-normed hidden state
    [M, H] (bf16 values in fp32, except in fp32 mode), what the kernel writes. Advances
    st.pos by one. dump, if a dict, receives per-layer h, h1 and the router's (ids,
    weights, logits). mode: "exact" (bf16 GEMV inputs), "emulate" (the kernel's fp8
    inputs), "fp32" (no bf16 rounding anywhere: for the HF fp32 comparison; use an fp32
    State)."""
    global MODE
    assert mode in ("exact", "emulate", "fp32")
    MODE = mode
    h = embed(W, tok)
    si = fi = 0
    for i in range(layers):
        x = rms(h, W.L(i, "input_layernorm.weight"))
        a = attn(W, i, x, st, fi if is_full(i) else si)
        if is_full(i):
            fi += 1
        else:
            si += 1
        h = ffn(W, i, h, a, dump)
    st.pos += 1
    x = rms(h, W["model.language_model.norm.weight"])
    return bf(x)


@torch.no_grad()
def prefill(W: Weights, ids: torch.Tensor, st: State, layers: int = NL, qblk: int = 256):
    """Exact-mode prefill of ids [M, L] into an empty State (positions 0..L-1, all rows
    at once): the same layer math as hidden_step, with causal (and, on sliding layers,
    window) masking. Leaves st.pos = L; returns the final-normed hidden states [M, L, H]
    (bf16 values in fp32)."""
    global MODE
    MODE = "exact"
    assert int(st.pos.abs().sum()) == 0
    M, L = ids.shape
    dev = ids.device
    h = embed(W, ids.reshape(-1))
    p = torch.arange(L, device=dev)
    si = fi = 0
    for i in range(layers):
        full = is_full(i)
        hd, nkv = (HD_F, NKV_F) if full else (HD_S, NKV_S)
        li = fi if full else si
        x = rms(h, W.L(i, "input_layernorm.weight"))
        pre = "self_attn."
        q = lin(x, W.lin(i, pre + "q_proj")).reshape(M, L, NQ, hd)
        k = lin(x, W.lin(i, pre + "k_proj")).reshape(M, L, nkv, hd)
        v = k if full else lin(x, W.lin(i, pre + "v_proj")).reshape(M, L, nkv, hd)
        q = rms(q, W.L(i, pre + "q_norm.weight"))
        k = rms(k, W.L(i, pre + "k_norm.weight"))
        v = rms(v)
        cos, sin = rope_cos_sin(p, full)
        q = bf(torch.stack([apply_rope(q[m], cos, sin) for m in range(M)]))
        k = torch.stack([apply_rope(k[m], cos, sin) for m in range(M)])
        kc, vc = (st.kf, st.vf) if full else (st.ks, st.vs)
        kc[li, :, :, :L] = k.permute(0, 2, 1, 3).to(kc.dtype)
        vc[li, :, :, :L] = v.permute(0, 2, 1, 3).to(vc.dtype)
        g = NQ // nkv
        out = torch.empty(M, L, NQ, hd, device=dev)
        for m in range(M):
            K, V = kc[li, m, :, :L].float(), vc[li, m, :, :L].float()  # [nkv, L, hd]
            for q0 in range(0, L, qblk):
                qi = p[q0 : q0 + qblk]
                qh = q[m, q0 : q0 + qblk].reshape(-1, nkv, g, hd)
                s = torch.einsum("ijgd,jld->jgil", qh, K)
                vis = p[None, :] <= qi[:, None]
                if not full:
                    vis &= p[None, :] > qi[:, None] - WINDOW
                s = s.masked_fill(~vis, float("-inf"))
                o = torch.einsum("jgil,jld->ijgd", torch.softmax(s, -1), V)
                out[m, q0 : q0 + qblk] = o.reshape(-1, NQ, hd)
        a = lin(out.reshape(M * L, NQ * hd), W.lin(i, pre + "o_proj"))
        if full:
            fi += 1
        else:
            si += 1
        h = ffn(W, i, h, a)
    st.pos += L
    return bf(rms(h, W["model.language_model.norm.weight"])).reshape(M, L, H)


def logits_of(W: Weights, hidden: torch.Tensor) -> torch.Tensor:
    """Tied LM head (bf16 hidden x bf16 embedding, fp32 accumulation) + final soft-cap."""
    z = bf(hidden) @ W.embed.float().t()
    return SOFTCAP * torch.tanh(z / SOFTCAP)


def step(W: Weights, tok: torch.Tensor, st: State, layers: int = NL, dump=None,
         mode="exact") -> torch.Tensor:
    """One decode step -> fp32 logits [M, VOCAB] (soft-capped)."""
    return logits_of(W, hidden_step(W, tok, st, layers, dump, mode))
