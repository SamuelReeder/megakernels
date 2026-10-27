"""PyTorch reference of one Qwen3.6-35B-A3B decode step (text model), M rows.

Two modes (docs/DESIGN.md, "Correctness oracle"):
- exact (emulate=False): fp32 everywhere. The weights are the checkpoint's, dequantized
  exactly to fp32 (FP8: e4m3fn x the 128 x 128 block scale). Caches hold fp32. This is
  what HF transformers computes when it runs the model in fp32 (tests/ref_vs_hf.py).
- emulate (emulate=True): the megakernel's rounding.
  - GEMV input of an FP8 weight: e4m3fnuz with one pow2 scale per (row, 128 K);
    of a bf16 weight (router, shared gate, in_proj_a/b, LM head, BF16 checkpoint): bf16.
  - Norm input: the bf16 copy of the fp32 residual times the fp32 rstd.
  - Router and shared-gate logits fp32 (input bf16); routing weights fp32.
  - Expert act: fp32, quantized as a GEMV input of the down projection.
  - Conv state, KV cache and q in QK^T: bf16.
  - The expert outputs of a row add up in slot order (top-k rank, then the shared expert).
Routing in both modes: top-8 by logit, ties to the lower expert id; weights = softmax over
the 8 selected logits (equal to HF's softmax over 256, top-8, renormalize).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
from safetensors import safe_open

from . import dims
from .dims import (
    CONV_DIM,
    CONV_K,
    DK,
    DV,
    EPS,
    HD,
    KEY_DIM,
    N_ATT,
    N_GDN,
    NE,
    NKH,
    NKV,
    NL,
    NQ,
    NVH,
    ROT,
    THETA,
    TOPK,
    VAL_DIM,
    is_attn,
)

P = "model.language_model."


def dequant_fp8(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """e4m3fn [..., N, K] x block scales [..., N/128, K/128] -> fp32 [..., N, K] (exact)."""
    sc = s.float().repeat_interleave(128, -2).repeat_interleave(128, -1)
    return w.float() * sc[..., : w.shape[-2], : w.shape[-1]]


class Weights:
    """Text-model weights on `device`.

    Dense matrices and small tensors: fp32 (FP8 dequantized exactly). Routed experts stay
    in the checkpoint's format and are dequantized when a step selects them.
    `fp8` holds the names of the matrices that are FP8 in the checkpoint. With keep_raw,
    `raw[name] = (e4m3fn weight, bf16 block scale)` on the CPU (for the kernel's packer).
    """

    def __init__(self, path: str, device="cuda", layers: int = NL, keep_raw=False):
        cfg = json.load(open(os.path.join(path, "config.json")))
        dims.check(cfg)
        qc = cfg.get("quantization_config")
        self.fmt = "bf16"
        if qc is not None:
            assert qc["quant_method"] == "fp8" and qc["fmt"] == "e4m3", qc
            assert qc["weight_block_size"] == [128, 128], qc
            self.fmt = "fp8"
        self.device, self.layers = device, layers
        self.t: dict[str, torch.Tensor] = {}
        self.fp8: set[str] = set()
        self.raw: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        idx = os.path.join(path, "model.safetensors.index.json")
        files = (
            sorted(set(json.load(open(idx))["weight_map"].values()))
            if os.path.exists(idx)
            else ["model.safetensors"]
        )

        def want(k):
            if not k.startswith(P) and k != "lm_head.weight":
                return False
            if ".layers." in k:
                return int(k.split(".layers.")[1].split(".")[0]) < layers
            return True

        cpu: dict[str, torch.Tensor] = {}
        for f in files:
            with safe_open(os.path.join(path, f), framework="pt", device="cpu") as sf:
                for k in sf.keys():
                    if want(k):
                        cpu[k] = sf.get_tensor(k)
        # routed experts: [NE, N, K] per layer and projection
        self.ex: dict[tuple[int, str], torch.Tensor] = {}
        self.exs: dict[tuple[int, str], torch.Tensor] = {}
        for i in range(layers):
            pre = f"{P}layers.{i}.mlp.experts."
            if self.fmt == "fp8":
                for pj in ("gate_proj", "up_proj", "down_proj"):
                    ks = [f"{pre}{e}.{pj}.weight" for e in range(NE)]
                    self.ex[(i, pj)] = torch.stack([cpu.pop(k) for k in ks]).to(device)
                    self.exs[(i, pj)] = torch.stack(
                        [cpu.pop(k + "_scale_inv") for k in ks]
                    ).to(device)
            else:
                gu = cpu.pop(pre + "gate_up_proj")  # [NE, 2 FI, H]: gate rows, then up
                n = gu.shape[1] // 2
                self.ex[(i, "gate_proj")] = gu[:, :n].contiguous().to(device)
                self.ex[(i, "up_proj")] = gu[:, n:].contiguous().to(device)
                self.ex[(i, "down_proj")] = cpu.pop(pre + "down_proj").to(device)
        for k in [k for k in cpu if k.endswith(".weight_scale_inv")]:
            base = k[: -len("_scale_inv")]
            w, s = cpu.pop(base), cpu.pop(k)
            assert w.dtype == torch.float8_e4m3fn, (base, w.dtype)
            self.fp8.add(base)
            if keep_raw:
                self.raw[base] = (w, s)
            self.t[base] = dequant_fp8(w.to(device), s.to(device))
        for k, v in cpu.items():
            self.t[k] = v.to(device).float()
        # the embedding and the LM head stay bf16 (2 x 1 GB); used through .float() rows
        for k in (P + "embed_tokens.weight", "lm_head.weight"):
            self.t[k] = self.t[k].to(torch.bfloat16)

    def __getitem__(self, k):
        return self.t[k]

    def L(self, i, name):
        return self.t[f"{P}layers.{i}.{name}"]

    def is_fp8(self, i, name) -> bool:
        return f"{P}layers.{i}.{name}" in self.fp8

    def expert(self, i: int, e: int):
        """fp32 (gate [FI, H], up [FI, H], down [H, FI]) of routed expert e of layer i."""
        out = []
        for pj in ("gate_proj", "up_proj", "down_proj"):
            w = self.ex[(i, pj)][e]
            out.append(dequant_fp8(w, self.exs[(i, pj)][e]) if self.fmt == "fp8" else w.float())
        return out


EMULATE = False  # set by step()


def bf16r(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


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


def lin(x32: torch.Tensor, w: torch.Tensor, fp8: bool) -> torch.Tensor:
    """x [M, K] fp32 @ w [N, K]^T, fp32 accumulation. Emulate mode rounds the input as the
    kernel's GEMV does for this weight format."""
    if EMULATE:
        x32 = quant_fp8(x32) if fp8 else bf16r(x32)
    return x32 @ w.float().t()


def rms0(x32: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Zero-centred RMSNorm in fp32: x * rsqrt(mean(x^2) + eps) * (1 + w)."""
    r = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS)
    return x32 * r * (1.0 + w.float())


def norm_in(h32: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The RMSNorm in front of a GEMV. Emulate: the kernel scales the bf16 copy of the
    residual (rstd from the fp32 residual)."""
    if not EMULATE:
        return rms0(h32, w)
    r = torch.rsqrt(h32.pow(2).mean(-1, keepdim=True) + EPS)
    return bf16r(h32) * r * (1.0 + w.float())


def cache_r(x: torch.Tensor) -> torch.Tensor:
    """A value stored in a bf16 cache (emulate) or an fp32 one (exact)."""
    return bf16r(x) if EMULATE else x


def rope_cos_sin(pos: torch.Tensor):
    inv = THETA ** (-torch.arange(0, ROT, 2, device=pos.device, dtype=torch.float64) / ROT)
    ang = pos.double()[:, None] * inv[None, :]
    ang = torch.cat([ang, ang], -1)
    return ang.cos().float(), ang.sin().float()


def apply_rope(x: torch.Tensor, cos, sin):
    """x [M, heads, 256] fp32; NeoX half-split rotation on the first ROT dims."""
    xr, xp = x[..., :ROT], x[..., ROT:]
    rot = torch.cat([-xr[..., ROT // 2 :], xr[..., : ROT // 2]], -1)
    xr = xr * cos[:, None, :] + rot * sin[:, None, :]
    return torch.cat([xr, xp], -1)


@dataclass
class State:
    """Per-sequence caches for M rows (fp32 tensors; emulate stores bf16 values)."""

    conv: torch.Tensor  # [30, M, 8192, 3] oldest input first
    ssm: torch.Tensor  # [30, M, 32 vh, 128 v, 128 k]
    k: torch.Tensor  # [10, M, 2, maxlen, 256]
    v: torch.Tensor  # [10, M, 2, maxlen, 256]
    pos: torch.Tensor  # [M] int64: tokens already in the cache

    @staticmethod
    def new(M: int, maxlen: int, device="cuda"):
        z = lambda *s: torch.zeros(*s, dtype=torch.float32, device=device)  # noqa: E731
        return State(
            z(N_GDN, M, CONV_DIM, CONV_K - 1),
            z(N_GDN, M, NVH, DV, DK),
            z(N_ATT, M, NKV, maxlen, HD),
            z(N_ATT, M, NKV, maxlen, HD),
            torch.zeros(M, dtype=torch.long, device=device),
        )

    def clone(self):
        return State(*(t.clone() for t in (self.conv, self.ssm, self.k, self.v, self.pos)))


def gdn(W: Weights, i: int, x: torch.Tensor, st: State, g_i: int) -> torch.Tensor:
    M = x.shape[0]
    pre = "linear_attn."

    def L(n):
        return lin(x, W.L(i, pre + n + ".weight"), W.is_fp8(i, pre + n + ".weight"))

    qkv, z, b, a = L("in_proj_qkv"), L("in_proj_z"), L("in_proj_b"), L("in_proj_a")
    cw = W.L(i, pre + "conv1d.weight")[:, 0, :]  # [8192, 4]
    win = torch.cat([st.conv[g_i], qkv[:, :, None]], -1)  # [M, 8192, 4]
    st.conv[g_i] = cache_r(win[:, :, 1:])
    y = (win * cw[None]).sum(-1)
    y = y * torch.sigmoid(y)  # silu
    q = y[:, :KEY_DIM].reshape(M, NKH, DK)
    k = y[:, KEY_DIM : 2 * KEY_DIM].reshape(M, NKH, DK)
    v = y[:, 2 * KEY_DIM :].reshape(M, NVH, DV)
    q = q / torch.sqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)
    q = q * DK**-0.5
    q = q.repeat_interleave(NVH // NKH, 1)  # V head h uses K head h // 2
    k = k.repeat_interleave(NVH // NKH, 1)
    beta = torch.sigmoid(b)
    g = -torch.exp(W.L(i, pre + "A_log"))[None] * torch.nn.functional.softplus(
        a + W.L(i, pre + "dt_bias")[None]
    )
    S = st.ssm[g_i] * torch.exp(g)[:, :, None, None]  # [M, 32, v, k]
    kv = torch.einsum("mhvk,mhk->mhv", S, k)
    S = S + ((v - kv) * beta[:, :, None])[:, :, :, None] * k[:, :, None, :]
    st.ssm[g_i] = S
    o = torch.einsum("mhvk,mhk->mhv", S, q)
    o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS) * W.L(i, pre + "norm.weight")
    zz = z.reshape(M, NVH, DV)
    o = (o * (zz * torch.sigmoid(zz))).reshape(M, VAL_DIM)
    return lin(o, W.L(i, pre + "out_proj.weight"), W.is_fp8(i, pre + "out_proj.weight"))


def attn(W: Weights, i: int, x: torch.Tensor, st: State, a_i: int) -> torch.Tensor:
    M = x.shape[0]
    pre = "self_attn."

    def L(n):
        return lin(x, W.L(i, pre + n + ".weight"), W.is_fp8(i, pre + n + ".weight"))

    qg = L("q_proj").reshape(M, NQ, 2 * HD)  # per head [q | gate]
    q, gate = qg[..., :HD], qg[..., HD:].reshape(M, NQ * HD)
    k = L("k_proj").reshape(M, NKV, HD)
    v = L("v_proj").reshape(M, NKV, HD)
    q = rms0(q, W.L(i, pre + "q_norm.weight"))
    k = rms0(k, W.L(i, pre + "k_norm.weight"))
    cos, sin = rope_cos_sin(st.pos)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    q = cache_r(q)  # q enters QK^T in bf16 in the kernel
    out = torch.empty(M, NQ, HD, device=x.device)
    for m in range(M):
        p = int(st.pos[m])
        st.k[a_i, m, :, p] = cache_r(k[m])
        st.v[a_i, m, :, p] = cache_r(v[m])
        K = st.k[a_i, m, :, : p + 1]  # [2, L, 256]
        V = st.v[a_i, m, :, : p + 1]
        qh = q[m].reshape(NKV, NQ // NKV, HD)  # q head h uses kv head h // 8
        s = torch.einsum("gqd,gld->gql", qh, K) * HD**-0.5
        out[m] = torch.einsum("gql,gld->gqd", torch.softmax(s, -1), V).reshape(NQ, HD)
    o = out.reshape(M, NQ * HD) * torch.sigmoid(gate)
    return lin(o, W.L(i, pre + "o_proj.weight"), W.is_fp8(i, pre + "o_proj.weight"))


def route(logits: torch.Tensor):
    """logits [M, NE] -> (expert ids [M, 8] in rank order, weights [M, 8] fp32).
    Ties go to the lower expert id (stable sort)."""
    idx = torch.sort(logits, dim=-1, descending=True, stable=True).indices[:, :TOPK]
    top = logits.gather(1, idx)
    w = torch.exp(top - top[:, :1])
    return idx, w / w.sum(-1, keepdim=True)


def moe(W: Weights, i: int, x: torch.Tensor, valid: torch.Tensor, tr: dict | None):
    """Routed experts + gated shared expert for the normed input x [M, H]. Rows with
    valid[m] False (padding) select no expert; their output is don't-care (zero here)."""
    M = x.shape[0]
    wr = torch.cat([W.L(i, "mlp.gate.weight"), W.L(i, "mlp.shared_expert_gate.weight")])
    lg = lin(x, wr, False)  # [M, 257] fp32 (emulate: bf16 input)
    idx, w = route(lg[:, :NE])
    sg = torch.sigmoid(lg[:, NE])

    def ffn(gate, up, down, fp8):
        g, u = lin(x, gate, fp8), lin(x, up, fp8)
        return lin(g * torch.sigmoid(g) * u, down, fp8)

    ys = {}
    for e in torch.unique(idx[valid]).tolist():
        ys[e] = ffn(*W.expert(i, e), W.fmt == "fp8")
    se = "mlp.shared_expert."
    ysh = ffn(
        W.L(i, se + "gate_proj.weight"),
        W.L(i, se + "up_proj.weight"),
        W.L(i, se + "down_proj.weight"),
        W.is_fp8(i, se + "gate_proj.weight"),
    )
    out = torch.zeros_like(x)
    for m in range(M):
        if not bool(valid[m]):
            continue
        acc = torch.zeros_like(x[m])
        for j in range(TOPK):
            acc = acc + w[m, j] * ys[int(idx[m, j])][m]
        out[m] = acc + sg[m] * ysh[m]
    if tr is not None:
        tr.update(router=lg, idx=idx, w=w, sg=sg)
    return out


@torch.no_grad()
def step(
    W: Weights,
    tok: torch.Tensor,
    st: State,
    layers: int = NL,
    emulate=False,
    trace: list | None = None,
    valid: torch.Tensor | None = None,
):
    """One decode step for M rows. tok [M] int64. Returns fp32 logits [M, VOCAB] and
    advances st.pos. trace, if a list, receives per layer a dict: h1 (after the mixer),
    h0 (after the MoE block), router (logits [M, 257]), idx, w, sg."""
    global EMULATE
    EMULATE = emulate
    M = tok.shape[0]
    if valid is None:
        valid = torch.ones(M, dtype=torch.bool, device=tok.device)
    h = W[P + "embed_tokens.weight"][tok].float()
    g_i = a_i = 0
    for i in range(layers):
        x = norm_in(h, W.L(i, "input_layernorm.weight"))
        if is_attn(i):
            h = h + attn(W, i, x, st, a_i)
            a_i += 1
        else:
            h = h + gdn(W, i, x, st, g_i)
            g_i += 1
        tr = {"h1": h.clone()} if trace is not None else None
        x = norm_in(h, W.L(i, "post_attention_layernorm.weight"))
        h = h + moe(W, i, x, valid, tr)
        if trace is not None:
            tr["h0"] = h.clone()
            trace.append(tr)
    st.pos += 1
    x = norm_in(h, W[P + "norm.weight"])
    return lin(x, W["lm_head.weight"], False)


def prefill(W: Weights, prompts: list[list[int]], st: State, emulate=False):
    """Feed prompts token by token (all rows the same length). Returns the logits of
    every position [L, M, VOCAB]."""
    L = len(prompts[0])
    assert all(len(p) == L for p in prompts)
    out = []
    for t in range(L):
        tok = torch.tensor([p[t] for p in prompts], device=st.pos.device)
        out.append(step(W, tok, st, emulate=emulate))
    return torch.stack(out)
