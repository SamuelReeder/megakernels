"""Packing of the FP8 checkpoint into the megakernel's layouts.

Per layer the GEMVs are (docs/DESIGN.md, "Per-layer stage list"):
  GDN layer:  inp  = [in_proj_qkv | in_proj_z]         fp8  (12288 x 2048)
              ba   = [in_proj_b | in_proj_a]           bf16 (64 x 2048)
              out  = out_proj                          fp8  (2048 x 4096)
  attn layer: qkv  = [q_proj | k_proj | v_proj]        fp8  (9216 x 2048)
              o    = o_proj                            fp8  (2048 x 4096)
  every layer router = [mlp.gate | shared_expert_gate | 0]  bf16 (272 x 2048)
              e1   = experts 0..255 + shared, [gate | up] rows in gu_order
                                                      fp8  (257 x 1024 x 2048)
              e2   = experts 0..255 + shared, down     fp8  (257 x 2048 x 512)
LM head: bf16 (pack_bf16). The small tensors stay bf16 as in the checkpoint.

fp8 format: e4m3fn bits read as e4m3fnuz (bias 8 instead of 7: the value halves), 0x80
(e4m3fn -0, e4m3fnuz NaN) -> 0x00, block scale x 2. Exact.
"""

from __future__ import annotations

import torch

from .dims import FI, H, NE, NL, NVH, is_attn
from mkcore.gemv import PAD_TILES

from .moe import LGN

P = "model.language_model."


def gu_order(t: torch.Tensor) -> torch.Tensor:
    """[..., gate; up rows, ...] (rows on dim -2) -> per 8 columns c: gate rows 8 c ..
    8 c + 7, then up rows 8 c .. 8 c + 7. One 16-row group holds both halves of 8 act
    columns (moe.EpiAct relies on it)."""
    *b, N2, K = t.shape
    n = N2 // 2
    g = t[..., :n, :].reshape(*b, n // 8, 8, K)
    u = t[..., n:, :].reshape(*b, n // 8, 8, K)
    return torch.cat([g, u], -2).reshape(*b, N2, K)


def row_scales(s: torch.Tensor, N: int) -> torch.Tensor:
    """Block scales [..., N/128, K/128] -> per-row fp32 [..., N, K/128]."""
    return s.float().repeat_interleave(128, -2)[..., :N, :]


def fp8_tiles(w: torch.Tensor, srow: torch.Tensor):
    """Convert e4m3fn [..., N, K] + per-row fp32 scales [..., N, K/128] to (tiles u8
    [..., G, KS, 2, 64, 16], scales fp32 [..., G, KS, 2]).

    Lane l (row r = l % 16, gl = l // 16), load q: MFMA steps d = 2 q, 2 q + 1, 8 bytes
    each at k = 128 st + 32 d + 8 gl + 0..7. Scales: rows 0-7 and rows 8-15 of the group
    (each half must have one scale)."""
    *b, N, K = w.shape
    G, KS = N // 16, K // 128
    assert N % 16 == 0 and K % 128 == 0
    u8 = w.view(torch.uint8)
    assert not ((u8 & 0x7F) == 0x7F).any(), "e4m3fn NaN in the checkpoint"
    u8 = torch.where(u8 == 0x80, torch.zeros_like(u8), u8)
    nb = len(b)
    t = u8.reshape(*b, G, 16, KS, 2, 2, 4, 8)  # g r st q dd gl byte
    perm = list(range(nb)) + [nb + i for i in (0, 2, 3, 5, 1, 4, 6)]
    tiles = t.permute(perm).reshape(*b, G, KS, 2, 64, 16).contiguous()
    sr = srow.reshape(*b, G, 2, 8, KS)
    assert torch.equal(sr, sr[..., :1, :].expand_as(sr)), "scale varies inside 8 rows"
    sc = (2.0 * sr[..., 0, :]).transpose(-1, -2).contiguous()  # [..., G, KS, 2]
    return tiles, sc


def pack_bf16(w: torch.Tensor) -> torch.Tensor:
    """bf16 [N, K] -> [G, KS, 4, 64, 8] bf16 (4 KiB per (group, superstep)).

    Lane l (row r = l % 16, gl = l // 16), quarter q (16 B): sub-steps s = 2q, 2q+1 of
    the 16x16x16 MFMA, each 4 bf16 at k = 128 st + 16 s + 4 gl + 0..3.
    """
    N, K = w.shape
    G, KS = N // 16, K // 128
    t = w.to(torch.bfloat16).reshape(G, 16, KS, 4, 2, 4, 4)  # g r st q h gl e
    t = t.permute(0, 2, 3, 5, 1, 4, 6)  # g st q gl r h e
    return t.reshape(G, KS, 4, 64, 8).contiguous()


def unit_order(t: torch.Tensor, nrest: int, C: int) -> torch.Tensor:
    """[..., G, KS, rest (nrest dims)] per-(group, superstep) data -> the stream's unit
    order (chunk, group, superstep), flat [(...) C G KSc, rest] (leading dims outermost,
    e.g. experts), plus PAD_TILES zero units (loads run ahead)."""
    rest = t.shape[t.dim() - nrest:]
    G, KS = t.shape[t.dim() - nrest - 2: t.dim() - nrest]
    u = t.reshape(-1, G, C, KS // C, *rest).transpose(1, 2).reshape(-1, *rest)
    pad = torch.zeros(PAD_TILES, *rest, dtype=t.dtype, device=t.device)
    return torch.cat([u, pad]).contiguous()


_uo = unit_order


def _fp8(W, names, i, device):
    """Concatenated raw fp8 rows of `names` (layer i) -> (w e4m3fn, per-row scales)."""
    ws, ss = [], []
    for n in names:
        w, s = W.raw[f"{P}layers.{i}.{n}.weight"]
        ws.append(w.to(device))
        ss.append(row_scales(s.to(device), w.shape[0]))
    return torch.cat(ws), torch.cat(ss)


GEMVS = {
    "gdn": {
        "inp": ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"],
        "out": ["linear_attn.out_proj"],
    },
    "attn": {
        "qkv": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        "o": ["self_attn.o_proj"],
    },
}


class Packed:
    """All megakernel weights on one device. W: reference.Weights of the FP8 checkpoint
    loaded with keep_raw=True. chunks: name -> C (kernel.Cfg.chunks())."""

    def __init__(self, W, chunks: dict, device="cuda", layers=NL):
        assert W.fmt == "fp8", "the BF16 kernel path is not built"
        self.keep = []
        self.chunks = dict(chunks)
        self.mats = {}  # (layer, name) -> (tiles, scales) ; bf16: (tiles, tiles)

        def put(key, tiles, scales):
            self.mats[key] = (tiles, scales)
            self.keep += [tiles, scales]

        for i in range(layers):
            for name, parts in GEMVS["attn" if is_attn(i) else "gdn"].items():
                w, sr = _fp8(W, parts, i, device)
                t, s = fp8_tiles(w, sr)
                put((i, name), _uo(t, 3, chunks[name]), _uo(s, 1, chunks[name]))
            if not is_attn(i):
                ba = torch.cat([W.L(i, "linear_attn.in_proj_b.weight"),
                                W.L(i, "linear_attn.in_proj_a.weight")])
                assert ba.shape == (2 * NVH, H)
                t = _uo(pack_bf16(ba), 3, chunks["ba"])
                put((i, "ba"), t, t)
            r = torch.zeros(LGN, H, device=device)
            r[:NE] = W.L(i, "mlp.gate.weight")
            r[NE] = W.L(i, "mlp.shared_expert_gate.weight")[0]
            t = _uo(pack_bf16(r), 3, chunks["router"])
            put((i, "router"), t, t)
            # routed experts + the shared expert as expert NE
            se = f"{P}layers.{i}.mlp.shared_expert."
            def stack(pj, i=i, se=se):
                w = W.ex[(i, pj)]
                s = row_scales(W.exs[(i, pj)], w.shape[1])
                ws, ss = W.raw[se + pj + ".weight"]
                w = torch.cat([w, ws.to(device)[None]])
                s = torch.cat([s, row_scales(ss.to(device), ws.shape[0])[None]])
                return w, s
            gw, gs = stack("gate_proj")
            uw, us = stack("up_proj")
            w1 = gu_order(torch.cat([gw, uw], 1))
            s1 = gu_order(torch.cat([gs, us], 1))
            t, s = fp8_tiles(w1, s1)
            put((i, "e1"), _uo(t, 3, chunks["e1"]), _uo(s, 1, chunks["e1"]))
            del gw, gs, uw, us, w1, s1
            dw, ds = stack("down_proj")
            t, s = fp8_tiles(dw, ds)
            put((i, "e2"), _uo(t, 3, chunks["e2"]), _uo(s, 1, chunks["e2"]))
            del dw, ds
        lm = W["lm_head.weight"].to(device)
        self.lm_head = _uo(pack_bf16(lm), 3, chunks["lm"])
        self.embed = W[P + "embed_tokens.weight"].to(device).to(torch.bfloat16)
        self.final_norm = W[P + "norm.weight"].to(device).to(torch.bfloat16)
        self.keep += [self.lm_head, self.embed, self.final_norm]
        self.small = {}
        for i in range(layers):
            names = ["input_layernorm.weight", "post_attention_layernorm.weight"]
            if is_attn(i):
                names += ["self_attn.q_norm.weight", "self_attn.k_norm.weight"]
            else:
                names += [
                    "linear_attn.A_log",
                    "linear_attn.dt_bias",
                    "linear_attn.conv1d.weight",
                    "linear_attn.norm.weight",
                ]
            for n in names:
                # the checkpoint's bf16 values (reference.Weights holds them as fp32)
                t = W.L(i, n).to(device).to(torch.bfloat16).contiguous()
                self.small[(i, n)] = t
                self.keep.append(t)
        assert FI == 512
