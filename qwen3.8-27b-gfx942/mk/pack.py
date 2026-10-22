"""Offline packing of the Quark MXFP4 checkpoint into the megakernel's layouts.

Per layer the GEMVs are:
  GDN layer:  inp  = [in_proj_qkv | in_proj_z | in_proj_b | in_proj_a]  (16480 x 5120)
              out  = out_proj                                         (5120 x 6144)
  attn layer: qkv  = [q_proj | k_proj | v_proj]                       (14336 x 5120)
              o    = o_proj                                           (5120 x 6144)
  every layer gu   = [gate_proj | up_proj]                            (34816 x 5120)
              down = down_proj                                        (5120 x 17408)
MXFP4 GEMVs use the mk4 layout of mk/fp4.py. The LM head stays bf16, packed for
v_mfma_f32_16x16x16_bf16 (see pack_bf16). Small tensors stay as in the checkpoint.
"""
from __future__ import annotations

import torch

from . import fp4
from .dims import NL, is_attn

L = "model.language_model.layers.{}."


def _cat(W, i, names):
    ws = [W.q[(L + n).format(i)] for n in names]
    return torch.cat([w for w, _ in ws]), torch.cat([s for _, s in ws])


def gu_order(t: torch.Tensor) -> torch.Tensor:
    """[gate; up] rows -> per 128-column block b: gate rows 128b.., then up rows 128b..
    (16 groups: 0..7 gate, 8..15 up; the gate_up epilogue relies on it)."""
    n = t.shape[0] // 2
    g, u = t[:n].reshape(n // 128, 128, -1), t[n:].reshape(n // 128, 128, -1)
    return torch.cat([g, u], 1).reshape(2 * n, -1)


GEMVS = {
    "gdn": {"inp": ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b",
                    "linear_attn.in_proj_a"],
            "out": ["linear_attn.out_proj"]},
    "attn": {"qkv": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
             "o": ["self_attn.o_proj"]},
    "mlp": {"gu": ["mlp.gate_proj", "mlp.up_proj"], "down": ["mlp.down_proj"]},
}


def pack_bf16(w: torch.Tensor) -> torch.Tensor:
    """bf16 [N, K] -> [G, KS, 4, 64, 8] bf16 (4 KiB per (group, superstep)).

    Lane l (row r = l % 16, gl = l // 16), quarter q (16 B): sub-steps s = 2q, 2q+1 of
    the 16x16x16 MFMA, each 4 bf16 at k = 128 st + 16 s + 4 gl + 0..3.
    """
    N, K = w.shape
    G, KS = N // 16, K // 128
    t = w.reshape(G, 16, KS, 4, 2, 4, 4)        # g r st q h gl e   (s = 2q + h)
    t = t.permute(0, 2, 3, 5, 1, 4, 6)          # g st q gl r h e
    return t.reshape(G, KS, 4, 64, 8).contiguous()


class Packed:
    """All megakernel weights on one device, plus an int64 pointer table per layer."""

    def __init__(self, W, device="cuda", layers=NL):
        self.keep = []                       # owns every tensor the tables point at
        self.mats = {}                       # (layer, name) -> (tiles, scales)
        for i in range(layers):
            kinds = ["attn" if is_attn(i) else "gdn", "mlp"]
            for kind in kinds:
                for name, parts in GEMVS[kind].items():
                    w, s = _cat(W, i, parts)
                    if name == "gu":
                        w, s = gu_order(w), gu_order(s)
                    tiles, scales = fp4.repack(w.to(device), s.to(device))
                    self.mats[(i, name)] = (tiles, scales)
                    self.keep += [tiles, scales]
        self.lm_head = pack_bf16(W["lm_head.weight"].to(device))
        self.embed = W["model.language_model.embed_tokens.weight"].to(device)
        self.final_norm = W["model.language_model.norm.weight"].to(device)
        self.keep += [self.lm_head, self.embed, self.final_norm]
        self.small = {}
        for i in range(layers):
            names = ["input_layernorm.weight", "post_attention_layernorm.weight"]
            if is_attn(i):
                names += ["self_attn.q_norm.weight", "self_attn.k_norm.weight"]
            else:
                names += ["linear_attn.A_log", "linear_attn.dt_bias", "linear_attn.conv1d.weight",
                          "linear_attn.norm.weight"]
            for n in names:
                t = W.L(i, n).to(device).contiguous()
                self.small[(i, n)] = t
                self.keep.append(t)
