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
LM head: bf16 (pack_bf16). The small tensors stay bf16 as in the checkpoint. The tile
layouts (kind "fp8_b128": row_scales, fp8_tiles_b128) are in common/mkcore/pack.py.
"""

from __future__ import annotations

import torch

from .dims import FI, H, NE, NL, NVH, is_attn
from mkcore.pack import fp8_tiles_b128, gu_order, pack_bf16, row_scales
from mkcore.pack import unit_order as _uo

from .moe import LGN

P = "model.language_model."


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
                t, s = fp8_tiles_b128(w, sr)
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
            t, s = fp8_tiles_b128(w1, s1)
            put((i, "e1"), _uo(t, 3, chunks["e1"]), _uo(s, 1, chunks["e1"]))
            del gw, gs, uw, us, w1, s1
            dw, ds = stack("down_proj")
            t, s = fp8_tiles_b128(dw, ds)
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
