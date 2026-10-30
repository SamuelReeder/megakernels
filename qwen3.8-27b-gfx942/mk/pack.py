"""Offline packing of the Quark MXFP4 checkpoint into the megakernel's layouts.

Per layer the GEMVs are:
  GDN layer:  inp  = [in_proj_qkv | in_proj_z | in_proj_b | in_proj_a]  (16480 x 5120)
              out  = out_proj                                         (5120 x 6144)
  attn layer: qkv  = [q_proj | k_proj | v_proj]                       (14336 x 5120)
              o    = o_proj                                           (5120 x 6144)
  every layer gu   = [gate_proj | up_proj]                            (34816 x 5120)
              down = down_proj                                        (5120 x 17408)
MXFP4 GEMVs use the mk4 tiles of mkcore/fp4.py. The LM head stays bf16, packed for
v_mfma_f32_16x16x16_bf16 (mkcore.pack.pack_bf16). Tiles are stored in the GEMV stream's
unit order (mkcore.pack.unit_order). Small tensors stay as in the checkpoint.
"""

from __future__ import annotations

import torch

from mkcore import fp4
from mkcore.pack import gu_order, pack_bf16, unit_order

from .kernel import chunks
from .dims import KEY_DIM, NL, VAL_DIM, is_attn

L = "model.language_model.layers.{}."

# vLLM's TP layout (docs/DESIGN-TP2.md section 4): column-parallel tensors keep rank
# r's slice of each logical segment of their rows (in_proj_qkv: q | k | v; conv1d the
# same), row-parallel tensors their K slice.
ROW_SEGMENTS = {
    "linear_attn.in_proj_qkv": (KEY_DIM, KEY_DIM, VAL_DIM),
    "linear_attn.conv1d.weight": (KEY_DIM, KEY_DIM, VAL_DIM),
}
ROW_PARALLEL = ("linear_attn.out_proj", "self_attn.o_proj", "mlp.down_proj")
REPLICATED = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "linear_attn.norm.weight",
)


def shard_rows(t, tp, rank, segments=None):
    """Rank `rank`'s rows: its 1 / tp of each segment (default: one segment)."""
    if tp == 1:
        return t
    segments = segments or (t.shape[0],)
    assert sum(segments) == t.shape[0] and all(n % tp == 0 for n in segments)
    out, o = [], 0
    for n in segments:
        out.append(t[o + rank * n // tp : o + (rank + 1) * n // tp])
        o += n
    return torch.cat(out)


def shard_cols(t, tp, rank):
    """Rank `rank`'s K slice of a [N, K'] tensor (packed MXFP4 bytes or e8m0 scales)."""
    if tp == 1:
        return t
    k = t.shape[1] // tp
    assert k * tp == t.shape[1]
    return t[:, rank * k : (rank + 1) * k]


def shard_linear(name, w, s, tp, rank):
    """A linear layer's raw MXFP4 (w [N, K/2], s [N, K/32]) -> this rank's shard."""
    if name in ROW_PARALLEL:
        return shard_cols(w, tp, rank), shard_cols(s, tp, rank)
    seg = ROW_SEGMENTS.get(name)
    return shard_rows(w, tp, rank, seg), shard_rows(s, tp, rank, seg)


def shard_small(name, t, tp, rank):
    if tp == 1 or name in REPLICATED:
        return t
    return shard_rows(t, tp, rank, ROW_SEGMENTS.get(name))


def _cat(W, i, names, tp=1, rank=0):
    ws = [shard_linear(n, *W.q[(L + n).format(i)], tp, rank) for n in names]
    return torch.cat([w for w, _ in ws]), torch.cat([s for _, s in ws])


GEMVS = {
    "gdn": {
        "inp": [
            "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_z",
            "linear_attn.in_proj_b",
            "linear_attn.in_proj_a",
        ],
        "out": ["linear_attn.out_proj"],
    },
    "attn": {
        "qkv": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        "o": ["self_attn.o_proj"],
    },
    "mlp": {"gu": ["mlp.gate_proj", "mlp.up_proj"], "down": ["mlp.down_proj"]},
}


class Packed:
    """All megakernel weights of TP rank `rank` (of `tp`) on one device."""

    def __init__(self, W, device="cuda", layers=NL, tp=1, rank=0, chunk_override=()):
        self.tp, self.rank = tp, rank
        self.chunks = tuple(sorted(dict(chunk_override).items()))  # Cfg.chunks
        CHUNKS = chunks(tp, self.chunks)
        self.keep = []  # owns every tensor the tables point at
        self.mats = {}  # (layer, name) -> (tiles, scales)
        for i in range(layers):
            kinds = ["attn" if is_attn(i) else "gdn", "mlp"]
            for kind in kinds:
                for name, parts in GEMVS[kind].items():
                    w, s = _cat(W, i, parts, tp, rank)
                    if name == "gu":
                        w, s = gu_order(w), gu_order(s)
                    tiles, scales = fp4.repack(w.to(device), s.to(device))
                    C = CHUNKS.get(name, 1)
                    tiles, scales = unit_order(tiles, 2, C), unit_order(scales, 1, C)
                    self.mats[(i, name)] = (tiles, scales)
                    self.keep += [tiles, scales]
        self.lm_head = unit_order(
            pack_bf16(shard_rows(W["lm_head.weight"], tp, rank).to(device)),
            3,
            CHUNKS["lm"],
        )
        self.embed = shard_rows(
            W["model.language_model.embed_tokens.weight"], tp, rank
        ).to(device).contiguous()
        self.final_norm = W["model.language_model.norm.weight"].to(device)
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
                t = shard_small(n, W.L(i, n), tp, rank).to(device).contiguous()
                self.small[(i, n)] = t
                self.keep.append(t)
