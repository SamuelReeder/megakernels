"""Packing of the FP8 checkpoint (RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic) into the
megakernel's layouts. The tile layouts (kind "fp8_row": fp8_tiles_row, one scale per
row) are in common/mkcore/pack.py.

Per layer i (docs/DESIGN.md "Stages and barriers"):
  qkv    = [q_proj | k_proj | v_proj]   fp8 (sliding 8192 x 2816; full [q | k] 9216 x 2816)
  o      = o_proj                       fp8 (2816 x 4096 sliding, 2816 x 8192 full)
  router = router.proj                  bf16 (128 x 2816)
  e1     = experts 0..127, then the dense MLP as 3 pseudo-experts (columns 704 k ..
           704 k + 703 of gate_proj / up_proj), [gate | up] rows in gu_order
                                        fp8 (131 x 1408 x 2816)
  e2     = the same experts' down_proj, K padded 704 -> 768 with zeros
                                        fp8 (131 x 2816 x 768)
LM head: the tied embedding, bf16 (pack_bf16). Small tensors stay bf16.
"""

from __future__ import annotations

import torch

from .dims import EFF, EFFP, H, NDS, NE, NL, is_full
from mkcore.pack import fp8_tiles_row, gu_order, gu_order_rows, pack_bf16
from mkcore.pack import unit_order as _uo

from .moe import LGN

P = "model.language_model."


def _lin(W, i, name, device):
    """(e4m3fn weight [N, K], fp32 row scales [N]) of layer i's `name`."""
    w, s = W.lin(i, name)
    assert w.dtype == torch.float8_e4m3fn and s is not None, name
    return w.to(device), s.to(device).float().reshape(-1)


class Packed:
    """All megakernel weights on one device. W: reference.Weights of the FP8 checkpoint.
    chunks: name -> C (kernel.Cfg.chunks())."""

    def __init__(self, W, chunks: dict, device="cuda", layers=NL):
        assert W.fmt == "fp8", "the BF16 kernel path is not built"
        self.keep = []
        self.chunks = dict(chunks)
        self.mats = {}  # (layer, name) -> (tiles, scales); bf16: (tiles, tiles)

        def put(key, tiles, scales):
            self.mats[key] = (tiles, scales)
            self.keep += [tiles, scales]

        def put_fp8(key, w, s, cname):
            t, sc = fp8_tiles_row(w, s)
            put(key, _uo(t, 3, chunks[cname]), _uo(sc, 1, chunks[cname]))

        a = "self_attn."
        for i in range(layers):
            parts = ["q_proj", "k_proj"] + ([] if is_full(i) else ["v_proj"])
            ws = [_lin(W, i, a + p, device) for p in parts]
            put_fp8((i, "qkv"), torch.cat([w for w, _ in ws]), torch.cat([s for _, s in ws]),
                    "qkv")
            put_fp8((i, "o"), *_lin(W, i, a + "o_proj", device), "o")
            r = W.L(i, "router.proj.weight").to(device)
            assert r.shape == (LGN, H)
            t = _uo(pack_bf16(r), 3, chunks["router"])
            put((i, "router"), t, t)
            # routed experts, then the dense MLP as NDS pseudo-experts of EFF columns
            gate, up, down = [], [], []
            for e in range(NE):
                (gw, gs), (uw, us), (dw, ds) = W.expert(i, e)
                gate.append((gw, gs))
                up.append((uw, us))
                down.append((dw, ds))
            mg = _lin(W, i, "mlp.gate_proj", device)
            mu = _lin(W, i, "mlp.up_proj", device)
            md = _lin(W, i, "mlp.down_proj", device)
            for k in range(NDS):
                sl = slice(k * EFF, (k + 1) * EFF)
                gate.append((mg[0][sl], mg[1][sl]))
                up.append((mu[0][sl], mu[1][sl]))
                down.append((md[0][:, sl], md[1]))
            st = lambda xs: torch.stack([x.to(device) for x in xs])  # noqa: E731
            gsc = lambda xs: torch.stack([x.to(device).float().reshape(-1) for x in xs])  # noqa: E731
            w1 = gu_order(torch.cat([st([g for g, _ in gate]), st([u for u, _ in up])], 1))
            s1 = gu_order_rows(torch.cat([gsc([s for _, s in gate]), gsc([s for _, s in up])], 1))
            put_fp8((i, "e1"), w1, s1, "e1")
            del w1, s1
            dw = st([d for d, _ in down])  # [NE + NDS, H, EFF]
            dpad = torch.zeros(dw.shape[0], H, EFFP, dtype=dw.dtype, device=device)
            dpad[..., :EFF] = dw
            put_fp8((i, "e2"), dpad, gsc([s for _, s in down]), "e2")
            del dw, dpad
        self.lm_head = _uo(pack_bf16(W.embed.to(device)), 3, chunks["lm"])
        self.embed = W.embed.to(device).to(torch.bfloat16).contiguous()
        self.final_norm = W[P + "norm.weight"].to(device).to(torch.bfloat16).contiguous()
        self.keep += [self.lm_head, self.embed, self.final_norm]
        # "layer -1" of the rebuild (kernel.py): ones and layer_scalar 1.0
        self.ones = torch.ones(H, dtype=torch.bfloat16, device=device)
        self.keep.append(self.ones)
        self.small = {}
        names = [
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight", "pre_feedforward_layernorm_2.weight",
            "post_feedforward_layernorm_1.weight", "post_feedforward_layernorm_2.weight",
            "post_feedforward_layernorm.weight", "router.scale", "router.per_expert_scale",
            a + "q_norm.weight", a + "k_norm.weight", "layer_scalar",
        ]
        for i in range(layers):
            for n in names:
                t = W.L(i, n).to(device).to(torch.bfloat16).reshape(-1)
                if n == "layer_scalar":  # read as one dword
                    t = torch.cat([t, t])
                t = t.contiguous()
                self.small[(i, n)] = t
                self.keep.append(t)
