"""Host side of the megakernel: buffers, pointer tables, one launch per decode step."""

from __future__ import annotations

import torch

from . import kernel as KK
from . import mixer as X
from .dims import (
    CONV_DIM,
    DK,
    DV,
    H,
    HD,
    INP_N,
    NKV,
    NL,
    NVH,
    QKV_N,
    FFN,
    VAL_DIM,
    VOCAB,
    is_attn,
)
from .pack import Packed
from .reference import rope_cos_sin

LAYER = "model.language_model.layers.{}."


CONV_BYTES = 3 * CONV_DIM * 2  # bf16 [3][CONV_DIM]
PAGE = CONV_BYTES + NVH * DV * DK * 4  # one state slot: conv, then SSM (as vLLM)


class Engine:
    """Decode engine for M rows (independent sequences), with its own caches in vLLM's
    layouts: KV blocks of `block` positions handed out in a shuffled order (block 0 is
    vLLM's null block), and GDN state slots 1..M (slot 0 is the null slot)."""

    def __init__(
        self,
        PK: Packed,
        M: int,
        maxlen: int = 4096,
        layers: int = NL,
        device="cuda",
        block: int = 208,
        xpack: bool = False,
    ):
        self.PK, self.M, self.maxlen, self.layers = PK, M, maxlen, layers
        cus = torch.cuda.get_device_properties(0).multi_processor_count
        nb = -(-maxlen // block)
        # a block as vLLM's ROCm backends keep it: a K and a V half of NKV HD block
        # elements, [position][head][HD] (TRITON_ATTN) or x-packed per head (ROCM_ATTN)
        kv_s = (
            2 * block * NKV * HD,
            HD * (block if xpack else 1),
            NKV * HD,
            block * NKV * HD,
        )
        self.cfg = KK.Cfg(M, cus, maxlen, block, nb, *kv_s, xpack, PAGE, PAGE)
        self.launch = KK.build(self.cfg)
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=device)  # noqa: E731
        ng, na = NL * 3 // 4, NL // 4
        self.hb = z(3, M, H)  # residual h0, h1 (fp32), then their bf16 copies
        self.ssq = z(2, M, H // 16)
        self.y = z(M, max(INP_N, QKV_N, 2 * FFN))
        self.y2 = z(M, VAL_DIM)
        self.hssq = z(M, NVH, 8)
        self.part = z(self.cfg.part_floats())
        self.cnt = z(8 * KK.CNT_STRIDE, dt=torch.int32)
        self.flags = z((cus + KK.EPOCH_WORD_LINE + 1) * 32, dt=torch.int32)
        self.rec = z(M * NKV * self.cfg.ncmax * X.REC)
        self.tok = z(M, dt=torch.int32)
        self.pos = z(M, dt=torch.int32)
        self.logits = z(M, VOCAB)
        cos, sin = rope_cos_sin(torch.arange(maxlen, device=device))
        self.rope = torch.cat([cos[:, :32], sin[:, :32]], 1).contiguous()
        g = torch.Generator().manual_seed(0)
        self.bt = (
            (torch.randperm(M * nb, generator=g) + 1)
            .reshape(M, nb)
            .to(torch.int32)
            .to(device)
        )
        self.slot = z(M, dt=torch.int32)  # >= 0: every row is live
        self.sidx = torch.arange(1, M + 1, dtype=torch.int32, device=device)
        self.kv = z(na, M * nb + 1, 2, block, NKV * HD, dt=torch.bfloat16)
        self.state = z(ng, M + 1, PAGE, dt=torch.uint8)
        g = [0] * KK.NGLOB
        g[KK.G_EMBED], g[KK.G_LM], g[KK.G_FNORM] = (
            PK.embed.data_ptr(),
            PK.lm_head.data_ptr(),
            PK.final_norm.data_ptr(),
        )
        g[KK.G_ROPE], g[KK.G_HB], g[KK.G_SSQ] = (
            self.rope.data_ptr(),
            self.hb.data_ptr(),
            self.ssq.data_ptr(),
        )
        g[KK.G_Y], g[KK.G_Y2], g[KK.G_HSSQ] = (
            self.y.data_ptr(),
            self.y2.data_ptr(),
            self.hssq.data_ptr(),
        )
        g[KK.G_PART], g[KK.G_CNT], g[KK.G_FLAGS] = (
            self.part.data_ptr(),
            self.cnt.data_ptr(),
            self.flags.data_ptr(),
        )
        g[KK.G_REC], g[KK.G_TOK], g[KK.G_POS] = (
            self.rec.data_ptr(),
            self.tok.data_ptr(),
            self.pos.data_ptr(),
        )
        g[KK.G_OUT] = self.logits.data_ptr()
        self.glob = torch.tensor(g, dtype=torch.int64, device=device)
        tab = torch.zeros(NL, KK.NTAB, dtype=torch.int64)
        gi = ai = 0
        for i in range(layers):
            sm = lambda n: PK.small[(i, n)].data_ptr()  # noqa: E731
            if is_attn(i):
                mt, ms = PK.mats[(i, "qkv")]
                ot, os_ = PK.mats[(i, "o")]
                tab[i, KK.T_QN], tab[i, KK.T_KN] = (
                    sm("self_attn.q_norm.weight"),
                    sm("self_attn.k_norm.weight"),
                )
                tab[i, KK.T_KV] = self.kv[ai].data_ptr()
                tab[i, KK.T_IDX], tab[i, KK.T_SLOT] = (
                    self.bt.data_ptr(),
                    self.slot.data_ptr(),
                )
                ai += 1
            else:
                mt, ms = PK.mats[(i, "inp")]
                ot, os_ = PK.mats[(i, "out")]
                tab[i, KK.T_ALOG], tab[i, KK.T_DTB] = (
                    sm("linear_attn.A_log"),
                    sm("linear_attn.dt_bias"),
                )
                tab[i, KK.T_CONVW], tab[i, KK.T_NORMW] = (
                    sm("linear_attn.conv1d.weight"),
                    sm("linear_attn.norm.weight"),
                )
                tab[i, KK.T_CONV] = self.state[gi].data_ptr()
                tab[i, KK.T_SSM] = self.state[gi].data_ptr() + CONV_BYTES
                tab[i, KK.T_IDX] = self.sidx.data_ptr()
                gi += 1
            gt, gs = PK.mats[(i, "gu")]
            dt, ds = PK.mats[(i, "down")]
            tab[i, : KK.T_POST_LN + 1] = torch.tensor(
                [
                    mt.data_ptr(),
                    ms.data_ptr(),
                    ot.data_ptr(),
                    os_.data_ptr(),
                    gt.data_ptr(),
                    gs.data_ptr(),
                    dt.data_ptr(),
                    ds.data_ptr(),
                    sm("input_layernorm.weight"),
                    sm("post_attention_layernorm.weight"),
                ]
            )
        self.tab = tab.to(device)

    def reset(self):
        for t in (self.state, self.kv, self.pos):
            t.zero_()

    def step(self, tok: torch.Tensor, stream=None):
        """One decode step for all M rows. tok: [M] int. Returns fp32 logits [M, VOCAB]
        (a view, overwritten by the next step)."""
        self.tok.copy_(tok.to(torch.int32))
        self.launch(
            self.tab.data_ptr(),
            self.glob.data_ptr(),
            self.layers,
            stream=stream or torch.cuda.current_stream(),
        )
        self.pos += 1
        return self.logits
