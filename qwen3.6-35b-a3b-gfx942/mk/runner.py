"""Host side of the megakernel: buffers, pointer tables, one launch per decode step."""

from __future__ import annotations

import dataclasses
import os

import torch

from . import kernel as KK
from . import mixer as X
from . import moe as MO
from .dims import (
    CONV_DIM,
    DK,
    DV,
    FI,
    H,
    HD,
    INP_N,
    NKV,
    NL,
    NVH,
    QKV_N,
    VAL_DIM,
    VOCAB,
    is_attn,
)
from .pack import Packed
from .reference import rope_cos_sin

CONV_BYTES = 3 * CONV_DIM * 2  # bf16 [3][CONV_DIM]
PAGE = CONV_BYTES + NVH * DV * DK * 4  # one state slot: conv, then SSM (as vLLM)


def default_chunks() -> dict:
    """The GEMV K chunks of a default Cfg (pack.Packed needs them). MK_CHUNKS="lm=4,
    out=2" overrides some (tuning); Engine then compiles a Cfg with those chunks."""
    f = {x.name: x.default for x in dataclasses.fields(KK.Cfg) if x.name.startswith("c_")}
    ch = {k[2:]: v for k, v in f.items()}
    for kv in filter(None, os.environ.get("MK_CHUNKS", "").split(",")):
        k, v = kv.split("=")
        assert k.strip() in ch, kv
        ch[k.strip()] = int(v)
    return ch


class Engine:
    """Decode engine for M rows (independent sequences), with its own caches in vLLM's
    layouts: KV blocks of `block` positions handed out in a shuffled order (block 0 is
    vLLM's null block), and GDN state slots 1..M (slot 0 is the null slot).
    valid[m] False marks a padding row: it selects no routed expert."""

    def __init__(
        self,
        PK: Packed,
        M: int,
        maxlen: int = 4096,
        layers: int = NL,
        device="cuda",
        block: int = 208,
        xpack: bool = False,
        lm_head: bool = True,
        debug_route: bool = False,
        fence: bool | None = None,
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
        if fence is None:  # MK_FENCE=1: L2 fences at every grid barrier (debug)
            fence = os.environ.get("MK_FENCE", "0") == "1"
        # MK_CFG="name=1,other=0": override Cfg flags (A/B experiments)
        over = dict(kv.split("=") for kv in filter(None, os.environ.get("MK_CFG", "").split(",")))
        self.cfg = KK.Cfg(
            M, cus, maxlen, block, nb, *kv_s, xpack, PAGE, PAGE, lm_head,
            debug_route=debug_route, fence=fence,
            **{"c_" + k: v for k, v in PK.chunks.items()},
            **{k: bool(int(v)) for k, v in over.items()},
        )
        self.launch = KK.build(self.cfg)
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=device)  # noqa: E731
        ng, na = NL * 3 // 4, NL // 4
        self.hb = z(3, M, H)  # residual h0, h1 (fp32), then their bf16 copies
        self.ssq = z(2, M, H // 16)
        self.y = z(M, max(INP_N, QKV_N))
        self.y2 = z(M, VAL_DIM)
        self.hssq = z(M, NVH, 8)
        self.part = z(self.cfg.part_floats())
        self.cnt = z(len(KK.CNT_REGION) * KK.CNT_STRIDE, dt=torch.int32)
        self.flags = z((cus + KK.EPOCH_WORD_LINE + 1) * 32, dt=torch.int32)
        self.rec = z(M * NKV * self.cfg.ncmax * X.REC)
        self.tok = z(M, dt=torch.int32)
        self.pos = z(M, dt=torch.int32)
        self.router = z(M, MO.LGN)  # router logits (debug: last layer's)
        self.act = z(M, MO.NSLOT, FI)
        self.rowv = torch.ones(M, dtype=torch.int32, device=device)
        # debug_route (moe.route_hash): per layer and CTA the hash of its routing
        # tables (invariant I5), then CTA 0's picks [NL][M][8] and weights [NL][M][9]
        self.dbg = z(NL * cus + NL * M * 8 + NL * M * 9, dt=torch.int32)
        self.route_hash = self.dbg[: NL * cus].view(NL, cus)
        self.route_pick = self.dbg[NL * cus : NL * cus + NL * M * 8].view(NL, M, 8)
        self.route_w = self.dbg[NL * cus + NL * M * 8 :].view(torch.float32).view(NL, M, 9)
        self.out = z(M, VOCAB) if lm_head else z(M, H, dt=torch.bfloat16)
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
        gp = [0] * KK.NGLOB
        gp[KK.G_EMBED] = PK.embed.data_ptr()
        gp[KK.G_LM] = PK.lm_head.data_ptr()
        gp[KK.G_FNORM] = PK.final_norm.data_ptr()
        gp[KK.G_ROPE] = self.rope.data_ptr()
        gp[KK.G_HB] = self.hb.data_ptr()
        gp[KK.G_SSQ] = self.ssq.data_ptr()
        gp[KK.G_Y] = self.y.data_ptr()
        gp[KK.G_Y2] = self.y2.data_ptr()
        gp[KK.G_HSSQ] = self.hssq.data_ptr()
        gp[KK.G_PART] = self.part.data_ptr()
        gp[KK.G_CNT] = self.cnt.data_ptr()
        gp[KK.G_FLAGS] = self.flags.data_ptr()
        gp[KK.G_REC] = self.rec.data_ptr()
        gp[KK.G_TOK] = self.tok.data_ptr()
        gp[KK.G_POS] = self.pos.data_ptr()
        gp[KK.G_OUT] = self.out.data_ptr()
        gp[KK.G_LOGIT] = self.router.data_ptr()
        gp[KK.G_ACT] = self.act.data_ptr()
        gp[KK.G_ROWV] = self.rowv.data_ptr()
        gp[KK.G_SPARE] = self.dbg.data_ptr()
        self.glob = torch.tensor(gp, dtype=torch.int64, device=device)
        tab = torch.zeros(NL, KK.NTAB, dtype=torch.int64)
        gi = ai = 0
        for i in range(layers):
            sm = lambda n: PK.small[(i, n)].data_ptr()  # noqa: E731
            mp = lambda n: [t.data_ptr() for t in PK.mats[(i, n)]]  # noqa: E731
            if is_attn(i):
                mt, ms = mp("qkv")
                ot, os_ = mp("o")
                tab[i, KK.T_QN] = sm("self_attn.q_norm.weight")
                tab[i, KK.T_KN] = sm("self_attn.k_norm.weight")
                tab[i, KK.T_KV] = self.kv[ai].data_ptr()
                tab[i, KK.T_IDX] = self.bt.data_ptr()
                tab[i, KK.T_SLOT] = self.slot.data_ptr()
                ai += 1
            else:
                mt, ms = mp("inp")
                ot, os_ = mp("out")
                tab[i, KK.T_BA] = mp("ba")[0]
                tab[i, KK.T_ALOG] = sm("linear_attn.A_log")
                tab[i, KK.T_DTB] = sm("linear_attn.dt_bias")
                tab[i, KK.T_CONVW] = sm("linear_attn.conv1d.weight")
                tab[i, KK.T_NORMW] = sm("linear_attn.norm.weight")
                tab[i, KK.T_CONV] = self.state[gi].data_ptr()
                tab[i, KK.T_SSM] = self.state[gi].data_ptr() + CONV_BYTES
                tab[i, KK.T_IDX] = self.sidx.data_ptr()
                gi += 1
            tab[i, KK.T_MIX_T], tab[i, KK.T_MIX_S] = mt, ms
            tab[i, KK.T_OUT_T], tab[i, KK.T_OUT_S] = ot, os_
            tab[i, KK.T_RT] = mp("router")[0]
            tab[i, KK.T_E1_T], tab[i, KK.T_E1_S] = mp("e1")
            tab[i, KK.T_E2_T], tab[i, KK.T_E2_S] = mp("e2")
            tab[i, KK.T_IN_LN] = sm("input_layernorm.weight")
            tab[i, KK.T_POST_LN] = sm("post_attention_layernorm.weight")
        self.tab = tab.to(device)

    def reset(self):
        for t in (self.state, self.kv, self.pos):
            t.zero_()

    def step(self, tok: torch.Tensor, valid: torch.Tensor | None = None, stream=None):
        """One decode step for all M rows. tok: [M] int. Returns fp32 logits [M, VOCAB]
        (or the bf16 normed hidden states; a view, overwritten by the next step)."""
        self.tok.copy_(tok.to(torch.int32))
        if valid is not None:
            self.rowv.copy_(valid.to(torch.int32))
        self.launch(
            self.tab.data_ptr(),
            self.glob.data_ptr(),
            self.layers,
            stream=stream or torch.cuda.current_stream(),
        )
        self.pos += 1
        return self.out

    def hidden(self):
        """The residual h0 after the last layer run (fp32 [M, H])."""
        return self.hb[0]
