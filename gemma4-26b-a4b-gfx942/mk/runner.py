"""Host side of the megakernel: buffers, pointer tables, one launch per decode step.

Structure from qwen3.6-35b-a3b-gfx942/mk/runner.py at megakernels e2a6fb8.
"""

from __future__ import annotations

import dataclasses
import os

import torch

from . import attention as A
from . import kernel as KK
from . import moe as MO
from .dims import EFFP, HD_F, HD_S, NKV_F, NKV_S, NL, NQ, QKV_F, VOCAB, H, is_full
from .pack import Packed
from .reference import rope_cos_sin


def default_chunks() -> dict:
    """The GEMV K chunks of a default Cfg (pack.Packed needs them). MK_CHUNKS="o=2,e2=3"
    overrides some (tuning); Engine then compiles a Cfg with those chunks."""
    f = {x.name: x.default for x in dataclasses.fields(KK.Cfg) if x.name.startswith("c_")}
    ch = {k[2:]: v for k, v in f.items()}
    for kv in filter(None, os.environ.get("MK_CHUNKS", "").split(",")):
        k, v = kv.split("=")
        assert k.strip() in ch, kv
        ch[k.strip()] = int(v)
    return ch


def rope_table(full: bool, maxlen: int, device="cuda"):
    """[maxlen][HD] fp32: cos of the HD/2 angle pairs, then their sin (attention.py)."""
    cos, sin = rope_cos_sin(torch.arange(maxlen, device=device), full)
    return torch.cat([cos, sin], 1).contiguous()


class Cache:
    """Paged KV of one layer type, vLLM TRITON_ATTN style: a [blocks, B, NKV, 2 HD] tensor
    per layer (K | V in the last dimension), viewed [blocks, NKV, B, 2 HD]. Block 0 is the
    null block; each row's blocks are handed out in a shuffled order."""

    def __init__(self, full, nlayers, M, maxlen, B, device, seed):
        self.B = B
        self.HD, self.NKV = (HD_F, NKV_F) if full else (HD_S, NKV_S)
        nb = -(-maxlen // B)
        self.nb = nb
        g = torch.Generator().manual_seed(seed)
        self.bt = (torch.randperm(M * nb, generator=g) + 1).reshape(M, nb).to(torch.int32)
        self.bt = self.bt.to(device)
        self.kv = torch.zeros(nlayers, M * nb + 1, B, self.NKV, 2 * self.HD,
                              dtype=torch.bfloat16, device=device)
        v = self.view(0).stride()
        self.strides = (v[0], v[1], v[2], self.HD)
        self.slot = torch.zeros(M, dtype=torch.int32, device=device)

    def view(self, li):
        """[blocks, NKV, B, 2 HD] (vLLM's per-layer view)."""
        return self.kv[li].permute(0, 2, 1, 3)

    def set_slots(self, pos, valid=None):
        p = pos.long()
        rows = torch.arange(p.shape[0], device=p.device)
        s = self.bt[rows, p // self.B].long() * self.B + p % self.B
        if valid is not None:
            s = torch.where(valid.bool(), s, torch.full_like(s, -1))
        self.slot.copy_(s.to(torch.int32))


class Engine:
    """Decode engine for M rows (independent sequences), with its own caches in vLLM's
    layouts. valid[m] False marks a padding row: it selects no routed expert and writes
    no KV."""

    def __init__(self, PK: Packed, M: int, maxlen: int = 4096, layers: int = NL,
                 device="cuda", block_s: int = 16, block_f: int = 32, lm_head: bool = True,
                 debug_route: bool = False, fence: bool | None = None):
        self.PK, self.M, self.maxlen, self.layers = PK, M, maxlen, layers
        cus = torch.cuda.get_device_properties(0).multi_processor_count
        ns = sum(not is_full(i) for i in range(layers))
        self.cs = Cache(False, max(ns, 1), M, maxlen, block_s, device, 1)
        self.cf = Cache(True, max(layers - ns, 1), M, maxlen, block_f, device, 2)
        if fence is None:  # MK_FENCE=1: L2 fences at every grid barrier (debug)
            fence = os.environ.get("MK_FENCE", "0") == "1"
        flags = {}
        ftype = {f.name: f.type for f in dataclasses.fields(KK.Cfg)}
        for kv in filter(None, os.environ.get("MK_CFG", "").split(",")):
            k, v = kv.split("=")
            flags[k] = int(v) if ftype[k] in (int, "int") else bool(int(v))
        self.cfg = KK.Cfg(
            M, cus, maxlen, block_s, self.cs.nb, self.cs.strides, block_f, self.cf.nb,
            self.cf.strides, lm_head, debug_route=debug_route, fence=fence,
            **{"c_" + k: v for k, v in PK.chunks.items()}, **flags,
        )
        self.launch = KK.build(self.cfg)
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=device)  # noqa: E731
        self.buf = {k: z(M, H) for k in ("h", "h1", "a", "d", "e")}
        self.ssq = {k: z(M, H // 16) for k in ("a", "d", "e")}
        self.rstd1 = z(M)
        # router_publish: the gate_up inputs, fp8 [2M][H] and scales [2M][H / 128]
        self.xg = z(2 * M * H // 4, dt=torch.int32)
        self.xgs = z(2 * M * (H // 128))
        # combine_rows: the next layer's qkv input, fp8 [M][H] and scales [M][H / 128]
        self.xq = z(M * H // 4, dt=torch.int32)
        self.xqs = z(M * (H // 128))
        self.y = z(M, QKV_F)
        self.o = z(M, NQ * HD_F)
        self.part = z(self.cfg.part_floats())
        self.cnt = z(len(KK.CNT_REGION) * KK.CNT_STRIDE, dt=torch.int32)
        self.flags = z((cus + KK.EPOCH_WORD_LINE + 1) * 32, dt=torch.int32)
        gs, gf = self.cfg.geo(False), self.cfg.geo(True)
        self.rec = z(max(gs.nrec(M), gf.nrec(M)))
        self.tok = z(M, dt=torch.int32)
        self.pos = z(M, dt=torch.int32)
        # router logits: [NL][M][LGN] with debug_route (every layer), else one layer's
        self.router = z(NL if debug_route else 1, M, MO.LGN)
        self.act = z(M, MO.NSLOT, EFFP)  # columns EFF.. stay 0 (the padded down K)
        self.rowv = torch.ones(M, dtype=torch.int32, device=device)
        # debug_route (moe.route_hash): per layer and CTA the hash of its routing
        # tables (invariant I5), then CTA 0's picks [NL][M][8] and weights [NL][M][NSLOT]
        nsl = MO.NSLOT
        self.dbg = z(NL * cus + NL * M * 8 + NL * M * nsl, dt=torch.int32)
        self.route_hash = self.dbg[: NL * cus].view(NL, cus)
        self.route_pick = self.dbg[NL * cus: NL * cus + NL * M * 8].view(NL, M, 8)
        self.route_w = self.dbg[NL * cus + NL * M * 8:].view(torch.float32).view(NL, M, nsl)
        self.out = z(M, VOCAB) if lm_head else z(M, H, dt=torch.bfloat16)
        self.rope_s = rope_table(False, maxlen, device)
        self.rope_f = rope_table(True, maxlen, device)
        gp = [0] * (KK.NGLOB + KK.NGEO)
        gp[KK.G_GEOM :] = self.cfg.geom_slots()
        gp[KK.G_EMBED] = PK.embed.data_ptr()
        gp[KK.G_LM] = PK.lm_head.data_ptr()
        gp[KK.G_FNORM] = PK.final_norm.data_ptr()
        gp[KK.G_ROPE_S] = self.rope_s.data_ptr()
        gp[KK.G_ROPE_F] = self.rope_f.data_ptr()
        for k, g in (("h", KK.G_H), ("h1", KK.G_H1), ("a", KK.G_A), ("d", KK.G_D),
                     ("e", KK.G_E)):
            gp[g] = self.buf[k].data_ptr()
        for k, g in (("a", KK.G_SSQA), ("d", KK.G_SSQD), ("e", KK.G_SSQE)):
            gp[g] = self.ssq[k].data_ptr()
        gp[KK.G_RSTD1] = self.rstd1.data_ptr()
        gp[KK.G_XG] = self.xg.data_ptr()
        gp[KK.G_XGS] = self.xgs.data_ptr()
        gp[KK.G_XQ] = self.xq.data_ptr()
        gp[KK.G_XQS] = self.xqs.data_ptr()
        gp[KK.G_Y] = self.y.data_ptr()
        gp[KK.G_O] = self.o.data_ptr()
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
        tab = torch.zeros(NL + 1, KK.NTAB, dtype=torch.int64)
        one = PK.ones.data_ptr()
        for k in (KK.T_POST1, KK.T_POST2, KK.T_POSTF, KK.T_LS):
            tab[0, k] = one  # layer -1: ones, layer_scalar 1.0 (bf16 1.0 in a dword)
        si = fi = 0
        for i in range(layers):
            sm = lambda n: PK.small[(i, n)].data_ptr()  # noqa: E731
            mp = lambda n: [t.data_ptr() for t in PK.mats[(i, n)]]  # noqa: E731
            r = tab[i + 1]
            r[KK.T_QKV_T], r[KK.T_QKV_S] = mp("qkv")
            r[KK.T_O_T], r[KK.T_O_S] = mp("o")
            r[KK.T_RT] = mp("router")[0]
            r[KK.T_E1_T], r[KK.T_E1_S] = mp("e1")
            r[KK.T_E2_T], r[KK.T_E2_S] = mp("e2")
            r[KK.T_IN_LN] = sm("input_layernorm.weight")
            r[KK.T_PA_LN] = sm("post_attention_layernorm.weight")
            r[KK.T_PF_LN] = sm("pre_feedforward_layernorm.weight")
            r[KK.T_PF2_LN] = sm("pre_feedforward_layernorm_2.weight")
            r[KK.T_RSCALE] = sm("router.scale")
            r[KK.T_PES] = sm("router.per_expert_scale")
            r[KK.T_QN] = sm("self_attn.q_norm.weight")
            r[KK.T_KN] = sm("self_attn.k_norm.weight")
            r[KK.T_POST1] = sm("post_feedforward_layernorm_1.weight")
            r[KK.T_POST2] = sm("post_feedforward_layernorm_2.weight")
            r[KK.T_POSTF] = sm("post_feedforward_layernorm.weight")
            r[KK.T_LS] = sm("layer_scalar")
            c = self.cf if is_full(i) else self.cs
            li = fi if is_full(i) else si
            r[KK.T_KV] = c.kv[li].data_ptr()
            r[KK.T_BT] = c.bt.data_ptr()
            r[KK.T_SLOT] = c.slot.data_ptr()
            if is_full(i):
                fi += 1
            else:
                si += 1
        self.tab = tab.to(device)
        self.valid = None

    def reset(self):
        for t in (self.cs.kv, self.cf.kv, self.pos):
            t.zero_()

    def load_kv(self, st, L):
        """Copy the first L positions of a reference.State's contiguous caches (all rows,
        all layers) into the paged caches, and set the positions to st.pos."""
        for c, ks, vs in ((self.cs, st.ks, st.vs), (self.cf, st.kf, st.vf)):
            p = torch.arange(L, device=ks.device)
            for m in range(self.M):
                blk = c.bt[m, p // c.B].long()
                for li in range(min(ks.shape[0], c.kv.shape[0])):
                    v = c.view(li)  # [blocks, NKV, B, 2 HD]
                    v[blk, :, p % c.B, : c.HD] = ks[li, m, :, :L].permute(1, 0, 2).to(v.dtype)
                    v[blk, :, p % c.B, c.HD:] = vs[li, m, :, :L].permute(1, 0, 2).to(v.dtype)
        self.pos.copy_(st.pos.to(torch.int32))

    def step(self, tok: torch.Tensor, valid: torch.Tensor | None = None, stream=None):
        """One decode step for all M rows. tok: [M] int. Returns fp32 logits [M, VOCAB]
        (soft-capped), or the bf16 normed hidden states (a view, overwritten by the next
        step)."""
        self.tok.copy_(tok.to(torch.int32))
        if valid is not None:
            self.rowv.copy_(valid.to(torch.int32))
        v = self.rowv
        self.cs.set_slots(self.pos, v)
        self.cf.set_slots(self.pos, v)
        self.launch(self.tab.data_ptr(), self.glob.data_ptr(), self.layers,
                    stream=stream or torch.cuda.current_stream())
        self.pos += 1
        return self.out


assert A.CH == 64
