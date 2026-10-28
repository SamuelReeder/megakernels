"""Tensor-parallel harness of the standalone engine (tests and benchmarks).

MK_TP=2 (one process per GPU, `srun -n2`; SLURM_PROCID is the rank) turns a test into
its TP2 run: each rank packs its shard and runs its own Engine; the ranks meet over a
gloo group (peer-buffer handshake, token broadcast, logits gather, rank checks).
MK_AR selects the all-reduce partials ("fp32" or "bf16"), MK_AR_LOCAL=1 the timing-only
build without peer traffic (Cfg.ar_local). Without MK_TP everything is the TP1 path.
The process group runs gloo for CPU tensors and RCCL for GPU tensors (benchmarks).
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .pack import Packed
from .runner import Engine


@dataclass
class Ctx:
    tp: int
    rank: int
    group: object
    ar_dtype: str
    ar_local: bool = False

    @property
    def main(self):
        return self.rank == 0


def init() -> Ctx:
    tp = int(os.environ.get("MK_TP", "1"))
    ar = os.environ.get("MK_AR", "fp32")
    local = os.environ.get("MK_AR_LOCAL", "0") == "1"
    if tp == 1:
        return Ctx(1, 0, None, ar)
    rank = int(os.environ["SLURM_PROCID"])
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl",
        init_method=f"tcp://127.0.0.1:{os.environ.get('MK_PORT', '29541')}",
        rank=rank,
        world_size=tp,
        timeout=datetime.timedelta(seconds=1800),
    )
    return Ctx(tp, rank, dist.group.WORLD, ar, local)


def packed(ctx: Ctx, W, **kw) -> Packed:
    """MK_CHUNKS="out=1,down=2" overrides GEMV K chunks (tuning experiments)."""
    ov = os.environ.get("MK_CHUNKS", "")
    ov = tuple((k, int(v)) for k, v in (x.split("=") for x in ov.split(",") if x))
    return Packed(W, tp=ctx.tp, rank=ctx.rank, chunk_override=ov, **kw)


class TPEngine:
    """One rank's Engine plus the host side of a TP step: rank 0's tokens go to every
    rank, every rank gets the full logits (the vocab slices in rank order), and the
    residual stream is compared bitwise across ranks after every step."""

    def __init__(self, ctx: Ctx, PK: Packed, M: int, **kw):
        self.ctx = ctx
        self.eng = Engine(
            PK, M, group=ctx.group, ar_dtype=ctx.ar_dtype, ar_local=ctx.ar_local,
            **kw
        )
        self.ranks_equal = True
        self.steps = 0

    def reset(self):
        self.eng.reset()

    def step(self, tok: torch.Tensor) -> torch.Tensor:
        t = tok.to(torch.int64).cpu().contiguous()
        dist.broadcast(t, 0, group=self.ctx.group)
        self.eng.step(t.cuda())
        torch.cuda.synchronize()
        parts = [torch.empty_like(self.eng.logits.cpu()) for _ in range(self.ctx.tp)]
        dist.all_gather(parts, self.eng.logits.cpu(), group=self.ctx.group)
        # the residual after the last layer (h0, fp32) and its bf16 copy
        h = self.eng.hb[[0, 2]].cpu().view(torch.int32).flatten()
        hs = [torch.empty_like(h) for _ in range(self.ctx.tp)]
        dist.all_gather(hs, h, group=self.ctx.group)
        same = all(torch.equal(hs[0], x) for x in hs[1:])
        if not same and self.ranks_equal:
            print(f"[rank {self.ctx.rank}] ranks differ at step {self.steps}", flush=True)
        self.ranks_equal &= same
        self.steps += 1
        return torch.cat(parts, -1).cuda()


def engine(ctx: Ctx, PK: Packed, M: int, **kw):
    if ctx.tp == 1:
        return Engine(PK, M, **kw)
    return TPEngine(ctx, PK, M, **kw)


def agree(ctx: Ctx, ok: bool) -> bool:
    """Every rank's verdict (all must pass)."""
    if ctx.tp == 1:
        return ok
    f = torch.tensor([int(ok)], dtype=torch.int32)
    dist.all_reduce(f, op=dist.ReduceOp.MIN, group=ctx.group)
    return bool(f.item())


def raw(eng):
    """The rank's Engine (no host work per step)."""
    return eng.eng if isinstance(eng, TPEngine) else eng


def next_token(ctx: Ctx, logits: torch.Tensor) -> torch.Tensor:
    """Greedy tokens from the ranks' vocab slices, on the device: each rank's (max, index),
    one RCCL all-gather, the first rank holding the largest value (the argmax of the
    concatenated logits)."""
    if ctx.tp == 1:
        return logits.argmax(-1)
    v, i = logits.max(-1)
    mine = torch.stack([v, i.float()], -1).contiguous()  # [M, 2]; i < 2^24 is exact
    allv = torch.empty(ctx.tp, *mine.shape, device=mine.device)
    dist.all_gather_into_tensor(allv, mine, group=ctx.group)
    r = allv[:, :, 0].argmax(0)  # first maximum: the lowest rank
    idx = allv[:, :, 1].gather(0, r[None])[0].long()
    return idx + r * logits.shape[-1]
