"""Microbenchmark: grid-wide barrier latency of a persistent FlyDSL kernel on gfx942.

Each CTA: thread 0 adds 1 to a device-scope counter, then spins (device-scope
loads, s_sleep) until the counter reaches (i + 1) * CTAS; then s_barrier.
Usage: python tests/bench_barrier.py [ITERS] [SLEEP]
"""
import os
import sys

import torch

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl._mlir import ir  # noqa: E402
from flydsl._mlir.dialects import llvm  # noqa: E402
from flydsl.compiler.ast_rewriter import ASTRewriter  # noqa: E402
from flydsl.expr import gpu, rocdl  # noqa: E402
from flydsl.expr.typing import Int32, Int64, T  # noqa: E402

from mk.vendor import buffer_ops as bo  # noqa: E402

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SLEEP = int(sys.argv[2]) if len(sys.argv) > 2 else 1
CUS = torch.cuda.get_device_properties(0).multi_processor_count
THREADS = 512
CM_DEV = 16  # sc1: device scope on gfx94x


def gptr(addr):
    return llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), fx.Int64(addr).ir_value()).result


@ASTRewriter.transform
def grid_barrier(rc, cnt_ptr, target, tid):
    if tid == 0:
        rocdl.s_waitcnt(vmcnt=0)
        fx.atomic_add(cnt_ptr, fx.Int32(1), syncscope=fx.rocdl.SyncScope.Agent)
        v = fx.Int32(bo.buffer_load(rc, 0, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))
        while v < target:
            rocdl.s_sleep(SLEEP)
            v = fx.Int32(bo.buffer_load(rc, 0, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))
    gpu.barrier()


@ASTRewriter.transform
def flag_barrier(rf, epoch, tid, bid):
    """CTA b stores epoch to flag b; threads t < CUS wait for flag t >= epoch."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        bo.buffer_store(epoch, rf, bid * STRIDE, cache_modifier=CM_DEV)
    if tid < CUS:
        v = fx.Int32(bo.buffer_load(rf, tid * STRIDE, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))
        while v < epoch:
            rocdl.s_sleep(SLEEP)
            v = fx.Int32(bo.buffer_load(rf, tid * STRIDE, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))
    gpu.barrier()


@ASTRewriter.transform
def xcd_barrier(rc, cbase, epoch, tid, bid):
    """Per-XCD counter (CTA b on XCD b % 8); the XCD's last arriver bumps the global
    counter (slot 0); everyone waits for the global counter to reach 8 epoch."""
    if tid == 0:
        rocdl.s_waitcnt(vmcnt=0)
        x = bid % 8
        per = (CUS - x + 7) // 8
        old = fx.atomic_add(gptr(cbase + (1 + x) * 128), fx.Int32(1), syncscope=fx.rocdl.SyncScope.Agent)
        if (old + 1) == per * epoch:
            fx.atomic_add(gptr(cbase), fx.Int32(1), syncscope=fx.rocdl.SyncScope.Agent)
        v = fx.Int32(bo.buffer_load(rc, 0, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))
        while v < 8 * epoch:
            rocdl.s_sleep(SLEEP)
            v = fx.Int32(bo.buffer_load(rc, 0, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))
    gpu.barrier()


@ASTRewriter.transform
def poll_ge(r, slot, val):
    v = fx.Int32(bo.buffer_load(r, slot, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))
    while v < val:
        rocdl.s_sleep(SLEEP)
        v = fx.Int32(bo.buffer_load(r, slot, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))


LINE = 32  # i32 per 128 B line


@ASTRewriter.transform
def tree_barrier(rf, epoch, tid, bid):
    """Arrive: CTA b stores epoch to its line. Leaders (CTA x < 8) wait for the CTAs
    of XCD x (b % 8 == x), then store epoch to leader line x. All wait for 8 leaders."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        bo.buffer_store(epoch, rf, bid * LINE, cache_modifier=CM_DEV)
    if bid < 8:
        nx = (CUS - bid + 7) // 8
        if tid < nx:
            poll_ge(rf, (bid + 8 * tid) * LINE, epoch)
        gpu.barrier()
        if tid == 0:
            bo.buffer_store(epoch, rf, (CUS + bid) * LINE, cache_modifier=CM_DEV)
    if tid < 8:
        poll_ge(rf, (CUS + tid) * LINE, epoch)
    gpu.barrier()


NWORD = (CUS + 15) // 16  # dwordx4 loads covering all arrival bytes


@ASTRewriter.transform
def byte_barrier(rf, epoch, tid, bid):
    """CTA b stores (epoch & 0xFF) to byte b; threads t < NWORD poll 16 bytes each
    until all equal the epoch byte (bytes past CUS are pre-set by the host)."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    e8 = epoch & 0xFF
    if tid == 0:
        bo.buffer_store(fx.Int8(e8), rf, bid, cache_modifier=CM_DEV)
    if tid < NWORD:
        want = e8 * 0x01010101
        v = fx.Vector(bo.buffer_load(rf, tid * 4, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV))
        bad = (v[0] != want) | (v[1] != want) | (v[2] != want) | (v[3] != want)
        while bad:
            rocdl.s_sleep(SLEEP)
            v = fx.Vector(bo.buffer_load(rf, tid * 4, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV))
            bad = (v[0] != want) | (v[1] != want) | (v[2] != want) | (v[3] != want)
    gpu.barrier()


KIND = os.environ.get("KIND", "atomic")
STRIDE = int(os.environ.get("STRIDE", "1"))


@flyc.kernel(name=f"bar_{KIND}_{ITERS}_{SLEEP}_{STRIDE}", known_block_size=[THREADS, 1, 1])
def kbar(cnt: Int64, base: Int32):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    rc = bo.create_buffer_resource_from_addr(cnt)
    cp = gptr(cnt)
    for i in range(0, ITERS):
        if KIND == "atomic":
            grid_barrier(rc, cp, (base + i + 1) * CUS, tid)
        elif KIND == "byte":
            byte_barrier(rc, base + i + 1, tid, bid)
        elif KIND == "tree":
            tree_barrier(rc, base + i + 1, tid, bid)
        elif KIND == "flag":
            flag_barrier(rc, base + i + 1, tid, bid)
        else:
            xcd_barrier(rc, cnt, base + i + 1, tid, bid)




@flyc.jit
def launch(cnt: Int64, base: Int32, stream: fx.Stream = fx.Stream(None)):
    kbar(cnt, base).launch(grid=(CUS,), block=(THREADS,), stream=stream)


def main():
    cnt = torch.zeros(CUS * 64 + 4096, dtype=torch.int32, device="cuda")
    launch(cnt.data_ptr(), 0)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    base = ITERS
    e0.record()
    launch(cnt.data_ptr(), base, stream=torch.cuda.current_stream())
    e1.record()
    torch.cuda.synchronize()
    ms = e0.elapsed_time(e1)
    print(f"{KIND} stride {STRIDE}: {ITERS} barriers, sleep {SLEEP}: {ms * 1e3 / ITERS:.2f} us per barrier")


if __name__ == "__main__":
    main()
