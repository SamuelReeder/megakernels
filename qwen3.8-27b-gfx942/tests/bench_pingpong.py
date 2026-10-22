"""Microbenchmark: one-way flag latency between two CTAs (device-scope store/poll).

CTA 0 and CTA PEER bounce a counter ITERS times; one-way latency = time / (2 ITERS).
CM selects the cache bits of both the store and the poll load (16 = sc1 device scope,
1 = sc0, 17 = sc0|sc1 system scope).
Usage: python tests/bench_pingpong.py PEER [CM] [ITERS]
"""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl.compiler.ast_rewriter import ASTRewriter  # noqa: E402
from flydsl.expr import rocdl  # noqa: E402
from flydsl.expr.typing import Int32, Int64, T  # noqa: E402

from mk.vendor import buffer_ops as bo  # noqa: E402

PEER = int(sys.argv[1]) if len(sys.argv) > 1 else 1
CM = int(sys.argv[2]) if len(sys.argv) > 2 else 16
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
CUS = torch.cuda.get_device_properties(0).multi_processor_count


@ASTRewriter.transform
def wait_ge(r, slot, val):
    v = fx.Int32(bo.buffer_load(r, slot, vec_width=1, dtype=T.i32, cache_modifier=CM))
    while v < val:
        rocdl.s_sleep(0)
        v = fx.Int32(bo.buffer_load(r, slot, vec_width=1, dtype=T.i32, cache_modifier=CM))


@fx.struct
class Big:
    pad: fx.Array[fx.Int32, 10240, 16]   # 40 KiB: one CTA per CU


@flyc.kernel(name=f"pp_{PEER}_{CM}_{ITERS}", known_block_size=[64, 1, 1])
def kpp(buf: Int64, base: Int32):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    pad = fx.SharedAllocator().allocate(Big).peek().pad.ptr
    fx.ptr_store(fx.Int32(0), pad + tid)
    r = bo.create_buffer_resource_from_addr(buf)
    if tid == 0:
        if bid == 0:
            for i in range(0, ITERS):
                bo.buffer_store(base + 2 * i + 1, r, 0, cache_modifier=CM)
                wait_ge(r, 64, base + 2 * i + 2)
        if bid == PEER:
            for i in range(0, ITERS):
                wait_ge(r, 0, base + 2 * i + 1)
                bo.buffer_store(base + 2 * i + 2, r, 64, cache_modifier=CM)


@flyc.jit
def launch(buf: Int64, base: Int32, stream: fx.Stream = fx.Stream(None)):
    kpp(buf, base).launch(grid=(CUS,), block=(64,), stream=stream)


def main():
    buf = torch.zeros(1024, dtype=torch.int32, device="cuda")
    launch(buf.data_ptr(), 0)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    launch(buf.data_ptr(), 2 * ITERS, stream=torch.cuda.current_stream())
    e1.record()
    torch.cuda.synchronize()
    ms = e0.elapsed_time(e1)
    assert buf[0].item() == 4 * ITERS - 1 and buf[64].item() == 4 * ITERS, (buf[0].item(), buf[64].item())
    print(f"peer {PEER} cm {CM}: one-way {ms * 1e3 / (2 * ITERS) * 1e3:.0f} ns")
    import time
    t0 = time.time()
    launch(buf.data_ptr(), 4 * ITERS, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    print(f"wall: one-way {(time.time() - t0) / (2 * ITERS) * 1e9:.0f} ns")


if __name__ == "__main__":
    main()
