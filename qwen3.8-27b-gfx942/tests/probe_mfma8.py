"""Probe: operand/result layout of v_mfma_f32_16x16x32_fp8_fp8 and the e2m1 decode.

Assumed: A lane l -> A[l % 16][8 (l // 16) + 0..7]; B lane l -> B[8 (l // 16) + 0..7][l % 16];
D lane l reg r -> D[4 (l // 16) + r][l % 16].
"""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl.expr.typing import Int64, T  # noqa: E402

from mk import fp4  # noqa: E402
from mk.vendor import buffer_ops as bo  # noqa: E402

H = fp4.device_helpers()


@flyc.kernel(name="probe_mfma8", known_block_size=[64, 1, 1])
def kp(ap: Int64, bp: Int64, wp: Int64, dp: Int64, d2p: Int64):
    lane = fx.thread_idx.x
    ra, rb, rw = (bo.create_buffer_resource_from_addr(p) for p in (ap, bp, wp))
    rd, rd2 = bo.create_buffer_resource_from_addr(dp), bo.create_buffer_resource_from_addr(d2p)
    a = fx.Vector(bo.buffer_load(ra, lane * 2, vec_width=2, dtype=T.i32)).bitcast(fx.Int64)[0]
    b = fx.Vector(bo.buffer_load(rb, lane * 2, vec_width=2, dtype=T.i32)).bitcast(fx.Int64)[0]
    d = H["mfma"](a, b, fx.Vector.filled(4, 0.0, fx.Float32))
    bo.buffer_store(d, rd, lane * 4)
    # decoded e2m1 dword (lane's B operand) against the same A
    w = fx.Int32(bo.buffer_load(rw, lane, vec_width=1, dtype=T.i32))
    d2 = H["mfma"](a, H["dec_dword"](w), fx.Vector.filled(4, 0.0, fx.Float32))
    bo.buffer_store(d2, rd2, lane * 4)


@flyc.jit
def launch(ap: Int64, bp: Int64, wp: Int64, dp: Int64, d2p: Int64, stream: fx.Stream = fx.Stream(None)):
    kp(ap, bp, wp, dp, d2p).launch(grid=(1,), block=(64,), stream=stream)


def main():
    torch.manual_seed(0)
    A = torch.randn(16, 32).to(torch.float8_e4m3fnuz)
    B = torch.randn(32, 16).to(torch.float8_e4m3fnuz)
    a_l = torch.stack([A[l % 16, 8 * (l // 16): 8 * (l // 16) + 8] for l in range(64)])
    b_l = torch.stack([B[8 * (l // 16): 8 * (l // 16) + 8, l % 16] for l in range(64)])
    # e2m1 weights: W[k, n] codes; lane l dword = codes of B column l%16, k 8gl..8gl+7
    codes = torch.randint(0, 16, (32, 16), dtype=torch.uint8)
    codes[codes == 8] = 0
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
    Wv = lut[codes.long()]
    wl = []
    for l in range(64):
        c = codes[8 * (l // 16): 8 * (l // 16) + 8, l % 16]
        byte = c[:4] | (c[4:] << 4)
        wl.append(int(byte[0]) | int(byte[1]) << 8 | int(byte[2]) << 16 | int(byte[3]) << 24)
    w_t = torch.tensor(wl, dtype=torch.int64).to(torch.int32).cuda()
    ap = a_l.view(torch.uint8).contiguous().cuda()
    bp = b_l.view(torch.uint8).contiguous().cuda()
    d = torch.zeros(64, 4, device="cuda")
    d2 = torch.zeros(64, 4, device="cuda")
    launch(ap.data_ptr(), bp.data_ptr(), w_t.data_ptr(), d.data_ptr(), d2.data_ptr())
    torch.cuda.synchronize()
    ref = A.float() @ B.float()
    ref2 = A.float() @ Wv
    D = torch.zeros(16, 16)
    D2 = torch.zeros(16, 16)
    for l in range(64):
        for r in range(4):
            D[4 * (l // 16) + r, l % 16] = d[l, r]
            D2[4 * (l // 16) + r, l % 16] = d2[l, r]
    print("layout err", (D - ref).abs().max().item(), "decode err", (D2 - ref2).abs().max().item())
    print("D[0,:4]", D[0, :4].tolist(), "ref", ref[0, :4].tolist())
    print("D2[0,:4]", D2[0, :4].tolist(), "ref2", ref2[0, :4].tolist())


if __name__ == "__main__":
    main()
