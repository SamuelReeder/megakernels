"""Probe: v_cvt_pk_fp8_f32 (gfx942) vs torch float8_e4m3fnuz rounding."""
import sys
import torch
sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl.expr.typing import Int64, T  # noqa: E402
from mk import fp4  # noqa: E402
from mk.vendor import buffer_ops as bo  # noqa: E402
H = fp4.device_helpers()
NT = 256

@flyc.kernel(name="probe_cvt8", known_block_size=[NT, 1, 1])
def kp(xp: Int64, yp: Int64):
    t = fx.thread_idx.x
    rx, ry = bo.create_buffer_resource_from_addr(xp), bo.create_buffer_resource_from_addr(yp)
    v = fx.Vector(bo.buffer_load(rx, t * 4, vec_width=4, dtype=T.f32))
    bo.buffer_store(H["fp8_pack4"](v[0], v[1], v[2], v[3]), ry, t)

@flyc.jit
def launch(xp: Int64, yp: Int64, stream: fx.Stream = fx.Stream(None)):
    kp(xp, yp).launch(grid=(1,), block=(NT,), stream=stream)

x = (torch.randn(NT * 4) * 60).cuda()
y = torch.zeros(NT, dtype=torch.int32, device="cuda")
launch(x.data_ptr(), y.data_ptr()); torch.cuda.synchronize()
hw = y.view(torch.uint8).view(torch.float8_e4m3fnuz).float()
tq = x.to(torch.float8_e4m3fnuz).float()
bad = (hw != tq).nonzero().flatten()
print("mismatches", bad.numel(), "of", x.numel())
for i in bad[:8].tolist():
    print(x[i].item(), "hw", hw[i].item(), "torch", tq[i].item())
