"""Attention stage (mk/attention.py) alone against a PyTorch model of the same step.

One FlyDSL launch runs every attention task of one layer: grid = CU count, 8 waves a CTA,
tasks strided over the waves, exactly as the megakernel's attention stage runs them.
The paged KV cache follows vLLM TRITON_ATTN ([blocks, kv heads, B, 2 HD], K | V in the last
dimension), in two physical layouts. Block tables are shuffled; on sliding layers the entries
wholly below the window point at the null block 0, and the null block and every
out-of-window block hold NaN: a read outside the window shows up as NaN in the output.
Row M - 1 may be a padding row (slot -1): no cache write, output untouched.
Checks per case: output o vs the model (max |err| / max |o| < 2e-3: the kernel's and torch's
fp32 q norms can round a q dim to bf16 one ulp apart, which moves a score by about 1e-2 and,
with peaked softmaxes, o by up to 1e-3; elsewhere the error is about 1e-6), new K / V in the
cache at the
slot, no NaN; three consecutive steps (counters reset by the last arrivers).
Usage: python tests/test_attention.py
"""
import sys
import zlib

import torch

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl.expr.typing import Int64  # noqa: E402

from mk import attention as A  # noqa: E402
from mk import dev as D  # noqa: E402
from mk import reference as R  # noqa: E402
from mk.dims import NQ, WINDOW  # noqa: E402

WAVES, NT = 8, 512
CUS = torch.cuda.get_device_properties(0).multi_processor_count


def build(geo: A.Geo, M: int):
    WLF = geo.WLF
    name = f"g4_attn_m{M}_{zlib.crc32(repr((geo, M, CUS)).encode()):08x}"

    @fx.struct
    class Lds:
        aw: fx.Array[fx.Float32, WAVES * WLF, 16]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kattn(args: Int64):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = D.uniform(tid // 64)
        c = dict(tid=tid, bid=bid, lane=lane, wave=wave, gl=lane // 16, col=lane % 16, cus=CUS)
        wl = fx.SharedAllocator().allocate(Lds).peek().aw.ptr + wave * WLF
        ra = D.rsrc(args)

        def a64(i):
            v = D.ld(ra, 2 * i, 2)
            return (fx.Int64(D.uniform(v[1])) << 32) | fx.Int64(fx.Uint32(D.uniform(v[0])))

        P = [a64(i) for i in range(11)]
        for t in range(wave * CUS + bid, geo.ntask(M), CUS * WAVES):
            A.attn_task(c, geo, t, *P, wl)

    @flyc.jit
    def launch(args: Int64, stream: fx.Stream = fx.Stream(None)):
        kattn(args).launch(grid=(CUS,), block=(NT,), stream=stream)

    return launch


def rope_table(full, maxlen):
    cos, sin = R.rope_cos_sin(torch.arange(maxlen, device="cuda"), full)
    return torch.cat([cos, sin], 1).contiguous()  # [maxlen][HD]


class Case:
    """Paged caches for M rows at positions pos[m] (tokens already cached)."""

    def __init__(self, full, M, pos, B, layout, pad, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.full, self.M, self.B, self.pad = full, M, B, pad
        self.pos = torch.tensor(pos, dtype=torch.int32, device="cuda")
        self.steps_max = 3
        maxlen = max(pos) + self.steps_max + 1
        self.maxlen = maxlen
        nbr = -(-maxlen // B)  # blocks per row
        HD = A.HD_F if full else A.HD_S
        NKV = A.NKV_F if full else A.NKV_S
        self.HD, self.NKV = HD, NKV
        nblocks = M * nbr + 1
        # physical layout -> the [blocks, NKV, B, 2 HD] view vLLM hands the backend
        if layout == "BHNC":
            phys = torch.randn(nblocks, NKV, B, 2 * HD, generator=g, device="cuda")
            self.kv = phys.to(torch.bfloat16)
            view = self.kv
        else:  # "BNHC": positions outer, heads inner
            phys = torch.randn(nblocks, B, NKV, 2 * HD, generator=g, device="cuda")
            self.kv = phys.to(torch.bfloat16)
            view = self.kv.permute(0, 2, 1, 3)
        self.view = view
        s = view.stride()
        self.strides = (s[0], s[1], s[2], HD)
        perm = torch.randperm(M * nbr, generator=g, device="cuda").to(torch.int32) + 1
        self.bt = perm.reshape(M, nbr).contiguous()
        self.bt_full = self.bt.clone()  # the real blocks (for the model)
        self.view[0] = float("nan")  # vLLM's null block
        self.y = torch.randn(M, A.QKV_F if full else A.QKV_S, generator=g, device="cuda") * 2
        self.qn = (torch.rand(HD, generator=g, device="cuda") + 0.5).to(torch.bfloat16)
        self.kn = (torch.rand(HD, generator=g, device="cuda") + 0.5).to(torch.bfloat16)
        self.rope = rope_table(full, maxlen)
        self.apply_window()

    def slot(self):
        p = self.pos.long()
        rows = torch.arange(self.M, device="cuda")
        s = (self.bt_full[rows, p // self.B].long() * self.B + p % self.B).to(torch.int32)
        if self.pad:
            s[-1] = -1
        return s

    def apply_window(self):
        """vLLM nulls the block-table entries wholly below pos - (W - 1); their blocks go to
        other requests (here: NaN)."""
        if self.full:
            return
        for m in range(self.M):
            lo = max(0, int(self.pos[m]) - (WINDOW - 1))
            for b in range(lo // self.B):
                blk = int(self.bt_full[m, b])
                if blk != 0:
                    self.view[blk] = float("nan")
                self.bt[m, b] = 0

    def new_kv(self):
        """The new position's k, v per row (fp32, before the cache's bf16 rounding)."""
        HD, NKV, M = self.HD, self.NKV, self.M
        y = self.y
        k = y[:, NQ * HD : NQ * HD + NKV * HD].reshape(M, NKV, HD)
        v = k if self.full else y[:, NQ * HD + NKV * HD :].reshape(M, NKV, HD)
        cos, sin = R.rope_cos_sin(self.pos.long(), self.full)
        return R.apply_rope(R.rms(k, self.kn), cos, sin), R.rms(v)

    def model(self, written):
        """o [M, NQ HD] as the reference computes it, with the new position's K / V taken
        from the cache as the kernel wrote them (written[m] = (k, v) bf16): the kernel's and
        torch's fp32 norms may round to bf16 one ulp apart, which a large softmax weight on
        the new position would show as a 1e-3 error."""
        HD, NKV, M = self.HD, self.NKV, self.M
        q = self.y[:, : NQ * HD].reshape(M, NQ, HD)
        cos, sin = R.rope_cos_sin(self.pos.long(), self.full)
        q = R.apply_rope(R.rms(q, self.qn), cos, sin).to(torch.bfloat16).float()
        o = torch.zeros(M, NQ, HD, device="cuda")
        g = NQ // NKV
        for m in written:
            p = int(self.pos[m])
            lo = 0 if self.full else max(0, p - WINDOW + 1)
            ps = torch.arange(lo, p + 1, device="cuda")
            blk = self.bt_full[m, ps // self.B].long()
            rows = self.view[blk, :, ps % self.B].float()  # [L, NKV, 2 HD]
            K, V = rows[..., :HD].clone(), rows[..., HD:].clone()
            K[-1], V[-1] = written[m][0].float(), written[m][1].float()
            s = torch.einsum("jgd,ljd->jgl", q[m].reshape(NKV, g, HD), K)
            o[m] = torch.einsum("jgl,ljd->jgd", torch.softmax(s, -1), V).reshape(NQ, HD)
        return o.reshape(M, NQ * HD)


def run_case(name, full, M, pos, B, layout, pad, seed=0, steps=3):
    cs = Case(full, M, pos, B, layout, pad, seed)
    s0, s1, s2, sv = cs.strides
    NC = (WINDOW // A.CH + 1) if not full else -(-cs.maxlen // A.CH)
    geo = A.Geo(full, NC, cs.bt.shape[1], B, s0, s1, s2, sv).check()
    launch = build(geo, M)
    rec = torch.zeros(geo.nrec(M), device="cuda")
    cnt = torch.zeros(geo.ncnt(M), dtype=torch.int32, device="cuda")
    o = torch.full((M, NQ * cs.HD), 7.0, device="cuda")
    ok = True
    for st in range(steps):
        slot = cs.slot()
        k32, v32 = cs.new_kv()
        args = torch.tensor(
            [cs.y.data_ptr(), cs.qn.data_ptr(), cs.kn.data_ptr(), cs.rope.data_ptr(),
             cs.kv.data_ptr(), cs.bt.data_ptr(), slot.data_ptr(), cs.pos.data_ptr(),
             rec.data_ptr(), cnt.data_ptr(), o.data_ptr()], dtype=torch.int64, device="cuda")
        o.fill_(7.0)
        launch(args.data_ptr(), stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        live = slot >= 0
        # the new K / V in the cache: within one bf16 ulp (<= 2^-7 relative) of the fp32 values
        written, wr = {}, True
        for m in range(M):
            if not live[m]:
                continue
            p = int(cs.pos[m])
            row = cs.view[int(cs.bt_full[m, p // B]), :, p % B]
            written[m] = (row[:, : cs.HD].clone(), row[:, cs.HD :].clone())
            for got, want in zip(written[m], (k32[m], v32[m])):
                wr &= bool(((got.float() - want).abs() <= want.abs() * 2.0**-7 + 1e-30).all())
        ref_o = cs.model(written)
        err = (o[live] - ref_o[live]).abs().max().item() / ref_o[live].abs().max().item()
        nan = torch.isnan(o[live]).any().item()
        if err > 1e-4:  # where: (row, head, worst dim)
            d = (o - ref_o).abs().reshape(M, NQ, cs.HD) * live[:, None, None]
            mi, hi, di = [int(x) for x in torch.unravel_index(d.argmax(), d.shape)]
            print(f"  worst at row {mi} (pos {int(cs.pos[mi])}) head {hi} dim {di}: "
                  f"kernel {o[mi, hi * cs.HD + di]:.5f} model {ref_o[mi, hi * cs.HD + di]:.5f}")
        untouched = True
        if pad:
            untouched = bool((o[-1] == 7.0).all())
        good = err < 2e-3 and not nan and wr and untouched and int(cnt.abs().sum()) == 0
        ok &= good
        print(f"{name} step {st}: pos {cs.pos.tolist()} rel err {err:.2e} nan {nan} kv write {wr} "
              f"pad untouched {untouched} counters zero {int(cnt.abs().sum()) == 0} "
              f"{'ok' if good else 'FAIL'}")
        # next step: the model's K / V of this step are in the cache (written by the kernel)
        cs.pos += 1
        cs.y = torch.randn_like(cs.y) * 2
        cs.apply_window()
    return ok


def main():
    W = WINDOW
    cases = [
        ("sliding M1 B16", False, 1, [3000], 16, "BHNC", False),
        ("sliding M8 B16 boundaries", False, 8, [0, 5, W - 2, W - 1, W, W + 1, 2 * W - 1, 2 * W + 37], 16, "BHNC", False),
        ("sliding M4 B32 BNHC pad", False, 4, [63, W + 63, 1500, 700], 32, "BNHC", True),
        ("full M1 B32", True, 1, [3000], 32, "BHNC", False),
        ("full M8 B32 BNHC pad", True, 8, [0, 1, 63, 64, W - 1, W, 2500, 4000], 32, "BNHC", True),
        ("full M2 B16", True, 2, [130, 2049], 16, "BHNC", False),
    ]
    ok = True
    for c in cases:
        ok &= run_case(*c)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
