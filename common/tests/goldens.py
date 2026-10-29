"""CPU goldens (common/DESIGN.md "Test strategy", oracle 1): byte hashes of the host pack
functions on seeded inputs and the host-side GEMV geometry of every model's Cfg.specs()
at M = 1..8, cus = 304 (Qwen3.8 also at TP2). No GPU.

Usage (repo root, the venv of setup_env.sh):
  python common/tests/goldens.py            compare with common/tests/goldens.txt
  python common/tests/goldens.py --record   write common/tests/goldens.txt
"""
import dataclasses
import hashlib
import importlib
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLD = os.path.join(ROOT, "common", "tests", "goldens.txt")
MODELS = ("qwen3.8-27b-gfx942", "qwen3.6-35b-a3b-gfx942", "gemma4-26b-a4b-gfx942")
CUS = 304
# geometry: attributes and methods of a Gemv / MoeGemv that the host side uses
ATTRS = ("U", "NW", "G", "KS", "KSc", "Kc", "MT", "XR", "XSR", "slots", "SEGMAX", "esize",
         "small", "NGC", "NPWC")
PER_WAVE = ("wave_lo",)  # f(gw), gw = 0 .. NW
PER_CTA = ("cta_range", "cta_groups")  # f(bid), bid = 0 .. cus - 1
PER_PIECE = ("nseg_host",)  # f(p), p = 0 .. U - 1


def load(model):
    """The model's `mk` package (one at a time: every model calls it `mk`)."""
    for k in [k for k in sys.modules if k == "mk" or k.startswith("mk.")]:
        del sys.modules[k]
    sys.path.insert(0, os.path.join(ROOT, model))
    try:
        return (importlib.import_module("mk.kernel"), importlib.import_module("mk.pack"))
    finally:
        sys.path.pop(0)


def h(*ts):
    d = hashlib.sha256()
    for t in ts:
        d.update(repr((t.dtype, tuple(t.shape))).encode() + t.contiguous().view(torch.uint8).numpy().tobytes())
    return d.hexdigest()[:16]


def call(f, *a):
    try:
        v = f(*a)
    except Exception as e:  # noqa: BLE001 (the error kind is part of the geometry)
        return type(e).__name__
    return v


def geometry(spec):
    out = []
    for a in ATTRS:
        if hasattr(type(spec), a) or a in getattr(spec, "__dict__", {}):
            out.append((a, call(getattr, spec, a)))
    for name, rng in ((n, range(getattr(spec, "NW", 0) + 1)) for n in PER_WAVE):
        if hasattr(spec, name):
            out.append((name, [call(getattr(spec, name), i) for i in rng]))
    for name in PER_CTA:
        if hasattr(spec, name) and getattr(spec, "cta", False):
            out.append((name, [call(getattr(spec, name), b) for b in range(spec.cus)]))
    for name in PER_PIECE:
        if hasattr(spec, name):
            out.append((name, [call(getattr(spec, name), p) for p in range(getattr(spec, "U", 0))]))
    if hasattr(spec, "part_floats"):
        out.append(("part_floats", call(spec.part_floats)))
    return out


def dummy_cfg(Cfg, **kw):
    """Cfg with every required field set to a neutral value: geometry reads M, cus, the
    chunk fields and the model's switches only."""
    vals = {}
    for f in dataclasses.fields(Cfg):
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            t = str(f.type)
            vals[f.name] = () if "tuple" in t else (False if "bool" in t else 1)
    vals.update(kw)
    return Cfg(**vals)


def geo_lines(model, KK):
    tps = (1, 2) if "tp" in {f.name for f in dataclasses.fields(KK.Cfg)} else (1,)
    for tp in tps:
        for M in range(1, 9):
            kw = dict(M=M, cus=CUS) | (dict(tp=tp) if tp > 1 else {})
            cfg = dummy_cfg(KK.Cfg, **kw)
            for name, spec in cfg.specs().items():
                g = geometry(spec)
                yield f"{model} geo tp{tp} M{M} {name} {hashlib.sha256(repr(g).encode()).hexdigest()[:16]}"
            yield f"{model} geo tp{tp} M{M} part_floats {cfg.part_floats()}"


def pack_lines(model, P):
    g = torch.Generator().manual_seed(1)
    w = torch.randn(64, 512, generator=g).to(torch.bfloat16)
    w2 = torch.randn(3, 64, 512, generator=g).to(torch.bfloat16)
    yield f"{model} pack pack_bf16 {h(P.pack_bf16(w))}"
    yield f"{model} pack gu_order {h(P.gu_order(w))}"
    tiles = P.pack_bf16(w)
    if "qwen3.8" in model:
        yield f"{model} pack unit_order {h(P.unit_order(tiles, 2))}"
        from mk import fp4
        wq = torch.randint(0, 256, (64, 256), generator=g, dtype=torch.uint8)
        sq = torch.randint(100, 140, (64, 16), generator=g, dtype=torch.uint8)
        yield f"{model} pack fp4.repack {h(*fp4.repack(wq, sq))}"
        return
    yield f"{model} pack gu_order_batched {h(P.gu_order(w2))}"
    yield f"{model} pack unit_order {h(P.unit_order(tiles, 3, 2))}"
    yield f"{model} pack unit_order_batched {h(P.unit_order(torch.stack([tiles, tiles + 1]), 3, 2))}"
    w8 = (torch.randn(64, 256, generator=g) * 4).to(torch.float8_e4m3fn)
    if "qwen3.6" in model:
        s = torch.rand(1, 2, generator=g) + 0.5
        yield f"{model} pack row_scales {h(P.row_scales(s, 64))}"
        yield f"{model} pack fp8_tiles {h(*P.fp8_tiles(w8, P.row_scales(s, 64)))}"
    else:
        s = torch.rand(64, generator=g) + 0.5
        yield f"{model} pack gu_order_rows {h(P.gu_order_rows(s))}"
        yield f"{model} pack fp8_tiles {h(*P.fp8_tiles(w8, s))}"


def lines():
    for model in MODELS:
        KK, P = load(model)
        sys.path.insert(0, os.path.join(ROOT, model))  # mk.fp4 import inside pack_lines
        try:
            yield from pack_lines(model, P)
            yield from geo_lines(model, KK)
        finally:
            sys.path.pop(0)


def main():
    got = list(lines())
    if "--record" in sys.argv:
        with open(GOLD, "w") as f:
            f.write("\n".join(got) + "\n")
        print(f"recorded {len(got)} goldens to {GOLD}")
        return
    want = open(GOLD).read().splitlines()
    bad = [(a, b) for a, b in zip(want, got) if a != b] + ([("count", len(got))] if len(want) != len(got) else [])
    for a, b in bad:
        print("DIFF", a, "->", b)
    print(f"goldens: {len(got)} lines, {len(bad)} differ")
    print("PASS" if not bad else "FAIL")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
