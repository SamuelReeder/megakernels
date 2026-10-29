"""Slice C: geometry key (crc32 of repr(Cfg with M=0)) of the slice A test geometry, built as
mk.runner.Engine builds its Cfg (maxlen 128, block 208, 304 CUs, TP1, no chunk override).
Also prints crc32(repr(Cfg)); it does not equal the slice A symbols, because megakernels'
kernel_name hashes the compile key (Cfg + kernel sources), not repr(Cfg). Run in the model dir."""
import dataclasses
import sys
import zlib

sys.path.insert(0, ".")
from mk import kernel as KK  # noqa: E402
from mk.dims import HD, Dims  # noqa: E402
from mk.runner import page_bytes  # noqa: E402

d = Dims(1)
maxlen, block, xpack, cus = 128, 208, False, 304
kv_s = (2 * block * d.nkv * HD, HD * (block if xpack else 1), d.nkv * HD, block * d.nkv * HD)
for M, sym in ((1, "8612ae82"), (8, "c6efe94d")):
    cfg = KK.Cfg(M, cus, maxlen, block, -(-maxlen // block), *kv_s, xpack, page_bytes(d),
                 page_bytes(d), tp=1, ar_dtype="fp32", ar_local=False, chunks=())
    crc = f"{zlib.crc32(repr(cfg).encode()):08x}"
    geom = f"{zlib.crc32(repr(dataclasses.replace(cfg, M=0)).encode()):08x}"
    print(f"M={M} symbol crc {crc} expected {sym} match {crc == sym} geom {geom} cfg {cfg!r}")
