"""Disposable: trace + compile the kernel for gfx942 on a non-gfx942 host
(FLYDSL_GPU_ARCH=gfx942). Linking fails there (lld); every earlier error is ours."""
import sys
import traceback

sys.path.insert(0, __file__.rsplit("/scratch/", 1)[0])
from mk import kernel as KK  # noqa: E402
from mk.dims import HD, Dims  # noqa: E402

import os
DEFER = os.environ.get("MK_AR_DEFER", "0") == "1"
for tp, ar, local, M in [(1, "fp32", False, 1), (2, "fp32", False, 1), (2, "bf16", False, 8),
                         (2, "fp32", True, 2), (2, "bf16", False, 1)][int(sys.argv[1]) if len(sys.argv) > 1 else 0:][:int(sys.argv[2]) if len(sys.argv) > 2 else 99]:
    d = Dims(tp)
    block, nb = 208, 3
    kv_s = (2 * block * d.nkv * HD, HD, d.nkv * HD, block * d.nkv * HD)
    cfg = KK.Cfg(M, 304, 512, block, nb, *kv_s, False, 1, 1, tp=tp, ar_dtype=ar, ar_local=local, ar_defer=DEFER, chunks=(("down", 2), ("o", 1), ("out", 1)) if tp > 1 else ())
    try:
        KK.build(cfg)(0, 0, 64)
        print("RAN?", cfg)
    except Exception as e:
        msg = str(e)
        ok = "lld" in msg or "serializ" in msg
        print(("TRACE OK " if ok else "FAIL ") + f"tp={tp} ar={ar} local={local} M={M}")
        if not ok:
            traceback.print_exc()
            sys.exit(1)
