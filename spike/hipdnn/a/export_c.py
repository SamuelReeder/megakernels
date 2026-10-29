"""Spike A (disposable): try FlyDSL 0.4.0 CompiledFunction.export_to_c on the Qwen3.8
launcher and check whether the extracted code object (hip_launch.py) sits byte-equal in
the exported host object. Usage (model dir): python export_c.py MODEL OUT CO M"""
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, os.getcwd())
import flydsl.compiler as flyc  # noqa: E402
import torch  # noqa: E402

from mk import reference as R  # noqa: E402
from mk import tp as TP  # noqa: E402
from mk.runner import Engine  # noqa: E402

path, out, co, M = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]), int(sys.argv[4])
L = 64
out.mkdir(parents=True, exist_ok=True)
ctx = TP.init()
W = R.Weights(path, "mxfp4", keep_raw=True, dequant=False, layers=L)
PK = TP.packed(ctx, W, layers=L)
eng = Engine(PK, M, maxlen=128, layers=L)
cf = flyc.compile(eng.launch, eng.tab.data_ptr(), eng.glob.data_ptr(), L, torch.cuda.current_stream())
try:
    cf.export_to_c(str(out), f"qwen38_m{M}")
    print("EXPORT ok", sorted(p.name for p in out.iterdir()), flush=True)
except Exception as ex:
    print(f"EXPORT FAILED {type(ex).__name__}: {ex}", flush=True)
    sys.exit(0)
for o in out.glob("*.o"):
    blob, cob = o.read_bytes(), co.read_bytes()
    print(f"OBJ {o.name} {len(blob)} B contains extracted co ({len(cob)} B) byte-equal: "
          f"{cob in blob}", flush=True)
    s = subprocess.run(["/opt/rocm/llvm/bin/llvm-readelf", "-sSW", str(o)], capture_output=True, text=True)
    (out / f"{o.stem}_readelf.txt").write_text(s.stdout + s.stderr)
for h in out.glob("*.h"):
    print(h.read_text()[:3000], flush=True)
