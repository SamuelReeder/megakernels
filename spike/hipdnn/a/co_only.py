"""Spike A (disposable): fresh process, no FlyDSL compile. Builds the runner's buffers and
tables (mk.runner.Engine; its KK.build only defines the launcher) and launches a saved
code object with hipModuleLaunchKernel. Prints the tests/logits_hash.py logits hash.
Usage (model dir): python co_only.py MODEL CO SYMBOL M [STEPS] [LAYERS]"""
import ctypes
import hashlib
import os
import pathlib
import sys

sys.path.insert(0, os.getcwd())
import torch  # noqa: E402

from mk import reference as R  # noqa: E402
from mk import tp as TP  # noqa: E402
from mk.runner import Engine  # noqa: E402

path, co, sym, M = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
steps = int(sys.argv[5]) if len(sys.argv) > 5 else 8
L = int(sys.argv[6]) if len(sys.argv) > 6 else 64
hip = ctypes.CDLL(str(pathlib.Path(torch.__file__).parent / "lib" / "libamdhip64.so"))
hip.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
assert hip.hipModuleLoad(ctypes.byref(mod), co.encode()) == 0
assert hip.hipModuleGetFunction(ctypes.byref(fn), mod, sym.encode()) == 0

ctx = TP.init()
W = R.Weights(path, "mxfp4", keep_raw=True, dequant=False, layers=L)
PK = TP.packed(ctx, W, layers=L)
eng = Engine(PK, M, maxlen=128, layers=L)
args = (ctypes.c_int64(eng.tab.data_ptr()), ctypes.c_int64(eng.glob.data_ptr()), ctypes.c_int32(L))
params = (ctypes.c_void_p * 3)(*[ctypes.addressof(a) for a in args])
toks = torch.randint(0, 200000, (steps, M), device="cuda",
                     generator=torch.Generator("cuda").manual_seed(7))
h = hashlib.sha256()
for t in range(steps):
    eng.tok.copy_(toks[t].to(torch.int32))
    rc = hip.hipModuleLaunchKernel(fn, eng.cfg.cus, 1, 1, 512, 1, 1, 0,
                                   torch.cuda.current_stream().cuda_stream, params, None)
    assert rc == 0, rc
    eng.pos += 1
    torch.cuda.synchronize()
    h.update(eng.logits.float().cpu().numpy().tobytes())
print(f"HASH co-only M={M} cus={eng.cfg.cus} logits {h.hexdigest()[:16]} "
      f"flydsl_compiled={getattr(eng.launch, '_last_compiled', None) is not None}", flush=True)
