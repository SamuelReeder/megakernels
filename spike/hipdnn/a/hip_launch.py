"""Spike A (disposable): take the FlyDSL-built Qwen3.8 decode kernel as a standalone code
object and launch it with hipModuleLoad + hipModuleLaunchKernel (ctypes, torch's HIP
runtime), on the runner's own tab/glob tables. Reproduces tests/logits_hash.py per M for
three launch paths: FlyDSL launcher, hipModuleLaunchKernel, and hipModuleLaunchKernel
captured in a torch CUDA (HIP) graph. Then times the host side of one launch.
Run in the model dir (common/scripts/mkrun): python hip_launch.py OUT MODEL [Ms] [STEPS] [LAYERS]
"""
import ctypes
import hashlib
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.getcwd())
import torch  # noqa: E402

from mk import reference as R  # noqa: E402
from mk import tp as TP  # noqa: E402
from mk.runner import Engine  # noqa: E402

from flydsl._mlir import ir  # noqa: E402
from flydsl._mlir.dialects import gpu  # noqa: E402
from flydsl.compiler.jit_function import _create_mlir_context  # noqa: E402

OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
PATH = sys.argv[2]
MS = [int(m) for m in (sys.argv[3] if len(sys.argv) > 3 else "1,8").split(",")]
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 8
L = int(sys.argv[5]) if len(sys.argv) > 5 else 64
LLVM = "/opt/rocm/llvm/bin"

_tl = pathlib.Path(torch.__file__).parent / "lib" / "libamdhip64.so"
hip = ctypes.CDLL(str(_tl) if _tl.exists() else "libamdhip64.so")
hip.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
hip.hipGetErrorString.restype = ctypes.c_char_p


def ck(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what}: hip error {rc} {hip.hipGetErrorString(rc).decode()}")


def log(*a):
    print(*a, flush=True)


def extract(eng, M):
    """Compile eng's FlyDSL launcher without running it (COMPILE_ONLY) and pull the GPU
    object out of the lowered module (gpu.binary objects[0], as FlyDSL's own
    max_blocks_per_cu does). Returns (code object path, kernel symbol, launch info)."""
    os.environ["COMPILE_ONLY"] = "1"
    t0 = time.time()
    r = eng.launch(eng.tab.data_ptr(), eng.glob.data_ptr(), eng.layers,
                   stream=torch.cuda.current_stream())
    del os.environ["COMPILE_ONLY"]
    assert r is None
    _, art = eng.launch._last_compiled
    info = dict(compile_s=round(time.time() - t0, 1),
                post_load_processors=len(art._post_load_processors),
                link_libs=str(getattr(art, "_link_libs", None)))
    (OUT / f"m{M}_lowered.mlir.txt").write_text(art._ir_text[:20000])
    objs, launches, calls, kref = [], [], set(), []

    def visit(op):
        if op.name == "gpu.binary":
            for o in ir.ArrayAttr(op.attributes["objects"]):
                oa = gpu.ObjectAttr(o)
                objs.append((ir.StringAttr(op.attributes["sym_name"]).value, str(oa.target),
                             oa.format, oa.object))
        elif op.name == "gpu.launch_func":
            launches.append(str(op))
            kref.append(ir.SymbolRefAttr(op.opview.kernel).value)
        elif op.name == "llvm.call":
            calls.add(str(op.attributes["callee"]) if "callee" in op.attributes else "?")
        return ir.WalkResult.ADVANCE

    with _create_mlir_context():
        ir.Module.parse(art._ir_text).operation.walk(visit)
    info["objects"] = [dict(binary=b, target=t, format=f, nbytes=len(o), magic=o[:24].hex())
                       for b, t, f, o in objs]
    info["launch_func"] = launches
    info["host_calls"] = sorted(calls)
    co = OUT / f"qwen38_decode_m{M}.co"
    co.write_bytes(objs[0][3])
    info["kernel_refs"] = [list(map(str, k)) for k in kref]
    info["symbol_guess"] = str(kref[0][-1]) if kref else None
    for tool, args in (("readelf_notes", ["llvm-readelf", "--notes"]),
                       ("readelf_syms", ["llvm-readelf", "-sW"]),
                       ("readelf_hdr", ["llvm-readelf", "-hSW"])):
        p = subprocess.run([f"{LLVM}/{args[0]}", *args[1:], str(co)], capture_output=True, text=True)
        (OUT / f"m{M}_{tool}.txt").write_text(p.stdout + p.stderr)
    return co, info


class HipKernel:
    def __init__(self, co, sym):
        self.mod, self.fn = ctypes.c_void_p(), ctypes.c_void_p()
        ck(hip.hipModuleLoad(ctypes.byref(self.mod), str(co).encode()), "hipModuleLoad")
        ck(hip.hipModuleGetFunction(ctypes.byref(self.fn), self.mod, sym.encode()), "hipModuleGetFunction")

    def bind(self, eng):
        """kernelParams for (tab: i64, glob: i64, nlayers: i32); fixed per engine."""
        a = (ctypes.c_int64(eng.tab.data_ptr()), ctypes.c_int64(eng.glob.data_ptr()),
             ctypes.c_int32(eng.layers))
        p = (ctypes.c_void_p * 3)(*[ctypes.addressof(x) for x in a])
        return (a, p, eng.cfg.cus)

    def launch(self, bound, stream):
        _, p, cus = bound
        ck(hip.hipModuleLaunchKernel(self.fn, cus, 1, 1, 512, 1, 1, 0, stream, p, None),
           "hipModuleLaunchKernel")


def hip_step(eng, hk, bound, tok):
    eng.tok.copy_(tok.to(torch.int32))
    hk.launch(bound, torch.cuda.current_stream().cuda_stream)
    eng.pos += 1
    return eng.logits


def run_hash(step, toks):
    h = hashlib.sha256()
    fin = True
    for t in range(STEPS):
        o = step(toks[t]).clone()
        torch.cuda.synchronize()
        h.update(o.float().cpu().numpy().tobytes())
        fin &= bool(torch.isfinite(o).all())
    return h.hexdigest()[:16], fin


def host_us(fn, n=200):
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        fn()
        ts.append((time.perf_counter_ns() - t0) / 1e3)
    torch.cuda.synchronize()
    ts.sort()
    return dict(median=round(statistics.median(ts), 2), p10=round(ts[n // 10], 2),
                p90=round(ts[9 * n // 10], 2))


def dev_ms(fn, n=20):
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn()
    torch.cuda.synchronize()
    e0.record()
    for _ in range(n):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return round(e0.elapsed_time(e1) / n, 4)


def main():
    maps = [ln.split()[-1] for ln in open("/proc/self/maps") if "amdhip64" in ln]
    res = dict(hip_lib=sorted(set(maps)), torch=torch.__version__, hip=torch.version.hip,
               dev=torch.cuda.get_device_name(0), host=os.uname().nodename, Ms={})
    ctx = TP.init()
    W = R.Weights(PATH, "mxfp4", keep_raw=True, dequant=False, layers=L)
    PK = TP.packed(ctx, W, layers=L)
    for M in MS:
        r = res["Ms"][M] = {}
        toks = torch.randint(0, 200000, (STEPS, M), device="cuda",
                             generator=torch.Generator("cuda").manual_seed(7))
        e_fly = Engine(PK, M, maxlen=128, layers=L)
        co, info = extract(e_fly, M)
        r.update(info, co=str(co), co_bytes=co.stat().st_size)
        log(f"M={M} extracted {co} {r['co_bytes']} B symbol {info['symbol_guess']} "
            f"objects {[(o['target'], o['format'], o['magic'][:16]) for o in info['objects']]}")
        hk = HipKernel(co, info["symbol_guess"])
        # 1. FlyDSL launcher (baseline, reuses the compiled artifact)
        r["hash_flydsl"] = run_hash(e_fly.step, toks)
        log(f"HASH M={M} flydsl {r['hash_flydsl']}")
        # 2. hipModuleLaunchKernel on a fresh engine
        e_hip = Engine(PK, M, maxlen=128, layers=L)
        b_hip = hk.bind(e_hip)
        r["hash_hip"] = run_hash(lambda tk: hip_step(e_hip, hk, b_hip, tk), toks)
        log(f"HASH M={M} hipModule {r['hash_hip']}")
        # 3. hipModuleLaunchKernel captured in a HIP graph (tok copy, launch, pos += 1)
        e_g = Engine(PK, M, maxlen=128, layers=L)
        b_g = hk.bind(e_g)
        static_tok = torch.zeros(M, dtype=toks.dtype, device="cuda")
        g = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(g):
                hip_step(e_g, hk, b_g, static_tok)

            def gstep(tk):
                static_tok.copy_(tk)
                g.replay()
                return e_g.logits

            r["hash_graph"] = run_hash(gstep, toks)
        except Exception as ex:  # record and go on
            r["hash_graph"] = f"FAILED {type(ex).__name__}: {ex}"
        log(f"HASH M={M} hipModule-graph {r['hash_graph']}")
        # 4. host overhead of one launch (state is garbage from here on; no pos change)
        stream_obj = torch.cuda.current_stream()
        sh = stream_obj.cuda_stream
        fly = lambda: e_fly.launch(e_fly.tab.data_ptr(), e_fly.glob.data_ptr(), L, stream=stream_obj)  # noqa: E731
        hl = lambda: hk.launch(b_hip, sh)  # noqa: E731
        r["host_us_flydsl"] = host_us(fly)
        r["host_us_hip"] = host_us(hl)
        # FlyDSL's precompiled fast path (flyc.compile runs one launch first)
        try:
            import flydsl.compiler as flyc
            tp_, gp_ = e_fly.tab.data_ptr(), e_fly.glob.data_ptr()
            cf = flyc.compile(e_fly.launch, tp_, gp_, L, stream_obj)
            r["host_us_flydsl_compiled"] = host_us(lambda: cf(tp_, gp_, L, stream_obj))
        except Exception as ex:
            r["host_us_flydsl_compiled"] = f"FAILED {type(ex).__name__}: {ex}"
        t0 = time.perf_counter()
        HipKernel(r["co"], info["symbol_guess"])
        r["module_load_ms"] = round((time.perf_counter() - t0) * 1e3, 2)
        if not isinstance(r["hash_graph"], str):
            r["host_us_graph_replay"] = host_us(g.replay)
        r["dev_ms_flydsl"] = dev_ms(fly)
        r["dev_ms_hip"] = dev_ms(hl)
        log(f"TIME M={M} host_us flydsl {r['host_us_flydsl']} flydsl-compiled "
            f"{r['host_us_flydsl_compiled']} hip {r['host_us_hip']} graph "
            f"{r.get('host_us_graph_replay')} dev_ms flydsl {r['dev_ms_flydsl']} hip "
            f"{r['dev_ms_hip']} module_load_ms {r['module_load_ms']}")
        del e_fly, e_hip, e_g, g
        torch.cuda.empty_cache()
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
    log(json.dumps({M: {k: v for k, v in r.items() if k.startswith(("hash", "host", "dev", "co_"))}
                    for M, r in res["Ms"].items()}, default=str))


if __name__ == "__main__":
    main()
