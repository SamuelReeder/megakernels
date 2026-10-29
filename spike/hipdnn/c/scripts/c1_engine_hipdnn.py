"""Slice C step 1-3 (disposable): mk.runner.Engine builds tab/glob/state with torch; the decode
step launches through hipDNN (custom op megakernel.qwen3_8_27b_decode -> hip-kernel-provider
megakernel pack -> FlyDSL code object from kpack) instead of the FlyDSL launcher.
1. logits hash per M (tests/logits_hash.py schedule: seed 7, STEPS steps, 64 layers):
   FlyDSL launcher, hipDNN execute, hipDNN execute captured in torch.cuda.graph.
2. LONG greedy steps, FlyDSL vs hipDNN engines side by side, logits compared bitwise per step.
3. Host time of one launch and TPOT per path.
Run in the model dir: python c1_engine_hipdnn.py OUT MODEL CO_DIR [Ms] [STEPS] [LONG]
"""
import ctypes
import hashlib
import json
import os
import pathlib
import statistics
import sys
import time

sys.path.insert(0, os.getcwd())
import torch  # noqa: E402

from mk import reference as R  # noqa: E402
from mk import tp as TP  # noqa: E402
from mk.runner import Engine  # noqa: E402

import hipdnn_frontend as hipdnn  # noqa: E402

OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
PATH = sys.argv[2]
CO_DIR = pathlib.Path(sys.argv[3])
MS = [int(m) for m in (sys.argv[4] if len(sys.argv) > 4 else "1,8").split(",")]
STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 8
LONG = int(sys.argv[6]) if len(sys.argv) > 6 else 96
L = 64
MAXLEN = 128
SYM = {1: "qwen38_decode_m1_8612ae82", 8: "qwen38_decode_m8_c6efe94d"}
MODEL = os.environ.get("MK_HIPDNN_MODEL", "qwen3.8-27b-flydsl")
# geometry key of the slice A test geometry (logs/geom_test_cfg.log)
GEOM = os.environ.get("MK_HIPDNN_GEOM", "e5cc820d")
DT = hipdnn.DataType


def log(*a):
    print(*a, flush=True)


def hip_lib(key="amdhip64"):
    maps = sorted({ln.split()[-1] for ln in open("/proc/self/maps") if key in ln})
    return maps


hip = ctypes.CDLL(hip_lib()[0]) if hip_lib() else None


def ck(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what}: hip error {rc}")


class HipDnnStep:
    """One hipDNN graph per engine: custom op over (tab, glob); the payload selects the code object."""

    def __init__(self, eng, handle):
        self.eng, self.handle = eng, handle
        g = hipdnn.Graph()
        g.set_name(f"qwen38_decode_m{eng.M}")
        g.set_io_data_type(DT.FLOAT).set_intermediate_data_type(DT.FLOAT).set_compute_data_type(DT.FLOAT)
        tab = hipdnn.Tensor.create([eng.tab.numel()], DT.INT64)
        tab.set_name("tab")
        glob = hipdnn.Tensor.create([eng.glob.numel()], DT.INT64)
        glob.set_name("glob")
        payload = {"model": MODEL, "M": eng.M, "nlayers": eng.layers, "geom": GEOM,
                   "geometry": {"grid": [eng.cfg.cus], "block": [512], "lds": 0}}
        attrs = hipdnn.CustomOpAttributes()
        attrs.set_custom_op_id("megakernel.qwen3_8_27b_decode")
        attrs.set_compute_data_type(DT.FLOAT)
        attrs.set_data(list(json.dumps(payload).encode()))
        g.custom_op([tab, glob], 0, attrs)
        t0 = time.perf_counter()
        for what in ("validate", "build_operation_graph", "create_execution_plans", "check_support",
                     "build_plans"):
            fn = getattr(g, what)
            err = fn(handle) if what == "build_operation_graph" else fn()
            if not err.is_good():
                raise RuntimeError(f"hipDNN {what}: {err.get_message()}")
        self.plan_ms = round((time.perf_counter() - t0) * 1e3, 1)
        eid = g.get_execution_plan_engine_id()
        eid = eid[-1] if isinstance(eid, tuple) else eid
        self.engine_name = handle.engine_id_to_name(eid)
        self.g = g
        self.vp = {tab.get_uid(): eng.tab.data_ptr(), glob.get_uid(): eng.glob.data_ptr()}

    def launch(self):
        err = self.g.execute(self.handle, self.vp, 0)
        if not err.is_good():
            raise RuntimeError(f"hipDNN execute: {err.get_message()}")

    def step(self, tok):
        self.eng.tok.copy_(tok.to(torch.int32))
        self.launch()
        self.eng.pos += 1
        return self.eng.logits


class HipModule:
    """Slice A path: hipModuleLaunchKernel on torch's HIP runtime."""

    def __init__(self, eng):
        hip.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.mod, self.fn = ctypes.c_void_p(), ctypes.c_void_p()
        ck(hip.hipModuleLoad(ctypes.byref(self.mod), str(CO_DIR / f"qwen38_decode_m{eng.M}.co").encode()),
           "hipModuleLoad")
        ck(hip.hipModuleGetFunction(ctypes.byref(self.fn), self.mod, SYM[eng.M].encode()), "GetFunction")
        self.a = (ctypes.c_int64(eng.tab.data_ptr()), ctypes.c_int64(eng.glob.data_ptr()),
                  ctypes.c_int32(eng.layers))
        self.p = (ctypes.c_void_p * 3)(*[ctypes.addressof(x) for x in self.a])
        self.cus = eng.cfg.cus

    def launch(self):
        ck(hip.hipModuleLaunchKernel(self.fn, self.cus, 1, 1, 512, 1, 1, 0,
                                     torch.cuda.current_stream().cuda_stream, self.p, None), "launch")


def run_hash(step, toks):
    h = hashlib.sha256()
    fin = True
    for t in range(STEPS):
        o = step(toks[t]).clone()
        torch.cuda.synchronize()
        h.update(o.float().cpu().numpy().tobytes())
        fin &= bool(torch.isfinite(o).all())
    return h.hexdigest()[:16], fin


def stats(ts):
    ts = sorted(ts)
    n = len(ts)
    return dict(median=round(statistics.median(ts), 2), p10=round(ts[n // 10], 2),
                p90=round(ts[9 * n // 10], 2))


def host_us(fn, n=200):
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        fn()
        ts.append((time.perf_counter_ns() - t0) / 1e3)
    torch.cuda.synchronize()
    return stats(ts)


def tpot(eng, step, first, n=64, sync=True):
    """Greedy decode loop of n steps from a reset engine; ms per step. sync: the host waits
    for each token (as a serving loop does); else the steps queue and one sync ends the loop."""
    eng.reset()
    tok = first.clone()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        tok = step(tok).argmax(-1)
        if sync:
            tok.cpu()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / n


def main():
    import flydsl.compiler as flyc
    res = dict(hip_lib=hip_lib(), torch=torch.__version__, hip=torch.version.hip,
               dev=torch.cuda.get_device_name(0), host=os.uname().nodename, model=MODEL, Ms={})
    ctx = TP.init()
    W = R.Weights(PATH, "mxfp4", keep_raw=True, dequant=False, layers=L)
    PK = TP.packed(ctx, W, layers=L)
    handle = hipdnn.create_handle()
    stream = torch.cuda.current_stream()
    handle.set_stream(stream.cuda_stream)
    res["hip_lib_after_hipdnn"] = hip_lib()
    res["hipdnn_libs"] = hip_lib("hipdnn") + hip_lib("hip_kernel_provider")
    log("HIP libs", res["hip_lib_after_hipdnn"], "hipDNN libs", res["hipdnn_libs"], "stream",
        hex(stream.cuda_stream))
    for M in MS:
        r = res["Ms"][M] = {}
        toks = torch.randint(0, 200000, (STEPS, M), device="cuda",
                             generator=torch.Generator("cuda").manual_seed(7))
        # 1a. FlyDSL launcher
        e_fly = Engine(PK, M, maxlen=MAXLEN, layers=L)
        r["hash_flydsl"] = run_hash(e_fly.step, toks)
        log(f"HASH M={M} flydsl {r['hash_flydsl']}")
        # 1b. hipDNN execute
        e_dnn = Engine(PK, M, maxlen=MAXLEN, layers=L)
        hs = HipDnnStep(e_dnn, handle)
        r["plan_ms"], r["engine"] = hs.plan_ms, hs.engine_name
        log(f"M={M} hipDNN engine {hs.engine_name} build_plans total {hs.plan_ms} ms")
        r["hash_hipdnn"] = run_hash(hs.step, toks)
        log(f"HASH M={M} hipdnn {r['hash_hipdnn']}")
        # 1c. hipDNN execute captured in torch.cuda.graph (tok copy, execute, pos += 1)
        e_g = Engine(PK, M, maxlen=MAXLEN, layers=L)
        hg = HipDnnStep(e_g, handle)
        static_tok = torch.zeros(M, dtype=toks.dtype, device="cuda")
        g = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(g):
                handle.set_stream(torch.cuda.current_stream().cuda_stream)
                hg.step(static_tok)
            handle.set_stream(stream.cuda_stream)

            def gstep(tk):
                static_tok.copy_(tk)
                g.replay()
                return e_g.logits

            r["hash_hipdnn_graph"] = run_hash(gstep, toks)
        except Exception as ex:
            handle.set_stream(stream.cuda_stream)
            gstep = None
            r["hash_hipdnn_graph"] = f"FAILED {type(ex).__name__}: {ex}"
        log(f"HASH M={M} hipdnn-graph {r['hash_hipdnn_graph']}")
        # 2. LONG greedy steps side by side, bitwise per step
        e_fly.reset()
        e_dnn.reset()
        tok = toks[0].clone()
        first_diff, maxabs = None, 0.0
        for t in range(LONG):
            a = e_fly.step(tok).clone()
            b = hs.step(tok).clone()
            if not torch.equal(a, b) and first_diff is None:
                first_diff = t
                maxabs = float((a - b).abs().max())
            tok = a.argmax(-1)
        torch.cuda.synchronize()
        r["long"] = dict(steps=LONG, first_diff=first_diff, maxabs=maxabs,
                         last_tok=tok.tolist())
        log(f"LONG M={M} steps={LONG} bitwise_equal={first_diff is None} first_diff={first_diff} "
            f"maxabs={maxabs} last_tok={tok.tolist()}")
        # 3. host time of one launch (state garbage from here; reset before TPOT)
        hm = HipModule(e_dnn)
        tp_, gp_ = e_fly.tab.data_ptr(), e_fly.glob.data_ptr()
        fly = lambda: e_fly.launch(tp_, gp_, L, stream=stream)  # noqa: E731
        cf = flyc.compile(e_fly.launch, tp_, gp_, L, stream)
        flyc_ = lambda: cf(tp_, gp_, L, stream)  # noqa: E731
        r["host_us"] = dict(flydsl=host_us(fly), flydsl_compiled=host_us(flyc_),
                            hipModuleLaunchKernel=host_us(hm.launch), hipdnn_execute=host_us(hs.launch))
        if gstep is not None:
            r["host_us"]["graph_replay_hipdnn"] = host_us(g.replay)
        log(f"HOST_US M={M} {json.dumps(r['host_us'])}")

        def fly_step(tk):
            e_fly.tok.copy_(tk.to(torch.int32))
            fly()
            e_fly.pos += 1
            return e_fly.logits

        def flyc_step(tk):
            e_fly.tok.copy_(tk.to(torch.int32))
            flyc_()
            e_fly.pos += 1
            return e_fly.logits

        def hm_step(tk):
            e_dnn.tok.copy_(tk.to(torch.int32))
            hm.launch()
            e_dnn.pos += 1
            return e_dnn.logits

        first = toks[0].clone()
        paths = dict(flydsl=(e_fly, fly_step), flydsl_compiled=(e_fly, flyc_step),
                     hipModuleLaunchKernel=(e_dnn, hm_step), hipdnn_execute=(e_dnn, hs.step))
        if gstep is not None:
            paths["graph_replay_hipdnn"] = (e_g, gstep)
        # interleaved rounds: every path once per round, so clock drift hits all paths alike
        rounds = int(os.environ.get("MK_TPOT_ROUNDS", "7"))
        for key, sync in (("tpot_ms", True), ("step_ms_async", False)):
            all_ = {k: [] for k in paths}
            for _ in range(rounds):
                for k, (e, f) in paths.items():
                    all_[k].append(tpot(e, f, first, sync=sync))
            r[key] = {k: round(statistics.median(v), 4) for k, v in all_.items()}
            r[key + "_minmax"] = {k: [round(min(v), 4), round(max(v), 4)] for k, v in all_.items()}
            log(f"{key.upper()} M={M} rounds={rounds} median {json.dumps(r[key])} "
                f"min/max {json.dumps(r[key + '_minmax'])}")
        del e_fly, e_dnn, e_g, g, hs, hg
        torch.cuda.empty_cache()
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
    log(json.dumps(res["Ms"], default=str))


if __name__ == "__main__":
    main()
