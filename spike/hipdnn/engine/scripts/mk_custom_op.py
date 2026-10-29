"""Slice B smoke: hipDNN custom op `megakernel.qwen3_8_27b_decode` -> hip-kernel-provider
megakernel ingestor pack -> stand-in kpack kernel. Checks the pattern, picks the M variant,
times host overhead per execute, and tries HIP stream capture around execute.

Usage: python mk_custom_op.py [--outputs N] [--iters N]
"""

import argparse
import ctypes
import json
import time

import numpy as np
MODEL = "qwen3.8-27b"
BUILD_ONLY = False

import hipdnn_frontend as hipdnn

CUS, NT, NLAYERS = 304, 512, 64
N = CUS * NT
TAB_WORDS = 64
TAB_MAGIC = 0x0123456789ABCDEF
DT = hipdnn.DataType

hip = ctypes.CDLL("libamdhip64.so")
for name in ("hipStreamCreate", "hipStreamSynchronize", "hipStreamBeginCapture",
             "hipStreamEndCapture", "hipGraphInstantiate", "hipGraphLaunch", "hipMemsetAsync",
             "hipGetErrorString", "hipGraphGetNodes"):
    getattr(hip, name).restype = ctypes.c_int if name != "hipGetErrorString" else ctypes.c_char_p


def check(err, what):
    if err != 0:
        raise RuntimeError(f"{what}: hip error {err} {hip.hipGetErrorString(err).decode()}")


def ok(err, what):
    good = err.is_good()
    print(f"  {what}: {'OK' if good else 'FAIL ' + err.get_message()}", flush=True)
    return good


def expected(m):
    i = np.arange(N, dtype=np.uint64)
    return (np.uint64(0x4D4B) << np.uint64(48)) | (np.uint64(m) << np.uint64(40)) | (
        np.uint64(NLAYERS & 0xFF) << np.uint64(32)) | i


def build(m, num_outputs, handle):
    g = hipdnn.Graph()
    g.set_name(f"megakernel_m{m}")
    g.set_io_data_type(DT.FLOAT).set_intermediate_data_type(DT.FLOAT).set_compute_data_type(
        DT.FLOAT)
    tab = hipdnn.Tensor.create([TAB_WORDS], DT.INT64)
    tab.set_name("tab")
    glob = hipdnn.Tensor.create([N + 1], DT.INT64)
    glob.set_name("glob")
    payload = {"model": MODEL, "M": m, "nlayers": NLAYERS,
               "geometry": {"grid": [CUS], "block": [NT], "lds": 0}}
    attrs = hipdnn.CustomOpAttributes()
    attrs.set_custom_op_id("megakernel.qwen3_8_27b_decode")
    attrs.set_compute_data_type(DT.FLOAT)
    attrs.set_data(list(json.dumps(payload).encode()))
    outs = g.custom_op([tab, glob], num_outputs, attrs)
    for o in outs:
        o.set_dim([1]).set_stride([1]).set_data_type(DT.INT64).set_output(True)
    steps = [("validate", lambda: g.validate()),
             ("build_operation_graph", lambda: g.build_operation_graph(handle)),
             ("create_execution_plans", lambda: g.create_execution_plans()),
             ("check_support", lambda: g.check_support()),
             ("build_plans", lambda: g.build_plans())]
    for what, fn in steps:
        if not ok(fn(), what):
            return None
    eid = g.get_execution_plan_engine_id()
    eid = eid[-1] if isinstance(eid, tuple) else eid
    print(f"  engine id {eid} name {handle.engine_id_to_name(eid)} "
          f"workspace {g.get_workspace_size()}", flush=True)
    return g, tab, glob, outs


def run(m, args, handle, stream):
    print(f"== M={m}", flush=True)
    t0 = time.perf_counter()
    built = build(m, args.outputs, handle)
    print(f"  graph build+plan ms {1e3 * (time.perf_counter() - t0):.1f}", flush=True)
    if built is None:
        return False
    g, tab, glob, outs = built
    tab_host = np.zeros(TAB_WORDS, dtype=np.uint64)
    tab_host[0] = TAB_MAGIC
    tab_buf = hipdnn.DeviceBuffer(tab_host.nbytes)
    tab_buf.copy_from_host(tab_host.tobytes())
    glob_buf = hipdnn.DeviceBuffer((N + 1) * 8)
    glob_buf.zeros()
    out_bufs = [hipdnn.DeviceBuffer(8) for _ in outs]
    vp = {tab.get_uid(): tab_buf.ptr(), glob.get_uid(): glob_buf.ptr()}
    for o, b in zip(outs, out_bufs):
        vp[o.get_uid()] = b.ptr()

    if BUILD_ONLY:
        return True
    if not ok(g.execute(handle, vp, 0), "execute"):
        return False
    check(hip.hipStreamSynchronize(stream), "sync")
    got = np.frombuffer(glob_buf.copy_to_host(), dtype=np.uint64)
    exp = expected(m)
    good = np.array_equal(got[:N], exp) and got[N] == TAB_MAGIC
    print(f"  pattern {'OK' if good else 'MISMATCH'}: glob[0]={got[0]:#x} glob[N-1]={got[N-1]:#x}"
          f" glob[N]={got[N]:#x}", flush=True)

    # Host overhead per execute: graph built once, execute N times, async, then one sync.
    for _ in range(10):
        g.execute(handle, vp, 0)
    check(hip.hipStreamSynchronize(stream), "sync")
    t0 = time.perf_counter()
    for _ in range(args.iters):
        g.execute(handle, vp, 0)
    t_host = time.perf_counter() - t0
    check(hip.hipStreamSynchronize(stream), "sync")
    t_all = time.perf_counter() - t0
    print(f"  execute x{args.iters}: host us/call {1e6 * t_host / args.iters:.2f}, "
          f"wall incl. GPU us/call {1e6 * t_all / args.iters:.2f}", flush=True)

    # Stream capture around execute, then replay.
    glob_buf.zeros()
    check(hip.hipStreamSynchronize(stream), "sync")
    graph = ctypes.c_void_p()
    exe = ctypes.c_void_p()
    try:
        check(hip.hipStreamBeginCapture(ctypes.c_void_p(stream), 0), "BeginCapture(global)")
        res = g.execute(handle, vp, 0)
        check(hip.hipStreamEndCapture(ctypes.c_void_p(stream), ctypes.byref(graph)), "EndCapture")
        print(f"  capture execute: {'OK' if res.is_good() else 'FAIL ' + res.get_message()}")
        nnodes = ctypes.c_size_t(0)
        check(hip.hipGraphGetNodes(graph, None, ctypes.byref(nnodes)), "GetNodes")
        check(hip.hipGraphInstantiate(ctypes.byref(exe), graph, None, None, 0), "Instantiate")
        got = np.frombuffer(glob_buf.copy_to_host(), dtype=np.uint64)
        print(f"  after capture (not launched) glob[0]={got[0]:#x} graph nodes={nnodes.value}")
        check(hip.hipGraphLaunch(exe, ctypes.c_void_p(stream)), "GraphLaunch")
        check(hip.hipStreamSynchronize(stream), "sync")
        got = np.frombuffer(glob_buf.copy_to_host(), dtype=np.uint64)
        good2 = np.array_equal(got[:N], exp) and got[N] == TAB_MAGIC
        print(f"  graph replay pattern {'OK' if good2 else 'MISMATCH'} glob[0]={got[0]:#x}")
        t0 = time.perf_counter()
        for _ in range(args.iters):
            hip.hipGraphLaunch(exe, ctypes.c_void_p(stream))
        t_host = time.perf_counter() - t0
        check(hip.hipStreamSynchronize(stream), "sync")
        print(f"  graph replay x{args.iters}: host us/call {1e6 * t_host / args.iters:.2f}, wall "
              f"us/call {1e6 * (time.perf_counter() - t0) / args.iters:.2f}")
    except RuntimeError as exc:
        print(f"  capture FAILED: {exc}")
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=int, default=0)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--m", type=int, nargs="*", default=[1, 8, 4])
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--build-only", action="store_true",
                    help="build plans (loads the code object) but do not execute")
    args = ap.parse_args()
    global MODEL, BUILD_ONLY
    MODEL, BUILD_ONLY = args.model, args.build_only
    handle = hipdnn.create_handle()
    stream = ctypes.c_void_p()
    check(hip.hipStreamCreate(ctypes.byref(stream)), "hipStreamCreate")
    handle.set_stream(stream.value)
    stream = stream.value
    print(f"stream {stream:#x}", flush=True)
    for m in args.m:
        run(m, args, handle, stream)


if __name__ == "__main__":
    main()
