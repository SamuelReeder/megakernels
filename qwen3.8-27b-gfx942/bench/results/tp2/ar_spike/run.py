"""Spike only (disposable). One process per GPU (srun -n2): checks and times the
in-kernel all-reduce (arkernel.py) against the barrier-only baseline.

Usage: srun -n2 python bench/results/tp2/ar_spike/run.py OUT_DIR [M ...]
Writes OUT_DIR/spike_r{rank}.json; rank 0 prints the summary.
"""

import json
import os
import socket
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(__file__))
import arkernel as K  # noqa: E402
from peer import PeerBuffer  # noqa: E402

ROUNDS = 128
LAUNCHES = 200
STAMP_LAUNCHES = 20


def log(*a):
    print(f"[r{RANK} {time.strftime('%H:%M:%S')}]", *a, flush=True)


def quant(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def run_mode(M, mode, peer, flags, src, out, cus, stamps_on):
    launch = K.build(M, cus, TP, RANK, mode, stamps_on)
    stbuf = torch.zeros(
        (STAMP_LAUNCHES if stamps_on else 1) * ROUNDS * cus * 2, dtype=torch.int64, device="cuda"
    )
    args = (peer.addresses.data_ptr(), flags.data_ptr(), src.data_ptr(), out.data_ptr(), stbuf.data_ptr())
    s = torch.cuda.current_stream()
    out.zero_()
    # first launch compiles; then warm up
    for i in range(5):
        launch(*args, 0, ROUNDS, stream=s)
    torch.cuda.synchronize()
    res = {}
    if stamps_on:
        for i in range(STAMP_LAUNCHES):
            launch(*args, i, ROUNDS, stream=s)
        torch.cuda.synchronize()
        t = stbuf.view(STAMP_LAUNCHES, ROUNDS, cus, 2).double().cpu() / 100.0  # us
        start = t[..., 0].max(dim=2).values  # last CTA's release into round r
        # round time: last release into round r -> last release into round r + 1
        rt = (start[:, 1:] - start[:, :-1]).flatten().tolist()
        # all-reduce phase: last CTA done minus last CTA started (same round)
        ph = (t[..., 1].max(dim=2).values - start).flatten().tolist()
        res.update(
            round_med=quant(rt, 0.5), round_p99=quant(rt, 0.99), round_max=max(rt),
            phase_med=quant(ph, 0.5), phase_p99=quant(ph, 0.99),
        )
    else:
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(LAUNCHES + 1)]
        dist.barrier(GROUP)
        ev[0].record(s)
        for i in range(LAUNCHES):
            launch(*args, 0, ROUNDS, stream=s)
            ev[i + 1].record(s)
        torch.cuda.synchronize()
        per = [ev[i].elapsed_time(ev[i + 1]) * 1000 / ROUNDS for i in range(LAUNCHES)]
        res.update(launch_per_round_med=quant(per, 0.5), launch_per_round_p99=quant(per, 0.99),
                   total_ms=ev[0].elapsed_time(ev[-1]))
    # correctness of the last launch: every round's sum, bitwise, and equal on all ranks
    torch.cuda.synchronize()
    o = out.view(ROUNDS, M, K.H)
    if mode.startswith("ar"):
        cast = (lambda t: t.bfloat16().float()) if mode == "ar_bf" else (lambda t: t)
        exp = cast(SRCS[0][:ROUNDS, :M]).clone()
        for p in range(1, TP):
            exp = exp + cast(SRCS[p][:ROUNDS, :M])
    else:
        exp = SRCS[RANK][:ROUNDS, :M] + SRCS[RANK][:ROUNDS, :M]
    ok = bool(torch.equal(o.cpu(), exp))
    res["exact"] = ok
    if mode.startswith("ar"):
        mine = o.cpu().contiguous()
        allo = [torch.empty_like(mine) for _ in range(TP)]
        dist.all_gather(allo, mine, group=GROUP)
        res["ranks_equal"] = all(torch.equal(allo[0], a) for a in allo)
    return res


def main():
    global RANK, TP, GROUP, SRCS
    outdir = sys.argv[1]
    Ms = [int(a) for a in sys.argv[2:]] or [1, 8]
    RANK = int(os.environ["SLURM_PROCID"])
    TP = int(os.environ["SLURM_NTASKS"])
    torch.cuda.set_device(RANK)
    port = int(os.environ.get("SPIKE_PORT", "29533"))
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=RANK, world_size=TP,
                            timeout=__import__("datetime").timedelta(seconds=300))
    GROUP = dist.group.WORLD
    props = torch.cuda.get_device_properties(RANK)
    cus = props.multi_processor_count
    log(socket.gethostname(), props.name, props.gcnArchName, cus, "CUs",
        "visible", torch.cuda.device_count(), "ROCR", os.environ.get("ROCR_VISIBLE_DEVICES"),
        "peer", [torch.cuda.can_device_access_peer(RANK, p) for p in range(torch.cuda.device_count()) if p != RANK])
    peer = PeerBuffer(2 * K.half_bytes(TP), GROUP, RANK, TP, torch.device("cuda", RANK))
    log("peer buffer", [hex(a) for a in peer.addresses.tolist()])
    # every rank's partials, deterministic, so each rank can compute the exact sum
    SRCS = []
    for p in range(TP):
        g = torch.Generator().manual_seed(1000 + p)
        SRCS.append(torch.randn(ROUNDS, 8, K.H, generator=g, dtype=torch.float32))
    results = {"host": socket.gethostname(), "gpu": props.name, "cus": cus, "rank": RANK, "tp": TP,
               "rounds": ROUNDS, "runs": []}
    for M in Ms:
        src = SRCS[RANK][:, :M].contiguous().cuda()
        out = torch.zeros(ROUNDS * M * K.H, dtype=torch.float32, device="cuda")
        for mode in os.environ.get("SPIKE_MODES", "local ar ar_bf").split():
            for stamps_on in (False, True):
                flags = torch.zeros((cus + K.EPOCH_WORD_LINE + 1) * 32, dtype=torch.int32, device="cuda")
                # every rank starts the same epoch sequence: the tags must agree
                peer.bytes.zero_()
                torch.cuda.synchronize()
                dist.barrier(GROUP)
                r = run_mode(M, mode, peer, flags, src, out, cus, stamps_on)
                r.update(M=M, mode=mode, stamps=stamps_on)
                results["runs"].append(r)
                log(json.dumps(r))
                dist.barrier(GROUP)
    json.dump(results, open(os.path.join(outdir, f"spike_r{RANK}.json"), "w"), indent=1)
    peer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
