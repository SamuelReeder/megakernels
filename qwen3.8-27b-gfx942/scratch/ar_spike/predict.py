"""Spike only (disposable). Go / no-go: predicted TP2 step = W / 2 + F + 128 x a, with
T = W + F the TP1 step (7.56 / 10.26 ms, bench/RESULTS.md), and a the all-reduce cost the
spike measured (round time with the all-reduce minus round time of the barrier-only
baseline, rank 0, stamped runs; p99 variant too).

Two splits of T into weight stream W and fixed cost F:
- "40 %": F = 0.40 T (the objective's figure, M = 1 only; applied to M = 8 as well);
- "bytes / LM-head rate": W = (12.94 GB MXFP4 linear + 2.54 GB bf16 LM head) / B, B the
  LM head's own stream rate in the TP1 timeline at the same M (bench/results/timeline).
Usage: python predict.py SPIKE_JSON [TIMELINE_DIR]
"""

import json
import sys

T1 = {1: 7.56, 8: 10.26}
LIN = 24.35e9 * 0.53125  # MXFP4 linear weights + e8m0 scales, bytes
LM = 248320 * 5120 * 2


def lm_rate(tl):
    d = json.load(open(tl))
    us = d["us"]
    rel = min(c[1] for c in us[-2])
    arr = max(c[0] for c in us[-1])
    return LM / ((arr - rel) * 1e-6), arr - rel


def main():
    spike = json.load(open(sys.argv[1]))
    tld = sys.argv[2] if len(sys.argv) > 2 else "bench/results/timeline"
    runs = {(r["M"], r["mode"], r["stamps"]): r for r in spike["runs"]}
    print(f"host {spike['host']} {spike['gpu']} cus {spike['cus']} tp {spike['tp']} rounds {spike['rounds']}")
    for M, mode in sorted({(k[0], k[1]) for k in runs if k[1] != "local"}):
        ar, lo = runs[(M, mode, True)], runs[(M, "local", True)]
        are, loe = runs[(M, mode, False)], runs[(M, "local", False)]
        a_med = ar["round_med"] - lo["round_med"]
        a_p99 = ar["round_p99"] - lo["round_med"]
        a_ev = are["launch_per_round_med"] - loe["launch_per_round_med"]
        print(f"\nM={M} {mode}: exact {ar['exact']} {are['exact']} ranks_equal {ar.get('ranks_equal')} {are.get('ranks_equal')}")
        print(f"  round (stamps) {mode} med {ar['round_med']:.2f} p99 {ar['round_p99']:.2f} max {ar['round_max']:.2f} us;"
              f" local med {lo['round_med']:.2f} p99 {lo['round_p99']:.2f} us")
        print(f"  all-reduce phase (stamps) med {ar['phase_med']:.2f} p99 {ar['phase_p99']:.2f} us;"
              f" local phase med {lo['phase_med']:.2f}")
        print(f"  launch/128 (events) {mode} {are['launch_per_round_med']:.2f} p99 {are['launch_per_round_p99']:.2f};"
              f" local {loe['launch_per_round_med']:.2f} us")
        print(f"  added cost per all-reduce: med {a_med:.2f} us, p99 {a_p99:.2f} us, events {a_ev:.2f} us")
        if M not in T1:
            continue
        rate, lm_us = lm_rate(f"{tld}/timeline_m{M}.json")
        T = T1[M]
        for split, W in (("40 %", 0.6 * T), (f"bytes / LM-head rate {rate / 1e12:.2f} TB/s", (LIN + LM) / rate * 1e3)):
            F = T - W
            for lab, a in (("med", a_med), ("p99", a_p99)):
                p = W / 2 + F + 128 * a * 1e-3
                print(f"  [{split:32s}] W {W:.2f} F {F:.2f} a({lab}) {a:.2f} -> TP2 {p:.2f} ms vs TP1 {T:.2f}:"
                      f" {100 * (1 - p / T):+.1f} % faster")


if __name__ == "__main__":
    main()
