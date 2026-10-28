#!/usr/bin/env python3
"""Summarize raw results under results/ into markdown table rows.

vLLM dirs (vllm[_variant]_<ckpt>): `vllm bench serve` JSON files c{B}.json, c1_rep{2,3}.json.
SGLang dirs (sglang_<ckpt>): `sglang.bench_one_batch` logs sweep.log, b1_rep{2,3}.log
(median decode latency per step = TPOT).
usage: summarize.py <results_dir>
"""
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])


def env(d):
    e = (d / "env.txt").read_text()
    node = re.search(r"^node: (\S+)", e, re.M).group(1).split(".")[0]
    date = re.search(r"^date: (\S+)", e, re.M).group(1)[:10]
    return node, date


def row(ckpt, engine, batch, tpot, node, date, note=""):
    print(f"| {ckpt} | {engine} | {batch} | {tpot:.2f} | {1000 / tpot:.1f} | {node} | {date} | {note} |")


for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("INVALID")):
    node, date = env(d)
    if d.name.startswith("vllm"):
        m = re.match(r"vllm(?:_(aiter\w*))?_(gemma-4-26B-A4B-it[^_]*)(_run1_partial)?$", d.name)
        variant, ckpt, run1 = m.groups()
        ver = re.search(r"^vllm (\S+)", (d / "env.txt").read_text(), re.M).group(1)
        engine = f"vLLM {ver}" + (" +AITER" if variant else "")
        names = [(f"c{b}", b, "") for b in (1, 2, 4, 8)] + [(f"c1_rep{r}", 1, f"rep {r}") for r in (2, 3)]
        for name, b, note in names:
            f = d / f"{name}.json"
            if not f.exists():
                continue
            j = json.loads(f.read_text())
            n = "; ".join(x for x in (note, "earlier run (partial)" if run1 else "") if x)
            n = (n + "; " if n else "") + f"p90 {j['p90_tpot_ms']:.2f}, TTFT {j['median_ttft_ms']:.0f} ms, n={j['completed']}"
            row(ckpt, engine, b, j["median_tpot_ms"], node, date, n)
    elif d.name.startswith("sglang"):
        ckpt = d.name[len("sglang_"):]
        ver = re.search(r"^sglang (\S+)", (d / "env.txt").read_text(), re.M).group(1)
        for log, note in (("sweep", ""), ("b1_rep2", "rep 2"), ("b1_rep3", "rep 3")):
            f = d / f"{log}.log"
            if not f.exists():
                continue
            txt = f.read_text()
            # each batch size prints its Decode steps then the median; the first pass is bench_one_batch's warmup
            bs = [int(x) for x in re.findall(r"Decode \d+\. Batch size: (\d+)", txt)]
            meds = [float(x) for x in re.findall(r"Decode\.\s+median latency: ([\d.]+) s", txt)]
            order = list(dict.fromkeys(bs))  # batch sizes in order of first appearance
            # warmup runs once (first batch size) before the measured sweep
            measured = meds[-len(order):] if len(meds) > len(order) else meds
            for b, med in zip(order, measured):
                row(ckpt, f"SGLang {ver}", b, med * 1000, node, date, note)
