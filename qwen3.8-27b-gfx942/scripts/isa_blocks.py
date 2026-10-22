"""Instruction mix of the basic blocks with MFMAs in a dumped ISA (GEMV loop bodies)."""
import collections
import re
import sys

L = open(sys.argv[1]).read().split("\n")
blocks, cur = [], None
for i, line in enumerate(L):
    m = re.match(r"^(\.LBB\d+_\d+):", line)
    if m:
        cur = [m.group(1), i, []]
        blocks.append(cur)
        continue
    if cur and line.startswith("\t") and not line.startswith(("\t.", "\t;")):
        cur[2].append(line.split()[0])
for name, i, ins in blocks:
    mf = sum(1 for x in ins if x.startswith("v_mfma"))
    if mf >= 4:
        c = collections.Counter(ins)
        valu = sum(v for k, v in c.items() if k.startswith("v_") and not k.startswith("v_mfma"))
        salu = sum(v for k, v in c.items() if k.startswith("s_"))
        print(name, i, "n", len(ins), "mfma", mf, "valu", valu, "salu", salu,
              "vmem", sum(v for k, v in c.items() if k.startswith("buffer_")))
