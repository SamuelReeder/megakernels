"""Disposable: compare two state_r<rank>.txt files of scratch/m6/statecheck line by line.
Prints the first differing line, which field differs (ids, pos, hidden, or the cache
layers before / after the call), and a count of differing lines.
Usage: python scratch/m6/statecmp.py A.txt B.txt"""
import sys


def parse(line):
    head, *rest = line.rstrip("\n").split(" | ")
    f = dict(zip(head.split()[2::2], head.split()[3::2])) if head.startswith("fwd") else {"launch": head}
    if rest:
        f["pre"] = rest[0][4:].split(";")
        f["post"] = rest[1][5:].split(";")
    return f


a, b = (open(p).readlines() for p in sys.argv[1:3])
print(f"lines {len(a)} / {len(b)}")
ndiff, first = 0, None
for i, (x, y) in enumerate(zip(a, b)):
    if x == y:
        continue
    ndiff += 1
    if first is None:
        first = i
        fx, fy = parse(x), parse(y)
        print(f"first differing line {i + 1}: {x[:60].strip()}")
        for k in fx:
            if k in ("pre", "post"):
                layers = [j for j, (u, v) in enumerate(zip(fx[k], fy[k])) if u != v]
                print(f"  {k}: {len(layers)} layers differ: {layers[:16]}")
            elif fx[k] != fy.get(k):
                print(f"  {k}: {fx[k]} vs {fy.get(k)}")
print(f"differing lines {ndiff} of {min(len(a), len(b))}")
