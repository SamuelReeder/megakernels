"""Disposable: for two statecheck runs, (1) whether every forward input (token count,
input_ids sum, positions sum) that occurs in both runs gives one hidden-state sum,
(2) the first call whose input differs, (3) per cache layer kind (GDN / attention) the
number of layers whose state differs before that call and after the call before it.
Usage: python bench/results/tp2/m6/stateinputs.py RUN1_DIR RUN2_DIR [rank]"""
import sys
from collections import defaultdict


def load(d, r):
    calls, names, launches = [], None, []
    for l in open(f"{d}/state_r{r}.txt"):
        if l.startswith("layers"):
            names = [x.split(":")[1] for x in l.split()[1:]]
        elif l.startswith("launch"):
            launches.append(l.strip())
        else:
            h, pre, post = l.rstrip("\n").split(" | ")
            f = h.split()
            calls.append(dict(n=int(f[1]), key=(f[3], f[5], f[7]), hidden=f[11],
                              pre=pre[4:].split(";"), post=post[5:].split(";")))
    return calls, names, launches


r = sys.argv[3] if len(sys.argv) > 3 else "0"
(a, names, la), (b, _, lb) = (load(d, r) for d in sys.argv[1:3])
hid = defaultdict(set)
for c in a + b:
    hid[c["key"]].add(c["hidden"])
real = [k for k in hid if k[1] != "0"]
print(f"calls {len(a)} / {len(b)}; real inputs {len(real)}, each with one hidden sum: "
      f"{all(len(hid[k]) == 1 for k in real)}; eager kernel launches equal: {la == lb} ({len(la)})")
i = next((i for i, (x, y) in enumerate(zip(a, b)) if x["key"] != y["key"]), None)
first_real = next(j for j, c in enumerate(a) if c["key"][1] != "0")
for j in (first_real, i):
    if j is None:
        continue
    x, y = a[j], b[j]
    kinds = defaultdict(lambda: [0, 0])
    for nm, u, v in zip(names, x["pre"], y["pre"]):
        kinds[nm][0] += 1
        kinds[nm][1] += u != v
    print(f"call {x['n']}: input {x['key']} / {y['key']}; cache layers differing before it: "
          + ", ".join(f"{k} {d}/{n}" for k, (n, d) in kinds.items()))
