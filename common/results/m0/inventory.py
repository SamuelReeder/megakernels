# M0 inventory: python3 common/results/m0/inventory.py mk/dev.py ... (repo root; per-name ast source hash, labels A-D, change vs 13b25e7 / qwen38-tp2)
import ast, hashlib, subprocess, sys
def show(ref, p):
    r = subprocess.run(["git","show",f"{ref}:{p}"],capture_output=True,text=True)
    return r.stdout if r.returncode==0 else None
def defs(src):
    out={}
    for n in ast.parse(src).body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef,ast.AsyncFunctionDef)): k=n.name
        elif isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name): k=n.targets[0].id
        elif isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name): k=n.target.id
        else: continue
        out[k]=hashlib.sha1(ast.get_source_segment(src,n).encode()).hexdigest()[:8]
    return out
M={"g":"gemma4-26b-a4b-gfx942","q6":"qwen3.6-35b-a3b-gfx942","q8":"qwen3.8-27b-gfx942"}
OLD={"g":"13b25e7","q6":"13b25e7","q8":"origin/users/sareeder/qwen38-tp2"}
NEW="origin/main"
files=sys.argv[1:]
for f in files:
    srcs={k:show(NEW,f"{d}/{f}") for k,d in M.items()}
    olds={k:show(OLD[k],f"{d}/{f}") for k,d in M.items()}
    print(f"\n### {f} lines", {k:(len(s.splitlines()) if s else None) for k,s in srcs.items()})
    if not f.endswith(".py"):
        print(" file-eq", {k:(hashlib.sha1(s.encode()).hexdigest()[:8] if s else None) for k,s in srcs.items()}, "changed-vs-old", {k:(srcs[k]!=olds[k]) for k in M})
        continue
    D={k:defs(s) for k,s in srcs.items() if s}; O={k:defs(s) for k,s in olds.items() if s}
    names=sorted(set().union(*D.values()))
    for n in names:
        hs={k:D[k].get(n) for k in D}
        lab={};row=[]
        for k in M:
            h=hs.get(k)
            if h is None: row.append("."); continue
            lab.setdefault(h,"ABCD"[len(lab)]); row.append(lab[h])
        ch=[k for k in D if hs[k] != (O.get(k,{}).get(n))]
        print(f"  {n:28s} {' '.join(row)}  changed_since_old:{','.join(ch) or '-'}")
