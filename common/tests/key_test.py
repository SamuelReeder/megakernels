"""The compile key of one code object (common/mkcore/key.py compile_key on Cfg.compiled()):
at rt_geom 2 two Cfgs that differ only in max_model_len (maxlen and every maxblocks* field)
have one key and one kernel name; at rt_geom 0 they differ; a compiled-in field (the KV
block size) changes the key at rt_geom 2 too. The hipDNN route keys its code objects on
this key (docs/hipdnn-route.md section 4, `src`). CPU only, no build.
Usage (venv, repo root): python common/tests/key_test.py [MODEL_DIR ...]
(default: Qwen3.8 at TP 1 and 2, Qwen3.6, Gemma; one process per model: each has `mk`)."""
import dataclasses
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODELS = ("qwen3.8-27b-gfx942", "qwen3.6-35b-a3b-gfx942", "gemma4-26b-a4b-gfx942")


def filler(f):
    t = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", str(f.type))
    return {"int": 64, "bool": False, "tuple": (1, 2, 3, 4)}[t]


def check(model_dir):
    sys.path.insert(0, str(model_dir))
    import mk  # noqa: F401  (puts common/ on sys.path)
    from mk import kernel as KK
    from mkcore import key as K

    pkg = pathlib.Path(KK.__file__).resolve().parent
    req = {f.name: filler(f) for f in dataclasses.fields(KK.Cfg)
           if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING}
    req.update(M=1, cus=304)
    names = {f.name for f in dataclasses.fields(KK.Cfg)}
    blk = next(n for n in ("block", "block_s") if n in names)
    tps = (1, 2) if "tp" in names else (None,)
    fails = 0
    for tp in tps:
        extra = {} if tp is None else {"tp": tp}

        def cfg(rt, maxlen, **kw):
            nb = -(-maxlen // 16)
            geo = {n: nb for n in names if n.startswith("maxblocks")}
            return KK.Cfg(**{**req, **extra, **geo, "maxlen": maxlen, "rt_geom": rt, **kw})

        def key(c):
            return K.compile_key(c, pkg)

        a2, b2 = cfg(2, 4096), cfg(2, 8192)
        rows = [
            ("rt2 maxlen 4096 == 8192", key(a2) == key(b2)),
            ("rt2 kernel name 4096 == 8192",
             K.kernel_name("k", a2, key(a2)) == K.kernel_name("k", b2, key(b2))),
            ("rt2 key(cfg) == key(cfg.compiled())", key(a2) == key(a2.compiled())),
            ("rt2 block 16 != 32", key(cfg(2, 4096, **{blk: 16})) != key(cfg(2, 4096, **{blk: 32}))),
            ("rt0 maxlen 4096 != 8192", key(cfg(0, 4096)) != key(cfg(0, 8192))),
            ("rt0 != rt2", key(cfg(0, 4096)) != key(a2)),
        ]
        for name, ok in rows:
            fails += not ok
            print(f"KEY {model_dir.name}{'' if tp is None else f' tp{tp}'} {name}: "
                  f"{'PASS' if ok else 'FAIL'}")
    return fails


def main():
    dirs = [pathlib.Path(d).resolve() for d in sys.argv[1:]]
    if len(dirs) == 1:
        sys.exit(1 if check(dirs[0]) else 0)
    rc = 0
    for d in dirs or [ROOT / m for m in MODELS]:
        rc |= subprocess.run([sys.executable, "-B", __file__, str(d)]).returncode
    print("KEY RESULT", "PASS" if rc == 0 else "FAIL")
    sys.exit(rc)


if __name__ == "__main__":
    main()
