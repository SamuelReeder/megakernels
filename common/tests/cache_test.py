"""FlyDSL disk-cache test of the compile key (common/DESIGN.md I-key; M1, repeated at M2
and M8). Qwen3.6 at M = 1, 4 layers, 4 fixed-token steps, each run a fresh process with
the cache on in a fresh directory (common/results/cache/cache_model.py does the runs).
Cases (each: base, changed with the same cache, changed with the cache off):
  A  two Cfg that differ in one field read as cfg.* (not a scalar closure value):
     A1 lm_head True -> False, A2 MK_CFG=cta_red=0
  B  a comment line appended to a source file the kernel traces (EDIT_FILE, default
     qwen3.6-35b-a3b-gfx942/mk/dev.py; M2: common/mkcore/key.py)
  C  the str key in the launcher closure: C0 the same sources and Cfg twice (cache hit,
     no new entry); C1 a helper edit (dev.silu x (1 + 2^-10)) that changes the bits
Pass: every changed run with the cache equals the cache-off run, makes a new cache entry
(A, B, C1), and C0 makes none.
Usage (GPU node, venv): python common/tests/cache_test.py REPO_ROOT FP8_DIR WORKDIR [A,B,C]
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "cache"))
from cache_model import EDIT, run  # noqa: E402

MODEL = "qwen3.6-35b-a3b-gfx942"


def main():
    root0, fp8, work = sys.argv[1:4]
    only = sys.argv[4].split(",") if len(sys.argv) > 4 else ["A", "B", "C"]
    edit_file = os.environ.get("EDIT_FILE", f"{MODEL}/mk/dev.py")
    os.makedirs(work, exist_ok=True)
    root = os.path.join(work, "root")
    src = os.path.join(root, MODEL)
    off = os.path.join(work, "cache_off")
    bad = []

    def fresh(name):
        shutil.rmtree(root, ignore_errors=True)
        shutil.copytree(os.path.join(root0, MODEL), src)
        if os.path.isdir(os.path.join(root0, "common", "mkcore")):
            shutil.copytree(os.path.join(root0, "common", "mkcore"), os.path.join(root, "common", "mkcore"))
        cache = os.path.join(work, "cache_" + name)
        shutil.rmtree(cache, ignore_errors=True)
        return cache

    def r(tag, cache, on, **kw):
        return run(tag, src, fp8, work, cache, on, **kw)

    def judge(name, a, b, c, new=True):
        ok = (not b["rc"] and b["hash"] == c["hash"] and (b["new_cache_files"] > 0) == new)
        print(f"VERDICT {name}: {'PASS' if ok else 'FAIL'} (base {a['hash']} new {a['new_cache_files']}; "
              f"changed+cache {b['hash']} new {b['new_cache_files']}; changed, cache off {c['hash']})",
              flush=True)
        if not ok:
            bad.append(name)

    if "A" in only:
        cache = fresh("a1")
        a = r("A1 lm_head=True", cache, True)
        b = r("A1 lm_head=False, same cache", cache, True, lm="0")
        c = r("A1 lm_head=False, cache off", off, False, lm="0")
        judge("A1 cfg.lm_head", a, b, c)
        cache = fresh("a2")
        a = r("A2 default", cache, True)
        b = r("A2 MK_CFG=cta_red=0, same cache", cache, True, env_extra={"MK_CFG": "cta_red=0"})
        c = r("A2 MK_CFG=cta_red=0, cache off", off, False, env_extra={"MK_CFG": "cta_red=0"})
        judge("A2 cfg.cta_red", a, b, c)
    if "B" in only:
        cache = fresh("b")
        a = r("B base", cache, True)
        with open(os.path.join(root, edit_file), "a") as f:
            f.write("# cache test: a comment line\n")
        b = r(f"B comment in {edit_file}, same cache", cache, True)
        c = r(f"B comment in {edit_file}, cache off", off, False)
        judge(f"B comment edit {edit_file}", a, b, c)
        if b["hash"] != a["hash"]:
            bad.append("B changed the bits")
    if "C" in only:
        cache = fresh("c")
        a = r("C0 first build", cache, True)
        b = r("C0 second build, same cache", cache, True)
        judge("C0 same key: cache hit", a, b, a, new=False)
        p = os.path.join(src, "mk", "dev.py")
        s = open(p).read()
        assert s.count(EDIT[0]) == 1
        open(p, "w").write(s.replace(*EDIT))
        b = r("C1 edited silu, same cache", cache, True)
        c = r("C1 edited silu, cache off", off, False)
        judge("C1 helper edit (str key in the launcher closure)", a, b, c)
        if b["hash"] == a["hash"]:
            bad.append("C1 edit did not change the bits")
    shutil.rmtree(root, ignore_errors=True)
    print("PASS" if not bad else f"FAIL {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
