#!/usr/bin/env python3
"""Dry run of driver.py on CPU: FakeSlurm jobs, stub stages (stage.sh PIPE_DRY=1), one
pipeline minute = PIPE_SCALE seconds. Checks the orchestration from the driver log:
  - every stage starts only after its dependencies ended, on a job with enough GPUs and
    with its estimate plus the margin left before the job's end;
  - no GPU of a job runs two stages at once, and a CPU stage has its job alone;
  - at most 3 jobs alive (pending or running) at any time; jobs 68209506 / sdpabwd never touched;
  - every stage ends PASS, FAIL, ERROR or SKIP as the scenario expects.
Scenarios: real (the held jobs' estimated starts; hvco tests take 60 min; one FAIL, one
ERROR then PASS) and chaos (the coexport job lands on a node without the shared home; the
driver is killed with stages running and restarted; e_g fails twice with ERROR).
Usage: python3 dryrun.py real|chaos   (output in /tmp/mkpipe_dry_<scenario>)
"""
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import time

here = pathlib.Path(__file__).resolve().parent
sc = sys.argv[1] if len(sys.argv) > 1 else "real"
scale = float(os.environ.get("PIPE_SCALE", "0.2"))
D = pathlib.Path(f"/tmp/mkpipe_dry_{sc}")
shutil.rmtree(D, ignore_errors=True)
(D / "p").mkdir(parents=True)
ST = D / "out/state"
ST.mkdir(parents=True)
for f in ("driver.py", "stage.sh"):
    shutil.copy(here / f, D / "p" / f)
(D / "p/pins.env").write_text("PIPE_REV=dry\n")
env = dict(os.environ, PIPE_DIR=str(D / "p"), PIPE_OUT=str(D / "out"), PIPE_DRY="1",
           PIPE_SCALE=str(scale), PIPE_POLL=str(scale * 2))
t0 = time.time() / scale  # fake minutes
jobs = {"68215720": dict(gpus=2, start_at=t0 + 10, limit=180, name="vrt2cc"),
        "68215745": dict(gpus=1, start_at=t0 + 10, limit=240, hvco_done_after=60, name="hvco"),
        "68215802": dict(gpus=2, start_at=t0 + 878, limit=360, name="coexport")}
outcome = {"c_rt0_g": "FAIL", "c_rt2_q6": "ERROR PASS"}
expect = {"c_rt0_g": "FAIL"}
if sc == "chaos":
    jobs["68215802"]["bad"] = True
    outcome = {"e_g": "ERROR ERROR", "a_cc": "FAIL"}
    expect = {"e_g": "ERROR", "a_cc": "FAIL"}
(ST / "fake_jobs.json").write_text(json.dumps(jobs))
for k, v in outcome.items():
    (ST / f"dry_{k}").write_text(v + "\n")

log = open(D / "out/driver.log", "a")
drv = lambda: subprocess.Popen([sys.executable, str(D / "p/driver.py")], stdout=log,
                               stderr=subprocess.STDOUT, env=env)
p = drv()
if sc == "chaos":  # kill the driver while stages run (minute ~100), restart it 10 minutes later
    time.sleep(100 * scale)
    p.send_signal(signal.SIGKILL)
    p.wait()
    log.write("dryrun: driver killed\n")
    log.flush()
    time.sleep(10 * scale)
    p = drv()
p.wait(timeout=5000 * scale)
log.close()

# ---- checks
sys.path.insert(0, str(D / "p"))
import driver as dr  # noqa: E402  (SPEC, MARGIN)

text = (D / "out/driver.log").read_text()
lines = text.splitlines()
st = json.loads((ST / "stages.json").read_text())
jb = json.loads((ST / "jobs.json").read_text())
ok = True


def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg)
    ok &= bool(cond)


ended, running, slots_busy, alive = set(), {}, {}, set(jobs)
max_alive = len(alive)
for i, ln in enumerate(lines):
    m = re.search(r"stage (\S+) started on job (\S+) gpus (\[.*?\]|-) \(estimate (\d+) min, (-?\d+) min left\)", ln)
    if m:
        name, job, gl, est, left = m.groups()
        sl = re.findall(r"'(\d+)'", gl)
        sp = dr.SPEC[name]
        check(all(d in ended for d in sp["deps"]), f"{name}: dependencies ended before start")
        if job != "-":
            g = jb[job]["gpus"]
            check(sp["gpus"] <= g and len(sl) == sp["gpus"], f"{name}: {sp['gpus']} GPU(s) on job {job} ({g} GPUs) slots {sl}")
            check(int(left) + dr.MARGIN / 2 >= int(est) + dr.MARGIN - 1, f"{name}: estimate {est} <= {left} min left on {job}")
            busy = slots_busy.setdefault(job, {})  # GPU index or "cpu" -> stage
            clash = "cpu" in busy or (not sl and busy) or any(x in busy for x in sl)
            check(not clash, f"{name}: job {job} free for it ({busy})")
            for x in sl or ["cpu"]:
                busy[x] = name
            running[name] = (job, sl or ["cpu"])
        continue
    m = re.search(r"stage (\S+) ended on job (\S+): rc (\S+) -> (\S+)", ln)
    if m:
        name = m.group(1)
        if name in running:
            job, keys = running.pop(name)
            for x in keys:
                slots_busy[job].pop(x, None)
        if not (i + 1 < len(lines) and f"stage {name}: retry" in lines[i + 1]):
            ended.add(name)
        continue
    m = re.search(r"stage (\S+): SKIP", ln)
    if m:
        ended.add(m.group(1))
    m = re.search(r"submitted job (\S+)", ln)
    if m:
        alive.add(m.group(1))
    m = re.search(r"job (\S+) (released|ended)", ln)
    if m:
        alive.discard(m.group(1))
    max_alive = max(max_alive, len(alive))
check(max_alive <= dr.MAX_JOBS, f"at most {dr.MAX_JOBS} jobs alive (max {max_alive})")
check("68209506" not in text and "sdpabwd" not in text, "68209506 / sdpabwd never touched")
check((ST / "ALL_DONE").exists(), "ALL_DONE written")
for name in dr.SPEC:
    want = expect.get(name, "PASS")
    check(st[name]["state"] == want, f"{name}: {st[name]['state']} (expected {want}), job {st[name].get('job')}, attempts {st[name]['attempts']}")
if sc == "chaos":
    check("dryrun: driver killed" in text and "adopted, still running" in text, "restart adopts the running stages")
    check(jb["68215802"]["probe"] == "bad" and jb["68215802"]["released"], "bad job 68215802 released")
fj = json.loads((ST / "fake_jobs.json").read_text())
print("jobs:", {k: (v["gpus"], v.get("name"), "cancelled" if v.get("cancelled") else "") for k, v in fj.items()})
print("DRYRUN", sc, "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
