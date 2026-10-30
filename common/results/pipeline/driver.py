#!/usr/bin/env python3
"""Unattended GPU validation pipeline of the hipDNN megakernel route.

Runs on the Alola login node in tmux (run.sh). Every POLL seconds it:
  1. reads the state of its SLURM jobs (the three held jobs, plus any it submits);
  2. probes a job once it runs (home shared, gfx942 visible; the hvco job only after its
     own tests finished, ~/mk/out/pipeline/state/hvco_done_<job>);
  3. collects finished stages (stage.sh writes state/<stage>.rc; 0 PASS, 1 FAIL, else ERROR;
     an ERROR stage is retried once, a stage whose process vanished counts as ERROR);
  4. starts each ready stage, in STAGES order, on a running job with free GPUs and enough
     time left for its estimate (best fit), as a detached process (survives the driver);
  5. submits a replacement job (at most MAX_JOBS jobs alive) when no alive job can hold a
     stage, and releases a running job that no remaining stage fits;
  6. stops when every stage is PASS, FAIL, ERROR (after retries) or SKIP (a hard dependency
     ended in ERROR or SKIP), writes summary.txt and releases its jobs.
State lives in ~/mk/out/pipeline/state (stages.json, jobs.json), so a restart resumes.
Dry run (PIPE_DRY=1): FakeSlurm (state/fake_jobs.json) and stub stages; PIPE_SCALE seconds
per minute.
"""
import datetime as dt
import fcntl
import json
import os
import pathlib
import re
import subprocess
import sys
import time

HOME = pathlib.Path.home()
P = pathlib.Path(os.environ.get("PIPE_DIR", HOME / "mk/pipeline"))      # scripts, bundles
O = pathlib.Path(os.environ.get("PIPE_OUT", HOME / "mk/out/pipeline"))  # logs, state
ST = O / "state"
DRY = os.environ.get("PIPE_DRY") == "1"
SCALE = float(os.environ.get("PIPE_SCALE", "60"))  # seconds per minute
POLL = float(os.environ.get("PIPE_POLL", "60"))    # seconds
MAX_JOBS = 3          # our jobs alive (pending or running) at one time
MARGIN = 10           # minutes kept free before a job's end
MAX_ATTEMPTS = 2      # an ERROR stage runs at most twice
PROBE_TRIES = 3
NEVER = {"68209506"}  # jobs and names (sdpabwd_*) this driver must never touch
EXCLUDE = "ctr-cx63-mi300x-12"

# name, GPUs (None: login node, 0: CPU only), estimate (minutes), dependencies, soft
# (soft: wait for the dependencies to end in any state; hard: SKIP after ERROR or SKIP)
STAGES = [
    ("a_cc", 2, 150, [], False),           # VllmRt2 gpu.sh: gates new vs main, vLLM smoke, TP2
    ("b_mkeng", 1, 75, [], False),         # MkEngine gpu.sh: IntegrationGpuMegakernel.*
    ("c_rt0_q8", 2, 70, [], False),        # CoExporter rt0 set (q8 includes TP2)
    ("c_rt0_q6", 1, 45, [], False),
    ("c_rt0_g", 1, 45, [], False),
    ("c_rt0_merge", None, 2, ["c_rt0_q8", "c_rt0_q6", "c_rt0_g"], False),
    ("c_rt2_q8", 2, 90, [], False),        # CoExporter rt2 set (q8: maxlen 4096 and 8192, TP2)
    ("c_rt2_q6", 1, 55, [], False),
    ("c_rt2_g", 1, 45, [], False),
    ("c_rt2_merge", None, 2, ["c_rt2_q8", "c_rt2_q6", "c_rt2_g"], False),
    ("d_wheel", 0, 150, ["c_rt2_merge"], False),  # gfx94X wheel + drop4 wheel from the rt2 set
    ("e_q8", 1, 50, ["d_wheel"], False),   # e2e_plan.sh runs (flydsl, hipdnn, default, nowheel)
    ("e_q8tp2", 2, 50, ["d_wheel"], False),
    ("e_q6", 1, 50, ["d_wheel"], False),
    ("e_g", 1, 45, ["d_wheel"], False),
    ("f_drop_q8", 1, 20, ["d_wheel"], False),  # e2e_plan.sh drop (H2): hipdnn on the drop4 wheel
    ("f_drop_q6", 1, 20, ["d_wheel"], False),
    ("g_check", None, 2, ["e_q8", "e_q8tp2", "e_q6", "e_g", "f_drop_q8", "f_drop_q6"], True),
]
SPEC = {s[0]: dict(gpus=s[1], est=s[2], deps=s[3], soft=s[4]) for s in STAGES}
TERMINAL = {"PASS", "FAIL", "ERROR", "SKIP"}
# the held jobs at the start: id -> GPUs, kind (holder: sbatch --wrap sleep; hvco: srun
# of ~/work/hvco_run.sh, which holds the allocation after its tests while the driver lives)
INITIAL_JOBS = {"68215720": (2, "holder"), "68215745": (1, "hvco"), "68215802": (2, "holder")}


def now():
    return time.time() / SCALE


def log(msg):
    print(f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}", flush=True)


def load(name, default):
    f = ST / name
    return json.loads(f.read_text()) if f.exists() else default


def save(name, obj):
    f = ST / name
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    tmp.replace(f)


class Slurm:
    def query(self, ids):
        """id -> dict(state, start, end, limit, node, name) in minutes; ended jobs absent."""
        if not ids:
            return {}
        r = subprocess.run(["squeue", "-h", "-j", ",".join(ids), "-o", "%i|%T|%S|%e|%l|%N|%j"],
                           capture_output=True, text=True)
        if r.returncode != 0 and "Invalid job id" not in r.stderr:
            raise RuntimeError(f"squeue: {r.stderr.strip()}")
        out = {}
        for line in r.stdout.splitlines():
            i, state, start, end, limit, node, name = line.split("|")
            out[i] = dict(state=state, start=tmin(start), end=tmin(end), limit=limlen(limit),
                          node=node, name=name)
        return out

    def probe(self, job, gpus):
        cmd = ("hostname; test -d $HOME/mk/models && echo HOME_OK; "
               "rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+'")
        try:
            r = subprocess.run(["srun", f"--jobid={job}", "--overlap", "-n1", "--time=3",
                                "bash", "-c", cmd], capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            return False, "probe timeout"
        ok = r.returncode == 0 and "HOME_OK" in r.stdout and (gpus == 0 or "gfx942" in r.stdout)
        return ok, " ".join((r.stdout + r.stderr).split())[:300]

    def submit(self, gpus, minutes):
        args = ["sbatch", "--parsable", "-o", str(ST / "job_%j.out"), f"--exclude={EXCLUDE}",
                f"-t{minutes}", "--wrap", f"sleep {minutes * 60}"]
        if gpus:
            args[1:1] = ["-p", "defq", "-J", "mkpipe_gpu", "--constraint=MARKHAM&GFX942",
                         f"--gres=gpu:gfx942-mi300x:{gpus}", "--cpus-per-task=32", "--mem=200G"]
        else:
            args[1:1] = ["-p", "am", "-J", "mkpipe_cpu", "--constraint=MARKHAM",
                         "--cpus-per-task=32", "--mem=96G"]
        r = subprocess.run(args, capture_output=True, text=True, check=True)
        return r.stdout.strip().split(";")[0]

    def cancel(self, job):
        subprocess.run(["scancel", job], check=False)


def tmin(s):
    """squeue time (local, ISO) -> minutes; None for N/A."""
    try:
        return dt.datetime.fromisoformat(s).timestamp() / 60
    except ValueError:
        return None


def limlen(s):
    m = re.fullmatch(r"(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)", s)
    if not m:
        return 10 ** 6
    d, h, mi, sec = (int(x or 0) for x in m.groups())
    return d * 1440 + h * 60 + mi + sec / 60


class FakeSlurm:
    """Dry run: jobs in state/fake_jobs.json, id -> dict(gpus, start_at, limit[, bad,
    hvco_done_after, cancelled]) in fake minutes (now())."""

    def _jobs(self):
        return load("fake_jobs.json", {})

    def query(self, ids):
        jobs, t, out = self._jobs(), now(), {}
        for i in ids:
            j = jobs.get(i)
            if not j or j.get("cancelled") or t >= j["start_at"] + j["limit"]:
                continue
            run = t >= j["start_at"]
            if run and "hvco_done_after" in j and t >= j["start_at"] + j["hvco_done_after"]:
                (ST / f"hvco_done_{i}").touch()
            out[i] = dict(state="RUNNING" if run else "PENDING", start=j["start_at"],
                          end=j["start_at"] + j["limit"], limit=j["limit"], node=f"fake{i}",
                          name=j.get("name", "fake"))
        return out

    def probe(self, job, gpus):
        bad = self._jobs()[job].get("bad")
        return (not bad), ("fake node, AUSTIN home" if bad else "fake node HOME_OK gfx942")

    def submit(self, gpus, minutes):
        jobs = self._jobs()
        i = str(90000000 + len(jobs))
        jobs[i] = dict(gpus=gpus, start_at=now() + 5, limit=minutes, name="mkpipe")
        save("fake_jobs.json", jobs)
        return i

    def cancel(self, job):
        jobs = self._jobs()
        jobs[job]["cancelled"] = True
        save("fake_jobs.json", jobs)


class Driver:
    def __init__(self):
        self.sl = FakeSlurm() if DRY else Slurm()
        self.jobs = load("jobs.json", None)
        if self.jobs is None:
            self.jobs = {i: dict(gpus=g, kind=k, probe=None, tries=0, released=False, gone=False)
                         for i, (g, k) in INITIAL_JOBS.items()}
        self.st = load("stages.json", {})
        for name in SPEC:
            self.st.setdefault(name, dict(state="WAIT", attempts=0))
        self.procs = {}  # name -> Popen of stages this process started
        self.q = {}

    # ---- stages
    def alive(self, name, s):
        p = self.procs.get(name)
        if p is not None:
            return p.poll() is None
        try:
            return f"stage.sh\0{name}\0" in pathlib.Path(f"/proc/{s['pid']}/cmdline").read_text()
        except (OSError, KeyError):
            return False

    def reap(self):
        for name, s in self.st.items():
            if s["state"] != "RUN" or self.alive(name, s):
                continue
            self.procs.pop(name, None)
            rcf = ST / f"{name}.rc"
            rc = int(rcf.read_text().strip() or 99) if rcf.exists() else None
            res = {0: "PASS", 1: "FAIL"}.get(rc, "ERROR")
            log(f"stage {name} ended on job {s.get('job')}: rc {rc} -> {res}")
            if res == "ERROR" and s["attempts"] < MAX_ATTEMPTS:
                s.update(state="WAIT")
                log(f"stage {name}: retry ({s['attempts']}/{MAX_ATTEMPTS} attempts used)")
            else:
                s.update(state=res, end=time.time())

    def propagate(self):
        changed = True
        while changed:
            changed = False
            for name, sp in SPEC.items():
                s = self.st[name]
                if s["state"] != "WAIT" or sp["soft"]:
                    continue
                bad = [d for d in sp["deps"] if self.st[d]["state"] in ("ERROR", "SKIP")]
                if bad:
                    s.update(state="SKIP", why=f"dependency {bad[0]} {self.st[bad[0]]['state']}")
                    log(f"stage {name}: SKIP ({s['why']})")
                    changed = True

    def ready(self, name):
        return all(self.st[d]["state"] in TERMINAL for d in SPEC[name]["deps"])

    def pfinish(self, name):
        s, sp = self.st[name], SPEC[name]
        if s["state"] in TERMINAL:
            return now()
        if s["state"] == "RUN":
            return max(now(), s["t0"] / SCALE + sp["est"])
        return self.pready(name) + sp["est"]

    def pready(self, name):
        return max([now()] + [self.pfinish(d) for d in SPEC[name]["deps"]])

    def launch(self, name, job, slots):
        s, sp = self.st[name], SPEC[name]
        s["attempts"] += 1
        tleft = int(self.q[job]["end"] - now() - MARGIN / 2) if job != "-" else 0
        (ST / f"{name}.rc").unlink(missing_ok=True)
        env = dict(os.environ, PIPE_DIR=str(P), PIPE_OUT=str(O), STAGE_EST=str(sp["est"]),
                   PIPE_SCALE=str(SCALE), PIPE_DRY="1" if DRY else "0", STAGE_ATTEMPT=str(s["attempts"]))
        f = open(O / f"{name}.log", "a")
        f.write(f"==== {dt.datetime.now().isoformat(timespec='seconds')} {name} attempt "
                f"{s['attempts']} job {job} gpus {','.join(slots) or '-'} tleft {tleft} min\n")
        f.flush()
        p = subprocess.Popen(["bash", str(P / "stage.sh"), name, job, ",".join(slots) or "-",
                              str(tleft)], stdout=f, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, env=env, start_new_session=True)
        f.close()
        self.procs[name] = p
        s.update(state="RUN", job=job, slots=slots, pid=p.pid, t0=time.time())
        log(f"stage {name} started on job {job} gpus {slots or '-'} (estimate {sp['est']} min,"
            f" {tleft} min left), pid {p.pid}")

    # ---- jobs
    def usable(self):
        return [i for i, j in self.jobs.items() if not j["gone"] and not j["released"]
                and j["probe"] == "ok" and self.q.get(i, {}).get("state") == "RUNNING"]

    def busy(self, job):
        return [n for n, s in self.st.items() if s["state"] == "RUN" and s.get("job") == job]

    def free_slots(self, job):
        g = self.jobs[job]["gpus"]
        used = {x for n in self.busy(job) for x in self.st[n]["slots"]}
        return [str(i) for i in range(g) if str(i) not in used]

    def fits(self, name, job, start=None):
        """name's estimate fits on running job before its end (from start, default now)."""
        sp, j, q = SPEC[name], self.jobs[job], self.q[job]
        if sp["gpus"] > j["gpus"]:
            return False
        return q["end"] - max(now(), start or now()) - MARGIN >= sp["est"]

    def place(self, name):
        """A running job (and GPU slots) that can take name now, best fit; None."""
        sp, best = SPEC[name], None
        for job in self.usable():
            if not self.fits(name, job):
                continue
            busy = self.busy(job)
            if any(SPEC[n]["gpus"] == 0 for n in busy):
                continue  # a CPU stage (build) holds the whole job
            if sp["gpus"] == 0:
                if busy:
                    continue
                key = (self.jobs[job]["gpus"], self.q[job]["end"])  # CPU jobs first
                slots = []
            else:
                free = self.free_slots(job)
                if len(free) < sp["gpus"]:
                    continue
                key = (self.jobs[job]["gpus"] - sp["gpus"], self.q[job]["end"])
                slots = free[:sp["gpus"]]
            if best is None or key < best[0]:
                best = (key, job, slots)
        return best and best[1:]

    def touch(self, job, gpus):
        return gpus == 0 or self.jobs[job]["gpus"] >= gpus

    def release(self, job, why):
        j = self.jobs[job]
        name = self.q.get(job, {}).get("name", "")
        if job in NEVER or name.startswith("sdpabwd") or job not in self.jobs:
            log(f"refusing to release job {job} ({name})")
            return
        if j["kind"] == "hvco":
            (ST / f"release_{job}").touch()  # hvco_run.sh's hold loop ends
        else:
            self.sl.cancel(job)
        j["released"] = True
        log(f"job {job} released: {why}")

    def refresh_jobs(self):
        ids = [i for i, j in self.jobs.items() if not j["gone"]]
        self.q = self.sl.query(ids)
        for i in ids:
            j = self.jobs[i]
            if i not in self.q:
                j["gone"] = True
                log(f"job {i} ended")
                continue
            if self.q[i]["state"] != "RUNNING" or j["probe"] is not None or j["released"]:
                continue
            if j["kind"] == "hvco" and not (ST / f"hvco_done_{i}").exists():
                continue  # its own tests still run on the GPU
            ok, info = self.sl.probe(i, j["gpus"])
            j["tries"] += 1
            log(f"job {i} probe {'ok' if ok else 'failed'} ({info})")
            if ok:
                j["probe"] = "ok"
            elif j["tries"] >= PROBE_TRIES:
                j["probe"] = "bad"
                self.release(i, "unusable node (no shared home or no gfx942)")

    def alive_jobs(self):
        return [i for i, j in self.jobs.items() if not j["gone"] and not j["released"]]

    def could_hold(self, job, name, start):
        """job (alive) can run name, from start, now or once it starts."""
        sp, j, q = SPEC[name], self.jobs[job], self.q.get(job, {})
        if not self.touch(job, sp["gpus"]) or j["probe"] == "bad":
            return False
        if q.get("state") == "RUNNING" and j["probe"] == "ok":
            return self.fits(name, job, start)
        return q.get("limit", 0) - MARGIN >= sp["est"]  # pending, or not probed yet

    def maybe_submit(self):
        alive = self.alive_jobs()
        if len(alive) >= MAX_JOBS:
            return
        for name, sp in SPEC.items():
            if self.st[name]["state"] != "WAIT" or sp["gpus"] is None:
                continue
            if sp["gpus"] == 0:
                # a ready CPU stage that no idle job takes now: a CPU job, unless one is coming
                if not self.ready(name) or self.place(name):
                    continue
                if any(self.jobs[i]["gpus"] == 0 and self.jobs[i]["probe"] is None for i in alive):
                    continue
                gpus, minutes = 0, sp["est"] + 60
            else:
                start = self.pready(name)
                if any(self.could_hold(i, name, start) for i in alive):
                    continue
                gpus, minutes = 2, max(360, sp["est"] + 60)
            job = self.sl.submit(gpus, minutes)
            self.jobs[job] = dict(gpus=gpus, kind="holder" if gpus else "cpu", probe=None,
                                  tries=0, released=False, gone=False, by="driver")
            log(f"submitted job {job} ({gpus} GPUs, {minutes} min) for stage {name}")
            return

    def maybe_release(self, done):
        for job in self.alive_jobs():
            state = self.q.get(job, {}).get("state")
            if done:
                if state == "RUNNING" or self.jobs[job]["kind"] != "hvco":
                    self.release(job, "all stages ended")
                continue
            if state != "RUNNING" or self.jobs[job]["probe"] != "ok" or self.busy(job):
                continue
            need = [n for n, sp in SPEC.items() if self.st[n]["state"] == "WAIT"
                    and sp["gpus"] is not None and self.fits(n, job, self.pready(n))]
            if not need:
                self.release(job, f"no remaining stage fits ({self.q[job]['end'] - now():.0f} min left)")

    # ---- loop
    def step(self):
        (ST / "driver.alive").touch()
        self.refresh_jobs()
        self.reap()
        self.propagate()
        for name, sp in SPEC.items():
            if self.st[name]["state"] != "WAIT" or not self.ready(name):
                continue
            if sp["gpus"] is None:
                self.launch(name, "-", [])
                continue
            where = self.place(name)
            if where:
                self.launch(name, *where)
        self.maybe_submit()
        done = all(s["state"] in TERMINAL for s in self.st.values())
        self.maybe_release(done)
        save("stages.json", self.st)
        save("jobs.json", self.jobs)
        self.summary()
        return done

    def summary(self):
        lines = [f"# pipeline summary {dt.datetime.now().isoformat(timespec='seconds')}"]
        for name, sp in SPEC.items():
            s = self.st[name]
            lines.append(f"{name:12} {s['state']:5} job {s.get('job', '-'):9} attempts "
                         f"{s['attempts']} {s.get('why', '')}")
        lines.append("# jobs")
        for i, j in self.jobs.items():
            q = self.q.get(i, {})
            lines.append(f"{i} {j['kind']:6} gpus {j['gpus']} {q.get('state', 'ENDED'):8} "
                         f"node {q.get('node', '-')} probe {j['probe']} released {j['released']}")
        lines.append("# verdict lines (PASS/FAIL) of each stage log")
        for name in SPEC:
            f = O / f"{name}.log"
            if f.exists():
                lines += [f"{name}: {x}" for x in f.read_text(errors="replace").splitlines()
                          if re.match(r"(PASS|FAIL|RESULT) ", x)]
        (O / "summary.txt").write_text("\n".join(lines) + "\n")

    def run(self):
        log(f"driver start: dry={DRY} scale={SCALE}s/min poll={POLL}s dir={P} out={O}")
        for name, s in self.st.items():
            if s["state"] == "RUN":
                log(f"stage {name}: {'adopted, still running' if self.alive(name, s) else 'ended while the driver was down'}")
        while not self.step():
            time.sleep(POLL)
        (ST / "ALL_DONE").touch()
        log("all stages ended; summary in " + str(O / "summary.txt"))


def main():
    ST.mkdir(parents=True, exist_ok=True)
    lock = open(ST / "driver.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("another driver holds " + str(ST / "driver.lock"))
    if (ST / "ALL_DONE").exists():
        log("ALL_DONE exists: nothing to do")
        return
    Driver().run()


if __name__ == "__main__":
    main()
