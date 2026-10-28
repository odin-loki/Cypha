"""Sequential job queue for the baseline runs (one heavy model at a time).

Jobs run in list order; a job whose input file does not exist yet is skipped
and retried after the others; the queue polls every 5 minutes until all done.
A job is done when its final output file exists. The job list is re-read from
jobs.txt on every pass, so jobs can be added while the queue runs.
jobs.txt lines:  <output file relative to R> | <command>
"""
import os, subprocess, sys, time
from pathlib import Path

SP = Path("/tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad")
R = SP / "run/baselines"
JOBS = SP / "bl/jobs.txt"
LOGS = R / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, HF_HOME=str(SP / "hf"), SCRIBE_BASELINES_ROOT=str(R), PYTHONUNBUFFERED="1")
failed = set()


def jobs():
    out = []
    for line in JOBS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        target, cmd = [x.strip() for x in line.split("|", 1)]
        out.append((target, cmd))
    return out


def log(m):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), m, flush=True)


while True:
    pending, ran = 0, False
    for target, cmd in jobs():
        if (R / target).exists() or target in failed:
            continue
        pending += 1
        inp = cmd.split()[cmd.split().index("INPUT") + 1] if "INPUT" in cmd.split() else None
        args = cmd.replace("INPUT ", "")
        if inp and not Path(inp).exists():
            continue
        # coordinator: nice 0 with 2 threads, 3 when the load is below 4
        threads = 3 if os.getloadavg()[0] < 4 else 2
        args += f" --threads {threads}"
        name = target.replace("/", "__").replace(".jsonl", "")
        log(f"start {target}: {args}")
        t0 = time.time()
        with open(LOGS / f"{name}.log", "a") as f:
            rc = subprocess.call(f"nice -n 0 {args}", shell=True, stdout=f, stderr=subprocess.STDOUT,
                                 env=env, cwd="/home/user/Cypha")
        log(f"end {target}: rc={rc} in {time.time() - t0:.0f}s")
        if rc != 0:
            failed.add(target)
        ran = True
        break  # re-read the job list after every job
    if not ran:
        if pending == 0:
            log("all jobs done" + (f"; failed: {sorted(failed)}" if failed else ""))
            break
        time.sleep(300)
