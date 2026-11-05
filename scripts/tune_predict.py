#!/usr/bin/env python3
"""T1 (docs/GENERATION_PLAN.md §13.42, §13.43): tune the settings of CyphaLM's prediction stack (context8k) on 1 MiB of
enwik8's training part, and keep the best. Odin Loch, 2026-11-05 (started), 2026-11-06 (resumed, §13.45).

The slices. context8k reads enwik8[0, --sample), trains on [0, --train) plus the Dolly chat, and scores the second half
of its holdout [--train, --sample). The standard holdout is [7340032, 8388608) (the defaults 7 MiB and 8 MiB). The two
tuning slices both end before it, so no byte of the standard holdout is read while tuning:
  small  --train 1048576 --sample 1179648 (the small setup): 1 MiB to train, holdout enwik8[1 MiB, 1.125 MiB), scored
         on [1.0625 MiB, 1.125 MiB). Every move is screened here (about 10 min a run).
  big    --train 6291456 --sample 7340032: the full setup one MiB earlier, holdout enwik8[6 MiB, 7 MiB), scored on its
         second half. Only the best few candidates are confirmed here, one run at a time (about an hour a run).
The score is context8k_bpc on the slice: the meta mix's headline arm (M1Gs) on the slice holdout's second half, after
learning on its first half, as the full run scores it.

Subcommands (python3 -I, in WSL):
  settings                     every flagged setting: default, kind (continuous or structural), how it is tuned
  dump    --slice S --flags J  one context8k run of flags J (a JSON list) on slice S that also writes the experts'
                               outputs (--full-mix-dump) for the replays
  expert  --start J            coordinate search over the experts' flags on the small slice: a ladder of values per
                               setting; each round tries one step either way, keeps the best single move or the
                               combination of improving moves, then steps further only where a move improved
  replay  --dump F             coordinate (pattern) search with random restarts over --full-mix-params and
                               --meta-mix-params, replaying the experts' outputs dumped once (meta-mix-replay)
  points  --dump F --points J  the replay score of given mixer points (e.g. small-slice optima on the 6 MiB dump)
  run     --slice S --flags J  one context8k run of flags J (a confirmation)
  sens    ...                  sensitivity curves (§13.43): bpc at theta* x {0.5, 0.7, 0.85, 1.15, 1.4, 2}, others at
                               theta*; mixers by replay, experts by small runs
  table                        writes bench/tune/t1_sensitivity.tsv from the logs
  pair    --flags J            the final pair: the tuned set and today's defaults, full default context8k on the
                               standard holdout, started together inside one turn of ~/.cypha_full.lock
  best                         the best trial of each log
Every trial goes to a TSV under T1_OUT (settings, bpc, seconds, peak RSS). A trial already logged is not run again, so a
stopped search resumes where it was.

Machine limits (§13.45): at most T1_SLOTS (2) runs at once, replays included; none starts while MemAvailable is under
10 GB; every full-size context8k (the 6 MiB slice and the final pair) holds the shared lock ~/.cypha_full.lock.

Environment: T1_BIN (default ~/t1r/bin/cyphalm_trie), T1_OUT (default /mnt/d/cypha_llm/tune/t1), T1_SLOTS (2).
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HOME = os.path.expanduser("~")
BIN = os.environ.get("T1_BIN", os.path.join(HOME, "t1r/bin/cyphalm_trie"))
OUT = os.environ.get("T1_OUT", "/mnt/d/cypha_llm/tune/t1")
DATA = os.path.join(HOME, "t1r_data")  # the dumps, on the WSL disk (replays read them fast there)
ENWIK8 = "/mnt/d/cypha_llm/enwik8"
DOLLY = "/mnt/d/cypha_llm/corpus/labelled/dolly/databricks-dolly-15k.jsonl"
SLICES = {"small": ["--train", "1048576", "--sample", "1179648"], "big": ["--train", "6291456", "--sample", "7340032"],
          "full": []}
PREFIX = {"small": "SM__", "big": "X__", "full": "F__"}
HOLDOUT = (7340032, 8388608)  # context8k's standard holdout (packed_trie.hpp Config: train 7 MiB, sample 8 MiB)
for _k in ("small", "big"):
    assert int(SLICES[_k][3]) <= HOLDOUT[0], "a tuning slice must end before the standard holdout"
MULTS = [0.5, 0.7, 0.85, 1.15, 1.4, 2.0]  # §13.43's sensitivity multipliers
TOL = 0.0001  # bpc: context8k and the replay print 4 decimals; a move must lower the score by at least this
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------------------------------------ the settings
# kind: "cont" (continuous: could change while the model runs, §13.43) or "struct" (table bits, orders, sizes, parts,
# switches). how: "run" (a context8k run on the small slice a trial), "replay" (meta-mix-replay over a dump), or
# "none" (not tuned, and why).
SETTINGS = [
    # name, flag, default, kind, how, note
    ("ngram-cap", "--ngram-cap", "60", "cont", "run", "n-gram count limit (counts halve past it)"),
    ("ngram-lr", "--ngram-lr", "0.005", "cont", "run", "n-gram order mixer's step"),
    ("sim T", "--sim default,T=", "0.05", "cont", "run", "similar-context softmax temperature (qa reads it too)"),
    ("ngram-orders", "--ngram-orders", "1,2,3,4,5,6,8,12", "struct", "run", "online n-gram orders"),
    ("lstm-window", "--lstm-window", "20", "struct", "run", "LSTM BPTT window (one Adam step a window)"),
    ("bkn", "--bkn", "6:4:2:1", "struct", "run", "byte Kneser-Ney order:prune-from:min-context:drop-singletons (qa too)"),
    ("cls", "--cls", "shape (= shape:6:3:cp2)", "struct", "run", "class expert tail:orders:storage (qa too)"),
    ("doc", "--doc", "cache", "struct", "run", "document experts' parts and the cache's word settings (cand, stop)"),
    ("ngram-tb", "--ngram-tb", "24", "struct", "none", "table bits: 23 lost (ONLINE_LEARNING_UPGRADES O1); 25 adds ~590 MB (+38%), over the +20% RAM limit"),
    ("lstm-size", "--lstm-size", "256", "struct", "none", "64, 128 or 256 only; 256 is the largest built, smaller ones lost (L4, C-L1)"),
    ("sim k", "--sim default,k=", "8192", "struct", "none", "neighbours a query; the similar-context loop is the row loop's critical path"),
    ("copy-min", "--copy-min", "8", "struct", "none", "clips only the frozen copy lambdas answers read; the online chain ignores it (content_mixer.hpp online_mix), so context8k's bpc cannot move"),
    ("final-gate", "--final-gate", "copy", "struct", "none", "ig or lstm turn the full and meta mix off"),
    ("two-rates", "--two-rates/--one-rate", "on", "struct", "none", "a switch (§13.21 measured it), not a setting"),
    ("lr1", "--full-mix-params lr1=", "0.001", "cont", "replay", "bitwise mixers' first-layer step (and the 2-input mixers')"),
    ("lr2", "--full-mix-params lr2=", "0.0005", "cont", "replay", "bitwise mixers' second-layer step"),
    ("lrb", "--full-mix-params lrb=", "0.0002", "cont", "replay", "geometric mixer's step"),
    ("apm", "--full-mix-params apm=", "0.03", "cont", "replay", "SSE (APM) rate"),
    ("eps", "--full-mix-params eps=", "1e-06", "cont", "replay", "stretch clamp"),
    ("floor", "--full-mix-params floor=", "1e-06", "cont", "replay", "geometric mixer's distribution floor"),
    ("decay", "--full-mix-params decay=", "0", "cont", "replay", "step decay per weight set (tried 3e-5, 1e-4)"),
    ("ssef", "--full-mix-params ssef=", "0", "struct", "replay", "SSE output: 0 fixed average, 1 learned"),
    ("a2bits", "--full-mix-params a2bits=", "18", "struct", "replay", "order-2 APM contexts (2^a2bits)"),
    ("m1lr1", "--meta-mix-params m1lr1=", "0.002", "cont", "replay", "M1 first-layer step"),
    ("m1lr2", "--meta-mix-params m1lr2=", "-1 (= lr2, 0.0005)", "cont", "replay", "M1 second-layer step"),
    ("m2eta", "--meta-mix-params m2eta=", "1", "cont", "replay", "M2A master's exponent"),
    ("m2share", "--meta-mix-params m2share=", "0.001", "cont", "replay", "M2A master's share"),
    ("m2geta", "--meta-mix-params m2geta=", "1", "cont", "replay", "M2G master's exponent"),
    ("m2gshare", "--meta-mix-params m2gshare=", "0.001", "cont", "replay", "M2G master's share"),
    ("l2", "--meta-mix-params l2=", "1", "struct", "replay", "M1 second layer by bit position (0) or partial byte (1)"),
    ("m1r", "--meta-mix-params m1r=", "1", "struct", "replay", "M1 at each rate scale (1) or one (0)"),
    ("scales", "--meta-mix-params scales=", "0.25:0.5:1:2:4", "struct", "replay", "M2 rate multiples"),
    ("m2m", "--meta-mix-params m2m=", "0", "struct", "replay", "M2A master: fixed share (0) or logistic (1)"),
    ("m3a, m3eta, m3l*", "--meta-mix-params", "1e-4, 1, ...", "cont", "none", "M3 and M3L only: not in the headline M1Gs"),
    ("sel", "--meta-mix-params sel=", "0:...:13", "struct", "none", "M1's selectors; META_MIXING found each adds"),
    ("headline", "--meta-mix-params headline=", "M1Gs", "struct", "none", "the arm context8k_bpc reads (every arm is logged)"),
    ("chain rates", "(no flag)", "eta 0.02, slow/fast 0.005/0.08", "cont", "none", "no flag (content_mixer.hpp); T1 changes no code"),
    ("LSTM rate", "(no flag)", "0.003", "cont", "none", "no flag (ByteLSTMT's constructor in cmd_context8k)"),
    ("trie Config", "(no flag)", "alpha, eta, ...", "cont", "none", "no flag"),
]

# ------------------------------------------------------------------------------------------------ logging


def tsv_append(path, row, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, os.SEEK_END)
        if f.tell() == 0:
            f.write("\t".join(header) + "\n")
        f.write("\t".join(str(row.get(h, "")) for h in header) + "\n")
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def tsv_read(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        lines = f.read().splitlines()
    if not lines:
        return []
    h = lines[0].split("\t")
    return [dict(zip(h, ln.split("\t"))) for ln in lines[1:] if ln.strip()]


def log(msg):
    print(time.strftime("%m-%d %H:%M:%S"), msg, flush=True)


# The PC is shared: at most T1_SLOTS of T1's runs at once (replays and slice runs together), none started while less
# than 10 GB is available, and every full-size context8k holds the PC's shared lock. The slot locks are on the WSL disk
# (flock does not hold on /mnt/d), shared by every driver process.
LOCKS = os.path.join(HOME, "t1r/locks")
FULL_LOCK = os.path.join(HOME, ".cypha_full.lock")
SLOTS = int(os.environ.get("T1_SLOTS", "2"))
MIN_AVAIL_GB = 10.0


def mem_avail_gb():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable:"):
            return int(ln.split()[1]) / 1048576.0
    return 0.0


class Slot:
    """One of T1's small-run slots (small context8k runs and replays: at most T1_SLOTS at once). With full=True, the
    PC's shared lock for full-size context8k runs instead: one such run on the PC at a time, whoever runs it, so it
    takes no small slot (and never holds the lock while waiting for one)."""

    def __init__(self, full=False, min_gb=MIN_AVAIL_GB):
        self.full, self.ff, self.f, self.min_gb = full, None, None, min_gb

    def __enter__(self):
        os.makedirs(LOCKS, exist_ok=True)
        if self.full:
            self.ff = open(FULL_LOCK, "a")
            log("waiting for %s" % FULL_LOCK)
            fcntl.flock(self.ff, fcntl.LOCK_EX)
            log("holding %s" % FULL_LOCK)
            while mem_avail_gb() < self.min_gb:  # the lock may come while others' runs still free memory
                time.sleep(15)
            return self
        while self.f is None:
            if mem_avail_gb() >= self.min_gb:
                for i in range(SLOTS):
                    f = open(os.path.join(LOCKS, "slot%d.lock" % i), "w")
                    try:
                        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        self.f = f
                        break
                    except BlockingIOError:
                        f.close()
            if self.f is None:
                time.sleep(15)
        return self

    def __exit__(self, *exc):
        for f in (self.ff, self.f):
            if f:
                fcntl.flock(f, fcntl.LOCK_UN)
                f.close()
        return False


def parse_time(path):
    wall, rss, user = 0.0, 0, 0.0
    try:
        for ln in open(path):
            if "Maximum resident set size" in ln:
                rss = int(ln.split(":")[-1])
            elif "Elapsed (wall clock)" in ln:
                t = ln.split(": ")[-1].strip().split(":")
                wall = sum(float(x) * 60 ** i for i, x in enumerate(reversed(t)))
            elif "User time" in ln:
                user = float(ln.split(":")[-1])
    except OSError:
        pass
    return wall, rss, user


ARM = re.compile(r"^(full|meta)_mix_arm\s+(\S+)\s+first_half\s+([\d.]+)\s+second_half\s+([\d.]+)")


def parse_arms(text):
    arms = {}
    for ln in text.splitlines():
        m = ARM.match(ln)
        if m:
            arms[m.group(2)] = (float(m.group(3)), float(m.group(4)))
    return arms


# ------------------------------------------------------------------------------------------------ replay trials

RLOG = os.path.join(OUT, "replay_trials.tsv")
RHEAD = ["trial", "phase", "dump", "fparams", "mparams", "score", "first_half", "headline", "best_arm", "best_arm_bpc",
         "ABs", "seconds", "peak_rss_mb", "status"]
# The coordinates the replay search moves: (name, group f|m, default, "log": multiplicative steps).
RCOORD = [("lr1", "f", 0.001, "log"), ("lr2", "f", 0.0005, "log"), ("lrb", "f", 0.0002, "log"), ("apm", "f", 0.03, "log"),
          ("eps", "f", 1e-6, "log"), ("floor", "f", 1e-6, "log"),
          ("m1lr1", "m", 0.002, "log"), ("m1lr2", "m", 0.0005, "log"), ("m2eta", "m", 1.0, "log"),
          ("m2share", "m", 1e-3, "log"), ("m2geta", "m", 1.0, "log"), ("m2gshare", "m", 1e-3, "log")]
# The categorical ones: (name, group, default, alternatives), tried in the first round of the main search.
RCAT = [("decay", "f", 0, [3e-5, 1e-4]), ("ssef", "f", 0, [1]), ("a2bits", "f", 18, [17, 19]), ("l2", "m", 1, [0]),
        ("m1r", "m", 1, [0]), ("m2m", "m", 0, [1]),
        ("scales", "m", "0.25:0.5:1:2:4", ["0.5:1:2", "0.125:0.25:0.5:1:2:4:8", "0.25:0.5:1:2"])]
RDEFAULT = {c[0]: c[2] for c in RCOORD}
RDEFAULT.update({c[0]: c[2] for c in RCAT})
RGROUP = {c[0]: c[1] for c in RCOORD}
RGROUP.update({c[0]: c[1] for c in RCAT})


def fmtv(v):
    return ("%.6g" % v) if isinstance(v, float) else str(v)


def rparams(x):
    """The two params strings of a mixer point x (every searched key written out, so the code's defaults do not
    matter; at RDEFAULT they equal the defaults, m1lr2 0.0005 = lr2's default)."""
    f = ",".join("%s=%s" % (k, fmtv(x[k])) for k in sorted(x) if RGROUP[k] == "f")
    m = ",".join("%s=%s" % (k, fmtv(x[k])) for k in sorted(x) if RGROUP[k] == "m")
    return f, m


def rdiff(x):
    """Only the settings of x that differ from RDEFAULT, as context8k flags (empty at the defaults)."""
    def differs(k):  # m1lr2's code default (-1) follows lr2, the replays' 0.0005 does not: write it when lr2 moved
        return fmtv(x[k]) != fmtv(RDEFAULT[k]) or (k == "m1lr2" and fmtv(x["lr2"]) != fmtv(RDEFAULT["lr2"]))
    f = ",".join("%s=%s" % (k, fmtv(x[k])) for k in sorted(x) if RGROUP[k] == "f" and differs(k))
    m = ",".join("%s=%s" % (k, fmtv(x[k])) for k in sorted(x) if RGROUP[k] == "m" and differs(k))
    out = []
    if f:
        out += ["--full-mix-params", f]
    if m:
        out += ["--meta-mix-params", m]
    return out


class Replay:
    def __init__(self, dump, headline="M1Gs", phase=""):
        self.dump, self.headline, self.phase = dump, headline, phase
        self.cache = {}
        for r in tsv_read(RLOG):
            if r.get("status") == "ok" and r["dump"] == os.path.basename(dump):
                self.cache[(r["fparams"], r["mparams"])] = r

    def run(self, x):
        f, m = rparams(x)
        if (f, m) in self.cache:
            r = self.cache[(f, m)]
            return float(r["score"]), r
        tag = hashlib.sha1(("%s|%s|%s" % (os.path.basename(self.dump), f, m)).encode()).hexdigest()[:12]
        d = os.path.join(OUT, "replay")
        os.makedirs(d, exist_ok=True)
        base = os.path.join(d, tag)
        with Slot(min_gb=MIN_AVAIL_GB):
            t0 = time.time()
            with open(base + ".out", "w") as so, open(base + ".err", "w") as se:
                rc = subprocess.call(["/usr/bin/time", "-v", "-o", base + ".time", BIN, "meta-mix-replay", self.dump, f, m],
                                     stdout=so, stderr=se)
            sec = time.time() - t0
        _, rss, _ = parse_time(base + ".time")
        arms = parse_arms(open(base + ".out").read())
        ok = rc == 0 and self.headline in arms
        best = min(arms.items(), key=lambda kv: kv[1][1]) if arms else ("", (0, 0))
        row = {"trial": tag, "phase": self.phase, "dump": os.path.basename(self.dump), "fparams": f, "mparams": m,
               "score": "%.4f" % arms[self.headline][1] if ok else "", "first_half": "%.4f" % arms[self.headline][0] if ok else "",
               "headline": self.headline, "best_arm": best[0], "best_arm_bpc": "%.4f" % best[1][1],
               "ABs": "%.4f" % arms["ABs"][1] if "ABs" in arms else "", "seconds": "%.0f" % sec,
               "peak_rss_mb": "%.0f" % (rss / 1024), "status": "ok" if ok else "rc=%d" % rc}
        tsv_append(RLOG, row, RHEAD)
        log("replay %s %.0fs %s | %s | %s" % (row["score"], sec, row["best_arm"], f, m))
        if ok:
            self.cache[(f, m)] = row
            return arms[self.headline][1], row
        return math.inf, row

    def batch(self, xs, jobs):
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            return list(ex.map(self.run, xs))


def moves(x, step, coords):
    out = []
    for k in coords:
        for s in (step, 1.0 / step):
            y = dict(x)
            y[k] = float("%.6g" % (x[k] * s))
            out.append((k, s, y))
    return out


def pattern_search(rp, x, jobs, steps=(2.0, 1.41), rounds=6, cats=True, label=""):
    """Coordinate (pattern) search: every coordinate's up and down move from x is tried (in parallel); the improving
    moves are combined (the combination is kept if it beats the best single move, else the best single move is); then
    the next round. When no move improves, the step shrinks. Coordinates that move the score by less than TOL either
    way in the first round are dropped."""
    s0, _ = rp.batch([x], 1)[0]
    log("%s start %.4f" % (label, s0))
    active = [c[0] for c in RCOORD]
    first = True
    for step in steps:
        for rnd in range(rounds):
            cand = moves(x, step, active)
            if cats and first:
                for k, _, d, alts in RCAT:
                    for a in alts:
                        y = dict(x)
                        y[k] = a
                        cand.append((k, a, y))
            res = rp.batch([y for _, _, y in cand], jobs)
            best = {}
            for (k, s, y), (sc, _) in zip(cand, res):
                if sc < s0 - TOL / 2 and (k not in best or sc < best[k][0]):
                    best[k] = (sc, y[k])
            if first:  # drop coordinates that move nothing either way
                eff = {}
                for (k, s, y), (sc, _) in zip(cand, res):
                    eff[k] = max(eff.get(k, 0), abs(sc - s0))
                active = [k for k in active if eff.get(k, 0) >= TOL / 2]
                log("%s active coordinates: %s" % (label, ",".join(active)))
                first = False
            if not best:
                log("%s step %.2f: no move improves %.4f" % (label, step, s0))
                break
            k1 = min(best, key=lambda k: best[k][0])
            single = dict(x)
            single[k1] = best[k1][1]
            sc_c = math.inf
            if len(best) > 1:
                comb = dict(x)
                for k, (_, v) in best.items():
                    comb[k] = v
                sc_c, _ = rp.batch([comb], 1)[0]
            if sc_c < best[k1][0] - TOL / 2:
                x, s0 = comb, sc_c
                log("%s step %.2f round %d: combined %s -> %.4f" % (label, step, rnd, ",".join(sorted(best)), s0))
            else:
                x, s0 = single, best[k1][0]
                log("%s step %.2f round %d: %s=%s -> %.4f" % (label, step, rnd, k1, fmtv(best[k1][1]), s0))
    return x, s0


def cmd_replay(a):
    rp = Replay(a.dump, phase="main")
    x0 = dict(RDEFAULT)
    if a.start:
        x0.update(json.load(open(a.start))["x"] if os.path.exists(a.start) else json.loads(a.start))
    xb, sb = pattern_search(rp, x0, a.jobs, steps=tuple(a.steps), rounds=a.rounds, cats=not a.no_cats, label="main")
    results = [("main", xb, sb)]
    rng = random.Random(a.seed)
    for r in range(a.restarts):
        rp.phase = "restart%d" % (r + 1)
        xr = dict(xb if a.restart_from_best else RDEFAULT)
        for k, _, d, _ in RCOORD:
            xr[k] = float("%.6g" % (xr[k] * math.exp(rng.uniform(-math.log(a.spread), math.log(a.spread)))))
        x, s = pattern_search(rp, xr, a.jobs, steps=tuple(a.restart_steps), rounds=a.restart_rounds, cats=False,
                              label=rp.phase)
        results.append((rp.phase, x, s))
    results.sort(key=lambda t: t[2])
    for ph, x, s in results:
        log("RESULT %s %.4f %s" % (ph, s, " ".join(rdiff(x))))
    name = a.name or os.path.basename(a.dump).split(".")[0]
    with open(os.path.join(OUT, "replay_best_%s.json" % name), "w") as fo:
        json.dump({"phase": results[0][0], "score": results[0][2], "x": results[0][1], "flags": rdiff(results[0][1]),
                   "all": [{"phase": ph, "score": s, "x": x, "flags": rdiff(x)} for ph, x, s in results]}, fo, indent=1)


def cmd_points(a):
    """The replay score of given points (a JSON file: a list of {"name": .., "x": {param: value}}) on a dump, e.g. the
    small slice's optima on the 6 MiB slice's dump. Prints one RESULT line a point."""
    pts = json.load(open(a.points))
    rp = Replay(a.dump, phase=a.phase)
    xs = []
    for p in pts:
        x = dict(RDEFAULT)
        x.update(p.get("x", {}))
        xs.append(x)
    res = rp.batch(xs, a.jobs)
    for p, x, (sc, row) in zip(pts, xs, res):
        log("RESULT points %s %.4f first_half %s best_arm %s %s | %s" % (p["name"], sc, row.get("first_half"),
                                                                        row.get("best_arm"), row.get("best_arm_bpc"),
                                                                        " ".join(rdiff(x))))


# ------------------------------------------------------------------------------------------------ context8k trials

XLOG = os.path.join(OUT, "expert_trials.tsv")
XHEAD = ["trial", "slice", "phase", "flags", "score", "first_half", "ABs", "online_final", "ngram_alone", "lstm_alone",
         "seconds", "user_s", "peak_rss_mb", "status"]


def slug(flags, sl="small"):
    s = []
    i = 0
    while i < len(flags):
        k = flags[i].lstrip("-")
        v = flags[i + 1] if i + 1 < len(flags) and not flags[i + 1].startswith("--") else ""
        i += 2 if v else 1
        if k in ("full-mix-params", "meta-mix-params") or len(v) > 40:  # long: a short hash
            v = hashlib.sha1(v.encode()).hexdigest()[:8]
        s.append(k + ("=" + v.replace(",", ".").replace(":", "_").replace("=", "~") if v else ""))
    return PREFIX[sl] + ("__".join(s) if s else "default")


def run_ctx(flags, phase, sl="small", extra_flags=(), name=None, threads=None):
    """One context8k run on slice sl with flags; returns (score, row). Reuses a finished run of the same name; waits for
    one still running. extra_flags (a dump) do not change the output and are not part of the name."""
    name = name or slug(flags, sl)
    d = os.path.join(OUT, "runs")
    os.makedirs(d, exist_ok=True)
    base = os.path.join(d, name)
    for r in tsv_read(XLOG):
        if r["trial"] == name and r["status"] == "ok":
            return float(r["score"]), r

    def alive():  # a run of this name still going (by hand, or by another driver)
        return subprocess.call(["pgrep", "-f", base + ".time"], stdout=subprocess.DEVNULL) == 0
    while not os.path.exists(base + ".wall") and alive():
        time.sleep(30)
    if not os.path.exists(base + ".wall"):
        with Slot(full=sl != "small"):
            th = threads or ("2" if sl == "small" else "4")
            env = dict(os.environ, CYPHA_LSTM_THREADS=th, CYPHA_NGRAM_THREADS=th)
            cmd = [BIN, "context8k", ENWIK8, DOLLY] + SLICES[sl] + list(flags) + list(extra_flags)
            with open(base + ".cmd", "w") as fo:
                fo.write("CYPHA_LSTM_THREADS=%s CYPHA_NGRAM_THREADS=%s %s\n" % (th, th, " ".join(cmd)))
            log("start %s" % name)
            t0 = time.time()
            with open(base + ".out", "w") as so, open(base + ".err", "w") as se:
                rc = subprocess.call(["/usr/bin/time", "-v", "-o", base + ".time"] + cmd, stdout=so, stderr=se, env=env,
                                     cwd=os.path.join(HOME, "t1r"))
            with open(base + ".wall", "w") as fo:
                fo.write("%s rc=%d wall=%ds\n" % (name, rc, time.time() - t0))
    txt = open(base + ".out").read()
    wall, rss, user = parse_time(base + ".time")
    m = re.search(r"^context8k_bpc ([\d.]+)", txt, re.M)
    arms = parse_arms(txt)

    def line(k):
        mm = re.search(r"^%s ([\d.]+)" % k, txt, re.M)
        return mm.group(1) if mm else ""
    ok = m is not None and "rc=0" in open(base + ".wall").read()
    row = {"trial": name, "slice": sl, "phase": phase, "flags": " ".join(flags), "score": m.group(1) if m else "",
           "first_half": "%.4f" % arms["M1Gs"][0] if "M1Gs" in arms else "", "ABs": "%.4f" % arms["ABs"][1] if "ABs" in arms else "",
           "online_final": line("online_final_bpc"), "ngram_alone": line("ngram_alone_bpc"), "lstm_alone": line("lstm_alone_bpc"),
           "seconds": "%.0f" % wall, "user_s": "%.0f" % user, "peak_rss_mb": "%.0f" % (rss / 1024),
           "status": "ok" if ok else "failed"}
    tsv_append(XLOG, row, XHEAD)
    log("ctx %s %s %ss %sMB %s" % (sl, row["score"], row["seconds"], row["peak_rss_mb"], name))
    return (float(row["score"]) if ok else math.inf), row


# The experts' ladders: (setting, flag, values in order, default). Ordered ladders step one value either way; a
# categorical setting (ordered=False) tries every alternative in the first round and is then fixed.
XCOORD = [
    ("ngram-cap", "--ngram-cap", ["8", "15", "30", "60", "120", "240", "480"], "60", True),
    ("ngram-lr", "--ngram-lr", ["0.0003125", "0.000625", "0.00125", "0.0025", "0.005", "0.01", "0.02", "0.04"], "0.005", True),
    ("lstm-window", "--lstm-window", ["5", "8", "12", "20", "30", "50"], "20", True),
    ("sim T", "--sim", ["default,T=0.0125", "default,T=0.025", "default,T=0.035", "default,T=0.05", "default,T=0.07",
                        "default,T=0.1", "default,T=0.2"], "default,T=0.05", True),
    ("bkn", "--bkn", ["4:4:2:1", "5:4:2:1", "6:4:2:1", "8:4:2:1", "10:4:2:1"], "6:4:2:1", True),
    ("cls", "--cls", ["shape:6:2:cp2", "shape:6:3:cp2", "shape:6:4:cp2", "shape:6:5:cp2"], "shape:6:3:cp2", True),
    ("ngram-orders", "--ngram-orders", ["1,2,3,4,5,6,8,12", "1,2,3,4,5,6,7,8,10,12,16", "1,2,3,4,6,8,12,16"],
     "1,2,3,4,5,6,8,12", False),
    ("doc", "--doc", ["cache", "trig,cache", "cache,cand=2048", "cache,stop=100", "cache,stop=400"], "cache", False),
]
XDEFAULT = {c[1]: c[3] for c in XCOORD}


def xflags(x, mix=()):
    """The context8k flags of an expert point x ({flag: value}): only the values that differ from the defaults, sorted,
    then the mixer flags."""
    out = []
    for f in sorted(x):
        if x[f] != XDEFAULT.get(f):
            out += [f, x[f]]
    return out + list(mix)


def cmd_expert(a):
    """The ladder search. Round 1 from the start: every ordered setting one step either way, every categorical one's
    alternatives. A round keeps the best single move, or the combination of all improving moves when that is better.
    The next round steps further only along the settings whose move improved, from the point kept."""
    mix = json.loads(a.mix) if a.mix else []
    x = dict(XDEFAULT)
    if a.start:
        x.update(json.loads(a.start))
    only = set(a.only.split(";")) if a.only else None
    coords = [c for c in XCOORD if only is None or c[0] in only]
    s0, _ = run_ctx(xflags(x, mix), "start")
    log("expert start %.4f %s" % (s0, " ".join(xflags(x, mix))))
    dirs = {c[1]: (-1, 1) if c[4] else None for c in coords}  # the directions still open per ordered setting
    rnd = 0
    while True:
        rnd += 1
        cand = []
        for name, flag, vals, dflt, ordered in coords:
            if ordered:
                if dirs[flag] is None:
                    continue
                i = vals.index(x[flag])
                for dd in dirs[flag]:
                    if 0 <= i + dd < len(vals):
                        y = dict(x)
                        y[flag] = vals[i + dd]
                        cand.append((flag, dd, y))
            elif rnd == 1:
                for v in vals:
                    if v != x[flag]:
                        y = dict(x)
                        y[flag] = v
                        cand.append((flag, 0, y))
        if not cand:
            break
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            res = list(ex.map(lambda c: run_ctx(xflags(c[2], mix), "round%d" % rnd), cand))
        best = {}
        for (flag, dd, y), (sc, _) in zip(cand, res):
            if sc < s0 - TOL / 2 and (flag not in best or sc < best[flag][0]):
                best[flag] = (sc, y[flag], dd)
        log("expert round %d improving: %s" % (rnd, {f: (round(v[0], 4), v[1]) for f, v in best.items()}))
        for flag in list(dirs):  # an ordered setting stays open only in the direction that improved
            if dirs[flag] is not None:
                dirs[flag] = (best[flag][2],) if flag in best else None
        if not best:
            break
        f1 = min(best, key=lambda f: best[f][0])
        single = dict(x)
        single[f1] = best[f1][1]
        sc_c = math.inf
        if len(best) > 1:
            comb = dict(x)
            for f, (_, v, _) in best.items():
                comb[f] = v
            sc_c, _ = run_ctx(xflags(comb, mix), "round%d_comb" % rnd)
        if sc_c < best[f1][0] - TOL / 2:
            x, s0 = comb, sc_c
            log("expert round %d: combined %s -> %.4f" % (rnd, ",".join(sorted(best)), s0))
        else:  # the other improving moves are tried again (same direction) from the point kept, in the next round
            x, s0 = single, best[f1][0]
            log("expert round %d: %s=%s -> %.4f" % (rnd, f1, best[f1][1], s0))
    log("RESULT expert %.4f %s" % (s0, " ".join(xflags(x, mix))))
    with open(os.path.join(OUT, "expert_best_%s.json" % (a.name or "small")), "w") as fo:
        json.dump({"score": s0, "x": x, "flags": xflags(x, mix)}, fo, indent=1)


def cmd_dump(a):
    flags = json.loads(a.flags) if a.flags else []
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, a.name + ".fmx")
    sc, row = run_ctx(flags, "dump", a.slice, extra_flags=["--full-mix-dump", path])
    log("RESULT dump %s %.4f %s" % (path, sc, row["trial"]))


def cmd_run(a):
    flags = json.loads(a.flags) if a.flags else []
    sc, row = run_ctx(flags, a.phase, a.slice, name=a.name or None)
    log("RESULT run %s %.4f %s %ss %sMB" % (a.slice, sc, row["trial"], row["seconds"], row["peak_rss_mb"]))


# ------------------------------------------------------------------------------------------------ sensitivity

SLOG = os.path.join(OUT, "sensitivity.tsv")
SHEAD = ["setting", "how", "theta_star", "mult", "value", "bpc", "delta", "first_half", "seconds"]


def cmd_sens(a):
    """--dump F --mix-best J: the mixers' continuous settings by replay over the tuned experts' dump, every other mixer
    setting at theta* (J's point). --flags J --x J: the experts' continuous settings by small runs, every other flag at
    its tuned value (the tuned mixers included)."""
    done = {(r["setting"], r["mult"]) for r in tsv_read(SLOG)}
    if a.dump:
        rp = Replay(a.dump, phase="sens")
        xs = dict(RDEFAULT)
        if a.mix_best:
            xs.update(json.load(open(a.mix_best))["x"])
        s0, r0 = rp.run(xs)
        if ("(theta*) replay", "1") not in done:
            tsv_append(SLOG, {"setting": "(theta*) replay", "how": "replay", "theta_star": "", "mult": "1", "value": "",
                              "bpc": "%.4f" % s0, "delta": "+0.0000", "first_half": r0.get("first_half", ""),
                              "seconds": r0.get("seconds", "")}, SHEAD)
        pts = []
        for k, _, _, _ in RCOORD + [("decay", "f", 0, None)]:
            if k == "decay" and not xs.get("decay"):
                continue  # decay 0 has no multiples
            for m in MULTS:
                y = dict(xs)
                y[k] = float("%.6g" % (xs[k] * m))
                pts.append((k, m, y))
        res = rp.batch([y for _, _, y in pts], a.jobs)
        for (k, m, y), (sc, row) in zip(pts, res):
            if (k, "%g" % m) not in done:
                tsv_append(SLOG, {"setting": k, "how": "replay", "theta_star": fmtv(xs[k]), "mult": "%g" % m,
                                  "value": fmtv(y[k]), "bpc": "%.4f" % sc, "delta": "%+.4f" % (sc - s0),
                                  "first_half": row.get("first_half", ""), "seconds": row.get("seconds", "")}, SHEAD)
    if a.x:  # the experts' continuous settings: small runs
        mix = json.loads(a.mix) if a.mix else []
        x = dict(XDEFAULT)
        x.update(json.loads(a.x))
        sx, r0 = run_ctx(xflags(x, mix), "sens")
        if ("(theta*) run", "1") not in done:
            tsv_append(SLOG, {"setting": "(theta*) run", "how": "run", "theta_star": "", "mult": "1", "value": "",
                              "bpc": "%.4f" % sx, "delta": "+0.0000", "first_half": r0.get("first_half", ""),
                              "seconds": r0.get("seconds", "")}, SHEAD)
        jobs = []
        for key, flag, fmt in (("ngram-cap", "--ngram-cap", "int"), ("ngram-lr", "--ngram-lr", "g"), ("sim T", "--sim", "T")):
            if fmt == "T":
                th = float(re.search(r"T=([\d.e-]+)", x[flag]).group(1))
            else:
                th = float(x[flag])
            for m in MULTS:
                y = dict(x)
                v = str(int(round(th * m))) if fmt == "int" else "%.6g" % (th * m)
                y[flag] = "default,T=%s" % v if fmt == "T" else v
                jobs.append((key, th, m, y, v))
        with ThreadPoolExecutor(max_workers=a.jobs_expert) as ex:
            rx = list(ex.map(lambda j: run_ctx(xflags(j[3], mix), "sens"), jobs))
        for (key, th, m, y, v), (sc, row) in zip(jobs, rx):
            if (key, "%g" % m) not in done:
                tsv_append(SLOG, {"setting": key, "how": "run", "theta_star": "%g" % th, "mult": "%g" % m, "value": v,
                                  "bpc": "%.4f" % sc, "delta": "%+.4f" % (sc - sx), "first_half": row.get("first_half", ""),
                                  "seconds": row.get("seconds", "")}, SHEAD)


def cmd_table(a):
    """bench/tune/t1_sensitivity.tsv: one row per continuous setting and multiple (from the sensitivity log), then one
    row per structural or untuned setting (kind, how, note), so every setting is marked."""
    rows = tsv_read(SLOG)
    out = os.path.join(REPO, "bench/tune/t1_sensitivity.tsv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    kinds = {s[0]: s for s in SETTINGS}
    with open(out, "w") as f:
        f.write("setting\tkind\thow\ttheta_star\tmult\tvalue\tbpc\tdelta_vs_theta_star\tnote\n")
        for r in rows:
            if r["setting"].startswith("("):
                f.write("%s\t-\t%s\t\t1\t\t%s\t+0.0000\tthe tuned point itself (small slice, second half)\n" %
                        (r["setting"], r["how"], r["bpc"]))
        for r in rows:
            if not r["setting"].startswith("("):
                s = kinds.get(r["setting"])
                f.write("\t".join([r["setting"], "continuous", r["how"], r["theta_star"], r["mult"], r["value"], r["bpc"],
                                   r["delta"], s[5] if s else ""]) + "\n")
        for s in SETTINGS:
            if s[3] == "struct" or s[4] == "none":
                f.write("\t".join([s[0], "structural" if s[3] == "struct" else "continuous (not tuned)", s[4], s[2], "",
                                   "", "", "", s[5]]) + "\n")
    log("wrote %s (%d curve rows)" % (out, sum(1 for r in rows if not r["setting"].startswith("("))))


# ------------------------------------------------------------------------------------------------ the final pair


def cmd_pair(a):
    """The tuned set and today's defaults, full default context8k on the standard holdout, started together inside one
    turn of the PC's shared lock (as flock ~/.cypha_full.lock bash -c 'run_tuned & run_default & wait'), once at least
    12 GB is free."""
    flags = json.loads(a.flags)
    d = os.path.join(OUT, "final")
    os.makedirs(d, exist_ok=True)
    runs = [("tuned", flags), ("default", [])]
    with open(FULL_LOCK, "a") as ff:
        log("waiting for %s" % FULL_LOCK)
        fcntl.flock(ff, fcntl.LOCK_EX)
        while mem_avail_gb() < 12.0:
            time.sleep(15)
        log("holding %s; %.1f GB available" % (FULL_LOCK, mem_avail_gb()))
        env = dict(os.environ, CYPHA_LSTM_THREADS=a.threads, CYPHA_NGRAM_THREADS=a.threads)
        procs = []
        for name, fl in runs:
            base = os.path.join(d, name)
            cmd = [BIN, "context8k", ENWIK8, DOLLY] + fl
            with open(base + ".cmd", "w") as fo:
                fo.write("CYPHA_LSTM_THREADS=%s CYPHA_NGRAM_THREADS=%s %s\n" % (a.threads, a.threads, " ".join(cmd)))
            so, se = open(base + ".out", "w"), open(base + ".err", "w")
            procs.append((name, base, subprocess.Popen(["/usr/bin/time", "-v", "-o", base + ".time"] + cmd, stdout=so,
                                                       stderr=se, env=env, cwd=os.path.join(HOME, "t1r")), so, se, time.time()))
            log("start final %s: %s" % (name, " ".join(fl)))
        for name, base, p, so, se, t0 in procs:
            rc = p.wait()
            so.close()
            se.close()
            with open(base + ".wall", "w") as fo:
                fo.write("%s rc=%d wall=%ds\n" % (name, rc, time.time() - t0))
            log("final %s rc=%d" % (name, rc))
        fcntl.flock(ff, fcntl.LOCK_UN)
    for name, base, *_ in procs:
        txt = open(base + ".out").read()
        m = re.search(r"^context8k_bpc ([\d.]+)", txt, re.M)
        wall, rss, user = parse_time(base + ".time")
        log("RESULT final %s %s wall %.0fs user %.0fs rss %.0fMB" % (name, m.group(1) if m else "?", wall, user, rss / 1024))


# ------------------------------------------------------------------------------------------------ the rest


def cmd_settings(a):
    print("setting\tflag\tdefault\tkind\thow\tnote")
    for s in SETTINGS:
        print("\t".join(s))


def cmd_best(a):
    for path in (XLOG, RLOG):
        rows = [r for r in tsv_read(path) if r.get("status") == "ok" and r.get("score")]
        for key in sorted({r.get("slice", r.get("dump")) for r in rows}):
            rr = [r for r in rows if r.get("slice", r.get("dump")) == key]
            b = min(rr, key=lambda r: float(r["score"]))
            print(os.path.basename(path), key, len(rr), "trials; best", b["score"], b["trial"],
                  b.get("flags", "") or (b.get("fparams", "") + " | " + b.get("mparams", "")))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("settings")
    sp.add_parser("best")
    sp.add_parser("table")
    e = sp.add_parser("expert")
    e.add_argument("--jobs", type=int, default=2)
    e.add_argument("--start", default="", help="JSON dict {flag: value} of expert settings to start from")
    e.add_argument("--mix", default="", help="JSON list of mixer flags added to every run (the tuned mixers)")
    e.add_argument("--only", default="", help="settings to move, ';'-separated")
    e.add_argument("--name", default="")
    du = sp.add_parser("dump")
    du.add_argument("--slice", default="small", choices=sorted(SLICES))
    du.add_argument("--flags", default="")
    du.add_argument("--name", required=True)
    ru = sp.add_parser("run")
    ru.add_argument("--slice", default="small", choices=sorted(SLICES))
    ru.add_argument("--flags", default="")
    ru.add_argument("--phase", default="confirm")
    ru.add_argument("--name", default="")
    r = sp.add_parser("replay")
    r.add_argument("--dump", required=True)
    r.add_argument("--name", default="")
    r.add_argument("--jobs", type=int, default=2)
    r.add_argument("--steps", type=float, nargs="+", default=[2.0, 1.41])
    r.add_argument("--rounds", type=int, default=6)
    r.add_argument("--no-cats", action="store_true")
    r.add_argument("--restarts", type=int, default=2)
    r.add_argument("--restart-steps", type=float, nargs="+", default=[1.41])
    r.add_argument("--restart-rounds", type=int, default=4)
    r.add_argument("--spread", type=float, default=2.5, help="restarts: each rate x exp(U(-ln s, ln s))")
    r.add_argument("--restart-from-best", action="store_true")
    r.add_argument("--seed", type=int, default=20261106)
    r.add_argument("--start", default="", help="JSON dict of params, or a replay_best JSON file, to start from")
    s = sp.add_parser("sens")
    s.add_argument("--dump", default="")
    s.add_argument("--mix-best", default="")
    s.add_argument("--x", default="", help="JSON dict: the tuned expert point")
    s.add_argument("--mix", default="", help="JSON list: the tuned mixer flags")
    s.add_argument("--jobs", type=int, default=2)
    s.add_argument("--jobs-expert", type=int, default=2)
    q = sp.add_parser("points")
    q.add_argument("--dump", required=True)
    q.add_argument("--points", required=True)
    q.add_argument("--phase", default="confirm")
    q.add_argument("--jobs", type=int, default=2)
    pa = sp.add_parser("pair")
    pa.add_argument("--flags", required=True, help="JSON list: the tuned flags")
    pa.add_argument("--threads", default="4")
    a = p.parse_args()
    {"settings": cmd_settings, "best": cmd_best, "table": cmd_table, "expert": cmd_expert, "dump": cmd_dump,
     "run": cmd_run, "replay": cmd_replay, "sens": cmd_sens, "points": cmd_points, "pair": cmd_pair}[a.cmd](a)


if __name__ == "__main__":
    main()
