#!/usr/bin/env python3
"""T1 (docs/GENERATION_PLAN.md §13.42, §13.43): tune the settings of CyphaLM's prediction stack (context8k) on 1 MB of
enwik8's training part, and keep the best. Odin Loch, 2026-11-05.

The slices. context8k reads enwik8[0, --sample), trains on [0, --train) plus the Dolly chat, and scores the second half
of the holdout [--train, --sample). The standard holdout is [7340032, 8388608) (the defaults 7 MiB and 8 MiB). Two
tuning slices, both inside the standard training part, so no byte of the standard holdout is read:
  small  --train 1048576 --sample 1179648 (the small setup): 1 MiB to train, holdout enwik8[1 MiB, 1.125 MiB), scored
         on [1.0625 MiB, 1.125 MiB). It is the 1 MB slice: every move is screened here (10-15 min a run).
  big    --train 6291456 --sample 7340032: the full setup one MiB earlier, holdout enwik8[6 MiB, 7 MiB), scored on its
         second half. The best few candidates are confirmed here, one run at a time (about an hour a run).
The score is context8k_bpc on the slice: the meta mix's headline arm on the slice holdout's second half, after learning
on its first half, as the full run scores.

Subcommands (run with python3 -I, in WSL, from anywhere):
  settings                    print every flagged setting with its default, kind and how it is tuned
  expert  [--slice S]         coordinate search over the experts' flags: one context8k run on the slice a trial
  replay  --dump F [--jobs N] coordinate search with random restarts over --full-mix-params and --meta-mix-params,
                              replaying the experts' outputs dumped once (cyphalm_trie meta-mix-replay)
  points  --dump F --points J the replay score of given points (a confirmation on another slice's dump)
  sens    --dump F            sensitivity curves: bits per byte at theta* x {0.5, 0.7, 0.85, 1.15, 1.4, 2}
  best                        the best trial of each log
Every trial goes to a TSV (settings, bits per byte, seconds, peak RSS). A trial already in the TSV is not run again,
so a stopped search resumes where it was.

Environment: T1_BIN (cyphalm_trie, default ~/t1/bin/cyphalm_trie), T1_OUT (default /mnt/d/cypha_llm/tune/t1).
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

BIN = os.environ.get("T1_BIN", os.path.expanduser("~/t1/bin/cyphalm_trie"))
OUT = os.environ.get("T1_OUT", "/mnt/d/cypha_llm/tune/t1")
ENWIK8 = "/mnt/d/cypha_llm/enwik8"
DOLLY = "/mnt/d/cypha_llm/corpus/labelled/dolly/databricks-dolly-15k.jsonl"
SLICES = {"small": ["--train", "1048576", "--sample", "1179648"], "big": ["--train", "6291456", "--sample", "7340032"]}
PREFIX = {"small": "SM__", "big": "X__"}
HOLDOUT = (7340032, 8388608)  # context8k's standard holdout (packed_trie.hpp Config: train 7 MiB, sample 8 MiB)
for _s in SLICES.values():
    assert int(_s[3]) <= HOLDOUT[0], "a slice must end before the standard holdout"
MULTS = [0.5, 0.7, 0.85, 1.15, 1.4, 2.0]  # §13.43's sensitivity multipliers
TOL = 0.0001  # bits per byte: context8k and the replay print 4 decimals

# ------------------------------------------------------------------------------------------------ the settings
# kind: "cont" (continuous: can change while the model runs, §13.43) or "struct" (table bits, orders, sizes, switches).
# how: "run" (needs a full context8k run), "replay" (meta-mix-replay over a dump), "none" (not tuned: why).
SETTINGS = [
    # name, flag/where, default, kind, how, note
    ("ngram-orders", "--ngram-orders", "1,2,3,4,5,6,8,12", "struct", "run", "online n-gram orders"),
    ("ngram-tb", "--ngram-tb", "24", "struct", "none", "table bits; 25 adds about 590 MB (+38%), over the +20% RAM limit"),
    ("ngram-cap", "--ngram-cap", "60", "cont", "run", "n-gram count limit (counts halve past it)"),
    ("ngram-lr", "--ngram-lr", "0.005", "cont", "run", "n-gram order mixer's step"),
    ("lstm-size", "--lstm-size", "256", "struct", "none", "64, 128 or 256 only; 256 is the largest built"),
    ("lstm-window", "--lstm-window", "20", "struct", "run", "LSTM BPTT window (one Adam step a window)"),
    ("bkn", "--bkn", "6:4:2:1", "struct", "run", "byte Kneser-Ney K:prune-from-order:min-context:drop-singletons (qa too)"),
    ("cls", "--cls", "shape (shape:6:3:cp2)", "struct", "run", "class expert tail:orders:storage (qa too)"),
    ("sim T", "--sim default,T=", "0.05", "cont", "run", "similar-context softmax temperature (qa too)"),
    ("sim k", "--sim default,k=", "8192", "struct", "none", "neighbours a query; sim is the row loop's critical path"),
    ("doc", "--doc", "cache", "struct", "run", "document experts' parts (trig, cache, topic)"),
    ("copy-min", "--copy-min", "8", "struct", "none", "only clips the frozen copy lambdas answers read; no effect on context8k's bpc online"),
    ("final-gate", "--final-gate", "copy", "struct", "none", "ig turns the full and meta mix off"),
    ("two-rates", "--two-rates/--one-rate", "on", "struct", "none", "the full mix replays two rates; a switch, not a setting"),
    ("lr1", "--full-mix-params lr1=", "0.001", "cont", "replay", "bitwise mixers' first-layer step (and the 2-input mixers')"),
    ("lr2", "--full-mix-params lr2=", "0.0005", "cont", "replay", "bitwise mixers' second-layer step"),
    ("lrb", "--full-mix-params lrb=", "0.0002", "cont", "replay", "geometric mixer's step"),
    ("apm", "--full-mix-params apm=", "0.03", "cont", "replay", "SSE (APM) rate"),
    ("eps", "--full-mix-params eps=", "1e-6", "cont", "replay", "stretch clamp"),
    ("floor", "--full-mix-params floor=", "1e-6", "cont", "replay", "geometric mixer's distribution floor"),
    ("decay", "--full-mix-params decay=", "0", "cont", "replay", "step decay per weight set"),
    ("ssef", "--full-mix-params ssef=", "0", "struct", "replay", "SSE output: 0 fixed average, 1 learned"),
    ("a2bits", "--full-mix-params a2bits=", "18", "struct", "replay", "order-2 APM contexts (2^a2bits)"),
    ("m1lr1", "--meta-mix-params m1lr1=", "0.002", "cont", "replay", "M1 first-layer step"),
    ("m1lr2", "--meta-mix-params m1lr2=", "-1 (= lr2)", "cont", "replay", "M1 second-layer step"),
    ("l2", "--meta-mix-params l2=", "1", "struct", "replay", "M1 second layer by bit position (0) or partial byte (1)"),
    ("m1r", "--meta-mix-params m1r=", "1", "struct", "replay", "M1 at each rate scale (1) or one (0)"),
    ("scales", "--meta-mix-params scales=", "0.25:0.5:1:2:4", "struct", "replay", "M2 rate multiples"),
    ("m2m", "--meta-mix-params m2m=", "0", "struct", "replay", "M2A master: fixed share (0) or logistic (1)"),
    ("m2eta", "--meta-mix-params m2eta=", "1", "cont", "replay", "M2A master's exponent"),
    ("m2share", "--meta-mix-params m2share=", "1e-3", "cont", "replay", "M2A master's share"),
    ("m2geta", "--meta-mix-params m2geta=", "1", "cont", "replay", "M2G master's exponent"),
    ("m2gshare", "--meta-mix-params m2gshare=", "1e-3", "cont", "replay", "M2G master's share"),
    ("m3a", "--meta-mix-params m3a=", "1e-4", "cont", "none", "M3 only; not in the headline M1Gs"),
    ("m3eta", "--meta-mix-params m3eta=", "1", "cont", "none", "M3 only"),
    ("m3lctx/m3llr1/m3llr2/m3linit", "--meta-mix-params", "0/-1/-1/3", "struct", "none", "M3L only"),
    ("sel", "--meta-mix-params sel=", "0:...:13", "struct", "none", "M1's selectors; META_MIXING found each adds"),
    ("headline", "--meta-mix-params headline=", "M1Gs", "struct", "replay", "the arm context8k_bpc reads (every arm is printed)"),
    ("trie Config", "(none)", "alpha 0.4, eta 0.05, ...", "cont", "none", "no flag"),
    ("chain rates", "(none)", "eta 0.05, 0.005/0.08", "cont", "none", "no flag (content_mixer.hpp)"),
    ("LSTM rate", "(none)", "0.003", "cont", "none", "no flag (ByteLSTMT's constructor in cmd_context8k)"),
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
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# The PC is shared (about 10 agents): at most T1_SLOTS tuning runs at once (replays and slice runs together), at most
# one full-size context8k at a time, and none started while less than 10 GB is available. Locks on the WSL disk
# (flock does not hold on /mnt/d), shared by every driver process.
LOCKS = os.path.expanduser("~/t1/locks")
SLOTS = int(os.environ.get("T1_SLOTS", "3"))
MIN_AVAIL_GB = 10.0


def mem_avail_gb():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable:"):
            return int(ln.split()[1]) / 1048576.0
    return 0.0


class Slot:
    def __init__(self, full=False):
        self.full, self.ff, self.f = full, None, None

    def __enter__(self):
        os.makedirs(LOCKS, exist_ok=True)
        if self.full:
            self.ff = open(os.path.join(LOCKS, "full.lock"), "w")
            fcntl.flock(self.ff, fcntl.LOCK_EX)
        while True:
            if mem_avail_gb() >= MIN_AVAIL_GB:
                for i in range(SLOTS):
                    f = open(os.path.join(LOCKS, "slot%d.lock" % i), "w")
                    try:
                        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        self.f = f
                        return self
                    except BlockingIOError:
                        f.close()
            time.sleep(15)

    def __exit__(self, *exc):
        for f in (self.f, self.ff):
            if f:
                fcntl.flock(f, fcntl.LOCK_UN)
                f.close()
        return False


def parse_time(path):
    wall, rss = 0.0, 0
    try:
        for ln in open(path):
            if "Maximum resident set size" in ln:
                rss = int(ln.split(":")[-1])
            elif "Elapsed (wall clock)" in ln:
                t = ln.split(": ")[-1].strip().split(":")
                wall = sum(float(x) * 60 ** i for i, x in enumerate(reversed(t)))
    except OSError:
        pass
    return wall, rss


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
# The coordinates the replay search moves: (name, group f|m, default, kind). log: multiplicative steps.
RCOORD = [("lr1", "f", 0.001, "log"), ("lr2", "f", 0.0005, "log"), ("lrb", "f", 0.0002, "log"), ("apm", "f", 0.03, "log"),
          ("eps", "f", 1e-6, "log"), ("floor", "f", 1e-6, "log"),
          ("m1lr1", "m", 0.002, "log"), ("m1lr2", "m", 0.0005, "log"), ("m2eta", "m", 1.0, "log"),
          ("m2share", "m", 1e-3, "log"), ("m2geta", "m", 1.0, "log"), ("m2gshare", "m", 1e-3, "log")]
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
    """The two params strings of a point x (every searched key written out, so the code's defaults do not matter)."""
    f = ",".join("%s=%s" % (k, fmtv(x[k])) for k in sorted(x) if RGROUP[k] == "f")
    m = ",".join("%s=%s" % (k, fmtv(x[k])) for k in sorted(x) if RGROUP[k] == "m")
    return f, m


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
            return float(self.cache[(f, m)]["score"]), self.cache[(f, m)]
        tag = hashlib.sha1(("%s|%s|%s" % (os.path.basename(self.dump), f, m)).encode()).hexdigest()[:12]
        d = os.path.join(OUT, "replay")
        os.makedirs(d, exist_ok=True)
        base = os.path.join(d, tag)
        with Slot():
            t0 = time.time()
            with open(base + ".out", "w") as so, open(base + ".err", "w") as se:
                rc = subprocess.call(["/usr/bin/time", "-v", "-o", base + ".time", BIN, "meta-mix-replay", self.dump, f, m],
                                     stdout=so, stderr=se)
            sec = time.time() - t0
        _, rss = parse_time(base + ".time")
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
            y[k] = x[k] * s
            out.append((k, s, y))
    return out


def pattern_search(rp, x, jobs, steps=(2.0, 1.41, 1.19), rounds=6, cats=True, label=""):
    """Coordinate (pattern) search: every coordinate's up and down move from x is tried in parallel; the improving moves
    are combined (the combination is kept if it beats the best single move, else the best single move is); then the
    next round. When no move improves, the step shrinks."""
    s0, _ = rp.batch([x], 1)[0]
    log("%s start %.4f" % (label, s0))
    active = [c[0] for c in RCOORD]
    for step in steps:
        for rnd in range(rounds):
            cand = moves(x, step, active)
            if cats and step == steps[0] and rnd == 0:
                for k, _, d, alts in RCAT:
                    for a in alts:
                        y = dict(x)
                        y[k] = a
                        cand.append((k, a, y))
            res = rp.batch([y for _, _, y in cand], jobs)
            best = {}
            for (k, s, y), (sc, _) in zip(cand, res):
                if sc < s0 - TOL and (k not in best or sc < best[k][0]):
                    best[k] = (sc, y[k])
            if step == steps[0] and rnd == 0:  # drop coordinates that move nothing either way
                eff = {}
                for (k, s, y), (sc, _) in zip(cand, res):
                    eff[k] = max(eff.get(k, 0), abs(sc - s0))
                active = [k for k in active if eff.get(k, 0) >= TOL]
                log("%s active coordinates: %s" % (label, ",".join(active)))
            if not best:
                log("%s step %.2f: no move improves %.4f" % (label, step, s0))
                break
            k1 = min(best, key=lambda k: best[k][0])
            single = dict(x)
            single[k1] = best[k1][1]
            if len(best) > 1:
                comb = dict(x)
                for k, (_, v) in best.items():
                    comb[k] = v
                sc_c, _ = rp.batch([comb], 1)[0]
            else:
                sc_c = math.inf
            if sc_c < best[k1][0]:
                x, s0 = comb, sc_c
                log("%s step %.2f round %d: combined %s -> %.4f" % (label, step, rnd, ",".join(best), s0))
            else:
                x, s0 = single, best[k1][0]
                log("%s step %.2f round %d: %s=%s -> %.4f" % (label, step, rnd, k1, fmtv(best[k1][1]), s0))
    return x, s0


def cmd_replay(a):
    rp = Replay(a.dump, phase="main")
    x0 = dict(RDEFAULT)
    if a.start:
        x0.update(json.loads(a.start))
    xb, sb = pattern_search(rp, x0, a.jobs, label="main")
    results = [("main", xb, sb)]
    rng = random.Random(a.seed)
    for r in range(a.restarts):
        rp.phase = "restart%d" % (r + 1)
        xr = dict(xb if a.restart_from_best else RDEFAULT)
        for k, _, d, _ in RCOORD:
            xr[k] = xr[k] * math.exp(rng.uniform(-math.log(a.spread), math.log(a.spread)))
        x, s = pattern_search(rp, xr, a.jobs, steps=(1.41, 1.19), rounds=4, cats=False, label=rp.phase)
        results.append((rp.phase, x, s))
    results.sort(key=lambda t: t[2])
    for ph, x, s in results:
        f, m = rparams(x)
        log("RESULT %s %.4f --full-mix-params %s --meta-mix-params %s" % (ph, s, f, m))
    with open(os.path.join(OUT, "replay_best_%s.json" % os.path.basename(a.dump).split(".")[0]), "w") as fo:
        json.dump({"phase": results[0][0], "score": results[0][2], "x": results[0][1]}, fo, indent=1)


# ------------------------------------------------------------------------------------------------ expert trials

XLOG = os.path.join(OUT, "expert_trials.tsv")
XHEAD = ["trial", "slice", "phase", "flags", "score", "first_half", "ABs", "online_final", "ngram_alone", "lstm_alone",
         "seconds", "peak_rss_mb", "status"]


def slug(flags, sl="big"):
    s = []
    i = 0
    while i < len(flags):
        k = flags[i].lstrip("-")
        v = flags[i + 1] if i + 1 < len(flags) and not flags[i + 1].startswith("--") else ""
        i += 2 if v else 1
        if k in ("full-mix-params", "meta-mix-params"):  # long: a short hash
            v = hashlib.sha1(v.encode()).hexdigest()[:8]
        s.append(k + ("=" + v.replace(",", ".").replace(":", "_").replace("=", "~") if v else ""))
    return PREFIX[sl] + ("__".join(s) if s else "default")


def run_expert(flags, phase, extra=None, sl="big", extra_flags=()):
    """One context8k run on slice sl with flags; returns (score, row). Reuses a finished run of the same name; waits
    for one still running (a run launched by hand under the same name). extra_flags (a dump) do not change the output
    and are not part of the name."""
    name = slug(flags, sl) if extra is None else extra
    d = os.path.join(OUT, "runs")
    os.makedirs(d, exist_ok=True)
    base = os.path.join(d, name)
    for r in tsv_read(XLOG):
        if r["trial"] == name and r["status"] == "ok":
            return float(r["score"]), r
    def alive():  # a run of this name still going (launched by hand, or by another driver)
        return subprocess.call(["pgrep", "-f", base + ".time"], stdout=subprocess.DEVNULL) == 0
    while not os.path.exists(base + ".wall") and alive():
        time.sleep(30)
    if not os.path.exists(base + ".wall"):
        with Slot(full=sl == "big"):
            env = dict(os.environ, CYPHA_LSTM_THREADS=os.environ.get("CYPHA_LSTM_THREADS", "2"),
                       CYPHA_NGRAM_THREADS=os.environ.get("CYPHA_NGRAM_THREADS", "2"))
            cmd = [BIN, "context8k", ENWIK8, DOLLY] + SLICES[sl] + list(flags) + list(extra_flags)
            with open(base + ".cmd", "w") as fo:
                fo.write(" ".join(cmd) + "\n")
            t0 = time.time()
            with open(base + ".out", "w") as so, open(base + ".err", "w") as se:
                rc = subprocess.call(["/usr/bin/time", "-v", "-o", base + ".time"] + cmd, stdout=so, stderr=se, env=env,
                                     cwd=os.path.expanduser("~/t1"))
            with open(base + ".wall", "w") as fo:
                fo.write("%s rc=%d wall=%ds\n" % (name, rc, time.time() - t0))
    txt = open(base + ".out").read()
    wall, rss = parse_time(base + ".time")
    m = re.search(r"^context8k_bpc ([\d.]+)", txt, re.M)
    arms = parse_arms(txt)

    def line(k):
        mm = re.search(r"^%s ([\d.]+)" % k, txt, re.M)
        return mm.group(1) if mm else ""
    ok = m is not None and "rc=0" in open(base + ".wall").read()
    row = {"trial": name, "slice": sl, "phase": phase, "flags": " ".join(flags), "score": m.group(1) if m else "",
           "first_half": "%.4f" % arms["M1Gs"][0] if "M1Gs" in arms else "", "ABs": "%.4f" % arms["ABs"][1] if "ABs" in arms else "",
           "online_final": line("online_final_bpc"), "ngram_alone": line("ngram_alone_bpc"), "lstm_alone": line("lstm_alone_bpc"),
           "seconds": "%.0f" % wall, "peak_rss_mb": "%.0f" % (rss / 1024), "status": "ok" if ok else "failed"}
    tsv_append(XLOG, row, XHEAD)
    log("expert %s %s %ss %sMB" % (row["score"], name, row["seconds"], row["peak_rss_mb"]))
    return (float(row["score"]) if ok else math.inf), row


# The registered first round: each move from the defaults, alone. (setting, flags)
XMOVES = [
    ("ngram-cap", ["--ngram-cap", "30"]), ("ngram-cap", ["--ngram-cap", "120"]),
    ("ngram-lr", ["--ngram-lr", "0.0025"]), ("ngram-lr", ["--ngram-lr", "0.01"]),
    ("lstm-window", ["--lstm-window", "12"]), ("lstm-window", ["--lstm-window", "30"]),
    ("ngram-orders", ["--ngram-orders", "1,2,3,4,5,6,7,8,10,12,16"]), ("ngram-orders", ["--ngram-orders", "1,2,3,4,6,8,12,16"]),
    ("bkn", ["--bkn", "8:4:2:1"]), ("bkn", ["--bkn", "6:4:1:1"]),
    ("sim T", ["--sim", "default,T=0.035"]), ("sim T", ["--sim", "default,T=0.07"]),
    ("cls", ["--cls", "shape:6:4:cp2"]), ("doc", ["--doc", "trig,cache"]),
]


def flags_dict(flags):
    d, i = {}, 0
    while i < len(flags):
        d[flags[i]] = flags[i + 1]
        i += 2
    return d


def dict_flags(d):
    out = []
    for k in sorted(d):
        out += [k, d[k]]
    return out


def cmd_expert(a):
    base = {}
    if a.start:
        base = flags_dict(json.loads(a.start))
    mv = XMOVES if not a.only else [m for m in XMOVES if m[0] in a.only.split(";")]
    cand = []
    for k, fl in mv:
        y = dict(base)
        y.update(flags_dict(fl))
        cand.append((k, y))
    bx = json.loads(a.base_extra) if a.base_extra else []
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:  # the start point first (it may write a dump), then round 1
        fb = ex.submit(run_expert, dict_flags(base), "base", a.base_name or None, a.slice, bx)
        res = [ex.submit(run_expert, dict_flags(y), "round1", None, a.slice) for _, y in cand]
        res = [r.result() for r in res]
        s0 = fb.result()[0]
    log("expert base %.4f %s" % (s0, base))
    best = {}
    for (k, y), (sc, _) in zip(cand, res):
        if sc < s0 - TOL and (k not in best or sc < best[k][0]):
            best[k] = (sc, y)
    log("expert round 1 improving: %s" % {k: (v[0], v[1]) for k, v in best.items()})
    if not best:
        print(json.dumps({"flags": base, "score": s0}))
        return
    k1 = min(best, key=lambda k: best[k][0])
    x, sx = best[k1][1], best[k1][0]
    if len(best) > 1:
        comb = dict(base)
        for k, (_, y) in best.items():
            comb.update({f: v for f, v in y.items() if f not in base or base[f] != v})
        sc, _ = run_expert(dict_flags(comb), "combined", None, a.slice)
        if sc < sx:
            x, sx = comb, sc
    log("RESULT expert %s %.4f %s" % (a.slice, sx, " ".join(dict_flags(x))))
    with open(os.path.join(OUT, "expert_best_%s.json" % a.slice), "w") as fo:
        json.dump({"score": sx, "flags": x, "base_score": s0, "improving": {k: v for k, v in best.items()}}, fo, indent=1)


# ------------------------------------------------------------------------------------------------ sensitivity

SLOG = os.path.join(OUT, "sensitivity.tsv")
SHEAD = ["setting", "how", "theta_star", "mult", "value", "bpc", "delta", "seconds"]


def cmd_sens(a):
    """The mixers' continuous settings by replay of the tuned experts' dump (others at theta*); with --flags (every tuned
    flag, a JSON list) and --base-name (the slice run of exactly those flags), the experts' continuous settings by full
    runs on the slice, every other flag at its tuned value."""
    done = {(r["setting"], r["mult"]) for r in tsv_read(SLOG)}
    if a.dump:
        rp = Replay(a.dump, phase="sens")
        xs = dict(RDEFAULT)
        xs.update(json.load(open(a.replay_best))["x"] if a.replay_best else {})
        s0, _ = rp.run(xs)
        pts = []
        for k, _, _, _ in RCOORD:
            for m in MULTS:
                y = dict(xs)
                y[k] = xs[k] * m
                pts.append((k, m, y))
        res = rp.batch([y for _, _, y in pts], a.jobs)
        for (k, m, y), (sc, row) in zip(pts, res):
            if (k, "%g" % m) not in done:
                tsv_append(SLOG, {"setting": k, "how": "replay", "theta_star": fmtv(xs[k]), "mult": "%g" % m,
                                  "value": fmtv(y[k]), "bpc": "%.4f" % sc, "delta": "%+.4f" % (sc - s0),
                                  "seconds": row.get("seconds", "")}, SHEAD)
    if a.flags:  # the experts' continuous settings: full runs on the slice
        xf = flags_dict(json.loads(a.flags))
        sx, _ = run_expert(dict_flags(xf), "sens", a.base_name or None, a.slice)
        jobsx = []
        for f, dflt in (("--ngram-cap", 60), ("--ngram-lr", 0.005)):
            th = float(xf.get(f, dflt))
            for m in MULTS:
                y = dict(xf)
                y[f] = str(int(round(th * m))) if f == "--ngram-cap" else "%.6g" % (th * m)
                jobsx.append((f.lstrip("-"), th, m, y, y[f]))
        mm = re.search(r"T=([\d.]+)", xf.get("--sim", ""))
        th = float(mm.group(1)) if mm else 0.05
        for m in MULTS:
            y = dict(xf)
            y["--sim"] = "default,T=%.6g" % (th * m)
            jobsx.append(("sim T", th, m, y, "%.6g" % (th * m)))
        with ThreadPoolExecutor(max_workers=a.jobs_expert) as ex:
            rx = list(ex.map(lambda j: run_expert(dict_flags(j[3]), "sens", None, a.slice), jobsx))
        for (key, th, m, y, v), (sc, row) in zip(jobsx, rx):
            if (key, "%g" % m) not in done:
                tsv_append(SLOG, {"setting": key, "how": "run", "theta_star": "%g" % th, "mult": "%g" % m, "value": v,
                                  "bpc": "%.4f" % sc, "delta": "%+.4f" % (sc - sx), "seconds": row.get("seconds", "")}, SHEAD)


# ------------------------------------------------------------------------------------------------ the rest


def cmd_settings(a):
    print("setting\tflag\tdefault\tkind\thow\tnote")
    for s in SETTINGS:
        print("\t".join(s))


def cmd_points(a):
    """The replay score of given points (JSON: a list of {"name": .., "x": {param: value}}) on a dump, e.g. the small
    slice's optima confirmed on the big slice's dump. Prints one RESULT line a point."""
    pts = json.load(open(a.points))
    rp = Replay(a.dump, phase=a.phase)
    xs = []
    for p in pts:
        x = dict(RDEFAULT)
        x.update(p.get("x", {}))
        xs.append(x)
    res = rp.batch(xs, a.jobs)
    for p, (sc, row) in zip(pts, res):
        log("RESULT points %s %.4f best_arm %s %s" % (p["name"], sc, row.get("best_arm"), row.get("best_arm_bpc")))


def cmd_best(a):
    for path, key in ((XLOG, "score"), (RLOG, "score")):
        rows = [r for r in tsv_read(path) if r.get("status") == "ok" and r.get(key)]
        if rows:
            b = min(rows, key=lambda r: float(r[key]))
            print(os.path.basename(path), len(rows), "trials; best", json.dumps(b))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("settings")
    sp.add_parser("best")
    e = sp.add_parser("expert")
    e.add_argument("--jobs", type=int, default=3)
    e.add_argument("--start", default="", help="JSON list of flags to start from")
    e.add_argument("--only", default="", help="settings to move, ';'-separated")
    e.add_argument("--base-name", default="", help="the run name of the start point (e.g. SM0, the dump run)")
    e.add_argument("--base-extra", default="", help="JSON list of flags for the start point's run only (a dump)")
    e.add_argument("--slice", default="small", choices=sorted(SLICES))
    r = sp.add_parser("replay")
    r.add_argument("--dump", required=True)
    r.add_argument("--jobs", type=int, default=6)
    r.add_argument("--restarts", type=int, default=2)
    r.add_argument("--spread", type=float, default=2.5, help="restarts: each rate x exp(U(-ln s, ln s))")
    r.add_argument("--restart-from-best", action="store_true")
    r.add_argument("--seed", type=int, default=20261105)
    r.add_argument("--start", default="", help="JSON dict of params to start from")
    s = sp.add_parser("sens")
    s.add_argument("--dump", default="")
    s.add_argument("--replay-best", default="")
    s.add_argument("--flags", default="", help="JSON list: every tuned flag (experts and mixers)")
    s.add_argument("--base-name", default="")
    s.add_argument("--slice", default="small", choices=sorted(SLICES))
    s.add_argument("--jobs", type=int, default=6)
    s.add_argument("--jobs-expert", type=int, default=3)
    q = sp.add_parser("points")
    q.add_argument("--dump", required=True)
    q.add_argument("--points", required=True)
    q.add_argument("--phase", default="confirm")
    q.add_argument("--jobs", type=int, default=2)
    a = p.parse_args()
    {"settings": cmd_settings, "best": cmd_best, "expert": cmd_expert, "replay": cmd_replay, "sens": cmd_sens,
     "points": cmd_points}[a.cmd](a)


if __name__ == "__main__":
    main()
