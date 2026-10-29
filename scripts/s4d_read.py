#!/usr/bin/env python3
"""S4d: apply the registered rules to pattern_learn's stdout (docs/reports/S4_UNSEEN_AND_FRONTIER.md, "S4d").

  python3 -I scripts/s4d_read.py dev  DEV.out
  python3 -I scripts/s4d_read.py test TEST.out CHOSEN [DEV.out]

dev: the dev choice of gate (a). test: gates (a) and (b), the threshold rule (c), and the predictions H1-H9.
It reads pattern_learn's SIGNAL lines only. Rule 10: these signals answer nothing.
"""
import math
import re
import sys

CANDS = ["O1-register", "O2-rare", "O3-topic", "O4-delta", "O5-novelty"]
FA_SIGNALS = ["unparsed", "vocabulary"] + CANDS
HIGH = "@A100k"
REPS = 200

FA_RE = re.compile(r"^  SIGNAL_FA (\S+) (running|fixed)\s*\| C scored (\d+) events ([\d.]+) alarms (\d+) \| "
                   r"S scored (\d+) events (\d+) rate ([\d.]+) alarms (\d+)")
DET_RE = re.compile(r"^  SIGNAL_DETECT (\S+) pool (\S+) pi ([\d.]+) (running|fixed)\s*\| within 50\s+(\d+)/(\d+) "
                    r"within 200\s+(\d+)/(\d+) \| delay median (\S+) p90 (\S+)")
SIG_RE = re.compile(r"^SIGNAL (\S+) units (.*?) \| strata")
OSET_RE = re.compile(r"^  SIGNAL_O_SET \S+ (\S+) pages (\d+) units (\d+)")


def wilson(k, n, z=1.959964):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def share(k, n=REPS):
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({k / n:.1%}, 95% {lo:.3f}-{hi:.3f})"


def med(s):
    return 10**9 if s == ">200" else int(s)


def parse(path):
    fa, det, units, oset = {}, {}, {}, {}
    alarms = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            m = FA_RE.match(line)
            if m:
                fa[(m[1], m[2])] = dict(c_alarms=int(m[5]), s_scored=int(m[6]), s_events=int(m[7]), s_rate=float(m[8]), s_alarms=int(m[9]))
                continue
            m = DET_RE.match(line)
            if m:
                det[(m[1], m[2], float(m[3]), m[4])] = dict(w50=int(m[5]), n=int(m[6]), w200=int(m[7]), median=m[9], p90=m[10])
                continue
            m = SIG_RE.match(line)
            if m:
                toks = m[2].split()
                units[m[1]] = {toks[i]: int(toks[i + 1]) for i in range(0, len(toks) - 1, 2)}
                continue
            m = OSET_RE.match(line)
            if m:
                oset[m[1]] = (int(m[2]), int(m[3]))
                continue
            if line.startswith("    ALARM ") and " running S unit " in line:
                alarms.setdefault(line.split()[1], []).append(line.strip())
    return fa, det, units, oset, alarms


def dev_choice(fa, det, say):
    say("Dev choice of gate (a) (A = 20,000, running reference; stream = Sdev, pools O and Pdev at pi = 1)")
    say("| Candidate | Sdev alarms | Pdev within 50 | O-dev within 50 | O-dev median delay | Eligible |")
    say("|---|---|---|---|---|---|")
    rows = []
    for i, c in enumerate(CANDS):
        a = fa[(c, "running")]["s_alarms"]
        p = det[(c, "Pdev", 1.0, "running")]
        o = det[(c, "O", 1.0, "running")]
        ok = a <= 1 and p["w50"] <= 0.10 * p["n"]
        rows.append((c, ok, o["w50"], med(o["median"]), i))
        say(f"| {c} | {a} | {share(p['w50'], p['n'])} | {share(o['w50'], o['n'])} | {o['median']} | {'yes' if ok else 'no'} |")
    elig = [r for r in rows if r[1]]
    pool = elig if elig else rows
    best = sorted(pool, key=lambda r: (-r[2], r[3], r[4]))[0]
    say("")
    say(f"CHOSEN {best[0]} ({'eligible' if elig else 'none eligible: read on test and reported as failed on dev'})")
    return best[0], bool(elig)


def main():
    mode, path = sys.argv[1], sys.argv[2]
    out = []
    say = lambda s="": out.append(s)
    fa, det, units, oset, alarms = parse(path)
    if mode == "dev":
        dev_choice(fa, det, say)
        say("")
        say("Reported only (dev): L2-dev and Y at pi = 1, running reference, both thresholds")
        say("| Signal | L2 units | L2 within 50 | L2 median | Y within 50 | Sdev alarms |")
        say("|---|---|---|---|---|---|")
        for c in CANDS + ["vocabulary"]:
            for t in ("", HIGH):
                l2 = det.get((c + t, "L2", 1.0, "running"))
                y = det.get((c + t, "Y", 1.0, "running"))
                a = fa[(c + t, "running")]["s_alarms"]
                say(f"| {c + t} | {units[c + t].get('L2', 0)} | {share(l2['w50'], l2['n']) if l2 else '-'} | {l2['median'] if l2 else '-'} | "
                    f"{share(y['w50'], y['n']) if y else '-'} | {a} |")
        print("\n".join(out))
        return

    chosen = sys.argv[3]
    dev_out = sys.argv[4] if len(sys.argv) > 4 else None

    def gate_a(c, t):
        f = fa[(c + t, "running")]["s_alarms"]
        o = det[(c + t, "O", 1.0, "running")]
        p = det[(c + t, "P2", 1.0, "running")]
        conds = [f <= 1, o["w50"] >= 0.90 * o["n"], p["w50"] <= 0.10 * p["n"]]
        return all(conds), conds, f, o, p

    def gate_b(c, t, with_p2=True):
        f = fa[(c + t, "running")]["s_alarms"]
        l2 = det.get((c + t, "L2", 1.0, "running"))
        n_l2 = units[c + t].get("L2", 0)
        p = det.get((c + t, "P2", 1.0, "running"))
        if n_l2 < 20 or l2 is None:
            return None, [], f, l2, p, n_l2
        conds = [f <= 1, l2["w50"] >= 0.90 * l2["n"]] + ([p["w50"] <= 0.10 * p["n"]] if with_p2 else [])
        return all(conds), conds, f, l2, p, n_l2

    yes = lambda b: "holds" if b else "fails"
    for t, A in (("", "20,000"), (HIGH, "100,000")):
        ok, conds, f, o, p = gate_a(chosen, t)
        say(f"Gate (a), {chosen}, A = {A}: F alarms {f} ({yes(conds[0])}); O-test within 50 {share(o['w50'], o['n'])} ({yes(conds[1])}); "
            f"P2 within 50 {share(p['w50'], p['n'])} ({yes(conds[2])}) -> {'PASS' if ok else 'FAIL'}")
        ok, conds, f, l2, p, n = gate_b("O5-novelty", t)
        if ok is None:
            say(f"Gate (b), O5-novelty, A = {A}: L2-test has {n} O5 units (< 20): not read")
        else:
            say(f"Gate (b), O5-novelty, A = {A}: F alarms {f} ({yes(conds[0])}); L2-test units {n}, within 50 {share(l2['w50'], l2['n'])} ({yes(conds[1])}); "
                f"P2 within 50 {share(p['w50'], p['n'])} ({yes(conds[2])}) -> {'PASS' if ok else 'FAIL'}")
        ok, conds, f, l2, p, n = gate_b("vocabulary", t, with_p2=False)
        if ok is None:
            say(f"V on L2-test, A = {A}: {n} units (< 20): not read")
        else:
            say(f"V on L2-test (conditions 1, 2, 4), A = {A}: F alarms {f} ({yes(conds[0])}); L2-test units {n}, within 50 {share(l2['w50'], l2['n'])} "
                f"({yes(conds[1])}) -> {'PASS' if ok else 'FAIL'}")
    say("")

    def fa_sum(t):
        return sum(fa[(s + t, "running")]["s_alarms"] for s in FA_SIGNALS)

    def pass_set(t):
        s = set()
        if gate_a(chosen, t)[0]:
            s.add((chosen, "O-test"))
        if gate_b("O5-novelty", t)[0]:
            s.add(("O5-novelty", "L2-test"))
        if gate_b("vocabulary", t, with_p2=False)[0]:
            s.add(("vocabulary", "L2-test"))
        return s

    fa20, fa100 = fa_sum(""), fa_sum(HIGH)
    p20, p100 = pass_set(""), pass_set(HIGH)
    adopt = fa100 < fa20 and p100 >= p20
    say("Threshold rule (c): alarms on F, running reference, per signal (A = 20,000 / A = 100,000)")
    say("| Signal | F units scored | F events | F event rate | alarms A = 20,000 | alarms A = 100,000 |")
    say("|---|---|---|---|---|---|")
    for s in FA_SIGNALS:
        a, b = fa[(s, "running")], fa[(s + HIGH, "running")]
        say(f"| {s} | {a['s_scored']} | {a['s_events']} | {a['s_rate']:.4f} | {a['s_alarms']} | {b['s_alarms']} |")
    say(f"FA(20,000) = {fa20}, FA(100,000) = {fa100}")
    say(f"PASS(20,000) = {sorted(p20)}; PASS(100,000) = {sorted(p100)}")
    say(f"THRESHOLD {'A = 100,000 adopted' if adopt else 'A stays 20,000'} (FA lower: {fa100 < fa20}; PASS kept: {p100 >= p20})")
    say("")
    say("Alarms on F (running reference):")
    for s in FA_SIGNALS:
        for t in ("", HIGH):
            for a in alarms.get(s + t, []):
                say("  " + a)
    say("")

    say("Detections at pi = 1, running reference (A = 20,000 | A = 100,000):")
    say("| Signal | Pool | within 50 | median | within 50 @100k | median @100k | +median |")
    say("|---|---|---|---|---|---|---|")
    for s in CANDS + ["vocabulary", "unparsed"]:
        for pool in ("O", "Y", "L2", "P2"):
            a, b = det.get((s, pool, 1.0, "running")), det.get((s + HIGH, pool, 1.0, "running"))
            if not a:
                continue
            dm = (str(med(b["median"]) - med(a["median"])) if a["median"] != ">200" and b["median"] != ">200" else "-")
            say(f"| {s} | {pool} | {share(a['w50'], a['n'])} | {a['median']} | {share(b['w50'], b['n'])} | {b['median']} | {dm} |")
    say("")

    # Predictions.
    say("Predictions:")
    dev_chosen = None
    if dev_out:
        dfa, ddet, _, doset, _ = parse(dev_out)
        dev_chosen, _ = dev_choice(dfa, ddet, lambda s="": None)
    say(f"H1 O1-register chosen on dev: {'yes' if dev_chosen == 'O1-register' else 'no (' + str(dev_chosen) + ')'}")
    ok, conds, f, o, p = gate_a(chosen, "")
    lo = wilson(o["w50"], o["n"])[0]
    say(f"H2 {chosen} passes gate (a) with O-test Wilson lower bound >= 0.85: {'yes' if ok and lo >= 0.85 else 'no'} (pass {ok}, lower {lo:.3f})")
    y = det[(chosen, "Y", 1.0, "running")]
    say(f"H3 Y-test within 50 in 40-90%: {'yes' if 0.40 * y['n'] <= y['w50'] <= 0.90 * y['n'] else 'no'} ({share(y['w50'], y['n'])})")
    ok5 = gate_b("O5-novelty", "")[0]
    say(f"H4 O5 passes gate (b): {'yes' if ok5 else 'no' if ok5 is False else 'not read'}")
    v = det.get(("vocabulary", "L2", 1.0, "running"))
    say(f"H5 V on L2-test within 50 < 90%: {('yes' if v['w50'] < 0.90 * v['n'] else 'no') + ' (' + share(v['w50'], v['n']) + ')' if v else 'not read'}")
    p2 = {c: det[(c, "P2", 1.0, "running")]["w50"] for c in CANDS}
    rates = {s: fa[(s, "running")]["s_rate"] for s in FA_SIGNALS}
    h6a = all(k <= 0.05 * REPS for k in p2.values())
    h6b = all(0.03 <= r <= 0.08 for r in rates.values())
    say(f"H6 P2 within 50 <= 5% for every candidate: {'yes' if h6a else 'no'} ({p2}); F event rates in [0.03, 0.08]: {'yes' if h6b else 'no'} "
        f"({ {k: round(v2, 4) for k, v2 in rates.items()} })")
    say(f"H7 FA(20,000) in 1-3 and FA(100,000) lower with PASS kept: {'yes' if 1 <= fa20 <= 3 and adopt else 'no'} (FA {fa20} / {fa100}, adopted {adopt})")
    both = []
    for (c, pool) in sorted(p20 & p100):
        pool_name = pool.split("-")[0]
        a, b = det[(c, pool_name, 1.0, "running")], det[(c + HIGH, pool_name, 1.0, "running")]
        both.append((c, pool, med(b["median"]) - med(a["median"])))
    h8 = bool(both) and all(1 <= d <= 8 for _, _, d in both)
    say(f"H8 median delay 1-8 units longer at A = 100,000 for pairs passing at both: {'yes' if h8 else 'no' if both else 'not read (no pair passes at both)'} ({both})")
    o_pages, o_units = oset.get("O", (0, 0))
    l2_units = oset.get("L2", (0, 0))[1]
    h9 = o_pages and o_units >= 0.75 * o_pages and l2_units >= 25
    say(f"H9 >= 75% of O pages are O units and L2-test >= 25 O units: {'yes' if h9 else 'no'} (O {o_units}/{o_pages} = {o_units / max(1, o_pages):.3f}; L2 {l2_units})")
    print("\n".join(out))


if __name__ == "__main__":
    main()
