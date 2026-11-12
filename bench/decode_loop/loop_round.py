#!/usr/bin/env python3
"""DECODE_LOOP.md: a round's blind packet, and its scoring into rounds.json. Odin Loch, 2026-11-13.

  loop_round.py packet N BEST S1 [S2]        round N: 20 rater prompts by rotation, seed 20261210 + N, bench/blind/loop_rN
  loop_round.py confirm C BEST START         confirmation C: all 50, seed 20261310 + C, bench/blind/loop_cC
  loop_round.py score NAME KEYFILE           reads bench/blind/NAME/R1..R4.json and the key; prints the result by the
                                             registered rule and logs every judgment in bench/decode_loop/rounds.json
KEY.json is moved to KEYDIR (env DLOOP_KEYDIR) as NAME_KEY.json; it never enters the repository.
"""
import json
import os
import random
import shutil
import statistics as st
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUNS = os.environ.get("DLOOP_RUNS", os.path.expanduser("~/dloop_runs"))
KEYDIR = os.environ.get("DLOOP_KEYDIR")
R50 = json.load(open(os.path.join(ROOT, "bench/blind/llm_rating/prompts.json")))["prompts"]


def rotation(n):
    s = (20 * (n - 1)) % 50
    return [R50[(s + j) % 50] for j in range(20)]


def rotation_differing(n, best, others):
    """Amendment 2 (rounds 3 on): walk the 50 from round n's start; take the first 20 prompts where every survivor's
    text (cut at 460 characters) differs from the best's; if fewer than 20, add those where at least one differs."""
    sys.path.insert(0, HERE)
    from blind_cut import cut_texts
    t = {lab: cut_texts(os.path.join(RUNS, lab + ".gen.txt")) for lab in [best] + others}
    s = (20 * (n - 1)) % 50
    order = [R50[(s + j) % 50] for j in range(50)]
    allv = [i for i in order if all(t[o][i] != t[best][i] for o in others)]
    some = [i for i in order if i not in allv and any(t[o][i] != t[best][i] for o in others)]
    return (allv + some)[:20]


def make(name, ids, best, others, seed):
    out = os.path.join(ROOT, "bench/blind", name)
    os.makedirs(out, exist_ok=True)
    json.dump({"prompts": ids}, open(os.path.join(out, "ids.json"), "w"))
    cmd = [sys.executable, "-I", os.path.join(ROOT, "scripts/blind_rate.py"), "make", "--out", out,
           "--ids", os.path.join(out, "ids.json"), "--mode", "pair", "--baseline", best, "--seed", str(seed),
           "--prompts", os.path.join(ROOT, "bench/eval/gen_prompts_dev.tsv")]
    for lab in [best] + others:
        cmd += ["--arm", f"{lab}={os.path.join(RUNS, lab + '.gen.txt')}"]
    subprocess.run(cmd, check=True, cwd=ROOT)
    for lab in [best] + others:  # the GEN files the packet was built from, committed beside it
        shutil.copy(os.path.join(RUNS, lab + ".gen.txt"), os.path.join(out, lab + ".gen.txt"))
    shutil.move(os.path.join(out, "KEY.json"), os.path.join(KEYDIR, f"{name}_KEY.json"))
    print(f"{name}: {len(ids)} prompts, best {best}, against {', '.join(others)}, seed {seed}; key in {KEYDIR}")


def score(name, keyfile):
    d = os.path.join(ROOT, "bench/blind", name)
    key = json.load(open(keyfile))
    judg = []
    for r in range(1, 5):
        for row in json.load(open(os.path.join(d, f"R{r}.json"))):
            n = row["item"]
            ka, kb = key[f"R{r}/{n}/A"], key[f"R{r}/{n}/B"]
            base = None
            arms = {ka["arm"], kb["arm"]}
            ch = row["choice"]
            judg.append(dict(r=r, n=n, dev=ka["dev"], A=ka["arm"], B=kb["arm"], choice=ch))
    log = json.load(open(os.path.join(HERE, "rounds.json")))
    ent = next(x for x in log["rounds"] if x["name"] == name)
    best = ent["best"]
    js, shares = [], {}
    rng = random.Random(0)
    for j in judg:
        arm = j["A"] if j["A"] != best else j["B"]
        win = 0.5 if j["choice"] == "tie" else float(key[f"R{j['r']}/{j['n']}/{j['choice']}"]["arm"] == arm)
        # raters: confirmations have four (R1..R4); rounds two (X reads R1 and R3, Y reads R2 and R4)
        rater = f"R{j['r']}" if ent["kind"] == "confirm" else ("X" if j["r"] in (1, 3) else "Y")
        js.append(dict(dev=j["dev"], arm=arm, rater=rater, win=win, packet=j["r"]))
    print(f"# {name}: against {best}")
    for arm in sorted({x["arm"] for x in js}):
        per = {}
        for x in js:
            if x["arm"] == arm:
                per.setdefault(x["dev"], []).append(x["win"])
        vals = [st.mean(v) for v in per.values()]
        bs = sorted(st.mean(rng.choices(vals, k=len(vals))) for _ in range(10000))
        m = st.mean(vals)
        by = {}
        for x in js:
            if x["arm"] == arm:
                key_ = x["rater"] if ent["kind"] == "round" else ("pair 1" if x["packet"] <= 2 else "pair 2")
                by.setdefault(key_, []).append(x["win"])
        parts = ", ".join(f"{k} {st.mean(v):.2f}" for k, v in sorted(by.items()))
        if ent["kind"] == "round":
            ok = m >= 0.60 and all(st.mean(v) >= 0.5 for v in by.values())
            verdict = "becomes the new best" if ok else "does not"
        else:
            ok = bs[250] > 0.5 and all(st.mean(v) > 0.5 for v in by.values())
            verdict = "reads better by §13.38" if ok else "does not pass §13.38"
        shares[arm] = m
        ent.setdefault("result", {})[arm] = dict(share=m, lo=bs[250], hi=bs[9749], by=by and {k: st.mean(v) for k, v in by.items()}, passes=ok)
        print(f"  {arm:12s} {m:.3f} [{bs[250]:.3f}, {bs[9749]:.3f}]  {parts}  prompts {len(vals)}: {verdict}")
    ent["judgments"] = js
    ent["shares"] = shares
    json.dump(log, open(os.path.join(HERE, "rounds.json"), "w"), indent=1)


def main():
    cmd = sys.argv[1]
    log = json.load(open(os.path.join(HERE, "rounds.json")))
    if cmd == "packet":
        n, best, others = int(sys.argv[2]), sys.argv[3], sys.argv[4:]
        name = f"loop_r{n}"
        ids = rotation(n) if n <= 2 else rotation_differing(n, best, others)
        make(name, ids, best, others, 20261210 + n)
        log["rounds"] = [x for x in log["rounds"] if x["name"] != name] + [dict(name=name, kind="round", best=best, arms=others)]
    elif cmd == "confirm":
        c, best, start = int(sys.argv[2]), sys.argv[3], sys.argv[4]
        name = f"loop_c{c}"
        make(name, R50, start, [best], 20261310 + c)
        log["rounds"] = [x for x in log["rounds"] if x["name"] != name] + [dict(name=name, kind="confirm", best=start, arms=[best])]
    elif cmd == "score":
        return score(sys.argv[2], sys.argv[3])
    json.dump(log, open(os.path.join(HERE, "rounds.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
