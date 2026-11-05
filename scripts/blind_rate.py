#!/usr/bin/env python3
"""Blind reading of generated text by independent raters (GENERATION_PLAN §13.31-13.37). Odin Loch, 2026-11-05.

Two steps, with the raters in between:

  make   builds rater packets and a key from GEN files (cyphalm_trie sample / scripts/gen_eval.sh output):
           scripts/blind_rate.py make --out DIR --ids bench/blind/llm_rating/prompts.json \
               --arm B0=path/B0.gen.txt --arm T2=path/T2.gen.txt [--human] [--mode rate|pair] [--baseline B0] [--seed S]
         mode rate: each item shows the prompt and every arm (plus the human text with --human) in random order, A, B, ...;
                    raters give 1-10 each. Two raters per prompt: raters 1-2 see the first half of the prompts, 3-4 the
                    second (each with its own order and labels).
         mode pair: each item shows the prompt and two continuations, the baseline and one other arm, in random order;
                    raters choose the one that reads better (or "tie"). Use it when the 1-10 scale sits at its floor
                    (§13.36: every K5 variant got exactly 2).
         Writes DIR/R1.txt..R4.txt (the packets), DIR/KEY.json (keep it away from the raters), DIR/key.b64 (to commit)
         and DIR/RATER_PROMPT.md (what to send each rater).
  score  reads DIR/R1.json..R4.json (each rater's JSON reply) and DIR/KEY.json and prints the result:
           scripts/blind_rate.py score --dir DIR --baseline B0
         rate: mean per arm, and each arm minus the baseline per prompt (two raters averaged), with a 95% bootstrap
               interval over prompts and the sign in each rater pair.
         pair: each arm's share of wins against the baseline (a tie counts half), with a 95% bootstrap interval.

Raters: four fresh, independent model instances (Claude subagents), each told to read exactly one packet file and
nothing else, with no tools beyond that read and no key. Register the arms, the rule and the predictions in
GENERATION_PLAN before running them; commit the packets and key.b64 first. The rule used so far: an arm reads better
than the baseline if its mean paired difference (rate) or win share minus 0.5 (pair) is above 0, the 95% interval
excludes 0, and both rater pairs agree on the sign.
"""
import argparse
import base64
import csv
import json
import os
import random
import statistics as st
import sys

LABELS = "ABCDEFGHIJ"


def cut(t, n=460):
    t = " ".join(t.split())
    if len(t) <= n:
        return t
    t = t[:n]
    t = t[: t.rfind(" ")]
    return t.rstrip(" ,;:") + " …"


def read_gen(path):
    out, cur = {}, None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("PROMPT\t"):
                cur = int(line.split("\t")[1])
            elif line.startswith("GEN\t") and cur is not None and cur not in out:
                out[cur] = line.split("\t", 2)[2]
    return out


def read_prompts(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            rows[int(row["id"].split("/")[1])] = row
    return rows


RATE_PROMPT = """You are a careful, independent reader rating short text continuations. This is a blind rating task.

Read exactly one file with the Read tool: {packet}
Do not open, list or search any other file, and do not run commands. Use only that file.

The file has {n} items. Each item is a PROMPT (the start of a Wikipedia-style passage) and {k} continuations, {labs}, from different sources, in random order. Every continuation is cut at about 460 characters; a trailing "…" is that cut, not a flaw.

Rate each continuation from 1 to 10, as a reader would, on how well it reads as coherent prose that continues the prompt:
- 1: word salad, not sentences.
- 3: mostly sentence-shaped, but they don't connect or make sense.
- 5: readable sentences with some sense, but drifting off the prompt's topic or not building on each other.
- 7: clear, connected and on topic, with small slips.
- 10: reads like a good human writer continuing this exact passage.
Judge each text on its own merits. Do not try to guess where it came from. Text that keeps repeating the prompt's keywords without saying anything coherent is not on topic in the sense meant here: rate what a reader gets out of it. Odd character codes (like "xe2x80x94") or odd spacing count only as far as they hurt readability.

Reply with JSON only, no other text: a list of {n} objects in item order, each {{"item": n, {fields}}}, with integer scores."""

PAIR_PROMPT = """You are a careful, independent reader comparing short text continuations. This is a blind comparison task.

Read exactly one file with the Read tool: {packet}
Do not open, list or search any other file, and do not run commands. Use only that file.

The file has {n} items. Each item is a PROMPT (the start of a Wikipedia-style passage) and two continuations, A and B, from different sources, in random order. Every continuation is cut at about 460 characters; a trailing "…" is that cut, not a flaw.

For each item, choose the continuation that reads better as coherent prose continuing the prompt: sentences that make sense, connect to each other and stay on the prompt's subject. Text that keeps repeating the prompt's keywords without saying anything coherent is not better for it. Odd character codes or spacing count only as far as they hurt readability. If you truly cannot prefer one, answer "tie", but choose whenever you have any preference.

Reply with JSON only, no other text: a list of {n} objects in item order, each {{"item": n, "choice": "A" | "B" | "tie"}}."""


def make(a):
    rows = read_prompts(a.prompts)
    ids = json.load(open(a.ids))
    ids = ids["prompts"] if isinstance(ids, dict) else ids
    arms = []
    for spec in a.arm:
        name, path = spec.split("=", 1)
        arms.append((name, read_gen(path)))
    for name, g in arms:
        miss = [i for i in ids if len(g.get(i, "").strip()) < 50]
        if miss:
            sys.exit(f"arm {name}: no usable text for prompts {miss[:5]} ({len(miss)})")
    os.makedirs(a.out, exist_ok=True)
    halves = [ids[: len(ids) // 2], ids[len(ids) // 2 :]]
    key = {}
    for r in range(4):
        rr = random.Random(a.seed + r)
        order = halves[r // 2][:]
        rr.shuffle(order)
        out, n = [], 0
        for i in order:
            prompt = " ".join(rows[i]["prompt"].split())
            if a.mode == "rate":
                texts = [(nm, g[i]) for nm, g in arms]
                if a.human:
                    texts.append(("human", rows[i]["continuation"]))
                rr.shuffle(texts)
                n += 1
                out.append(f"### Item {n}\nPROMPT: {prompt}")
                for lab, (arm, t) in zip(LABELS, texts):
                    out.append(f"{lab}: {cut(t)}")
                    key[f"R{r + 1}/{n}/{lab}"] = {"arm": arm, "dev": i}
                out.append("")
            else:
                base = dict(arms)[a.baseline]
                others = [(nm, g) for nm, g in arms if nm != a.baseline]
                if a.human:
                    others.append(("human", {i: rows[i]["continuation"]}))
                rr.shuffle(others)
                for nm, g in others:
                    pair = [(a.baseline, base[i]), (nm, g[i])]
                    rr.shuffle(pair)
                    n += 1
                    out.append(f"### Item {n}\nPROMPT: {prompt}")
                    for lab, (arm, t) in zip("AB", pair):
                        out.append(f"{lab}: {cut(t)}")
                        key[f"R{r + 1}/{n}/{lab}"] = {"arm": arm, "dev": i}
                    out.append("")
        path = os.path.join(a.out, f"R{r + 1}.txt")
        open(path, "w", encoding="utf-8").write("\n".join(out))
    json.dump(key, open(os.path.join(a.out, "KEY.json"), "w"))
    open(os.path.join(a.out, "key.b64"), "w").write(base64.encodebytes(json.dumps(key).encode()).decode())
    k = len(arms) + (1 if a.human else 0)
    with open(os.path.join(a.out, "RATER_PROMPT.md"), "w", encoding="utf-8") as f:
        for r in range(4):
            packet = os.path.abspath(os.path.join(a.out, f"R{r + 1}.txt"))
            items = open(packet, encoding="utf-8").read().count("### Item ")
            if a.mode == "rate":
                labs = ", ".join(LABELS[:k])
                fields = ", ".join(f'"{x}": s' for x in LABELS[:k])
                f.write(f"## Rater {r + 1}\n\n" + RATE_PROMPT.format(packet=packet, n=items, k=k, labs=labs, fields=fields) + "\n\n")
            else:
                f.write(f"## Rater {r + 1}\n\n" + PAIR_PROMPT.format(packet=packet, n=items) + "\n\n")
    print(f"wrote 4 packets, KEY.json, key.b64 and RATER_PROMPT.md to {a.out}")


def score(a):
    key = json.load(open(os.path.join(a.dir, "KEY.json")))
    rng = random.Random(0)
    first_rater_ids = set()
    if any("choice" in row for row in json.load(open(os.path.join(a.dir, "R1.json")))):
        wins = {}  # (dev, arm) -> list of win values for arm vs baseline
        for r in range(1, 5):
            for row in json.load(open(os.path.join(a.dir, f"R{r}.json"))):
                n = row["item"]
                ka, kb = key[f"R{r}/{n}/A"], key[f"R{r}/{n}/B"]
                arm = ka["arm"] if ka["arm"] != a.baseline else kb["arm"]
                ch = row["choice"]
                v = 0.5 if ch == "tie" else (1.0 if key[f"R{r}/{n}/{ch}"]["arm"] == arm else 0.0)
                wins.setdefault((ka["dev"], arm), []).append((r, v))
                if r <= 2:
                    first_rater_ids.add(ka["dev"])
        arms = sorted({arm for _, arm in wins})
        print(f"pairwise against {a.baseline}: share of wins (tie = 0.5), 95% bootstrap over prompts")
        for arm in arms:
            per = {d: st.mean(v for _, v in vs) for (d, x), vs in wins.items() if x == arm}
            vals = list(per.values())
            bs = sorted(st.mean(rng.choices(vals, k=len(vals))) for _ in range(10000))
            p1 = st.mean(v for (d, x), vs in wins.items() if x == arm for r, v in vs if r <= 2)
            p2 = st.mean(v for (d, x), vs in wins.items() if x == arm for r, v in vs if r > 2)
            print(f"  {arm:12s} {st.mean(vals):.3f} [{bs[250]:.3f}, {bs[9749]:.3f}]  rater pairs {p1:.2f} {p2:.2f}  prompts {len(vals)}")
        return
    score_ = {}
    for r in range(1, 5):
        for row in json.load(open(os.path.join(a.dir, f"R{r}.json"))):
            for lab in LABELS:
                if lab not in row:
                    continue
                k = key[f"R{r}/{row['item']}/{lab}"]
                score_.setdefault((k["dev"], k["arm"]), []).append(row[lab])
                if r <= 2:
                    first_rater_ids.add(k["dev"])
    arms = sorted({arm for _, arm in score_})
    devs = sorted({d for d, _ in score_})
    print("mean score per arm (1-10)")
    for arm in arms:
        v = [x for (d, x2), xs in score_.items() if x2 == arm for x in xs]
        print(f"  {arm:12s} {st.mean(v):.2f}  rated 6+: {sum(x >= 6 for x in v)}/{len(v)}")
    if a.baseline not in arms:
        return
    print(f"each arm minus {a.baseline}, per prompt (raters averaged), 95% bootstrap over prompts")
    for arm in arms:
        if arm in (a.baseline, "human"):
            continue
        diff = {d: st.mean(score_[(d, arm)]) - st.mean(score_[(d, a.baseline)]) for d in devs}
        vals = list(diff.values())
        bs = sorted(st.mean(rng.choices(vals, k=len(vals))) for _ in range(10000))
        h1 = st.mean(v for d, v in diff.items() if d in first_rater_ids)
        h2 = st.mean(v for d, v in diff.items() if d not in first_rater_ids)
        print(f"  {arm:12s} {st.mean(vals):+.3f} [{bs[250]:+.3f}, {bs[9749]:+.3f}]  rater pairs {h1:+.2f} {h2:+.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--out", required=True)
    m.add_argument("--ids", required=True, help="JSON list of GEN-A dev prompt numbers, or {'prompts': [...]}")
    m.add_argument("--prompts", default="bench/eval/gen_prompts_dev.tsv")
    m.add_argument("--arm", action="append", required=True, help="NAME=GEN_FILE (repeatable)")
    m.add_argument("--human", action="store_true", help="add the human continuation as an arm")
    m.add_argument("--mode", choices=["rate", "pair"], default="rate")
    m.add_argument("--baseline", default=None, help="pair mode: the arm every other arm is compared with")
    m.add_argument("--seed", type=int, default=20261105)
    s = sub.add_parser("score")
    s.add_argument("--dir", required=True)
    s.add_argument("--baseline", required=True)
    a = p.parse_args()
    if a.cmd == "make":
        if a.mode == "pair" and not a.baseline:
            sys.exit("pair mode needs --baseline")
        make(a)
    else:
        score(a)


if __name__ == "__main__":
    main()
