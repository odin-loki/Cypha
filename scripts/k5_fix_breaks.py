#!/usr/bin/env python3
"""K5_FIXES (F1): breaks marked blind, as X1 marked them (docs/reports/K5_FIXES.md, "Breaks"). Odin Loch, 2026-11-13.

  make   k5_fix_breaks.py make --out DIR --seed S ARM=GEN [ARM=GEN ...]
         The 30 prompts of WHERE_IT_BREAKS (bench/where_it_breaks/items.tsv), each arm's first 600 bytes, shuffled
         with no labels into DIR/READ.md (the reader's one file). DIR/key.json maps each text id to its arm and prompt.
  score  k5_fix_breaks.py score --dir DIR --baseline T8
         Reads DIR/marks.json (the reader's reply) and DIR/key.json. Prints breaks per 100 words (words as X1 counts
         them), the count capped at 3 a text, whole-sentence share and breaks by kind, each arm against the baseline
         paired by prompt with a 95% bootstrap interval. Writes DIR/marks.tsv and DIR/summary.md.
"""
import argparse
import json
import random
import re
import sys

IDS = [0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 28, 29, 31, 32, 33, 35, 36]
TOKRE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*|[0-9]+(?:[.,][0-9]+)*|\n|[^\s]")  # X1's word tokens (logprob_map.py)


def unesc(s):
    return s.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")


def read_gen(path):
    out, cur = {}, None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("PROMPT\t"):
                cur = int(line.split("\t")[1])
            elif line.startswith("GEN\t") and cur is not None and cur not in out:
                out[cur] = unesc(line.split("\t", 2)[2])
    return out


def first_bytes(t, n=600):
    return t.encode("utf-8")[:n].decode("utf-8", "ignore")


def prompts():
    p = {}
    with open("bench/eval/gen_prompts_dev.tsv", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            p[i] = line.rstrip("\n").split("\t")[2]
    return p


INSTR = """# Reading for broken sentences

You are a careful reader of English. Below are {n} short texts. Each continues a prompt (shown, cut to its last
part). Read each text on its own merits. Do not compare texts with each other. Some texts continue the same prompt.

For each text:
1. Count its sentences. A sentence ends at . ! or ? (or at the end of the text). Count the last, cut-off one too.
2. In each sentence that stops working, mark the **first word where it stops working**. One mark per broken sentence,
   at most. A sentence that works gets no mark.
   - **grammar**: from this word the clause is abandoned or cannot be parsed as English.
   - **sense**: the sentence is grammatical, but this word makes it false, absurd or self-contradictory
     (for example "Auckland is a town located *throughout* the animal kingdom").
   - Do not mark a sentence only because it is cut off by the end of the text, or only for a wrong fact you cannot
     know is wrong. Do mark text that runs words together, repeats itself senselessly, or changes subject mid-clause.
3. For each mark give the marked word exactly as written, the (up to) four words just before it exactly as written,
   and the kind.

Reply with JSON only, in this form, one entry per text, in the order given:

{{"items": [{{"id": "t001", "sentences": 5, "breaks": [{{"before": "the four words before", "word": "marked", "kind": "grammar"}}]}}]}}

"""


def make(a):
    P = prompts()
    arms = []
    for spec in a.arm:
        name, path = spec.split("=", 1)
        arms.append((name, read_gen(path)))
    items = [(name, i, first_bytes(g[i])) for name, g in arms for i in IDS]
    rng = random.Random(a.seed)
    rng.shuffle(items)
    key, body = {}, [INSTR.format(n=len(items))]
    for k, (name, i, t) in enumerate(items):
        tid = f"t{k + 1:03d}"
        key[tid] = {"arm": name, "prompt": i}
        pr = " ".join(P[i].split())
        pr = pr if len(pr) <= 300 else "…" + pr[-300:]
        body.append(f"## {tid}\n\nPrompt: {pr}\n\nText:\n\n> " + " ".join(t.split()) + "\n")
    import os
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/READ.md", "w", encoding="utf-8") as f:
        f.write("\n".join(body))
    with open(f"{a.out}/key.json", "w") as f:
        json.dump({"seed": a.seed, "key": key}, f, indent=1)
    print(f"{len(items)} texts -> {a.out}/READ.md")


def boot(d, n=10000, seed=1):
    r = random.Random(seed)
    m = sum(d) / len(d)
    bs = sorted(sum(r.choice(d) for _ in d) / len(d) for _ in range(n))
    return m, bs[int(0.025 * n)], bs[int(0.975 * n) - 1]


def score(a):
    key = json.load(open(f"{a.dir}/key.json"))["key"]
    marks = json.load(open(f"{a.dir}/marks.json"))
    texts = {}
    for line in open(f"{a.dir}/READ.md", encoding="utf-8"):
        pass
    body = open(f"{a.dir}/READ.md", encoding="utf-8").read()
    for m in re.finditer(r"^## (t\d+)\n\nPrompt: .*?\n\nText:\n\n> (.*?)\n", body, re.M | re.S):
        texts[m.group(1)] = m.group(2)
    per = {}  # (arm, prompt) -> dict
    rows, unfound = [], 0
    for it in marks["items"]:
        tid = it["id"]
        k = key[tid]
        t = texts[tid]
        nw = len(TOKRE.findall(t))
        br = it.get("breaks", [])
        for b in br:
            found = (b["before"] + " " + b["word"]).strip() in t or b["word"] in t
            unfound += not found
            rows.append((tid, k["arm"], k["prompt"], b["kind"], b["before"], b["word"], int(found)))
        per[(k["arm"], k["prompt"])] = {"words": nw, "breaks": len(br), "cap3": min(3, len(br)), "sents": max(int(it["sentences"]), len(br)),
                                        "grammar": sum(b["kind"] == "grammar" for b in br), "sense": sum(b["kind"] == "sense" for b in br)}
    with open(f"{a.dir}/marks.tsv", "w", encoding="utf-8") as f:
        f.write("id\tarm\tprompt\tkind\tbefore\tword\tfound\n")
        for r in sorted(rows, key=lambda r: (r[1], r[2])):
            f.write("\t".join(str(x).replace("\t", " ") for x in r) + "\n")
    arms = sorted({k for k, _ in per}, key=lambda x: (x != a.baseline, x))
    out = ["| Arm | texts | words | breaks | per 100 words | arm − baseline [95%] | capped at 3 a text | whole sentences | grammar / sense |",
           "|---|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        ps = [i for i in IDS if (arm, i) in per and (a.baseline, i) in per]
        A = [per[(arm, i)] for i in ps]
        W, Bk = sum(x["words"] for x in A), sum(x["breaks"] for x in A)
        S = sum(x["sents"] for x in A)
        rate = 100 * Bk / W
        # paired difference of per-text rates (breaks per 100 words), bootstrapped over prompts
        d = [100 * per[(arm, i)]["breaks"] / per[(arm, i)]["words"] - 100 * per[(a.baseline, i)]["breaks"] / per[(a.baseline, i)]["words"] for i in ps]
        m, lo, hi = boot(d)
        diff = "—" if arm == a.baseline else f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]"
        out.append(f"| {arm} | {len(A)} | {W} | {Bk} | {rate:.2f} | {diff} | {sum(x['cap3'] for x in A)} | {(S - Bk) / S:.3f} ({S - Bk} of {S}) | "
                   f"{sum(x['grammar'] for x in A)} / {sum(x['sense'] for x in A)} |")
    out.append(f"\nMarks whose words were not found in the text as quoted: {unfound} of {len(rows)}.")
    s = "\n".join(out)
    open(f"{a.dir}/summary.md", "w").write(s + "\n")
    print(s)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("make")
    m.add_argument("--out", required=True)
    m.add_argument("--seed", type=int, default=20261213)
    m.add_argument("arm", nargs="+")
    s = sp.add_parser("score")
    s.add_argument("--dir", required=True)
    s.add_argument("--baseline", default="T8")
    a = ap.parse_args()
    make(a) if a.cmd == "make" else score(a)


if __name__ == "__main__":
    sys.exit(main())
