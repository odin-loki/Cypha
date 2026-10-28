"""Question-answering control ladder for SCRIBE's prompted profile (PLAN 6.4 V1, Q/A form), from Dolly 15k.

Dolly rows with a response of 100-1,500 bytes are split by row index into pack (rows 0, 3, 6, ...: two thirds),
reference and test (the rest, alternating). The prompt is the instruction, followed by the context when there is one.
Test rungs, best first:
  correct      the human response
  other        the human response to another test question of the same category (fluent, wrong question)
  echo         the prompt repeated to the response's length (the question echoed)
  shuffled     the response's words shuffled inside each sentence (word salad on the right topic)
Usage: qa_ladder.py --in dolly.jsonl --out DIR [--n 600] [--seed 3]
Writes DIR/pack.jsonl, DIR/ref.jsonl (prompt + text, for --ref-prompts and as the reference), DIR/test.jsonl.
Author: Odin Loch
"""
import argparse
import json
import os
import random
import re


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    rows = []
    for line in open(a.inp, encoding="utf-8"):
        r = json.loads(line)
        resp = " ".join(r.get("response", r.get("text", "")).split())
        if not 100 <= len(resp.encode()) <= 1500:
            continue
        ctx = " ".join((r.get("context") or "").split())
        prompt = " ".join(r["instruction"].split()) + ((" " + ctx) if ctx else "")
        rows.append({"prompt": prompt, "text": resp, "category": r.get("category", "")})
    os.makedirs(a.out, exist_ok=True)
    pack, ref, test = [], [], []
    for i, r in enumerate(rows):
        (pack if i % 3 != 2 else (ref if (i // 3) % 2 == 0 else test)).append(r)
    rng = random.Random(a.seed)
    rng.shuffle(test)
    test = test[: a.n]
    with open(os.path.join(a.out, "pack.jsonl"), "w") as o:
        for i, r in enumerate(pack):
            o.write(json.dumps({"id": f"qa/pack/{i}", "system": "human", "text": r["text"]}) + "\n")
    with open(os.path.join(a.out, "ref.jsonl"), "w") as o:
        for i, r in enumerate(ref):
            o.write(json.dumps({"id": f"qa/ref/{i}", "system": "human", "prompt": r["prompt"], "text": r["text"]}) + "\n")
    bycat = {}
    for i, r in enumerate(test):
        bycat.setdefault(r["category"], []).append(i)
    with open(os.path.join(a.out, "test.jsonl"), "w") as o:
        for i, r in enumerate(test):
            same = [j for j in bycat[r["category"]] if j != i] or [j for j in range(len(test)) if j != i]
            other = test[rng.choice(same)]["text"]
            echo = r["prompt"]
            while len(echo) < len(r["text"]):
                echo += " " + r["prompt"]
            sh = []
            for s in re.split(r"(?<=[.!?])\s+", r["text"]):
                w = s.split()
                if len(w) > 1:
                    tail = re.search(r"[.!?]+$", w[-1])
                    t = tail.group(0) if tail else ""
                    w[-1] = w[-1][: len(w[-1]) - len(t)] if t else w[-1]
                    rng.shuffle(w)
                    w = [x for x in w if x]
                    s = " ".join(w) + t
                    s = s[:1].upper() + s[1:]
                sh.append(s)
            for rung, text in (("correct", r["text"]), ("other", other), ("echo", echo), ("shuffled", " ".join(sh))):
                o.write(json.dumps({"id": f"qa/{rung}/{i:04d}", "set": "qa", "system": rung, "prompt": r["prompt"], "text": text,
                                    "meta": {"category": r["category"]}}) + "\n")
    print(f"{len(rows)} usable rows: pack {len(pack)}, ref {len(ref)}, test {len(test)} (x4 rungs)")


if __name__ == "__main__":
    main()
