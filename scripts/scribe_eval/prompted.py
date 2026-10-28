"""Prompted control ladder for SCRIBE's relevance axis R (docs/SCRIBE_PAPER_PLAN.md section 5.1, prompted form).

From prompt / true-continuation pairs (`scribe prompts`), writes one JSON line per (rung, prompt):
  true       the true human continuation of the prompt
  offprompt  a true human continuation of a different document (fluent, on the wrong topic): the failure the 8K
             count model showed, a fluent answer to a different question
  echo_half  the prompt's own sentences, then the true continuation cut to the same total length (half echo)
  echo       the prompt repeated until it is as long as the true continuation (the question echoed)
Expected order, best first: true > offprompt > echo_half > echo for a relevance-aware metric. (echo_half keeps
real, on-topic text, so placing it below offprompt is a value judgment: copying the prompt is worse than drifting.)

Usage: prompted.py --in prompts.jsonl --out ladder.jsonl [--seed 11]
Author: Odin Loch
"""
import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
    rng = random.Random(a.seed)
    n = len(rows)
    # A derangement across documents: each prompt gets the continuation of a row from another document.
    perm = list(range(n))
    for _ in range(100):
        rng.shuffle(perm)
        if all(rows[i]["meta"]["doc"] != rows[perm[i]]["meta"]["doc"] for i in range(n)):
            break
    for i in range(n):
        j = perm[i]
        while rows[j]["meta"]["doc"] == rows[i]["meta"]["doc"]:
            j = rng.randrange(n)
        perm[i] = j
    with open(a.out, "w", encoding="utf-8") as o:
        for i, r in enumerate(rows):
            p, t = r["prompt"], r["ref"]
            off = rows[perm[i]]["ref"]
            half = (p + " " + t)[: len(t)]
            cut = half.rfind(". ")
            half = half[: cut + 1] if cut > len(p) else half
            echo = p
            while len(echo) < len(t):
                echo = echo + " " + p
            for rung, text in (("true", t), ("offprompt", off), ("echo_half", half), ("echo", echo)):
                o.write(json.dumps({"id": f"{r['id']}/{rung}", "set": r["set"], "system": rung, "prompt": p, "text": text, "ref": t},
                                   ensure_ascii=False) + "\n")
    print(f"{4 * n} texts from {n} prompts")


if __name__ == "__main__":
    main()
