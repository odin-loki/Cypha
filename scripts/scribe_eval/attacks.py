"""Adversarial attacks on SCRIBE (PLAN 6.4 V3; docs/SCRIBE_PAPER_PLAN.md section 5.5).

Each attack tries to score high with text a person would rate low, or exposes a blind spot. All are built from
prompt / true-continuation pairs (`scribe prompts`) and are scored with the prompt, so both the open profile (v2)
and the prompted profile (v2p) can be read from one run.

  human           the true continuation (the bar to compare with)
  copy_pack       a human sample cut from the pack, the text SCRIBE was trained on (--copy FILE, `scribe samples`)
  offprompt       the true continuation of another document: fluent, on the wrong topic
  echo            the prompt repeated to the continuation's length
  stuffing        the true continuation with every third word replaced by one of the prompt's rarest words
  generic         one fixed, fluent, contentless paragraph, the same for every prompt ("it depends")
  generic_topic   the same paragraph with the prompt's rarest words put in as the topic (gaming key-term coverage)
  numbers         the true continuation with every number replaced by a random one (a factual error SCRIBE cannot see)
  negation        the true continuation with "not" put after the first is/was/are/were/has/had/can of each sentence
  fwloop          a loop of function words, punctuated as sentences

Usage: attacks.py --in prompts.jsonl --copy pack_samples.jsonl --vocab pack.vocab.tsv --out attacks.jsonl [--seed 5]
Author: Odin Loch
"""
import argparse
import json
import random
import re

GENERIC = ("This is a topic that has many different aspects. It is important to consider the wider context before "
           "reaching any conclusion. Experts have offered a range of views on the question over the years. In general, "
           "the answer depends on the situation and on the people involved. More research will be needed before the "
           "matter can be understood fully.")
GENERIC_TOPIC = ("The topic of {a} has many different aspects. It is important to consider {b} in a wider context before "
                 "reaching any conclusion. Experts have offered a range of views on {a} over the years. In general, the "
                 "answer depends on {c} and on the people involved. More research on {b} will be needed before the matter "
                 "can be understood fully.")
FUNCTION_WORDS = ["the", "of", "and", "to", "in", "a", "is", "that", "for", "it", "as", "was", "with", "on", "by"]
AUX = re.compile(r"\b(is|was|are|were|has|had|can|will|would|could|should|does|did)\b")
SENT = re.compile(r'(?<=[.!?])["\')\]]*\s+(?=["(\[]?[A-Z0-9])')


def rare_words(text, rank, k=3):
    ws = [w for w in re.findall(r"[A-Za-z][A-Za-z'\-]+", text) if len(w) > 3]
    ws = sorted(set(ws), key=lambda w: -rank.get(w.lower(), 10 ** 9))
    return ws[:k] or ["it"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--copy", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
    copies = [json.loads(l)["text"] for l in open(a.copy, encoding="utf-8")]
    rank = {}
    for i, line in enumerate(open(a.vocab, encoding="utf-8")):
        rank[line.split("\t")[0]] = i
    rng = random.Random(a.seed)
    n = len(rows)
    perm = list(range(n))
    rng.shuffle(perm)
    perm = [p if p != i else (i + 1) % n for i, p in enumerate(perm)]
    out = []
    for i, r in enumerate(rows):
        p, t = r["prompt"], r["ref"]
        rw = rare_words(p, rank)
        echo = p
        while len(echo) < len(t):
            echo += " " + p
        toks = t.split()
        stuffed = " ".join(rw[k % len(rw)] if k % 3 == 2 and re.match(r"^[A-Za-z]", w) else w for k, w in enumerate(toks))
        nums = re.sub(r"\d+", lambda m: str(rng.randint(1, 10 ** len(m.group(0)) - 1)), t)
        neg = " ".join(AUX.sub(lambda m: m.group(0) + " not", s, count=1) for s in SENT.split(t))
        loop = []
        while len(" ".join(loop)) < len(t):
            words = [rng.choice(FUNCTION_WORDS) for _ in range(rng.randint(6, 12))]
            loop.append(words[0].capitalize() + " " + " ".join(words[1:]) + ".")
        variants = {
            "human": t,
            "copy_pack": copies[i % len(copies)],
            "offprompt": rows[perm[i]]["ref"],
            "echo": echo,
            "stuffing": stuffed,
            "generic": GENERIC,
            "generic_topic": GENERIC_TOPIC.format(a=rw[0], b=rw[1 % len(rw)], c=rw[2 % len(rw)]),
            "numbers": nums,
            "negation": neg,
            "fwloop": " ".join(loop),
        }
        for k, v in variants.items():
            out.append({"id": f"attack/{k}/{i:04d}", "set": "attack", "system": k, "prompt": p, "text": v, "ref": t})
    with open(a.out, "w", encoding="utf-8") as o:
        for x in out:
            o.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"{len(out)} texts, {len(out) // n} attacks x {n} prompts")


if __name__ == "__main__":
    main()
