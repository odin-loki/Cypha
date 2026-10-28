"""Held-out corruption families for the SCRIBE v2 dose-response test (docs/SCRIBE_PAPER_PLAN.md section 5.2).

None of these families was used to design SCRIBE v2; they are test data for every metric. Each family damages a
human sample at strengths 0, 10, 25, 50, 75 and 100 percent (strength 0 is the untouched sample), so a good metric
should fall as the strength rises.

Families
  subst     replace that share of word tokens by frequency-weighted random words from the pack vocabulary
  swap      swap disjoint adjacent word pairs inside sentences, each with probability = the strength
  reorder   permute that share of the sentence positions of the sample (100% = a full shuffle of its own sentences)
  repeat    replace that share of sentences (after the first) by a copy of an earlier sentence
  nounswap  replace that share of content words by content words drawn from other samples
  typo      give that share of words one random character edit (insert, delete, substitute or transpose)
  truncate  cut that share of sentences after half their words (the terminal punctuation is kept)
  offtopic  replace that share of sentences by sentences from other samples
  dropword  delete that share of words

Usage
  corrupt.py --in human.jsonl --vocab pack.vocab.tsv --out corrupt.jsonl [--n 200] [--seed 7]
  human.jsonl: JSON lines with "id" and "text" (e.g. the human rung of `scribe ladder --export`).
  pack.vocab.tsv: "word<TAB>count" lines (ppmi_svd's vocabulary of the pack), for frequency-weighted draws.
Output lines: {"id", "set": "corrupt", "system": "<family>@<strength>", "text", "meta": {"family", "strength", "source"}}.

Author: Odin Loch
"""
import argparse
import bisect
import json
import random
import re
import zlib

STRENGTHS = [0, 10, 25, 50, 75, 100]
FAMILIES = ["subst", "swap", "reorder", "repeat", "nounswap", "typo", "truncate", "offtopic", "dropword"]

SENT_RE = re.compile(r'(?<=[.!?])["\')\]]*\s+(?=["(\[]?[A-Z0-9])')
WORD_RE = re.compile(r"^([^A-Za-z]*)([A-Za-z][A-Za-z'\-]*)([^A-Za-z]*)$")


def sentences(text):
    return [s for s in SENT_RE.split(text) if s.strip()]


def pick_k(n, share, rng):
    k = round(n * share / 100.0)
    return set(rng.sample(range(n), k)) if k > 0 else set()


def keep_case(src, word):
    if src[:1].isupper() and word:
        return word[0].upper() + word[1:]
    return word


class Vocab:
    """Frequency-weighted draws from a word list; the top 200 words are treated as function words."""

    def __init__(self, path):
        self.words, cum, total = [], [], 0
        for line in open(path, encoding="utf-8"):
            w, c = line.rstrip("\n").split("\t")
            if not w.isalpha():
                continue
            total += float(c)
            self.words.append(w)
            cum.append(total)
        self.cum, self.total = cum, total
        self.function = set(self.words[:200])

    def draw(self, rng):
        return self.words[bisect.bisect_right(self.cum, rng.random() * self.total)]


def split_token(tok):
    m = WORD_RE.match(tok)
    return (m.group(1), m.group(2), m.group(3)) if m else None


def subst(text, s, rng, vocab, pool):
    toks = text.split()
    idx = [i for i, t in enumerate(toks) if split_token(t)]
    for i in pick_k(len(idx), s, rng):
        pre, core, post = split_token(toks[idx[i]])
        toks[idx[i]] = pre + keep_case(core, vocab.draw(rng)) + post
    return " ".join(toks)


def swap(text, s, rng, vocab, pool):
    out = []
    for sent in sentences(text):
        toks = sent.split()
        if len(toks) < 3:
            out.append(sent)
            continue
        tail = ""
        m = re.search(r'[.!?]["\')\]]*$', toks[-1])
        if m:
            tail = toks[-1][m.start():]
            toks[-1] = toks[-1][: m.start()]
        # Disjoint pairs, left to right: each pair (i, i + 1) is swapped with probability s, then both are skipped.
        # (Swapping overlapping pairs in sequence would rotate the sentence instead of damaging it.)
        i = 0
        while i + 1 < len(toks):
            if rng.random() < s / 100.0:
                toks[i], toks[i + 1] = toks[i + 1], toks[i]
                i += 2
            else:
                i += 1
        toks = [t for t in toks if t]
        if toks:
            toks[0] = toks[0][:1].upper() + toks[0][1:]
        out.append(" ".join(toks) + tail)
    return " ".join(out)


def reorder(text, s, rng, vocab, pool):
    ss = sentences(text)
    pos = sorted(pick_k(len(ss), s, rng))
    moved = [ss[p] for p in pos]
    if len(moved) > 1:
        for _ in range(10):
            rng.shuffle(moved)
            if any(moved[i] != ss[p] for i, p in enumerate(pos)):
                break
    for i, p in enumerate(pos):
        ss[p] = moved[i]
    return " ".join(ss)


def repeat(text, s, rng, vocab, pool):
    ss = sentences(text)
    if len(ss) < 2:
        return text
    for p in sorted(pick_k(len(ss) - 1, s, rng)):
        ss[p + 1] = ss[rng.randrange(0, p + 1)]
    return " ".join(ss)


def nounswap(text, s, rng, vocab, pool):
    toks = text.split()
    idx = [i for i, t in enumerate(toks) if split_token(t) and split_token(t)[1].lower() not in vocab.function and len(split_token(t)[1]) > 2]
    for i in pick_k(len(idx), s, rng):
        pre, core, post = split_token(toks[idx[i]])
        toks[idx[i]] = pre + keep_case(core, rng.choice(pool["content"])) + post
    return " ".join(toks)


def typo(text, s, rng, vocab, pool):
    toks = text.split()
    idx = [i for i, t in enumerate(toks) if split_token(t) and len(split_token(t)[1]) >= 3]
    letters = "abcdefghijklmnopqrstuvwxyz"
    for i in pick_k(len(idx), s, rng):
        pre, core, post = split_token(toks[idx[i]])
        c = list(core)
        op = rng.randrange(4)
        j = rng.randrange(len(c))
        if op == 0:
            c.insert(j, rng.choice(letters))
        elif op == 1 and len(c) > 1:
            del c[j]
        elif op == 2:
            c[j] = rng.choice(letters)
        elif len(c) > 1:
            j = min(j, len(c) - 2)
            c[j], c[j + 1] = c[j + 1], c[j]
        toks[idx[i]] = pre + "".join(c) + post
    return " ".join(toks)


def truncate(text, s, rng, vocab, pool):
    ss = sentences(text)
    for p in pick_k(len(ss), s, rng):
        toks = ss[p].split()
        if len(toks) >= 4:
            m = re.search(r'[.!?]["\')\]]*$', toks[-1])
            end = toks[-1][m.start():] if m else "."
            cut = toks[: len(toks) // 2]
            cut[-1] = re.sub(r"[,;:]+$", "", cut[-1])
            ss[p] = " ".join(cut) + end
    return " ".join(ss)


def offtopic(text, s, rng, vocab, pool):
    ss = sentences(text)
    for p in pick_k(len(ss), s, rng):
        ss[p] = rng.choice(pool["sents"])
    return " ".join(ss)


def dropword(text, s, rng, vocab, pool):
    out = []
    for sent in sentences(text):
        toks = sent.split()
        drop = pick_k(len(toks), s, rng)
        kept = [t for i, t in enumerate(toks) if i not in drop]
        if not kept:
            kept = toks[-1:]
        if not re.search(r'[.!?]["\')\]]*$', kept[-1]):
            kept[-1] = kept[-1] + "."
        kept[0] = kept[0][:1].upper() + kept[0][1:]
        out.append(" ".join(kept))
    return " ".join(out)


FUNCS = {f: globals()[f] for f in FAMILIES}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
    rows = [r for r in rows if r.get("text")][: a.n]
    vocab = Vocab(a.vocab)
    rng = random.Random(a.seed)
    # Pools for nounswap and offtopic come from the same samples; each sample draws from the others' sentences.
    all_sents = [(i, s) for i, r in enumerate(rows) for s in sentences(r["text"])]
    content = [w for r in rows for t in r["text"].split() for w in [split_token(t)] if w and w[1].lower() not in vocab.function and len(w[1]) > 2]
    n = 0
    with open(a.out, "w", encoding="utf-8") as o:
        for i, r in enumerate(rows):
            pool = {"sents": [s for j, s in all_sents if j != i], "content": [c[1].lower() for c in content]}
            for fam in FAMILIES:
                for s in STRENGTHS:
                    rr = random.Random(zlib.crc32(f"{a.seed}/{i}/{fam}/{s}".encode()))
                    t = r["text"] if s == 0 else FUNCS[fam](r["text"], s, rr, vocab, pool)
                    o.write(json.dumps({"id": f"corrupt/{fam}@{s}/{i:04d}", "set": "corrupt", "system": f"{fam}@{s}", "text": t,
                                        "meta": {"family": fam, "strength": s, "source": r["id"]}}, ensure_ascii=False) + "\n")
                    n += 1
    print(f"{n} texts from {len(rows)} samples, {len(FAMILIES)} families x {len(STRENGTHS)} strengths")


if __name__ == "__main__":
    main()
