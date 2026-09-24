"""Shared LM metrics over dumped next-byte distributions.

A dump is float32 natural-log probabilities, 256 per byte (row i predicts
text[i]); `cyphalm_lm_quality --dump-dist` and `byte_lm.py eval --dump` both
write this layout, so every model is scored by the same code.
"""
import math
import re

import numpy as np

LN2 = math.log(2)


def load_dump(path, n):
    lp = np.fromfile(path, dtype=np.float32).reshape(-1, 256)
    if lp.shape[0] != n:
        raise ValueError(f"{path}: {lp.shape[0]} rows, expected {n}")
    return lp.astype(np.float64)


def byte_class(b):
    if 97 <= b <= 122:
        return "lower"
    if 65 <= b <= 90:
        return "upper"
    if 48 <= b <= 57:
        return "digit"
    if b in (32, 10, 9, 13):
        return "space"
    if b >= 128:
        return "non-ascii"
    return "punct/markup"


def words(text):
    """Spans [s, e) of alphabetic words, each with the byte that ends it."""
    raw = np.asarray(text, dtype=np.uint8).tobytes()
    return [(m.start(), m.end()) for m in re.finditer(rb"[A-Za-z]+", raw) if m.end() < len(raw)]


def score(lp, text, pos_buckets=(0, 256, 1024, 4096, 16384, 1 << 30)):
    text = np.asarray(text, dtype=np.int64)
    n = len(text)
    idx = np.arange(n)
    lpt = lp[idx, text]
    bits = -lpt / LN2
    p = np.exp(lp)
    order = np.argsort(-lp, axis=1)
    top1 = order[:, 0] == text
    top5 = (order[:, :5] == text[:, None]).any(1)
    pmax = p.max(1)
    ent = -(p * np.where(p > 0, lp, 0)).sum(1) / LN2
    # ECE of top-1 confidence, 10 bins (as cyphalm_lm_quality).
    bins = np.minimum(9, (pmax * 10).astype(int))
    ece = sum(abs(pmax[bins == k].mean() - top1[bins == k].mean()) * (bins == k).mean()
              for k in range(10) if (bins == k).any())
    temps = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    tnll = {}
    for t in temps:
        z = lp / t
        z = z - z.max(1, keepdims=True)
        tnll[t] = float((-(z[idx, text] - np.log(np.exp(z).sum(1)))).mean() / LN2)
    by_pos = {}
    for a, b in zip(pos_buckets, pos_buckets[1:]):
        if a < n:
            by_pos[f"{a}-{min(b, n)}"] = float(bits[a:b].mean())
    classes = np.array([byte_class(int(b)) for b in text])
    by_class = {c: {"share": float((classes == c).mean()), "bits": float(bits[classes == c].mean())}
                for c in sorted(set(classes))}
    # Next-word greedy accuracy: greedy decoding from the true prefix produces
    # the true word (and its terminator) iff every byte is the argmax.
    ws = words(text)
    word_ok = [bool(top1[s:e + 1].all()) for s, e in ws]
    word_first = [bool(top1[s]) for s, e in ws]
    return {
        "bytes": n,
        "bits_per_byte": float(bits.mean()),
        "perplexity_per_byte": float(2 ** bits.mean()),
        "word_perplexity": float(2 ** (bits.sum() / max(1, len(ws)))),
        "top1": float(top1.mean()),
        "top5": float(top5.mean()),
        "mean_entropy_bits": float(ent.mean()),
        "ece_top1": float(ece),
        "best_temperature": min(tnll, key=tnll.get),
        "bits_at_best_temperature": min(tnll.values()),
        "bits_by_position": by_pos,
        "bits_by_class": by_class,
        "next_word_greedy_acc": float(np.mean(word_ok)) if ws else None,
        "word_first_byte_top1": float(np.mean(word_first)) if ws else None,
        "words": len(ws),
    }


def distinct_ngrams(s, n=4):
    g = [s[i:i + n] for i in range(len(s) - n + 1)]
    return len(set(g)) / max(1, len(g))


def copy_rate(prompt, comp, n=12):
    """Share of n-byte windows of the continuation already seen earlier."""
    full = prompt + comp
    hits = 0
    tot = 0
    for i in range(len(prompt), len(full) - n + 1):
        tot += 1
        if full[i:i + n] in full[:i + n - 1]:  # an occurrence starting before i
            hits += 1
    return hits / max(1, tot)


def word_validity(comp, vocab):
    ws = [w.lower() for w in re.findall(r"[A-Za-z]+", comp)]
    if not ws:
        return None
    return sum(w in vocab for w in ws) / len(ws)
