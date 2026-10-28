"""Pair-aware pilot subset of the probe set (scratch; described in mined/probe/README.md).
Keeps probe.txt order and text exactly. Keeps: every word, phrase and title; both sides of up to
PER_CLASS seeded pairs per class of classes.tsv; the first EXTRA held-out enwik8 sentences in probe order.
Writes probe_pilot.txt, pos_pairs_pilot.tsv and classes_pilot.tsv (rows whose two sides are both kept).
Usage: python -I pilot_subset.py PROBE_DIR PER_CLASS EXTRA [SEED] [MAX_EACH_WORD_PHRASE_TITLE] [SUFFIX] [BASE]
With BASE (e.g. pilot), pairs are drawn from classes_BASE.tsv and items only from probe_BASE.txt, so the result nests."""
import collections
import random
import sys

d, per_class, extra = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 20261028
cap = int(sys.argv[5]) if len(sys.argv) > 5 else 0
suffix = sys.argv[6] if len(sys.argv) > 6 else "pilot"
base = sys.argv[7] if len(sys.argv) > 7 else None
rng = random.Random(seed)
texts = open(f"{d}/probe.txt", encoding="utf-8").read().split("\n")[:-1]
meta = [ln.rstrip("\n").split("\t") for ln in open(f"{d}/probe_meta.tsv", encoding="utf-8")][1:]
if base:
    allowed = set(open(f"{d}/probe_{base}.txt", encoding="utf-8").read().split("\n")[:-1])
    texts, meta = map(list, zip(*[(t, m) for t, m in zip(texts, meta) if t in allowed]))
keep = set()
taken = collections.Counter()
for t, m in zip(texts, meta):
    if m[1] in ("word", "phrase", "title") and (not cap or taken[m[1]] < cap):
        keep.add(t)
        taken[m[1]] += 1
cls = [ln.rstrip("\n").split("\t") for ln in open(f"{d}/classes{'_' + base if base else ''}.tsv", encoding="utf-8")][1:]
by = collections.defaultdict(list)
for r in cls:
    by[r[2]].append(r)
chosen = []
for c in sorted(by):
    rows = by[c]
    chosen += rows if len(rows) <= per_class else rng.sample(rows, per_class)
for r in chosen:
    keep.add(r[0])
    keep.add(r[1])
n_extra = 0
for t, m in zip(texts, meta):
    if n_extra >= extra:
        break
    if m[1] == "sentence" and m[2] == "enwik8_heldout" and t not in keep:
        keep.add(t)
        n_extra += 1
out = [t for t in texts if t in keep]
with open(f"{d}/probe_{suffix}.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
chosen_set = {tuple(r[:3]) for r in chosen}
with open(f"{d}/classes_{suffix}.tsv", "w", encoding="utf-8") as f:
    f.write("a\tb\tclass\tsplit\n")
    for r in cls:
        if tuple(r[:3]) in chosen_set:
            f.write("\t".join(r) + "\n")
pos = [ln.rstrip("\n").split("\t") for ln in open(f"{d}/pos_pairs.tsv", encoding="utf-8")][1:]
n_pos = 0
with open(f"{d}/pos_pairs_{suffix}.tsv", "w", encoding="utf-8") as f:
    f.write("anchor\tpositive\tsource\tsplit\n")
    for r in pos:
        if r[0] in keep and r[1] in keep:
            f.write("\t".join(r) + "\n")
            n_pos += 1
kinds = collections.Counter(m[1] + "/" + m[2] for t, m in zip(texts, meta) if t in keep)
print(f"items {len(out)}; class pairs {len(chosen)}; positive pairs {n_pos}; extra wiki sentences {n_extra}")
print(dict(sorted(kinds.items())))
