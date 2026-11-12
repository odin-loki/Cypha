#!/usr/bin/env python3
"""skf_read.py: K5_SKELFREE screen helpers.
  skf_read.py show IDS K1.gen ARM.gen ...   -- first 460 chars of each GEN for the given dev ids, side by side
  skf_read.py stats K1.gen ARM.gen ...      -- samples unlike K1 (all, rated 50); ended-sentence share in 460 chars
"""
import json, re, sys

def gens(path):
    out, cur = {}, None
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        if line.startswith("PROMPT\t"):
            cur = int(line.split("\t")[1])
        elif line.startswith("GEN\t") and cur is not None:
            parts = line.split("\t")
            out[cur] = parts[-1]
    return out

def unesc(s):
    return s.encode("utf-8").decode("unicode_escape", errors="ignore") if "\\" in s else s

def ended_share(t):
    t = t[:460]
    # sentences: split at . ! ? followed by space
    parts = re.split(r"(?<=[.!?])\s+", t.strip())
    if not parts: return 0, 0
    n = len(parts)
    ended = sum(1 for p in parts if p.rstrip().endswith((".", "!", "?")))
    return ended, n

def main():
    cmd = sys.argv[1]
    if cmd == "show":
        ids = [int(x) for x in sys.argv[2].split(",")]
        files = sys.argv[3:]
        G = [gens(f) for f in files]
        for i in ids:
            print(f"===== dev {i}")
            for f, g in zip(files, G):
                print(f"--- {f.split('/')[-1]}:\n{g.get(i, '')[:460]}\n")
    elif cmd == "stats":
        rated = set(json.load(open("bench/blind/llm_rating/prompts.json"))["prompts"])
        base = gens(sys.argv[2])
        for f in sys.argv[2:]:
            g = gens(f)
            diff = sum(1 for i in g if g[i] != base.get(i))
            diffr = sum(1 for i in rated if g.get(i) != base.get(i))
            e = n = 0
            for i in g:
                a, b = ended_share(g[i]); e += a; n += b
            print(f"{f.split('/')[-1]}\tsamples {len(g)}\tunlike K1 {diff} (rated {diffr} of 50)\tended sentences in 460 chars {e}/{n} = {e/max(n,1):.3f}")

main()
