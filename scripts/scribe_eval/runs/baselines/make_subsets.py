"""Build the evaluation subsets the coordinator specified (cuts of 2026-10-27 23:30).

Every subset keeps the original lines unchanged (same ids, fields and order of the
source file), so SCRIBE and every baseline score exactly the same texts.
"""
import json
from pathlib import Path

SP = Path("/tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad")
D, R = SP / "data", SP / "run/baselines"
OUT = R / "subsets"
OUT.mkdir(parents=True, exist_ok=True)


def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def write(name, rows, note):
    p = OUT / name
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    assert len({r["id"] for r in rows}) == len(rows), name
    print(f"{name}\t{len(rows)}\t{note}")


def first_per(rows, key, n, keep=None):
    seen, out = {}, []
    for r in rows:
        k = key(r)
        if keep is not None and k not in keep:
            continue
        if seen.get(k, 0) < n:
            out.append(r)
            seen[k] = seen.get(k, 0) + 1
    return out


def earlier_plan():
    # plan of 23:30 (superseded at 23:48)
    # A. fresh Wikipedia ladder
    wf = read(SP / "run/fresh/wiki_fresh/texts.jsonl")
    W = first_per(wf, lambda r: r["system"], 150)
    write("wiki_fresh_W.jsonl", W, "first 150 texts (file order) of each of the 11 rungs")
    write("wiki_fresh_W60.jsonl", first_per(W, lambda r: r["system"], 60), "first 60 of each rung of W (judge)")
    human = [r for r in wf if r["system"] == "human"]
    write("wiki_fresh_mauve_p.jsonl", human[:250], "MAUVE reference p: first 250 human texts (file order)")
    q = human[250:500]
    for rung in ["sent_shuffled", "word_shuffled", "random_lexicon", "random_letters", "repeated_phrase", "trie"]:
        q += [r for r in wf if r["system"] == rung][:250]
    write("wiki_fresh_mauve_q.jsonl", q, "MAUVE q: human texts 250-499 and the first 250 of 6 rungs")

    # B. MAUVE human evaluation: the first 1,000 (and 300) pairs and the union of their texts
    mt = read(D / "mauve_he/texts.jsonl")
    pairs = read(D / "mauve_he/pairs.jsonl")
    for n, name in [(1000, "mauve_he_M.jsonl"), (300, "mauve_he_M300.jsonl")]:
        ids = {x for p in pairs[:n] for x in (p["a"], p["b"])}
        sel = [r for r in mt if r["id"] in ids]
        assert len(sel) == len(ids), (name, len(sel), len(ids))
        write(name, sel, f"texts of the first {n} pairs of pairs.jsonl (texts.jsonl order)")

    # C. held-out corruptions: texts of the first 40 (and 10) distinct sources
    co = read(SP / "run/fresh/corrupt.jsonl")
    srcs = []
    for r in co:
        if r["meta"]["source"] not in srcs:
            srcs.append(r["meta"]["source"])
    for n in (40, 10):
        keep = set(srcs[:n])
        write(f"corrupt_S{n}.jsonl", [r for r in co if r["meta"]["source"] in keep],
              f"all families and strengths of the first {n} sources")

    # D. GPT-2 output dataset samples (coordinator's run/gpt2out/samples.jsonl)
    gs = read(SP / "run/gpt2out/samples.jsonl")
    g500 = first_per(gs, lambda r: r["system"], 500)
    write("gpt2out_500.jsonl", g500, "first 500 of each of the 9 systems")
    gh = [r for r in g500 if r["system"] == "human"]
    write("gpt2out_mauve_p.jsonl", gh[:250], "MAUVE reference p: first 250 human (WebText)")
    write("gpt2out_mauve_q.jsonl", [r for r in g500 if r["system"] != "human"] + gh[250:500],
          "MAUVE q: each model's 500 and human texts 250-499")




# ---- minimal plan of 23:48 (90-minute cap) ----
def minimal():
    co = read(SP / "run/fresh/corrupt.jsonl")
    srcs = []
    for r in co:
        if r["meta"]["source"] not in srcs:
            srcs.append(r["meta"]["source"])
    keep = set(srcs[:20])
    write("corrupt_S20.jsonl", [r for r in co if r["meta"]["source"] in keep],
          "all families and strengths of the first 20 sources")
    at = read(SP / "run/attacks/attacks.jsonl")
    write("attacks_50.jsonl", [r for r in at if int(r["id"].rsplit("/", 1)[1]) < 50],
          "the first 50 prompts of each attack (ids attack/<name>/0000-0049)")
    gs = read(SP / "run/gpt2out/samples.jsonl")
    write("gpt2out_200.jsonl", first_per(gs, lambda r: r["system"], 200), "first 200 of each of the 9 systems")
    W = read(OUT / "wiki_fresh_W.jsonl")
    hw = [r for r in W if r["system"] == "human"]
    write("wiki_fresh_W_mauve_p.jsonl", hw[:100], "MAUVE reference p: first 100 human texts of W")
    q = hw[100:150]
    for rung in ["sent_shuffled", "word_shuffled", "random_lexicon", "random_letters", "repeated_phrase", "trie"]:
        q += [r for r in W if r["system"] == rung][:100]
    write("wiki_fresh_W_mauve_q.jsonl", q, "MAUVE q: next 50 human of W and the first 100 of 6 rungs of W")


if __name__ == "__main__":
    import sys
    minimal() if "minimal" in sys.argv else earlier_plan()
