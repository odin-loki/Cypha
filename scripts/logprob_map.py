#!/usr/bin/env python3
"""Where today's writer breaks (docs/reports/WHERE_IT_BREAKS.md). Odin Loch, 2026-11-13.

  logprob_map.py items --ids IDS --out ITEMS.tsv --t8 T8.gen.txt [--mr MR.gen.txt] [--prompts gen_prompts_dev.tsv]
      the probe's input: per prompt id, the human continuation, T8's text and (if given) D4's MR text
  logprob_map.py report --probe PREFIX --marks MARKS.tsv --out DIR
      reads k5_probe's map, gate and beam files (and the broken spots marked by reading) and writes the tables and
      the annotated example maps the report quotes (DIR/tables.md, DIR/examples.md, DIR/numbers.json)

The texts are cut at 600 bytes (T8's own length; the human text at the last space before 600 bytes).
"""
import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict

CUT = 600


def unesc_tsv(s):
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            n = s[i + 1]
            out.append({"n": "\n", "t": "\t", "r": "\r", "\\": "\\"}.get(n, n))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def unprintable(s):
    """packed_trie.hpp printable(): \\n and \\xHH escapes (bytes); returns str (utf-8, errors kept as surrogates)."""
    b = bytearray()
    i = 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s) and s[i + 1] == "n":
            b += b"\n"
            i += 2
        elif s[i] == "\\" and i + 3 < len(s) and s[i + 1] == "x":
            b.append(int(s[i + 2:i + 4], 16))
            i += 4
        else:
            b += s[i].encode("utf-8")
            i += 1
    return bytes(b).decode("utf-8", errors="surrogateescape")


def esc(s):
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")


def read_gen(path):
    gens, pid = {}, None
    with open(path, encoding="utf-8", errors="surrogateescape") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("PROMPT\t"):
                pid = int(line.split("\t")[1])
            elif line.startswith("GEN\t") and pid is not None:
                gens[pid] = unprintable(line.split("\t", 2)[2])
    return gens


def cut_human(t):
    b = t.encode("utf-8", errors="surrogateescape")
    if len(b) <= CUT:
        return t
    s = b[:CUT].decode("utf-8", errors="ignore")
    j = s.rfind(" ")
    return s[:j] if j > 0 else s


def cmd_items(a):
    ids = [int(x) for x in a.ids.split(",")]
    prompts = {}
    with open(a.prompts, encoding="utf-8", errors="surrogateescape") as f:
        next(f)
        for i, line in enumerate(f):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                prompts[i] = (unesc_tsv(p[2]), " " + unesc_tsv(p[3]))
    t8 = read_gen(a.t8)
    mr = read_gen(a.mr) if a.mr else {}
    with open(a.out, "w", encoding="utf-8", errors="surrogateescape") as f:
        for i in ids:
            pr, hu = prompts[i]
            f.write(f"{i}\thuman\t{esc(pr)}\t{esc(cut_human(hu))}\n")
            f.write(f"{i}\tT8\t{esc(pr)}\t{esc(t8[i])}\n")
            if i in mr:
                f.write(f"{i}\tMR\t{esc(pr)}\t{esc(mr[i])}\n")
    print(f"{len(ids)} prompts -> {a.out}")


# ---------------------------------------------------------------- report

def read_tsv(path):
    with open(path, encoding="utf-8", errors="surrogateescape") as f:
        head = f.readline().rstrip("\n").split("\t")
        rows = []
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(head):
                p += [""] * (len(head) - len(p))
            rows.append(dict(zip(head, p)))
    return rows


NUM = {"lp_full", "lp_k5", "lp_top1", "nuc_before", "lp_t", "lp_mix", "lp_prior"}
INT = {"wi", "sent", "unk", "ctx_ord", "ans_ord", "ov_ctx", "ov_ans", "rank", "ncand", "in_cand", "rep_b", "cap_b", "nuc_k", "nuc_in", "has_mix"}


def typed(rows):
    for r in rows:
        for k in NUM:
            if k in r:
                r[k] = float(r[k])
        for k in INT:
            if k in r:
                r[k] = int(r[k])
        r["tok"] = unesc_tsv(r["tok"])
        r["top1"] = unesc_tsv(r.get("top1", ""))
    return rows


def boot_mean(xs, n=2000, seed=20261113):
    if not xs:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    m = sum(xs) / len(xs)
    bs = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return m, bs[int(0.025 * n)], bs[int(0.975 * n) - 1]


def pct(a, b):
    return f"{a / b:.3f}" if b else "—"


def is_word(t):
    return bool(re.match(r"[A-Za-z0-9]", t))


STOP = {".", "!", "?", "\n"}


def cmd_report(a):
    rows = typed(read_tsv(a.probe + ".map.tsv"))
    gate = read_tsv(a.probe + ".gate.tsv") if os.path.exists(a.probe + ".gate.tsv") else []
    beam = read_tsv(a.probe + ".beam.tsv") if os.path.exists(a.probe + ".beam.tsv") else []
    info = read_tsv(a.probe + ".info.tsv")
    marks = []
    if a.marks and os.path.exists(a.marks):
        marks = [r for r in read_tsv(a.marks)]
    os.makedirs(a.out, exist_ok=True)
    by = defaultdict(list)  # (item, kind) -> rows
    for r in rows:
        by[(r["item"], r["kind"])].append(r)
    kinds = [k for k in ("human", "T8", "MR") if any(kk == k for (_, kk) in by)]
    num = {}
    T = []

    def emit(s=""):
        T.append(s)

    # 1. per kind: the whole-text picture
    emit("## Table 1. Every word, by text (30 prompts; first 600 bytes)\n")
    emit("| | human | T8 | MR |")
    emit("|---|---|---|---|")
    stats = {}
    for k in kinds:
        R = [r for (i, kk), v in by.items() if kk == k for r in v]
        n = len(R)
        lp = [r["lp_full"] for r in R if not r["unk"]]
        s = {
            "words": n,
            "bits_mean": -sum(lp) / max(len(lp), 1),
            "unk": sum(r["unk"] for r in R) / n,
            "ans5": sum(r["ans_ord"] == 5 for r in R) / n,
            "ans4": sum(r["ans_ord"] == 4 for r in R) / n,
            "ans3": sum(r["ans_ord"] == 3 for r in R) / n,
            "ans2": sum(r["ans_ord"] == 2 for r in R) / n,
            "ans1": sum(r["ans_ord"] == 1 for r in R) / n,
            "ctx5": sum(r["ctx_ord"] == 5 for r in R) / n,
            "ov_ans": sum(r["ov_ans"] > 0 for r in R) / n,
            "ov_gain": sum(r["lp_full"] - r["lp_k5"] for r in R if not r["unk"]) / max(len(lp), 1),
            "nuc_in": sum(r["nuc_in"] for r in R) / n,
            "rank1": sum(r["rank"] == 1 for r in R) / n,
            "rank_le10": sum(r["rank"] <= 10 for r in R) / n,
            "tail": sum(r["lp_full"] < -10 for r in R if not r["unk"]) / n,
            "p_lt_01": sum(r["lp_full"] < math.log2(0.01) for r in R if not r["unk"]) / n,
            "nuc_k_med": sorted(r["nuc_k"] for r in R)[n // 2],
            "mix_bits": -sum(r["lp_mix"] for r in R if r["has_mix"]) / max(sum(r["has_mix"] for r in R), 1),
        }
        stats[k] = s
    num["by_kind"] = stats
    lines = [
        ("words", "words", "{:.0f}"),
        ("mean bits a word under K5 + overlay (<unk> left out)", "bits_mean", "{:.2f}"),
        ("<unk> words (outside the 32,000-word vocabulary)", "unk", "{:.3f}"),
        ("answered by order 5 (the word seen after the 4 words before)", "ans5", "{:.3f}"),
        ("answered by order 4", "ans4", "{:.3f}"),
        ("answered by order 3", "ans3", "{:.3f}"),
        ("answered by order 2 (only the word before)", "ans2", "{:.3f}"),
        ("from the class floor (no context held the word)", "ans1", "{:.3f}"),
        ("an order-5 context existed (4 words back matched)", "ctx5", "{:.3f}"),
        ("the overlay's document held the word in context", "ov_ans", "{:.3f}"),
        ("overlay's gain, bits a word", "ov_gain", "{:+.2f}"),
        ("rank 1 (K5's most probable word)", "rank1", "{:.3f}"),
        ("rank 1-10", "rank_le10", "{:.3f}"),
        ("P < 0.01", "p_lt_01", "{:.3f}"),
        ("inside T8's nucleus (T 0.8, nucleus 0.8, after the bans)", "nuc_in", "{:.3f}"),
        ("words in the nucleus (median)", "nuc_k_med", "{:.0f}"),
        ("learner's bits a word (primed, frozen; copy off)", "mix_bits", "{:.2f}"),
    ]
    for label, key, fmt in lines:
        emit("| " + label + " | " + " | ".join(fmt.format(stats[k][key]) if k in stats else "—" for k in ("human", "T8", "MR")) + " |")
    emit("")

    # 2. the order that answered, around sentence ends
    emit("## Table 2. The first two words of a sentence against the rest\n")
    emit("| | human, first 2 | human, rest | T8, first 2 | T8, rest |")
    emit("|---|---|---|---|---|")
    pos = {}
    for k in ("human", "T8"):
        for part in ("first", "rest"):
            R = []
            for (i, kk), v in by.items():
                if kk != k:
                    continue
                since = 99
                for j, r in enumerate(v):
                    first = since < 2
                    if (part == "first") == first:
                        R.append(r)
                    since = 0 if r["tok"] in STOP else since + 1
                    if j == 0:
                        pass
            n = len(R) or 1
            pos[(k, part)] = {
                "n": len(R),
                "bits": -sum(r["lp_full"] for r in R if not r["unk"]) / n,
                "ans5": sum(r["ans_ord"] == 5 for r in R) / n,
                "ans_le2": sum(1 <= r["ans_ord"] <= 2 for r in R) / n,
                "nuc_k": sorted(r["nuc_k"] for r in R)[len(R) // 2] if R else 0,
            }
    for label, key, fmt in [("words", "n", "{:.0f}"), ("bits a word", "bits", "{:.2f}"), ("answered by order 5", "ans5", "{:.3f}"),
                            ("answered by order 2 or the floor", "ans_le2", "{:.3f}"), ("nucleus size (median)", "nuc_k", "{:.0f}")]:
        emit("| " + label + " | " + " | ".join(fmt.format(pos[(k, p)][key]) for k in ("human", "T8") for p in ("first", "rest")) + " |")
    num["sentence_start"] = {f"{k}_{p}": v for (k, p), v in pos.items()}
    emit("")

    # 3. human words K5 finds very unlikely (model failures)
    H = [r for (i, kk), v in by.items() if kk == "human" for r in v]
    fail = [r for r in H if r["unk"] or r["lp_full"] < -15]
    kinds_f = Counter()
    for r in fail:
        if r["unk"]:
            kinds_f["<unk> (vocabulary hole)"] += 1
        elif r["ans_ord"] == 1:
            kinds_f["class floor (no context held it)"] += 1
        elif r["ans_ord"] == 2:
            kinds_f["order 2 only"] += 1
        else:
            kinds_f["order 3+ held it, still below 2^-15"] += 1
    emit("## Table 3. Human words K5 finds very unlikely (P < 2^-15, or <unk>)\n")
    emit(f"{len(fail)} of {len(H)} human words ({pct(len(fail), len(H))}).\n")
    emit("| kind | words | share of human words |")
    emit("|---|---|---|")
    for k, v in kinds_f.most_common():
        emit(f"| {k} | {v} | {pct(v, len(H))} |")
    caps = sum(1 for r in fail if r["tok"][:1].isupper())
    digs = sum(1 for r in fail if r["tok"][:1].isdigit())
    emit(f"\nOf these, capitalised: {caps}; numbers: {digs}.\n")
    num["human_fail"] = {"n": len(fail), "of": len(H), "kinds": dict(kinds_f), "caps": caps, "digits": digs}
    emit("Human words outside T8's nucleus: " + pct(sum(1 for r in H if not r["nuc_in"]), len(H)) +
         f" ({sum(1 for r in H if not r['nuc_in'])} of {len(H)}).\n")

    # 4. T8's broken spots (marked by reading)
    T8 = {i: v for (i, kk), v in by.items() if kk == "T8"}
    marked = []
    for mk in marks:
        i = mk["item"]
        if i not in T8:
            continue
        wi = int(mk["wi"])
        v = T8[i]
        if wi >= len(v):
            continue
        r = dict(v[wi])
        r["why"] = mk.get("why", "")
        r["prev"] = " ".join(x["tok"] for x in v[max(0, wi - 5):wi])
        marked.append(r)
    allT = [r for v in T8.values() for r in v]
    if marked:
        n = len(marked)
        emit(f"## Table 4. T8's broken spots, marked by reading ({n} spots in {len(set(m['item'] for m in marks))} texts), against all T8 words\n")
        emit("| | broken spots | all T8 words |")
        emit("|---|---|---|")

        def share(R, f):
            return sum(1 for r in R if f(r)) / max(len(R), 1)

        rowsdef = [
            ("answered by order 5", lambda r: r["ans_ord"] == 5),
            ("answered by order 4", lambda r: r["ans_ord"] == 4),
            ("answered by order 3", lambda r: r["ans_ord"] == 3),
            ("answered by order 2", lambda r: r["ans_ord"] == 2),
            ("class floor", lambda r: r["ans_ord"] == 1),
            ("context matched 4 words back (order-5 context existed)", lambda r: r["ctx_ord"] == 5),
            ("context matched 2 or fewer words back", lambda r: r["ctx_ord"] <= 3),
            ("overlay held the word in context", lambda r: r["ov_ans"] > 0),
            ("rank 1", lambda r: r["rank"] == 1),
            ("rank 2-10", lambda r: 2 <= r["rank"] <= 10),
            ("rank above 10 (the tail of the nucleus)", lambda r: r["rank"] > 10),
            ("P >= 0.1", lambda r: r["lp_full"] >= math.log2(0.1)),
            ("P < 0.01", lambda r: r["lp_full"] < math.log2(0.01)),
        ]
        brk = {}
        for label, f in rowsdef:
            brk[label] = (share(marked, f), share(allT, f))
            emit(f"| {label} | {brk[label][0]:.3f} | {brk[label][1]:.3f} |")
        mb = -sum(r["lp_full"] for r in marked) / n
        ma = -sum(r["lp_full"] for r in allT) / len(allT)
        emit(f"| mean bits | {mb:.2f} | {ma:.2f} |")
        nk = sorted(r["nuc_k"] for r in marked)[n // 2]
        emit(f"| nucleus size (median) | {nk} | {sorted(r['nuc_k'] for r in allT)[len(allT) // 2]} |")
        emit("")
        tail = sum(1 for r in marked if r["rank"] > 10 or r["lp_full"] < math.log2(0.02))
        hp = sum(1 for r in marked if r["rank"] <= 3 and r["lp_full"] >= math.log2(0.1))
        emit(f"Sampling into the tail (rank above 10 or P < 0.02): **{tail} of {n}**. A high-probability wrong word (rank 1-3 and P >= 0.1): **{hp} of {n}**.\n")
        why = Counter(r["why"] for r in marked)
        emit("Kinds of break, by reading: " + "; ".join(f"{k} {v}" for k, v in why.most_common()) + ".\n")
        num["breaks"] = {"n": n, "tail": tail, "highprob": hp, "shares": brk, "mean_bits": mb, "why": dict(why)}
        emit("| item | before | broken word | P | rank | order answered | context depth | overlay | K5's top word | kind |")
        emit("|---|---|---|---|---|---|---|---|---|---|")
        for r in marked:
            emit(f"| {r['item']} | …{r['prev']} | **{r['tok']}** | {2 ** r['lp_full']:.3f} | {r['rank']} | {r['ans_ord']} | {r['ctx_ord'] - 1} words | "
                 f"{'yes' if r['ov_ans'] else 'no'} | {r['top1']} ({2 ** r['lp_top1']:.2f}) | {r['why']} |")
        emit("")

    # 5. the gate
    if gate:
        G = [g for g in gate if g["path"] != "fact"]
        ranks = [int(g["rank_flu"]) for g in G]
        n = len(G)
        k5 = [g for g in G if g["path"] == "k5"]
        agree = sum(1 for g in k5 if g["pick"] == g["match"])
        emit("## Table 5. The gate: which of its 16 drafts T8 kept\n")
        emit(f"Sentences whose 16 drafts were reproduced exactly: {n} (plain gate {len(k5)}, discourse free sentences {n - len(k5)}); "
             f"fact sentences or unmatched: {sum(1 for g in gate if g['path'] == 'fact')}. The plain gate's choice, recomputed, equals the kept draft in {agree} of {len(k5)}.\n")
        if n:
            flu_k = sum(float(g["flu_kept"]) for g in G) / n
            flu_b = sum(float(g["flu_best"]) for g in G) / n
            flu_m = sum(float(g["flu_mean"]) for g in G) / n
            emit("| | value |")
            emit("|---|---|")
            emit(f"| kept draft's fluency rank among 16 (1: most fluent), mean | {sum(ranks) / n:.2f} (uniform: 8.5) |")
            emit(f"| kept draft is the most fluent | {pct(sum(1 for x in ranks if x == 1), n)} |")
            emit(f"| kept draft in the less fluent half (rank 9-16) | {pct(sum(1 for x in ranks if x > 8), n)} |")
            emit(f"| mean log2 P a word: kept / mean of 16 / most fluent | {flu_k:.2f} / {flu_m:.2f} / {flu_b:.2f} |")
            emit(f"| kept draft's z(prompt association) + z(topic), mean | {sum(float(g['zp_kept']) + float(g['zt_kept']) for g in G) / n:+.2f} |")
            num["gate"] = {"n": n, "k5": len(k5), "agree": agree, "rank_mean": sum(ranks) / n, "rank1": sum(1 for x in ranks if x == 1) / n,
                           "rank_gt8": sum(1 for x in ranks if x > 8) / n, "flu_kept": flu_k, "flu_mean": flu_m, "flu_best": flu_b}
        emit("")

    # 6. the most probable output
    if beam:
        emit("## Table 6. The most probable output against the sampled and the human text\n")
        emit("| text | mean bits a word under K5 + overlay | words in 460+ bytes |")
        emit("|---|---|---|")
        modes = []
        for b in beam:
            if b["mode"] not in modes:
                modes.append(b["mode"])
        bm = {}
        for mo in modes:
            B = [b for b in beam if b["mode"] == mo]
            bpw = sum(float(b["bits_per_word"]) for b in B) / len(B)
            wds = sum(int(b["words"]) for b in B) / len(B)
            bm[mo] = {"bits": bpw, "words": wds}
            emit(f"| {mo} | {bpw:.2f} | {wds:.0f} |")
        for k in kinds:
            emit(f"| {k} (sampled or written) | {stats[k]['bits_mean']:.2f} | — |")
        num["beam"] = bm
        # repetition: share of repeated word 3-grams
        emit("\nRepetition (share of word 3-grams that occur earlier in the same text), and the share of sentences that end:\n")
        emit("| text | repeated 3-grams | sentences per 100 words |")
        emit("|---|---|---|")

        def rep3(t):
            w = re.findall(r"\w+|[^\w\s]", t)
            g = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
            seen, r = set(), 0
            for x in g:
                r += x in seen
                seen.add(x)
            return r / max(len(g), 1), 100 * sum(1 for x in w if x in (".", "!", "?")) / max(len(w), 1)

        for mo in modes:
            B = [rep3(unesc_tsv(b["text"])) for b in beam if b["mode"] == mo]
            emit(f"| {mo} | {sum(x for x, _ in B) / len(B):.3f} | {sum(y for _, y in B) / len(B):.1f} |")
        for k in kinds:
            texts = []
            for (i, kk), v in by.items():
                if kk == k:
                    texts.append(" ".join(r["tok"] for r in v))
            B = [rep3(t) for t in texts]
            emit(f"| {k} | {sum(x for x, _ in B) / len(B):.3f} | {sum(y for _, y in B) / len(B):.1f} |")
        emit("")
    with open(os.path.join(a.out, "tables.md"), "w", encoding="utf-8", errors="surrogateescape") as f:
        f.write("\n".join(T) + "\n")
    with open(os.path.join(a.out, "numbers.json"), "w") as f:
        json.dump(num, f, indent=1, default=str)

    # examples: human and T8 maps side by side, one line per word
    E = []
    ids = a.examples.split(",") if a.examples else []
    for i in ids:
        for k in ("human", "T8"):
            v = by.get((i, k), [])
            if not v:
                continue
            E.append(f"### Prompt {i}, {k}\n")
            E.append("Each word: log2 P under K5 + overlay, the order that answered (5 to 2; F the class floor; U <unk>), and N if outside T8's nucleus. "
                     "Words below 2^-10 are bold.\n")
            out = []
            for r in v:
                o = {0: "U", 1: "F"}.get(r["ans_ord"], str(r["ans_ord"]))
                w = r["tok"].replace("\n", "\\n")
                cell = f"{w}<sub>{r['lp_full']:.0f}·{o}{'·N' if not r['nuc_in'] else ''}</sub>"
                if r["lp_full"] < -10 or r["unk"]:
                    cell = f"**{w}**<sub>{r['lp_full']:.0f}·{o}{'·N' if not r['nuc_in'] else ''}</sub>"
                out.append(cell)
            E.append(" ".join(out) + "\n")
    with open(os.path.join(a.out, "examples.md"), "w", encoding="utf-8", errors="surrogateescape") as f:
        f.write("\n".join(E) + "\n")
    print("\n".join(T))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("items")
    p.add_argument("--ids", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--t8", required=True)
    p.add_argument("--mr")
    p.add_argument("--prompts", default="bench/eval/gen_prompts_dev.tsv")
    p = sp.add_parser("report")
    p.add_argument("--probe", required=True)
    p.add_argument("--marks")
    p.add_argument("--out", required=True)
    p.add_argument("--examples", default="")
    a = ap.parse_args()
    {"items": cmd_items, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
