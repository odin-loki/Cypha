#!/usr/bin/env python3
"""Build the QE2 probe set (described in mined/probe/README.md). Odin Loch, 2026-10-28.

Inputs: held-out enwik8 (bytes after 8 MiB), Dolly, GLUE STS-B/MRPC, PAWS labeled_final, SNLI
(train/validation splits only; every test split is kept out), NP chunks from the E1 chunker.
T-QA questions (bench/eval/t_qa_wiki*.tsv) are read only to exclude them.

Usage: python -I build_probe.py OUT_DIR DATA_DIR REPO_DIR CHUNKER MODEL
(ENWIK8 and DOLLY in the environment override the corpus paths below.)
"""
import collections
import hashlib
import html
import json
import os
import random
import re
import subprocess
import sys

OUT, DATA, REPO, CHUNKER, CMODEL = sys.argv[1:6]
SEED = 20261028
ENWIK = os.environ.get("ENWIK8", "/root/cdata/enwik8")
DOLLY = os.environ.get("DOLLY", "/root/cdata/dolly.jsonl")
HELD_OUT_START = 8 * 1024 * 1024
os.makedirs(OUT, exist_ok=True)
rng = random.Random(SEED)


# ------------------------------------------------------------------ text hygiene

def norm_ws(s):
    return re.sub(r"\s+", " ", s.replace(" ", " ")).strip()


def safe(s):
    """Items must survive the index TSV unchanged: no tab, newline or backslash."""
    s = norm_ws(s)
    return s if s and "\\" not in s and "\t" not in s else None


def qkey(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


TQA = set()
for fn in ("t_qa_wiki.tsv", "t_qa_wiki_dev.tsv", "t_qa_wiki_seed.tsv"):
    with open(os.path.join(REPO, "bench/eval", fn), encoding="utf-8") as f:
        for ln in f:
            q = ln.split("\t")[0]
            if q and q != "question":
                TQA.add(qkey(q))


def is_tqa(s):
    k = qkey(s)
    return k in TQA or any(q and q in k for q in TQA if len(q) > 15)


def split_of(s):
    return "test" if int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % 5 == 0 else "dev"


# ------------------------------------------------------------------ enwik8 cleaning

NS = re.compile(r"^\s*:?\s*(image|file|category|media|wikipedia|template|help|portal|user|talk|special|wp|wiktionary|"
                r"[a-z]{2,3}(-[a-z]+)?|simple|meta|commons|wikt|q|s|b|n|v)\s*:", re.I)


def strip_nested(s, open_, close):
    pat = re.compile(re.escape(open_) + r"(?:(?!" + re.escape(open_) + r"|" + re.escape(close) + r").)*" + re.escape(close), re.S)
    for _ in range(10):
        s2 = pat.sub(" ", s)
        if s2 == s:
            break
        s = s2
    return s


def links(s):
    pat = re.compile(r"\[\[([^\[\]]*)\]\]")

    def rep(m):
        inner = m.group(1)
        if NS.match(inner):
            return " "
        parts = inner.split("|")
        return parts[-1] if len(parts) > 1 else parts[0]
    for _ in range(6):
        s2 = pat.sub(rep, s)
        if s2 == s:
            break
        s = s2
    return s


def clean_wiki(text):
    t = html.unescape(html.unescape(text))
    t = re.sub(r"<!--.*?-->", " ", t, flags=re.S)
    t = re.sub(r"<ref[^>]*/>", " ", t, flags=re.I)
    t = re.sub(r"<ref[^>]*>.*?</ref>", " ", t, flags=re.I | re.S)
    t = re.sub(r"<(math|code|pre|gallery|timeline|table|source|nowiki|div|imagemap)[^>]*>.*?</\1>", " ", t, flags=re.I | re.S)
    t = re.sub(r"<[^<>]{0,200}>", " ", t)
    t = strip_nested(t, "{{", "}}")
    t = strip_nested(t, "{|", "|}")
    t = links(t)
    t = re.sub(r"\[(?:https?|ftp)://\S+\s+([^\]]*)\]", r"\1", t)
    t = re.sub(r"\[(?:https?|ftp)://\S+\]", " ", t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"'{2,}", "", t)
    return t


ABBR = re.compile(r"\b(Mr|Mrs|Ms|Dr|St|Jr|Sr|Prof|Gen|Col|Lt|Sgt|Capt|Rev|Mt|No|vs|etc|Inc|Ltd|Co|Corp|Fig|ca|c|approx|Vol|pp|ed|eds|U\.S|U\.K|i\.e|e\.g|[A-Z])\.$")
SPLIT = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(]?[A-Z0-9])")


def sentences(par):
    out, buf = [], ""
    for piece in SPLIT.split(par):
        buf = (buf + " " + piece).strip() if buf else piece.strip()
        if ABBR.search(buf):
            continue
        out.append(buf)
        buf = ""
    if buf:
        out.append(buf)
    return out


BAD = set("[]{}|<>=*#_\\~^")


def good_sentence(s, lo=6, hi=40):
    w = s.split()
    if not (lo <= len(w) <= hi) or not s[0].isupper() or s[-1] not in ".!?":
        return False
    if any(c in BAD for c in s) or "()" in s or "( " in s or " ," in s or " ." in s or "  " in s or ";;" in s:
        return False
    if s.count("(") != s.count(")") or s.count('"') % 2:
        return False
    ns = [c for c in s if not c.isspace()]
    return sum(c.isalpha() for c in ns) >= 0.75 * len(ns)


def enwik_pages():
    with open(ENWIK, "rb") as f:
        f.seek(HELD_OUT_START)
        raw = f.read().decode("utf-8", errors="replace")
    for chunk in raw.split("<page>")[1:]:
        m = re.search(r"<title>(.*?)</title>", chunk)
        tm = re.search(r"<text[^>]*>(.*?)</text>", chunk, flags=re.S)
        if not m or not tm:
            continue
        title = html.unescape(m.group(1))
        body = tm.group(1)
        if body.lstrip()[:9].lower() == "#redirect" or ":" in title:
            continue
        yield title, clean_wiki(body)


pages = []  # (title, [paragraph sentence lists])
wc, surf = collections.Counter(), collections.defaultdict(collections.Counter)
for title, body in enwik_pages():
    paras = []
    for ln in body.split("\n"):
        s = ln.strip()
        if not s or s[0] in "=*#:;|!{ " or ln[:1] == " " or len(s) < 40:
            continue
        s = norm_ws(s)
        for w in re.findall(r"[A-Za-z][A-Za-z'-]*[A-Za-z]|[A-Za-z]", s):
            wc[w.lower()] += 1
            surf[w.lower()][w] += 1
        sents = [x for x in sentences(s) if good_sentence(x)]
        if sents:
            paras.append(sents)
    if paras:
        pages.append((norm_ws(title), paras))
print(f"enwik held-out: {len(pages)} articles with prose, {sum(len(p) for _, ps in pages for p in ps)} clean sentences",
      file=sys.stderr)

with open(os.path.join(OUT, "word_counts.tsv"), "w", encoding="utf-8") as f:
    f.write("word\tcount\n")
    for w, c in wc.most_common():
        if c >= 2:
            f.write(f"{w}\t{c}\n")

# ------------------------------------------------------------------ items and pairs

items = collections.OrderedDict()  # text -> (kind, source)
pos, cls = [], []  # (a, b, source/class)


def add(text, kind, source):
    t = safe(text) if text else None
    if not t or is_tqa(t):
        return None
    if t not in items:
        items[t] = (kind, source)
    return t


def pair(lst, a, b, label):
    if a and b and a != b:
        lst.append((a, b, label))


# Sentences from held-out enwik8: up to 4 per article for the probe; the rest feed held_out.txt and the chunker.
rng.shuffle(pages)
probe_sents, rest = [], []
for title, paras in pages:
    flat = [(pi, si, s) for pi, p in enumerate(paras) for si, s in enumerate(p)]
    pick = set(rng.sample(range(len(flat)), min(4, len(flat))))
    for k, x in enumerate(flat):
        (probe_sents if k in pick else rest).append((title, x))
probe_sents = probe_sents[:30000]
for title, (_pi, _si, s) in probe_sents:
    add(s, "sentence", "enwik8_heldout")

# Classes built on enwik: adjacent sentences, same article not adjacent, unrelated, title -> lead.
by_title = {t: ps for t, ps in pages}
n_adj = n_same = n_lead = 0
for title, paras in pages:
    if n_adj < 3000:
        cand = [(p[i], p[i + 1]) for p in paras for i in range(len(p) - 1)]
        if cand:
            a, b = rng.choice(cand)
            a, b = add(a, "sentence", "enwik8_heldout"), add(b, "sentence", "enwik8_heldout")
            pair(pos, a, b, "wiki_adjacent")
            pair(cls, a, b, "wiki_adjacent")
            n_adj += 1
    if n_same < 3000:
        flat = [s for p in paras for s in p]
        if len(flat) >= 6:
            i, j = sorted(rng.sample(range(len(flat)), 2))
            if j - i >= 3:
                a, b = add(flat[i], "sentence", "enwik8_heldout"), add(flat[j], "sentence", "enwik8_heldout")
                pair(cls, a, b, "wiki_same_article")
                n_same += 1
    if n_lead < 2000 and 1 <= len(title.split()) <= 6:
        t, l = add(title, "title", "enwik8_heldout"), add(paras[0][0], "sentence", "enwik8_heldout")
        pair(pos, t, l, "wiki_title_lead")
        pair(cls, t, l, "wiki_title_lead")
        n_lead += 1
wiki_items = [t for t, (k, s) in items.items() if s == "enwik8_heldout" and k == "sentence"]
for _ in range(3000):
    a, b = rng.sample(wiki_items, 2)
    pair(cls, a, b, "unrelated")

# held_out.txt: sentences not in the probe, 8-25 words, for sensitivity and occlusion.
rest_s = [s for _t, (_pi, _si, s) in rest if s not in items and good_sentence(s, 8, 25) and not is_tqa(s)]
rng.shuffle(rest_s)
held_out = list(dict.fromkeys(x for x in (safe(s) for s in rest_s) if x))[:4000]

# NP chunks from the E1 chunker, on other held-out sentences.
chunk_in = [s for _t, (_pi, _si, s) in rest if s not in set(held_out)][:120000]
p = subprocess.run([CHUNKER, "tag", CMODEL, "--chunks"], input="\n".join(chunk_in) + "\n", capture_output=True, text=True)
npc = collections.Counter()
DET = {"the", "a", "an", "this", "that", "these", "those", "its", "his", "her", "their", "our", "my", "your"}
for ln in p.stdout.splitlines():
    if not ln.startswith("  "):
        continue
    for m in re.finditer(r"\[NP ([^\[\]]+)\]", ln):
        w = m.group(1).split()
        while w and w[0].lower() in DET:
            w = w[1:]
        if 2 <= len(w) <= 5 and all(re.fullmatch(r"[A-Za-z][A-Za-z'-]*", x) for x in w):
            npc[" ".join(w)] += 1
lower_seen, phrases = set(), []
for ph, c in npc.most_common():
    k = ph.lower()
    if k not in lower_seen:
        lower_seen.add(k)
        phrases.append(ph)
    if len(phrases) >= 20000:
        break
for ph in phrases:
    add(ph, "phrase", "e1_np_chunk")

# Words: the 20k most frequent held-out word types, in their most common casing.
nw = 0
for w, c in wc.most_common():
    if len(w) < 2 and w not in ("a", "i"):
        continue
    if add(surf[w].most_common(1)[0][0], "word", "enwik8_heldout_freq"):
        nw += 1
    if nw >= 20000:
        break

# Dolly: questions.txt from question-form instructions; the probe gets other records (instruction + response lead).
with open(DOLLY, encoding="utf-8") as f:
    dolly = [json.loads(ln) for ln in f if ln.strip()]
QCATS = {"open_qa", "general_qa", "closed_qa"}
qs = [d for d in dolly if d.get("category") in QCATS and norm_ws(d["instruction"]).endswith("?")
      and 4 <= len(d["instruction"].split()) <= 30 and safe(d["instruction"]) and not is_tqa(d["instruction"])]
rng.shuffle(qs)
questions = list(dict.fromkeys(safe(d["instruction"]) for d in qs))[:2000]
qset = set(questions)


def lead(text, cap=400):
    out = ""
    for s in sentences(norm_ws(text)):
        if out and len(out) + 1 + len(s) > cap:
            break
        out = (out + " " + s).strip()
    return out[:cap * 2] if out else None


rest_d = [d for d in dolly if safe(d["instruction"]) not in qset and len(d["instruction"]) <= 400]  # no pasted passages
rng.shuffle(rest_d)
by_cat = collections.defaultdict(list)
for d in rest_d[:8000]:
    ins, res = add(d["instruction"], "instruction", "dolly"), add(lead(d["response"]), "response", "dolly")
    if ins and res:
        pair(pos, ins, res, "dolly_instruction_response")
        pair(cls, ins, res, "dolly_instruction_response")
        by_cat[d["category"]].append((ins, res))
for cat, lst in by_cat.items():  # same category, another record's response: same task type, wrong answer
    for ins, _res in lst[: max(1, 3000 * len(lst) // 8000)]:
        other = rng.choice(lst)[1]
        pair(cls, ins, other, "dolly_wrong_response")

# Pair datasets (train/validation only).
import pyarrow.parquet as pq  # noqa: E402


def rows(fn):
    return pq.read_table(os.path.join(DATA, fn)).to_pylist()


for fn in ("glue_stsb_train.parquet", "glue_stsb_validation.parquet"):
    for r in rows(fn):
        a, b, y = add(r["sentence1"], "sentence", "stsb"), add(r["sentence2"], "sentence", "stsb"), float(r["label"])
        if y < 0:
            continue
        pair(cls, a, b, f"stsb_{min(int(y), 4)}to{min(int(y), 4) + 1}")
        if y >= 4.0:
            pair(pos, a, b, "stsb_ge4")
for fn in ("glue_mrpc_train.parquet", "glue_mrpc_validation.parquet"):
    for r in rows(fn):
        a, b = add(r["sentence1"], "sentence", "mrpc"), add(r["sentence2"], "sentence", "mrpc")
        pair(cls, a, b, "mrpc_paraphrase" if r["label"] == 1 else "mrpc_not_paraphrase")
        if r["label"] == 1:
            pair(pos, a, b, "mrpc_paraphrase")
paws = rows("paws_labeled_final_validation.parquet")
for r in paws:
    a, b = add(r["sentence1"], "sentence", "paws"), add(r["sentence2"], "sentence", "paws")
    pair(cls, a, b, "paws_paraphrase" if r["label"] == 1 else "paws_not_paraphrase")
    if r["label"] == 1:
        pair(pos, a, b, "paws_paraphrase")
SNLI = {0: "snli_entailment", 1: "snli_neutral", 2: "snli_contradiction"}
for r in rows("snli_validation.parquet"):
    if r["label"] not in SNLI:
        continue
    a, b = add(r["premise"], "sentence", "snli"), add(r["hypothesis"], "sentence", "snli")
    pair(cls, a, b, SNLI[r["label"]])
    if r["label"] == 0:
        pair(pos, a, b, "snli_entailment")

# ------------------------------------------------------------------ write

order = list(items)
rng.shuffle(order)  # so the first 1,000 lines (the timing subset, --limit 1000) are a fair sample
with open(os.path.join(OUT, "probe.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(order) + "\n")
with open(os.path.join(OUT, "probe_meta.tsv"), "w", encoding="utf-8") as f:
    f.write("row\tkind\tsource\tsplit\n")
    for i, t in enumerate(order):
        f.write(f"{i}\t{items[t][0]}\t{items[t][1]}\t{split_of(t)}\n")
with open(os.path.join(OUT, "timing_1000.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(order[:1000]) + "\n")


def pair_split(a, b):
    return "test" if "test" in (split_of(a), split_of(b)) else "dev"


seen = set()
with open(os.path.join(OUT, "pos_pairs.tsv"), "w", encoding="utf-8") as f:
    f.write("anchor\tpositive\tsource\tsplit\n")
    for a, b, s in pos:
        if (a, b) not in seen:
            seen.add((a, b))
            f.write(f"{a}\t{b}\t{s}\t{pair_split(a, b)}\n")
seen = set()
with open(os.path.join(OUT, "classes.tsv"), "w", encoding="utf-8") as f:
    f.write("a\tb\tclass\tsplit\n")
    for a, b, c in cls:
        if a in items and b in items and (a, b, c) not in seen:
            seen.add((a, b, c))
            f.write(f"{a}\t{b}\t{c}\t{pair_split(a, b)}\n")
with open(os.path.join(OUT, "held_out.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(held_out) + "\n")
with open(os.path.join(OUT, "questions.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(questions) + "\n")

kinds = collections.Counter((k, s) for k, s in items.values())
summary = {"seed": SEED, "items": len(items), "by_kind_source": {f"{k}/{s}": n for (k, s), n in sorted(kinds.items())},
           "pos_pairs": collections.Counter(s for _a, _b, s in pos), "classes": collections.Counter(c for _a, _b, c in cls),
           "held_out": len(held_out), "questions": len(questions), "np_chunk_input_sentences": len(chunk_in),
           "tqa_questions_excluded_set": len(TQA), "enwik_articles": len(pages)}
with open(os.path.join(OUT, "build_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2), file=sys.stderr)
