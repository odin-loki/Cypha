#!/usr/bin/env python3
"""Text signals of CyphaLM's writing against the blind raters (GENERATION_PLAN §13.14). Odin Loch, 2026-11-12.

Which measurable signals separate human text from CyphaLM's text, and which of them pick the text the blind
raters picked? Report and registered predictions: docs/reports/WRITING_SIGNAL.md. Outputs: bench/writing_signal/.

Steps (run from the repository root, in order):

  judgments  every committed blind round in bench/blind/*/ with packets (R1..R4.txt), the raters' replies
             (R1..R4.json) and key.b64. Each rated pair is one judgment. Pair rounds: the rater's choice. Rate rounds
             (1-10 scores): every two texts of an item with different scores are a pair, the higher score preferred.
             The texts are taken from the packets, exactly as the raters saw them (the trailing cut mark removed).
             -> judgments.jsonl, texts.jsonl
  gapset     the human continuations of the 250 GEN-A dev prompts and every dev GEN file of the writing benches,
             each cut near 460 characters as blind_rate.py cuts them. -> gap_texts.jsonl
  features   the registered signals of every text (a word trigram Kneser-Ney model over --train for the surprise
             signals). -> features.jsonl, features_meta.json
  scribe     SCRIBE v2 and v3.1 of every text (the scribe binary's `prepare --mode continuation`, as the ladder, then
             `score` with the prompt). -> scribe.jsonl
  gap        human against writer for each signal. -> gap.md, gap.json
  agree      for each signal, the share of rated pairs where it picks the rater's choice, with 95% cluster
             bootstrap intervals (clusters: round x prompt). -> agree.md, agree.json
  explore    not registered: a leave-one-round-out logistic combination, and agreement by group of rounds.
             -> explore.md, explore.json

  python3 -I scripts/surprise_profile.py all --train /mnt/d/cypha_llm/fuzzy/train.txt --scribe ~/gq_bin/scribe
"""
import argparse
import base64
import collections
import csv
import glob
import hashlib
import json
import math
import os
import random
import re
import statistics as st
import subprocess

OUT = "bench/writing_signal"
csv.field_size_limit(1 << 30)

# ---------------------------------------------------------------------------------------------------------------
# Text handling
# ---------------------------------------------------------------------------------------------------------------

FUNCTION = set("""a about above after again against all also although am an and any are as at be because been before
being below between both but by can could did do does doing down during each either even ever every few for from
further had has have having he her here hers herself him himself his how however i if in into is it its itself just
least less many may me might more most much must my myself neither no nor not now of off often on once one only or
other others our ours ourselves out over own per quite rather same several shall she should since so some such than
that the their theirs them themselves then there these they this those though through thus to too toward towards
under until up upon us very was we were what whatever when where whether which while who whom whose why will with
within without would yet you your yours""".split())

TOK = re.compile(r"\\x[0-9a-fA-F]{2}|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*|[^\sA-Za-z\d]")
SENT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def cut(t, n=460):  # as scripts/blind_rate.py cuts
    t = " ".join(t.split())
    if len(t) <= n:
        return t
    t = t[:n]
    t = t[: t.rfind(" ")]
    return t.rstrip(" ,;:") + " …"


def uncut(t):
    t = t.strip()
    return t[:-2].rstrip() if t.endswith(" …") else t


def tid(prompt, text):
    return hashlib.sha1((prompt + "\x00" + text).encode()).hexdigest()[:16]


def is_word(t):
    return t[0].isalpha() and not t.startswith("\\x")


def stem(w):
    w = w.lower()
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def is_content(t):
    return is_word(t) and len(t) >= 3 and t.lower() not in FUNCTION


def sentences(text):
    return [s for s in SENT.split(text) if s.strip()]


def kn_tokens(toks):
    return [t.lower() if not t[0].isdigit() else "<num>" for t in toks]


# ---------------------------------------------------------------------------------------------------------------
# Word trigram Kneser-Ney (interpolated, one discount per order)
# ---------------------------------------------------------------------------------------------------------------


class KN3:
    def __init__(self, path, min_count=2, d=0.75):
        self.d = d
        raw = open(path, encoding="utf-8", errors="replace").read()
        toks = []
        for line in raw.split("\n"):
            if line.strip():
                toks.append("<s>")
                toks.extend(kn_tokens(TOK.findall(line)))
        uni = collections.Counter(toks)
        self.vocab = {w for w, c in uni.items() if c >= min_count}
        toks = [w if w in self.vocab else "<unk>" for w in toks]
        self.vocab.add("<unk>")
        c3 = collections.Counter(zip(toks, toks[1:], toks[2:]))
        self.c3 = c3
        self.ctx3 = collections.Counter()
        self.typ3 = collections.Counter()
        n2 = collections.Counter()  # N1+(. v w)
        for (u, v, w), c in c3.items():
            self.ctx3[(u, v)] += c
            self.typ3[(u, v)] += 1
            n2[(v, w)] += 1
        self.n2 = n2
        self.ctx2 = collections.Counter()
        self.typ2 = collections.Counter()
        n1 = collections.Counter()  # N1+(. w)
        for (v, w), c in n2.items():
            self.ctx2[v] += c
            self.typ2[v] += 1
            n1[w] += 1
        self.n1 = n1
        self.n1tot = sum(n1.values())
        self.V = len(self.vocab)
        self.ntok = len(toks)

    def p1(self, w):
        return (self.n1.get(w, 0) + 0.5) / (self.n1tot + 0.5 * self.V)

    def p2(self, v, w):
        c = self.ctx2.get(v, 0)
        if not c:
            return self.p1(w)
        return max(self.n2.get((v, w), 0) - self.d, 0) / c + self.d * self.typ2[v] / c * self.p1(w)

    def p3(self, u, v, w):
        c = self.ctx3.get((u, v), 0)
        if not c:
            return self.p2(v, w)
        return max(self.c3.get((u, v, w), 0) - self.d, 0) / c + self.d * self.typ3[(u, v)] / c * self.p2(v, w)

    def surprise(self, prompt_toks, toks):
        """Bits for each token of toks, the prompt's last two tokens as the first context."""
        m = lambda w: w if w in self.vocab else "<unk>"  # noqa: E731
        ctx = [m(w) for w in kn_tokens(prompt_toks)[-2:]]
        ctx = (["<s>", "<s>"] + ctx)[-2:]
        out = []
        for w in kn_tokens(toks):
            w = m(w)
            out.append(-math.log2(self.p3(ctx[0], ctx[1], w)))
            ctx = [ctx[1], w]
        return out


# ---------------------------------------------------------------------------------------------------------------
# The registered signals. Direction: + means a higher value should read better, - lower. Names ending in _h are a
# distance from the human median. Registered in WRITING_SIGNAL.md before any comparison with the raters.
# ---------------------------------------------------------------------------------------------------------------

SIGNALS = [
    ("surp_mean", "-", "mean surprise per word (bits, word trigram KN on train.txt)"),
    ("surp_sd", "+", "standard deviation of per-word surprise"),
    ("surp_ac1", "+", "lag-1 autocorrelation of per-word surprise (burstiness of surprise)"),
    ("surp_cf", "+", "mean surprise of content words minus that of function words"),
    ("burst_cw", "+", "share of content-word tokens that repeat an earlier content word of the text"),
    ("commit_text", "+", "share of sentences 2+ that share a content word with an earlier sentence"),
    ("commit_prompt", "+", "share of sentences that share a content word with the prompt"),
    ("ent_reuse", "+", "share of mid-sentence capitalised words already seen (prompt or text)"),
    ("sl_mean_h", "-", "|log mean sentence length - log human median|"),
    ("sl_cv_h", "-", "|sentence-length CV - human median|"),
    ("fw_js", "-", "Jensen-Shannon divergence of function-word frequencies from human text"),
    ("fw_share_h", "-", "|share of function words - human median|"),
    ("rep3", "-", "share of word trigrams that repeat an earlier trigram of the text"),
    ("junk", "-", "byte escapes, spaced hyphens and orphan single letters per word"),
    ("oov", "-", "share of words outside train.txt's vocabulary (count >= 2)"),
]
SCRIBE_SIGNALS = [("scribe_v2", "+", "SCRIBE v2 (score of record)"), ("scribe_v31", "+", "SCRIBE v3.1")]


def raw_features(kn, prompt, text):
    toks = TOK.findall(text)
    ptoks = TOK.findall(prompt)
    words = [t for t in toks if is_word(t)]
    f = {"n_words": len(words)}
    s = kn.surprise(ptoks, toks)
    ws = [b for t, b in zip(toks, s) if is_word(t)]
    if len(ws) >= 3:
        f["surp_mean"] = st.mean(ws)
        f["surp_sd"] = st.pstdev(ws)
        m = f["surp_mean"]
        var = sum((x - m) ** 2 for x in ws)
        f["surp_ac1"] = sum((ws[i] - m) * (ws[i + 1] - m) for i in range(len(ws) - 1)) / var if var else 0.0
        cw = [b for t, b in zip(toks, s) if is_content(t)]
        fw = [b for t, b in zip(toks, s) if is_word(t) and t.lower() in FUNCTION]
        f["surp_cf"] = (st.mean(cw) - st.mean(fw)) if cw and fw else 0.0
    cws = [stem(t) for t in toks if is_content(t)]
    seen, rep = set(), 0
    for w in cws:
        rep += w in seen
        seen.add(w)
    f["burst_cw"] = rep / len(cws) if cws else 0.0
    sents = sentences(text)
    scw = [{stem(t) for t in TOK.findall(x) if is_content(t)} for x in sents]
    pcw = {stem(t) for t in ptoks if is_content(t)}
    if len(scw) >= 2:
        hits, prev = 0, set(scw[0])
        for c in scw[1:]:
            hits += bool(c & prev)
            prev |= c
        f["commit_text"] = hits / (len(scw) - 1)
    else:
        f["commit_text"] = 0.0
    f["commit_prompt"] = sum(bool(c & pcw) for c in scw) / len(scw) if scw else 0.0
    caps_seen = {t.lower() for t in ptoks if is_word(t) and t[0].isupper()}
    ent, ent_rep = 0, 0
    for x in sents:
        st_toks = TOK.findall(x)
        for i, t in enumerate(st_toks):
            if is_word(t) and t[0].isupper():
                if i > 0 and len(t) > 1 and st_toks[i - 1] not in ('"', "(", ":"):
                    ent += 1
                    ent_rep += t.lower() in caps_seen
                caps_seen.add(t.lower())
    f["ent_reuse"] = ent_rep / ent if ent else 0.0
    sl = [len([t for t in TOK.findall(x) if is_word(t)]) for x in sents]
    sl = [x for x in sl if x > 0] or [len(words) or 1]
    f["sl_mean"] = st.mean(sl)
    f["sl_cv"] = st.pstdev(sl) / st.mean(sl) if len(sl) > 1 else 0.0
    lw = [w.lower() for w in words]
    fwc = collections.Counter(w for w in lw if w in FUNCTION)
    f["fw_share"] = sum(fwc.values()) / len(lw) if lw else 0.0
    f["_fw"] = dict(fwc)
    tri = list(zip(lw, lw[1:], lw[2:]))
    seen, rep = set(), 0
    for t in tri:
        rep += t in seen
        seen.add(t)
    f["rep3"] = rep / len(tri) if tri else 0.0
    esc = sum(t.startswith("\\x") for t in toks)
    hyph = len(re.findall(r"\s-\s", text))
    orphan = sum(1 for t in toks if is_word(t) and len(t) == 1 and t not in ("a", "A", "I"))
    f["junk"] = (esc + hyph + orphan) / max(len(words), 1)
    f["oov"] = sum(kn_tokens([w])[0] not in kn.vocab for w in words) / max(len(words), 1)
    return f


def js(p, q):
    keys = set(p) | set(q)
    sp, sq = sum(p.values()) or 1, sum(q.values()) or 1
    out = 0.0
    for k in keys:
        a, b = p.get(k, 0) / sp, q.get(k, 0) / sq
        m = (a + b) / 2
        if a:
            out += 0.5 * a * math.log2(a / m)
        if b:
            out += 0.5 * b * math.log2(b / m)
    return out


def finish_features(rows, human_ids):
    """Human-referenced signals: medians and the function-word profile from the 250 human dev continuations."""
    hum = [r for r in rows if r["id"] in human_ids]
    med_sl = st.median(r["sl_mean"] for r in hum)
    med_cv = st.median(r["sl_cv"] for r in hum)
    med_fw = st.median(r["fw_share"] for r in hum)
    href = collections.Counter()
    for r in hum:
        href.update(r["_fw"])
    for r in rows:
        r["sl_mean_h"] = abs(math.log(r["sl_mean"]) - math.log(med_sl))
        r["sl_cv_h"] = abs(r["sl_cv"] - med_cv)
        r["fw_share_h"] = abs(r["fw_share"] - med_fw)
        r["fw_js"] = js(r["_fw"], href)
        del r["_fw"]
    return {"human_median_sl_mean": med_sl, "human_median_sl_cv": med_cv, "human_median_fw_share": med_fw}


# ---------------------------------------------------------------------------------------------------------------
# Step: judgments
# ---------------------------------------------------------------------------------------------------------------


def read_packet(path):
    items = {}
    for block in open(path, encoding="utf-8").read().split("### Item ")[1:]:
        lines = block.split("\n")
        n = int(lines[0].strip())
        prompt, texts = None, {}
        for ln in lines[1:]:
            if ln.startswith("PROMPT: "):
                prompt = ln[8:]
            elif len(ln) > 3 and ln[1:3] == ": " and ln[0] in "ABCDEFGHIJ":
                texts[ln[0]] = uncut(ln[3:])
        items[n] = (prompt, texts)
    return items


def step_judgments(a):
    texts, judg, rounds = {}, [], []
    for d in sorted(glob.glob("bench/blind/*/")):
        rnd = os.path.basename(d.rstrip("/"))
        if not all(os.path.exists(os.path.join(d, f)) for f in ["key.b64"] + [f"R{r}.json" for r in range(1, 5)]):
            continue
        key = json.loads(base64.b64decode(open(os.path.join(d, "key.b64")).read()))
        mode = None
        for r in range(1, 5):
            pk = read_packet(os.path.join(d, f"R{r}.txt"))
            for row in json.load(open(os.path.join(d, f"R{r}.json"))):
                n = row["item"]
                prompt, tx = pk[n]
                ids = {}
                for lab, t in tx.items():
                    k = key[f"R{r}/{n}/{lab}"]
                    i = tid(prompt, t)
                    texts[i] = {"id": i, "prompt": prompt, "text": t}
                    ids[lab] = (i, k["arm"], k["dev"])
                if "choice" in row:
                    mode = "pair"
                    (ia, aa, dev), (ib, ab, _) = ids["A"], ids["B"]
                    ch = row["choice"]
                    pref = 0.5 if ch == "tie" else (1.0 if ch == "A" else 0.0)
                    judg.append({"round": rnd, "mode": mode, "rater": r, "item": n, "dev": dev, "x": ia, "y": ib,
                                 "arm_x": aa, "arm_y": ab, "pref": pref})
                else:
                    mode = "rate"
                    labs = sorted(ids)
                    for i1 in range(len(labs)):
                        for i2 in range(i1 + 1, len(labs)):
                            l1, l2 = labs[i1], labs[i2]
                            s1, s2 = row[l1], row[l2]
                            judg.append({"round": rnd, "mode": mode, "rater": r, "item": n, "dev": ids[l1][2],
                                         "x": ids[l1][0], "y": ids[l2][0], "arm_x": ids[l1][1], "arm_y": ids[l2][1],
                                         "pref": 0.5 if s1 == s2 else (1.0 if s1 > s2 else 0.0)})
        rounds.append((rnd, mode))
    for j in judg:
        j["human"] = "human" in (j["arm_x"], j["arm_y"])
        j["identical"] = texts[j["x"]]["text"] == texts[j["y"]]["text"]
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/judgments.jsonl", "w") as f:
        for j in judg:
            f.write(json.dumps(j) + "\n")
    with open(f"{OUT}/texts.jsonl", "w") as f:
        for t in texts.values():
            f.write(json.dumps(t) + "\n")
    print(f"{len(rounds)} rounds ({', '.join(f'{r}:{m}' for r, m in rounds)}), {len(judg)} judgments, {len(texts)} texts")


# ---------------------------------------------------------------------------------------------------------------
# Step: gapset
# ---------------------------------------------------------------------------------------------------------------


def read_gen(path):
    out, cur = {}, None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("PROMPT\t"):
                cur = int(line.split("\t")[1])
            elif line.startswith("GEN\t") and cur is not None and cur not in out:
                out[cur] = line.split("\t", 2)[2]
    return out


GEN_GLOBS = ["bench/gen_quality/runs/*.gen.txt", "bench/tune_write/runs/*.gen.txt", "bench/k5_coherence/runs/*.gen.txt",
             "bench/k5_planner/*.gen.txt"]


def step_gapset(a):
    rows = {}
    with open("bench/eval/gen_prompts_dev.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            rows[int(row["id"].split("/")[1])] = row
    out = []
    for dev, row in rows.items():
        p = " ".join(row["prompt"].split())
        t = uncut(cut(row["continuation"]))
        out.append({"id": tid(p, t), "prompt": p, "text": t, "arm": "human", "dev": dev})
    for g in GEN_GLOBS:
        for path in sorted(glob.glob(g)):
            arm = os.path.basename(path).replace(".gen.txt", "")
            if arm.endswith("_long"):
                continue
            arm = f"{path.split('/')[1]}/{arm}"
            for dev, t in read_gen(path).items():
                if dev not in rows or len(t.strip()) < 50:
                    continue
                p = " ".join(rows[dev]["prompt"].split())
                t = uncut(cut(t))
                out.append({"id": tid(p, t), "prompt": p, "text": t, "arm": arm, "dev": dev})
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/gap_texts.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"gap set: {len(out)} texts, {len({r['arm'] for r in out})} arms (human included)")


def all_texts():
    seen = {}
    for p in (f"{OUT}/texts.jsonl", f"{OUT}/gap_texts.jsonl"):
        for ln in open(p):
            r = json.loads(ln)
            seen.setdefault(r["id"], {"id": r["id"], "prompt": r["prompt"], "text": r["text"]})
    return list(seen.values())


# ---------------------------------------------------------------------------------------------------------------
# Step: features, scribe
# ---------------------------------------------------------------------------------------------------------------


def step_features(a):
    kn = KN3(a.train)
    print(f"KN3 on {a.train}: {kn.ntok} tokens, vocabulary {kn.V}, {len(kn.c3)} trigrams", flush=True)
    texts = all_texts()
    rows = [{"id": t["id"], **raw_features(kn, t["prompt"], t["text"])} for t in texts]
    human_ids = {r["id"] for r in map(json.loads, open(f"{OUT}/gap_texts.jsonl")) if r["arm"] == "human"}
    ref = finish_features(rows, human_ids)
    with open(f"{OUT}/features.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump({"train": a.train, "tokens": kn.ntok, "vocab": kn.V, "trigrams": len(kn.c3), **ref},
              open(f"{OUT}/features_meta.json", "w"), indent=1)
    print(f"features of {len(rows)} texts; human reference {ref}")


def step_scribe(a):
    texts = all_texts()
    tmp_in, tmp_out = f"{OUT}/.scribe_in.jsonl", f"{OUT}/.scribe_out.jsonl"
    with open(tmp_in, "w") as f:
        for t in texts:
            f.write(json.dumps({"id": t["id"], "set": "ws", "system": "ws", "text": t["text"], "prompt": t["prompt"]}) + "\n")
    e = a.enwik8
    # As scripts/gen_eval.sh's ladder does: `prepare --mode continuation` first (white space collapsed, the
    # unterminated tail cut, texts under 100 bytes dropped), so a cut mid-sentence is not scored as a flaw.
    tmp_prep = f"{OUT}/.scribe_prep.jsonl"
    subprocess.run([a.scribe, "prepare", "--in", tmp_in, "--out", tmp_prep, "--mode", "continuation"], check=True)
    os.replace(tmp_prep, tmp_in)
    subprocess.run([a.scribe, "score", "--pack", f"enwik:{e}:16:48", "--ref", f"enwik:{e}:48:56", "--vectors", a.vectors,
                    "--profile", "v31", "--in", tmp_in, "--out", tmp_out], check=True)
    with open(f"{OUT}/scribe.jsonl", "w") as f:
        for ln in open(tmp_out):
            r = json.loads(ln)
            f.write(json.dumps({"id": r["id"], "scribe_v2": r["scribe_v2"], "scribe_v31": r["scribe_v31"]}) + "\n")
    os.remove(tmp_in)
    os.remove(tmp_out)
    print(f"SCRIBE of {len(texts)} texts ({sum(1 for _ in open(f'{OUT}/scribe.jsonl'))} kept by prepare)")


def load_features():
    feats = {}
    for ln in open(f"{OUT}/features.jsonl"):
        r = json.loads(ln)
        feats[r["id"]] = r
    if os.path.exists(f"{OUT}/scribe.jsonl"):
        for ln in open(f"{OUT}/scribe.jsonl"):
            r = json.loads(ln)
            feats.setdefault(r["id"], {}).update(r)
    return feats


# ---------------------------------------------------------------------------------------------------------------
# Step: gap
# ---------------------------------------------------------------------------------------------------------------


def auc(pos, neg):
    """P(pos > neg), ties half (Mann-Whitney)."""
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank, i, rs = 1, 0, 0.0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        r = (rank + rank + (j - i) - 1) / 2
        rs += r * sum(1 for k in range(i, j) if allv[k][1])
        rank += j - i
        i = j
    n1, n0 = len(pos), len(neg)
    return (rs - n1 * (n1 + 1) / 2) / (n1 * n0)


def step_gap(a):
    feats = load_features()
    gap = [json.loads(ln) for ln in open(f"{OUT}/gap_texts.jsonl")]
    by_arm = collections.defaultdict(list)
    for r in gap:
        by_arm[r["arm"]].append(feats[r["id"]])
    arms = sorted(x for x in by_arm if x != "human")
    hum = by_arm["human"]
    writer = [f for x in arms for f in by_arm[x]]
    cur = by_arm.get(a.current, [])
    res = {"arms": arms, "current": a.current, "n_human": len(hum), "n_writer": len(writer), "signals": {}}
    lines = ["# Human against writer, GEN-A dev (cut near 460 characters)\n",
             f"Human: {len(hum)} texts. Writer: {len(writer)} texts from {len(arms)} dev GEN files. "
             f"Current default: {a.current} ({len(cur)} texts).\n",
             "AUC = P(a human text's value > a writer text's value), ties half; 0.5 = no separation. "
             "Raw values, not oriented by the registered direction.\n",
             "| signal | dir | human mean | writer mean | current mean | AUC human vs all writers | AUC human vs current |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for name, d, _ in SIGNALS + SCRIBE_SIGNALS:
        h = [f[name] for f in hum if name in f]
        w = [f[name] for f in writer if name in f]
        c = [f[name] for f in cur if name in f]
        if not h or not w:
            continue
        a1 = auc(h, w)
        a2 = auc(h, c) if c else float("nan")
        res["signals"][name] = {"dir": d, "human": st.mean(h), "writer": st.mean(w), "current": st.mean(c) if c else None,
                                "auc_all": a1, "auc_current": a2,
                                "per_arm": {x: st.mean(f[name] for f in by_arm[x] if name in f) for x in arms}}
        lines.append(f"| {name} | {d} | {st.mean(h):.3f} | {st.mean(w):.3f} | {st.mean(c) if c else float('nan'):.3f} | "
                     f"{a1:.3f} | {a2:.3f} |")
    for name in ("sl_mean", "sl_cv", "fw_share"):
        h = st.mean(f[name] for f in hum)
        w = st.mean(f[name] for f in writer)
        c = st.mean(f[name] for f in cur) if cur else float("nan")
        lines.append(f"| ({name}, raw) | | {h:.3f} | {w:.3f} | {c:.3f} | | |")
    open(f"{OUT}/gap.md", "w").write("\n".join(lines) + "\n")
    json.dump(res, open(f"{OUT}/gap.json", "w"), indent=1)
    print("\n".join(lines))


# ---------------------------------------------------------------------------------------------------------------
# Step: agree
# ---------------------------------------------------------------------------------------------------------------


def hit(fx, fy, d, pref):
    diff = (fx - fy) * (1 if d == "+" else -1)
    if diff == 0:
        return 0.5
    return 1.0 if (diff > 0) == (pref == 1.0) else 0.0


def boot(clusters, n=2000, seed=0):
    """clusters: list of (sum of hits, count). The pooled share and its 95% interval over resampled clusters."""
    rng = random.Random(seed)
    tot = sum(s for s, _ in clusters) / max(sum(c for _, c in clusters), 1)
    k = len(clusters)
    bs = []
    for _ in range(n):
        s = c = 0
        for _ in range(k):
            x = clusters[rng.randrange(k)]
            s += x[0]
            c += x[1]
        bs.append(s / c if c else 0.5)
    bs.sort()
    return tot, bs[int(0.025 * n)], bs[int(0.975 * n) - 1]


def agreement(js_, feats, name, d):
    cl = collections.defaultdict(lambda: [0.0, 0])
    for j in js_:
        fx, fy = feats[j["x"]].get(name), feats[j["y"]].get(name)
        if fx is None or fy is None:
            continue
        c = cl[(j["round"], j["dev"])]
        c[0] += hit(fx, fy, d, j["pref"])
        c[1] += 1
    return [tuple(v) for v in cl.values()]


def paired_vs(js_, feats, name, d, ref="scribe_v2", n=2000, seed=0):
    """Agreement of name minus agreement of ref on the same pairs (both defined), 95% cluster bootstrap."""
    cl = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for j in js_:
        fx, fy = feats[j["x"]].get(name), feats[j["y"]].get(name)
        rx, ry = feats[j["x"]].get(ref), feats[j["y"]].get(ref)
        if None in (fx, fy, rx, ry):
            continue
        c = cl[(j["round"], j["dev"])]
        c[0] += hit(fx, fy, d, j["pref"])
        c[1] += hit(rx, ry, "+", j["pref"])
        c[2] += 1
    v = list(cl.values())
    rng = random.Random(seed)
    tot = (sum(x[0] for x in v) - sum(x[1] for x in v)) / max(sum(x[2] for x in v), 1)
    bs = []
    for _ in range(n):
        smp = [v[rng.randrange(len(v))] for _ in range(len(v))]
        bs.append((sum(x[0] for x in smp) - sum(x[1] for x in smp)) / max(sum(x[2] for x in smp), 1))
    bs.sort()
    return tot, bs[int(0.025 * n)], bs[int(0.975 * n) - 1]


def rater_ceiling(judg):
    """Pair rounds: raters 1-2 (and 3-4) judged the same pairs. Share of pairs, both raters non-tie, where they agree."""
    by = collections.defaultdict(dict)
    for j in judg:
        if j["mode"] != "pair" or j["identical"]:
            continue
        pair = tuple(sorted([j["x"], j["y"]]))
        win = None if j["pref"] == 0.5 else (j["x"] if j["pref"] == 1.0 else j["y"])
        by[(j["round"], j["dev"], pair)][j["rater"]] = win
    cl = collections.defaultdict(lambda: [0.0, 0])
    for (rnd, dev, pair), v in by.items():
        rs = sorted(v)
        if len(rs) == 2 and v[rs[0]] is not None and v[rs[1]] is not None:
            c = cl[(rnd, dev)]
            c[0] += v[rs[0]] == v[rs[1]]
            c[1] += 1
    return [tuple(v) for v in cl.values()]


def step_agree(a):
    feats = load_features()
    judg = [json.loads(ln) for ln in open(f"{OUT}/judgments.jsonl")]
    usable = [j for j in judg if not j["identical"] and j["pref"] != 0.5]
    ww = [j for j in usable if not j["human"]]
    hw = [j for j in usable if j["human"]]
    rounds = sorted({j["round"] for j in judg})
    res = {"n_judgments": len(judg), "n_identical": sum(j["identical"] for j in judg),
           "n_ties": sum(j["pref"] == 0.5 and not j["identical"] for j in judg), "n_ww": len(ww), "n_hw": len(hw),
           "rounds": {r: sum(1 for j in ww if j["round"] == r) for r in rounds}, "signals": {}}
    ceil = boot(rater_ceiling(judg))
    res["rater_ceiling"] = ceil
    lines = ["# Agreement with the blind raters\n",
             f"Judgments {len(judg)}: identical texts {res['n_identical']}, rater ties {res['n_ties']}; "
             f"used: writer vs writer {len(ww)}, human vs writer {len(hw)}.\n",
             f"Rater against rater (pair rounds, same pair, both non-tie): {ceil[0]:.3f} [{ceil[1]:.3f}, {ceil[2]:.3f}].\n",
             "Share of rated pairs where the signal picks the rater's choice (signal tie = 0.5); 95% bootstrap over "
             "round x prompt clusters.\n",
             "| signal | dir | writer vs writer | minus SCRIBE v2, same pairs | human vs writer |", "|---|---|---|---|---|"]
    shown = [r for r in rounds if res["rounds"][r]]
    for name, d, _ in SIGNALS + SCRIBE_SIGNALS:
        aw = boot(agreement(ww, feats, name, d))
        ah = boot(agreement(hw, feats, name, d)) if hw else (float("nan"),) * 3
        per = {}
        for r in shown:
            cl = agreement([j for j in ww if j["round"] == r], feats, name, d)
            per[r] = (sum(s for s, _ in cl) / max(sum(c for _, c in cl), 1), sum(c for _, c in cl))
        dv = paired_vs(ww, feats, name, d)
        res["signals"][name] = {"dir": d, "ww": aw, "hw": ah, "vs_scribe_v2": dv, "per_round": per}
        lines.append(f"| {name} | {d} | {aw[0]:.3f} [{aw[1]:.3f}, {aw[2]:.3f}] | {dv[0]:+.3f} [{dv[1]:+.3f}, {dv[2]:+.3f}] | "
                     f"{ah[0]:.3f} [{ah[1]:.3f}, {ah[2]:.3f}] |")
    lines += ["\n## Per round, writer vs writer (share of pairs)\n",
              "| signal | " + " | ".join(shown) + " |", "|---|" + "---:|" * len(shown)]
    for name, _, _ in SIGNALS + SCRIBE_SIGNALS:
        per = res["signals"][name]["per_round"]
        lines.append(f"| {name} | " + " | ".join(f"{per[r][0]:.2f}" for r in shown) + " |")
    lines.append("| pairs | " + " | ".join(str(res["rounds"][r]) for r in shown) + " |")
    open(f"{OUT}/agree.md", "w").write("\n".join(lines) + "\n")
    json.dump(res, open(f"{OUT}/agree.json", "w"), indent=1)
    print("\n".join(lines))


# ---------------------------------------------------------------------------------------------------------------
# Step: explore (not registered; read after agree)
# ---------------------------------------------------------------------------------------------------------------


def logit_fit(X, y, l2=1.0, iters=500, lr=0.5):
    import numpy as np
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) / len(y) + l2 * w / len(y)
        w -= lr * g
    return w


def step_explore(a):
    """Exploratory, not ruled on: can a combination of the signals predict the raters on a round it never saw?
    Logistic regression on the pair's signal differences (no intercept, so swapping X and Y flips the prediction),
    leave one round out; and surp_mean without gq_pair, the one round whose writers count from different texts."""
    import numpy as np
    feats = load_features()
    judg = [json.loads(ln) for ln in open(f"{OUT}/judgments.jsonl")]
    ww = [j for j in judg if not j["identical"] and j["pref"] != 0.5 and not j["human"]]
    lines = ["# Exploratory (not registered)\n"]
    sets = {"text signals (15)": [n for n, _, _ in SIGNALS], "text signals + SCRIBE (17)": [n for n, _, _ in SIGNALS + SCRIBE_SIGNALS]}
    res = {}
    for label, names in sets.items():
        rows = [j for j in ww if all(feats[j[k]].get(n) is not None for k in ("x", "y") for n in names)]
        X = np.array([[feats[j["x"]][n] - feats[j["y"]][n] for n in names] for j in rows])
        sd = X.std(0)
        sd[sd == 0] = 1
        X = X / sd
        y = np.array([j["pref"] for j in rows])
        rnds = sorted({j["round"] for j in rows})
        pred = np.zeros(len(rows))
        for r in rnds:
            tr = np.array([j["round"] != r for j in rows])
            w = logit_fit(X[tr], y[tr])
            pred[~tr] = X[~tr] @ w
        hits = np.where(pred == 0, 0.5, ((pred > 0) == (y == 1)).astype(float))
        cl = collections.defaultdict(lambda: [0.0, 0])
        for j, h in zip(rows, hits):
            c = cl[(j["round"], j["dev"])]
            c[0] += h
            c[1] += 1
        b = boot([tuple(v) for v in cl.values()])
        per = {r: float(hits[[j["round"] == r for j in rows]].mean()) for r in rnds}
        wall = logit_fit(X, y)
        res[label] = {"loro": b, "per_round": per, "weights_all": dict(zip(names, map(float, wall))), "n": len(rows)}
        lines.append(f"**{label}**, leave one round out: {b[0]:.3f} [{b[1]:.3f}, {b[2]:.3f}] on {len(rows)} pairs. Per round: "
                     + ", ".join(f"{r} {v:.2f}" for r, v in per.items()) + ".\n")
        lines.append("Weights fitted on all rounds (standardised differences): "
                     + ", ".join(f"{n} {v:+.2f}" for n, v in sorted(zip(names, wall), key=lambda t: -abs(t[1]))) + ".\n")
    for name, d in (("surp_mean", "-"), ("scribe_v2", "+")):
        sub = [j for j in ww if j["round"] != "gq_pair"]
        b = boot(agreement(sub, feats, name, d))
        res[f"{name}_without_gq_pair"] = b
        lines.append(f"{name} without gq_pair: {b[0]:.3f} [{b[1]:.3f}, {b[2]:.3f}] ({len(sub)} pairs).\n")
    for grp, rs in (("plain K5 variants (llm_rating, llm_rating2, tw_pair, ng_pair, coh_pair, coh_long)",
                     {"llm_rating", "llm_rating2", "tw_pair", "ng_pair", "coh_pair", "coh_long"}),
                    ("planner and fill rounds (p1t_pair, planner_pair, q1t_pair, q3_pair, q3_long, fill_pair)",
                     {"p1t_pair", "planner_pair", "q1t_pair", "q3_pair", "q3_long", "fill_pair"})):
        sub = [j for j in ww if j["round"] in rs]
        for name, d in (("surp_mean", "-"), ("surp_sd", "+"), ("commit_text", "+"), ("scribe_v2", "+")):
            b = boot(agreement(sub, feats, name, d))
            res[f"{name}|{grp}"] = b
            lines.append(f"{grp}: {name} {b[0]:.3f} [{b[1]:.3f}, {b[2]:.3f}] ({len(sub)} pairs).\n")
    sub = [j for j in ww if j["round"] in {"tw_pair", "ng_pair", "coh_pair", "coh_long"}]
    for name, d in (("surp_mean", "-"), ("scribe_v2", "+"), ("scribe_v31", "+")):
        b = boot(agreement(sub, feats, name, d))
        dv = paired_vs(sub, feats, name, d)
        res[f"{name}|today"] = {"agree": b, "minus_scribe_v2": dv}
        lines.append(f"today's writer, K5 from 337 MB (tw_pair, ng_pair, coh_pair, coh_long): {name} {b[0]:.3f} "
                     f"[{b[1]:.3f}, {b[2]:.3f}], minus SCRIBE v2 {dv[0]:+.3f} [{dv[1]:+.3f}, {dv[2]:+.3f}] ({len(sub)} pairs).\n")
    open(f"{OUT}/explore.md", "w").write("\n".join(lines))
    json.dump(res, open(f"{OUT}/explore.json", "w"), indent=1)
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["judgments", "gapset", "features", "scribe", "gap", "agree", "explore", "all"])
    p.add_argument("--train", default="/mnt/d/cypha_llm/fuzzy/train.txt")
    p.add_argument("--scribe", default=os.path.expanduser("~/gq_bin/scribe"))
    p.add_argument("--enwik8", default="/mnt/d/cypha_llm/enwik8")
    p.add_argument("--vectors", default="/mnt/d/cypha_llm/scribe/v32")
    p.add_argument("--current", default="k5_coherence/L1", help="the current default's dev GEN file (gap table)")
    a = p.parse_args()
    steps = ["judgments", "gapset", "features", "scribe", "gap", "agree", "explore"] if a.step == "all" else [a.step]
    for s in steps:
        globals()[f"step_{s}"](a)


if __name__ == "__main__":
    main()
