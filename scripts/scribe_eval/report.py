"""Results driver for the SCRIBE paper: every analysis of the plan, through meta.py.

Author: Odin Loch

Runs each analysis of docs/SCRIBE_PAPER_PLAN.md §5 (the list in
build_analyses) on the files that exist, and writes <out>/<analysis>.md,
.csv and .json, plus <out>/index.md listing every output. An analysis whose
required inputs are missing is skipped and listed in index.md with the reason;
optional inputs that are missing (a baseline family that has not finished) are
listed next to the analysis, which runs without them. Partial files
(*.partial.jsonl) are never read. Rerun it as files land: every output is
rewritten.

    python scripts/scribe_eval/report.py                 # everything that can run
    python scripts/scribe_eval/report.py --list          # what would run, what would be skipped
    python scripts/scribe_eval/report.py --only hanna_corr wiki_fresh_scribe

Inputs live under --sp (default $SCRIBE_SP, else the session scratchpad):
D = <sp>/data, RUN = <sp>/run, R = <sp>/run/baselines.

Metrics: METRICS is the one table of metric names, directions and labels
(plus hanna_auto_metric, the rule for the 72 automatic metrics shipped with
HANNA). SCRIBE files of different variants (v2, v2p, two-sided, no I1) share
field names, so each source renames its fields (the scribe_v2 field of a
.v2p.jsonl file becomes scribe_v2p). The merged scores of each analysis go to
a work directory (--work), where meta.py reads them.
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import meta  # noqa: E402  (first: it limits BLAS threads before numpy loads)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = HERE.parents[1]
DEFAULT_SP = os.environ.get(
    "SCRIBE_SP", "/tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad")
DEFAULT_OUT = REPO / "paper" / "scribe" / "results"


# ---------------------------------------------------------------------------
# Metrics: name -> (direction, label). Table rows follow this order.
# ---------------------------------------------------------------------------

METRICS = {
    # SCRIBE (merged names; see scribe())
    "scribe_v2": ("+", "SCRIBE v2"),
    "scribe_v2p": ("+", "SCRIBE v2p"),
    "scribe_v2ts": ("+", "SCRIBE v2, two-sided (exploratory)"),
    "scribe_v2_noi1": ("+", "SCRIBE v2 without I1"),
    "scribe_v1": ("+", "SCRIBE v1"),
    "surface_s": ("+", "surface S"),
    "v2_D": ("-", "v2 deviance D"),
    "v2_DL": ("-", "v2 D_L"),
    "v2_DS": ("-", "v2 D_S"),
    "v2_DC": ("-", "v2 D_C"),
    "v2_DI": ("-", "v2 D_I"),
    "v2_DR": ("-", "v2 D_R"),
    "v2_DI_flag": ("-", "share with D_I > 0"),
    # per-text baselines (baselines.py, judge.py)
    "distinct_1": ("+", "distinct-1"),
    "distinct_2": ("+", "distinct-2"),
    "distinct_3": ("+", "distinct-3"),
    "distinct_4": ("+", "distinct-4"),
    "rep_2": ("-", "rep-2"),
    "rep_3": ("-", "rep-3"),
    "rep_4": ("-", "rep-4"),
    "seq_rep_4": ("-", "seq-rep-4"),
    "diversity": ("+", "diversity"),
    "ttr": ("+", "type-token ratio"),
    "gzip_ratio": ("+", "gzip ratio"),
    "ppl_gpt2": ("-", "GPT-2 PPL"),
    "ppl_cond_gpt2": ("-", "GPT-2 PPL given prompt"),
    "ppl_gpt2_large": ("-", "GPT-2-large PPL"),
    "ppl_cond_gpt2_large": ("-", "GPT-2-large PPL given prompt"),
    "cola_mean": ("+", "CoLA acceptability, mean"),
    "cola_min": ("+", "CoLA acceptability, min"),
    "coh_adjacent": ("+", "MiniLM coherence, adjacent"),
    "coh_allpairs": ("+", "MiniLM coherence, all pairs"),
    "prompt_cos": ("+", "MiniLM prompt cosine"),
    "det_p_human": ("+", "GPT-2 detector P(human)"),
    "judge_score": ("+", "LLM judge (Qwen2.5-1.5B)"),
    # set level (baselines.py setlevel); zipf_gap = |Zipf - mean Zipf of the human sets|
    "mauve": ("+", "MAUVE"),
    "self_bleu": ("-", "self-BLEU"),
    "zipf_gap": ("-", "|Zipf − human Zipf|"),
    # shipped with HANNA (selection and labels of run/meta/hanna_metrics.json)
    "BLEU Ξ§": ("+", "BLEU"),
    "ROUGE-L F-Score Ξ§": ("+", "ROUGE-L"),
    "METEOR Ξ§": ("+", "METEOR"),
    "chrF Ξ§": ("+", "chrF"),
    "BERTScore F1 Ξε": ("+", "BERTScore F1"),
    "MoverScore Ξε": ("+", "MoverScore"),
    "BARTScore-SH ΞΔ": ("+", "BARTScore (s->h)"),
    "BaryScore-W Ξε": ("-", "BaryScore-W"),
    "SUPERT-PS ¤ε": ("+", "SUPERT (prompt)"),
    "BARTScore-PS ¤Δ": ("+", "BARTScore (prompt->story)"),
    "Repetition-3 ¤§": ("-", "Repetition-3"),
    "Novelty-1 ¤§": ("+", "Novelty-1"),
    "Text length ¤§": ("+", "Text length"),
    "ChatGPT AVG 1": ("+", "ChatGPT (avg)"),
    "Mistral-7B AVG 1": ("+", "Mistral-7B (avg)"),
    "Llama-13B AVG 1": ("+", "Llama-13B (avg)"),
    "Beluga-13B AVG 1": ("+", "Beluga-13B (avg)"),
}

HANNA_SHORT = [k for k in METRICS if re.search(r"(Ξ|¤)\S*$|AVG 1$", k)]
HANNA_MARK = re.compile(r"\s+(Ξ§|Ξε|ΞΔ|¤§|¤ε|¤Δ)$")
HANNA_LLM = re.compile(r"^(ChatGPT|Mistral-7B|Llama-13B|Beluga-13B|OrcaPlatypus) ")


def hanna_auto_metric(name):
    """Direction and label of an automatic metric shipped with HANNA: higher is
    better except distances (BaryScore, DepthScore, InfoLM Fisher-Rao) and
    Repetition-n. Coverage, Density and Compression have no natural direction
    and are taken as higher is better."""
    if name in METRICS:
        return METRICS[name]
    label = HANNA_MARK.sub("", name)
    lower = label.startswith(("BaryScore", "DepthScore", "InfoLM", "Repetition"))
    return ("-" if lower else "+", label)


SCRIBE_FIELDS = {"scribe_v2": "scribe_v2", "scribe_v1": "scribe_v1", "surface_s": "surface_s"}
SCRIBE_PARTS = {"v2.D": "v2_D", "v2.DL": "v2_DL", "v2.DS": "v2_DS", "v2.DC": "v2_DC",
                "v2.DI": "v2_DI", "v2.DR": "v2_DR"}
VARIANT = {"v2p": "scribe_v2p", "ts": "scribe_v2ts", "noi1": "scribe_v2_noi1"}

# Baseline families (globs; partial files never match). The plan in force (baselines PROGRESS.md,
# 23:48, "minimal set") has no gpt2-large, no LLM judge and no ppl_cond; families outside it are
# read when present but not listed as missing.
PER_TEXT = ["stats.jsonl", "ppl_gpt2.jsonl", "cola.jsonl", "coherence.jsonl", "detector.jsonl"]
EXTRA = ["ppl_gpt2_large.jsonl", "judge_*.jsonl"]
JUDGE = ["judge_*.jsonl"]


def fams(folder, expected, optional=()):
    """Baseline specs: expected families (listed when missing) and optional ones."""
    rest = [g for g in PER_TEXT + EXTRA if g not in expected and g not in optional] + list(optional)
    return [(folder, list(expected), False), (folder, rest, None)]

LADDER = ["human", "sent_shuffled", "word_shuffled", "random_lexicon", "random_letters", "repeated_phrase"]
LABELS = {"human": "held-out human", "sent_shuffled": "sentences shuffled",
          "word_shuffled": "words shuffled", "random_lexicon": "random lexicon words",
          "random_letters": "random letters", "repeated_phrase": "one phrase repeated",
          "trie": "trie samples (pooled)"}
GPT2_LABELS = {"human": "WebText (human)", "xl-1542M": "GPT-2 xl", "large-762M": "GPT-2 large",
               "medium-345M": "GPT-2 medium", "small-117M": "GPT-2 small"}
GPT2_LABELS.update({k + "-k40": v + ", top-k 40" for k, v in list(GPT2_LABELS.items()) if k != "human"})

# numbers known from earlier runs, checked after every report run
KNOWN = {"wiki_fresh_scribe": {"scribe_v2": [0.975, 0.999, 0.998, 0.998, 1.000],
                               "surface_s": [0.627, 0.341, 0.750, 1.000, 0.559]}}
MAUVE_PUBLISHED = ("MAUVE paper (Pillutla et al. 2021), published system-level Spearman between MAUVE "
                   "(GPT-2-large features) and the Bradley-Terry scores over the 8 models, judgments with "
                   "te > 25 s: q1 0.810, q2 0.857, q3 0.952.")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass
class Src:
    """One score file: fields (dotted) -> merged metric names; None takes every
    top-level field named in METRICS. derive adds computed metrics."""
    path: Path
    fields: dict = None
    derive: dict = None
    required: bool = True


def scribe(path, variant="v2", parts=False, required=True, derive=None):
    if variant == "v2":
        f = dict(SCRIBE_FIELDS)
        if parts:
            f.update(SCRIBE_PARTS)
    else:
        f = {"scribe_v2": VARIANT[variant]}
    return Src(Path(path), f, derive, required)


def di_flag(rec):
    v = meta.get_path(rec, "v2.DI")
    return None if v is None else float(float(v) > 0)


@dataclass
class Analysis:
    name: str                       # output stem
    section: str                    # heading in index.md
    title: str                      # one line
    cmd: str                        # meta.py subcommand, or "profile"
    texts: Path = None
    scribe: list = field(default_factory=list)       # Src
    extra: list = field(default_factory=list)        # Src (e.g. shipped metrics)
    baselines: list = field(default_factory=list)    # (folder under R, [globs], need): need True =
    #                                                  required, False = expected (listed when missing),
    #                                                  None = used if present, not listed
    need_baselines: object = False  # skip when no baseline file exists yet (True, or a folder name)
    systems: object = None          # list of names, or "auto"
    labels: dict = None
    args: list = field(default_factory=list)
    pairs: Path = None
    pairs_limit: int = None
    pairs_te: Path = None           # raw MAUVE CSV: attach answer times
    transform: object = None        # text record -> record
    setlevel: list = field(default_factory=list)     # (path, [fields], required)
    post: object = None             # writes the Markdown from meta's JSON
    note: str = ""


def read_jsonl(path):
    return meta.read_jsonl(path)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def find_baselines(R, specs):
    """Existing final files for the (folder, globs, required) specs, and what is missing."""
    found, missing, missing_req = [], [], []
    for folder, globs, req in specs:
        for g in globs:
            hits = sorted(p for p in (R / folder).glob(g) if not p.name.endswith(".partial.jsonl"))
            if hits:
                found += [Src(p, None, None, False) for p in hits]
            elif req is not None:
                partial = list((R / folder).glob(g.replace(".jsonl", ".partial.jsonl")))
                tag = f"{folder}/{g}" + (" (partial)" if partial else "")
                (missing_req if req else missing).append(tag)
    return found, missing, missing_req


def merge_scores(ids, sources):
    """id -> {metric: value} for the given ids; the first source wins on a clash."""
    out = {i: {} for i in ids}
    clashes = {}
    for s in sources:
        if not s.path.exists():
            continue
        for rec in read_jsonl(s.path):
            i = str(rec.get("id"))
            if i not in out:
                continue
            if s.fields is None:
                items = [(k, k) for k in rec if k in METRICS]
            else:
                items = list(s.fields.items())
            row = out[i]
            for fld, name in items:
                v = num(rec[fld] if fld in rec else meta.get_path(rec, fld))
                if name in row:
                    if v is not None and row[name] is not None and abs(v - row[name]) > 1e-9 * max(1, abs(v)):
                        clashes[name] = clashes.get(name, 0) + 1
                    continue
                row[name] = v
            for name, fn in (s.derive or {}).items():
                row.setdefault(name, fn(rec))
    return out, clashes


def metric_config(scores, extra_labels=None):
    """Metrics with at least one value, in METRICS order, as a meta.py config."""
    present = []
    for row in scores.values():
        for k, v in row.items():
            if v is not None and k not in present:
                present.append(k)
    order = {k: i for i, k in enumerate(METRICS)}
    present.sort(key=lambda k: (order.get(k, len(order)), k))
    cfg = []
    for k in present:
        d, lab = METRICS.get(k) or (extra_labels or {}).get(k) or hanna_auto_metric(k)
        cfg.append({"name": k, "dir": d, "label": lab})
    return cfg


def systems_arg(names, labels=None):
    labels = labels or {}
    return ";".join(f"{n}={labels.get(n, n)}" for n in names)


def auto_systems(texts):
    seen = []
    for r in texts:
        s = str(r.get("system"))
        if s not in seen:
            seen.append(s)
    if "human" in seen:
        seen.remove("human")
        seen.insert(0, "human")
    return seen


def attach_te(pairs, csv_path):
    """Answer times from the raw MAUVE CSV. The pairs file is the CSV row for row
    (pair_id = mauve_he/pair/<HITId>); every row is checked before use."""
    df = pd.read_csv(csv_path, index_col=0)
    if len(df) < len(pairs):
        raise ValueError("the MAUVE CSV has fewer rows than the pairs file")
    by_hit = {int(h): r for h, r in zip(df["HITId"], df.to_dict("records"))}
    out = []
    for p in pairs:
        h = int(str(p["pair_id"]).rsplit("/", 1)[-1])
        r = by_hit.get(h)
        if r is None or int(r["Input.idx"]) != int(p["idx"]) or r["WorkerId"] != p["worker"] or \
                any(str(r[f"Answer.{q}"]) != str(p[q]) for q in ("q1", "q2", "q3")):
            raise ValueError(f"pair {p['pair_id']} does not match row HITId {h} of the MAUVE CSV")
        out.append(dict(p, te=float(r["Answer.te"])))
    return out


# rating layouts -> meta.py's {criterion: {"mean", "raters"}}

def hanna_ratings(rec):
    m = dict(rec.get("meta") or {})
    hum, raters = m.get("human") or {}, m.get("raters") or {}
    m["human"] = {c: {"mean": v, "raters": raters.get(c, [])} for c, v in hum.items()}
    return dict(rec, meta=m)


def openmeva_ratings(rec):
    m = dict(rec.get("meta") or {})
    h = m.get("human") or {}
    m["human"] = {"overall": {"mean": h.get("overall"), "raters": h.get("scores", [])}}
    return dict(rec, meta=m)


def load_setlevel(specs):
    """Merge set-level rows of several files on (group, subset); add zipf_gap."""
    rows, used = {}, []
    for path, fields, _ in specs:
        if not path.exists():
            continue
        used.append(path)
        for r in read_jsonl(path):
            key = (str(r.get("group")), r.get("subset"))
            row = rows.setdefault(key, {"group": key[0], "subset": key[1]})
            for f in fields:
                if f in r and f not in row:
                    row[f] = num(r[f])
    vals = list(rows.values())
    hz = [r["zipf"] for r in vals if r["group"] == "human" and r.get("zipf") is not None]
    if hz:
        mz = float(np.mean(hz))
        for r in vals:
            if r.get("zipf") is not None:
                r["zipf_gap"] = abs(r["zipf"] - mz)
    return vals, used


# ---------------------------------------------------------------------------
# The analyses
# ---------------------------------------------------------------------------

def build_analyses(P):
    D, RUN, R = P.D, P.RUN, P.R
    S = R / "subsets"
    A = []
    lad = ["--systems", systems_arg(LADDER, LABELS)]

    # 1. fresh Wikipedia ladder
    F = RUN / "fresh" / "wiki_fresh"
    fs = F / "scores.jsonl"
    sec = "1. Fresh Wikipedia control ladder"
    A.append(Analysis("wiki_fresh_scribe", sec, "SCRIBE v2, v1, surface S and the v2 deviance parts; "
                      "500 texts per rung", "ladder", F / "texts.jsonl", [scribe(fs, parts=True)], args=lad))
    A.append(Analysis("wiki_fresh_W_all", sec, "All per-text metrics on subset W (150 texts per rung)",
                      "ladder", S / "wiki_fresh_W.jsonl", [scribe(fs)],
                      baselines=fams("wiki_fresh_W", PER_TEXT), need_baselines=True, args=lad))
    A.append(Analysis("wiki_fresh_W60_judge", sec, "The LLM judge and every other metric on W60 "
                      "(60 texts per rung)", "ladder", S / "wiki_fresh_W60.jsonl", [scribe(fs)],
                      baselines=[("wiki_fresh_W60", JUDGE, True)] + fams("wiki_fresh_W", PER_TEXT),
                      args=lad))
    mv = R / "wiki_fresh_mauve"
    A.append(Analysis("wiki_fresh_W_setlevel", sec, "Set-level metrics on W, one set per rung (q: 100 "
                      "texts per rung; MAUVE with GPT-2 small features against 100 other human texts)",
                      "setladder", setlevel=[(R / "wiki_fresh_W_mauve" / "setlevel.jsonl",
                                              ["mauve", "self_bleu", "zipf"], True)], args=lad))
    A.append(Analysis("wiki_fresh_setlevel", sec, "Set-level metrics, one set of 250 texts per rung "
                      "(MAUVE with GPT-2-large features, reference: 250 other human texts)", "setladder",
                      setlevel=[(mv / "setlevel.jsonl", ["mauve", "self_bleu", "zipf"], True)], args=lad))
    A.append(Analysis("wiki_fresh_setlevel_K5", sec, "Set-level metrics on 5 disjoint subsets of 50 "
                      "texts per rung", "setladder",
                      setlevel=[(mv / "setlevel_K5.jsonl", ["mauve", "self_bleu", "zipf"], True)], args=lad))
    trie_order = ["trie"] + LADDER
    A.append(Analysis("wiki_fresh_trie", sec, "Where the order-4 trie falls: P(trie > control rung), "
                      "SCRIBE on all texts", "ladder", F / "texts.jsonl", [scribe(fs, parts=True)],
                      args=["--systems", systems_arg(trie_order, LABELS)], post=post_trie))
    A.append(Analysis("wiki_fresh_W_trie", sec, "Where the trie falls, all per-text metrics on W", "ladder",
                      S / "wiki_fresh_W.jsonl", [scribe(fs)], baselines=fams("wiki_fresh_W", PER_TEXT),
                      need_baselines=True, args=["--systems", systems_arg(trie_order, LABELS)],
                      post=post_trie))

    # 2. genre ladders
    for g in ("web", "books"):
        G = RUN / "genre" / g
        A.append(Analysis(f"genre_{g}", "2. Genre control ladders", f"{g.capitalize()} ladder, each genre "
                          f"with its own pack and reference, SCRIBE only", "ladder", G / "texts.jsonl",
                          [scribe(G / "scores.jsonl")], args=lad))

    # 3. dose-response
    C = RUN / "fresh"
    sec = "3. Held-out corruptions, dose-response"
    A.append(Analysis("dose_scribe", sec, "SCRIBE on all corruptions (9 families x 6 strengths x 200 "
                      "texts)", "dose", C / "corrupt.jsonl", [scribe(C / "corrupt.scribe.jsonl")]))
    A.append(Analysis("dose_S20_all", sec, "All per-text metrics on S20 (every family and strength of "
                      "the first 20 sources)", "dose", S / "corrupt_S20.jsonl",
                      [scribe(C / "corrupt.scribe.jsonl")], baselines=fams("corrupt_S20", PER_TEXT),
                      need_baselines=True))
    A.append(Analysis("dose_S40_all", sec, "All per-text metrics on S40 (earlier plan; runs if its files "
                      "appear)", "dose", S / "corrupt_S40.jsonl", [scribe(C / "corrupt.scribe.jsonl")],
                      baselines=fams("corrupt_S40", []) + [("corrupt", ["stats.jsonl"], None)],
                      need_baselines="corrupt_S40"))
    A.append(Analysis("dose_S10_judge", sec, "The LLM judge and every other metric on S10 (first 10 "
                      "sources)", "dose", S / "corrupt_S10.jsonl", [scribe(C / "corrupt.scribe.jsonl")],
                      baselines=[("corrupt_S10", JUDGE, True)] + fams("corrupt_S20", [])
                      + fams("corrupt_S40", []) + [("corrupt", ["stats.jsonl"], None)]))

    # 4. MAUVE human evaluation
    M = D / "mauve_he"
    msc = [scribe(RUN / "human" / "mauve_he.v2.jsonl"), scribe(RUN / "human" / "mauve_he.v2p.jsonl", "v2p"),
           scribe(RUN / "ts" / "mauve_he.ts.jsonl", "ts", required=False)]
    csv = P.SP / "dl" / "mauve_he" / "mauve_human_eval.csv"
    sec = "4. MAUVE human evaluation (pairwise)"
    A.append(Analysis("mauve_he_scribe", sec, "SCRIBE on all 3,240 judgments", "pairs", M / "texts.jsonl",
                      msc, pairs=M / "pairs.jsonl", note=MAUVE_PUBLISHED))
    A.append(Analysis("mauve_he_scribe_te25", sec, "SCRIBE, judgments with an estimated answer time over "
                      "25 s only (the MAUVE paper's filter; times from the raw CSV)", "pairs",
                      M / "texts.jsonl", msc, pairs=M / "pairs.jsonl", pairs_te=csv, note=MAUVE_PUBLISHED))
    A.append(Analysis("mauve_he_M300_all", sec, "All per-text metrics on M300 (first 300 pairs)", "pairs",
                      M / "texts.jsonl", msc, pairs=M / "pairs.jsonl", pairs_limit=300,
                      baselines=fams("mauve_he_M300", ["stats.jsonl", "ppl_gpt2.jsonl", "detector.jsonl"])
                      + [("mauve_he", ["stats.jsonl"], None)], need_baselines="mauve_he_M300",
                      note=MAUVE_PUBLISHED))
    A.append(Analysis("mauve_he_M_all", sec, "All per-text metrics on M (first 1,000 pairs; only stats "
                      "under the plan in force)", "pairs", M / "texts.jsonl", msc, pairs=M / "pairs.jsonl",
                      pairs_limit=1000, baselines=fams("mauve_he_M", ["stats.jsonl"]) + [("mauve_he", ["stats.jsonl"], None)],
                      need_baselines=True, note=MAUVE_PUBLISHED))
    A.append(Analysis("mauve_he_M300_judge", sec, "The LLM judge and every other metric on M300 (first "
                      "300 pairs)", "pairs", M / "texts.jsonl", msc, pairs=M / "pairs.jsonl",
                      pairs_limit=300, baselines=[("mauve_he_M300", JUDGE, True)] + fams("mauve_he_M300", [])
                      + [("mauve_he", ["stats.jsonl"], None)], note=MAUVE_PUBLISHED))

    # 5. per-item human ratings
    sec = "5. Human ratings (HANNA, OpenMEVA)"
    hsc = [scribe(RUN / "human" / "hanna.v2.jsonl"), scribe(RUN / "human" / "hanna.v2p.jsonl", "v2p"),
           scribe(RUN / "ts" / "hanna.ts.jsonl", "ts", required=False)]
    shipped = D / "hanna" / "shipped_metrics.jsonl"
    hargs = ["--criteria", "RE,CH,EM,SU,EG,CX", "--human-system", "human"]
    A.append(Analysis("hanna_corr", sec, "HANNA, 6 criteria, with and without human stories: SCRIBE, our "
                      "baselines, 13 shipped automatic metrics and 4 shipped LLM judges", "corr",
                      D / "hanna" / "texts.jsonl", hsc,
                      extra=[Src(shipped, {k: k for k in HANNA_SHORT}, None, True)],
                      baselines=fams("hanna", ["stats.jsonl", "ppl_gpt2.jsonl"]), transform=hanna_ratings, args=hargs,
                      note=HANNA_NOTE))
    auto = hanna_auto_names(shipped)
    A.append(Analysis("hanna_corr_full", sec, f"HANNA, SCRIBE against all {len(auto) or 72} shipped "
                      f"automatic metrics", "corr", D / "hanna" / "texts.jsonl", hsc[:1],
                      extra=[Src(shipped, {k: k for k in auto}, None, True)], transform=hanna_ratings,
                      args=hargs, note=HANNA_NOTE))
    for c in ("roc", "wp"):
        osc = [scribe(RUN / "human" / f"mans_{c}.v2.jsonl"), scribe(RUN / "human" / f"mans_{c}.v2p.jsonl", "v2p"),
               scribe(RUN / "ts" / f"mans_{c}.ts.jsonl", "ts", required=False)]
        A.append(Analysis(f"mans_{c}_corr", sec, f"OpenMEVA MANS {c.upper()}, overall quality (5 ratings "
                          f"per story)", "corr", D / "openmeva" / f"mans_{c}.jsonl", osc,
                          baselines=fams(f"mans_{c}", ["stats.jsonl", "ppl_gpt2.jsonl"]), transform=openmeva_ratings,
                          args=["--criteria", "overall"]))

    # 6. model ladders
    sec = "6. Model ladders (known order)"
    G = RUN / "gpt2out"
    gsc = [scribe(G / "samples.scribe.jsonl"), scribe(G / "samples.ts.jsonl", "ts", required=False)]
    pure = ["human", "xl-1542M", "large-762M", "medium-345M", "small-117M"]
    k40 = ["human"] + [s + "-k40" for s in pure[1:]]
    for tag, order, what in (("pure", pure, "pure sampling"), ("k40", k40, "top-k 40")):
        sa = ["--systems", systems_arg(order, GPT2_LABELS)]
        A.append(Analysis(f"gpt2out_{tag}_scribe", sec, f"GPT-2 output dataset, {what}: human > xl > large "
                          f"> medium > small, SCRIBE on 1,000 texts per system", "models",
                          G / "samples.jsonl", gsc, args=sa))
        A.append(Analysis(f"gpt2out_{tag}_200_all", sec, f"GPT-2 output dataset, {what}, all per-text "
                          f"metrics on gpt2out_200", "models", S / "gpt2out_200.jsonl", gsc,
                          baselines=fams("gpt2out_200", ["stats.jsonl", "ppl_gpt2.jsonl"]),
                          need_baselines="gpt2out_200", args=sa))
        A.append(Analysis(f"gpt2out_{tag}_500_all", sec, f"GPT-2 output dataset, {what}, per-text metrics "
                          f"on gpt2out_500 (stats only under the plan in force)", "models",
                          S / "gpt2out_500.jsonl", gsc, baselines=fams("gpt2out_500", ["stats.jsonl"]),
                          need_baselines="gpt2out_500", args=sa))
        A.append(Analysis(f"gpt2out_{tag}_setlevel", sec, f"GPT-2 output dataset, {what}, set-level "
                          f"metrics (one set per system: self-BLEU and Zipf on gpt2out_200; MAUVE with "
                          f"GPT-2 small features if run)", "models", args=sa,
                          setlevel=[(R / "gpt2out_200" / "setlevel.jsonl", ["self_bleu", "zipf"], False),
                                    (R / "gpt2out_mauve" / "setlevel.jsonl", ["mauve"], False)]))
    Mo = RUN / "models"
    A.append(Analysis("ngram_ladder", sec, "KN word n-grams: human > 5 > 4 > 3 > 2 > 1", "models",
                      Mo / "ladder_ngram.fresh.jsonl", [scribe(Mo / "ladder_ngram.fresh.scribe.jsonl")],
                      args=["--systems", systems_arg(["human", "ngram5", "ngram4", "ngram3", "ngram2",
                                                      "ngram1"])]))
    Cy = RUN / "cyphalm"
    A.append(Analysis("cyphalm_cls", sec, "CyphaLM: human, trie, trie + class expert (no known order: "
                      "means and pairwise accuracies)", "ladder", Cy / "cls_prepared.jsonl",
                      [scribe(Cy / "cls.scribe.jsonl"), scribe(Cy / "cls.scribe.ts.jsonl", "ts", required=False)],
                      systems=["human", "trie+cls:shape", "trie"], post=post_means))
    for lad_name, order, what in (
            ("pythia_steps", ["human", "step143000", "step64000", "step16000", "step4000", "step1000"],
             "Pythia-410M training checkpoints: later > earlier"),
            ("pythia_size", ["human", "pythia-1.4b", "pythia-1b", "pythia-410m", "pythia-160m", "pythia-70m"],
             "Pythia sizes: larger > smaller")):
        A.append(Analysis(lad_name, sec, what, "models", Mo / f"{lad_name}.jsonl",
                          [scribe(Mo / f"{lad_name}.scribe.jsonl")], args=["--systems", systems_arg(order)]))
    A.append(Analysis("gpt2_decoding", sec, "GPT-2-large decoding (no known order): means, pairwise "
                      "accuracies and the share of texts flagged degenerate (D_I > 0)", "ladder",
                      Mo / "gpt2_decoding.jsonl",
                      [scribe(Mo / "gpt2_decoding.scribe.jsonl", derive={"v2_DI_flag": di_flag})],
                      systems="auto", post=post_means))

    # 7. attacks
    At = RUN / "attacks"
    A.append(Analysis("attacks", "7. Attacks", "Mean per attack and parity with human text (mean / mean "
                      "of human); P(human > attack)", "ladder", At / "attacks.jsonl",
                      [scribe(At / "attacks.v2.jsonl"), scribe(At / "attacks.v2p.jsonl", "v2p"),
                       scribe(At / "attacks.v2_noi1.jsonl", "noi1")],
                      systems="auto", post=post_attacks))
    A.append(Analysis("attacks_50_all", "7. Attacks", "The same on attacks_50 (first 50 prompts) with "
                      "every baseline", "ladder", S / "attacks_50.jsonl",
                      [scribe(At / "attacks.v2.jsonl"), scribe(At / "attacks.v2p.jsonl", "v2p"),
                       scribe(At / "attacks.v2_noi1.jsonl", "noi1")],
                      baselines=fams("attacks_50", ["stats.jsonl", "ppl_gpt2.jsonl", "detector.jsonl"]),
                      need_baselines="attacks_50", systems="auto", post=post_attacks))

    # 8. length
    sec = "8. Correlation with length"
    largs = ["--where", "system=human|human_short", "--by", "system"]
    A.append(Analysis("length_wiki_scribe", sec, "Held-out human text, standard (human) and short "
                      "(human_short) buckets and pooled, SCRIBE", "length", F / "texts.jsonl", [scribe(fs)],
                      args=largs))
    A.append(Analysis("length_wiki_W_all", sec, "Held-out human text on W (150 per bucket), all per-text "
                      "metrics", "length", S / "wiki_fresh_W.jsonl", [scribe(fs)],
                      baselines=fams("wiki_fresh_W", PER_TEXT), need_baselines=True, args=largs))

    # 9. Q/A ladder
    Q = RUN / "qa"
    A.append(Analysis("qa_ladder", "9. Q/A ladder", "correct > other answer > prompt echo > shuffled "
                      "words", "ladder", Q / "test.jsonl",
                      [scribe(Q / "test.v2.jsonl"), scribe(Q / "test.v2p.jsonl", "v2p")],
                      args=["--systems", systems_arg(["correct", "other", "echo", "shuffled"])]))

    # 10. profiling
    A.append(Analysis("profiling", "10. Ablations and cost", "Ablation ladder table (as explore/abl.py), "
                      "TIMING and peak RAM of SCRIBE, throughput of every baseline", "profile"))
    return A


HANNA_NOTE = ("Shipped metrics: Ξ marks a metric that compares with the human reference story (so it is "
              "degenerate on the human stories themselves), ¤ a reference-free one. Directions: higher is "
              "better except BaryScore, DepthScore, InfoLM Fisher-Rao (distances) and Repetition-n; "
              "Coverage, Density and Compression have no natural direction and are taken as higher is better.")


def hanna_auto_names(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        first = json.loads(f.readline())
    return [k for k in first if k not in ("id", "story_id", "model") and not HANNA_LLM.match(k)]


# ---------------------------------------------------------------------------
# Custom Markdown built from meta.py's JSON
# ---------------------------------------------------------------------------

def load_tables(json_path):
    return json.loads(Path(json_path).read_text(encoding="utf-8"))["tables"]


def g(row, key):
    v = row.get(key)
    return math.nan if v is None else v


def post_trie(a, rep, tables, cfg):
    """P(trie > control) for every control rung."""
    pairs = [r for r in tables["pairs"] if r["upper"] == "trie"]
    controls = []
    for r in pairs:
        if r["lower"] not in controls:
            controls.append(r["lower"])
    rows = []
    for m in cfg:
        cells = []
        for c in controls:
            r = next((x for x in pairs if x["metric"] == m["name"] and x["lower"] == c), None)
            cells.append(meta.fmci(g(r, "acc"), g(r, "acc_lo"), g(r, "acc_hi")) if r else "n/a")
        rows.append([m["label"]] + cells)
    rep.p("Share of (trie, control) pairs in which the trie text scores higher (ties one half), with 95% "
          "bootstrap intervals. 0.5 is no information; the trie writes word salad, so a correct metric "
          "puts it below fluent rungs.")
    rep.table(["metric"] + [f"trie > {LABELS.get(c, c)}" for c in controls], rows)
    rung = [r for r in tables["rungs"]]
    rep.p("Rung means [95% interval]:")
    names = []
    for r in rung:
        if r["rung"] not in names:
            names.append(r["rung"])
    rep.table(["metric"] + [LABELS.get(n, n) for n in names],
              [[m["label"]] + [meta.fmci(g(r, "mean"), g(r, "lo"), g(r, "hi")) for n in names
                               for r in rung if r["metric"] == m["name"] and r["rung"] == n] for m in cfg])


def post_means(a, rep, tables, cfg):
    """Means per system and P(row > column) for every pair of systems."""
    rung = tables["rungs"]
    systems = []
    for r in rung:
        if r["rung"] not in systems:
            systems.append(r["rung"])
    lab = {r["rung"]: r["rung_label"] for r in rung}
    rep.p("Means per system [95% interval]; no order is assumed.")
    nrow = {r["rung"]: r["n"] for r in rung if r["metric"] == cfg[0]["name"]}
    rep.table(["system", "n"] + [m["label"] for m in cfg],
              [[lab[s], nrow.get(s, "")] + [meta.fmci(g(r, "mean"), g(r, "lo"), g(r, "hi")) for m in cfg
                                            for r in rung if r["metric"] == m["name"] and r["rung"] == s]
               for s in systems])
    rep.p("Pairwise accuracy, P(row > column) over text pairs with ties one half, on the oriented score "
          "(lower-is-better metrics negated).")
    for m in cfg:
        acc = {}
        for r in tables["pairs"]:
            if r["metric"] == m["name"] and r.get("acc") is not None:
                acc[r["upper"], r["lower"]] = r["acc"]
                acc[r["lower"], r["upper"]] = 1 - r["acc"]
        rep.h(m["label"], 3)
        rep.table(["row > column"] + [lab[s] for s in systems],
                  [[lab[s]] + ["" if s == t else meta.f3(acc.get((s, t), math.nan)) for t in systems]
                   for s in systems])


def post_attacks(a, rep, tables, cfg):
    """Mean per attack, parity with human text and P(human > attack)."""
    rung = tables["rungs"]
    attacks = []
    for r in rung:
        if r["rung"] not in attacks:
            attacks.append(r["rung"])
    mean = {(r["metric"], r["rung"]): r for r in rung}
    hum = attacks[0]
    rep.p(f"Mean per attack with its 95% interval and, in brackets, parity = mean / mean of `{hum}` text. "
          f"For a lower-is-better metric (marked −) a parity above 1 means the attack looks worse.")
    rows = []
    for s in attacks:
        cells = []
        for m in cfg:
            r = mean.get((m["name"], s))
            h = mean.get((m["name"], hum))
            par = r["mean"] / h["mean"] if r and h and h["mean"] not in (None, 0) and r["mean"] is not None else math.nan
            cells.append(f"{meta.fmci(g(r, 'mean'), g(r, 'lo'), g(r, 'hi'))} ({meta.f3(par, 2)})" if r else "n/a")
        rows.append([s, mean[(cfg[0]['name'], s)]["n"]] + cells)
    rep.table(["attack", "n"] + [m["label"] + (" (−)" if m["dir"] == "-" else "") for m in cfg], rows)
    rep.p(f"P({hum} > attack) over text pairs, ties one half (1.0: the metric always prefers the human text):")
    acc = {(r["metric"], r["lower"]): r for r in tables["pairs"] if r["upper"] == hum}
    rep.table(["attack"] + [m["label"] for m in cfg],
              [[s] + [meta.fmci(g(acc[(m["name"], s)], "acc"), g(acc[(m["name"], s)], "acc_lo"),
                                g(acc[(m["name"], s)], "acc_hi")) if (m["name"], s) in acc else "n/a"
                      for m in cfg] for s in attacks[1:]])
    parity = [{"metric": m["name"], "label": m["label"], "attack": s,
               "mean": mean[(m["name"], s)]["mean"], "human_mean": mean[(m["name"], hum)]["mean"],
               "parity": (mean[(m["name"], s)]["mean"] / mean[(m["name"], hum)]["mean"]
                          if mean[(m["name"], hum)]["mean"] else None)}
              for m in cfg for s in attacks if (m["name"], s) in mean and (m["name"], hum) in mean]
    rep.rows("parity", parity)


# ---------------------------------------------------------------------------
# Profiling (no meta.py)
# ---------------------------------------------------------------------------

def parse_time_v(text):
    """Fields of a /usr/bin/time -v block."""
    out = {}
    for key, pat in (("command", r'Command being timed: "(.*)"'), ("user_s", r"User time \(seconds\): ([\d.]+)"),
                     ("wall", r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)"),
                     ("max_rss_kb", r"Maximum resident set size \(kbytes\): (\d+)")):
        m = re.search(pat, text)
        if m:
            out[key] = m.group(1)
    if "wall" in out:
        parts = [float(x) for x in out["wall"].split(":")]
        out["wall_s"] = sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
    return out


def run_profile(a, P, out_dir):
    prof = P.RUN / "profile"
    rep = meta.Report("Ablations and cost")
    used = []
    # ablation ladders (the table of explore/abl.py)
    rows = []
    for f in sorted(prof.glob("abl_*.md")):
        txt = f.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"### control ladder \(no model rung\)(.*?)Verdict", txt, re.S)
        if not m:
            continue
        accs = re.findall(r"\| [^|]+ > [^|]+ \| ([0-9.]+) \| (yes|no) \| (PASS|FAIL)", m.group(1))
        hm = re.search(r"\| held-out human \| \d+ \| ([0-9.]+)", txt)
        tr = re.search(r"\| trie samples \(pooled\) \| \d+ \| ([0-9.]+)", txt)
        rows.append({"variant": f.stem[4:], "acc": [float(x[0]) for x in accs],
                     "passed": sum(x[2] == "PASS" for x in accs),
                     "human": float(hm.group(1)) if hm else None, "trie": float(tr.group(1)) if tr else None})
        used.append(f)
    if rows:
        rep.h("Ablations: fresh control ladder without the model rung", 2)
        rep.p(f"{len(rows)} runs of `scribe ladder` (run/profile/abl_*.md). Accuracy of each adjacent pair "
              f"(ties one half) and the number of pairs that pass; human and trie are the rung means.")
        heads = ["h>ss", "ss>ws", "ws>rl", "rl>let", "let>rep"]
        rep.table(["variant"] + heads + ["passed", "human", "trie"],
                  [[r["variant"]] + [meta.f3(x) for x in r["acc"]] + [""] * (5 - len(r["acc"]))
                   + [r["passed"], meta.f3(r["human"]), meta.f3(r["trie"])] for r in rows])
        rep.rows("ablations", [{"variant": r["variant"], **{h: (r["acc"][i] if i < len(r["acc"]) else None)
                                                           for i, h in enumerate(heads)},
                                "passed": r["passed"], "human": r["human"], "trie": r["trie"]} for r in rows])
    # TIMING lines of `scribe score --timing`
    trows = []
    for f in sorted(prof.glob("timing_*.err")):
        txt = f.read_text(encoding="utf-8", errors="replace")
        for line in txt.splitlines():
            if line.startswith("TIMING "):
                try:
                    d = json.loads(line[7:])
                except json.JSONDecodeError:
                    continue
                d.update({"file": f.name, **{k: v for k, v in parse_time_v(txt).items() if k != "command"}})
                trows.append(d)
        used.append(f)
    if trows:
        rep.h("SCRIBE scoring cost (scribe score --timing)", 2)
        rep.table(["run", "pack MiB", "pack model s", "pack v2 s", "pack vectors s", "texts", "score s",
                   "texts/s", "extract ms/text", "peak RAM MB", "wall s"],
                  [[r["file"], meta.f3(r.get("pack_bytes", math.nan) / 2 ** 20, 1), meta.f3(r.get("pack_model_s"), 2),
                    meta.f3(r.get("pack_v2_s"), 2), meta.f3(r.get("pack_vectors_s"), 2), r.get("score_texts", ""),
                    meta.f3(r.get("score_s"), 2), meta.f3(r.get("texts_per_s"), 1), meta.f3(r.get("extract_ms"), 3),
                    meta.f3(int(r["max_rss_kb"]) / 1024, 0) if "max_rss_kb" in r else "n/a",
                    meta.f3(r.get("wall_s"), 1)] for r in trows])
        rep.rows("scribe_timing", trows)
    else:
        rep.note("no TIMING lines yet (run/profile/timing_*.err)")
    # peak RAM and time of every /usr/bin/time -v block in the profile folder
    mrows = []
    for f in sorted(list(prof.glob("*.err")) + list(prof.glob("*.time"))):
        d = parse_time_v(f.read_text(encoding="utf-8", errors="replace"))
        if "max_rss_kb" in d:
            what = "ppmi_svd (word vectors)" if f.suffix == ".time" else ("scribe score" if f.name.startswith("timing")
                                                                            else "scribe ladder")
            mrows.append({"file": f.name, "what": what, "wall_s": d.get("wall_s"), "user_s": num(d.get("user_s")),
                          "max_rss_mb": int(d["max_rss_kb"]) / 1024})
            used.append(f)
    if mrows:
        rep.h("Peak RAM and time (/usr/bin/time -v)", 2)
        rep.table(["file", "what", "wall s", "user s", "peak RAM MB"],
                  [[r["file"], r["what"], meta.f3(r["wall_s"], 1), meta.f3(r["user_s"], 1),
                    meta.f3(r["max_rss_mb"], 0)] for r in mrows])
        rep.rows("ram", mrows)
    # baselines throughput
    brows = []
    for f in sorted(P.R.glob("*/*.time.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        brows.append({"set": f.parent.name, "family": d.get("family", f.stem.replace(".time", "")),
                      "model": d.get("model") or d.get("featurize_model") or "",
                      "n": d.get("n", d.get("n_texts")), "scored": d.get("n_scored_this_run"),
                      "seconds": d.get("seconds", d.get("set_metric_seconds")),
                      "texts_per_s": d.get("texts_per_s", d.get("features_per_s")), "threads": d.get("threads")})
        used.append(f)
    if brows:
        rep.h("Baseline throughput (run/baselines/*/*.time.json)", 2)
        rep.p("texts/s of the run that wrote the file (for set level: MAUVE features per second); "
              "`scored` is the number scored in that run (a resumed run scores fewer than n). "
              "The machine was shared, so these are lower bounds on speed.")
        rep.table(["set", "family", "model", "n", "scored", "seconds", "texts/s", "threads"],
                  [[r["set"], r["family"], r["model"], r["n"], r["scored"] if r["scored"] is not None else "",
                    meta.f3(r["seconds"], 1), meta.f3(r["texts_per_s"], 2), r["threads"] or ""] for r in brows])
        rep.rows("baselines_time", brows)
    if not used:
        return None, "no profiling files yet"
    write(rep, out_dir, a.name, a)
    return rep, None


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def write(rep, out_dir, name, a):
    args = SimpleNamespace(out=str(out_dir / f"{name}.md"), csv=str(out_dir / f"{name}.csv"), cmd=a.cmd)
    meta.write_outputs(rep, args, ["report.py", name], quiet=True)


def required_missing(a):
    req = []
    if a.texts:
        req.append(a.texts)
    req += [s.path for s in a.scribe + a.extra if s.required]
    if a.pairs:
        req.append(a.pairs)
    if a.pairs_te:
        req.append(a.pairs_te)
    req += [p for p, _, r in a.setlevel if r]
    return [str(p) for p in req if not Path(p).exists()]


def header(a, info):
    lines = [f"# {a.title}", "", f"Analysis `{a.name}` ({a.section}). meta.py `{a.cmd}`."]
    lines += [f"Inputs: {', '.join(info['inputs'])}."]
    if info.get("missing_optional"):
        lines += [f"Not available yet (left out): {', '.join(info['missing_optional'])}."]
    if info.get("dropped"):
        lines += [f"Metrics with no value on these texts (left out): {', '.join(info['dropped'])}."]
    if info.get("clashes"):
        lines += [f"Values that differ between two sources (the first source was kept): {info['clashes']}."]
    if a.note:
        lines += [a.note]
    return "\n".join(lines) + "\n\n"


def run_analysis(a, P, args, out_dir, work):
    """Run one analysis; returns a status dict."""
    st = {"name": a.name, "section": a.section, "title": a.title, "cmd": a.cmd}
    t0 = time.time()
    if a.cmd == "profile":
        if args.list:
            st.update(status="would run")
            return st
        rep, why = run_profile(a, P, out_dir)
        if rep is None:
            st.update(status="skipped", why=why)
        else:
            st.update(status="ran", outputs=[f"{a.name}.md", f"{a.name}.csv", f"{a.name}.json"],
                      seconds=time.time() - t0)
        return st
    rel = lambda p: str(Path(p)).replace(str(P.SP), "SP")
    miss = required_missing(a)
    found, missing_opt, missing_req = find_baselines(P.R, a.baselines)
    if a.setlevel and not any(Path(p).exists() for p, _, _ in a.setlevel):
        miss += [str(p) for p, _, _ in a.setlevel if str(p) not in miss]
    if miss or missing_req:
        st.update(status="skipped", why="missing: " + ", ".join([rel(m) for m in miss] + [f"R/{m}" for m in missing_req]))
        return st
    need = a.need_baselines
    if need and not [s for s in found if need is True or s.path.parent.name == need]:
        want = [m for m in missing_opt if need is True or m.startswith(f"{need}/")] or [f"{need}/*"]
        st.update(status="skipped", why="no baseline output yet: " + ", ".join(f"R/{m}" for m in want))
        return st
    if args.list:
        st.update(status="would run", missing_optional=[f"R/{m}" for m in missing_opt])
        return st

    wd = work / a.name
    wd.mkdir(parents=True, exist_ok=True)
    info = {"inputs": [], "missing_optional": [f"R/{m}" for m in missing_opt]}
    argv = [a.cmd]
    if a.setlevel:
        vals, used = load_setlevel(a.setlevel)
        info["inputs"] += [rel(p) for p in used]
        info["missing_optional"] += [rel(p) for p, _, _ in a.setlevel if not Path(p).exists()]
        names = [k for k in METRICS if any(r.get(k) is not None for r in vals)]
        cfg = [{"name": k, "dir": METRICS[k][0], "label": METRICS[k][1]} for k in names]
        if a.cmd == "setladder":
            write_jsonl(wd / "values.jsonl", vals)
            argv += ["--values", str(wd / "values.jsonl")]
        else:   # one "text" per set, so meta.py models gives means and the order statistics
            write_jsonl(wd / "texts.jsonl", [{"id": f"{r['group']}#{r['subset']}", "system": r["group"],
                                              "text": ""} for r in vals])
            write_jsonl(wd / "scores.jsonl", [{"id": f"{r['group']}#{r['subset']}", **{k: r.get(k) for k in names}}
                                              for r in vals])
            argv += ["--texts", str(wd / "texts.jsonl"), "--scores", str(wd / "scores.jsonl")]
        n_units = len(vals)
    else:
        texts = read_jsonl(a.texts)
        info["inputs"].append(rel(a.texts))
        if a.transform:
            texts = [a.transform(r) for r in texts]
        if a.pairs:
            prs = read_jsonl(a.pairs)[: a.pairs_limit] if a.pairs_limit else read_jsonl(a.pairs)
            info["inputs"].append(rel(a.pairs) + (f" (first {a.pairs_limit})" if a.pairs_limit else ""))
            if a.pairs_te:
                try:
                    prs = attach_te(prs, a.pairs_te)
                except ValueError as e:
                    st.update(status="skipped", why=f"answer times: {e}")
                    return st
                info["inputs"].append(rel(a.pairs_te) + " (answer times)")
            need = {str(p["a"]) for p in prs} | {str(p["b"]) for p in prs}
            texts = [r for r in texts if str(r["id"]) in need]
            write_jsonl(wd / "pairs.jsonl", prs)
            argv += ["--pairs", str(wd / "pairs.jsonl")]
        if a.transform or a.pairs:
            write_jsonl(wd / "texts.jsonl", texts)
            tpath = wd / "texts.jsonl"
        else:
            tpath = a.texts
        sources = a.scribe + a.extra + found
        info["inputs"] += [rel(s.path) for s in sources if s.path.exists()]
        info["missing_optional"] += [rel(s.path) for s in a.scribe + a.extra if not s.path.exists()]
        scores, clashes = merge_scores([str(r["id"]) for r in texts], sources)
        info["clashes"] = clashes
        cfg = metric_config(scores)
        names_all = {n for s in sources for n in ((s.fields or {}).values())}
        info["dropped"] = sorted(n for n in names_all if n not in {c["name"] for c in cfg})
        write_jsonl(wd / "scores.jsonl", [{"id": i, **row} for i, row in scores.items()])
        argv += ["--texts", str(tpath), "--scores", str(wd / "scores.jsonl")]
        n_units = len(texts)
        if a.systems:
            order = auto_systems(texts) if a.systems == "auto" else a.systems
            argv += ["--systems", systems_arg(order, a.labels or LABELS)]
    if not cfg:
        st.update(status="skipped", why="no metric has a value on these texts")
        return st
    (wd / "metrics.json").write_text(json.dumps({"metrics": cfg}, ensure_ascii=False, indent=1), encoding="utf-8")
    md = out_dir / (f"{a.name}.full.md" if a.post else f"{a.name}.md")
    argv += ["--metrics-config", str(wd / "metrics.json"), *a.args, "--boot", str(args.boot),
             "--seed", str(args.seed), "--out", str(md), "--csv", str(out_dir / f"{a.name}.csv")]
    try:
        meta.main(argv, quiet=True)
    except SystemExit as e:
        if e.code not in (0, None):
            st.update(status="failed", why=f"meta.py: {e.code}")
            return st
    hdr = header(a, info)
    body = md.read_text(encoding="utf-8")
    body = body.split("\n", 1)[1] if body.startswith("# ") else body
    md.write_text(hdr + body.lstrip("\n"), encoding="utf-8")
    outputs = [md.name, f"{a.name}.csv", f"{a.name}.json"]
    if a.post:
        rep = meta.Report(a.title)
        rep.lines = hdr.rstrip("\n").split("\n") + [""]
        a.post(a, rep, load_tables(out_dir / f"{a.name}.json"), cfg)
        rep.p(f"Full meta.py output: [{a.name}.full.md]({a.name}.full.md).")
        (out_dir / f"{a.name}.md").write_text(rep.markdown(), encoding="utf-8")
        if rep.tables:
            pd.concat([pd.DataFrame(r).assign(table=k) for k, r in rep.tables.items()]).to_csv(
                out_dir / f"{a.name}.extra.csv", index=False)
            outputs.append(f"{a.name}.extra.csv")
        outputs.insert(0, f"{a.name}.md")
    st.update(status="ran", outputs=outputs, n=n_units, n_metrics=len(cfg),
              missing_optional=info["missing_optional"], seconds=time.time() - t0)
    st["checks"] = run_checks(a, out_dir)
    return st


def run_checks(a, out_dir):
    exp = KNOWN.get(a.name)
    if not exp:
        return []
    df = pd.read_csv(out_dir / f"{a.name}.csv")
    adj = df[(df["table"] == "pairs") & (df["adjacent"].astype(str) == "True")]
    res = []
    for mname, values in exp.items():
        got = adj[adj["metric"] == mname]["acc"].tolist()
        for i, v in enumerate(values):
            gv = got[i] if i < len(got) else math.nan
            pair = f"{LADDER[i]} > {LADDER[i + 1]}"
            res.append({"analysis": a.name, "metric": mname, "pair": pair, "expected": v, "got": gv,
                        "ok": bool(np.isfinite(gv) and round(gv, 3) == round(v, 3))})
    return res


def write_index(out_dir, statuses, P, args, started):
    L = ["# SCRIBE paper results", "",
         f"Written by `scripts/scribe_eval/report.py` on {time.strftime('%Y-%m-%d %H:%M')} "
         f"({time.time() - started:.0f} s). Inputs under SP = `{P.SP}`. Statistics: "
         f"`scripts/scribe_eval/meta.py`, {args.boot} bootstrap resamples, seed {args.seed}. "
         f"Each analysis has a Markdown file, a CSV of every table (column `table` names the table) and "
         f"a JSON file with the same tables. Rerun the script as inputs land; analyses listed under "
         f"Skipped run once their inputs exist.", ""]
    ran = [s for s in statuses if s["status"] in ("ran", "would run")]
    L += ["## Outputs", "", "| analysis | section | what | units | metrics | not available yet |", "|---|---|---|---|---|---|"]
    for s in ran:
        link = f"[{s['name']}]({s['name']}.md)" if s["status"] == "ran" else s["name"]
        L.append(f"| {link} | {s['section'].split('.')[0]} | {meta.esc(s['title'])} | {s.get('n', '')} | "
                 f"{s.get('n_metrics', '')} | {meta.esc(', '.join(s.get('missing_optional') or [])) or '—'} |")
    sk = [s for s in statuses if s["status"] in ("skipped", "failed")]
    L += ["", "## Skipped", ""]
    if sk:
        L += ["| analysis | section | what | why |", "|---|---|---|---|"]
        L += [f"| {s['name']} | {s['section'].split('.')[0]} | {meta.esc(s['title'])} | "
              f"{meta.esc(s['status'] + ': ' + str(s.get('why', '')))} |" for s in sk]
    else:
        L.append("Nothing.")
    checks = [c for s in statuses for c in s.get("checks", [])]
    if checks:
        L += ["", "## Checks against known numbers", "",
              "| analysis | metric | pair | expected | got | match |", "|---|---|---|---|---|---|"]
        L += [f"| {c['analysis']} | {c['metric']} | {c['pair']} | {meta.f3(c['expected'])} | {meta.f3(c['got'])} | "
              f"{'yes' if c['ok'] else '**no**'} |" for c in checks]
    (out_dir / "index.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run every analysis of the SCRIBE paper through meta.py.")
    ap.add_argument("--sp", default=DEFAULT_SP, help="scratchpad root holding data/ and run/ (default $SCRIBE_SP)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help=f"output folder (default {DEFAULT_OUT})")
    ap.add_argument("--work", help="folder for merged inputs (default <sp>/run/report_work)")
    ap.add_argument("--only", nargs="+", help="run only these analyses (names as in index.md)")
    ap.add_argument("--list", action="store_true", help="list what would run and what would be skipped")
    ap.add_argument("--boot", type=int, default=meta.DEFAULT_BOOT)
    ap.add_argument("--seed", type=int, default=meta.DEFAULT_SEED)
    args = ap.parse_args(argv)
    sp = Path(args.sp)
    P = SimpleNamespace(SP=sp, D=sp / "data", RUN=sp / "run", R=sp / "run" / "baselines")
    out_dir, work = Path(args.out), Path(args.work) if args.work else sp / "run" / "report_work"
    out_dir.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    analyses = build_analyses(P)
    if args.only:
        unknown = set(args.only) - {a.name for a in analyses}
        if unknown:
            raise SystemExit(f"unknown analyses: {sorted(unknown)}; known: {[a.name for a in analyses]}")
        analyses = [a for a in analyses if a.name in args.only]
    started = time.time()
    statuses = []
    for a in analyses:
        st = run_analysis(a, P, args, out_dir, work)
        statuses.append(st)
        extra = st.get("why") or ", ".join(st.get("outputs", []))
        print(f"{st['status']:9s} {a.name}: {extra}" + (f" ({st['seconds']:.0f} s)" if "seconds" in st else ""),
              flush=True)
        for c in st.get("checks", []):
            if not c["ok"]:
                print(f"  CHECK FAILED {c['metric']} {c['pair']}: expected {c['expected']}, got {c['got']}")
    if not args.list:
        if args.only and (out_dir / "index.md").exists():
            statuses = merge_index_statuses(out_dir, statuses)
        write_index(out_dir, statuses, P, args, started)
        (out_dir / "index.json").write_text(json.dumps(meta.jsonable(statuses), indent=1, ensure_ascii=False),
                                            encoding="utf-8")
    return 0


def merge_index_statuses(out_dir, new):
    """With --only, keep the other analyses' entries of the previous run."""
    try:
        old = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return new
    names = {s["name"] for s in new}
    merged = [s for s in old if s["name"] not in names]
    order = {s["name"]: i for i, s in enumerate(old)}
    merged += new
    merged.sort(key=lambda s: order.get(s["name"], len(order)))
    return merged


if __name__ == "__main__":
    sys.exit(main())
