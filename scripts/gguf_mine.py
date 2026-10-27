#!/usr/bin/env python3
"""Statistically mine a GGUF language model for Cypha's data pack.

Offline only. The teacher model never runs inside Cypha; what comes out of
it is data (word lists, vectors, clusters, facts, relations, patterns,
text statistics), and every mined item carries the statistics needed to
weight it as evidence (docs/PLAN.md section 7, ECS). The full list of what
can be mined, why, and how each output feeds the plan:
docs/GGUF_MINING.md.

Static subcommands (read the weights; need numpy + gguf):
  info        metadata, tensor table, licence check
  vocab       tokenizer: words, frequency priors, merges, morphology, scripts
  embed       token embeddings: neighbours, clusters, hierarchy (Brown-style
              bit paths), principal axes, relation offsets, direct-path bigrams
  neurons     FFN neurons projected onto the vocabulary (what each reads/writes)
  heads       attention OV circuits: copying scores and token mappings
  spectra     singular-value spectra, power-law alpha, effective rank, outliers

Dynamic subcommands (run the model; also need llama-cpp-python):
  facts       relation probing (Wikidata relations, paraphrased templates)
  crawl       recursive knowledge-graph crawl from seed entities
  lexical     synonyms, antonyms, hypernyms, parts, definitions, POS,
              inflections, polarity, entity type
  taxonomy    is-a chains
  commonsense ConceptNet-style relations
  paraphrase  question-pattern paraphrases, verified by answer agreement
  qa          question/answer pairs from mined facts
  sample      synthetic corpus + n-gram, collocation and Zipf statistics
  score       bits per byte of a corpus under the model (data selection)
  vectors     model-pooled word vectors (--pooling mean|last|none) and senses
  calibrate   accuracy and calibration against gold facts -> ECS priors

Embedding geometry (docs/EMBEDDING_GEOMETRY.md; teacher e.g. Qwen3-Embedding):
  embedtexts  batched text embeddings: last-token pooling with the EOS check,
              query-side instruction, MRL prefix, resume, timing
  sensitivity B10 minimal pairs (negation, role swap, number, entity, shuffle,
              typo, synonym, padding) or an external pairs TSV
  occlusion   B15 word importance by dropping words; B16 additivity fits
  instruct    B8 instruction shifts (offset vs affine), B9 query/document asymmetry
  pairs       C1-C3 synthetic queries, hard negatives, yes/no verification,
              optional embedder cosine filter, false-negative audit
  geometry    (no model) B1, B3-B7, B13, B17 on any .npy + index: floor,
              anisotropy, PCA, TwoNN, hubness/CSLS, MRL curves, temperature
  compare     (no model) two vector sets: pair Spearman, kNN overlap, CKA,
              Procrustes, force-match KL
  wdiff       (static) A4 weight difference between two GGUFs

Utility:
  canon       attach IDs to mined strings from a label table (e.g. Wikidata)
  selftest    build a tiny random GGUF from a vocab-only GGUF and run all of
              the above, plus known-answer checks of the geometry maths
              (plumbing test; a random model knows nothing)

Every output directory gets provenance.json (model hash, metadata, licence,
versions, command, seed). Licences: prefer models whose terms allow using
outputs to build other systems (Apache-2.0, MIT, ...); `info` warns otherwise.
Do not use this to extract memorised training text; see GGUF_MINING.md.

Author: Odin Loch. Licence: as the repository.
"""

from __future__ import annotations

import argparse
import collections
import datetime as _dt
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata

import numpy as np

VERSION = "1.0"

PERMISSIVE = {
    "apache-2.0", "apache 2.0", "apache2", "mit", "bsd-2-clause", "bsd-3-clause",
    "cc-by-4.0", "cc0-1.0", "cc0", "unlicense", "isc", "mpl-2.0",
}

TOKEN_TYPES = {1: "normal", 2: "unknown", 3: "control", 4: "user_defined", 5: "unused", 6: "byte"}


# --------------------------------------------------------------------------- utilities

def log(msg: str) -> None:
    print(f"[gguf_mine {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(2)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def sha256_file(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def esc(s: str) -> str:
    """One-line, tab-safe rendering of a token or string."""
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")


def write_tsv(path: str, header: list[str], rows) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(esc(str(x)) if isinstance(x, str) else fmt(x) for x in r) + "\n")
            n += 1
    return n


def fmt(x) -> str:
    if isinstance(x, (float, np.floating)):
        if not math.isfinite(float(x)):
            return str(float(x))
        return f"{float(x):.6g}"
    return str(x)


def jsonl_append(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def done_keys(path: str, key_fields: tuple[str, ...]) -> set:
    """Keys already present in a JSONL output, for --resume."""
    keys = set()
    if not os.path.exists(path):
        return keys
    with open(path, encoding="utf-8") as f:
        for ln in f:
            try:
                o = json.loads(ln)
                keys.add(tuple(o.get(k) for k in key_fields))
            except json.JSONDecodeError:
                continue
    return keys


def logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + math.log(float(np.sum(np.exp(x - m))))


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    return x - logsumexp(x)


def entropy(p) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if p.size else 0.0


ARTICLES = re.compile(r"^(the|a|an)\s+", re.I)


def norm_key(s: str) -> str:
    """Comparison key for an answer string: case, punctuation and article folded."""
    s = unicodedata.normalize("NFKC", s).strip().lower()
    s = s.strip(" \t\"'`“”‘’.,;:!?()[]{}")
    s = ARTICLES.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s


DONT_KNOW = re.compile(r"^(don'?t know|do not know|unknown|n/?a|none|not sure|i don'?t know|no idea|\?+)$", re.I)


def clean_answer(s: str) -> str:
    s = s.strip()
    s = re.split(r"\n", s, maxsplit=1)[0]
    s = re.split(r"(?<=[a-z0-9\)])\.(\s|$)", s, maxsplit=1)[0]
    s = s.strip(" \t\"'`“”‘’,;:")
    return s.strip()


# --------------------------------------------------------------------------- GGUF reading

def open_gguf(path: str):
    try:
        from gguf import GGUFReader
    except ImportError:
        die("the static commands need the gguf package: pip install gguf numpy")
    if not os.path.exists(path):
        die(f"no such model file: {path}")
    return GGUFReader(path)


def field_value(f):
    try:
        return f.contents()
    except Exception:
        # Older gguf: decode by hand.
        from gguf import GGUFValueType
        t = f.types[0] if f.types else None
        if t == GGUFValueType.STRING:
            return bytes(f.parts[f.data[0]]).decode("utf-8", errors="replace")
        if t == GGUFValueType.ARRAY:
            sub = f.types[-1]
            if sub == GGUFValueType.STRING:
                return [bytes(f.parts[i]).decode("utf-8", errors="replace") for i in f.data]
            return [f.parts[i].tolist()[0] for i in f.data]
        return f.parts[f.data[0]].tolist()[0]


def meta(reader, key: str, default=None):
    f = reader.fields.get(key)
    if f is None:
        return default
    try:
        return field_value(f)
    except Exception:
        return default


def all_meta(reader, max_list: int = 8) -> dict:
    out = {}
    for k, f in reader.fields.items():
        if k.startswith("GGUF."):
            continue
        try:
            v = field_value(f)
        except Exception as e:  # pragma: no cover
            v = f"<unreadable: {e}>"
        if isinstance(v, list) and len(v) > max_list:
            v = {"length": len(v), "first": v[:max_list]}
        out[k] = v
    return out


def tensor_map(reader) -> dict:
    return {t.name: t for t in reader.tensors}


def dequant(t) -> np.ndarray:
    """Float32 array in row-major (outer, ..., inner) order, i.e. GGUF ne reversed."""
    from gguf.quants import dequantize
    a = np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32)
    shape = tuple(int(x) for x in reversed([int(v) for v in t.shape]))
    return a.reshape(shape)


def arch_key(reader, suffix: str, default=None):
    arch = meta(reader, "general.architecture", "llama")
    return meta(reader, f"{arch}.{suffix}", default)


def licence_status(reader) -> tuple[str, bool]:
    lic = meta(reader, "general.license", None)
    if lic is None:
        return "unknown (no general.license key)", False
    s = str(lic).strip().lower()
    return str(lic), s in PERMISSIVE


# --------------------------------------------------------------------------- tokenizer

def _bytes_to_unicode() -> dict:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


_U2B = {v: k for k, v in _bytes_to_unicode().items()}
_BYTE_TOK = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def token_text(tok: str, tmodel: str) -> str:
    """Surface text of a vocabulary entry (leading space kept)."""
    if tmodel in ("gpt2", "bpe"):
        try:
            return bytes(_U2B[c] for c in tok).decode("utf-8", errors="replace")
        except KeyError:
            return tok
    m = _BYTE_TOK.match(tok)
    if m:
        b = int(m.group(1), 16)
        return chr(b) if b < 128 else f"<0x{m.group(1)}>"
    return tok.replace("▁", " ")


def script_of(text: str) -> str:
    for ch in text:
        if ch.isalpha():
            try:
                name = unicodedata.name(ch)
            except ValueError:
                return "other"
            first = name.split(" ")[0]
            if first == "CJK":
                return "CJK"
            return first
    if any(ch.isdigit() for ch in text):
        return "DIGIT"
    if text.strip() == "":
        return "SPACE"
    return "PUNCT"


class Vocab:
    def __init__(self, reader):
        self.model = meta(reader, "tokenizer.ggml.model", "llama")
        toks = meta(reader, "tokenizer.ggml.tokens", None)
        if toks is None:
            die("this GGUF has no tokenizer.ggml.tokens")
        self.tokens: list[str] = list(toks)
        self.n = len(self.tokens)
        self.types = list(meta(reader, "tokenizer.ggml.token_type", [1] * self.n))
        scores = meta(reader, "tokenizer.ggml.scores", None)
        self.scores = np.asarray(scores, dtype=np.float64) if scores is not None else None
        merges = meta(reader, "tokenizer.ggml.merges", None)
        self.merges: list[str] = list(merges) if merges is not None else []
        self.text = [token_text(t, self.model) for t in self.tokens]
        self.merge_rank = {}
        for i, m in enumerate(self.merges):
            parts = m.split(" ")
            if len(parts) == 2:
                self.merge_rank.setdefault(parts[0] + parts[1], i)
        self.id_of = {}
        for i, t in enumerate(self.text):
            self.id_of.setdefault(t, i)

    def freq_rank_order(self) -> np.ndarray:
        """Token ids ordered most-frequent first, by the best prior the tokenizer has."""
        if self.scores is not None and np.ptp(self.scores) > 0:
            return np.argsort(-self.scores, kind="stable")
        if self.merge_rank:
            big = 10 ** 9
            ranks = np.array([self.merge_rank.get(t, big if len(t) > 1 else -1) for t in self.tokens])
            return np.argsort(ranks, kind="stable")
        return np.arange(self.n)

    def is_word(self, i: int, min_len: int = 3) -> bool:
        if TOKEN_TYPES.get(self.types[i], "normal") != "normal":
            return False
        t = self.text[i]
        return t.startswith(" ") and len(t) - 1 >= min_len and t[1:].isalpha()

    def word_ids(self, min_len: int = 3, limit: int | None = None, dictionary: set | None = None) -> list[int]:
        """Word-initial alphabetic tokens, most frequent first. Without a dictionary these are
        candidates: SPM/BPE also keep word-initial fragments ("con", "pro")."""
        order = self.freq_rank_order()
        ids = [int(i) for i in order if self.is_word(int(i), min_len)
               and (dictionary is None or self.text[int(i)][1:].lower() in dictionary)]
        return ids[:limit] if limit else ids

    def single_token_id(self, word: str) -> int | None:
        for form in (" " + word, word):
            if form in self.id_of:
                return self.id_of[form]
        return None


# --------------------------------------------------------------------------- provenance

def load_dictionary(path: str | None) -> set | None:
    if not path:
        return None
    return {w.strip().lower() for w in read_lines(path)}


def write_provenance(out: str, args, reader=None, extra: dict | None = None) -> None:
    p = {
        "tool": "scripts/gguf_mine.py",
        "tool_version": VERSION,
        "command": sys.argv,
        "time_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "seed": getattr(args, "seed", None),
        "model_path": os.path.abspath(args.model) if getattr(args, "model", None) else None,
        "numpy": np.__version__,
    }
    try:
        from importlib.metadata import version
        p["gguf"] = version("gguf")
    except Exception:
        pass
    try:
        from importlib.metadata import version
        p["llama_cpp_python"] = version("llama_cpp_python")
    except Exception:
        pass
    if getattr(args, "model", None) and os.path.exists(args.model):
        p["model_bytes"] = os.path.getsize(args.model)
        if not getattr(args, "no_hash", False):
            p["model_sha256"] = sha256_file(args.model)
    if reader is not None:
        lic, ok = licence_status(reader)
        p["model_name"] = meta(reader, "general.name")
        p["architecture"] = meta(reader, "general.architecture")
        p["file_type"] = meta(reader, "general.file_type")
        p["licence"] = lic
        p["licence_permissive"] = ok
    if extra:
        p.update(extra)
    with open(os.path.join(out, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(p, f, indent=2, default=_json_default)


def licence_warning(reader) -> None:
    lic, ok = licence_status(reader)
    if not ok:
        log(f"WARNING: model licence is '{lic}'. Check the terms allow using its outputs to build "
            f"another system before shipping anything mined from it (docs/GGUF_MINING.md, licences).")


# --------------------------------------------------------------------------- static: info

def cmd_info(args) -> None:
    r = open_gguf(args.model)
    out = ensure_dir(args.out)
    m = all_meta(r)
    rows = []
    total = 0
    for t in r.tensors:
        shape = [int(x) for x in t.shape]
        nbytes = int(t.n_bytes) if hasattr(t, "n_bytes") else int(np.asarray(t.data).nbytes)
        total += nbytes
        rows.append((t.name, t.tensor_type.name, "x".join(map(str, shape)), int(t.n_elements), nbytes))
    write_tsv(os.path.join(out, "tensors.tsv"), ["name", "type", "shape_ne", "elements", "bytes"], rows)
    lic, ok = licence_status(r)
    info = {
        "metadata": m,
        "n_tensors": len(r.tensors),
        "tensor_bytes": total,
        "parameters": int(sum(int(t.n_elements) for t in r.tensors)),
        "licence": lic,
        "licence_permissive": ok,
        "has_chat_template": "tokenizer.chat_template" in r.fields,
        "tied_output": "output.weight" not in tensor_map(r),
    }
    with open(os.path.join(out, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False, default=_json_default)
    licence_warning(r)
    write_provenance(out, args, r)
    log(f"info: {len(r.tensors)} tensors, {info['parameters']:,} parameters, licence {lic} -> {out}")


# --------------------------------------------------------------------------- static: vocab

COMMON_SUFFIXES = ["s", "es", "ed", "ing", "ly", "er", "est", "ness", "ment", "tion", "sion", "ity", "able",
                   "ible", "al", "ial", "ful", "less", "ous", "ive", "ize", "ise", "ist", "ism", "ship", "hood"]


def cmd_vocab(args) -> None:
    r = open_gguf(args.model)
    out = ensure_dir(args.out)
    v = Vocab(r)
    order = v.freq_rank_order()
    rank = np.empty(v.n, dtype=np.int64)
    rank[order] = np.arange(v.n)
    rows = []
    by_type = collections.Counter()
    by_script = collections.Counter()
    lengths = []
    word_initial = 0
    for i in range(v.n):
        t = v.text[i]
        ty = TOKEN_TYPES.get(v.types[i], str(v.types[i]))
        sc = script_of(t)
        by_type[ty] += 1
        by_script[sc] += 1
        lengths.append(len(t))
        wi = t.startswith(" ")
        word_initial += wi
        rows.append((i, ty, t, int(wi), sc, len(t),
                     float(v.scores[i]) if v.scores is not None else "",
                     v.merge_rank.get(v.tokens[i], ""), int(rank[i])))
    write_tsv(os.path.join(out, "vocab.tsv"),
              ["id", "type", "text", "word_initial", "script", "chars", "spm_score", "merge_rank", "freq_rank"], rows)

    dic = load_dictionary(args.dictionary)
    word_tok = v.word_ids(min_len=args.min_len)
    # Prefix fan-out: how many longer word-initial tokens extend this one. Fragments such as
    # "con" or "pro" have large fan-out and are usually not words on their own.
    lw_sorted = sorted({v.text[i][1:].lower() for i in word_tok})
    import bisect

    def fanout(w: str) -> int:
        lo = bisect.bisect_left(lw_sorted, w)
        hi = bisect.bisect_left(lw_sorted, w + "\uffff")
        return hi - lo - 1
    words = [(v.text[i][1:], i, int(rank[i]), float(v.scores[i]) if v.scores is not None else "",
              fanout(v.text[i][1:].lower()), "" if dic is None else int(v.text[i][1:].lower() in dic))
             for i in word_tok]
    write_tsv(os.path.join(out, "words.tsv"),
              ["word", "id", "freq_rank", "spm_score", "prefix_fanout", "in_dictionary"], words)

    # Morphology: suffix statistics over whole-word tokens, and continuation pieces.
    suf = collections.Counter()
    for w, *_ in words:
        lw = w.lower()
        for s in COMMON_SUFFIXES:
            if lw.endswith(s) and len(lw) > len(s) + 2:
                suf[s] += 1
    cont = [(v.text[i], i, int(rank[i])) for i in order
            if TOKEN_TYPES.get(v.types[i], "normal") == "normal" and not v.text[i].startswith(" ")
            and v.text[i].isalpha() and 1 <= len(v.text[i]) <= 6][: args.max_pieces]
    write_tsv(os.path.join(out, "suffix_counts.tsv"), ["suffix", "whole_words_ending_with_it"],
              sorted(suf.items(), key=lambda kv: -kv[1]))
    write_tsv(os.path.join(out, "continuation_pieces.tsv"), ["piece", "id", "freq_rank"], cont)

    if v.merges:
        write_tsv(os.path.join(out, "merges.tsv"), ["rank", "left", "right", "result_text"],
                  ((k, *(m.split(" ") + ["", ""])[:2], token_text("".join(m.split(" ")), v.model))
                   for k, m in enumerate(v.merges)))

    stats = {
        "tokenizer_model": v.model,
        "vocab_size": v.n,
        "by_type": dict(by_type),
        "by_script": dict(by_script.most_common()),
        "mean_chars": float(np.mean(lengths)),
        "word_initial_fraction": word_initial / max(1, v.n),
        "whole_word_tokens": len(words),
        "has_scores": v.scores is not None,
        "n_merges": len(v.merges),
        "frequency_prior": "spm scores" if v.scores is not None and np.ptp(v.scores) > 0
        else ("merge order" if v.merge_rank else "token id"),
    }
    with open(os.path.join(out, "vocab_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    write_provenance(out, args, r)
    log(f"vocab: {v.n} tokens, {len(words)} whole words, prior = {stats['frequency_prior']} -> {out}")


# --------------------------------------------------------------------------- static: embeddings

BUILTIN_PAIRS = {
    "capital_of": [("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"),
                   ("Japan", "Tokyo"), ("China", "Beijing"), ("Russia", "Moscow"), ("Egypt", "Cairo"),
                   ("Greece", "Athens"), ("Canada", "Ottawa"), ("Poland", "Warsaw"), ("Austria", "Vienna")],
    "female_of": [("king", "queen"), ("man", "woman"), ("boy", "girl"), ("father", "mother"),
                  ("brother", "sister"), ("son", "daughter"), ("uncle", "aunt"), ("husband", "wife"),
                  ("prince", "princess"), ("actor", "actress")],
    "plural": [("cat", "cats"), ("dog", "dogs"), ("car", "cars"), ("house", "houses"), ("tree", "trees"),
               ("book", "books"), ("bird", "birds"), ("road", "roads"), ("king", "kings"), ("hand", "hands")],
    "past_tense": [("walk", "walked"), ("play", "played"), ("jump", "jumped"), ("look", "looked"),
                   ("work", "worked"), ("call", "called"), ("start", "started"), ("open", "opened"),
                   ("move", "moved"), ("help", "helped")],
    "comparative": [("big", "bigger"), ("small", "smaller"), ("fast", "faster"), ("old", "older"),
                    ("long", "longer"), ("high", "higher"), ("young", "younger"), ("strong", "stronger"),
                    ("cold", "colder"), ("short", "shorter")],
}


def l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def topk_rows(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[1])
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    part = np.take_along_axis(scores, idx, axis=1)
    o = np.argsort(-part, axis=1)
    return np.take_along_axis(idx, o, axis=1)


def kmeans_cos(x: np.ndarray, k: int, iters: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Spherical k-means with k-means++ seeding. x rows must be unit length."""
    n = x.shape[0]
    k = min(k, n)
    cent = np.empty((k, x.shape[1]), dtype=np.float32)
    cent[0] = x[rng.integers(n)]
    d2 = 1.0 - x @ cent[0]
    for j in range(1, k):
        p = np.clip(d2, 1e-12, None)
        p = p / p.sum()
        cent[j] = x[rng.choice(n, p=p)]
        d2 = np.minimum(d2, 1.0 - x @ cent[j])
    lab = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        sim = x @ cent.T
        new = np.argmax(sim, axis=1)
        if np.array_equal(new, lab) and _ > 0:
            break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any():
                c = x[m].sum(axis=0)
                cent[j] = c / (np.linalg.norm(c) + 1e-12)
            else:
                cent[j] = x[rng.integers(n)]
    return lab, cent


def agglomerate(cent: np.ndarray, sizes: np.ndarray) -> list[tuple[int, int, int, float]]:
    """Average-linkage merges over centroids (cosine distance). Returns (new, a, b, distance)."""
    k = cent.shape[0]
    d = 1.0 - cent @ cent.T
    d = d.astype(np.float64)
    np.fill_diagonal(d, np.inf)
    active = list(range(k))
    ids = list(range(k))
    size = sizes.astype(np.float64).tolist()
    merges = []
    nxt = k
    alive = np.ones(k, dtype=bool)
    while alive.sum() > 1:
        sub = np.where(alive)[0]
        dd = d[np.ix_(sub, sub)]
        flat = int(np.argmin(dd))
        i, j = sub[flat // len(sub)], sub[flat % len(sub)]
        dist = float(d[i, j])
        si, sj = size[i], size[j]
        newrow = (d[i] * si + d[j] * sj) / (si + sj)
        d[i] = newrow
        d[:, i] = newrow
        d[i, i] = np.inf
        alive[j] = False
        d[j, :] = np.inf
        d[:, j] = np.inf
        merges.append((nxt, ids[i], ids[j], dist))
        ids[i] = nxt
        size[i] = si + sj
        nxt += 1
    del active
    return merges


def bit_paths(merges, k: int) -> dict[int, str]:
    """Brown-cluster-style bit strings for each leaf cluster from a merge list."""
    children = {new: (a, b) for new, a, b, _ in merges}
    if not merges:
        return {0: ""}
    root = merges[-1][0]
    paths = {}
    stack = [(root, "")]
    while stack:
        node, path = stack.pop()
        if node < k:
            paths[node] = path
        else:
            a, b = children[node]
            stack.append((a, path + "0"))
            stack.append((b, path + "1"))
    return paths


def randomized_pca(x: np.ndarray, k: int, rng: np.random.Generator, iters: int = 3):
    xc = x - x.mean(axis=0, keepdims=True)
    q = rng.standard_normal((x.shape[1], k + 10)).astype(np.float32)
    y = xc @ q
    for _ in range(iters):
        y, _r = np.linalg.qr(y)
        y = xc @ (xc.T @ y)
    qy, _r = np.linalg.qr(y)
    b = qy.T @ xc
    _u, s, vt = np.linalg.svd(b, full_matrices=False)
    total = float((xc ** 2).sum())
    return vt[:k], (s[:k] ** 2) / max(total, 1e-12)


def rankdata(a) -> np.ndarray:
    """Ranks from 1, ties given their average rank."""
    a = np.asarray(a, dtype=np.float64)
    o = np.argsort(a, kind="mergesort")
    sa = a[o]
    first = np.r_[True, sa[1:] != sa[:-1]] if a.size else np.zeros(0, dtype=bool)
    starts = np.flatnonzero(first)
    ends = np.r_[starts[1:], a.size]
    r = np.empty(a.size)
    r[o] = ((starts + ends - 1) / 2.0 + 1.0)[np.cumsum(first) - 1]
    return r


def spearman(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def load_unembed(r, tm):
    U = dequant(tm["output.weight"]) if "output.weight" in tm else dequant(tm["token_embd.weight"])
    g = dequant(tm["output_norm.weight"]) if "output_norm.weight" in tm else None
    return U, g


def cmd_embed(args) -> None:
    r = open_gguf(args.model)
    out = ensure_dir(args.out)
    tm = tensor_map(r)
    if "token_embd.weight" not in tm:
        die("no token_embd.weight in this GGUF")
    rng = np.random.default_rng(args.seed)
    v = Vocab(r)
    E = dequant(tm["token_embd.weight"])
    if E.shape[0] != v.n and E.shape[1] == v.n:
        E = E.T
    log(f"embed: token_embd {E.shape}")
    ids = v.word_ids(min_len=args.min_len, limit=args.max_tokens, dictionary=load_dictionary(args.dictionary))
    if len(ids) < 10:
        die("too few whole-word tokens to analyse")
    X = E[ids].astype(np.float32)
    Xn = l2n(X)
    words = [v.text[i][1:] for i in ids]

    # Norms against the frequency prior.
    norms = np.linalg.norm(X, axis=1)
    order = v.freq_rank_order()
    rank = np.empty(v.n, dtype=np.int64)
    rank[order] = np.arange(v.n)
    rho = spearman(norms, rank[ids])
    write_tsv(os.path.join(out, "norms.tsv"), ["word", "id", "freq_rank", "norm"],
              ((w, i, int(rank[i]), float(nm)) for w, i, nm in zip(words, ids, norms)))

    # Nearest neighbours.
    k = args.neighbours
    with open(os.path.join(out, "neighbours.tsv"), "w", encoding="utf-8") as f:
        f.write("word\tid\tneighbours\n")
        for s in range(0, len(ids), 1024):
            S = Xn[s:s + 1024] @ Xn.T
            for row in range(S.shape[0]):
                S[row, s + row] = -np.inf
            top = topk_rows(S, k)
            for row in range(S.shape[0]):
                nb = " ".join(f"{esc(words[j])}:{S[row, j]:.3f}" for j in top[row])
                f.write(f"{esc(words[s + row])}\t{ids[s + row]}\t{nb}\n")

    # Clusters and a Brown-style hierarchy over cluster centroids.
    lab, cent = kmeans_cos(Xn, args.clusters, args.kmeans_iters, rng)
    sizes = np.bincount(lab, minlength=cent.shape[0])
    merges = agglomerate(cent, np.maximum(sizes, 1))
    paths = bit_paths(merges, cent.shape[0])
    sim_to_c = (Xn * cent[lab]).sum(axis=1)
    write_tsv(os.path.join(out, "clusters.tsv"), ["word", "id", "cluster", "bit_path", "cos_to_centroid"],
              ((w, i, int(c), paths.get(int(c), ""), float(sc)) for w, i, c, sc in zip(words, ids, lab, sim_to_c)))
    rep = []
    for c in range(cent.shape[0]):
        m = np.where(lab == c)[0]
        if m.size == 0:
            continue
        best = m[np.argsort(-sim_to_c[m])[:12]]
        rep.append((c, paths.get(c, ""), int(m.size), " ".join(words[j] for j in best)))
    write_tsv(os.path.join(out, "cluster_summary.tsv"), ["cluster", "bit_path", "size", "core_words"], rep)
    write_tsv(os.path.join(out, "hierarchy.tsv"), ["node", "left", "right", "distance"], merges)

    # Principal axes.
    comps, var = randomized_pca(X, min(args.pca, X.shape[1]), rng)
    proj = (X - X.mean(axis=0)) @ comps.T
    prow = []
    for j in range(comps.shape[0]):
        o = np.argsort(proj[:, j])
        prow.append((j, float(var[j]), " ".join(words[t] for t in o[-15:][::-1]), " ".join(words[t] for t in o[:15])))
    write_tsv(os.path.join(out, "pca.tsv"), ["component", "variance_share", "positive_end", "negative_end"], prow)

    # Relation offsets (linear relational structure in the static space).
    pairs = dict(BUILTIN_PAIRS)
    if args.pairs:
        pairs = collections.defaultdict(list)
        for ln in read_lines(args.pairs):
            p = ln.split("\t")
            if len(p) >= 3:
                pairs[p[0]].append((p[1], p[2]))
    rel_rows = []
    En = l2n(E.astype(np.float32))
    word_mask = np.zeros(v.n, dtype=bool)
    word_mask[ids] = True
    for rel, plist in pairs.items():
        tp = [(v.single_token_id(a), v.single_token_id(b)) for a, b in plist]
        tp = [(a, b) for a, b in tp if a is not None and b is not None]
        if len(tp) < 3:
            rel_rows.append((rel, len(tp), "", "", "too few single-token pairs"))
            continue
        hit1 = hit10 = 0
        for j, (a, b) in enumerate(tp):
            rest = [tp[t] for t in range(len(tp)) if t != j]
            off = np.mean([En[bb] - En[aa] for aa, bb in rest], axis=0)
            q = l2n(En[a] + off)
            sims = En @ q
            sims[~word_mask] = -np.inf
            sims[a] = -np.inf
            top = np.argsort(-sims)[:10]
            hit1 += int(top[0] == b)
            hit10 += int(b in top)
        rel_rows.append((rel, len(tp), hit1 / len(tp), hit10 / len(tp), "leave-one-out offset"))
    write_tsv(os.path.join(out, "relation_offsets.tsv"), ["relation", "pairs", "acc@1", "acc@10", "method"], rel_rows)

    # Direct-path bigrams: E -> final norm -> unembedding, skipping every layer.
    U, g = load_unembed(r, tm)
    if U.shape[0] != v.n and U.shape[1] == v.n:
        U = U.T
    nb_ids = ids[: args.bigram_tokens]
    xin = E[nb_ids] * (g if g is not None else 1.0)
    with open(os.path.join(out, "direct_bigrams.tsv"), "w", encoding="utf-8") as f:
        f.write("word\tid\tnext_tokens_direct_path\n")
        for s in range(0, len(nb_ids), 256):
            lg = xin[s:s + 256] @ U.T
            top = topk_rows(lg, 10)
            for row in range(lg.shape[0]):
                f.write(f"{esc(words[s + row])}\t{nb_ids[s + row]}\t"
                        + " ".join(esc(v.text[t]) for t in top[row]) + "\n")

    np.save(os.path.join(out, "vectors.npy"), X.astype(np.float16))
    write_tsv(os.path.join(out, "vectors_index.tsv"), ["row", "word", "id"], ((j, w, i) for j, (w, i) in enumerate(zip(words, ids))))
    summary = {"tokens_analysed": len(ids), "dim": int(X.shape[1]), "clusters": int(cent.shape[0]),
               "spearman_norm_vs_freq_rank": rho, "pca_variance_share": [float(x) for x in var]}
    with open(os.path.join(out, "embed_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    write_provenance(out, args, r)
    log(f"embed: {len(ids)} words, {cent.shape[0]} clusters, rho(norm, freq rank) = {rho:.3f} -> {out}")


# --------------------------------------------------------------------------- static: neurons and heads

def layer_count(r, tm) -> int:
    n = arch_key(r, "block_count", None)
    if n is not None:
        return int(n)
    return 1 + max((int(m.group(1)) for m in (re.match(r"blk\.(\d+)\.", t) for t in tm) if m), default=-1)


def parse_layers(spec: str | None, n: int) -> list[int]:
    if not spec:
        return list(range(n))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [x for x in out if 0 <= x < n]


def cmd_neurons(args) -> None:
    r = open_gguf(args.model)
    out = ensure_dir(args.out)
    tm = tensor_map(r)
    rng = np.random.default_rng(args.seed)
    v = Vocab(r)
    E = dequant(tm["token_embd.weight"])
    if E.shape[0] != v.n:
        E = E.T
    U, g = load_unembed(r, tm)
    if U.shape[0] != v.n:
        U = U.T
    Ug = U * (g[None, :] if g is not None else 1.0)
    word_ids = np.array(v.word_ids(min_len=2, limit=args.read_tokens))
    Ew = E[word_ids]
    path = os.path.join(out, "neurons.tsv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("layer\texpert\tneuron\twrites_top\twrites_bottom\treads_top\twrite_entropy\twrite_max_logit\n")
        for L in parse_layers(args.layers, layer_count(r, tm)):
            down_name = next((nm for nm in (f"blk.{L}.ffn_down.weight", f"blk.{L}.ffn_down_exps.weight") if nm in tm), None)
            up_name = next((nm for nm in (f"blk.{L}.ffn_up.weight", f"blk.{L}.ffn_up_exps.weight") if nm in tm), None)
            if down_name is None:
                log(f"neurons: layer {L} has no ffn_down tensor; skipped")
                continue
            D = dequant(tm[down_name])
            Up = dequant(tm[up_name]) if up_name else None
            experts = [(None, D, Up)] if D.ndim == 2 else [(e, D[e], Up[e] if Up is not None else None) for e in range(D.shape[0])]
            for ex, Dm, Um in experts:
                n_embd = Ug.shape[1]
                if Dm.shape[0] != n_embd:
                    Dm = Dm.T
                n_ff = Dm.shape[1]
                sel = np.arange(n_ff) if args.max_neurons <= 0 or args.max_neurons >= n_ff else np.sort(rng.choice(n_ff, args.max_neurons, replace=False))
                for s in range(0, sel.size, args.batch):
                    b = sel[s:s + args.batch]
                    lg = Ug @ Dm[:, b]  # vocab x b
                    for j, nid in enumerate(b):
                        col = lg[:, j]
                        top = np.argpartition(-col, 15)[:15]
                        top = top[np.argsort(-col[top])]
                        bot = np.argpartition(col, 15)[:15]
                        bot = bot[np.argsort(col[bot])]
                        p = np.exp(log_softmax(col))
                        reads = ""
                        if Um is not None:
                            key = Um[nid] if Um.shape[0] == n_ff else Um[:, nid]
                            act = Ew @ key
                            rt = np.argsort(-act)[:15]
                            reads = " ".join(esc(v.text[word_ids[t]].strip()) for t in rt)
                        f.write(f"{L}\t{'' if ex is None else ex}\t{int(nid)}\t"
                                + " ".join(esc(v.text[t].strip()) for t in top) + "\t"
                                + " ".join(esc(v.text[t].strip()) for t in bot) + "\t"
                                + reads + f"\t{entropy(p):.4f}\t{float(col.max()):.4f}\n")
            log(f"neurons: layer {L} done")
    write_provenance(out, args, r)
    log(f"neurons -> {path}")


def cmd_heads(args) -> None:
    r = open_gguf(args.model)
    out = ensure_dir(args.out)
    tm = tensor_map(r)
    v = Vocab(r)
    E = dequant(tm["token_embd.weight"])
    if E.shape[0] != v.n:
        E = E.T
    U, g = load_unembed(r, tm)
    if U.shape[0] != v.n:
        U = U.T
    Ug = U * (g[None, :] if g is not None else 1.0)
    n_head = int(arch_key(r, "attention.head_count", 1))
    n_kv = int(arch_key(r, "attention.head_count_kv", n_head) or n_head)
    ids = np.array(v.word_ids(min_len=3, limit=args.tokens))
    rows = []
    for L in parse_layers(args.layers, layer_count(r, tm)):
        names = [f"blk.{L}.attn_v.weight", f"blk.{L}.attn_output.weight"]
        if not all(n in tm for n in names):
            log(f"heads: layer {L} has no separate attn_v/attn_output; skipped")
            continue
        Wv = dequant(tm[names[0]])  # (n_kv*hd, n_embd)
        Wo = dequant(tm[names[1]])  # (n_embd, n_head*hd)
        n_embd = E.shape[1]
        if Wv.shape[1] != n_embd:
            Wv = Wv.T
        if Wo.shape[0] != n_embd:
            Wo = Wo.T
        hd = Wo.shape[1] // n_head
        gn = dequant(tm[f"blk.{L}.attn_norm.weight"]) if f"blk.{L}.attn_norm.weight" in tm else 1.0
        x = E[ids] * gn
        for h in range(n_head):
            kv = h // max(1, n_head // n_kv)
            ov_in = x @ Wv[kv * hd:(kv + 1) * hd].T          # tokens x hd
            ov_out = ov_in @ Wo[:, h * hd:(h + 1) * hd].T     # tokens x n_embd
            lg = ov_out @ Ug.T                                # tokens x vocab
            top = topk_rows(lg, 5)
            c1 = float(np.mean(top[:, 0] == ids))
            c5 = float(np.mean((top == ids[:, None]).any(axis=1)))
            ex = " ".join(f"{esc(v.text[ids[t]].strip())}>{esc(v.text[top[t, 0]].strip())}" for t in range(min(8, len(ids))))
            rows.append((L, h, c1, c5, ex))
    write_tsv(os.path.join(out, "heads.tsv"), ["layer", "head", "copy@1", "copy@5", "examples_token>top_output"], rows)
    write_provenance(out, args, r)
    log(f"heads: {len(rows)} heads -> {out}")


# --------------------------------------------------------------------------- static: spectra

def powerlaw_alpha(ev: np.ndarray) -> tuple[float, float, float]:
    """Continuous power-law MLE on the eigenvalue tail with xmin chosen by KS distance.
    Returns (alpha, xmin, ks)."""
    x = np.sort(ev[ev > 0])
    if x.size < 20:
        return float("nan"), float("nan"), float("nan")
    best = (float("nan"), float("nan"), float("inf"))
    cands = np.unique(np.quantile(x, np.linspace(0.5, 0.95, 19)))
    for xmin in cands:
        tail = x[x >= xmin]
        n = tail.size
        if n < 10:
            continue
        a = 1.0 + n / float(np.sum(np.log(tail / xmin)))
        cdf_emp = np.arange(1, n + 1) / n
        cdf_fit = 1.0 - (tail / xmin) ** (1.0 - a)
        ks = float(np.max(np.abs(cdf_emp - cdf_fit)))
        if ks < best[2]:
            best = (a, float(xmin), ks)
    return best


def cmd_spectra(args) -> None:
    r = open_gguf(args.model)
    out = ensure_dir(args.out)
    rng = np.random.default_rng(args.seed)
    pat = re.compile(args.pattern) if args.pattern else None
    rows = []
    for t in r.tensors:
        if len(t.shape) != 2 or (pat and not pat.search(t.name)):
            continue
        W = dequant(t).astype(np.float64)
        if min(W.shape) < 8:
            continue
        sampled = ""
        if W.shape[0] > args.max_rows:
            W = W[rng.choice(W.shape[0], args.max_rows, replace=False)]
            sampled = f"rows sampled to {args.max_rows}"
        G = W @ W.T if W.shape[0] <= W.shape[1] else W.T @ W
        ev = np.clip(np.linalg.eigvalsh(G), 0, None)[::-1]
        sv = np.sqrt(ev)
        fro2 = float(ev.sum())
        p = sv / max(sv.sum(), 1e-30)
        eff_rank = float(math.exp(entropy(p)))
        stable_rank = fro2 / max(float(ev[0]), 1e-30)
        alpha, xmin, ks = powerlaw_alpha(ev)
        k = np.arange(1, sv.size + 1)
        m = sv > 0
        slope = float(np.polyfit(np.log(k[m][: max(5, m.sum() // 2)]), np.log(sv[m][: max(5, m.sum() // 2)]), 1)[0])
        w = W.ravel()
        sd = float(w.std())
        kurt = float(((w - w.mean()) ** 4).mean() / max(sd ** 4, 1e-30))
        rows.append((t.name, "x".join(map(str, W.shape)), float(sv[0]), stable_rank, eff_rank, alpha, xmin, ks,
                     slope, kurt, float(np.abs(w).max() / max(sd, 1e-30)), sampled))
        log(f"spectra: {t.name} alpha={alpha:.3f} slope={slope:.3f}")
    write_tsv(os.path.join(out, "spectra.tsv"),
              ["tensor", "shape", "sigma_max", "stable_rank", "effective_rank", "esd_alpha", "esd_xmin", "esd_ks",
               "sv_rank_slope", "kurtosis", "max_abs_over_std", "note"], rows)
    write_provenance(out, args, r)
    log(f"spectra: {len(rows)} tensors -> {out}")


# --------------------------------------------------------------------------- dynamic: model runner

class Runner:
    """Thin wrapper over llama-cpp-python with teacher-forced scoring."""

    def __init__(self, path: str, n_ctx: int, threads: int, seed: int, embedding: bool = False, pooling: str = "mean"):
        try:
            import llama_cpp
        except ImportError:
            die("the dynamic commands need llama-cpp-python: pip install llama-cpp-python")
        self.lc = llama_cpp
        kw = dict(model_path=path, n_ctx=n_ctx, n_threads=threads, seed=seed, verbose=False)
        self.pooling = pooling if embedding else None
        if embedding:
            kw["embedding"] = True
            kw["pooling_type"] = {"mean": llama_cpp.LLAMA_POOLING_TYPE_MEAN,
                                  "last": llama_cpp.LLAMA_POOLING_TYPE_LAST,
                                  "none": llama_cpp.LLAMA_POOLING_TYPE_NONE}.get(pooling, llama_cpp.LLAMA_POOLING_TYPE_MEAN)
            kw["n_batch"] = kw["n_ubatch"] = n_ctx  # a whole sequence in one ubatch (pooled outputs)
        else:
            kw["logits_all"] = True
        self.llm = llama_cpp.Llama(**kw)
        self.n_ctx = n_ctx
        self.seed = seed
        self.n_vocab = self.llm.n_vocab()
        yes = self._first_ids([" Yes", " yes", "Yes", "yes", " YES"])
        no = self._first_ids([" No", " no", "No", "no", " NO"])
        both = set(yes) & set(no)
        self._yes = [i for i in yes if i not in both]
        self._no = [i for i in no if i not in both]

    def tok(self, text: str, bos: bool = True, special: bool = False) -> list[int]:
        return self.llm.tokenize(text.encode("utf-8"), add_bos=bos, special=special)

    def piece(self, tid: int) -> str:
        if tid is None or tid < 0:
            return ""
        return self.llm.detokenize([tid], special=True).decode("utf-8", errors="replace")

    def _ptoks(self, prompt: str, special: bool) -> list[int]:
        """Prompt tokens. A rendered chat template (special=True) parses control tokens and brings
        its own BOS when the model needs one, so BOS is added only when it is not already there."""
        if not special:
            return self.tok(prompt) or self._start()
        b = self.piece(self.llm.token_bos())
        return self.tok(prompt, bos=not (b and prompt.startswith(b)), special=True) or self._start()

    def eos_plan(self, reader, mode: str = "auto") -> dict:
        """How a last-token-pooled input must end (Qwen3-Embedding: with <|endoftext|>).
        Reads tokenizer.ggml.add_eos_token / add_sep_token and checks the tokenizer itself:
        does tokenising with special tokens on append an end token? If not, we append one."""
        llm = self.llm
        eot = self.tok("<|endoftext|>", bos=False, special=True)
        if len(eot) == 1 and len(self.tok("<|endoftext|>", bos=False)) > 1:
            eos_id, src = eot[0], "<|endoftext|> in vocabulary"
        else:
            eos_id, src = llm.token_eos(), "tokenizer.ggml.eos_token_id"
        probe, bare = self.tok("hello world", bos=True), self.tok("hello world", bos=False)
        ends = {eos_id, llm.token_eos(), meta(reader, "tokenizer.ggml.seperator_token_id", -1)}
        appends = len(probe) > len(bare) and probe[-1] in ends
        plan = {"add_eos_token_meta": meta(reader, "tokenizer.ggml.add_eos_token", None),
                "add_sep_token_meta": meta(reader, "tokenizer.ggml.add_sep_token", None),
                "tokenizer_appends_end": bool(appends), "tokenizer_end_token_id": int(probe[-1]) if appends else None,
                "eos_token_id": int(eos_id), "eos_token": self.piece(eos_id), "eos_source": src, "mode": mode}
        if mode == "never":
            plan["action"] = "none"
        elif mode == "append" or not appends:
            plan["action"] = "appended"
        else:
            plan["action"] = "tokenizer"
            if probe[-1] != eos_id:
                log(f"WARNING: the tokenizer appends token {probe[-1]} ({self.piece(probe[-1])!r}), "
                    f"not {self.piece(eos_id)!r}; check the model card (pass --eos append to force it)")
        log(f"last pooling: EOS {plan['action']} (add_eos_token={plan['add_eos_token_meta']}, tokenizer appends: "
            f"{appends}, token {eos_id} {plan['eos_token']!r})")
        return plan

    def pooled_tokens(self, text: str, plan: dict | None, special: bool = False) -> list[int]:
        """Tokens of one embedding input, cut to the context, ending in EOS when the plan says so."""
        t = self.tok(text, bos=True, special=special)
        cap = self.n_ctx
        action = (plan or {}).get("action", "none")
        if action == "appended":
            t = t[: cap - 1] + [plan["eos_token_id"]]
        elif action == "tokenizer" and len(t) > cap:
            t = t[: cap - 1] + [t[-1]]
        else:
            t = t[:cap]
        return t or self._start()

    def embed_tokens(self, seqs: list[list[int]]) -> list[np.ndarray]:
        """Embed token sequences, several per decode (as Llama.embed does, but from tokens, so an
        appended EOS survives). Pooled vectors (n_embd,) or, for pooling none, (len, n_embd)."""
        llm, lc = self.llm, self.lc
        d, nb, nmax = llm.n_embd(), llm.n_batch, max(1, int(llm.context_params.n_seq_max))
        batch, out, sizes, used = llm._batch, [], [], 0
        seqs = [s[:nb] for s in seqs]
        i = 0
        while i <= len(seqs):
            if sizes and (i == len(seqs) or used + len(seqs[i]) > nb or len(sizes) >= nmax):
                llm._ctx.kv_cache_clear()
                llm._ctx.decode(batch)
                if self.pooling == "none":
                    flat = np.ctypeslib.as_array(lc.llama_get_embeddings(llm._ctx.ctx), shape=(used * d,))
                    pos = 0
                    for s in sizes:
                        out.append(flat[pos * d:(pos + s) * d].reshape(s, d).copy())
                        pos += s
                else:
                    for k in range(len(sizes)):
                        out.append(np.ctypeslib.as_array(lc.llama_get_embeddings_seq(llm._ctx.ctx, k), shape=(d,)).copy())
                batch.reset()
                sizes, used = [], 0
            if i == len(seqs):
                break
            if not sizes:
                batch.reset()
            batch.add_sequence(seqs[i], len(sizes), True)
            sizes.append(len(seqs[i]))
            used += len(seqs[i])
            i += 1
        return out

    def _start(self) -> list[int]:
        """A token to condition on when a prompt tokenizes to nothing (no BOS in some vocabularies)."""
        b = self.llm.token_bos()
        return [b] if b is not None and b >= 0 else self.tok("\n", bos=False)

    def _first_ids(self, forms) -> list[int]:
        """First content token of each form (a bare-space piece, as SPM emits, is skipped)."""
        ids = set()
        for s in forms:
            for t in self.tok(s, bos=False):
                piece = self.llm.detokenize([t]).decode("utf-8", errors="replace")
                if piece.strip():
                    ids.add(t)
                    break
        return sorted(ids)

    def _eval(self, toks: list[int]):
        """Evaluate toks, reusing any shared prefix already in the context.
        Returns (logit rows for every position, the tokens actually evaluated)."""
        if len(toks) > self.n_ctx:
            toks = toks[-self.n_ctx:]
        cur = self.llm.input_ids[: self.llm.n_tokens].tolist() if self.llm.n_tokens else []
        k = 0
        while k < min(len(cur), len(toks) - 1) and cur[k] == toks[k]:
            k += 1
        try:
            self.llm.n_tokens = k
            self.llm.eval(toks[k:])
        except Exception:
            self.llm.reset()
            self.llm.eval(toks)
        return np.asarray(self.llm.scores[: len(toks)], dtype=np.float32), toks

    def cont_logprobs(self, prompt: str, cont: str) -> list[float]:
        """Log-probabilities of the tokens of `cont` after `prompt` (teacher forcing)."""
        p = self.tok(prompt)
        full = self.tok(prompt + cont)
        if not p:
            p = self._start()
            full = p + full
        k = 0
        while k < min(len(p), len(full)) and p[k] == full[k]:
            k += 1
        if k >= len(full):
            return []
        logits, full = self._eval(full)
        out = []
        for i in range(max(k, 1), len(full)):
            row = logits[i - 1]
            out.append(float(row[full[i]] - logsumexp(row)))
        return out

    def next_logprobs(self, prompt: str, special: bool = False) -> np.ndarray:
        toks = self._ptoks(prompt, special)
        logits, toks = self._eval(toks)
        return log_softmax(logits[len(toks) - 1])

    def option_logprob(self, prompt: str, options: list[str]) -> np.ndarray:
        return np.array([sum(self.cont_logprobs(prompt, o)) for o in options], dtype=np.float64)

    def yes_no(self, prompt: str, special: bool = False) -> float:
        lp = self.next_logprobs(prompt, special)
        ly = logsumexp(lp[self._yes]) if self._yes else -1e9
        ln = logsumexp(lp[self._no]) if self._no else -1e9
        return float(ly - ln)

    def generate(self, prompt: str, n: int, temperature: float, max_tokens: int, stop, seed: int, top_p: float = 0.95,
                 special: bool = False) -> list[str]:
        outs = []
        ptoks = self._ptoks(prompt, special)
        for j in range(n):
            res = self.llm.create_completion(ptoks, max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                                             top_k=0 if temperature > 0 else 1, min_p=0.0, stop=stop or [],
                                             seed=seed + j)
            outs.append(res["choices"][0]["text"])
        self.llm.reset()
        return outs

    def bits_per_byte(self, text: str, stride: int) -> tuple[float, int]:
        toks = self.tok(text)
        nbytes = len(text.encode("utf-8"))
        if len(toks) < 2 or nbytes == 0:
            return float("nan"), nbytes
        win = self.n_ctx
        keep = max(0, win - max(1, stride))  # context carried into each new window
        total = 0.0
        pos = 1  # next token to score; the first token (BOS) has no prediction
        while pos < len(toks):
            lo = max(0, pos - keep)
            hi = min(len(toks), lo + win)
            seg = toks[lo:hi]
            self.llm.reset()
            logits, seg = self._eval(seg)
            for i in range(max(pos - lo, 1), len(seg)):
                row = logits[i - 1]
                total += -(float(row[seg[i]] - logsumexp(row)))
            pos = hi
        return total / math.log(2) / nbytes, nbytes


# --------------------------------------------------------------------------- relation tables

# Wikidata relations: (property, label, cloze templates with {s} and {o}, question, answer type).
WIKIDATA_RELATIONS = [
    ("P36", "capital", ["The capital of {s} is {o}.", "{s}'s capital city is {o}.", "The capital city of {s} is called {o}."], "What is the capital of {s}?", "place"),
    ("P1376", "capital of", ["{s} is the capital of {o}.", "{s} is the capital city of {o}."], "{s} is the capital of which country?", "place"),
    ("P50", "author", ["{s} was written by {o}.", "The author of {s} is {o}.", "{s} is a book by {o}."], "Who wrote {s}?", "person"),
    ("P19", "place of birth", ["{s} was born in {o}.", "The birthplace of {s} is {o}.", "{s} was born in the city of {o}."], "Where was {s} born?", "place"),
    ("P20", "place of death", ["{s} died in {o}.", "{s} passed away in {o}.", "The place of death of {s} is {o}."], "Where did {s} die?", "place"),
    ("P569", "date of birth", ["{s} was born in the year {o}.", "The year of birth of {s} is {o}.", "{s} (born {o})"], "In what year was {s} born?", "date"),
    ("P570", "date of death", ["{s} died in the year {o}.", "The year of death of {s} is {o}."], "In what year did {s} die?", "date"),
    ("P27", "country of citizenship", ["{s} is a citizen of {o}.", "{s} has the nationality of {o}.", "{s} is from the country of {o}."], "What is the nationality of {s}?", "place"),
    ("P106", "occupation", ["{s} worked as a {o}.", "The occupation of {s} is {o}.", "{s} is known professionally as a {o}."], "What was the occupation of {s}?", "thing"),
    ("P69", "educated at", ["{s} studied at {o}.", "{s} was educated at {o}.", "{s} graduated from {o}."], "Where did {s} study?", "organization"),
    ("P108", "employer", ["{s} worked for {o}.", "{s} was employed by {o}.", "The employer of {s} is {o}."], "Who did {s} work for?", "organization"),
    ("P26", "spouse", ["{s} is married to {o}.", "The spouse of {s} is {o}.", "{s}'s wife or husband is {o}."], "Who is {s} married to?", "person"),
    ("P22", "father", ["The father of {s} is {o}.", "{s}'s father was {o}."], "Who is the father of {s}?", "person"),
    ("P25", "mother", ["The mother of {s} is {o}.", "{s}'s mother was {o}."], "Who is the mother of {s}?", "person"),
    ("P40", "child", ["A child of {s} is {o}.", "{s} is the parent of {o}."], "Who is a child of {s}?", "person"),
    ("P166", "award received", ["{s} received the {o}.", "{s} was awarded the {o}.", "{s} won the {o}."], "What award did {s} receive?", "thing"),
    ("P17", "country", ["{s} is located in the country of {o}.", "{s} is in {o}.", "The country of {s} is {o}."], "In which country is {s}?", "place"),
    ("P131", "located in the administrative territorial entity", ["{s} is located in {o}.", "{s} is a place in {o}."], "Where is {s} located?", "place"),
    ("P30", "continent", ["{s} is located on the continent of {o}.", "{s} is in {o}.", "The continent of {s} is {o}."], "On which continent is {s}?", "place"),
    ("P37", "official language", ["The official language of {s} is {o}.", "In {s}, the official language is {o}.", "People in {s} officially speak {o}."], "What is the official language of {s}?", "thing"),
    ("P38", "currency", ["The currency of {s} is the {o}.", "In {s}, people pay with the {o}."], "What is the currency of {s}?", "thing"),
    ("P35", "head of state", ["The head of state of {s} is {o}.", "{o} is the head of state of {s}."], "Who is the head of state of {s}?", "person"),
    ("P6", "head of government", ["The head of government of {s} is {o}.", "{o} leads the government of {s}."], "Who is the head of government of {s}?", "person"),
    ("P1082", "population", ["The population of {s} is about {o}.", "{s} has a population of {o}."], "What is the population of {s}?", "number"),
    ("P571", "inception", ["{s} was founded in {o}.", "{s} was established in the year {o}."], "When was {s} founded?", "date"),
    ("P112", "founded by", ["{s} was founded by {o}.", "The founder of {s} is {o}."], "Who founded {s}?", "person"),
    ("P159", "headquarters location", ["The headquarters of {s} is in {o}.", "{s} is headquartered in {o}."], "Where is {s} headquartered?", "place"),
    ("P452", "industry", ["{s} operates in the {o} industry.", "The industry of {s} is {o}."], "What industry is {s} in?", "thing"),
    ("P127", "owned by", ["{s} is owned by {o}.", "The owner of {s} is {o}."], "Who owns {s}?", "organization"),
    ("P176", "manufacturer", ["{s} is made by {o}.", "{s} is manufactured by {o}.", "The manufacturer of {s} is {o}."], "Who makes {s}?", "organization"),
    ("P495", "country of origin", ["{s} originated in {o}.", "{s} comes from {o}.", "The country of origin of {s} is {o}."], "Where does {s} come from?", "place"),
    ("P136", "genre", ["{s} belongs to the {o} genre.", "The genre of {s} is {o}."], "What genre is {s}?", "thing"),
    ("P175", "performer", ["{s} is performed by {o}.", "{s} is a song by {o}."], "Who performs {s}?", "person"),
    ("P86", "composer", ["{s} was composed by {o}.", "The composer of {s} is {o}."], "Who composed {s}?", "person"),
    ("P57", "director", ["{s} was directed by {o}.", "The director of {s} is {o}."], "Who directed {s}?", "person"),
    ("P161", "cast member", ["{o} starred in {s}.", "A cast member of {s} is {o}."], "Who starred in {s}?", "person"),
    ("P361", "part of", ["{s} is part of {o}.", "{s} is a component of {o}."], "What is {s} part of?", "thing"),
    ("P527", "has part", ["{s} consists of {o}.", "{s} includes {o}."], "What does {s} consist of?", "thing"),
    ("P31", "instance of", ["{s} is a {o}.", "{s} is an example of a {o}."], "What kind of thing is {s}?", "thing"),
    ("P279", "subclass of", ["A {s} is a kind of {o}.", "{s} is a type of {o}."], "What is {s} a type of?", "thing"),
    ("P101", "field of work", ["{s} works in the field of {o}.", "The field of {s} is {o}."], "What field did {s} work in?", "thing"),
    ("P737", "influenced by", ["{s} was influenced by {o}.", "{o} influenced {s}."], "Who influenced {s}?", "person"),
    ("P1412", "languages spoken", ["{s} spoke {o}.", "{s} could speak {o}."], "What language did {s} speak?", "thing"),
    ("P463", "member of", ["{s} is a member of {o}.", "{s} belongs to {o}."], "What is {s} a member of?", "organization"),
    ("P39", "position held", ["{s} served as {o}.", "{s} held the position of {o}."], "What position did {s} hold?", "thing"),
    ("P102", "member of political party", ["{s} is a member of the {o} party.", "{s} belongs to the {o}."], "Which party does {s} belong to?", "organization"),
    ("P364", "original language", ["{s} was originally written in {o}.", "The original language of {s} is {o}."], "What language was {s} originally in?", "thing"),
    ("P61", "discoverer or inventor", ["{s} was invented by {o}.", "{s} was discovered by {o}."], "Who invented {s}?", "person"),
    ("P138", "named after", ["{s} is named after {o}.", "{s} was named in honour of {o}."], "Who or what is {s} named after?", "thing"),
    ("P47", "shares border with", ["{s} shares a border with {o}.", "{s} borders {o}."], "Which country borders {s}?", "place"),
    ("P206", "located next to body of water", ["{s} lies on the {o}.", "{s} is located next to the {o}."], "What body of water is {s} next to?", "place"),
    ("P2048", "height", ["{s} is {o} tall.", "The height of {s} is {o}."], "How tall is {s}?", "number"),
]

INVERSE = {"P36": "P1376", "P1376": "P36", "P361": "P527", "P527": "P361"}

LEXICAL_RELATIONS = [
    ("synonym", ["A synonym of '{s}' is '{o}'.", "Another word for {s} is {o}.", "{s} means the same as {o}."]),
    ("antonym", ["The opposite of '{s}' is '{o}'.", "The antonym of {s} is {o}.", "{s} is the opposite of {o}."]),
    ("hypernym", ["A {s} is a kind of {o}.", "{s} is a type of {o}.", "Every {s} is a {o}."]),
    ("hyponym", ["One kind of {s} is a {o}.", "An example of a {s} is a {o}."]),
    ("meronym", ["A part of a {s} is the {o}.", "A {s} has a {o}."]),
    ("holonym", ["A {s} is part of a {o}.", "You find a {s} as part of a {o}."]),
    ("definition", ["Definition of '{s}': {o}", "The word '{s}' means {o}"]),
    ("plural", ["The plural of '{s}' is '{o}'.", "One {s}, two {o}."]),
    ("past_tense", ["The past tense of '{s}' is '{o}'.", "Today I {s}; yesterday I {o}."]),
]

LEXICAL_CHOICES = [
    ("pos", "In grammar, the word '{s}' is most often used as a", [" noun", " verb", " adjective", " adverb", " pronoun", " preposition", " conjunction", " determiner", " interjection", " numeral"]),
    ("polarity", "The word '{s}' has a", [" positive", " negative", " neutral"]),
    ("entity_type", "{s} is the name of a", [" person", " place", " organization", " product", " event", " work of art", " concept", " date", " number", " animal", " plant"]),
    ("concreteness", "Is '{s}' something you can see or touch? Answer:", [" Yes", " No"]),
    ("formality", "The word '{s}' is", [" formal", " informal", " neutral"]),
]

COMMONSENSE_RELATIONS = [
    ("UsedFor", ["A {s} is used for {o}.", "You use a {s} to {o}."]),
    ("CapableOf", ["A {s} can {o}.", "A {s} is able to {o}."]),
    ("HasProperty", ["A {s} is usually {o}.", "Most {s}s are {o}."]),
    ("AtLocation", ["You are likely to find a {s} in a {o}.", "A {s} is usually found in a {o}."]),
    ("MadeOf", ["A {s} is made of {o}.", "A {s} is made from {o}."]),
    ("HasA", ["A {s} has {o}.", "Every {s} has a {o}."]),
    ("Causes", ["{s} causes {o}.", "{s} can lead to {o}."]),
    ("HasPrerequisite", ["Before you can {s}, you need to {o}.", "To {s}, you first need {o}."]),
    ("Desires", ["A {s} wants {o}.", "A {s} likes {o}."]),
    ("CreatedBy", ["A {s} is created by {o}.", "A {s} is made by {o}."]),
    ("PartOf", ["A {s} is part of a {o}.", "A {s} belongs to a {o}."]),
    ("MotivatedByGoal", ["You would {s} because you want to {o}.", "People {s} in order to {o}."]),
]

STOP = ["\n", ". ", ".\n", ",", ";", " (", "\"", " which ", " who "]


def split_template(tpl: str) -> tuple[str, str]:
    """Prefix before the object, and the separator the object needs."""
    i = tpl.index("{o}")
    pre = tpl[:i]
    if pre.endswith(" "):
        return pre[:-1], " "
    return pre, ""


def fill(tpl: str, s: str, o: str = "") -> str:
    return tpl.replace("{s}", s).replace("{o}", o)


def cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


# --------------------------------------------------------------------------- dynamic: probing core

def probe(runner: Runner, subject: str, templates: list[str], args, question: str | None = None,
          verify: bool = True, extra_candidates: list[str] | None = None) -> dict:
    """Generate and score candidate objects for one (subject, relation).

    Statistics per candidate:
      choice_p     model probability among the candidate set, averaged over templates
      lp_mean      mean token log-probability, averaged over templates
      para_agree   fraction of templates whose greedy answer is this candidate
      sample_frac  fraction of sampled answers that are this candidate
      margin       log-probability gap to the best other candidate (mean over templates)
      verify       log P(Yes) - log P(No) for "Is it true that <statement>?"
      inverse      filled in by the caller when an inverse relation exists
    Per (subject, relation): sem_entropy over sampled-answer clusters and dk_rate.
    """
    prompts = []
    for tpl in templates:
        pre, sep = split_template(fill(tpl, cap(subject) if tpl.startswith("{s}") else subject, "{o}"))
        prompts.append((tpl, pre, sep))
    raw: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    greedy_keys = []
    samples = []
    dk = 0
    total = 0
    for j, (tpl, pre, sep) in enumerate(prompts):
        stop = STOP + (["'"] if pre.endswith("'") else [])
        g = runner.generate(pre, 1, 0.0, args.max_tokens, stop, args.seed + 1000 * j)
        gs = runner.generate(pre, args.samples, args.temperature, args.max_tokens, stop, args.seed + 1000 * j + 1) if args.samples else []
        for k, txt in enumerate(g + gs):
            a = clean_answer(txt)
            total += 1
            if not a or DONT_KNOW.match(a):
                dk += 1
                if k == 0:
                    greedy_keys.append(None)
                continue
            key = norm_key(a)
            if not key:
                if k == 0:
                    greedy_keys.append(None)
                continue
            raw[key][a] += 1
            if k == 0:
                greedy_keys.append(key)
            else:
                samples.append(key)
    if question:
        qa_prompt = f"Question: {fill(question, subject)}\nAnswer:"
        for txt in runner.generate(qa_prompt, 1 + args.samples, args.temperature if args.samples else 0.0,
                                   args.max_tokens, STOP, args.seed + 7):
            a = clean_answer(txt)
            if a and not DONT_KNOW.match(a) and norm_key(a):
                raw[norm_key(a)][a] += 1
    for c in extra_candidates or []:
        if norm_key(c):
            raw[norm_key(c)][c] += 1
    if not raw:
        return {"subject": subject, "candidates": [], "dk_rate": dk / max(1, total), "sem_entropy": 0.0}

    keys = sorted(raw, key=lambda k: -sum(raw[k].values()))[: args.max_candidates]
    display = {k: raw[k].most_common(1)[0][0] for k in keys}
    lp_sum = np.zeros((len(prompts), len(keys)))
    lp_mean = np.zeros((len(prompts), len(keys)))
    for j, (tpl, pre, sep) in enumerate(prompts):
        for c, k in enumerate(keys):
            lps = runner.cont_logprobs(pre, sep + display[k])
            lp_sum[j, c] = sum(lps) if lps else -1e9
            lp_mean[j, c] = (sum(lps) / len(lps)) if lps else -1e9
    choice = np.exp(lp_sum - lp_sum.max(axis=1, keepdims=True))
    choice /= choice.sum(axis=1, keepdims=True)
    # Length-normalised alternative: whole-string probability favours short answers.
    choice_tok = np.exp(lp_mean - lp_mean.max(axis=1, keepdims=True))
    choice_tok /= choice_tok.sum(axis=1, keepdims=True)
    samp = collections.Counter(samples)
    clusters = np.array(list(samp.values()), dtype=np.float64)
    sem_h = entropy(clusters / clusters.sum()) if clusters.size else 0.0
    cands = []
    for c, k in enumerate(keys):
        others = np.delete(lp_sum, c, axis=1)
        margin = float(np.mean(lp_sum[:, c] - (others.max(axis=1) if others.size else lp_sum[:, c])))
        rec = {
            "object": display[k],
            "key": k,
            "choice_p": float(choice[:, c].mean()),
            "choice_p_min": float(choice[:, c].min()),
            "choice_p_tok": float(choice_tok[:, c].mean()),
            "lp_mean": float(lp_mean[:, c].mean()),
            "lp_mean_sd": float(lp_mean[:, c].std()),
            "para_agree": sum(1 for gk in greedy_keys if gk == k) / max(1, len(prompts)),
            "sample_frac": samp.get(k, 0) / max(1, len(samples)),
            "margin": margin,
        }
        cands.append(rec)
    cands.sort(key=lambda r: -r["choice_p"])
    if verify:
        st_tpl = templates[0]
        for rec in cands[: args.verify_top]:
            st = fill(st_tpl, subject, rec["object"]).rstrip(".")
            rec["verify"] = runner.yes_no(f"Question: Is it true that {st}?\nAnswer:")
    return {"subject": subject, "candidates": cands, "dk_rate": dk / max(1, total), "sem_entropy": sem_h,
            "n_templates": len(prompts), "n_samples": len(samples)}


FEATURES = ["choice_p", "para_agree", "sample_frac", "margin", "verify", "lp_mean", "sem_entropy", "dk_rate", "choice_p_tok"]


def feature_vec(rec: dict, res: dict) -> np.ndarray:
    v = [rec.get("choice_p", 0.0), rec.get("para_agree", 0.0), rec.get("sample_frac", 0.0),
         float(np.clip(rec.get("margin", 0.0), -20, 20)) / 10.0, float(np.clip(rec.get("verify", 0.0), -20, 20)) / 10.0,
         float(np.clip(rec.get("lp_mean", -10.0), -20, 0)) / 10.0, res.get("sem_entropy", 0.0), res.get("dk_rate", 0.0),
         rec.get("choice_p_tok", 0.0)]
    return np.array(v, dtype=np.float64)


class Calibration:
    def __init__(self, path: str | None):
        self.c = None
        if path:
            with open(path, encoding="utf-8") as f:
                self.c = json.load(f)

    @staticmethod
    def heuristic(rec: dict) -> float:
        """Uncalibrated stand-in: the model must both prefer the answer and give it under
        several phrasings. Not a probability of truth; run `calibrate` for that."""
        return float(rec.get("choice_p", 0.0) * max(rec.get("para_agree", 0.0), rec.get("sample_frac", 0.0)))

    def confidence(self, rel: str, rec: dict, res: dict) -> tuple[float, bool]:
        if not self.c:
            return self.heuristic(rec), False
        m = self.c["models"].get(rel) or self.c["models"].get("_pooled")
        if not m:
            return self.heuristic(rec), False
        z = float(np.dot(np.array(m["w"]), feature_vec(rec, res)) + m["b"])
        return 1.0 / (1.0 + math.exp(-z)), True

    def prior(self, rel: str) -> tuple[float, float]:
        if self.c and rel in self.c.get("beta_prior", {}):
            a, b = self.c["beta_prior"][rel]
            return float(a), float(b)
        if self.c and "_pooled" in self.c.get("beta_prior", {}):
            a, b = self.c["beta_prior"]["_pooled"]
            return float(a), float(b)
        return 1.0, 3.0  # distilled-NN default prior until calibrated (PLAN 7.1)


def ecs_evidence(model_id: str, method: str, rel: str, conf: float, calibrated: bool, cal: Calibration) -> dict:
    a, b = cal.prior(rel)
    return {"source": model_id, "method": method, "source_trust_beta": [a, b],
            "source_trust_mean": a / (a + b), "extraction_conf": conf, "calibrated": calibrated}


def model_id_of(args) -> str:
    try:
        r = open_gguf(args.model)
        return f"{meta(r, 'general.name', os.path.basename(args.model))} [{os.path.basename(args.model)}]"
    except SystemExit:
        return os.path.basename(args.model)


def load_relations(args) -> list[tuple]:
    rels = WIKIDATA_RELATIONS
    if args.relations_file:
        with open(args.relations_file, encoding="utf-8") as f:
            rels = [tuple(x) for x in json.load(f)]
    if args.relations:
        want = set(args.relations.split(","))
        rels = [r for r in rels if r[0] in want or r[1] in want]
    return rels


def read_subjects(path: str) -> list[tuple[str, str | None]]:
    out = []
    for ln in read_lines(path):
        p = ln.split("\t")
        out.append((p[0].strip(), p[1].strip() if len(p) > 1 and p[1].strip() else None))
    return out


# --------------------------------------------------------------------------- dynamic: facts

def cmd_facts(args) -> None:
    out = ensure_dir(args.out)
    reader = open_gguf(args.model)
    licence_warning(reader)
    runner = Runner(args.model, args.ctx, args.threads, args.seed)
    cal = Calibration(args.calibration)
    mid = model_id_of(args)
    rels = load_relations(args)
    all_rel = {r[0]: r for r in WIKIDATA_RELATIONS}
    subjects = read_subjects(args.subjects)
    path = os.path.join(out, "facts.jsonl")
    done = done_keys(path, ("subject", "relation")) if args.resume else set()
    if not args.resume and os.path.exists(path):
        os.remove(path)
    n = 0
    t0 = time.time()
    for s, sid in subjects:
        for pid, label, tpls, question, atype in rels:
            if (s, pid) in done:
                continue
            res = probe(runner, s, tpls, args, question)
            inv = INVERSE.get(pid)
            for rec in res["candidates"]:
                conf, calibrated = cal.confidence(pid, rec, res)
                rec["confidence"] = conf
                rec["calibrated"] = calibrated
            if inv and inv in all_rel and res["candidates"] and args.inverse:
                for rec in res["candidates"][: args.verify_top]:
                    inv_tpls = all_rel[inv][2]
                    lps = []
                    for tpl in inv_tpls:
                        pre, sep = split_template(fill(tpl, rec["object"], "{o}"))
                        lp = runner.cont_logprobs(pre, sep + s)
                        lps.append(sum(lp) / len(lp) if lp else -1e9)
                    rec["inverse_lp_mean"] = float(np.mean(lps))
            best = res["candidates"][0] if res["candidates"] else None
            obj = {
                "subject": s, "subject_id": sid, "relation": pid, "relation_label": label, "answer_type": atype,
                "object": best["object"] if best else None,
                "confidence": best["confidence"] if best else 0.0,
                "candidates": res["candidates"], "sem_entropy": res["sem_entropy"], "dk_rate": res["dk_rate"],
                "evidence": ecs_evidence(mid, "facts/cloze+qa", pid, best["confidence"] if best else 0.0,
                                         best["calibrated"] if best else False, cal),
            }
            jsonl_append(path, obj)
            n += 1
            if n % 10 == 0:
                log(f"facts: {n} probes, {(time.time() - t0) / n:.1f} s each")
    write_provenance(out, args, reader, {"relations": [r[0] for r in rels]})
    log(f"facts: {n} (subject, relation) probes -> {path}")


# --------------------------------------------------------------------------- dynamic: crawl

CRAWL_FEWSHOT = """Facts about Paris:
country: France
located next to body of water: Seine
instance of: city
capital of: France

Facts about Albert Einstein:
occupation: physicist
place of birth: Ulm
award received: Nobel Prize in Physics
field of work: theoretical physics

Facts about Atlantis:
instance of: legendary island
country: Don't know

Facts about {e}:
"""

ENTITY_LIKE = re.compile(r"^[A-Z0-9][\w\-'.]*( [A-Za-z0-9&][\w\-'.]*){0,6}$")


def relation_templates(label: str) -> list[str]:
    for pid, lab, tpls, q, at in WIKIDATA_RELATIONS:
        if lab == label.lower():
            return tpls
    return [f"The {label} of {{s}} is {{o}}.", f"{{s}}'s {label} is {{o}}.", f"{{s}} - {label}: {{o}}"]


def cmd_crawl(args) -> None:
    out = ensure_dir(args.out)
    reader = open_gguf(args.model)
    licence_warning(reader)
    runner = Runner(args.model, args.ctx, args.threads, args.seed)
    cal = Calibration(args.calibration)
    mid = model_id_of(args)
    path = os.path.join(out, "triples.jsonl")
    if os.path.exists(path) and not args.resume:
        os.remove(path)
    seen_fact = set()
    seen_ent = {}
    queue = collections.deque((s, 0) for s in read_lines(args.seeds))
    facts = 0
    while queue and facts < args.max_facts and len(seen_ent) < args.max_entities:
        ent, depth = queue.popleft()
        ek = norm_key(ent)
        if ek in seen_ent:
            continue
        seen_ent[ek] = depth
        listing = runner.generate(CRAWL_FEWSHOT.replace("{e}", ent), 1, 0.0, args.list_tokens, ["\n\n", "Facts about"], args.seed)[0]
        pairs = []
        for ln in listing.split("\n"):
            if ":" in ln:
                rel, obj = ln.split(":", 1)
                rel, obj = rel.strip().lower(), clean_answer(obj)
                if rel and obj and len(rel) < 60:
                    pairs.append((rel, obj))
        for rel, obj in pairs[: args.max_relations]:
            if DONT_KNOW.match(obj):
                continue
            tpls = relation_templates(rel)
            res = probe(runner, ent, tpls, args, None, verify=True, extra_candidates=[obj])
            okey = norm_key(obj)
            rec = next((c for c in res["candidates"] if c["key"] == okey), None)
            if rec is None:
                continue
            agree_templates = rec["para_agree"] * len(tpls)
            kept = agree_templates >= min(2, len(tpls))  # LMCRAWL rule: two paraphrases must agree
            fk = (ek, rel, okey)
            if fk in seen_fact:
                continue
            seen_fact.add(fk)
            conf, calibrated = cal.confidence(rel, rec, res)
            jsonl_append(path, {
                "subject": ent, "relation": rel, "object": rec["object"], "depth": depth, "kept": kept,
                "listed_by_model": True, "stats": rec, "sem_entropy": res["sem_entropy"], "dk_rate": res["dk_rate"],
                "confidence": conf,
                "evidence": ecs_evidence(mid, "crawl/list+probe", rel, conf, calibrated, cal),
            })
            facts += 1
            if kept and depth + 1 <= args.depth and ENTITY_LIKE.match(rec["object"]) and not rec["object"][:1].isdigit():
                queue.append((rec["object"], depth + 1))
        log(f"crawl: {ent} (depth {depth}) -> {len(pairs)} listed, {facts} facts total, queue {len(queue)}")
    write_tsv(os.path.join(out, "entities.tsv"), ["entity_key", "depth"], sorted(seen_ent.items(), key=lambda kv: kv[1]))
    write_provenance(out, args, reader)
    log(f"crawl: {facts} facts over {len(seen_ent)} entities -> {path}")


# --------------------------------------------------------------------------- dynamic: lexical, taxonomy, commonsense

def word_list(args, reader) -> list[str]:
    if args.words:
        return read_lines(args.words)
    v = Vocab(reader)
    dic = load_dictionary(getattr(args, "dictionary", None))
    if dic is None:
        log("no --words or --dictionary given: using word-initial tokenizer pieces, which include fragments")
    return [v.text[i][1:] for i in v.word_ids(min_len=3, limit=args.top_words, dictionary=dic)]


def cmd_lexical(args) -> None:
    out = ensure_dir(args.out)
    reader = open_gguf(args.model)
    licence_warning(reader)
    runner = Runner(args.model, args.ctx, args.threads, args.seed)
    cal = Calibration(args.calibration)
    mid = model_id_of(args)
    path = os.path.join(out, "lexical.jsonl")
    done = done_keys(path, ("word",)) if args.resume else set()
    if not args.resume and os.path.exists(path):
        os.remove(path)
    words = word_list(args, reader)
    want = set(args.kinds.split(",")) if args.kinds else None
    for n, w in enumerate(words):
        if (w,) in done:
            continue
        rec = {"word": w, "relations": {}, "choices": {}}
        for name, tpls in LEXICAL_RELATIONS:
            if want and name not in want:
                continue
            res = probe(runner, w, tpls, args, None, verify=name not in ("definition",))
            for c in res["candidates"]:
                c["confidence"], c["calibrated"] = cal.confidence(name, c, res)
            rec["relations"][name] = {"candidates": res["candidates"][: args.keep], "sem_entropy": res["sem_entropy"],
                                      "dk_rate": res["dk_rate"]}
        for name, tpl, options in LEXICAL_CHOICES:
            if want and name not in want:
                continue
            lp = runner.option_logprob(fill(tpl, w), options)
            p = np.exp(lp - logsumexp(lp))
            rec["choices"][name] = {"labels": [o.strip() for o in options], "p": [float(x) for x in p],
                                    "best": options[int(np.argmax(p))].strip(), "entropy": entropy(p)}
        rec["evidence"] = ecs_evidence(mid, "lexical", "lexical", 0.0, False, cal)
        jsonl_append(path, rec)
        if (n + 1) % 10 == 0:
            log(f"lexical: {n + 1}/{len(words)}")
    write_provenance(out, args, reader)
    log(f"lexical: {len(words)} words -> {path}")


def cmd_taxonomy(args) -> None:
    out = ensure_dir(args.out)
    reader = open_gguf(args.model)
    licence_warning(reader)
    runner = Runner(args.model, args.ctx, args.threads, args.seed)
    tpls = dict(LEXICAL_RELATIONS)["hypernym"]
    rows = []
    for term in word_list(args, reader):
        cur, chain = term, [norm_key(term)]
        for d in range(args.depth):
            res = probe(runner, cur, tpls, args, None, verify=True)
            if not res["candidates"]:
                break
            best = res["candidates"][0]
            parent = best["object"]
            pk = norm_key(parent)
            cyc = pk in chain
            rows.append((term, d, cur, parent, best["choice_p"], best["para_agree"], best.get("verify", ""),
                         res["sem_entropy"], int(cyc)))
            if cyc or best["choice_p"] < args.min_p:
                break
            chain.append(pk)
            cur = parent
        log(f"taxonomy: {term}: {' > '.join(chain)}")
    write_tsv(os.path.join(out, "taxonomy.tsv"),
              ["term", "depth", "child", "parent", "choice_p", "para_agree", "verify", "sem_entropy", "cycle"], rows)
    write_provenance(out, args, reader)


def cmd_commonsense(args) -> None:
    out = ensure_dir(args.out)
    reader = open_gguf(args.model)
    licence_warning(reader)
    runner = Runner(args.model, args.ctx, args.threads, args.seed)
    cal = Calibration(args.calibration)
    mid = model_id_of(args)
    path = os.path.join(out, "commonsense.jsonl")
    if os.path.exists(path) and not args.resume:
        os.remove(path)
    for c in word_list(args, reader):
        for rel, tpls in COMMONSENSE_RELATIONS:
            res = probe(runner, c, tpls, args, None, verify=True)
            for rec in res["candidates"]:
                rec["confidence"], rec["calibrated"] = cal.confidence(rel, rec, res)
            jsonl_append(path, {"concept": c, "relation": rel, "candidates": res["candidates"][: args.keep],
                                "sem_entropy": res["sem_entropy"], "dk_rate": res["dk_rate"],
                                "evidence": ecs_evidence(mid, "commonsense", rel, 0.0, False, cal)})
        log(f"commonsense: {c}")
    write_provenance(out, args, reader)


# --------------------------------------------------------------------------- dynamic: paraphrase and qa

PARAPHRASE_FEWSHOT = """Rewrite each question in a different way, keeping its meaning and the name in it.
Question: What is the capital of France?
Rewrite: Which city is the capital of France?
Question: Who wrote Hamlet?
Rewrite: Who is the author of Hamlet?
Question: {q}
Rewrite:"""


def cmd_paraphrase(args) -> None:
    out = ensure_dir(args.out)
    reader = open_gguf(args.model)
    licence_warning(reader)
    runner = Runner(args.model, args.ctx, args.threads, args.seed)
    rels = load_relations(args)
    subjects = [s for s, _ in read_subjects(args.subjects)]
    path = os.path.join(out, "patterns.jsonl")
    open(path, "w").close()
    for pid, label, tpls, question, atype in rels:
        probe_subject = subjects[0]
        q = fill(question, probe_subject)
        gens = runner.generate(PARAPHRASE_FEWSHOT.replace("{q}", q), args.n, args.temperature, 40, ["\n"], args.seed)
        seen = set()
        for gtxt in gens:
            para = gtxt.strip()
            if probe_subject not in para or norm_key(para) in seen or norm_key(para) == norm_key(q):
                continue
            seen.add(norm_key(para))
            pattern = para.replace(probe_subject, "{s}")
            agree = []
            for s in subjects[: args.check]:
                a1 = clean_answer(runner.generate(f"Question: {fill(question, s)}\nAnswer:", 1, 0.0, 12, STOP, args.seed)[0])
                a2 = clean_answer(runner.generate(f"Question: {fill(pattern, s)}\nAnswer:", 1, 0.0, 12, STOP, args.seed)[0])
                if norm_key(a1):
                    agree.append(norm_key(a1) == norm_key(a2))
            jsonl_append(path, {"relation": pid, "label": label, "pattern": pattern, "source_question": question,
                                "answer_agreement": float(np.mean(agree)) if agree else None, "checked": len(agree)})
        log(f"paraphrase: {pid} {label}: {len(seen)} patterns")
    write_provenance(out, args, reader)


def cmd_qa(args) -> None:
    out = ensure_dir(args.out)
    rels = {r[0]: r for r in WIKIDATA_RELATIONS}
    patterns = collections.defaultdict(list)
    if args.patterns and not os.path.exists(args.patterns):
        log(f"qa: no patterns file at {args.patterns}; using the built-in questions only")
    elif args.patterns:
        for ln in read_lines(args.patterns):
            o = json.loads(ln)
            if (o.get("answer_agreement") or 0) >= args.min_agreement:
                patterns[o["relation"]].append(o["pattern"])
    path = os.path.join(out, "qa.jsonl")
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for ln in read_lines(args.facts):
            o = json.loads(ln)
            if not o.get("object") or o.get("confidence", 0) < args.min_confidence:
                continue
            r = rels.get(o["relation"])
            if not r:
                continue
            for q in [r[3]] + patterns.get(o["relation"], []):
                f.write(json.dumps({"question": fill(q, o["subject"]), "answer": o["object"],
                                    "relation": o["relation"], "confidence": o["confidence"],
                                    "trust": "synthetic, from a teacher model"}, ensure_ascii=False) + "\n")
                n += 1
    log(f"qa: {n} pairs -> {path}")


# --------------------------------------------------------------------------- dynamic: sample and score

DEFAULT_PROMPTS = ["", "The", "In", "A", "It", "When", "One of the", "There are", "According to", "Once upon a time"]
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def text_stats(docs: list[str]) -> dict:
    uni = collections.Counter()
    bi = collections.Counter()
    for d in docs:
        w = [x.lower() for x in WORD_RE.findall(d)]
        uni.update(w)
        bi.update(zip(w, w[1:]))
    n = sum(uni.values())
    freqs = np.array(sorted(uni.values(), reverse=True), dtype=np.float64)
    zipf = float(np.polyfit(np.log(np.arange(1, freqs.size + 1)), np.log(freqs), 1)[0]) if freqs.size > 10 else float("nan")
    pmi = []
    nb = max(1, sum(bi.values()))
    for (a, b), c in bi.items():
        if c >= 5:
            pmi.append((a, b, c, math.log((c / nb) / ((uni[a] / n) * (uni[b] / n)))))
    pmi.sort(key=lambda t: -t[3])
    return {"tokens": n, "types": len(uni), "type_token_ratio": len(uni) / max(1, n), "zipf_slope": zipf,
            "top_unigrams": uni.most_common(50), "top_collocations_pmi": pmi[:100]}


def cmd_sample(args) -> None:
    out = ensure_dir(args.out)
    reader = open_gguf(args.model)
    licence_warning(reader)
    runner = Runner(args.model, args.ctx, args.threads, args.seed)
    prompts = read_lines(args.prompts) if args.prompts else DEFAULT_PROMPTS
    docs = []
    with open(os.path.join(out, "corpus.txt"), "w", encoding="utf-8") as f:
        for i in range(args.n):
            p = prompts[i % len(prompts)]
            t = runner.generate(p, 1, args.temperature, args.max_tokens, None, args.seed + i, top_p=args.top_p)[0]
            doc = p + t
            docs.append(doc)
            f.write(doc.replace("<|doc|>", "") + "\n\n<|doc|>\n\n")
            if (i + 1) % 10 == 0:
                log(f"sample: {i + 1}/{args.n}")
    st = text_stats(docs)
    with open(os.path.join(out, "text_stats.json"), "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2, ensure_ascii=False)
    write_provenance(out, args, reader, {"temperature": args.temperature, "top_p": args.top_p})
    log(f"sample: {args.n} documents, {st['tokens']} words, Zipf slope {st['zipf_slope']:.3f} -> {out}")


def cmd_score(args) -> None:
    out = ensure_dir(args.out)
    reader = open_gguf(args.model)
    runner = Runner(args.model, args.ctx, args.threads, args.seed)
    rows = []
    for path in args.inputs:
        if path.endswith(".jsonl"):
            docs = [json.loads(ln).get(args.field, "") for ln in read_lines(path)]
        else:
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
            docs = [d for d in re.split(r"\n\s*<\|doc\|>\s*\n|\n{2,}", text) if d.strip()] if args.split else [text]
        for j, d in enumerate(docs[: args.max_docs]):
            bpb, nb = runner.bits_per_byte(d, args.stride)
            rows.append((path, j, nb, bpb))
    write_tsv(os.path.join(out, "scores.tsv"), ["file", "doc", "bytes", "bits_per_byte"], rows)
    vals = np.array([r[3] for r in rows if math.isfinite(r[3])])
    write_provenance(out, args, reader, {"mean_bpb": float(vals.mean()) if vals.size else None})
    log(f"score: {len(rows)} documents, mean {vals.mean() if vals.size else float('nan'):.4f} bits/byte -> {out}")


# --------------------------------------------------------------------------- dynamic: vectors and senses

def cmd_vectors(args) -> None:
    out = ensure_dir(args.out)
    reader = open_gguf(args.model)
    licence_warning(reader)
    words = word_list(args, reader)
    rng = np.random.default_rng(args.seed)
    emb = Runner(args.model, args.ctx, args.threads, args.seed, embedding=True, pooling=args.pooling)
    eos = emb.eos_plan(reader, args.eos) if args.pooling == "last" else {"action": "none", "reason": f"pooling {args.pooling}"}
    bos = emb.llm.token_bos()
    V = []
    for i, w in enumerate(words):
        text = w if not args.template else args.template.replace("{w}", w)
        if args.pooling == "mean":  # unchanged behaviour: llama.cpp mean pooling over every token
            e = np.asarray(emb.llm.embed(text), dtype=np.float32)
            V.append(e.reshape(-1, e.shape[-1]).mean(axis=0))
        elif args.pooling == "last":
            V.append(emb.embed_tokens([emb.pooled_tokens(text, eos)])[0])
        else:  # none: per-token states averaged over the text's own tokens (BOS left out)
            toks = emb.tok(text) or emb._start()
            e = emb.embed_tokens([toks])[0]
            lo = 1 if len(toks) > 1 and toks[0] == bos else 0
            V.append(e[lo:].mean(axis=0))
        if (i + 1) % 200 == 0:
            log(f"vectors: {i + 1}/{len(words)}")
    V = np.stack(V)
    if args.dims and args.dims < V.shape[1] and V.shape[0] > args.dims:
        comps, var = randomized_pca(V, args.dims, rng)
        V = (V - V.mean(axis=0)) @ comps.T
    np.save(os.path.join(out, "model_vectors.npy"), V.astype(np.float16))
    write_tsv(os.path.join(out, "model_vectors_index.tsv"), ["row", "word"], enumerate(words))
    Vn = l2n(V)
    with open(os.path.join(out, "model_neighbours.tsv"), "w", encoding="utf-8") as f:
        f.write("word\tneighbours\n")
        S = Vn @ Vn.T
        np.fill_diagonal(S, -np.inf)
        top = topk_rows(S, args.neighbours)
        for i, w in enumerate(words):
            f.write(f"{esc(w)}\t" + " ".join(f"{esc(words[j])}:{S[i, j]:.3f}" for j in top[i]) + "\n")

    if args.senses and args.corpus:
        tok = Runner(args.model, args.ctx, args.threads, args.seed, embedding=True, pooling="none")
        with open(args.corpus, encoding="utf-8", errors="replace") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        res = []
        for target in read_lines(args.senses):
            pat = re.compile(rf"\b{re.escape(target)}\b", re.I)
            ctxs = [ln for ln in lines if pat.search(ln)][: args.max_contexts]
            vecs, used = [], []
            for c in ctxs:
                m = pat.search(c)
                n_pre = len(tok.tok(c[: m.start()]))   # BOS + text before the word
                n_all = len(tok.tok(c[: m.end()]))     # ... + the word itself
                e = np.asarray(tok.llm.embed(c), dtype=np.float32)
                if e.ndim != 2 or n_all > e.shape[0]:
                    continue
                vecs.append(e[n_pre:n_all].mean(axis=0) if n_all > n_pre else e[n_all - 1])
                used.append(c)
            if len(vecs) < 4:
                res.append({"word": target, "contexts": len(vecs), "senses": []})
                continue
            X = l2n(np.stack(vecs))
            best = (float("-inf"), np.zeros(len(vecs), dtype=np.int64))
            min_size = max(2, int(0.05 * len(vecs)))
            for k in range(2, min(6, len(vecs) // min_size + 1)):
                lab, cent = kmeans_cos(X, k, 30, rng)
                if np.bincount(lab, minlength=k).min() < min_size:
                    continue
                sim = X @ cent.T
                own = sim[np.arange(len(lab)), lab]
                other = np.where(np.eye(cent.shape[0], dtype=bool)[lab], -np.inf, sim).max(axis=1)
                sil = float(np.mean((own - other) / np.maximum(1 - other, 1e-6)))
                if sil > best[0]:
                    best = (sil, lab)
            sil, lab = best  # one cluster (no split found) when every k left a tiny cluster
            senses = []
            for c in sorted(set(lab.tolist())):
                ex = [used[i] for i in np.where(lab == c)[0][:3]]
                senses.append({"cluster": int(c), "size": int((lab == c).sum()), "examples": ex})
            res.append({"word": target, "contexts": len(vecs), "silhouette_like": sil, "senses": senses})
            log(f"senses: {target}: {len(senses)} clusters from {len(vecs)} contexts")
        with open(os.path.join(out, "senses.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
    write_provenance(out, args, reader, {"pooling": args.pooling, "eos": eos})
    log(f"vectors: {len(words)} words, dim {V.shape[1]}, pooling {args.pooling} -> {out}")


# --------------------------------------------------------------------------- embedding geometry: maths
# docs/EMBEDDING_GEOMETRY.md section 6. Each function is checked on known answers in `selftest`.

def qsummary(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"n": 0}
    q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"n": int(x.size), "mean": float(x.mean()), "sd": float(x.std()), "min": float(x.min()), "q05": float(q[0]),
            "q25": float(q[1]), "median": float(q[2]), "q75": float(q[3]), "q95": float(q[4]), "max": float(x.max()),
            "iqr": float(q[3] - q[1])}


def sigmoid(v: float) -> float:
    return 1.0 / (1.0 + math.exp(-v)) if v >= 0 else math.exp(v) / (1.0 + math.exp(v))


def wilson(k: int, n: int, z: float = 1.96) -> list:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


def linear_cka(X, Y) -> float:
    """Linear CKA (Kornblith et al. 2019) between two row-aligned sets of vectors."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    if X.shape[0] < min(X.shape[1], Y.shape[1]):
        K, L = X @ X.T, Y @ Y.T
        return float((K * L).sum() / max(math.sqrt(float((K * K).sum() * (L * L).sum())), 1e-300))
    xy = float(np.linalg.norm(Y.T @ X) ** 2)
    return xy / max(float(np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)), 1e-300)


def procrustes(X, Y):
    """Orthogonal R minimising ||XR - Y||_F. Returns (R, ||XR - Y||^2 / ||Y||^2)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    U, _s, Vt = np.linalg.svd(X.T @ Y)
    R = U @ Vt
    return R, float(np.linalg.norm(X @ R - Y) ** 2 / max(float(np.linalg.norm(Y) ** 2), 1e-300))


def pca_project(X, k: int, rng) -> np.ndarray:
    Xc = np.asarray(X, dtype=np.float64)
    Xc = Xc - Xc.mean(axis=0)
    if min(Xc.shape) <= 2048:
        _u, _s, vt = np.linalg.svd(Xc, full_matrices=False)
        return Xc @ vt[:k].T
    comps, _v = randomized_pca(Xc.astype(np.float32), k, rng)
    return Xc @ comps.T.astype(np.float64)


def pca_spectrum(X) -> np.ndarray:
    """Covariance eigenvalues, largest first."""
    Xc = np.asarray(X, dtype=np.float64)
    Xc = Xc - Xc.mean(axis=0)
    G = Xc @ Xc.T if Xc.shape[0] < Xc.shape[1] else Xc.T @ Xc
    return np.clip(np.linalg.eigvalsh(G), 0, None)[::-1] / max(Xc.shape[0] - 1, 1)


def spectrum_stats(ev) -> dict:
    tot = float(np.sum(ev))
    if tot <= 0:
        return {}
    p = np.asarray(ev) / tot
    c = np.cumsum(p)
    d = {f"dims_{int(round(q * 100))}pct": int(np.searchsorted(c, q - 1e-12) + 1) for q in (0.9, 0.95, 0.99)}
    d["effective_rank"] = float(math.exp(entropy(p)))
    d["participation_ratio"] = float(tot ** 2 / float(np.sum(np.asarray(ev) ** 2)))
    d["top1_share"] = float(p[0])
    return d


def twonn(X, discard: float = 0.1, chunk: int = 1024) -> float:
    """TwoNN intrinsic dimension (Facco et al. 2017). mu = r2/r1 is Pareto with exponent d, so
    -log(1 - F(mu)) = d log mu: a line through the origin, fitted with the largest `discard` of mu dropped."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 10:
        return float("nan")
    sq = (X * X).sum(axis=1)
    r = np.empty((n, 2))
    for s in range(0, n, chunk):
        d2 = sq[s:s + chunk, None] + sq[None, :] - 2.0 * (X[s:s + chunk] @ X.T)
        d2[np.arange(d2.shape[0]), np.arange(s, s + d2.shape[0])] = np.inf
        r[s:s + chunk] = np.sqrt(np.sort(np.partition(np.maximum(d2, 0), 1, axis=1)[:, :2], axis=1))
    ok = r[:, 0] > 1e-12
    mu = np.sort(r[ok, 1] / r[ok, 0])
    m = mu.size
    if m < 10:
        return float("nan")
    keep = int(m * (1 - discard))
    x = np.log(mu[:keep])
    y = -np.log(1.0 - np.arange(1, keep + 1) / m)
    return float((x * y).sum() / max(float((x * x).sum()), 1e-300))


def knn_cos(Xn, k: int, Q=None, chunk: int = 1024):
    """Top-k cosine neighbours of each row of Q (default: of Xn itself, self excluded)."""
    self_q = Q is None
    Q = Xn if self_q else Q
    k = min(k, Xn.shape[0] - (1 if self_q else 0))
    idx = np.empty((Q.shape[0], k), dtype=np.int64)
    sims = np.empty((Q.shape[0], k), dtype=np.float32)
    for s in range(0, Q.shape[0], chunk):
        S = Q[s:s + chunk] @ Xn.T
        if self_q:
            S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])] = -np.inf
        t = topk_rows(S, k)
        idx[s:s + chunk] = t
        sims[s:s + chunk] = np.take_along_axis(S, t, axis=1)
    return idx, sims


def csls_knn(Xn, k: int, chunk: int = 1024):
    """Neighbours under CSLS (Conneau et al. 2018): 2 cos(x, y) - r(x) - r(y), r = mean cosine to the k nearest."""
    _i, s = knn_cos(Xn, k, chunk=chunk)
    r = s.mean(axis=1)
    k = s.shape[1]
    idx = np.empty((Xn.shape[0], k), dtype=np.int64)
    for st in range(0, Xn.shape[0], chunk):
        S = 2.0 * (Xn[st:st + chunk] @ Xn.T) - r[None, :]
        S[np.arange(S.shape[0]), np.arange(st, st + S.shape[0])] = -np.inf
        idx[st:st + chunk] = topk_rows(S, k)
    return idx, r


def skewness(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    return float(((x - x.mean()) ** 3).mean() / s ** 3) if s > 0 else 0.0


def auc(pos, neg) -> float:
    """P(a pos score beats a neg score), ties counted half (Mann-Whitney U / n_pos n_neg)."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if not pos.size or not neg.size:
        return float("nan")
    r = rankdata(np.r_[pos, neg])
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def golden_min(f, lo: float, hi: float, iters: int = 80):
    """Minimum of a unimodal f on [lo, hi]."""
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    x = (a + b) / 2
    return x, f(x)


def softmax_nll(S, pos, tau: float) -> float:
    Z = np.asarray(S, dtype=np.float64) / tau
    m = Z.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(Z - m).sum(axis=1))
    return float(np.mean(lse - Z[np.arange(len(pos)), pos]))


def fit_temperature(S, pos, lo: float = 1e-3, hi: float = 10.0):
    """Maximum-likelihood tau of softmax(S / tau) picking column pos[i] in row i (in-batch negatives).
    The NLL is convex in 1/tau, so a golden-section search over log tau finds the optimum."""
    pos = np.asarray(pos, dtype=np.int64)
    lt, v = golden_min(lambda t: softmax_nll(S, pos, math.exp(t)), math.log(lo), math.log(hi))
    return math.exp(lt), v


def align_uniform(Xn, a, b, rng, max_n: int = 2000) -> tuple[float, float]:
    """Wang & Isola (2020): L_align = E||f(x) - f(y)||^2 over positives; L_unif = log E exp(-2||f(x) - f(y)||^2)."""
    al = float(np.mean(((Xn[a] - Xn[b]) ** 2).sum(axis=1)))
    s = rng.choice(Xn.shape[0], min(max_n, Xn.shape[0]), replace=False)
    Y = Xn[s].astype(np.float64)
    iu = np.triu_indices(len(s), 1)
    d2 = np.maximum(2.0 - 2.0 * (Y @ Y.T), 0)[iu]
    return al, float(math.log(np.mean(np.exp(-2.0 * d2))))


def stratified_pairs(Xn, rng, n_anchor: int = 500, near_k: int = 10, per: int = 2, chunk: int = 256):
    """Pairs stratified by the first set's neighbour rank (EMBEDDING_GEOMETRY 7.1): near (top near_k),
    mid (ranks near_k .. n/10) and random, so random pairs do not swamp the agreement score."""
    n = Xn.shape[0]
    anchors = rng.choice(n, min(n_anchor, n), replace=False)
    I, J, T = [], [], []
    nk = min(near_k, n - 1)
    hi = min(max(nk + 2, n // 10), n - 1)
    for s in range(0, len(anchors), chunk):
        A = anchors[s:s + chunk]
        S = Xn[A] @ Xn.T
        S[np.arange(len(A)), A] = -np.inf
        order = np.argsort(-S, axis=1)
        for r, i in enumerate(A):
            for _ in range(per):
                I.append(i), J.append(order[r, rng.integers(nk)]), T.append("near")
                if hi > nk:
                    I.append(i), J.append(order[r, rng.integers(nk, hi)]), T.append("mid")
                j = int(rng.integers(n - 1))
                I.append(i), J.append(j + (j >= i)), T.append("random")
    return np.array(I, dtype=np.int64), np.array(J, dtype=np.int64), np.array(T)


def rowcos(Xn, i, j) -> np.ndarray:
    return (Xn[i] * Xn[j]).sum(axis=1)


def mrl_prefixes(D: int, spec: str | None = None) -> list[int]:
    if spec:
        return sorted({min(int(x), D) for x in spec.split(",") if x.strip()})
    return [2 ** k for k in range(3, 20) if 2 ** k < D] + [D]


def conj_grad(A, b, iters: int = 1000, tol: float = 1e-10) -> np.ndarray:
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rs = float(r @ r)
    b2 = max(float(b @ b), 1e-300)
    for _ in range(iters):
        if rs <= tol * tol * b2:
            break
        Ap = A(p)
        alpha = rs / float(p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        rn = float(r @ r)
        p = r + (rn / rs) * p
        rs = rn
    return x


def ridge_word_weights(sent_words, E, Y, lam_rel: float = 1e-2, iters: int = 2000):
    """Per-word scalar weights a_w for y_s ~ sum_w m_sw a_w e_w (m_sw = count / sentence length), ridge
    towards a_w = 1 (the plain mean). The normal equations are sparse (only words that share a sentence
    interact), so they are solved exactly by conjugate gradients. Returns (a, seen-in-training mask)."""
    V = E.shape[0]
    rows, cols, vals = [], [], []
    b = np.zeros(V)
    for (w, m), y in zip(sent_words, Y):
        if not len(w):
            continue
        Es = E[w].astype(np.float64)
        rows.append(np.repeat(w, len(w)))
        cols.append(np.tile(w, len(w)))
        vals.append(((Es @ Es.T) * np.outer(m, m)).ravel())
        np.add.at(b, w, m * (Es @ np.asarray(y, dtype=np.float64)))
    if not rows:
        return np.ones(V), np.zeros(V, dtype=bool)
    rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)

    def gmul(v):
        return np.bincount(rows, weights=vals * v[cols], minlength=V)

    on = rows == cols
    diag = np.bincount(rows[on], weights=vals[on], minlength=V)
    seen = diag > 0
    lam = lam_rel * max(float(diag[seen].mean()), 1e-12)
    beta = conj_grad(lambda v: gmul(v) + lam * v, b - gmul(np.ones(V)), iters)
    return 1.0 + beta, seen


def ridge_map(P, Y, P_test, lam_rel: float = 0.1) -> np.ndarray:
    """Linear map W = argmin ||PW - Y||^2 + mu ||W||^2 (mu relative to the mean eigenvalue), applied to P_test."""
    P = np.asarray(P, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    mu = lam_rel * max(float((P * P).sum()) / min(P.shape), 1e-12)
    if P.shape[0] < P.shape[1]:
        return (np.asarray(P_test, dtype=np.float64) @ P.T) @ np.linalg.solve(P @ P.T + mu * np.eye(P.shape[0]), Y)
    return np.asarray(P_test, dtype=np.float64) @ np.linalg.solve(P.T @ P + mu * np.eye(P.shape[1]), P.T @ Y)


def r2_multi(Y, P) -> float:
    Y = np.asarray(Y, dtype=np.float64)
    st = float(((Y - Y.mean(axis=0)) ** 2).sum())
    return 1.0 - float(((Y - P) ** 2).sum()) / st if st > 0 else float("nan")


def pair_cos_spearman(Y, P, rng, max_pairs: int = 20000) -> float:
    n = Y.shape[0]
    if n < 3:
        return float("nan")
    if n * (n - 1) // 2 <= max_pairs:
        i, j = np.triu_indices(n, 1)
    else:
        i = rng.integers(n, size=max_pairs)
        j = rng.integers(n - 1, size=max_pairs)
        j = j + (j >= i)
    Yn, Pn = l2n(np.asarray(Y, dtype=np.float64)), l2n(np.asarray(P, dtype=np.float64))
    return spearman(rowcos(Yn, i, j), rowcos(Pn, i, j))


def shift_stats(X, Y, tr, te, ridge: float = 0.1) -> dict:
    """How an instruction moves vectors X -> Y (B8): the share of the held-out shift explained by one
    offset (Q10), by an affine map, and the effective rank of A - I for the affine fit."""
    X = np.asarray(X, dtype=np.float64)
    D = np.asarray(Y, dtype=np.float64) - X
    e_te = float((D[te] ** 2).sum())
    t = D[tr].mean(axis=0)
    off = 1.0 - float(((D[te] - t) ** 2).sum()) / e_te if e_te > 0 else float("nan")
    xb, Xc, Dc = X[tr].mean(axis=0), X[tr] - X[tr].mean(axis=0), D[tr] - t
    mu = ridge * max(float((Xc * Xc).sum()) / min(Xc.shape), 1e-12)
    B = np.linalg.solve(Xc @ Xc.T + mu * np.eye(len(tr)), Dc)  # A - I = Xc^T B (dual ridge)
    pred = ((X[te] - xb) @ Xc.T) @ B + t
    aff = 1.0 - float(((D[te] - pred) ** 2).sum()) / e_te if e_te > 0 else float("nan")
    sv = np.linalg.svd(np.linalg.qr(Xc.T)[1] @ B, compute_uv=False)
    tot = float(sv.sum())
    eff = float(math.exp(entropy(sv / tot))) if tot > 0 else 0.0
    e2 = np.cumsum(sv ** 2) / max(float((sv ** 2).sum()), 1e-300)
    return {"offset_explained": off, "affine_explained": aff, "offset_norm": float(np.linalg.norm(t)),
            "A_minus_I_effective_rank": eff, "A_minus_I_rank90": int(np.searchsorted(e2, 0.9 - 1e-12) + 1), "offset": t}


def randomized_svals(A, k: int, rng, iters: int = 4, over: int = 10) -> np.ndarray:
    """Top-k singular values: randomized range finder with power iterations (Halko et al. 2011)."""
    A = np.asarray(A, dtype=np.float32)
    m, n = A.shape
    p = min(k + over, m, n)
    Q = np.linalg.qr(A @ rng.standard_normal((n, p)).astype(np.float32))[0]
    for _ in range(iters):
        Q = np.linalg.qr(A.T @ Q)[0]
        Q = np.linalg.qr(A @ Q)[0]
    return np.linalg.svd(Q.T @ A, compute_uv=False)[:k].astype(np.float64)


def sumsq(A, rows: int = 4096) -> float:
    A = A.reshape(A.shape[0], -1) if A.ndim > 1 else A.reshape(1, -1)
    return float(sum(np.einsum("ij,ij->", A[s:s + rows], A[s:s + rows], dtype=np.float64) for s in range(0, A.shape[0], rows)))


def delta_stats(Wa, Wb, rng, max_full: int = 1024, top_frac: float = 0.05, iters: int = 4) -> dict:
    """A4: size and shape of dW = Wa - Wb. top_share is the part of ||dW||_F^2 in the top `top_frac` of
    singular values (prediction Q4); randomized SVD (slightly low, so conservative) for big matrices."""
    D = np.asarray(Wa, dtype=np.float32) - np.asarray(Wb, dtype=np.float32)
    fro2, base2 = sumsq(D), sumsq(np.asarray(Wb, dtype=np.float32))
    o = {"fro": math.sqrt(fro2), "base_fro": math.sqrt(base2), "rel": math.sqrt(fro2 / base2) if base2 > 0 else float("nan"),
         "k_top": None, "top_share": float("nan"), "sigma_max": float("nan"), "stable_rank": float("nan"),
         "esd_alpha": float("nan"), "method": "norm only"}
    if D.ndim != 2 or min(D.shape) < 2 or fro2 <= 0:
        return o
    r = min(D.shape)
    k = max(1, int(math.ceil(top_frac * r)))
    if r <= max_full:
        G = (D @ D.T if D.shape[0] <= D.shape[1] else D.T @ D).astype(np.float64)
        ev = np.clip(np.linalg.eigvalsh(G), 0, None)[::-1]
        sv = np.sqrt(ev)
        o["esd_alpha"] = powerlaw_alpha(ev)[0]
        o["method"] = "full"
    else:
        sv = randomized_svals(D, k, rng, iters)
        o["method"] = f"randomized k={k}"
    o.update(k_top=k, top_share=float((sv[:k] ** 2).sum()) / fro2, sigma_max=float(sv[0]),
             stable_rank=fro2 / max(float(sv[0]) ** 2, 1e-300))
    return o


# --------------------------------------------------------------------------- offline: vectors in, geometry out

def load_vectors(path: str, index: str | None = None):
    """A (n, d) .npy (fp16 or fp32) and its index TSV with a header (default: <stem>_index.tsv or
    index.tsv beside it). Cypha's exported vectors work the same way."""
    if not os.path.exists(path):
        die(f"no such vectors file: {path}")
    X = np.load(path, allow_pickle=False)
    if X.ndim != 2:
        die(f"{path}: expected a 2-D array, got shape {X.shape}")
    X = X.astype(np.float32)
    if index is None:
        stem = os.path.splitext(path)[0]
        for c in (stem + "_index.tsv", os.path.join(os.path.dirname(path) or ".", "index.tsv")):
            if os.path.exists(c):
                index = c
                break
    rows = None
    if index:
        with open(index, encoding="utf-8") as f:
            hdr = f.readline().rstrip("\n").split("\t")
            rows = [dict(zip(hdr, ln.rstrip("\n").split("\t"))) for ln in f if ln.rstrip("\n")]
        if len(rows) != X.shape[0]:
            die(f"{index}: {len(rows)} rows for {X.shape[0]} vectors")
    return X, rows, index


def row_label(rows, i: int) -> str:
    if rows:
        for c in ("text", "word", "item", "id"):
            if c in rows[i]:
                return rows[i][c][:120]
    return str(i)


def row_resolver(rows, key: str | None = None):
    """Map a pairs-file token to a row: a value of the index's key column, else a row number."""
    m = {}
    if rows:
        col = key or next((c for c in ("id", "text", "word", "item") if c in rows[0]), None)
        if col in rows[0]:
            for i in range(len(rows) - 1, -1, -1):
                m[rows[i][col]] = i

    def f(tok: str):
        tok = tok.strip()
        if tok in m:
            return m[tok]
        try:
            return int(tok)
        except ValueError:
            return None
    return f


def read_pair_rows(path: str, resolve, n: int, extra: int = 0) -> list[tuple]:
    out, bad = [], 0
    for ln in read_lines(path):
        p = ln.split("\t")
        if len(p) < 2 + extra:
            bad += 1
            continue
        a, b = resolve(p[0]), resolve(p[1])
        if a is None or b is None or not (0 <= a < n and 0 <= b < n):
            bad += 1
            continue
        out.append((a, b) + tuple(x.strip() for x in p[2:2 + extra]))
    if bad:
        log(f"{path}: {bad} lines skipped (header, or not rows/keys of the index)")
    return out


def read_lengths(spec: str, rows, n: int) -> np.ndarray:
    if spec.startswith("index"):
        col = spec.split(":", 1)[1] if ":" in spec else "n_tokens"
        if not rows or col not in rows[0]:
            die(f"--lengths {spec}: the index has no column {col!r}")
        L = [float(r[col]) for r in rows]
    else:
        L = []
        for ln in read_lines(spec):
            try:
                L.append(float(ln.split("\t")[-1]))
            except ValueError:
                continue
    if len(L) != n:
        die(f"--lengths: {len(L)} lengths for {n} vectors")
    return np.asarray(L, dtype=np.float64)


def input_hashes(paths) -> dict:
    return {os.path.abspath(p): sha256_file(p) for p in paths if p and os.path.exists(p)}


def cmd_geometry(args) -> None:
    out = ensure_dir(args.out)
    rng = np.random.default_rng(args.seed)
    X, rows, index = load_vectors(args.vectors, args.index)
    n, d = X.shape
    if n < 3:
        die("geometry needs at least 3 vectors")
    norms = np.linalg.norm(X, axis=1)
    Xn = l2n(X)
    res = {"n": n, "dim": d, "index": index, "norm_mean": float(norms.mean()),
           "norm_cv": float(norms.std() / max(float(norms.mean()), 1e-12)), "predictions": {}}
    samp = np.sort(rng.choice(n, min(n, args.max_items), replace=False))
    Xs = Xn[samp]
    m = int(min(args.n_pairs, n * (n - 1) // 2))
    i = rng.integers(n, size=m)
    j = rng.integers(n - 1, size=m)
    j = j + (j >= i)
    cos = rowcos(Xn, i, j)
    # B1: the similarity floor and anisotropy
    res["random_pair_cos"] = qsummary(cos)
    tot = Xn.astype(np.float64).sum(axis=0)
    mu = tot / n
    res["mean_vector_norm"] = float(np.linalg.norm(mu))
    res["mean_cos_all_pairs"] = float((tot @ tot - (norms > 0).sum()) / (n * (n - 1)))
    Xc = l2n(Xn - mu.astype(np.float32))
    res["random_pair_cos_centred"] = qsummary(rowcos(Xc, i, j))
    share = mu ** 2 / max(float(mu @ mu), 1e-30)
    res["rogue_dims"] = [{"dim": int(t), "share_of_mean_cos": float(share[t])} for t in np.argsort(-share)[:5]]
    res["predictions"]["Q1_floor_0.05_to_0.5"] = bool(0.05 <= res["random_pair_cos"]["mean"] <= 0.5)
    h, edges = np.histogram(cos, bins=40, range=(-1, 1))
    write_tsv(os.path.join(out, "floor.tsv"), ["cos_lo", "cos_hi", "count"], zip(edges[:-1], edges[1:], h))
    # B6: PCA spectrum and TwoNN
    ev = pca_spectrum(Xs)
    res["pca"] = spectrum_stats(ev)
    c = np.cumsum(ev) / max(float(ev.sum()), 1e-300)
    write_tsv(os.path.join(out, "spectrum.tsv"), ["k", "eigenvalue", "share", "cumulative"],
              ((k + 1, ev[k], ev[k] / max(float(ev.sum()), 1e-300), c[k]) for k in range(min(len(ev), 2048))))
    res["twonn"] = twonn(Xs[: args.twonn_items])
    # B5: hubness before and after CSLS
    k = min(args.knn, len(samp) - 1)
    kidx, _s = knn_cos(Xs, k)
    cidx, _r = csls_knn(Xs, k)
    Nk, Nc = np.bincount(kidx.ravel(), minlength=len(samp)), np.bincount(cidx.ravel(), minlength=len(samp))
    res["hubness"] = {"k": k, "items": len(samp), "skew": skewness(Nk), "skew_csls": skewness(Nc),
                      "max_k_occurrence": int(Nk.max()), "max_k_occurrence_csls": int(Nc.max()),
                      "anti_hub_frac": float((Nk == 0).mean()), "anti_hub_frac_csls": float((Nc == 0).mean())}
    top = np.argsort(-Nk)[:50]
    write_tsv(os.path.join(out, "hubs.tsv"), ["row", "item", "k_occurrence", "k_occurrence_csls"],
              ((int(samp[t]), row_label(rows, int(samp[t])), int(Nk[t]), int(Nc[t])) for t in top))
    # B7 (+B6 per prefix): MRL truncation curves
    I, J, T = stratified_pairs(Xs, rng, args.anchors)
    full = rowcos(Xs, I, J)
    mrl, curve = [], {}
    for p in mrl_prefixes(d, args.prefixes):
        P = l2n(Xs[:, :p])
        cp = rowcos(P, I, J)
        pk, _ = knn_cos(P, k)
        jac = float(np.mean([len(set(a) & set(b)) / max(len(set(a) | set(b)), 1) for a, b in zip(kidx, pk)]))
        st = {t: spearman(full[T == t], cp[T == t]) for t in ("near", "mid", "random")}
        sp = spearman(full, cp)
        curve[p] = sp
        mrl.append((p, sp, st["near"], st["mid"], st["random"], jac, twonn(P[: args.twonn_items]),
                    spectrum_stats(pca_spectrum(P)).get("dims_95pct", 0)))
    write_tsv(os.path.join(out, "mrl.tsv"), ["dims", "pair_spearman_vs_full", "near", "mid", "random", "knn10_jaccard_vs_full",
                                             "twonn", "pca_dims_95pct"], mrl)
    res["mrl"] = {str(p): s for p, s in curve.items()}
    if 256 in curve and 32 in curve:
        res["predictions"]["Q3_256_keeps_95pct"] = bool(curve[256] >= 0.95)
        res["predictions"]["Q3_32_keeps_80pct"] = bool(curve[32] >= 0.80)
    resolve = row_resolver(rows, args.key)
    # B4 and B3: alignment, uniformity, temperature
    if args.pairs:
        pr = read_pair_rows(args.pairs, resolve, n)
        if len(pr) >= 2:
            a = np.array([x[0] for x in pr])
            b = np.array([x[1] for x in pr])
            al, un = align_uniform(Xn, a, b, rng)
            B = min(args.batch_size, len(a))
            nb = len(a) // B
            perm = rng.permutation(len(a))
            S = np.concatenate([Xn[a[perm[t * B:(t + 1) * B]]] @ Xn[b[perm[t * B:(t + 1) * B]]].T for t in range(nb)])
            pos = np.tile(np.arange(B), nb)
            tau, nll = fit_temperature(S, pos)
            res["pairs"] = {"n": len(a), "positive_cos": qsummary(rowcos(Xn, a, b)), "align": al, "uniformity": un,
                            "tau_ml": tau, "nll_at_tau": nll, "batch": B, "batches": nb,
                            "in_batch_top1": float(np.mean(np.argmax(S, axis=1) == pos))}
            res["predictions"]["Q2_tau_0.01_to_0.1"] = bool(0.01 <= tau <= 0.1)
    # B2: per-class similarity and pairwise AUCs
    if args.labelled_pairs:
        groups = collections.defaultdict(list)
        for a, b, cl in read_pair_rows(args.labelled_pairs, resolve, n, extra=1):
            groups[cl].append(float(Xn[a] @ Xn[b]))
        cls = sorted(groups, key=lambda g: -float(np.mean(groups[g])))
        res["classes"] = {g: qsummary(groups[g]) for g in cls}
        write_tsv(os.path.join(out, "classes.tsv"), ["class", "n", "mean", "median", "q25", "q75"],
                  ((g, len(groups[g]), *(res["classes"][g][q] for q in ("mean", "median", "q25", "q75"))) for g in cls))
        write_tsv(os.path.join(out, "class_auc.tsv"), ["higher", "lower", "auc"],
                  ((g1, g2, auc(groups[g1], groups[g2])) for x, g1 in enumerate(cls) for g2 in cls[x + 1:]))
    # B17: length bias
    if args.lengths:
        ll = np.log(np.maximum(read_lengths(args.lengths, rows, n), 1.0))
        dl, ml = np.abs(ll[i] - ll[j]), (ll[i] + ll[j]) / 2
        A = np.stack([np.ones_like(dl), dl, ml], axis=1)
        coef = np.linalg.lstsq(A, cos.astype(np.float64), rcond=None)[0]
        res["length_bias"] = {"coef_intercept": float(coef[0]), "coef_abs_dlog_len": float(coef[1]),
                              "coef_mean_log_len": float(coef[2]), "r2": r2_multi(cos[:, None], (A @ coef)[:, None]),
                              "spearman_cos_vs_abs_dlog_len": spearman(cos, dl),
                              "spearman_log_len_vs_centroid_cos": spearman(ll, Xn @ (mu / max(float(np.linalg.norm(mu)), 1e-30)))}
    # B13: kNN graph
    if not args.no_knn_graph:
        g = np.arange(n) if n <= args.knn_max else np.sort(rng.choice(n, args.knn_max, replace=False))
        gi, gs = knn_cos(Xn[g], args.knn)
        with open(os.path.join(out, "knn_graph.tsv"), "w", encoding="utf-8") as f:
            f.write("row\titem\tneighbours\n")
            for r_, (ii, ss) in enumerate(zip(gi, gs)):
                f.write(f"{int(g[r_])}\t{esc(row_label(rows, int(g[r_])))}\t" + " ".join(f"{int(g[x])}:{s:.4f}" for x, s in zip(ii, ss)) + "\n")
        res["knn_graph"] = {"items": len(g), "k": int(gi.shape[1])}
    with open(os.path.join(out, "geometry.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=_json_default)
    write_provenance(out, args, extra={"inputs": input_hashes([args.vectors, index, args.pairs, args.labelled_pairs])})
    log(f"geometry: n={n} d={d} floor mean {res['random_pair_cos']['mean']:.3f}, twonn {res['twonn']:.1f}, "
        f"hub skew {res['hubness']['skew']:.2f} -> {args.out}")


def cmd_compare(args) -> None:
    out = ensure_dir(args.out)
    rng = np.random.default_rng(args.seed)
    A, ra, ia = load_vectors(args.a, args.index_a)
    B, rb, ib = load_vectors(args.b, args.index_b)
    A0 = A
    if args.key:
        if not ra or not rb or args.key not in ra[0] or args.key not in rb[0]:
            die(f"--key {args.key}: both index files need that column")
        kb = {r[args.key]: x for x, r in enumerate(rb)}
        pairs = [(x, kb[r[args.key]]) for x, r in enumerate(ra) if r[args.key] in kb]
        A, B = A[[p for p, _ in pairs]], B[[q for _, q in pairs]]
        log(f"compare: {len(pairs)} items matched on {args.key}")
    elif A.shape[0] != B.shape[0]:
        die(f"different item counts ({A.shape[0]} vs {B.shape[0]}); pass --key to align by an index column")
    n = A.shape[0]
    if n < 4:
        die("compare needs at least 4 items")
    samp = np.sort(rng.choice(n, min(n, args.max_items), replace=False))
    An, Bn = l2n(A[samp]), l2n(B[samp])
    m = len(samp)
    I, J, T = stratified_pairs(An, rng, args.anchors)
    ca, cb = rowcos(An, I, J), rowcos(Bn, I, J)
    res = {"n": n, "items_used": m, "dim_a": A.shape[1], "dim_b": B.shape[1], "pair_spearman": spearman(ca, cb),
           "pair_spearman_by_stratum": {t: spearman(ca[T == t], cb[T == t]) for t in ("near", "mid", "random")},
           "pairs": int(len(I))}
    k = min(10, m - 1)
    kA, _ = knn_cos(An, k)
    kB, _ = knn_cos(Bn, k)
    kB50, _ = knn_cos(Bn, min(50, m - 1))
    res["knn10_jaccard"] = float(np.mean([len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(kA, kB)]))
    res["recall_top10_in_top50"] = float(np.mean([len(set(x) & set(y)) / len(x) for x, y in zip(kA, kB50)]))
    res["linear_cka"] = linear_cka(An, Bn)
    kp = min(args.procrustes_dim, An.shape[1], Bn.shape[1], m - 1)
    PA, PB = pca_project(An, kp, rng), pca_project(Bn, kp, rng)
    PA /= max(float(np.linalg.norm(PA)), 1e-300)
    PB /= max(float(np.linalg.norm(PB)), 1e-300)
    _R, resid = procrustes(PA, PB)
    res["procrustes"] = {"dims": kp, "residual": resid, "r2": 1.0 - resid}
    # Force match: KL(p_T || p_S) over each anchor's softmax on the other items (section 5.1).
    tau_t, tau_src = args.tau, "--tau"
    if tau_t is None and args.pairs:
        pr = read_pair_rows(args.pairs, row_resolver(ra, args.key), A0.shape[0])
        if len(pr) >= 2:
            Xa = l2n(A0)
            a_, b_ = np.array([x[0] for x in pr]), np.array([x[1] for x in pr])
            tau_t, _ = fit_temperature(Xa[a_] @ Xa[b_].T, np.arange(len(a_)))
            tau_src = "fitted on --pairs (teacher, in-batch)"
    if tau_t is None:
        tau_t, tau_src = 0.05, "default 0.05 (pass --tau, or --pairs to fit it)"
    anc = rng.permutation(m)[: min(args.anchors, m)]
    ST, SS = (An[anc] @ An.T).astype(np.float64), (Bn[anc] @ Bn.T).astype(np.float64)
    ST[np.arange(len(anc)), anc] = -1e6
    SS[np.arange(len(anc)), anc] = -1e6
    lpT = ST / tau_t - (ST / tau_t).max(axis=1, keepdims=True)
    lpT -= np.log(np.exp(lpT).sum(axis=1, keepdims=True))
    pT = np.exp(lpT)

    def kl(tau_s: float) -> float:
        z = SS / tau_s
        z = z - z.max(axis=1, keepdims=True)
        lpS = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(np.mean((pT * (lpT - lpS)).sum(axis=1)))

    lt, kl_fit = golden_min(lambda t: kl(math.exp(t)), math.log(1e-3), math.log(10.0))
    res["force_match"] = {"tau_teacher": tau_t, "tau_teacher_source": tau_src, "kl_at_teacher_tau": kl(tau_t),
                          "tau_student_fitted": math.exp(lt), "kl_at_fitted_tau": kl_fit, "anchors": int(len(anc))}
    if args.tau_student:
        res["force_match"]["kl_at_given_student_tau"] = kl(args.tau_student)
    write_tsv(os.path.join(out, "compare_strata.tsv"), ["stratum", "pairs", "spearman"],
              ((t, int((T == t).sum()), res["pair_spearman_by_stratum"][t]) for t in ("near", "mid", "random")))
    with open(os.path.join(out, "compare.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=_json_default)
    write_provenance(out, args, extra={"inputs": input_hashes([args.a, args.b, ia, ib, args.pairs])})
    log(f"compare: spearman {res['pair_spearman']:.3f}, knn10 jaccard {res['knn10_jaccard']:.3f}, "
        f"CKA {res['linear_cka']:.3f}, procrustes r2 {1 - resid:.3f} -> {args.out}")


def cmd_wdiff(args) -> None:
    out = ensure_dir(args.out)
    rng = np.random.default_rng(args.seed)
    ra, rb = open_gguf(args.model), open_gguf(args.base)
    ta, tb = tensor_map(ra), tensor_map(rb)
    pat = re.compile(args.pattern) if args.pattern else None
    names = [t.name for t in ra.tensors if t.name in tb and (not pat or pat.search(t.name))]
    only = {"model_only": sorted(set(ta) - set(tb)), "base_only": sorted(set(tb) - set(ta))}
    rows, layers = [], collections.defaultdict(lambda: [0.0, 0.0, [], 0])
    for nm in names:
        Wa, Wb = dequant(ta[nm]), dequant(tb[nm])
        if Wa.shape != Wb.shape:
            rows.append((nm, "x".join(map(str, Wa.shape)) + " vs " + "x".join(map(str, Wb.shape)), *([float("nan")] * 7), "", "shape mismatch"))
            continue
        note = ""
        if Wa.ndim == 2 and args.max_rows and Wa.shape[0] > args.max_rows:
            sel = np.sort(rng.choice(Wa.shape[0], args.max_rows, replace=False))
            Wa, Wb = Wa[sel], Wb[sel]
            note = f"rows sampled to {args.max_rows}"
        s = delta_stats(Wa, Wb, rng, args.max_full, args.top_frac, args.iters)
        rows.append((nm, "x".join(map(str, Wa.shape)), s["fro"], s["base_fro"], s["rel"], s["top_share"], s["sigma_max"],
                     s["stable_rank"], s["esd_alpha"], s["method"], note))
        m = re.match(r"blk\.(\d+)\.", nm)
        L = layers[int(m.group(1)) if m else -1]
        L[0] += s["fro"] ** 2
        L[1] += s["base_fro"] ** 2
        if math.isfinite(s["top_share"]):
            L[2].append(s["top_share"])
        L[3] += 1
        del Wa, Wb
    write_tsv(os.path.join(out, "wdiff.tsv"), ["tensor", "shape", "dW_fro", "W_base_fro", "rel_change", "top_share",
                                               "sigma_max", "stable_rank", "esd_alpha", "method", "note"], rows)
    lay = [(("other" if k < 0 else k), v[3], math.sqrt(v[0]), math.sqrt(v[0] / v[1]) if v[1] > 0 else float("nan"),
            float(np.mean(v[2])) if v[2] else float("nan")) for k, v in sorted(layers.items())]
    write_tsv(os.path.join(out, "wdiff_layers.tsv"), ["layer", "tensors", "dW_fro", "rel_change", "mean_top_share"], lay)
    shares = [r[5] for r in rows if isinstance(r[5], float) and math.isfinite(r[5])]
    summary = {"tensors_compared": len(rows), "matrices_with_spectrum": len(shares), "top_frac": args.top_frac,
               "Q4_frac_matrices_top_share_ge_0.5": float(np.mean([x >= 0.5 for x in shares])) if shares else None,
               "Q4_holds_in_most": bool(shares and np.mean([x >= 0.5 for x in shares]) > 0.5),
               "total_rel_change": math.sqrt(sum(v[0] for v in layers.values()) / max(sum(v[1] for v in layers.values()), 1e-300)),
               **only}
    with open(os.path.join(out, "wdiff_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    extra = {"base_path": os.path.abspath(args.base), "base_name": meta(rb, "general.name")}
    if not args.no_hash:
        extra["base_sha256"] = sha256_file(args.base)
    write_provenance(out, args, ra, extra)
    log(f"wdiff: {len(rows)} tensors, relative change {summary['total_rel_change']:.4g} -> {args.out}")


# --------------------------------------------------------------------------- dynamic: embedding-mode teacher

DEFAULT_QUERY_TEMPLATE = "Instruct: {instruction}\nQuery:{text}"
DEFAULT_INSTRUCTIONS = [
    "Given a web search query, retrieve relevant passages that answer the query",
    "Retrieve semantically similar text",
    "Given a question, retrieve questions that are semantically equivalent to the given question",
    "Classify the topic of the given text",
    "Given a claim, find documents that refute the claim",
]


class Embedder:
    """Text embeddings from a GGUF in embedding mode (EMBEDDING_GEOMETRY 2.1): pooling, the EOS check,
    the query-side instruction, the MRL prefix and timing, shared by the B-commands and `pairs`."""

    def __init__(self, args, model: str | None = None):
        self.path = model or args.model
        self.reader = open_gguf(self.path)
        self.pooling = args.pooling
        self.r = Runner(self.path, args.ctx, args.threads, args.seed, embedding=True, pooling=self.pooling)
        self.eos = (self.r.eos_plan(self.reader, args.eos) if self.pooling == "last"
                    else {"action": "none", "reason": f"pooling {self.pooling}"})
        self.instruction = getattr(args, "instruction", None)
        self.template = getattr(args, "template", None) or DEFAULT_QUERY_TEMPLATE
        self.dims = getattr(args, "dims", 0) or 0
        self.batch = max(1, getattr(args, "batch", 32))
        self.n_items = self.n_tokens = 0
        self.seconds = 0.0
        self.cache: dict = {}

    def form(self, text: str, side: str = "document", instruction: str | None = None) -> str:
        ins = self.instruction if instruction is None else instruction
        if side == "query" and ins:
            return self.template.replace("{instruction}", ins).replace("{text}", text)
        return text

    def raw(self, forms: list[str]) -> tuple[np.ndarray, list[int]]:
        """Unnormalised full-width vectors of formatted inputs, and their token counts."""
        V, lens = [], []
        for s in range(0, len(forms), self.batch):
            toks = [self.r.pooled_tokens(f, self.eos) for f in forms[s:s + self.batch]]
            t0 = time.time()
            V.extend(self.r.embed_tokens(toks))
            self.seconds += time.time() - t0
            lens += [len(t) for t in toks]
        self.n_items += len(forms)
        self.n_tokens += sum(lens)
        return (np.stack(V).astype(np.float32) if V else np.zeros((0, self.r.llm.n_embd()), dtype=np.float32)), lens

    def prefix(self, V: np.ndarray) -> np.ndarray:
        """MRL prefix: truncate, then renormalise."""
        if self.dims and self.dims < V.shape[1]:
            V = V[:, :self.dims]
        return l2n(V)

    def encode(self, texts, side: str = "document", instruction: str | None = None) -> np.ndarray:
        forms = [self.form(t, side, instruction) for t in texts]
        need = [f for f in dict.fromkeys(forms) if f not in self.cache]
        if need:
            V, _ = self.raw(need)
            for f, v in zip(need, self.prefix(V)):
                self.cache[f] = v
        return np.stack([self.cache[f] for f in forms]) if forms else np.zeros((0, self.dims or self.r.llm.n_embd()), np.float32)

    def timing(self) -> dict:
        s = max(self.seconds, 1e-9)
        return {"items": self.n_items, "tokens": self.n_tokens, "seconds": round(self.seconds, 3),
                "items_per_s": self.n_items / s if self.n_items else None,
                "tokens_per_s": self.n_tokens / s if self.n_items else None,
                "note": "decode time only (model load excluded); size jobs from this (EMBEDDING_GEOMETRY 9.1)"}

    def info(self) -> dict:
        return {"embedding_model": os.path.abspath(self.path), "pooling": self.pooling, "eos": self.eos,
                "instruction": self.instruction, "template": self.template if self.instruction else None,
                "dims": self.dims or None, "full_dim": self.r.llm.n_embd(), "ctx": self.r.n_ctx, "timing": self.timing()}


def read_items(args) -> list[tuple[str, str, str]]:
    """(id, side, text) per item: one per line, or one column of a TSV with a header."""
    if args.column is None:
        return [(str(i), args.side, t) for i, t in enumerate(read_lines(args.input))]
    out = []
    with open(args.input, encoding="utf-8") as f:
        hdr = f.readline().rstrip("\n").split("\t")

        def col(c):
            if c is None:
                return None
            if c in hdr:
                return hdr.index(c)
            if c.isdigit():
                return int(c)
            die(f"no column {c!r} in {args.input} (header: {hdr})")
        ci, ii, si = col(args.column), col(args.id_column), col(args.side_column)
        for n, ln in enumerate(f):
            p = ln.rstrip("\n").split("\t")
            if len(p) <= ci or not p[ci].strip():
                continue
            side = args.side
            if si is not None and si < len(p):
                side = "query" if p[si].strip().lower() in ("q", "query") else "document"
            out.append((p[ii] if ii is not None and ii < len(p) else str(n), side, p[ci]))
    return out


def cmd_embedtexts(args) -> None:
    out = ensure_dir(args.out)
    items = read_items(args)
    if args.limit:
        items = items[: args.limit]
    if not items:
        die("no input items")
    emb = Embedder(args)
    licence_warning(emb.reader)
    parts = ensure_dir(os.path.join(out, "parts"))
    cfg = {"model_bytes": os.path.getsize(args.model), "pooling": args.pooling, "eos": emb.eos.get("action"),
           "instruction": args.instruction, "template": args.template, "dims": args.dims, "ctx": args.ctx,
           "chunk": args.chunk, "n_items": len(items), "input": os.path.abspath(args.input), "column": args.column}
    cfg_path = os.path.join(parts, "config.json")
    if args.resume and os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            if json.load(f) != cfg:
                die(f"--resume: the settings differ from the run recorded in {cfg_path}; use a new --out")
    elif not args.resume:
        for fn in os.listdir(parts):
            if fn.startswith("part_"):
                os.remove(os.path.join(parts, fn))
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    hdr = ["row", "id", "side", "n_tokens", "norm", "text"]
    starts, resumed = list(range(0, len(items), args.chunk)), 0
    for start in starts:
        stem = os.path.join(parts, f"part_{start:09d}")
        if args.resume and os.path.exists(stem + ".npy") and os.path.exists(stem + ".tsv"):
            resumed += 1
            continue
        sub = items[start:start + args.chunk]
        raw, lens = emb.raw([emb.form(t, side) for _, side, t in sub])
        np.save(stem + ".tmp.npy", emb.prefix(raw).astype(np.float16))
        os.replace(stem + ".tmp.npy", stem + ".npy")
        norms = np.linalg.norm(raw, axis=1)
        write_tsv(stem + ".tmp.tsv", hdr, ((start + x, iid, side, lens[x], norms[x], t) for x, (iid, side, t) in enumerate(sub)))
        os.replace(stem + ".tmp.tsv", stem + ".tsv")
        tm = emb.timing()
        log(f"embedtexts: {start + len(sub)}/{len(items)} items, {tm['items_per_s']:.2f} items/s, {tm['tokens_per_s']:.1f} tokens/s")
    V = np.concatenate([np.load(os.path.join(parts, f"part_{s:09d}.npy")) for s in starts])
    np.save(os.path.join(out, "embeddings.npy"), V)
    with open(os.path.join(out, "embeddings_index.tsv"), "w", encoding="utf-8") as f:
        f.write("\t".join(hdr) + "\n")
        for s in starts:
            with open(os.path.join(parts, f"part_{s:09d}.tsv"), encoding="utf-8") as g:
                g.readline()
                f.write(g.read())
    write_provenance(out, args, emb.reader, {"embedding": emb.info(), "n_items": len(items), "dim": int(V.shape[1]),
                                             "parts": len(starts), "parts_resumed": resumed, "timing": emb.timing()})
    log(f"embedtexts: {len(items)} items, dim {V.shape[1]}, pooling {args.pooling}, EOS {emb.eos.get('action')} -> {out}")


# Minimal-pair rules for `sensitivity` (B10). Small built-in lists; an external TSV takes generated suites (C6).
NEG_AUX = {"is", "are", "was", "were", "can", "will", "does", "do", "did", "has", "have", "had", "should", "would",
           "could", "must", "may", "might"}
NEG_CONTRACT = {"isn't": "is", "aren't": "are", "wasn't": "was", "weren't": "were", "can't": "can", "cannot": "can",
                "won't": "will", "doesn't": "does", "don't": "do", "didn't": "did", "hasn't": "has", "haven't": "have",
                "hadn't": "had", "shouldn't": "should", "wouldn't": "would", "couldn't": "could", "mustn't": "must"}
SVO_NOUNS = ["dog", "cat", "teacher", "student", "doctor", "patient", "police officer", "thief", "company", "customer",
             "lawyer", "judge", "boy", "girl", "manager", "worker"]
SVO_VERBS = ["chased", "praised", "called", "helped", "followed", "paid", "hired", "blamed", "thanked", "visited",
             "warned", "watched"]
SVO_TEMPLATES = ["The {a} {v} the {b}.", "Yesterday the {a} {v} the {b}.", "The {a} {v} the {b} in the morning."]
SVO_RE = re.compile(r"^(The|A|An|the|a|an) ([A-Za-z]+) ([A-Za-z]+(?:ed|s)) (the|a|an) ([A-Za-z]+)([.!?]?)$")
ENTITIES = {
    "country": ["France", "Germany", "Italy", "Spain", "Japan", "China", "India", "Brazil", "Canada", "Australia", "Egypt", "Mexico"],
    "city": ["Paris", "Berlin", "Rome", "Madrid", "Tokyo", "Beijing", "Delhi", "London", "Sydney", "Cairo", "Toronto", "Moscow"],
    "person": ["Albert Einstein", "Marie Curie", "Isaac Newton", "Charles Darwin", "Ada Lovelace", "Alan Turing",
               "William Shakespeare", "Jane Austen", "Nikola Tesla", "Leonardo da Vinci"],
    "first_name": ["John", "Mary", "James", "Linda", "Robert", "Sarah", "David", "Emma", "Michael", "Anna"],
}
SYNONYMS = {"big": "large", "large": "big", "small": "little", "little": "small", "quick": "fast", "fast": "quick",
            "happy": "glad", "glad": "happy", "begin": "start", "start": "begin", "buy": "purchase", "bought": "purchased",
            "purchase": "buy", "car": "automobile", "help": "assist", "assist": "help", "answer": "reply", "reply": "answer",
            "children": "kids", "kids": "children", "house": "home", "home": "house", "difficult": "hard", "hard": "difficult",
            "easy": "simple", "simple": "easy", "speak": "talk", "talk": "speak", "smart": "clever", "clever": "smart",
            "rich": "wealthy", "wealthy": "rich", "end": "finish", "finish": "end", "city": "town", "job": "occupation",
            "won": "received", "muddy": "dirty"}
PAD_PREFIX = ["To be honest, ", "Well, ", "As far as I know, "]
PAD_SUFFIX = [" That is all.", " Thank you.", " (See above.)"]
NUMBER_WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
WORD_PUNCT = "\"'“”‘’.,;:!?()[]{}"


def _case_like(src: str, new: str) -> str:
    return new[:1].upper() + new[1:] if src[:1].isupper() else new


def edit_negation(s: str, rng=None):
    w = s.split(" ")
    for k, t in enumerate(w):
        core = t.strip(WORD_PUNCT)
        low = core.lower()
        if low in NEG_CONTRACT:
            w[k] = t.replace(core, _case_like(core, NEG_CONTRACT[low]), 1)
            return " ".join(w), f"negation removed ({core})"
        if low in NEG_AUX and core == t:
            if k + 1 < len(w) and w[k + 1].strip(WORD_PUNCT).lower() == "not":
                return " ".join(w[:k + 1] + w[k + 2:]), "negation removed (not)"
            if low == "can":
                w[k] = t + "not"
                return " ".join(w), "can -> cannot"
            return " ".join(w[:k + 1] + ["not"] + w[k + 1:]), f"not after {low}"
    return None


def edit_role(s: str, rng=None):
    m = SVO_RE.match(s.strip())
    if not m or m.group(2).lower() == m.group(5).lower():
        return None
    d1, a, v, d2, b, p = m.groups()
    return f"{d1} {b} {v} {d2} {a}{p}", "subject and object swapped"


def edit_number(s: str, rng):
    m = NUM_RE.search(s)
    if m:
        x = m.group(0)
        if "." in x:
            new = f"{float(x) * 2 + 1:.{len(x.split('.')[1])}f}"
        else:
            new = str(int(x) + int(rng.integers(1, 10 if int(x) < 1000 else 100)))
        return s[:m.start()] + new + s[m.end():], f"{x}->{new}"
    for wd in NUMBER_WORDS:
        m = re.search(rf"\b{wd}\b", s, re.I)
        if m:
            other = [x for x in NUMBER_WORDS if x != wd]
            new = _case_like(m.group(0), other[int(rng.integers(len(other)))])
            return s[:m.start()] + new + s[m.end():], f"{m.group(0)}->{new}"
    return None


def edit_entity(s: str, rng):
    for kind, names in ENTITIES.items():
        for nm in names:
            m = re.search(rf"\b{re.escape(nm)}\b", s)
            if m:
                other = [x for x in names if x != nm]
                new = other[int(rng.integers(len(other)))]
                return s[:m.start()] + new + s[m.end():], f"{kind}: {nm}->{new}"
    return None


def edit_shuffle(s: str, rng):
    core, end = s.rstrip(), ""
    if core and core[-1] in ".!?":
        core, end = core[:-1], core[-1]
    w = core.split()
    if len(w) < 3 or len(set(w)) < 2:
        return None
    for _ in range(20):
        new = " ".join(w[i] for i in rng.permutation(len(w)))
        if new != core:
            return new + end, "words shuffled"
    return None


def edit_typo(s: str, rng, n: int = 2):
    w = s.split(" ")
    cand = [k for k, t in enumerate(w) if len(t.strip(WORD_PUNCT)) >= 4 and t.strip(WORD_PUNCT).isalpha()]
    if not cand:
        return None
    notes = []
    for k in rng.choice(cand, min(n, len(cand)), replace=False):
        t = w[k]
        core = t.strip(WORD_PUNCT)
        i0 = t.index(core)
        c = list(core)
        j = int(rng.integers(1, len(c) - 1))
        op = int(rng.integers(4))
        if op == 0:
            c[j], c[j + 1] = c[j + 1], c[j]
        elif op == 1:
            del c[j]
        elif op == 2:
            c.insert(j, c[j])
        else:
            c[j] = chr(ord("a") + (ord(c[j].lower()) - ord("a") + 1 + int(rng.integers(25))) % 26)
        new = "".join(c)
        w[k] = t[:i0] + new + t[i0 + len(core):]
        notes.append(f"{core}->{new}")
    out = " ".join(w)
    return (out, "; ".join(notes)) if out != s else None


def edit_synonym(s: str, rng=None):
    w = s.split(" ")
    for k, t in enumerate(w):
        core = t.strip(WORD_PUNCT)
        if core.lower() in SYNONYMS:
            new = _case_like(core, SYNONYMS[core.lower()])
            w[k] = t.replace(core, new, 1)
            return " ".join(w), f"{core}->{new}"
    return None


def edit_padding(s: str, rng):
    v = int(rng.integers(3))
    if v == 0:
        p = PAD_PREFIX[int(rng.integers(len(PAD_PREFIX)))]
        return p + (s[:1].lower() + s[1:] if s[:2].istitle() and s.split(" ")[0].lower() in ("the", "a", "an", "it", "this") else s), "filler prefix"
    if v == 1:
        return s + PAD_SUFFIX[int(rng.integers(len(PAD_SUFFIX)))], "filler suffix"
    return s + " " * 8, "trailing whitespace"


EDITS = {"negation": edit_negation, "role_swap": edit_role, "number": edit_number, "entity": edit_entity,
         "shuffle": edit_shuffle, "typo": edit_typo, "synonym": edit_synonym, "padding": edit_padding}


def cmd_sensitivity(args) -> None:
    out = ensure_dir(args.out)
    rng = np.random.default_rng(args.seed)
    kinds = args.kinds.split(",") if args.kinds else list(EDITS)
    sents = read_lines(args.sentences) if args.sentences else []
    pairs = []  # (kind, a, b, note, source)
    for kind in kinds:
        if kind not in EDITS:
            die(f"unknown kind {kind!r}; choose from {', '.join(EDITS)}")
        got = 0
        for s in sents:
            if got >= args.per_kind:
                break
            r = edit_typo(s, rng, args.typos) if kind == "typo" else EDITS[kind](s, rng)
            if r and r[0] != s:
                pairs.append((kind, s, r[0], r[1], "rule"))
                got += 1
        if kind == "role_swap":
            for t in range(min(args.role_templates, args.per_kind)):
                a, b = rng.choice(len(SVO_NOUNS), 2, replace=False)
                v, tpl = SVO_VERBS[int(rng.integers(len(SVO_VERBS)))], SVO_TEMPLATES[t % len(SVO_TEMPLATES)]
                pairs.append(("role_swap", tpl.format(a=SVO_NOUNS[a], v=v, b=SVO_NOUNS[b]),
                              tpl.format(a=SVO_NOUNS[b], v=v, b=SVO_NOUNS[a]), "built-in SVO template", "template"))
    if len(sents) >= 2:  # control: the floor, so each kind can be read against unrelated pairs
        for _ in range(min(args.per_kind, len(sents))):
            i, j = rng.choice(len(sents), 2, replace=False)
            pairs.append(("unrelated", sents[i], sents[j], "control", "random"))
    if args.pairs:
        for ln in read_lines(args.pairs):
            p = ln.split("\t")
            if len(p) >= 3 and p[2].strip().lower() != "kind":
                pairs.append((p[2].strip(), p[0], p[1], p[3] if len(p) > 3 else "", "external"))
    if not pairs:
        die("no pairs: give --sentences and/or --pairs")
    emb = Embedder(args)
    side = "query" if args.instruction else "document"
    V = emb.encode([p[1] for p in pairs] + [p[2] for p in pairs], side)
    n = len(pairs)
    cos = (V[:n] * V[n:]).sum(axis=1)
    write_tsv(os.path.join(out, "pairs.tsv"), ["kind", "cos", "note", "source", "a", "b"],
              ((k, float(c), note, src, a, b) for (k, a, b, note, src), c in zip(pairs, cos)))
    by = collections.OrderedDict()
    for (k, *_r), c in zip(pairs, cos):
        by.setdefault(k, []).append(float(c))
    summ = {k: qsummary(v) for k, v in by.items()}
    floor = summ.get("unrelated", {}).get("median")
    rows = []
    for k, q in summ.items():
        above = (q["median"] - floor) / (1 - floor) if floor is not None and floor < 1 else float("nan")
        rows.append((k, q["n"], q["mean"], q["median"], q["q25"], q["q75"], q["iqr"], q["min"], q["max"], above))
    write_tsv(os.path.join(out, "summary.tsv"), ["kind", "n", "mean", "median", "q25", "q75", "iqr", "min", "max",
                                                 "median_above_floor"], rows)
    res = {"kinds": summ, "floor_median": floor}
    if "negation" in summ:
        res["Q6_negation_median_ge_0.8"] = bool(summ["negation"]["median"] >= 0.8)
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    write_provenance(out, args, emb.reader, {"embedding": emb.info(), "pairs": n})
    log(f"sensitivity: {n} pairs over {len(summ)} kinds -> {out}")


def word_key(t: str) -> str:
    return t.strip(WORD_PUNCT).lower()


def cmd_occlusion(args) -> None:
    out = ensure_dir(args.out)
    rng = np.random.default_rng(args.seed)
    sents = read_lines(args.sentences)[: args.max_sentences]
    toks = [s.split() for s in sents]
    cnt = collections.Counter(word_key(t) for ws in toks for t in ws if word_key(t))
    if args.freq:
        cnt = collections.Counter()
        for ln in read_lines(args.freq):
            p = ln.split("\t")
            try:
                cnt[p[0].strip().lower()] += float(p[1])
            except (IndexError, ValueError):
                continue
    total = max(float(sum(cnt.values())), 1.0)
    emb = Embedder(args)
    Y = emb.encode(sents)
    # B15: drop each word, record how far the sentence vector moves
    variants, owners = [], []
    for si, ws in enumerate(toks):
        if len(ws) < 2:
            continue
        for k in range(len(ws)):
            if word_key(ws[k]):
                variants.append(" ".join(ws[:k] + ws[k + 1:]))
                owners.append((si, word_key(ws[k])))
    drops = collections.defaultdict(list)
    if variants:
        Vv = emb.encode(variants)
        dd = 1.0 - (Vv * Y[[o[0] for o in owners]]).sum(axis=1)
        for (_si, key), x in zip(owners, dd):
            drops[key].append(float(x))
    # B16: additivity over single-word vectors
    surf = collections.defaultdict(collections.Counter)
    for ws in toks:
        for t in ws:
            if word_key(t):
                surf[word_key(t)][t.strip(WORD_PUNCT)] += 1
    types = sorted(surf)
    widx = {k: i for i, k in enumerate(types)}
    E = emb.encode([surf[k].most_common(1)[0][0] for k in types]).astype(np.float64)
    sent_words = []
    for ws in toks:
        c = collections.Counter(word_key(t) for t in ws if word_key(t))
        L = max(sum(c.values()), 1)
        sent_words.append((np.array([widx[k] for k in c], dtype=np.int64), np.array([c[k] / L for k in c])))
    n = len(sents)
    perm = rng.permutation(n)
    if n >= 6:
        n_te = max(2, int(round(args.heldout * n)))
        te, tr = np.sort(perm[:n_te]), np.sort(perm[n_te:])
        split = "held-out"
    else:
        te = tr = np.arange(n)
        split = "in-sample (fewer than 6 sentences)"
        log("occlusion: too few sentences for a held-out split; numbers are in-sample")

    def predict(a, idx):
        P = np.zeros((len(idx), E.shape[1]))
        for r_, s in enumerate(idx):
            w, m = sent_words[s]
            if len(w):
                P[r_] = (m * a[w]) @ E[w]
        return P

    p_w = np.array([cnt.get(k, 1.0) / total for k in types])
    sif = args.sif_a / (args.sif_a + p_w)
    alpha, seen = ridge_word_weights([sent_words[s] for s in tr], E, Y[tr], args.ridge)
    fits, rows = {}, []
    for name, a in (("a_uniform_mean", np.ones(len(types))), ("b_sif_mean", sif), ("c_ridge_word_weights", alpha)):
        Ptr, Pte = predict(a, tr), predict(a, te)
        if name != "c_ridge_word_weights":  # one global scale, fitted on train (a mean of unit vectors is short)
            sc = float((Ptr * Y[tr]).sum()) / max(float((Ptr * Ptr).sum()), 1e-300)
            Ptr, Pte = Ptr * sc, Pte * sc
        fits[name] = (Ptr, Pte)
    Ptr_c, Pte_c = fits["c_ridge_word_weights"]
    fits["d_ridge_weights_then_linear_map"] = (ridge_map(Ptr_c, Y[tr], Ptr_c, args.ridge_map), ridge_map(Ptr_c, Y[tr], Pte_c, args.ridge_map))
    res = {"sentences": n, "word_types": len(types), "split": split, "train": int(len(tr)), "test": int(len(te)), "fits": {}}
    for name, (Ptr, Pte) in fits.items():
        r = {"r2_test": r2_multi(Y[te], Pte), "pair_spearman_test": pair_cos_spearman(Y[te], Pte, rng),
             "mean_cos_test": float((l2n(Pte) * Y[te]).sum(axis=1).mean()),
             "r2_train": r2_multi(Y[tr], Ptr)}
        res["fits"][name] = r
        rows.append((name, r["r2_test"], r["pair_spearman_test"], r["mean_cos_test"], r["r2_train"], len(tr), len(te)))
    write_tsv(os.path.join(out, "additivity.tsv"), ["fit", "r2_test", "pair_spearman_test", "mean_cos_test", "r2_train",
                                                    "n_train", "n_test"], rows)
    best = max((res["fits"][k]["pair_spearman_test"] for k in list(res["fits"])[:3]
                if math.isfinite(res["fits"][k]["pair_spearman_test"])), default=float("nan"))
    res["Q5_weighted_sum_pair_spearman"] = best
    res["Q5_holds"] = bool(math.isfinite(best) and best >= 0.8)
    wrows = []
    for k in types:
        d = drops.get(k, [])
        se = float(np.std(d, ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
        wrows.append((k, len(d), float(np.mean(d)) if d else float("nan"), se, cnt.get(k, 0), cnt.get(k, 1.0) / total,
                      float(sif[widx[k]]), float(alpha[widx[k]]), int(seen[widx[k]])))
    wrows.sort(key=lambda r: -(r[2] if math.isfinite(r[2]) else -1))
    write_tsv(os.path.join(out, "word_weights.tsv"), ["word", "occlusions", "mean_cos_drop", "se", "corpus_count", "p_w",
                                                      "sif_weight", "fitted_weight", "fitted_on_train"], wrows)
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    write_provenance(out, args, emb.reader, {"embedding": emb.info(), "sif_a": args.sif_a, "ridge": args.ridge,
                                             "ridge_map": args.ridge_map})
    log(f"occlusion: {n} sentences, {len(types)} word types, best weighted-sum pair Spearman {best:.3f} -> {out}")


def cmd_instruct(args) -> None:
    out = ensure_dir(args.out)
    rng = np.random.default_rng(args.seed)
    queries = read_lines(args.queries)[: args.max_queries]
    if len(queries) < 4:
        die("instruct needs at least 4 queries")
    instructions = read_lines(args.instructions) if args.instructions else DEFAULT_INSTRUCTIONS
    emb = Embedder(args)
    X = emb.encode(queries, "document")  # no instruction: the text as a document is
    n = len(queries)
    perm = rng.permutation(n)
    n_te = max(2, int(round(args.heldout * n)))
    te, tr = np.sort(perm[:n_te]), np.sort(perm[n_te:])
    rows, items, offs = [], [], []
    for k, ins in enumerate(instructions):
        Y = emb.encode(queries, "query", instruction=ins)
        asym = (X * Y).sum(axis=1)  # B9: the same text as query and as document
        st = shift_stats(X, Y, tr, te, args.ridge)
        offs.append(st.pop("offset"))
        top1 = float(np.mean(np.argmax(Y @ X.T, axis=1) == np.arange(n)))
        q = qsummary(asym)
        rows.append((k, ins, n, q["mean"], q["median"], q["q25"], q["q75"], top1, st["offset_norm"], st["offset_explained"],
                     st["affine_explained"], st["A_minus_I_effective_rank"], st["A_minus_I_rank90"]))
        items += [(k, queries[x], float(asym[x]), float(np.linalg.norm(Y[x] - X[x]))) for x in range(n)]
    write_tsv(os.path.join(out, "instruct.tsv"), ["instruction_id", "instruction", "queries", "query_doc_cos_mean",
                                                  "query_doc_cos_median", "q25", "q75", "own_doc_top1", "offset_norm",
                                                  "offset_explained_test", "affine_explained_test", "A_minus_I_effective_rank",
                                                  "A_minus_I_rank90"], rows)
    write_tsv(os.path.join(out, "instruct_items.tsv"), ["instruction_id", "query", "query_doc_cos", "shift_norm"], items)
    O = l2n(np.stack(offs))
    write_tsv(os.path.join(out, "offset_cos.tsv"), ["instruction_a", "instruction_b", "cos"],
              ((a, b, float(O[a] @ O[b])) for a in range(len(offs)) for b in range(a + 1, len(offs))))
    fr = [r[9] for r in rows if math.isfinite(r[9])]
    res = {"queries": n, "train": int(len(tr)), "test": int(len(te)), "instructions": len(instructions),
           "median_offset_explained": float(np.median(fr)) if fr else None,
           "Q10_frac_instructions_offset_ge_0.7": float(np.mean([x >= 0.7 for x in fr])) if fr else None,
           "template": emb.template}
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    write_provenance(out, args, emb.reader, {"embedding": emb.info(), "instructions": instructions})
    log(f"instruct: {n} queries x {len(instructions)} instructions, median offset share {res['median_offset_explained']} -> {out}")


# Synthetic pairs, Qwen style (C1-C3), with a generative teacher such as Qwen3.5-9B.
QUERY_TYPES = {"keyword": "a short keyword-style search query (key terms only, not a sentence)",
               "factual": "a factual question", "summary": "a request for a summary or overview",
               "judgement": "a question asking for a judgement, opinion or evaluation"}
QUERY_LENGTHS = {"short": "Use at most 5 words.", "medium": "Use 6 to 12 words.", "long": "Use 13 to 25 words."}
QUERY_DIFFICULTY = {"easy": "Make it easy: it may reuse words from the document.",
                    "medium": "Make it moderately hard: paraphrase instead of copying words.",
                    "hard": "Make it hard: it should need real understanding of the document and share few words with it."}
NEG_TASKS = {"topic": "Write one sentence on the same topic as the document that does NOT answer the query.",
             "negation": "Rewrite the document so that it says the opposite of its main claim. Change nothing else.",
             "role_swap": "Rewrite the document with the roles of the main participants swapped (who did what to whom).",
             "entity": "Rewrite the document with its main named entity replaced by a different one of the same kind."}
NEG_RULES = {"negation": edit_negation, "role_swap": edit_role, "entity": edit_entity}
VERIFY_Q = "Document: {doc}\nQuery: {query}\nDoes the document answer the query? Answer Yes or No."
AUDIT_Q = ("Query: {query}\nPassage: {doc}\nIs the passage relevant to the query, that is, does it answer the query "
           "or say the same thing? Answer Yes or No.")


class PromptFormat:
    """Plain prompts, or the GGUF's own jinja chat template (Qwen3.5 needs it), thinking off by default.
    A model with no template, or one llama-cpp-python cannot render, gets plain prompts (logged)."""

    def __init__(self, runner: Runner, reader, mode: str = "auto"):
        self.mode, self.fmt, self.warn = "plain", None, None
        tpl = meta(reader, "tokenizer.chat_template", None)
        self.has_think = bool(tpl and "<think>" in tpl)
        if mode == "off":
            return
        if not tpl:
            if mode == "on":
                log("--chat-template on: the model has no chat template, so prompts stay plain (no-op)")
            return
        try:
            from llama_cpp.llama_chat_format import Jinja2ChatFormatter
            self.fmt = Jinja2ChatFormatter(template=tpl, eos_token=runner.piece(runner.llm.token_eos()),
                                           bos_token=runner.piece(runner.llm.token_bos()))
            self.render("ping", False)
            self.mode = "chat"
        except Exception as e:
            self.fmt, self.warn = None, f"{type(e).__name__}: {e}"
            log(f"WARNING: could not apply the model's chat template ({self.warn}); falling back to plain prompts")

    def render(self, user: str, think: bool) -> str:
        p = self.fmt(messages=[{"role": "user", "content": user}], enable_thinking=think).prompt
        if self.has_think and not think and "</think>" not in p[-64:]:
            p += "<think>\n\n</think>\n\n"  # a template that ignores enable_thinking: close the block ourselves
        return p

    def prompt(self, user: str, plain_cue: str, think: bool = False) -> tuple[str, bool]:
        if self.mode == "chat":
            return self.render(user, think), True
        return f"{user}\n{plain_cue}", False

    @property
    def can_think(self) -> bool:
        return self.mode == "chat" and self.has_think

    def info(self) -> dict:
        return {"prompt_mode": self.mode, "template_warning": self.warn, "template_has_think": self.has_think}


def clean_gen(s: str) -> str:
    for ln in s.splitlines():
        ln = re.sub(r"^(query|new text|answer)\s*:\s*", "", ln.strip(), flags=re.I).strip(" \"'`“”")
        if ln:
            return ln
    return ""


def judge(runner: Runner, pf: PromptFormat, user: str, think: bool, think_tokens: int, seed: int) -> tuple[float, int]:
    """log P(Yes) - log P(No) for a yes/no question; with thinking, a capped thought first."""
    think = think and pf.can_think
    prompt, special = pf.prompt(user, "Answer:", think=think)
    thought = ""
    if think:
        thought = runner.generate(prompt, 1, 0.0, think_tokens, ["</think>"], seed, special=True)[0]
        prompt = prompt + thought + "</think>\n\n"
    return runner.yes_no(prompt, special), len(thought)


def pairs_embed_filter(path: str, args) -> dict:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(ln) for ln in f if ln.strip()]
    if not rows:
        return {}
    emb = Embedder(args, model=args.embedder)
    Q = emb.encode([r["query"] for r in rows], "query")
    D = emb.encode([r["document"] for r in rows], "document")
    cos = (Q * D).sum(axis=1)
    pos = {}
    for r, c in zip(rows, cos):
        r["cos"] = float(c)
        if r["kind"] == "positive":
            pos[(r["doc_id"], r["query"])] = float(c)
    tag = {"model": os.path.basename(args.embedder), "pooling": emb.pooling, "eos": emb.eos.get("action"),
           "instruction": emb.instruction}
    for r in rows:
        r["embedder"] = tag
        if r["kind"] == "positive":
            r["pass_cos"] = r["cos"] > args.min_cos
            r["kept"] = r["verify"] > args.min_verify and r["pass_cos"]
        else:
            cp = pos.get((r["doc_id"], r["query"]))
            r["cos_positive"] = cp
            r["hard"] = cp is not None and r["cos"] >= cp - args.neg_margin
            r["kept"] = r["verify"] < -args.min_verify and r["hard"]
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")
    os.replace(path + ".tmp", path)
    return emb.info()


def cmd_pairs(args) -> None:
    out = ensure_dir(args.out)
    if not args.docs and not args.audit:
        die("pairs needs --docs (C1/C2) and/or --audit (C3)")
    reader = open_gguf(args.model)
    licence_warning(reader)
    runner = Runner(args.model, args.ctx, args.threads, args.seed)
    pf = PromptFormat(runner, reader, args.chat_template)
    if args.think and not pf.can_think:
        log("--think: the chat template in use has no <think> block, so judging runs without thinking (no-op)")
    prov = {"generator": model_id_of(args), "tool_version": VERSION, "temperature": args.temperature, **pf.info()}
    types = args.types.split(",")
    lengths, diffs = args.length.split(","), args.difficulty.split(",")
    negs = [x for x in args.negatives.split(",") if x]
    for x, table in ((types, QUERY_TYPES), (lengths, QUERY_LENGTHS), (diffs, QUERY_DIFFICULTY), (negs, NEG_TASKS)):
        bad = [v for v in x if v not in table]
        if bad:
            die(f"unknown choice(s) {bad}; choose from {', '.join(table)}")
    personas = read_lines(args.personas) if args.personas else []
    summary = {"prompt": pf.info()}
    path = os.path.join(out, "pairs.jsonl")
    if args.docs:
        if not args.resume and os.path.exists(path):
            os.remove(path)
        done = done_keys(path, ("doc_id",)) if args.resume else set()
        docs = read_lines(args.docs)[: args.max_docs or None]
        stats = collections.Counter()
        for di, doc in enumerate(docs):
            if (di,) in done:
                continue
            drng = np.random.default_rng([args.seed, di])  # per document, so --resume reproduces the controls
            persona = personas[int(drng.integers(len(personas)))] if personas else None
            rows = []
            for ti, qt in enumerate(types):
                for rep in range(args.n):
                    ctrl = {"type": qt, "length": lengths[int(drng.integers(len(lengths)))],
                            "difficulty": diffs[int(drng.integers(len(diffs)))], "persona": persona}
                    seed = args.seed + 7919 * di + 101 * ti + rep
                    who = f"You are {persona}. " if persona else ""
                    user = (f"{who}Here is a document.\n\nDocument: {doc}\n\nWrite {QUERY_TYPES[qt]} that this document "
                            f"answers. {QUERY_LENGTHS[ctrl['length']]} {QUERY_DIFFICULTY[ctrl['difficulty']]} "
                            f"Reply with the query only, on one line.")
                    prompt, sp = pf.prompt(user, "Query:")
                    q = clean_gen(runner.generate(prompt, 1, args.temperature, args.max_tokens, ["\n"], seed, special=sp)[0])
                    if not q:
                        stats["empty_query"] += 1
                        continue
                    v, _ = judge(runner, pf, VERIFY_Q.format(doc=doc, query=q), False, 0, seed)
                    rows.append({"doc_id": di, "kind": "positive", "query": q, "document": doc, "controls": ctrl,
                                 "verify": v, "p_yes": sigmoid(v), "kept": v > args.min_verify, "method": "generated",
                                 "seed": seed, "provenance": prov})
                    for ni, nk in enumerate(negs):
                        neg, method = None, "rule"
                        if nk in NEG_RULES:
                            r = NEG_RULES[nk](doc, drng)
                            neg = r[0] if r else None
                        if neg is None:
                            method = "generated"
                            prompt, sp = pf.prompt(f"Document: {doc}\nQuery: {q}\n\n{NEG_TASKS[nk]} Reply with the new text only, on one line.", "New text:")
                            neg = clean_gen(runner.generate(prompt, 1, args.temperature, args.neg_tokens, ["\n"], seed + 1 + ni, special=sp)[0])
                        if not neg or neg == doc:
                            stats[f"no_negative_{nk}"] += 1
                            continue
                        vn, _ = judge(runner, pf, VERIFY_Q.format(doc=neg, query=q), False, 0, seed)
                        rows.append({"doc_id": di, "kind": "negative", "neg_kind": nk, "query": q, "document": neg,
                                     "positive": doc, "verify": vn, "p_yes": sigmoid(vn), "kept": vn < -args.min_verify,
                                     "method": method, "seed": seed, "provenance": prov})
            for r in rows:
                jsonl_append(path, r)
            stats["docs"] += 1
            log(f"pairs: document {di + 1}/{len(docs)}: {len(rows)} rows")
        summary["generation"] = dict(stats)
    if args.audit:
        arows = []
        for k, ln in enumerate(read_lines(args.audit)):
            p = ln.split("\t")
            if len(p) < 2 or (k == 0 and p[0].strip().lower() == "query"):
                continue
            v, nth = judge(runner, pf, AUDIT_Q.format(query=p[0], doc=p[1]), args.think, args.think_tokens, args.seed + k)
            arows.append({"query": p[0], "candidate": p[1], "extra": p[2:], "verify": v, "p_yes": sigmoid(v),
                          "relevant": v > 0, "thinking": bool(args.think and pf.can_think), "thought_chars": nth,
                          "provenance": prov})
        with open(os.path.join(out, "audit.jsonl"), "w", encoding="utf-8") as f:
            for r in arows:
                f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")
        kk = sum(r["relevant"] for r in arows)
        summary["audit"] = {"n": len(arows), "judged_relevant": kk, "mask_precision": kk / len(arows) if arows else None,
                            "wilson95": wilson(kk, len(arows))}
    del runner  # free the generator before loading the embedder
    if args.embedder and args.docs and os.path.exists(path):
        summary["embedder"] = pairs_embed_filter(path, args)
    if args.docs and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(ln) for ln in f if ln.strip()]
        kinds = collections.Counter((r["kind"], r.get("neg_kind")) for r in rows)
        kept = collections.Counter((r["kind"], r.get("neg_kind")) for r in rows if r.get("kept"))
        summary["rows"] = {f"{a}{':' + b if b else ''}": {"n": kinds[(a, b)], "kept": kept[(a, b)]} for a, b in kinds}
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    write_provenance(out, args, reader, {"prompt": pf.info(), "thinking_for_judging": bool(args.think and pf.can_think)})
    log(f"pairs: done -> {out}")


# --------------------------------------------------------------------------- calibrate

def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 2000, lr: float = 0.1):
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        gw = X.T @ (p - y) / len(y) + l2 * w / len(y)
        gb = float(np.mean(p - y))
        w -= lr * gw
        b -= lr * gb
    return w, b


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def cmd_calibrate(args) -> None:
    """Gold TSV: subject <TAB> property id <TAB> object [<TAB> alias|alias...]."""
    out = ensure_dir(args.out)
    reader = open_gguf(args.model)
    runner = Runner(args.model, args.ctx, args.threads, args.seed)
    rels = {r[0]: r for r in WIKIDATA_RELATIONS}
    data = collections.defaultdict(list)
    rows = []
    for ln in read_lines(args.gold):
        p = ln.split("\t")
        if len(p) < 3 or p[1] not in rels:
            continue
        s, pid, o = p[0], p[1], p[2]
        golds = {norm_key(o)} | ({norm_key(a) for a in p[3].split("|")} if len(p) > 3 else set())
        res = probe(runner, s, rels[pid][2], args, rels[pid][3])
        if not res["candidates"]:
            data[pid].append((feature_vec({"choice_p": 0.0}, res), 0.0, 0.0))
            rows.append((s, pid, o, "", 0, 0.0))
            continue
        top = res["candidates"][0]
        y = float(top["key"] in golds or any(g and (g in top["key"] or top["key"] in g) for g in golds))
        data[pid].append((feature_vec(top, res), y, top["choice_p"]))
        rows.append((s, pid, o, top["object"], int(y), top["choice_p"]))
    write_tsv(os.path.join(out, "calibration_items.tsv"), ["subject", "relation", "gold", "top", "correct", "choice_p"], rows)
    models, report, beta = {}, {}, {}
    pooled = [x for v in data.values() for x in v]
    groups = dict(data)
    groups["_pooled"] = pooled
    for rel, items in groups.items():
        if not items:
            continue
        X = np.stack([x for x, _, _ in items])
        y = np.array([t for _, t, _ in items])
        raw = np.array([p for _, _, p in items])
        correct = int(y.sum())
        beta[rel] = [1.0 + correct, 1.0 + len(y) - correct]
        rep = {"n": len(y), "accuracy": float(y.mean()), "ece_raw": ece(raw, y), "brier_raw": float(np.mean((raw - y) ** 2))}
        if len(y) >= args.min_items and 0 < correct < len(y):
            w, b = fit_logistic(X, y)
            pc = 1.0 / (1.0 + np.exp(-(X @ w + b)))
            models[rel] = {"w": w.tolist(), "b": float(b), "features": FEATURES}
            rep.update({"ece_calibrated_in_sample": ece(pc, y), "brier_calibrated_in_sample": float(np.mean((pc - y) ** 2))})
        report[rel] = rep
    cal = {"models": models, "beta_prior": beta, "report": report, "features": FEATURES,
           "note": "In-sample numbers; hold out gold items to measure calibration honestly."}
    with open(os.path.join(out, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)
    write_provenance(out, args, reader)
    log(f"calibrate: {len(pooled)} gold facts, pooled accuracy "
        f"{report.get('_pooled', {}).get('accuracy', float('nan')):.3f} -> {out}")


# --------------------------------------------------------------------------- canon

def cmd_canon(args) -> None:
    """Labels TSV: label <TAB> id [<TAB> alias|alias...]. Adds *_id fields by exact, then folded match."""
    table = {}
    for ln in read_lines(args.labels):
        p = ln.split("\t")
        if len(p) < 2:
            continue
        for lab in [p[0]] + (p[2].split("|") if len(p) > 2 else []):
            table.setdefault(norm_key(lab), p[1])
    n = hit = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for ln in read_lines(args.input):
            o = json.loads(ln)
            for fld in args.fields.split(","):
                if o.get(fld):
                    n += 1
                    o[fld + "_id"] = table.get(norm_key(str(o[fld])))
                    hit += o[fld + "_id"] is not None
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    log(f"canon: {hit}/{n} strings matched -> {args.output}")


# --------------------------------------------------------------------------- selftest

QWEN3_STYLE_TEMPLATE = ("{%- for message in messages %}{{- '<|im_start|>' + message.role + '\\n' + message.content + "
                        "'<|im_end|>\\n' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
                        "{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}"
                        "{%- endif %}{%- endif %}")


def build_tiny_model(vocab_gguf: str, out_path: str, seed: int, n_embd: int = 64, n_layer: int = 2,
                     n_head: int = 4, n_ff: int = 128, n_ctx: int = 512, chat_template: str | None = None) -> None:
    from gguf import GGUFWriter, GGUFValueType
    r = open_gguf(vocab_gguf)
    rng = np.random.default_rng(seed)
    toks = meta(r, "tokenizer.ggml.tokens")
    n_vocab = len(toks)
    w = GGUFWriter(out_path, "llama")
    w.add_name("cypha-selftest-random")
    w.add_license("apache-2.0")
    w.add_context_length(n_ctx)
    w.add_embedding_length(n_embd)
    w.add_block_count(n_layer)
    w.add_feed_forward_length(n_ff)
    w.add_head_count(n_head)
    w.add_head_count_kv(n_head)
    w.add_rope_dimension_count(n_embd // n_head)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_file_type(0)
    for k, f in r.fields.items():
        if not k.startswith("tokenizer.") or (chat_template and k == "tokenizer.chat_template"):
            continue
        v = field_value(f)
        t = f.types[0]
        if t == GGUFValueType.ARRAY:
            w.add_array(k, v)
        elif t == GGUFValueType.STRING:
            w.add_string(k, v)
        elif t == GGUFValueType.BOOL:
            w.add_bool(k, bool(v))
        elif t == GGUFValueType.FLOAT32:
            w.add_float32(k, float(v))
        else:
            w.add_uint32(k, int(v))
    if chat_template:
        w.add_string("tokenizer.chat_template", chat_template)
    s = 0.02

    def rnd(*shape):
        return (rng.standard_normal(shape) * s).astype(np.float32)

    w.add_tensor("token_embd.weight", rnd(n_vocab, n_embd))
    for L in range(n_layer):
        w.add_tensor(f"blk.{L}.attn_norm.weight", np.ones(n_embd, dtype=np.float32))
        w.add_tensor(f"blk.{L}.attn_q.weight", rnd(n_embd, n_embd))
        w.add_tensor(f"blk.{L}.attn_k.weight", rnd(n_embd, n_embd))
        w.add_tensor(f"blk.{L}.attn_v.weight", rnd(n_embd, n_embd))
        w.add_tensor(f"blk.{L}.attn_output.weight", rnd(n_embd, n_embd))
        w.add_tensor(f"blk.{L}.ffn_norm.weight", np.ones(n_embd, dtype=np.float32))
        w.add_tensor(f"blk.{L}.ffn_gate.weight", rnd(n_ff, n_embd))
        w.add_tensor(f"blk.{L}.ffn_up.weight", rnd(n_ff, n_embd))
        w.add_tensor(f"blk.{L}.ffn_down.weight", rnd(n_embd, n_ff))
    w.add_tensor("output_norm.weight", np.ones(n_embd, dtype=np.float32))
    w.add_tensor("output.weight", rnd(n_vocab, n_embd))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def selftest_maths(seed: int) -> list[tuple[str, bool, str]]:
    """Known-answer checks for the embedding-geometry maths."""
    rng = np.random.default_rng(seed)
    res = []

    def chk(name, ok, val):
        res.append((f"math:{name}", bool(ok), val))

    # Spearman, worked by hand: d = (-1, 1, -1, 1, 0), rho = 1 - 6*4/(5*24) = 0.8.
    # With ties, y ranks (1, 2, 3.5, 5, 3.5): rho = 8 / sqrt(10 * 9.5) = 8 / sqrt(95).
    s1, s2 = spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]), spearman([1, 2, 3, 4, 5], [5, 6, 7, 8, 7])
    chk("spearman_hand", abs(s1 - 0.8) < 1e-12 and abs(s2 - 8 / math.sqrt(95)) < 1e-12, f"{s1:.6f}=0.8, {s2:.6f}={8 / math.sqrt(95):.6f}")
    a = auc([1, 2, 3], [0, 2])  # wins 1+0+1+0.5+1+1 = 4.5 of 6
    chk("auc_hand", abs(a - 0.75) < 1e-12, f"{a:.4f}=0.75")
    X = rng.standard_normal((300, 24))
    R = np.linalg.qr(rng.standard_normal((24, 24)))[0]
    c, c0 = linear_cka(X, 3.0 * X @ R), linear_cka(X, rng.standard_normal((300, 24)))
    Xw = rng.standard_normal((20, 50))
    Rw = np.linalg.qr(rng.standard_normal((50, 50)))[0]
    cw = linear_cka(Xw, Xw @ Rw)  # the n < d (Gram) branch
    chk("cka_rotation", abs(c - 1) < 1e-9 and abs(cw - 1) < 1e-9 and c0 < 0.3, f"CKA(X,3XR)={c:.10f}, gram {cw:.10f}, random {c0:.3f}")
    R2, resid = procrustes(X, X @ R)
    _r, resid_n = procrustes(X, X @ R + 0.01 * rng.standard_normal(X.shape))
    chk("procrustes_rotation", np.abs(R2 - R).max() < 1e-8 and resid < 1e-20 and resid_n < 1e-3,
        f"max|R'-R|={np.abs(R2 - R).max():.1e}, residual {resid:.1e}, noisy {resid_n:.1e}")
    tw = []
    for d in (2, 5, 8):
        U = rng.random((2000, d)) @ np.linalg.qr(rng.standard_normal((20, d)))[0].T  # uniform cube, rotated into 20-D
        tw.append((d, twonn(U)))
    chk("twonn_uniform", all(abs(e - d) <= 0.25 * d for d, e in tw), ", ".join(f"d={d}: {e:.2f}" for d, e in tw))
    S = rng.uniform(-1, 1, (4000, 32))
    P = np.exp(S / 0.05)
    P /= P.sum(axis=1, keepdims=True)
    pos = (P.cumsum(axis=1) > rng.random((4000, 1))).argmax(axis=1)
    tau, _ = fit_temperature(S, pos)
    chk("temperature_planted", abs(tau - 0.05) / 0.05 < 0.1, f"tau={tau:.4f} (planted 0.05)")
    V, D = 50, 32
    E = l2n(rng.standard_normal((V, D)))
    al = rng.uniform(0.5, 2.0, V)
    sw, Yp = [], []
    for _ in range(400):
        w = rng.choice(V, int(rng.integers(3, 9)), replace=False)
        m = np.full(len(w), 1.0 / len(w))
        sw.append((w, m))
        Yp.append((m * al[w]) @ E[w])
    ah, _seen = ridge_word_weights(sw, E, np.array(Yp), lam_rel=1e-10)
    chk("ridge_weights_planted", np.abs(ah - al).max() < 1e-4, f"max|a'-a|={np.abs(ah - al).max():.1e}")
    A = rng.standard_normal((300, 200)) @ np.diag(0.8 ** np.arange(200)) @ np.linalg.qr(rng.standard_normal((200, 200)))[0]
    sf, sr = np.linalg.svd(A, compute_uv=False)[:8], randomized_svals(A, 8, rng)
    Wb = rng.standard_normal((120, 80))
    Wa = Wb + np.outer(rng.standard_normal(120), rng.standard_normal(80))
    t_full = delta_stats(Wa, Wb, rng)["top_share"]
    t_rand = delta_stats(Wa, Wb, rng, max_full=10)["top_share"]
    t_iso = delta_stats(Wb + rng.standard_normal(Wb.shape), Wb, rng)["top_share"]
    chk("svd_and_top_share", np.abs(sr - sf).max() / sf[0] < 1e-3 and t_full > 0.999 and t_rand > 0.999 and t_iso < 0.25,
        f"rsvd err {np.abs(sr - sf).max() / sf[0]:.1e}; rank-1 dW top-5% share {t_full:.4f}/{t_rand:.4f}; noise {t_iso:.3f}")
    Xq = l2n(rng.standard_normal((60, 16)))
    tr, te = np.arange(40), np.arange(40, 60)
    so = shift_stats(Xq, Xq + rng.standard_normal(16) * 0.3, tr, te, 1e-6)
    M = np.outer(rng.standard_normal(16), rng.standard_normal(16)) * 0.2
    sl = shift_stats(Xq, Xq @ (np.eye(16) + M), tr, te, 1e-6)
    chk("offset_vs_affine", abs(so["offset_explained"] - 1) < 1e-9 and sl["affine_explained"] > 0.999
        and sl["offset_explained"] < 0.5 and sl["A_minus_I_rank90"] == 1,
        f"pure offset {so['offset_explained']:.4f}; rank-1 map: affine {sl['affine_explained']:.4f}, "
        f"offset {sl['offset_explained']:.3f}, rank90 {sl['A_minus_I_rank90']}")
    Xu = l2n(rng.standard_normal((500, 8)))
    al0, un = align_uniform(Xu, np.arange(500), np.arange(500), rng)
    chk("align_uniform", al0 == 0.0 and -4.0 < un < -1.0, f"align(identical)={al0}, uniformity {un:.3f}")
    return res


def selftest_pooling(model: str) -> list[tuple[str, bool, str]]:
    """The token-level embedding path against llama-cpp-python's own, and last pooling against per-token states."""
    res = []
    texts = ["The river bank was muddy.", "hello world", "A longer sentence about the bank of the river and the money in it."]
    rm = Runner(model, 256, 1, 0, embedding=True, pooling="mean")
    ours = rm.embed_tokens([rm.tok(t) for t in texts])  # several sequences in one decode
    ref = [np.asarray(rm.llm.embed(t), dtype=np.float32).reshape(-1) for t in texts]
    single = [rm.embed_tokens([rm.tok(t)])[0] for t in texts]
    cm = min(float(l2n(a) @ l2n(b)) for a, b in zip(ours, ref))
    db = max(float(np.abs(a - b).max()) for a, b in zip(ours, single))
    res.append(("pool:mean_matches_llama_embed", cm > 0.99999, f"min cos {cm:.7f}"))
    res.append(("pool:batched_equals_single", db < 1e-4, f"max |diff| {db:.1e}"))
    rl = Runner(model, 256, 1, 0, embedding=True, pooling="last")
    rn = Runner(model, 256, 1, 0, embedding=True, pooling="none")
    plan = rl.eos_plan(open_gguf(model))
    toks = [rl.pooled_tokens(t, plan) for t in texts]
    last, none = rl.embed_tokens(toks), rn.embed_tokens(toks)
    cl = min(float(l2n(a) @ l2n(b[-1])) for a, b in zip(last, none))
    res.append(("pool:last_is_last_token_state", cl > 0.99999, f"min cos {cl:.7f}"))
    ends = all(t[-1] == plan["eos_token_id"] for t in toks)
    res.append(("pool:eos_plan", plan["action"] in ("appended", "tokenizer") and ends,
                f"{plan['action']} {plan['eos_token']!r} id {plan['eos_token_id']} (add_eos_token={plan['add_eos_token_meta']})"))
    return res


def cmd_selftest(args) -> None:
    out = ensure_dir(args.out)
    model = os.path.join(out, "tiny_random.gguf")
    build_tiny_model(args.vocab, model, args.seed)
    model_b = os.path.join(out, "tiny_random_b.gguf")
    build_tiny_model(args.vocab, model_b, args.seed + 1, chat_template=QWEN3_STYLE_TEMPLATE)
    log(f"selftest: built {model} and {model_b} (random weights; outputs are meaningless by design)")
    sentences = ["The cat is on the mat.", "Paris is the capital of France.", "The dog chased the cat.",
                 "Albert Einstein was born in 1879.", "She bought 3 apples and two pears.", "The teacher praised the student.",
                 "This is a big house near the river.", "The company hired the lawyer.", "It is not easy to begin a new job.",
                 "Tokyo is a large city in Japan.", "The quick brown fox jumps over the lazy dog.",
                 "Marie Curie won the Nobel Prize in 1903.", "He can speak four languages.",
                 "The river bank was muddy after the rain.", "Water boils at 100 degrees Celsius.",
                 "The children were happy to go home.", "John visited Berlin last summer.", "The manager called the worker."]
    with open(os.path.join(out, "sentences.txt"), "w") as f:
        f.write("\n".join(sentences) + "\n")
    with open(os.path.join(out, "queries.txt"), "w") as f:
        f.write("capital of France\nwho discovered radium\nhow hot does water boil\nwhat did the dog chase\n"
                "largest city in Japan\nwhen was Einstein born\nhow many languages can he speak\nwhy was the bank muddy\n")
    with open(os.path.join(out, "docs.txt"), "w") as f:
        f.write("Paris is the capital of France.\nThe dog chased the cat.\n")
    with open(os.path.join(out, "personas.txt"), "w") as f:
        f.write("a curious student\na retired engineer\n")
    with open(os.path.join(out, "ext_pairs.tsv"), "w") as f:
        f.write("a\tb\tkind\nThe cat sat.\tThe cat did not sit.\tnegation_c6\nParis is big.\tParis is large.\tparaphrase_c6\n")
    with open(os.path.join(out, "audit.tsv"), "w") as f:
        f.write("query\tcandidate\ncapital of France\tParis is the capital of France.\ncapital of France\tThe dog chased the cat.\n")
    with open(os.path.join(out, "gpairs.tsv"), "w") as f:
        f.write("".join(f"{i}\t{i + 1}\n" for i in range(0, 16, 2)))
    with open(os.path.join(out, "glabelled.tsv"), "w") as f:
        f.write("a\tb\tclass\n" + "".join(f"{i}\t{i + 1}\tnear\n{i}\t{(i + 9) % 18}\tfar\n" for i in range(0, 16, 2)))
    srng = np.random.default_rng(args.seed)
    syn = srng.standard_normal((300, 32)).astype(np.float32)
    np.save(os.path.join(out, "syn_a.npy"), syn)
    np.save(os.path.join(out, "syn_b.npy"), syn @ np.linalg.qr(srng.standard_normal((32, 32)))[0].astype(np.float32))
    with open(os.path.join(out, "syn_pairs.tsv"), "w") as f:
        f.write("".join(f"{i}\t{i + 1}\n" for i in range(0, 300, 2)))
    with open(os.path.join(out, "subjects.tsv"), "w") as f:
        f.write("France\tQ142\nAlbert Einstein\tQ937\n")
    with open(os.path.join(out, "gold.tsv"), "w") as f:
        f.write("France\tP36\tParis\nGermany\tP36\tBerlin\nItaly\tP36\tRome\nSpain\tP36\tMadrid\n")
    with open(os.path.join(out, "words.txt"), "w") as f:
        f.write("house\nriver\nhappy\n")
    with open(os.path.join(out, "seeds.txt"), "w") as f:
        f.write("Paris\n")
    with open(os.path.join(out, "corpus.txt"), "w") as f:
        f.write("The river bank was muddy.\nShe went to the bank to get money.\nThe bank raised rates.\n"
                "We sat on the bank of the river.\nThe bank closed early.\nFish swam near the bank.\n")
    with open(os.path.join(out, "senses.txt"), "w") as f:
        f.write("bank\n")
    with open(os.path.join(out, "labels.tsv"), "w") as f:
        f.write("France\tQ142\nParis\tQ90\n")
    py = sys.executable
    me = os.path.abspath(__file__)
    common = ["--model", model, "--no-hash"]
    steps = [
        ["info", *common, "--out", f"{out}/info"],
        ["vocab", *common, "--out", f"{out}/vocab"],
        ["embed", *common, "--out", f"{out}/embed", "--max-tokens", "2000", "--clusters", "16", "--bigram-tokens", "50"],
        ["neurons", *common, "--out", f"{out}/neurons", "--layers", "0", "--max-neurons", "16", "--read-tokens", "500"],
        ["heads", *common, "--out", f"{out}/heads", "--tokens", "300"],
        ["spectra", *common, "--out", f"{out}/spectra", "--pattern", "blk\\.0\\."],
        ["wdiff", *common, "--out", f"{out}/wdiff", "--base", model_b, "--max-full", "32"],
        ["geometry", "--out", f"{out}/geometry_syn", "--vectors", f"{out}/syn_a.npy", "--pairs", f"{out}/syn_pairs.tsv",
         "--batch-size", "32"],
        ["compare", "--out", f"{out}/compare_syn", "--a", f"{out}/syn_a.npy", "--b", f"{out}/syn_b.npy",
         "--pairs", f"{out}/syn_pairs.tsv"],
    ]
    et = f"{out}/embedtexts"

    def drop_part():  # keep the full run, delete one part, so the --resume step must rebuild exactly that part
        import shutil
        shutil.copy(f"{et}/embeddings.npy", f"{out}/embeddings_full.npy")
        os.remove(f"{et}/parts/part_{14:09d}.npy")
        return True, "removed the part at row 14"

    def check_resume():
        with open(f"{et}/provenance.json") as f:
            pv = json.load(f)
        same = np.array_equal(np.load(f"{et}/embeddings.npy"), np.load(f"{out}/embeddings_full.npy"))
        ok = same and pv["parts_resumed"] == pv["parts"] - 1
        return ok, f"identical after resume: {same}; parts resumed {pv['parts_resumed']}/{pv['parts']}; EOS {pv['embedding']['eos']['action']}"

    def check_compare_syn():
        with open(f"{out}/compare_syn/compare.json") as f:
            c = json.load(f)
        ok = (c["linear_cka"] > 0.999999 and c["procrustes"]["residual"] < 1e-6 and c["pair_spearman"] > 0.9999
              and c["knn10_jaccard"] > 0.999 and c["force_match"]["kl_at_teacher_tau"] < 1e-6)
        return ok, (f"X vs XR: CKA {c['linear_cka']:.7f}, procrustes residual {c['procrustes']['residual']:.1e}, "
                    f"spearman {c['pair_spearman']:.5f}, jaccard {c['knn10_jaccard']:.3f}, KL {c['force_match']['kl_at_teacher_tau']:.1e}")

    def check_think():  # model_b carries a Qwen3-style template: thinking off closes the block, on leaves it open
        class Stub:
            class llm:
                token_eos = token_bos = staticmethod(lambda: -1)
            piece = staticmethod(lambda t: "")
        pf = PromptFormat(Stub, open_gguf(model_b), "auto")
        off, on = pf.render("hi", False), pf.render("hi", True)
        with open(f"{out}/pairs_think/audit.jsonl") as f:
            thought = [json.loads(ln)["thinking"] for ln in f]
        ok = pf.can_think and off.endswith("<think>\n\n</think>\n\n") and on.endswith("assistant\n") and all(thought)
        return ok, f"mode {pf.mode}, no-think prompt closes <think>: {off.endswith('</think>' + chr(10) * 2)}, audit thought: {all(thought)}"

    steps.append(("check:compare_X_vs_XR", check_compare_syn))
    sent, emb_c = f"{out}/sentences.txt", common + ["--batch", "4"]
    dyn = [
        ["facts", *common, "--out", f"{out}/facts", "--subjects", f"{out}/subjects.tsv", "--relations", "P36,P30", "--samples", "2", "--inverse"],
        ["calibrate", *common, "--out", f"{out}/calibrate", "--gold", f"{out}/gold.tsv", "--samples", "1", "--min-items", "2"],
        ["lexical", *common, "--out", f"{out}/lexical", "--words", f"{out}/words.txt", "--samples", "1", "--kinds", "synonym,antonym,pos,polarity"],
        ["taxonomy", *common, "--out", f"{out}/taxonomy", "--words", f"{out}/words.txt", "--samples", "1", "--depth", "2"],
        ["commonsense", *common, "--out", f"{out}/commonsense", "--words", f"{out}/words.txt", "--samples", "0"],
        ["crawl", *common, "--out", f"{out}/crawl", "--seeds", f"{out}/seeds.txt", "--samples", "1", "--max-facts", "4", "--depth", "1"],
        ["paraphrase", *common, "--out", f"{out}/paraphrase", "--subjects", f"{out}/subjects.tsv", "--relations", "P36", "--n", "3", "--check", "1"],
        ["qa", "--out", f"{out}/qa", "--facts", f"{out}/facts/facts.jsonl", "--patterns", f"{out}/paraphrase/patterns.jsonl", "--min-confidence", "0", "--min-agreement", "0"],
        ["sample", *common, "--out", f"{out}/sample", "--n", "3", "--max-tokens", "24"],
        ["score", *common, "--out", f"{out}/score", f"{out}/corpus.txt", "--split"],
        ["vectors", *common, "--out", f"{out}/vectors", "--words", f"{out}/words.txt", "--senses", f"{out}/senses.txt", "--corpus", f"{out}/corpus.txt"],
        ["canon", "--input", f"{out}/facts/facts.jsonl", "--labels", f"{out}/labels.tsv", "--output", f"{out}/facts/facts_canon.jsonl"],
        ["vectors", *common, "--out", f"{out}/vectors_last", "--words", f"{out}/words.txt", "--pooling", "last"],
        ["embedtexts", *emb_c, "--out", et, "--input", sent, "--instruction", "Retrieve semantically similar text", "--chunk", "2"],
        ("embedtexts:drop_part", drop_part),
        ["embedtexts", *emb_c, "--out", et, "--input", sent, "--instruction", "Retrieve semantically similar text", "--chunk", "2", "--resume"],
        ("check:embedtexts_resume", check_resume),
        ["embedtexts", *emb_c, "--out", f"{out}/embedtexts_mean", "--input", sent, "--pooling", "mean", "--dims", "32", "--chunk", "100"],
        ["geometry", "--out", f"{out}/geometry", "--vectors", f"{et}/embeddings.npy", "--pairs", f"{out}/gpairs.tsv",
         "--labelled-pairs", f"{out}/glabelled.tsv", "--lengths", "index", "--batch-size", "4"],
        ["compare", "--out", f"{out}/compare", "--a", f"{et}/embeddings.npy", "--b", f"{out}/embedtexts_mean/embeddings.npy"],
        ["sensitivity", *emb_c, "--out", f"{out}/sensitivity", "--sentences", sent, "--pairs", f"{out}/ext_pairs.tsv", "--role-templates", "4"],
        ["occlusion", *emb_c, "--out", f"{out}/occlusion", "--sentences", sent],
        ["instruct", *emb_c, "--out", f"{out}/instruct", "--queries", f"{out}/queries.txt"],
        ["pairs", *common, "--out", f"{out}/pairs", "--docs", f"{out}/docs.txt", "--types", "factual,keyword",
         "--personas", f"{out}/personas.txt", "--embedder", model, "--audit", f"{out}/audit.tsv", "--think",
         "--think-tokens", "8", "--max-tokens", "12", "--neg-tokens", "16"],
        ["pairs", "--model", model_b, "--no-hash", "--out", f"{out}/pairs_think", "--docs", f"{out}/docs.txt", "--types",
         "factual", "--negatives", "negation", "--audit", f"{out}/audit.tsv", "--think", "--think-tokens", "8", "--max-tokens", "8"],
        ("check:chat_template_thinking", check_think),
    ]
    import importlib.util
    have_llama = importlib.util.find_spec("llama_cpp") is not None
    if have_llama:
        steps += dyn
    else:
        log("selftest: llama-cpp-python not installed; dynamic commands skipped")
    if args.only:
        want = set(args.only.split(","))
        steps = [s for s in steps if (s[0] if isinstance(s, tuple) else s[0]) in want or (isinstance(s, tuple) and "check" in want)]
    import subprocess
    results = []
    for st in steps:
        t0 = time.time()
        if isinstance(st, tuple):  # an in-process step: (label, fn) -> (ok, detail)
            try:
                ok, detail = st[1]()
            except Exception as e:
                ok, detail = False, f"{type(e).__name__}: {e}"
            results.append((st[0], ok, time.time() - t0, detail))
            continue
        p = subprocess.run([py, me, *st], capture_output=True, text=True)
        ok = p.returncode == 0
        label = (os.path.basename(st[st.index("--out") + 1]) if "--out" in st else st[0]) + (" --resume" if "--resume" in st else "")
        results.append((label, ok, time.time() - t0, ""))
        if not ok:
            print(p.stderr[-3000:], file=sys.stderr)
    t0 = time.time()
    checks = selftest_maths(args.seed)
    if have_llama and not args.only:
        try:
            checks += selftest_pooling(model)
        except Exception as e:
            checks.append(("pool:checks", False, f"{type(e).__name__}: {e}"))
    results += [(n, ok, (time.time() - t0) / max(len(checks), 1), d) for n, ok, d in checks]
    w = max(len(r[0]) for r in results)
    for name, ok, dt, detail in results:
        print(f"{name.ljust(w)}  {'ok ' if ok else 'FAIL'}  {dt:6.1f} s  {detail}".rstrip())
    if not all(r[1] for r in results):
        sys.exit(1)


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def base(name, help_, model=True):
        p = sub.add_parser(name, help=help_)
        if model:
            p.add_argument("--model", required=True, help="path to the .gguf file")
            p.add_argument("--no-hash", action="store_true", help="skip the SHA-256 of the model in provenance")
        p.add_argument("--out", required=name != "canon", help="output directory")
        p.add_argument("--seed", type=int, default=1234)
        return p

    def dyn(p):
        p.add_argument("--ctx", type=int, default=512, help="context size for probing")
        p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
        p.add_argument("--samples", type=int, default=4, help="sampled answers per template")
        p.add_argument("--temperature", type=float, default=0.8)
        p.add_argument("--max-tokens", type=int, default=12)
        p.add_argument("--max-candidates", type=int, default=6)
        p.add_argument("--verify-top", type=int, default=3)
        p.add_argument("--calibration", help="calibration.json from `calibrate`")
        p.add_argument("--resume", action="store_true", help="append to existing output, skipping done items")
        return p

    p = base("info", "metadata, tensors, licence")
    p.set_defaults(fn=cmd_info)
    p = base("vocab", "tokenizer mining")
    p.add_argument("--min-len", type=int, default=3)
    p.add_argument("--dictionary", help="one word per line; marks which candidates are real words")
    p.add_argument("--max-pieces", type=int, default=5000)
    p.set_defaults(fn=cmd_vocab)
    p = base("embed", "token-embedding mining")
    p.add_argument("--min-len", type=int, default=3)
    p.add_argument("--dictionary", help="one word per line; analyse only these words")
    p.add_argument("--max-tokens", type=int, default=20000)
    p.add_argument("--neighbours", type=int, default=10)
    p.add_argument("--clusters", type=int, default=256)
    p.add_argument("--kmeans-iters", type=int, default=25)
    p.add_argument("--pca", type=int, default=10)
    p.add_argument("--pairs", help="TSV: relation, a, b (default: built-in pairs)")
    p.add_argument("--bigram-tokens", type=int, default=500)
    p.set_defaults(fn=cmd_embed)
    p = base("neurons", "FFN neurons in vocabulary space")
    p.add_argument("--layers", help="e.g. 0-3,10")
    p.add_argument("--max-neurons", type=int, default=2048, help="per layer (0 = all)")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--read-tokens", type=int, default=20000)
    p.set_defaults(fn=cmd_neurons)
    p = base("heads", "attention OV copying circuits")
    p.add_argument("--layers")
    p.add_argument("--tokens", type=int, default=2000)
    p.set_defaults(fn=cmd_heads)
    p = base("spectra", "weight spectra")
    p.add_argument("--pattern", help="regex on tensor names")
    p.add_argument("--max-rows", type=int, default=8192)
    p.set_defaults(fn=cmd_spectra)

    p = dyn(base("facts", "relation probing"))
    p.add_argument("--subjects", required=True, help="TSV: subject [<TAB> id]")
    p.add_argument("--relations", help="comma list of property ids or labels")
    p.add_argument("--relations-file", help="JSON list of [pid, label, [templates], question, type]")
    p.add_argument("--inverse", action="store_true", help="score inverse relations where defined")
    p.set_defaults(fn=cmd_facts)
    p = dyn(base("crawl", "recursive knowledge-graph crawl"))
    p.add_argument("--seeds", required=True)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--max-facts", type=int, default=1000)
    p.add_argument("--max-entities", type=int, default=500)
    p.add_argument("--max-relations", type=int, default=12)
    p.add_argument("--list-tokens", type=int, default=120)
    p.set_defaults(fn=cmd_crawl)
    for name, fn, h in (("lexical", cmd_lexical, "lexical relations"), ("taxonomy", cmd_taxonomy, "is-a chains"),
                        ("commonsense", cmd_commonsense, "ConceptNet-style relations")):
        p = dyn(base(name, h))
        p.add_argument("--words", help="one word per line (default: top tokenizer words)")
        p.add_argument("--dictionary", help="filter tokenizer words to these (when --words is not given)")
        p.add_argument("--top-words", type=int, default=1000)
        p.add_argument("--keep", type=int, default=5)
        if name == "lexical":
            p.add_argument("--kinds", help="subset, e.g. synonym,antonym,pos")
        if name == "taxonomy":
            p.add_argument("--depth", type=int, default=4)
            p.add_argument("--min-p", type=float, default=0.3)
        p.set_defaults(fn=fn)
    p = dyn(base("paraphrase", "question-pattern paraphrases"))
    p.add_argument("--subjects", required=True)
    p.add_argument("--relations")
    p.add_argument("--relations-file")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--check", type=int, default=5)
    p.set_defaults(fn=cmd_paraphrase)
    p = base("qa", "question/answer pairs from mined facts", model=False)
    p.add_argument("--facts", required=True)
    p.add_argument("--patterns")
    p.add_argument("--min-confidence", type=float, default=0.8)
    p.add_argument("--min-agreement", type=float, default=0.8)
    p.set_defaults(fn=cmd_qa)
    p = dyn(base("sample", "synthetic corpus + text statistics"))
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--prompts")
    p.add_argument("--top-p", type=float, default=0.95)
    p.set_defaults(fn=cmd_sample, max_tokens=256, temperature=1.0)
    p = dyn(base("score", "bits per byte of a corpus"))
    p.add_argument("inputs", nargs="+")
    p.add_argument("--split", action="store_true", help="score paragraphs / <|doc|> blocks separately")
    p.add_argument("--field", default="text", help="JSONL field to score")
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--max-docs", type=int, default=100000)
    p.set_defaults(fn=cmd_score)
    p = dyn(base("vectors", "model-pooled word vectors and senses"))
    p.add_argument("--words")
    p.add_argument("--dictionary")
    p.add_argument("--top-words", type=int, default=5000)
    p.add_argument("--template", help="e.g. 'The word {w}'")
    p.add_argument("--dims", type=int, default=0)
    p.add_argument("--neighbours", type=int, default=10)
    p.add_argument("--senses", help="words to induce senses for")
    p.add_argument("--corpus", help="text with one sentence per line, for senses")
    p.add_argument("--max-contexts", type=int, default=200)
    p.add_argument("--pooling", choices=("mean", "last", "none"), default="mean",
                   help="mean (default, as before); last = last-token state, EOS checked/appended (Qwen3-Embedding); "
                        "none = per-token states averaged over the word's own tokens")
    p.add_argument("--eos", choices=("auto", "append", "never"), default="auto",
                   help="last pooling: auto appends EOS only when the tokenizer does not")
    p.set_defaults(fn=cmd_vectors)

    def emb(p, pooling="last"):
        p.add_argument("--pooling", choices=("mean", "last"), default=pooling)
        p.add_argument("--eos", choices=("auto", "append", "never"), default="auto",
                       help="last pooling: auto appends EOS (<|endoftext|> when the vocabulary has it) only when the "
                            "tokenizer does not; the choice is logged and in provenance")
        p.add_argument("--instruction", help="query-side instruction (documents get none)")
        p.add_argument("--template", default=DEFAULT_QUERY_TEMPLATE, help="query form, with {instruction} and {text}")
        p.add_argument("--dims", type=int, default=0, help="MRL prefix: truncate to N dims, then renormalise")
        p.add_argument("--batch", type=int, default=32, help="texts per embedding call")
        return p

    p = emb(dyn(base("embedtexts", "batched text embeddings (fp16 .npy + index TSV + timing)")))
    p.add_argument("--input", required=True, help="one item per line, or a TSV with a header (see --column)")
    p.add_argument("--column", help="TSV column (name or 0-based index) holding the text")
    p.add_argument("--id-column")
    p.add_argument("--side", choices=("query", "document"), default="query", help="the instruction applies to queries only")
    p.add_argument("--side-column", help="TSV column with query/document per row")
    p.add_argument("--chunk", type=int, default=1000, help="items per resumable part")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(fn=cmd_embedtexts)
    p = emb(dyn(base("sensitivity", "B10 minimal-pair sensitivity")))
    p.add_argument("--sentences", help="one sentence per line; minimal pairs are made from it by rule")
    p.add_argument("--pairs", help="external TSV: a, b, kind [, note] (e.g. a generated C6 suite)")
    p.add_argument("--kinds", help=f"subset of {','.join(EDITS)}")
    p.add_argument("--per-kind", type=int, default=200)
    p.add_argument("--typos", type=int, default=2, help="typos per sentence")
    p.add_argument("--role-templates", type=int, default=20, help="built-in SVO role-swap pairs")
    p.set_defaults(fn=cmd_sensitivity)
    p = emb(dyn(base("occlusion", "B15 word importance and B16 additivity")))
    p.add_argument("--sentences", required=True)
    p.add_argument("--max-sentences", type=int, default=2000)
    p.add_argument("--freq", help="TSV word, count for the SIF p(w) (default: counts in --sentences)")
    p.add_argument("--sif-a", type=float, default=1e-3)
    p.add_argument("--ridge", type=float, default=1e-2, help="per-word weight ridge, relative to the mean diagonal")
    p.add_argument("--ridge-map", type=float, default=0.1, help="linear-map ridge, relative to the mean eigenvalue")
    p.add_argument("--heldout", type=float, default=0.2)
    p.set_defaults(fn=cmd_occlusion)
    p = emb(dyn(base("instruct", "B8 instruction effects and B9 query/document asymmetry")))
    p.add_argument("--queries", required=True)
    p.add_argument("--instructions", help="one per line (default: five Qwen/MTEB-style instructions)")
    p.add_argument("--max-queries", type=int, default=2000)
    p.add_argument("--heldout", type=float, default=0.3)
    p.add_argument("--ridge", type=float, default=0.1, help="affine-map ridge, relative to the mean eigenvalue")
    p.set_defaults(fn=cmd_instruct)
    p = emb(dyn(base("pairs", "C1-C3 synthetic pairs, hard negatives and false-negative audit")))
    p.add_argument("--docs", help="one document per line")
    p.add_argument("--max-docs", type=int, default=0)
    p.add_argument("--types", default=",".join(QUERY_TYPES))
    p.add_argument("--length", default=",".join(QUERY_LENGTHS), help="query lengths to draw from")
    p.add_argument("--difficulty", default=",".join(QUERY_DIFFICULTY))
    p.add_argument("--personas", help="one persona per line; one is drawn per document")
    p.add_argument("--n", type=int, default=1, help="queries per document and type")
    p.add_argument("--negatives", default=",".join(NEG_TASKS), help="hard-negative kinds ('' for none)")
    p.add_argument("--neg-tokens", type=int, default=64)
    p.add_argument("--chat-template", choices=("auto", "on", "off"), default="auto",
                   help="use the GGUF's jinja chat template (thinking off); a no-op when the model has none")
    p.add_argument("--think", action="store_true", help="thinking on, capped, for audit judging (C3)")
    p.add_argument("--think-tokens", type=int, default=256)
    p.add_argument("--min-verify", type=float, default=0.0, help="keep positives with verify above this, negatives below minus it")
    p.add_argument("--embedder", help="embedding GGUF for the cosine filter (loaded after the generator is freed)")
    p.add_argument("--min-cos", type=float, default=0.7)
    p.add_argument("--neg-margin", type=float, default=0.1, help="a hard negative is within this of the positive's cosine")
    p.add_argument("--audit", help="TSV: query, candidate [, ...]: candidate false negatives to judge (C3)")
    p.set_defaults(fn=cmd_pairs, max_tokens=48, instruction=DEFAULT_INSTRUCTIONS[0])

    p = base("geometry", "B1, B3-B7, B13, B17 on one set of vectors (no model)", model=False)
    p.add_argument("--vectors", required=True, help=".npy (n, d)")
    p.add_argument("--index", help="index TSV with a header (default: <stem>_index.tsv or index.tsv beside it)")
    p.add_argument("--key", help="index column that pairs files name items by (default id/text/word, else row numbers)")
    p.add_argument("--max-items", type=int, default=5000, help="sample for the O(n^2) statistics")
    p.add_argument("--n-pairs", type=int, default=20000, help="random pairs for the floor")
    p.add_argument("--anchors", type=int, default=500, help="anchors for stratified pairs (MRL curves)")
    p.add_argument("--prefixes", help="MRL prefixes, e.g. 32,64,128,256 (default: powers of two)")
    p.add_argument("--twonn-items", type=int, default=5000)
    p.add_argument("--knn", type=int, default=10)
    p.add_argument("--knn-max", type=int, default=20000, help="items in the exported kNN graph")
    p.add_argument("--no-knn-graph", action="store_true")
    p.add_argument("--pairs", help="TSV anchor, positive: alignment, uniformity and the ML temperature")
    p.add_argument("--batch-size", type=int, default=128, help="in-batch negatives for the temperature fit")
    p.add_argument("--labelled-pairs", help="TSV a, b, class: per-class cosines and pairwise AUCs (B2)")
    p.add_argument("--lengths", help="one length per line in row order, or 'index[:column]' (default column n_tokens)")
    p.set_defaults(fn=cmd_geometry)
    p = base("compare", "two vector sets over the same items (section 7.1, no model)", model=False)
    p.add_argument("--a", required=True, help="teacher .npy")
    p.add_argument("--b", required=True, help="other .npy")
    p.add_argument("--index-a")
    p.add_argument("--index-b")
    p.add_argument("--key", help="align the sets on this index column (default: row order)")
    p.add_argument("--max-items", type=int, default=5000)
    p.add_argument("--anchors", type=int, default=500)
    p.add_argument("--procrustes-dim", type=int, default=256)
    p.add_argument("--tau", type=float, help="teacher temperature for the force match")
    p.add_argument("--tau-student", type=float, help="also report KL at this student temperature")
    p.add_argument("--pairs", help="TSV anchor, positive in --a's rows: fits the teacher tau when --tau is not given")
    p.set_defaults(fn=cmd_compare)
    p = base("wdiff", "A4 weight difference between two GGUFs")
    p.add_argument("--base", required=True, help="the parent model; dW = W_model - W_base")
    p.add_argument("--pattern", help="regex on tensor names (e.g. 'blk\\.' to skip the embedding matrices)")
    p.add_argument("--max-full", type=int, default=1024, help="exact SVD up to this min dimension, randomized above")
    p.add_argument("--top-frac", type=float, default=0.05)
    p.add_argument("--iters", type=int, default=4, help="power iterations for the randomized SVD")
    p.add_argument("--max-rows", type=int, default=32768, help="sample rows of taller matrices (0 = all)")
    p.set_defaults(fn=cmd_wdiff)
    p = dyn(base("calibrate", "calibration against gold facts"))
    p.add_argument("--gold", required=True, help="TSV: subject, property id, object [, alias|alias]")
    p.add_argument("--min-items", type=int, default=30)
    p.set_defaults(fn=cmd_calibrate)
    p = base("canon", "attach IDs from a label table", model=False)
    p.add_argument("--input", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fields", default="subject,object")
    p.set_defaults(fn=cmd_canon)
    p = sub.add_parser("selftest", help="tiny random model + every command")
    p.add_argument("--vocab", required=True, help="a vocab-only GGUF, e.g. llama.cpp models/ggml-vocab-llama-spm.gguf")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--only", help="comma list of step labels to run (development aid); maths checks always run")
    p.set_defaults(fn=cmd_selftest)

    args = ap.parse_args(argv)
    if args.cmd in ("qa", "canon") and not getattr(args, "out", None):
        args.out = "."
    args.fn(args)


if __name__ == "__main__":
    main()
