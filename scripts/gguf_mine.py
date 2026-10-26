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
  vectors     model-pooled word vectors and induced word senses
  calibrate   accuracy and calibration against gold facts -> ECS priors

Utility:
  canon       attach IDs to mined strings from a label table (e.g. Wikidata)
  selftest    build a tiny random GGUF from a vocab-only GGUF and run all of
              the above (plumbing test; a random model knows nothing)

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


def spearman(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
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
        if embedding:
            kw["embedding"] = True
            kw["pooling_type"] = {"mean": llama_cpp.LLAMA_POOLING_TYPE_MEAN,
                                  "none": llama_cpp.LLAMA_POOLING_TYPE_NONE}.get(pooling, llama_cpp.LLAMA_POOLING_TYPE_MEAN)
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

    def tok(self, text: str, bos: bool = True) -> list[int]:
        return self.llm.tokenize(text.encode("utf-8"), add_bos=bos, special=False)

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

    def next_logprobs(self, prompt: str) -> np.ndarray:
        toks = self.tok(prompt) or self._start()
        logits, toks = self._eval(toks)
        return log_softmax(logits[len(toks) - 1])

    def option_logprob(self, prompt: str, options: list[str]) -> np.ndarray:
        return np.array([sum(self.cont_logprobs(prompt, o)) for o in options], dtype=np.float64)

    def yes_no(self, prompt: str) -> float:
        lp = self.next_logprobs(prompt)
        ly = logsumexp(lp[self._yes]) if self._yes else -1e9
        ln = logsumexp(lp[self._no]) if self._no else -1e9
        return float(ly - ln)

    def generate(self, prompt: str, n: int, temperature: float, max_tokens: int, stop, seed: int, top_p: float = 0.95) -> list[str]:
        outs = []
        ptoks = self.tok(prompt) or self._start()
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
    emb = Runner(args.model, args.ctx, args.threads, args.seed, embedding=True, pooling="mean")
    V = []
    for i, w in enumerate(words):
        e = np.asarray(emb.llm.embed(w if not args.template else args.template.replace("{w}", w)), dtype=np.float32)
        V.append(e.reshape(-1, e.shape[-1]).mean(axis=0))
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
    write_provenance(out, args, reader)
    log(f"vectors: {len(words)} words, dim {V.shape[1]} -> {out}")


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

def build_tiny_model(vocab_gguf: str, out_path: str, seed: int, n_embd: int = 64, n_layer: int = 2,
                     n_head: int = 4, n_ff: int = 128, n_ctx: int = 512) -> None:
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
        if not k.startswith("tokenizer."):
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


def cmd_selftest(args) -> None:
    out = ensure_dir(args.out)
    model = os.path.join(out, "tiny_random.gguf")
    build_tiny_model(args.vocab, model, args.seed)
    log(f"selftest: built {model} (random weights; outputs are meaningless by design)")
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
    ]
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
    ]
    import importlib.util
    if importlib.util.find_spec("llama_cpp") is not None:
        steps += dyn
    else:
        log("selftest: llama-cpp-python not installed; dynamic commands skipped")
    import subprocess
    results = []
    for st in steps:
        t0 = time.time()
        p = subprocess.run([py, me, *st], capture_output=True, text=True)
        ok = p.returncode == 0
        results.append((st[0], ok, time.time() - t0))
        if not ok:
            print(p.stderr[-3000:], file=sys.stderr)
    w = max(len(r[0]) for r in results)
    for name, ok, dt in results:
        print(f"{name.ljust(w)}  {'ok ' if ok else 'FAIL'}  {dt:6.1f} s")
    if not all(ok for _, ok, _ in results):
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
    p.set_defaults(fn=cmd_vectors)
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
    p.set_defaults(fn=cmd_selftest)

    args = ap.parse_args(argv)
    if args.cmd in ("qa", "canon") and not getattr(args, "out", None):
        args.out = "."
    args.fn(args)


if __name__ == "__main__":
    main()
