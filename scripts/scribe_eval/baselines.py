"""Baseline metrics for the SCRIBE meta-evaluation (paper plan §6).

Author: Odin Loch

Every baseline scores exactly the texts SCRIBE scores. Input is one JSONL file
of texts, one object per line:
    {"id", "set", "system", "text", "prompt"?, "ref"?, "meta"?}
Only "text" (the continuation) is scored, except ppl_cond and prompt_cos,
which also read "prompt".

Output: <results-root>/<name>/<family>.jsonl with one {"id", <metric>: value}
line per input id, in input order (null where a metric is undefined, e.g. an
empty text or a one-sentence text for coherence), plus <family>.time.json
with the model, its revision, the settings and the timing.

Subcommands (families)
  stats      distinct-1..4, rep-2/3/4, seq-rep-4, diversity, type-token ratio,
             gzip ratio, mean word length, word/byte/sentence counts. Pure Python.
  ppl        GPT-2 perplexity (--model gpt2 or gpt2-large), and the perplexity
             of the text given its prompt (ppl_cond) when a prompt exists
             (--no-cond skips it). --max-len 257 scores the first 256 text tokens.
  cola       RoBERTa CoLA classifier, P(acceptable) per sentence: mean and min.
  coherence  all-MiniLM-L6-v2 sentence cosines: adjacent, all pairs, prompt-text.
  detector   OpenAI GPT-2 output detector (RoBERTa base), P(real) per text.
  setlevel   per group of texts: self-BLEU, Zipf coefficient, MAUVE with
             GPT-2-large features (cached per text), optionally on K subsets.
  convert-gpt2out  the raw OpenAI GPT-2 output dataset files to the schema above.

Definitions
  Words are lower-cased runs of letters/digits with inner apostrophes
  (WORD_RE); punctuation is not a word. Sentences come from a simple regex
  split (split_sentences): newlines, and whitespace after . ! ? (with closing
  quotes or brackets). Abbreviations such as "Mr." are split; this is the same
  simple rule for every text and system.
  distinct-n = unique word n-grams / total word n-grams within the text, so at
  the text level rep-n = 1 - distinct-n, seq-rep-4 = rep-4 (Welleck et al.
  2020) and TTR = distinct-1; all are written because the literature names
  them separately. diversity = prod_{n=2..4} (1 - rep-n) (Su et al. 2022).

Runs are resumable: finished batches go to <family>.partial.jsonl and a rerun
skips the ids already there. Threads default to 2 (the machine is shared).
"""

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

DEFAULT_ROOT = os.environ.get("SCRIBE_BASELINES_ROOT", "run/baselines")
GPT2_MODELS = {"gpt2": "openai-community/gpt2", "gpt2-large": "openai-community/gpt2-large"}
COLA_MODEL = "textattack/roberta-base-CoLA"
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DETECTOR_MODEL = "openai-community/roberta-base-openai-detector"
MAUVE_MODEL = "openai-community/gpt2-large"  # mauve-text's featurize_model_name "gpt2-large"

WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")
SENT_END_RE = re.compile(r"([.!?]+[\"'”’)\]]*)\s+")
HAS_WORD_RE = re.compile(r"[^\W_]")


# ---------------------------------------------------------------- utilities

def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    # Write to a temporary file first so a crash never leaves half a file.
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def words_of(text):
    return WORD_RE.findall(text.lower())


def split_sentences(text):
    """Simple regex sentence split; keeps only pieces that contain a word character."""
    out = []
    for line in text.split("\n"):
        for piece in SENT_END_RE.sub(r"\1\n", line).split("\n"):
            piece = piece.strip()
            if piece and HAS_WORD_RE.search(piece):
                out.append(piece)
    return out


def finite(x):
    """Plain float for JSON, None for nan/inf/None."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def hf_revision(model_id):
    """Commit sha of the cached snapshot of a Hugging Face model (None if unknown)."""
    try:
        from huggingface_hub import constants
        ref = Path(constants.HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}" / "refs" / "main"
        return ref.read_text().strip()
    except Exception:
        return None


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def setup_torch(threads):
    import torch
    torch.set_num_threads(threads)
    return torch


def token_batches(lengths, max_tokens, max_batch):
    """Indices sorted by length (longest first), cut so batch * longest <= max_tokens."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batch, longest = [], 0
    for i in order:
        L = max(1, lengths[i])
        if batch and (len(batch) + 1 > max_batch or (len(batch) + 1) * max(longest, L) > max_tokens):
            yield batch
            batch, longest = [], 0
        batch.append(i)
        longest = max(longest, L)
    if batch:
        yield batch


def default_name(path):
    p = Path(path)
    stem = p.name[:-6] if p.name.endswith(".jsonl") else p.stem
    return p.parent.name if stem in ("texts", "data", "all") else stem


def load_rows(args):
    rows = read_jsonl(args.input)
    seen = set()
    for r in rows:
        if r["id"] in seen:
            sys.exit(f"duplicate id {r['id']!r} in {args.input}: ids must be unique")
        seen.add(r["id"])
        r["text"] = r.get("text") or ""
    if args.limit:
        rows = rows[: args.limit]
    return rows


def out_dir(args):
    d = Path(args.results_root) / (args.name or default_name(args.input))
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_family(args, family, rows, setup, meta):
    """Score rows with a resumable batch loop.

    setup() loads the model and returns score(batch_rows) -> list of result dicts
    (without "id"), plus a batching function batches(rows) -> list of row lists.
    """
    d = out_dir(args)
    final, part = d / f"{family}.jsonl", d / f"{family}.partial.jsonl"
    done = {}
    if part.exists() and not args.fresh:
        for r in read_jsonl(part):
            done[r["id"]] = r
    if final.exists() and not args.fresh and not done:
        prev = {r["id"]: r for r in read_jsonl(final)}
        if all(r["id"] in prev for r in rows):
            log(f"{family}: {final} already complete ({len(rows)} ids); use --fresh to redo")
            return
        done = prev
    todo = [r for r in rows if r["id"] not in done]
    log(f"{family}: {len(rows)} texts, {len(done)} already done, {len(todo)} to score -> {final}")
    t_load = time.time()
    score, batches = setup() if todo else (None, None)
    load_s = time.time() - t_load
    t0 = time.time()
    n_new, last_log = 0, t0
    with open(part, "a", encoding="utf-8") as f:
        for batch in (batches(todo) if todo else []):
            for r, res in zip(batch, score(batch)):
                out = {"id": r["id"], **res}
                done[r["id"]] = out
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            n_new += len(batch)
            if time.time() - last_log > 60:
                el = time.time() - t0
                log(f"{family}: {n_new}/{len(todo)} in {el:.0f}s ({n_new / el:.2f} texts/s)")
                last_log = time.time()
    secs = time.time() - t0
    write_jsonl(final, [done[r["id"]] for r in rows])
    part.unlink(missing_ok=True)
    timing = {
        "family": family, "input": str(args.input), "n": len(rows), "n_scored_this_run": n_new,
        "seconds": round(secs, 3), "load_seconds": round(load_s, 3),
        "texts_per_s": round(n_new / secs, 3) if n_new and secs > 0 else None,
        "threads": args.threads, "cpu": cpu_name(), "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        **meta,
    }
    tpath = d / f"{family}.time.json"
    if tpath.exists() and not args.fresh and n_new < len(rows):
        # Resumed run: keep the earlier timing next to this one.
        try:
            timing["previous_runs"] = [json.loads(tpath.read_text())]
        except ValueError:
            pass
    write_json(tpath, timing)
    log(f"{family}: wrote {final} ({len(rows)} lines); {n_new} scored in {secs:.1f}s"
        + (f" = {n_new / secs:.2f} texts/s" if n_new and secs > 0 else ""))


# ---------------------------------------------------------------- stats

def text_stats(text):
    w = words_of(text)
    b = text.encode("utf-8")
    out = {}
    for n in (1, 2, 3, 4):
        grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
        out[f"distinct_{n}"] = len(set(grams)) / len(grams) if grams else None
    for n in (2, 3, 4):
        d = out[f"distinct_{n}"]
        out[f"rep_{n}"] = None if d is None else 1.0 - d
    out["seq_rep_4"] = out["rep_4"]
    reps = [out[f"rep_{n}"] for n in (2, 3, 4)]
    out["diversity"] = None if any(x is None for x in reps) else float(np.prod([1 - x for x in reps]))
    out["ttr"] = len(set(w)) / len(w) if w else None
    # gzip ratio: compressed / raw bytes (level 9, mtime 0; includes the 18-byte gzip frame).
    out["gzip_ratio"] = len(gzip.compress(b, compresslevel=9, mtime=0)) / len(b) if b else None
    out["mean_word_len"] = sum(map(len, w)) / len(w) if w else None
    out["n_words"] = len(w)
    out["n_bytes"] = len(b)
    out["n_sents"] = len(split_sentences(text))
    return out


def cmd_stats(args):
    rows = load_rows(args)
    meta = {"word_re": WORD_RE.pattern, "sentence_split": SENT_END_RE.pattern + " and newlines"}

    def setup():
        return (lambda batch: [text_stats(r["text"]) for r in batch]), (lambda rs: [rs[i:i + 1000] for i in range(0, len(rs), 1000)])
    run_family(args, "stats", rows, setup, meta)


# ---------------------------------------------------------------- ppl

def cmd_ppl(args):
    torch = setup_torch(args.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = GPT2_MODELS.get(args.model, args.model)
    tag = model_id.split("/")[-1].replace("-", "_").replace(".", "_")
    rows = load_rows(args)
    tok = AutoTokenizer.from_pretrained(model_id)
    max_len = args.max_len
    eos = tok.eos_token_id

    def joiner(prompt, text):
        # Context tokens that join prompt and text. The text tokens are the same
        # as in the unconditional score, so ppl and ppl_cond cover the same tokens.
        if args.cond_joiner == "none":
            return prompt
        if args.cond_joiner == "newline":
            return prompt + "\n\n"
        # auto: add a paragraph break only if neither side has whitespace at the seam
        if prompt and text and not prompt[-1].isspace() and not text[0].isspace():
            return prompt + "\n\n"
        return prompt

    # Pre-tokenise: unconditional input = [eos] + text[:max_len-1];
    # conditional input = [eos] + prompt tail + text tokens (same text tokens).
    for r in rows:
        full = tok(r["text"]).input_ids if r["text"] else []
        ids = full[: max_len - 1]
        r["_ids"], r["_trunc"] = ids, len(full) > len(ids)
        r["_ctx"] = None
        if r.get("prompt") and ids and not args.no_cond:
            room = max_len - 1 - len(ids)
            ctx = tok(joiner(r["prompt"], r["text"])).input_ids
            r["_ctx"] = ctx[-room:] if room > 0 else []

    def seqs_of(r):
        out = [("u", [eos] + r["_ids"], 1)] if r["_ids"] else []
        if r["_ctx"] is not None:
            out.append(("c", [eos] + r["_ctx"] + r["_ids"], 1 + len(r["_ctx"])))
        return out

    def setup():
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()

        @torch.inference_mode()
        def nll_many(seqs):
            # seqs: list of (input ids, index of the first scored target). Right padding.
            L = max(len(s) for s, _ in seqs)
            x = torch.full((len(seqs), L), eos, dtype=torch.long)
            m = torch.zeros((len(seqs), L), dtype=torch.long)
            for i, (s, _) in enumerate(seqs):
                x[i, : len(s)] = torch.tensor(s)
                m[i, : len(s)] = 1
            logits = model(input_ids=x, attention_mask=m).logits.float()[:, :-1]
            # log p(next token) = its logit - logsumexp (3x cheaper than a transposed cross_entropy)
            lp = logits.gather(-1, x[:, 1:, None])[..., 0] - torch.logsumexp(logits, -1)
            out = []
            for i, (s, start) in enumerate(seqs):
                # target position t predicts token t+1; scored tokens are s[start:]
                tok_lp = lp[i, start - 1: len(s) - 1]
                out.append((float(-tok_lp.mean()), int(tok_lp.numel())))
            return out

        def score(batch):
            jobs = [(k, kind, s, st) for k, r in enumerate(batch) for kind, s, st in seqs_of(r)]
            res = {}
            for b in token_batches([len(j[2]) for j in jobs], args.max_batch_tokens, args.batch_size):
                for j, (nll, n) in zip(b, nll_many([(jobs[i][2], jobs[i][3]) for i in b])):
                    res[(jobs[j][0], jobs[j][1])] = (nll, n)
            out = []
            for k, r in enumerate(batch):
                u, c = res.get((k, "u")), res.get((k, "c"))
                out.append({
                    f"nll_{tag}": finite(u[0]) if u else None,
                    f"ppl_{tag}": finite(math.exp(u[0])) if u else None,
                    f"ntok_{tag}": u[1] if u else 0,
                    f"trunc_{tag}": r["_trunc"],
                    f"nll_cond_{tag}": finite(c[0]) if c else None,
                    f"ppl_cond_{tag}": finite(math.exp(c[0])) if c else None,
                    f"nctx_{tag}": len(r["_ctx"]) if c else None,
                })
            return out

        def batches(rs):
            # Outer chunks of ~64 texts, sorted by length so padding stays small.
            rs = sorted(rs, key=lambda r: -len(r["_ids"]))
            return [rs[i:i + 64] for i in range(0, len(rs), 64)]
        return score, batches

    meta = {"model": model_id, "revision": hf_revision(model_id), "dtype": "float32",
            "max_len": max_len, "bos": "<|endoftext|> prepended; every text token scored",
            "nll": "mean token NLL in nats; ppl = exp(nll)",
            "cond": "not computed (--no-cond)" if args.no_cond else
                    f"[eos] + prompt tail + text tokens, joiner={args.cond_joiner}; only text tokens scored",
            "text_tokens_max": max_len - 1,
            "max_batch_tokens": args.max_batch_tokens}
    run_family(args, f"ppl_{tag}", rows, setup, meta)


# ---------------------------------------------------------------- sentence classifiers

def _classifier(torch, model_id, max_len, threads):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, dtype=torch.float32).eval()

    @torch.inference_mode()
    def probs(texts, max_tokens, max_batch):
        enc = [tok(t, truncation=True, max_length=max_len).input_ids for t in texts]
        out = [None] * len(texts)
        for b in token_batches([len(e) for e in enc], max_tokens, max_batch):
            L = max(len(enc[i]) for i in b)
            x = torch.full((len(b), L), tok.pad_token_id, dtype=torch.long)
            m = torch.zeros((len(b), L), dtype=torch.long)
            for k, i in enumerate(b):
                x[k, : len(enc[i])] = torch.tensor(enc[i])
                m[k, : len(enc[i])] = 1
            p = torch.softmax(model(input_ids=x, attention_mask=m).logits.float(), -1)
            for k, i in enumerate(b):
                out[i] = p[k].tolist()
        return out, enc
    return model, tok, probs


def cmd_cola(args):
    torch = setup_torch(args.threads)
    rows = load_rows(args)
    state = {}

    def setup():
        model, tok, probs = _classifier(torch, COLA_MODEL, args.max_len, args.threads)
        # textattack CoLA models use GLUE labels: 0 unacceptable, 1 acceptable.
        # Check the order on obvious pairs before scoring anything.
        good = ["The cat sat on the mat.", "She has finished her homework.", "They are going to the market tomorrow."]
        bad = ["Cat the mat on sat the.", "She have finish homework her the.", "Market the going tomorrow are to they."]
        p, _ = probs(good + bad, 4096, 16)
        acc = args.acceptable_label
        pg, pb = [x[acc] for x in p[:3]], [x[acc] for x in p[3:]]
        log(f"cola label check: P(label {acc}) good={np.round(pg, 3).tolist()} bad={np.round(pb, 3).tolist()}")
        if min(pg) <= max(pb):
            sys.exit("cola: label check failed; pass --acceptable-label to fix the label order")
        state["check"] = {"good": pg, "bad": pb}

        def score(batch):
            sents = [split_sentences(r["text"]) for r in batch]
            flat = [s for ss in sents for s in ss]
            p, enc = probs(flat, args.max_batch_tokens, args.batch_size) if flat else ([], [])
            out, k = [], 0
            for ss in sents:
                v = [p[k + j][acc] for j in range(len(ss))]
                ntr = sum(len(enc[k + j]) >= args.max_len for j in range(len(ss)))
                k += len(ss)
                out.append({"cola_mean": finite(np.mean(v)) if v else None,
                            "cola_min": finite(np.min(v)) if v else None,
                            "cola_nsent": len(v), "cola_ntrunc": ntr})
            return out
        return score, (lambda rs: [rs[i:i + 32] for i in range(0, len(rs), 32)])

    meta = {"model": COLA_MODEL, "revision": hf_revision(COLA_MODEL), "dtype": "float32",
            "acceptable_label": args.acceptable_label, "max_len_per_sentence": args.max_len,
            "sentence_split": SENT_END_RE.pattern + " and newlines", "label_check": state}
    run_family(args, "cola", rows, setup, meta)


def cmd_detector(args):
    torch = setup_torch(args.threads)
    rows = load_rows(args)
    meta = {"model": DETECTOR_MODEL, "revision": hf_revision(DETECTOR_MODEL), "dtype": "float32",
            "max_len": args.max_len, "output": "P(real) = P(human-written)"}

    def setup():
        model, tok, probs = _classifier(torch, DETECTOR_MODEL, args.max_len, args.threads)
        labels = {v.lower(): int(k) for k, v in model.config.id2label.items()}
        if "real" not in labels:
            sys.exit(f"detector: unexpected labels {model.config.id2label}")
        real = labels["real"]
        meta["id2label"] = model.config.id2label

        def score(batch):
            texts = [r["text"] for r in batch]
            idx = [i for i, t in enumerate(texts) if t.strip()]
            p, enc = probs([texts[i] for i in idx], args.max_batch_tokens, args.batch_size) if idx else ([], [])
            out = [{"det_p_human": None, "det_ntok": 0, "det_trunc": False} for _ in batch]
            for k, i in enumerate(idx):
                out[i] = {"det_p_human": finite(p[k][real]), "det_ntok": len(enc[k]),
                          "det_trunc": len(enc[k]) >= args.max_len}
            return out

        def batches(rs):
            rs = sorted(rs, key=lambda r: -len(r["text"]))
            return [rs[i:i + 32] for i in range(0, len(rs), 32)]
        return score, batches
    run_family(args, "detector", rows, setup, meta)


# ---------------------------------------------------------------- coherence

def cmd_coherence(args):
    torch = setup_torch(args.threads)
    rows = load_rows(args)

    def setup():
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(SBERT_MODEL, device="cpu")

        def enc(texts):
            if not texts:
                return np.zeros((0, 384), dtype=np.float32)
            return model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False)

        def score(batch):
            sents = [split_sentences(r["text"]) for r in batch]
            E = enc([s for ss in sents for s in ss])
            prompts = [r.get("prompt") or "" for r in batch]
            pidx = [i for i, p in enumerate(prompts) if p.strip()]
            P = enc([prompts[i] for i in pidx])
            pmap = {i: P[k] for k, i in enumerate(pidx)}
            out, k = [], 0
            for i, ss in enumerate(sents):
                e = E[k:k + len(ss)]
                k += len(ss)
                res = {"coh_adjacent": None, "coh_allpairs": None, "coh_nsent": len(ss), "prompt_cos": None}
                if len(ss) >= 2:
                    S = e @ e.T
                    res["coh_adjacent"] = finite(np.mean(np.diag(S, 1)))
                    res["coh_allpairs"] = finite(S[np.triu_indices(len(ss), 1)].mean())
                if len(ss) >= 1 and i in pmap:
                    # text vector = normalised mean of its sentence vectors (covers the whole text)
                    c = e.mean(0)
                    c = c / (np.linalg.norm(c) + 1e-12)
                    res["prompt_cos"] = finite(float(c @ pmap[i]))
                out.append(res)
            return out
        return score, (lambda rs: [rs[i:i + 64] for i in range(0, len(rs), 64)])

    meta = {"model": SBERT_MODEL, "revision": hf_revision(SBERT_MODEL),
            "pooling": "mean, L2-normalised (sentence-transformers default for this model)",
            "max_seq_length": 256, "sentence_split": SENT_END_RE.pattern + " and newlines",
            "prompt_cos": "cosine of the prompt vector and the normalised mean of the text's sentence vectors"}
    run_family(args, "coherence", rows, setup, meta)


# ---------------------------------------------------------------- setlevel

def self_bleu(texts, n_sample, seed):
    """Mean sacrebleu sentence BLEU (0-1) of each sampled text against all other sampled texts."""
    from sacrebleu.metrics import BLEU
    texts = [t for t in texts if t.strip()]
    if len(texts) < 2:
        return None, len(texts)
    rng = random.Random(seed)
    sample = rng.sample(texts, min(n_sample, len(texts)))
    bleu = BLEU(effective_order=True)
    scores = [bleu.sentence_score(h, sample[:i] + sample[i + 1:]).score / 100.0 for i, h in enumerate(sample)]
    return float(np.mean(scores)), len(sample)


def zipf_coeff(texts, top=1000):
    """Zipf s from least squares log f = a - s log r over the top words (Holtzman et al. 2020)."""
    c = Counter(w for t in texts for w in words_of(t))
    f = np.array(sorted(c.values(), reverse=True)[:top], dtype=np.float64)
    if len(f) < 10:
        return None, None
    x, y = np.log(np.arange(1, len(f) + 1)), np.log(f)
    slope, icept = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (slope * x + icept)) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(-slope), float(r2)


class FeatureCache:
    """GPT-2-large features (last hidden state at the last of the first L tokens, as
    mauve-text computes them), cached per (id, text hash) in append-only .npy shards."""

    def __init__(self, cache_dir, model_id, max_len):
        self.dir = Path(cache_dir) / f"{model_id.split('/')[-1]}_L{max_len}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.model_id, self.max_len = model_id, max_len
        self.feats = {}
        for kf in sorted(self.dir.glob("keys_*.json")):
            ff = kf.with_name(kf.name.replace("keys_", "feats_").replace(".json", ".npy"))
            if ff.exists():
                keys, arr = json.loads(kf.read_text()), np.load(ff)
                self.feats.update(zip(keys, arr))
        self.model = None
        self.seconds, self.n_new, self.tokens = 0.0, 0, 0

    @staticmethod
    def key(row, ids):
        return f"{row['id']}#{hashlib.sha1(json.dumps(ids).encode()).hexdigest()[:16]}"

    def get(self, rows, ids_list, threads, max_tokens=4096, max_batch=16, shard_size=256):
        keys = [self.key(r, ids) for r, ids in zip(rows, ids_list)]
        todo = [i for i, k in enumerate(keys) if k not in self.feats and ids_list[i]]
        if todo:
            torch = setup_torch(threads)
            if self.model is None:
                from transformers import AutoModel
                log(f"mauve features: loading {self.model_id}")
                self.model = AutoModel.from_pretrained(self.model_id, dtype=torch.float32).eval()
            log(f"mauve features: {len(todo)} texts to featurize ({len(keys) - len(todo)} cached)")
            t0, last = time.time(), time.time()
            for s in range(0, len(todo), shard_size):
                chunk = todo[s:s + shard_size]
                new_keys, new_feats = [], []
                for b in token_batches([len(ids_list[i]) for i in chunk], max_tokens, max_batch):
                    b = [chunk[j] for j in b]
                    L = max(len(ids_list[i]) for i in b)
                    x = torch.zeros((len(b), L), dtype=torch.long)
                    m = torch.zeros((len(b), L), dtype=torch.long)
                    for k, i in enumerate(b):
                        x[k, : len(ids_list[i])] = torch.tensor(ids_list[i])
                        m[k, : len(ids_list[i])] = 1
                    with torch.inference_mode():
                        # last_hidden_state = hidden_states[-1] (after the final layer norm)
                        h = self.model(input_ids=x, attention_mask=m).last_hidden_state
                    for k, i in enumerate(b):
                        new_keys.append(keys[i])
                        new_feats.append(h[k, len(ids_list[i]) - 1].float().numpy())
                        self.tokens += len(ids_list[i])
                arr = np.stack(new_feats).astype(np.float32)
                stamp = f"{time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}_{s:06d}"
                np.save(self.dir / f"feats_{stamp}.npy", arr)
                (self.dir / f"keys_{stamp}.json").write_text(json.dumps(new_keys))
                self.feats.update(zip(new_keys, arr))
                self.n_new += len(new_keys)
                if time.time() - last > 60 or s + shard_size >= len(todo):
                    el = time.time() - t0
                    log(f"mauve features: {s + len(chunk)}/{len(todo)} in {el:.0f}s ({(s + len(chunk)) / el:.2f} texts/s)")
                    last = time.time()
            self.seconds += time.time() - t0
        return keys


def cmd_setlevel(args):
    rows = load_rows(args)
    gf = args.group_field
    if args.ref_file:
        ref_rows = read_jsonl(args.ref_file)
        for r in ref_rows:
            r[gf] = "__ref__"
            r["text"] = r.get("text") or ""
        ref_name = "__ref__"
        rows = rows + ref_rows
    else:
        ref_name = args.ref_system
    groups = {}
    for r in rows:
        groups.setdefault(str(r.get(gf)), []).append(r)
    if ref_name not in groups:
        sys.exit(f"setlevel: reference group {ref_name!r} not found in {sorted(groups)}")
    log(f"setlevel: groups { {g: len(v) for g, v in groups.items()} }, reference {ref_name!r}")

    # Truncate every text to its first --truncate-tokens GPT-2 tokens (as the MAUVE
    # paper does); self-BLEU and Zipf read the decoded truncated text, MAUVE the ids.
    from transformers import AutoTokenizer
    feat_model = GPT2_MODELS.get(args.feat_model, args.feat_model)
    tok = AutoTokenizer.from_pretrained(feat_model)
    L = args.truncate_tokens or 1024  # 0 means only GPT-2's own 1024-token limit
    for r in rows:
        ids = tok(r["text"]).input_ids if r["text"] else []
        r["_ids"] = ids[:L]
        r["_t"] = tok.decode(r["_ids"]) if (args.truncate_tokens and len(ids) > L) else r["text"]

    cache = FeatureCache(args.feat_cache or (Path(args.results_root) / "_mauve_cache"), feat_model, L)
    t0 = time.time()
    fkeys = {}
    if not args.no_mauve:
        allrows = [r for r in rows if r["_ids"]]
        for r, k in zip(allrows, cache.get(allrows, [r["_ids"] for r in allrows], args.threads,
                                           args.max_batch_tokens, args.batch_size)):
            fkeys[id(r)] = k
    feat_s = time.time() - t0

    import mauve as mauve_lib
    # k-means (faiss, OpenMP) and PCA (BLAS) would otherwise use every core
    try:
        import faiss
        faiss.omp_set_num_threads(args.threads)
    except Exception:
        pass
    from threadpoolctl import threadpool_limits
    threadpool_limits(args.threads)

    def feats(rs):
        return np.stack([cache.feats[fkeys[id(r)]] for r in rs if id(r) in fkeys])

    def mauve_of(p_rows, q_rows):
        if args.no_mauve or len(p_rows) < 2 or len(q_rows) < 2:
            return {}
        o = mauve_lib.compute_mauve(p_features=feats(p_rows), q_features=feats(q_rows), device_id=-1,
                                    max_text_length=L, featurize_model_name="gpt2-large",
                                    verbose=False)
        return {"mauve": finite(o.mauve), "frontier_integral": finite(o.frontier_integral),
                "num_buckets": int(o.num_buckets)}

    def set_metrics(g, subset, rs, ref_rs, seed):
        t = time.time()
        sb, sb_n = self_bleu([r["_t"] for r in rs], args.bleu_sample, seed)
        t_sb = time.time() - t
        z, z_r2 = zipf_coeff([r["_t"] for r in rs], args.zipf_top)
        t = time.time()
        m = mauve_of(ref_rs, rs) if ref_rs is not None else {}
        t_m = time.time() - t
        return {"group": g, "subset": subset, "n": len(rs), "n_ref": len(ref_rs) if ref_rs is not None else None,
                "self_bleu": finite(sb), "self_bleu_n": sb_n, "zipf": z, "zipf_r2": z_r2,
                "mauve": m.get("mauve"), "frontier_integral": m.get("frontier_integral"),
                "num_buckets": m.get("num_buckets"), "seconds_self_bleu": round(t_sb, 2),
                "seconds_mauve": round(t_m, 2)}

    out = []
    t1 = time.time()
    ref = groups[ref_name]
    K = args.subsets
    if K <= 1:
        for g, rs in groups.items():
            log(f"setlevel: group {g} (n={len(rs)})")
            out.append(set_metrics(g, None, rs, None if g == ref_name else ref, args.seed))
            log(f"  {out[-1]}")
    else:
        # K random disjoint subsets of equal size per group (fixed seed). Subset k of a
        # group is compared with subset k of the reference; the reference's own subset k
        # is compared with its subset k+1 (a human-vs-human ceiling).
        def split(rs, salt):
            rng = random.Random(f"{args.seed}:{salt}")
            idx = list(range(len(rs)))
            rng.shuffle(idx)
            m = len(rs) // K
            return [[rs[i] for i in idx[k * m:(k + 1) * m]] for k in range(K)]
        ref_sub = split(ref, ref_name)
        for g, rs in groups.items():
            subs = ref_sub if g == ref_name else split(rs, g)
            for k, sub in enumerate(subs):
                rk = ref_sub[(k + 1) % K] if g == ref_name else ref_sub[k]
                out.append(set_metrics(g, k, sub, rk, args.seed + k))
                log(f"  {g}[{k}] n={len(sub)} self_bleu={out[-1]['self_bleu']} zipf={out[-1]['zipf']} mauve={out[-1]['mauve']}")
    set_s = time.time() - t1

    d = out_dir(args)
    fam = "setlevel" if K <= 1 else f"setlevel_K{K}"
    write_jsonl(d / f"{fam}.jsonl", out)
    write_json(d / f"{fam}.time.json", {
        "family": fam, "input": str(args.input), "ref": args.ref_file or ref_name, "group_field": gf,
        "n_texts": len(rows), "n_groups": len(groups), "subsets": K,
        "truncate_tokens": L, "featurize_model": None if args.no_mauve else feat_model,
        "featurize_revision": None if args.no_mauve else hf_revision(feat_model),
        "mauve_settings": "mauve-text compute_mauve defaults (num_buckets auto = n/10, PCA 0.9 var, "
                          "5 k-means redos, 500 iters, scaling 5, seed 25), features passed in",
        "mauve_text_version": getattr(mauve_lib, "__version__", None) or _pkg_version("mauve-text"),
        "self_bleu": f"sacrebleu {_pkg_version('sacrebleu')} sentence BLEU (effective order, exp smoothing) "
                     f"vs the other sampled texts, sample {args.bleu_sample}, seed {args.seed}",
        "zipf": f"top {args.zipf_top} words, least squares log f on log rank, reported as -slope",
        "feature_seconds": round(feat_s, 2), "features_new": cache.n_new, "feature_tokens_new": cache.tokens,
        "features_per_s": round(cache.n_new / cache.seconds, 3) if cache.n_new else None,
        "feature_tokens_per_s": round(cache.tokens / cache.seconds, 1) if cache.n_new else None,
        "set_metric_seconds": round(set_s, 2), "threads": args.threads, "cpu": cpu_name(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"setlevel: wrote {d / (fam + '.jsonl')} ({len(out)} lines)")


def _pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


# ---------------------------------------------------------------- convert

def cmd_convert_gpt2out(args):
    """Raw GPT-2 output dataset files ({id, ended, length, text}) to the shared schema."""
    src = Path(args.src)
    files = args.files or ["webtext.test", "small-117M.test", "small-117M-k40.test", "medium-345M.test",
                           "medium-345M-k40.test", "large-762M.test", "large-762M-k40.test",
                           "xl-1542M.test", "xl-1542M-k40.test"]
    out = []
    for name in files:
        system, split = name.rsplit(".", 1)
        rows = read_jsonl(src / f"{name}.jsonl")[: args.n]
        for r in rows:
            out.append({"id": f"gpt2out/{system}/{r['id']}", "set": f"gpt2out.{split}", "system": system,
                        "text": r["text"], "meta": {"orig_id": r["id"], "ended": r.get("ended"),
                                                    "length": r.get("length"), "file": f"{name}.jsonl"}})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, out)
    log(f"convert-gpt2out: {len(out)} texts from {len(files)} files -> {args.out}")


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, threads=2, batch=16, tokens=4096, max_len=None):
        p.add_argument("input", help="input JSONL")
        p.add_argument("--name", help="results subdirectory (default: from the input path)")
        p.add_argument("--results-root", default=DEFAULT_ROOT, help="default $SCRIBE_BASELINES_ROOT or run/baselines")
        p.add_argument("--limit", type=int, default=0, help="score only the first N texts (tests)")
        p.add_argument("--threads", type=int, default=threads)
        p.add_argument("--batch-size", type=int, default=batch)
        p.add_argument("--max-batch-tokens", type=int, default=tokens)
        p.add_argument("--fresh", action="store_true", help="ignore earlier partial or final output")
        if max_len:
            p.add_argument("--max-len", type=int, default=max_len)
        return p

    p = common(sub.add_parser("stats", help="statistical per-text metrics"))
    p.set_defaults(func=cmd_stats)
    p = common(sub.add_parser("ppl", help="GPT-2 perplexity"), max_len=1024)
    p.add_argument("--model", default="gpt2", help="gpt2, gpt2-large or a Hugging Face id")
    p.add_argument("--cond-joiner", default="auto", choices=["auto", "none", "newline"])
    p.add_argument("--no-cond", action="store_true", help="skip ppl_cond even when a prompt exists")
    p.set_defaults(func=cmd_ppl)
    p = common(sub.add_parser("cola", help="RoBERTa CoLA acceptability"), batch=64, tokens=8192, max_len=128)
    p.add_argument("--acceptable-label", type=int, default=1)
    p.set_defaults(func=cmd_cola)
    p = common(sub.add_parser("coherence", help="MiniLM sentence-embedding coherence"), batch=64)
    p.set_defaults(func=cmd_coherence)
    p = common(sub.add_parser("detector", help="OpenAI GPT-2 output detector"), batch=16, tokens=8192, max_len=512)
    p.set_defaults(func=cmd_detector)
    p = common(sub.add_parser("setlevel", help="self-BLEU, Zipf, MAUVE per group"))
    p.add_argument("--group-field", default="system")
    p.add_argument("--ref-system", default="human")
    p.add_argument("--ref-file", help="JSONL of reference texts (instead of a reference group)")
    p.add_argument("--subsets", type=int, default=1, help="K disjoint random subsets per group")
    p.add_argument("--truncate-tokens", type=int, default=256, help="GPT-2 tokens per text (0 = 1024 cap off)")
    p.add_argument("--bleu-sample", type=int, default=200)
    p.add_argument("--zipf-top", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20261028)
    p.add_argument("--feat-cache", help="feature cache dir (default <results-root>/_mauve_cache)")
    p.add_argument("--no-mauve", action="store_true")
    p.add_argument("--feat-model", default="gpt2-large", help="MAUVE feature model (gpt2-large as published; others for tests)")
    p.set_defaults(func=cmd_setlevel)
    p = sub.add_parser("convert-gpt2out", help="convert the raw GPT-2 output dataset")
    p.add_argument("--src", required=True, help="directory with webtext.test.jsonl etc.")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=1000, help="first N texts per file")
    p.add_argument("--files", nargs="*", help="file stems (default: webtext.test and the 8 model files)")
    p.set_defaults(func=cmd_convert_gpt2out)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
