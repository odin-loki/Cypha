"""Model-ladder generations for the SCRIBE meta-evaluation (paper plan §5.3).

Author: Odin Loch

A model ladder is a set of text generators whose quality order is known in
advance (more training steps, a bigger model, a higher n-gram order). Every
generator here continues the same Wikipedia prompts, so a metric can be
checked against the known order.

Subcommands
  hf       continuations from a Hugging Face causal LM (batched, left padded)
  ngram    continuations from word n-gram generators (interpolated Kneser-Ney)
  ladders  the jobs of §5.3 in order (JOBS below), resumable; one model load per
           run of jobs on the same checkpoint; each finished model is deleted
           from the HF cache; one line per finished file in <gen-dir>/PROGRESS.txt
  estimate time one batch per model and print the projected hours of the plan
  summary  count, mean characters, tokens/s and wall time of output files

Ladders: pythia_steps (Pythia-410M at steps 1k, 4k, 16k, 64k, 143k), pythia_size
(Pythia 70M, 160M, 410M, 1B, 1.4B), gpt2_decoding (GPT-2 large: greedy, nucleus
0.95, pure sampling at T = 0.7, 1.0, 1.3, top-k 40, beam 4), ngram (orders 1-5).
Pythia runs use nucleus 0.95; pythia_size/pythia-410m is a copy of
pythia_steps/pythia-410m-step143000 (the main revision).

Output: one JSONL line per text, written to <gen-dir>/<ladder>/<system>.<set>.jsonl
  {"id": "<ladder>/<system>/<prompt id>", "set", "system", "prompt",
   "text" (continuation only, raw), "ref" (human continuation),
   "meta": {"model", "revision", "decoding", "params", "seed",
            "n_new_tokens", "stop" (eos | length), "seconds", ...}}
A continuation stops at the model's end-of-text token or at max_new_tokens
(144); it can be empty when the model ends the document at once.

Prompts: JSONL lines {"id", "set", "prompt", "ref" or "text"} in files named
prompts_<set>.jsonl; <set> (dev, fresh) is taken from the file name.

Reproducibility: prompts are sorted by token length and cut into batches; the
batch layout depends only on the prompt file and the batch size, and each batch
is generated under its own seed, recorded in meta.seed. An interrupted run
resumes at batch granularity with the batch size of its partial file, so a
resumed file equals an uninterrupted one. Batched and single-prompt greedy
decoding agree except at near ties (float rounding of the padded batch).

Example (the run behind the paper's files):
  generate.py --data-root $SP/data ladders --sets fresh --limit 150 --batch-size 50 \\
      --batch-override gpt2-large=38 pythia-1b=38 pythia-1.4b=25
"""

import argparse
import gc
import json
import os
import re
import shutil
import sys
import time
import zlib
from pathlib import Path

import numpy as np

DEFAULT_SEED = 20261028


# ---------------------------------------------------------------- utilities

def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def read_jsonl(path, tolerant=False):
    """JSONL lines; with tolerant=True a cut-off line (a killed writer) is skipped."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if not tolerant:
                    raise
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_prompts(path, limit=None):
    """Prompt lines, keeping the order of the file; the human continuation is "ref" (or "text").
    The set name of the outputs comes from the file name prompts_<set>.jsonl (dev, fresh);
    the prompt line's own "set" is kept as meta.prompt_set."""
    rows = read_jsonl(path)
    if limit:
        rows = rows[:limit]
    m = re.fullmatch(r"prompts_(\w+)\.jsonl", Path(path).name)
    out = []
    for r in rows:
        out.append({"id": str(r["id"]), "set": m.group(1) if m else r.get("set", Path(path).stem),
                    "prompt_set": r.get("set"), "prompt_meta": r.get("meta"),
                    "prompt": r["prompt"], "ref": r.get("ref", r.get("text"))})
    sets = {r["set"] for r in out}
    if len(sets) != 1:
        raise SystemExit(f"{path}: expected one set, found {sorted(sets)}")
    return out


def prompt_meta(p):
    """The prompt's own set and meta, carried into each output's meta."""
    return {"prompt_set": p["prompt_set"], "prompt_meta": p["prompt_meta"]}


def out_path(gen_dir, ladder, system, set_name):
    return Path(gen_dir) / ladder / f"{system}.{set_name}.jsonl"


def finished(path, n):
    """True when the output file exists and holds at least n lines (a file made
    from more prompts covers a shorter run: its first n lines are the first n prompts)."""
    if not Path(path).exists():
        return False
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip()) >= n


def batch_seed(base_seed, set_name, batch_index):
    # One seed per batch; depends on the set so dev and fresh do not share draws.
    return (base_seed * 1_000_003 + zlib.crc32(f"{set_name}:{batch_index}".encode())) % (2**31 - 1)


def free_gb(path="/"):
    return shutil.disk_usage(path).free / 2**30


# ------------------------------------------------------- Hugging Face models

def set_threads(n):
    import torch
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already set in this process


def load_hf(model_id, revision=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.perf_counter()
    rev = revision or "main"
    tok = AutoTokenizer.from_pretrained(model_id, revision=rev)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=rev, dtype=torch.float32)
    model.eval()
    log(f"loaded {model_id}@{rev} in {time.perf_counter() - t0:.1f}s "
        f"({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters)")
    return model, tok


def decoding_config(decoding, max_new_tokens, tok, temperature=1.0, top_k=0, top_p=1.0,
                    num_beams=4, min_new_tokens=0):
    """GenerationConfig plus the params dict recorded in meta. Every knob is set
    explicitly so no library default (such as top_k=50) leaks in."""
    from transformers import GenerationConfig
    common = dict(max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens or None,
                  pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                  bos_token_id=tok.bos_token_id, repetition_penalty=1.0,
                  no_repeat_ngram_size=0, use_cache=True)
    if decoding == "greedy":
        params = {}
        gen_cfg = GenerationConfig(do_sample=False, num_beams=1, **common)
    elif decoding == "beam":
        params = {"num_beams": num_beams, "length_penalty": 1.0, "early_stopping": False}
        gen_cfg = GenerationConfig(do_sample=False, num_beams=num_beams, length_penalty=1.0,
                              early_stopping=False, **common)
    elif decoding == "sample":
        params = {"temperature": temperature, "top_k": top_k, "top_p": top_p}
        gen_cfg = GenerationConfig(do_sample=True, num_beams=1, temperature=temperature,
                              top_k=top_k, top_p=top_p, **common)
    else:
        raise ValueError(decoding)
    params.update(max_new_tokens=max_new_tokens, repetition_penalty=1.0)
    if min_new_tokens:
        params["min_new_tokens"] = min_new_tokens
    return gen_cfg, params


def make_batches(tok, prompts, batch_size):
    """Prompt indices sorted by token length (ties by file order), cut into batches."""
    lens = [len(tok(p["prompt"]).input_ids) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: (lens[i], i))
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


def generate_batch(model, tok, texts, gen_cfg, seed):
    """Generate for one batch; returns (continuations, n_new_tokens, stops, seconds)."""
    import torch
    enc = tok(texts, return_tensors="pt", padding=True)
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**enc, generation_config=gen_cfg)
    dt = time.perf_counter() - t0
    new = out[:, enc["input_ids"].shape[1]:]
    eos = tok.eos_token_id
    conts, ns, stops = [], [], []
    for row in new.tolist():
        # pad == eos, so the first eos marks the end of this continuation
        n = row.index(eos) if eos in row else len(row)
        conts.append(tok.decode(row[:n], skip_special_tokens=True))
        ns.append(n)
        stops.append("eos" if n < len(row) else "length")
    return conts, ns, stops, dt


def run_hf_system(model, tok, model_id, revision, prompts, ladder, system, gen_dir,
                  decoding, params_kw, max_new_tokens, seed, batch_size, threads,
                  min_new_tokens=0):
    """Generate one system on one prompt set, resuming from a .partial file."""
    import torch
    import transformers
    set_name = prompts[0]["set"]
    path = out_path(gen_dir, ladder, system, set_name)
    if finished(path, len(prompts)):
        log(f"skip {path} (done)")
        return path
    gen_cfg, params = decoding_config(decoding, max_new_tokens, tok, min_new_tokens=min_new_tokens,
                                 **params_kw)
    partial = path.with_name(path.name + ".partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if partial.exists():
        # resume: whole batches are written at once, so a batch is either all there or
        # (if the writer was killed mid-batch) regenerated with its own seed
        for r in read_jsonl(partial, tolerant=True):
            done[r["id"]] = r
        # the batch layout depends on the prompt list, so a partial file made from
        # another number of prompts cannot be resumed
        if any(r["meta"].get("prompts_in_run", len(prompts)) != len(prompts) for r in done.values()):
            log(f"discard {partial.name}: made from another number of prompts")
            done = {}
        # rewrite the partial file without a cut-off last line
        with open(partial, "w", encoding="utf-8") as f:
            for r in done.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # keep the batch layout the partial file was made with
        if done:
            metas = [r["meta"] for r in done.values()]
            old = max(m.get("batch_size_nominal", m["batch_size"]) for m in metas)
            if old != batch_size:
                log(f"resume {path.name} with its batch size {old} (not {batch_size})")
                batch_size = old
    batches = make_batches(tok, prompts, batch_size)
    log(f"{ladder}/{system} [{set_name}] {len(prompts)} prompts, {len(batches)} batches, "
        f"{len(done)} already done")
    t_all = time.perf_counter()
    with open(partial, "a", encoding="utf-8") as f:
        for b, idx in enumerate(batches):
            ids = [f"{ladder}/{system}/{prompts[i]['id']}" for i in idx]
            if all(i in done for i in ids):
                continue
            s = batch_seed(seed, set_name, b)
            conts, ns, stops, dt = generate_batch(model, tok, [prompts[i]["prompt"] for i in idx], gen_cfg, s)
            tps = sum(ns) / dt if dt > 0 else 0.0
            for i, gid, text, n, stop in zip(idx, ids, conts, ns, stops):
                p = prompts[i]
                row = {"id": gid, "set": set_name, "system": system, "prompt": p["prompt"],
                       "text": text, "ref": p["ref"],
                       "meta": {"model": model_id, "revision": revision or "main",
                                "decoding": decoding, "params": params, "seed": s,
                                "n_new_tokens": n, "stop": stop,
                                "seconds": round(dt / len(idx), 4),
                                "batch_seconds": round(dt, 3), "batch_size": len(idx),
                                "batch_size_nominal": batch_size, "prompts_in_run": len(prompts),
                                "batch_index": b, "tokens_per_second": round(tps, 2),
                                "threads": threads, "dtype": "float32",
                                "torch": torch.__version__,
                                "transformers": transformers.__version__, **prompt_meta(p)}}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                done[gid] = row
            f.flush()
            log(f"  batch {b + 1}/{len(batches)}: {len(idx)} x {max(ns)} tok in {dt:.1f}s "
                f"({tps:.1f} tok/s)")
    # final file in the order of the prompt file
    rows = [done[f"{ladder}/{system}/{p['id']}"] for p in prompts]
    write_jsonl(path, rows)
    partial.unlink()
    log(f"wrote {path} ({len(rows)} lines, {time.perf_counter() - t_all:.0f}s this run)")
    return path


def cmd_hf(args):
    set_threads(args.threads)
    model, tok = load_hf(args.model, args.revision)
    system = args.system or args.model.split("/")[-1]
    params_kw = {}
    if args.decoding == "sample":
        params_kw = dict(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    elif args.decoding == "beam":
        params_kw = dict(num_beams=args.num_beams)
    for pf in args.prompts:
        prompts = load_prompts(pf, args.limit)
        if args.out and len(args.prompts) == 1:
            # explicit output file: write there via a one-off gen dir layout
            tmp = run_hf_system(model, tok, args.model, args.revision, prompts, args.ladder, system,
                                args.gen_dir, args.decoding, params_kw, args.max_new_tokens,
                                args.seed, args.batch_size, args.threads, args.min_new_tokens)
            if Path(args.out).resolve() != tmp.resolve():
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(tmp, args.out)
        else:
            run_hf_system(model, tok, args.model, args.revision, prompts, args.ladder, system,
                          args.gen_dir, args.decoding, params_kw, args.max_new_tokens,
                          args.seed, args.batch_size, args.threads, args.min_new_tokens)


# ------------------------------------------------------------- word n-grams

# Words (case kept), with internal hyphens, apostrophes and number separators
# ("well-known", "don't", "1,000", "3.5", "U.S"); every other non-space
# character is a token of its own.
TOKEN_RE = re.compile(r"\w+(?:[-'’.,]\w+)*|[^\w\s]")
NO_SPACE_BEFORE = set(".,;:!?)]}%'’”…")
NO_SPACE_AFTER = set("([{$‘“")


def word_tokens(text):
    return TOKEN_RE.findall(text)


def is_word(tok):
    return any(ch.isalnum() for ch in tok)


def detokenise(tokens, leading_space=True):
    """Join tokens with spaces, except before closing and after opening punctuation.
    Straight double quotes alternate between opening and closing."""
    out = []
    no_space_next = not leading_space
    quotes_open = False
    for t in tokens:
        space = not no_space_next
        no_space_next = False
        if t == '"':
            if quotes_open:
                space = False
            else:
                no_space_next = True
            quotes_open = not quotes_open
        elif t in NO_SPACE_BEFORE:
            space = False
        elif t in NO_SPACE_AFTER:
            no_space_next = True
        out.append((" " if space else "") + t)
    return "".join(out)


class NgramLevel:
    """Counts of the k-grams at one level of the model, sorted lexicographically.
    cols[j] is column j of the sorted unique k-grams; counts[i] the count of row i
    (raw counts at the top level, continuation counts below it)."""

    def __init__(self, grams, counts, discount):
        self.k = grams.shape[1]
        self.cols = [np.ascontiguousarray(grams[:, j]) for j in range(self.k)]
        self.counts = counts.astype(np.int64)
        self.D = float(discount)
        self.csum = np.concatenate([[0], np.cumsum(self.counts)])          # prefix of counts
        self.dsum = np.concatenate([[0.0], np.cumsum(self.counts - self.D)])  # prefix of (count - D)

    def find(self, ctx):
        """Row range [lo, hi) whose first k-1 columns equal ctx."""
        lo, hi = 0, len(self.counts)
        for j, w in enumerate(ctx):
            col = self.cols[j][lo:hi]
            w = np.int32(w)  # same dtype as the column, so numpy does not copy it to compare
            a = int(np.searchsorted(col, w, "left"))
            b = int(np.searchsorted(col, w, "right"))
            lo, hi = lo + a, lo + b
            if lo == hi:
                break
        return lo, hi


def kn_discount(counts):
    n1 = int((counts == 1).sum())
    n2 = int((counts == 2).sum())
    return n1 / (n1 + 2 * n2) if n1 + 2 * n2 > 0 else 0.5


def unique_rows(a):
    """Unique rows of a 2-D int32 array in lexicographic order, with counts."""
    return np.unique(a, axis=0, return_counts=True)


class KneserNey:
    """Interpolated Kneser-Ney word n-gram model of a given order, one discount per level.

    P(w | h) = max(c(h,w) - D, 0) / c(h) + D * N1+(h .) / c(h) * P(w | h')
    with raw counts c at the top level and continuation counts N1+(. h w) below.
    Sampling uses the mixture form: with probability (c(h) - D N1+(h .)) / c(h)
    draw w from the discounted counts of h, otherwise recurse on the shorter
    context h'. An unseen context backs off completely. The unigram level is
    sampled in proportion to its counts. Order 1 is the plain unigram model."""

    def __init__(self, seq, order):
        self.order = order
        self.levels = {}
        windows = {}
        for k in range(1, order + 1):
            w = np.lib.stride_tricks.sliding_window_view(seq, k)
            windows[k] = w[(w >= 0).all(axis=1)]  # drop windows that cross a document boundary
        top, top_counts = unique_rows(windows[order])
        self.levels[order] = NgramLevel(top, top_counts, kn_discount(top_counts) if order > 1 else 0.0)
        upper = top
        for k in range(order - 1, 0, -1):
            # continuation count of a k-gram: number of distinct words seen before it
            grams, cont = unique_rows(upper[:, 1:])
            self.levels[k] = NgramLevel(grams, cont, kn_discount(cont) if k > 1 else 0.0)
            # the (k)-grams that occur at least once as a suffix of a (k+1)-gram,
            # plus the doc-initial ones that never do, for the next level down
            all_k, _ = unique_rows(windows[k])
            upper = all_k
        self.discounts = {k: round(lv.D, 4) for k, lv in self.levels.items()}

    def sample(self, hist, rng):
        """Draw the next word id given the history (list of ids)."""
        k = min(self.order, len(hist) + 1)
        while k > 1:
            lv = self.levels[k]
            ctx = hist[len(hist) - (k - 1):]
            lo, hi = lv.find(ctx)
            if hi > lo:
                total = lv.csum[hi] - lv.csum[lo]
                u = rng.random() * total
                if u < total - lv.D * (hi - lo):
                    i = int(np.searchsorted(lv.dsum, lv.dsum[lo] + u, "right")) - 1
                    return int(lv.cols[k - 1][min(max(i, lo), hi - 1)])
            k -= 1
        lv = self.levels[1]
        u = rng.random() * lv.csum[-1]
        i = int(np.searchsorted(lv.csum, u, "right")) - 1
        return int(lv.cols[0][i])


def load_train_tokens(path):
    """Token ids of the training documents, -1 between documents, and the vocabulary."""
    vocab, words = {}, []
    ids = []
    n_docs = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            for t in word_tokens(doc["text"]):
                i = vocab.get(t)
                if i is None:
                    i = vocab[t] = len(words)
                    words.append(t)
                ids.append(i)
            ids.append(-1)
            n_docs += 1
    return np.array(ids, dtype=np.int32), vocab, words, n_docs


def cmd_ngram(args):
    import random
    t0 = time.perf_counter()
    seq, vocab, words, n_docs = load_train_tokens(args.train)
    n_tok = int((seq >= 0).sum())
    log(f"train: {n_docs} documents, {n_tok} tokens, vocabulary {len(words)} "
        f"({time.perf_counter() - t0:.1f}s)")
    prompt_sets = [load_prompts(pf, args.limit) for pf in args.prompts]
    for order in args.orders:
        system = f"ngram{order}"
        t1 = time.perf_counter()
        lm = KneserNey(seq, order)
        build = time.perf_counter() - t1
        log(f"order {order}: built in {build:.1f}s, discounts {lm.discounts}")
        for prompts in prompt_sets:
            set_name = prompts[0]["set"]
            path = out_path(args.gen_dir, args.ladder, system, set_name)
            rows = []
            t_set = time.perf_counter()
            for p in prompts:
                # one generator per prompt, seeded by (seed, system, set, prompt id), so a
                # text does not depend on the order of the prompts and the orders do not
                # share their random draws
                seed_str = f"{args.seed}/{system}/{set_name}/{p['id']}"
                rng = random.Random(seed_str)
                # context: the last n-1 prompt tokens; an unseen word breaks it, so only
                # the known tail after the last unseen word is kept
                tail = []
                for t in reversed(word_tokens(p["prompt"])):
                    if t not in vocab or len(tail) >= order - 1:
                        break
                    tail.append(vocab[t])
                hist = tail[::-1]
                n_ctx = len(hist)
                ts = time.perf_counter()
                out_ids, n_words = [], 0
                while n_words < args.n_words:
                    w = lm.sample(hist, rng)
                    out_ids.append(w)
                    n_words += is_word(words[w])
                    if order > 1:
                        hist = (hist + [w])[-(order - 1):]
                text = detokenise([words[i] for i in out_ids], leading_space=True)
                dt = time.perf_counter() - ts
                rows.append({"id": f"{args.ladder}/{system}/{p['id']}", "set": set_name,
                             "system": system, "prompt": p["prompt"], "text": text, "ref": p["ref"],
                             "meta": {"model": f"kn-word-{order}gram", "revision": Path(args.train).name,
                                      "decoding": "sample",
                                      "params": {"order": order, "n_words": args.n_words,
                                                 "discounts": lm.discounts, "train_tokens": n_tok,
                                                 "vocab": len(words), "context_tokens": n_ctx},
                                      "seed": seed_str, "n_new_tokens": len(out_ids),
                                      "seconds": round(dt, 5), **prompt_meta(p)}})
            write_jsonl(path, rows)
            log(f"wrote {path} ({len(rows)} lines, {time.perf_counter() - t_set:.1f}s)")
        del lm


# ------------------------------------------------------------- the ladders

def nucleus(p=0.95):
    return ("sample", {"temperature": 1.0, "top_k": 0, "top_p": p})


def pure(t):
    return ("sample", {"temperature": t, "top_k": 0, "top_p": 1.0})


PYTHIA_STEPS = ["step1000", "step4000", "step16000", "step64000", "step143000"]

# The jobs in the order of work: (model id, revision, ladder, system, decoding, params).
# Consecutive jobs on the same model share one load. Beam search is last (the
# costliest, and not in the plan's §5.3 text list), so it can be cut first.
JOBS = (
    [("EleutherAI/pythia-70m", "main", "pythia_size", "pythia-70m", *nucleus())]
    + [("EleutherAI/pythia-410m", s, "pythia_steps", f"pythia-410m-{s}", *nucleus())
       for s in PYTHIA_STEPS]
    + [("openai-community/gpt2-large", "main", "gpt2_decoding", name, dec, params)
       for name, dec, params in [
           ("greedy", "greedy", {}),
           ("nucleus-p0.95", *nucleus()),
           ("pure-t1.0", *pure(1.0)),
           ("topk40", "sample", {"temperature": 1.0, "top_k": 40, "top_p": 1.0}),
           ("pure-t0.7", *pure(0.7)),
           ("pure-t1.3", *pure(1.3))]]
    + [("EleutherAI/pythia-160m", "main", "pythia_size", "pythia-160m", *nucleus()),
       ("EleutherAI/pythia-1b", "main", "pythia_size", "pythia-1b", *nucleus()),
       ("EleutherAI/pythia-1.4b", "main", "pythia_size", "pythia-1.4b", *nucleus()),
       ("openai-community/gpt2-large", "main", "gpt2_decoding", "beam4", "beam", {"num_beams": 4})]
)
# pythia_size/pythia-410m is pythia_steps/pythia-410m-step143000 (main = step143000)
ALIASES = [(("pythia_steps", "pythia-410m-step143000"), ("pythia_size", "pythia-410m"))]
# Shared with other workstreams (baselines); never deleted from the HF cache here.
SHARED = {"openai-community/gpt2-large", "openai-community/gpt2"}


def copy_alias(gen_dir, src, dst, set_name):
    """Copy a system's file under another (ladder, system) name, rewriting id and system."""
    sp = out_path(gen_dir, src[0], src[1], set_name)
    dp = out_path(gen_dir, dst[0], dst[1], set_name)
    rows = read_jsonl(sp)
    for r in rows:
        pid = r["id"].split("/", 2)[2]
        r["id"] = f"{dst[0]}/{dst[1]}/{pid}"
        r["system"] = dst[1]
        r["meta"]["copied_from"] = f"{src[0]}/{src[1]}"
    write_jsonl(dp, rows)
    log(f"wrote {dp} (copy of {sp.name})")
    return dp


def delete_from_cache(model_id, revision):
    """Remove one revision of a model from the HF cache (blobs shared with another
    revision or repo are kept by huggingface_hub)."""
    from huggingface_hub import scan_cache_dir
    info = scan_cache_dir()
    for repo in info.repos:
        if repo.repo_id != model_id:
            continue
        hashes = [r.commit_hash for r in repo.revisions if revision in r.refs]
        if hashes:
            before = free_gb()
            strategy = info.delete_revisions(*hashes)
            strategy.execute()
            log(f"cache: deleted {model_id}@{revision} (expected {strategy.expected_freed_size_str}; "
                f"free disk {before:.1f} -> {free_gb():.1f} GB)")


def note_progress(gen_dir, path):
    """Append one line about a finished file to <gen-dir>/PROGRESS.txt (once per file)."""
    prog = Path(gen_dir) / "PROGRESS.txt"
    name = f"{Path(path).parent.name}/{Path(path).name}"
    if prog.exists() and any(line.split()[1:2] == [name] for line in prog.read_text().splitlines()):
        return
    s = file_stats(path)
    with open(prog, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {name} n={s['n']} mean_chars={s['chars']:.0f} "
                f"mean_tokens={s['tokens']:.0f} tok/s={s['tok_s']:.1f} wall_s={s['wall_s']:.0f} "
                f"eos_stops={s['eos']} empty={s['empty']} seq_rep4={s['rep4']:.3f}\n")


def cmd_ladders(args):
    set_threads(args.threads)
    prompt_files = [Path(args.wiki_dir) / f"prompts_{s}.jsonl" for s in args.sets]
    full_sets = [load_prompts(pf) for pf in prompt_files]
    skip = set(args.skip or [])
    only = set(args.only or [])
    bs_over = dict(kv.split("=") for kv in (args.batch_override or []))
    for f in sorted(Path(args.gen_dir).glob("*/*.jsonl")):
        note_progress(args.gen_dir, f)  # files finished before this run
    if not args.no_ngram and (not only or "ngram" in only):
        ns = argparse.Namespace(train=str(Path(args.wiki_dir) / "gen_train.jsonl"),
                                prompts=[str(p) for p in prompt_files], orders=[1, 2, 3, 4, 5],
                                n_words=args.n_words, seed=args.seed, gen_dir=args.gen_dir,
                                ladder="ngram", limit=None)
        if all(finished(out_path(args.gen_dir, "ngram", f"ngram{o}", ps[0]["set"]), len(ps))
               for o in ns.orders for ps in full_sets):
            log("skip ngram ladder (done)")
        else:
            cmd_ngram(ns)
            for f in sorted((Path(args.gen_dir) / "ngram").glob("*.jsonl")):
                note_progress(args.gen_dir, f)
    jobs = [j for j in JOBS if j[3] not in skip and (not only or j[2] in only)]

    def sets_for(model_id):
        name = model_id.split("/")[-1]
        limit = args.limit_large if (args.limit_large and name in (args.large_models or [])) else args.limit
        return [ps[:limit] if limit else ps for ps in full_sets]

    def job_done(j):
        return all(finished(out_path(args.gen_dir, j[2], j[3], ps[0]["set"]), len(ps))
                   for ps in sets_for(j[0]))

    k = 0
    while k < len(jobs):
        # a group: consecutive jobs on the same (model, revision)
        model_id, rev = jobs[k][:2]
        group = []
        while k < len(jobs) and jobs[k][:2] == (model_id, rev):
            group.append(jobs[k])
            k += 1
        sets = sets_for(model_id)
        todo = [j for j in group if not job_done(j)]
        if todo:
            model, tok = load_hf(model_id, rev)
            name = model_id.split("/")[-1]
            for _, _, ladder, system, decoding, params_kw in todo:
                bs = int(bs_over.get(name, args.batch_size))
                if decoding == "beam":  # batch size counts rows: bs / num_beams prompts
                    bs = max(1, bs // params_kw["num_beams"])
                for ps in sets:
                    path = run_hf_system(model, tok, model_id, rev, ps, ladder, system, args.gen_dir,
                                         decoding, params_kw, args.max_new_tokens, args.seed, bs,
                                         args.threads)
                    note_progress(args.gen_dir, path)
            del model
            gc.collect()
        else:
            log(f"skip {model_id}@{rev} (done)")
        for src, dst in ALIASES:
            for ps in sets:
                sp = out_path(args.gen_dir, src[0], src[1], ps[0]["set"])
                dp = out_path(args.gen_dir, dst[0], dst[1], ps[0]["set"])
                if finished(sp, len(ps)) and not finished(dp, len(ps)):
                    note_progress(args.gen_dir, copy_alias(args.gen_dir, src, dst, ps[0]["set"]))
        # free the disk once every job of this model is done and no later job needs it
        # (also when a restart finds it done but still cached); shared models are kept
        later = any(j[:2] == (model_id, rev) for j in jobs[k:])
        all_done = all(job_done(j) for j in jobs if j[:2] == (model_id, rev))
        if model_id not in SHARED and all_done and not later and (
                not args.keep_cache or free_gb() < args.min_free_gb):
            delete_from_cache(model_id, rev)
    log("ladders: all done")


# ------------------------------------------------------- timing estimate

def cmd_estimate(args):
    """Time one median-length batch of real prompts per model (sampling, and beam
    search where the plan has it) and project the hours of the plan."""
    set_threads(args.threads)
    prompts = load_prompts(Path(args.wiki_dir) / f"prompts_{args.sets[0]}.jsonl")
    n_total = sum(len(load_prompts(Path(args.wiki_dir) / f"prompts_{s}.jsonl")) for s in args.sets)
    timed = {}
    for model_id, rev, *_ in JOBS:
        if model_id in timed or (args.models and model_id.split("/")[-1] not in args.models):
            continue
        kinds = sorted({"beam" if j[4] == "beam" else "step" for j in JOBS if j[0] == model_id})
        model, tok = load_hf(model_id, rev)
        batches = make_batches(tok, prompts, args.batch_size)
        mid = batches[len(batches) // 2]
        res = {}
        for kind in kinds:
            if kind == "beam":
                gen_cfg, _ = decoding_config("beam", args.max_new_tokens, tok, num_beams=4,
                                             min_new_tokens=args.max_new_tokens)
                idx = mid[:max(1, args.batch_size // 4)]
            else:
                gen_cfg, _ = decoding_config("sample", args.max_new_tokens, tok, top_p=0.95,
                                             min_new_tokens=args.max_new_tokens)
                idx = mid
            _, ns, _, dt = generate_batch(model, tok, [prompts[i]["prompt"] for i in idx], gen_cfg, 1)
            res[kind] = dt / len(idx)  # seconds per text at full length
            log(f"  {model_id} {kind}: {len(idx)} texts x {max(ns)} tok in {dt:.1f}s "
                f"= {res[kind]:.2f} s/text")
        timed[model_id] = res
        del model
    total = 0.0
    print("\nprojected hours (full length, all prompts):")
    for model_id, rev, ladder, system, decoding, _ in JOBS:
        if model_id in timed:
            h = timed[model_id]["beam" if decoding == "beam" else "step"] * n_total / 3600
            print(f"  {ladder}/{system}: {h:.2f} h")
            total += h
    print(f"  total: {total:.2f} h for {n_total} prompts per system")


# --------------------------------------------------------------- summary

def seq_rep4(text):
    """Share of repeated word 4-grams in a text (Welleck et al. 2020, seq-rep-4)."""
    w = text.split()
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def file_stats(path):
    """Count, mean characters, empty texts, eos stops, mean new tokens, tokens/s,
    wall seconds (sum of per-text shares of batch time) and mean seq-rep-4."""
    rows = read_jsonl(path)
    n = len(rows)
    ntok = sum(r["meta"]["n_new_tokens"] for r in rows)
    secs = sum(r["meta"]["seconds"] for r in rows)
    return {"n": n, "chars": float(np.mean([len(r["text"]) for r in rows])) if n else 0.0,
            "empty": sum(1 for r in rows if not r["text"].strip()),
            "eos": sum(1 for r in rows if r["meta"].get("stop") == "eos"),
            "tokens": ntok / n if n else 0.0, "tok_s": ntok / secs if secs else 0.0,
            "wall_s": secs, "rep4": float(np.mean([seq_rep4(r["text"]) for r in rows])) if n else 0.0}


def cmd_summary(args):
    files = sorted(Path(args.gen_dir).glob("*/*.jsonl"))
    print(f"{'file':52s} {'n':>4s} {'chars':>6s} {'empty':>5s} {'eos':>4s} {'tok':>5s} "
          f"{'tok/s':>7s} {'wall_s':>8s} {'rep4':>5s}")
    for f in files:
        s = file_stats(f)
        name = f"{f.parent.name}/{f.name}"
        print(f"{name:52s} {s['n']:4d} {s['chars']:6.0f} {s['empty']:5d} {s['eos']:4d} "
              f"{s['tokens']:5.0f} {s['tok_s']:7.1f} {s['wall_s']:8.0f} {s['rep4']:5.3f}")


# ------------------------------------------------------------------ main

def main(argv=None):
    sp = os.environ.get("SP", "")
    data_root = os.environ.get("SCRIBE_DATA", os.path.join(sp, "data") if sp else "data")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", default=data_root, help="data root (default $SCRIBE_DATA or $SP/data)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--gen-dir", default=None, help="output root (default <data-root>/gen)")
        p.add_argument("--seed", type=int, default=DEFAULT_SEED)
        p.add_argument("--threads", type=int, default=2)

    p = sub.add_parser("hf", help="continuations from a Hugging Face causal LM")
    common(p)
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--decoding", choices=["greedy", "sample", "beam"], default="sample")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0, help="0 = off")
    p.add_argument("--top-p", type=float, default=1.0, help="1.0 = off")
    p.add_argument("--num-beams", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=144)
    p.add_argument("--min-new-tokens", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--prompts", nargs="+", required=True)
    p.add_argument("--limit", type=int, default=None, help="first N prompts of each file")
    p.add_argument("--ladder", default="adhoc")
    p.add_argument("--system", default=None, help="default: the model name")
    p.add_argument("--out", default=None, help="output file (one prompts file only)")
    p.set_defaults(func=cmd_hf)

    p = sub.add_parser("ngram", help="continuations from interpolated Kneser-Ney word n-grams")
    common(p)
    p.add_argument("--train", default=None, help="default <data-root>/wiki/gen_train.jsonl")
    p.add_argument("--orders", type=lambda s: [int(x) for x in s.split(",")], default=[1, 2, 3, 4, 5])
    p.add_argument("--n-words", type=int, default=110, help="words per continuation (punctuation not counted)")
    p.add_argument("--prompts", nargs="+", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ladder", default="ngram")
    p.set_defaults(func=cmd_ngram)

    for name, func, hlp in [("ladders", cmd_ladders, "run the whole §5.3 plan (resumable)"),
                            ("estimate", cmd_estimate, "time one batch per model, project hours")]:
        p = sub.add_parser(name, help=hlp)
        common(p)
        p.add_argument("--wiki-dir", default=None, help="default <data-root>/wiki")
        p.add_argument("--sets", nargs="+", default=["dev", "fresh"])
        p.add_argument("--batch-size", type=int, default=16)
        p.add_argument("--max-new-tokens", type=int, default=144)
        p.add_argument("--n-words", type=int, default=110)
        if name == "ladders":
            p.add_argument("--skip", nargs="*", help="system names to skip (e.g. beam4)")
            p.add_argument("--only", nargs="*", help="ladders to run (default all)")
            p.add_argument("--no-ngram", action="store_true")
            p.add_argument("--limit", type=int, default=None,
                           help="first N prompts of each file for every Hugging Face job")
            p.add_argument("--limit-large", type=int, default=None,
                           help="first N prompts of each file for the --large-models (overrides --limit)")
            p.add_argument("--large-models", nargs="*", default=["pythia-1b", "pythia-1.4b"])
            p.add_argument("--batch-override", nargs="*",
                           help="per-model batch size, e.g. pythia-1.4b=25 (rows, for beam search)")
            p.add_argument("--keep-cache", action="store_true",
                           help="keep finished models in the HF cache (default: delete each one when done)")
            p.add_argument("--min-free-gb", type=float, default=8.0,
                           help="with --keep-cache, still delete a finished model below this free disk")
        else:
            p.add_argument("--models", nargs="*", help="model names to time (default all)")
        p.set_defaults(func=func)

    p = sub.add_parser("summary", help="summarise the output files")
    common(p)
    p.set_defaults(func=cmd_summary)

    args = ap.parse_args(argv)
    if getattr(args, "gen_dir", None) is None:
        args.gen_dir = os.path.join(args.data_root, "gen")
    if hasattr(args, "wiki_dir") and args.wiki_dir is None:
        args.wiki_dir = os.path.join(args.data_root, "wiki")
    if hasattr(args, "train") and args.train is None:
        args.train = os.path.join(args.data_root, "wiki", "gen_train.jsonl")
    args.func(args)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    sys.exit(main())
