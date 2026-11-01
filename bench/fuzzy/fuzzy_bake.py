#!/usr/bin/env python3
"""Fuzzy matching without a network: the bake-off (docs/GENERATION_PLAN.md §13.7). Odin Loch, 2026-11-01.

Word-level next-word models on about 8 MB of clean enwik8 prose ([0, 16) MiB, SCRIBE's cleaner). Every fuzzy
model is mixed with an interpolated modified Kneser-Ney 5-gram (K5), with its weight fit on held-out prose, and
scored by bits per byte on held-out prose and by SCRIBE v2 on GEN-A's 250 dev prompts (T 1.0, nucleus 0.9, the
byte sampler's 11-byte repeat ban lifted to words). numpy only. Instruments for choosing what to build in C++;
nothing here is CyphaLM's default.

Arms: K5; F1 class substitution; F2 gapped contexts; F3 SimHash buckets; F4 memory-based (IB1-IG over word and
class features); F5 random class forest (an approximation of Xu and Jelinek's random forests: random partitions
of the history, not grown by likelihood); F6 indirect contexts (PAQ); F7 CCA states + kNN; F8a sparse distributed
memory over the last words; F8b sparse distributed memory over a decayed state of the whole past; ALL.

F9 is a zero-order Takagi-Sugeno fuzzy rule base (fuzzy c-means sets, product firing, consequents learned by
counting: a normalised RBF network without gradient descent); F10 is fuzzy ART, a self-organising category learner.

Usage (one job per arm, each on one core; bench/fuzzy/run_bake.sh runs them all):
       fuzzy_bake.py prep          data, vocabulary, vectors and classes (shared, cached in FZ_WORK)
       fuzzy_bake.py arm NAME      K5 + one arm (or ALL): fit, score, generate, SCRIBE -> FZ_WORK/res_NAME.json
       fuzzy_bake.py report        FZ_WORK/summary.md from every res_*.json
       FZ_SMALL=1 ...              a smoke test on a small slice
"""
import json
import math
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SMALL = os.environ.get("FZ_SMALL") == "1"
WORK = os.environ.get("FZ_WORK", "/root/work/fz" + ("_small" if SMALL else ""))
E8 = os.environ.get("ENWIK8", "/root/cdata/enwik8")
SCRIBE = os.environ.get("SCRIBE", "/root/work/chk/scribe")
SCRIBE_VEC = os.environ.get("SCRIBE_VEC", "/root/work/scribe/vec32")  # used by SCRIBE only, never by a model
VMAX = 32000
UNK, BOS = 0, 1
TOKRE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*|[0-9]+(?:[.,][0-9]+)*|\n|[^\sA-Za-z0-9]")
NO_SPACE_BEFORE = set(".,;:!?)]}%") | {"'s", "'"}
NO_SPACE_AFTER = set("([{$")
SEED = 20261101
ARMS = ["K5", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8a", "F8b", "F9", "F10"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def piece(tok, prev):
    if tok == "\n":
        return "\n"
    if prev is None or prev == "\n" or prev in NO_SPACE_AFTER or tok in NO_SPACE_BEFORE:
        return tok
    return " " + tok


def unesc(s):
    out, i = [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            out.append({"t": "\t", "n": "\n", "\\": "\\"}.get(s[i + 1], s[i + 1]))
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def esc(s):
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")


# ----------------------------------------------------------------------------------------------- data
def prep():
    if os.path.exists(f"{WORK}/train.npy"):
        return
    os.makedirs(WORK, exist_ok=True)
    clean = f"{WORK}/clean.jsonl"
    if not os.path.exists(clean):
        hi = "2" if SMALL else "16"
        subprocess.run([SCRIBE, "docs", f"enwik:{E8}:0:{hi}", "--out", clean], check=True, stdout=subprocess.DEVNULL)
    bench = set()
    for split in ("dev", "test"):  # only the continuations' 32-byte spans are read from test, to drop contaminated pages
        for line in open(f"{ROOT}/bench/eval/gen_prompts_{split}.tsv", encoding="utf-8").read().split("\n")[1:]:
            f = line.split("\t")
            if len(f) >= 4:
                b = unesc(f[3]).encode("utf-8")
                bench.update(b[i:i + 32] for i in range(len(b) - 31))
    docs, dropped = [], 0
    for line in open(clean, encoding="utf-8"):
        t = json.loads(line)["text"]
        b = t.encode("utf-8")
        if any(b[i:i + 32] in bench for i in range(len(b) - 31)):
            dropped += 1
            continue
        docs.append(t)
    val = [d for i, d in enumerate(docs) if i % 20 == 0]
    train = [d for i, d in enumerate(docs) if i % 20 != 0]
    counts = {}
    toks_train = [TOKRE.findall(d) for d in train]
    for ts in toks_train:
        for t in ts:
            counts[t] = counts.get(t, 0) + 1
    vocab = ["<unk>", "<s>"] + [t for t, c in sorted(counts.items(), key=lambda x: -x[1]) if c >= 2][:VMAX - 2]
    idx = {t: i for i, t in enumerate(vocab)}

    def enc(tok_docs):
        ids, doc_of = [], []
        for di, ts in enumerate(tok_docs):
            ids.append(BOS)
            doc_of.append(di)
            for t in ts:
                ids.append(idx.get(t, UNK))
                doc_of.append(di)
        return np.array(ids, np.int32), np.array(doc_of, np.int32)

    tr, _ = enc(toks_train)
    va, vdoc = enc([TOKRE.findall(d) for d in val])
    np.save(f"{WORK}/train.npy", tr)
    np.save(f"{WORK}/val.npy", va)
    np.save(f"{WORK}/val_doc.npy", vdoc)
    json.dump(vocab, open(f"{WORK}/vocab.json", "w"))
    open(f"{WORK}/train.txt", "w", encoding="utf-8").write("\n".join(train))
    info = dict(docs=len(docs), dropped=dropped, train_docs=len(train), val_docs=len(val),
                train_bytes=sum(len(d.encode()) for d in train), val_bytes=sum(len(d.encode()) for d in val),
                train_tokens=int(len(tr)), val_tokens=int(len(va)), vocab=len(vocab),
                oov_train=float(np.mean(tr == UNK)), oov_val=float(np.mean(va == UNK)))
    json.dump(info, open(f"{WORK}/data.json", "w"), indent=1)
    log("prep", info)


# ----------------------------------------------------------------------------------------------- tables
U16 = np.uint64(16)


def pack(cols):
    """cols[i] is position i+1 back (w-1 first); 16 bits each."""
    k = np.zeros(len(cols[0]), np.uint64)
    for i, c in enumerate(cols):
        k |= c.astype(np.uint64) << np.uint64(16 * i)
    return k


def pack1(vals):
    k = 0
    for i, v in enumerate(vals):
        k |= int(v) << (16 * i)
    return np.uint64(k)


class Table:
    """key -> next-word counts (CSR, keys sorted, next ids sorted within a key)."""

    def __init__(self, keys, nexts, weights=None):
        o = np.lexsort((nexts, keys))
        k, w = keys[o], nexts[o]
        if len(k) == 0:
            self.keys = np.zeros(0, np.uint64)
            self.off = np.zeros(1, np.int64)
            self.ids = np.zeros(0, np.int32)
            self.cnt = np.zeros(0, np.float32)
            self.tot = np.zeros(0)
            return
        brk = np.flatnonzero((k[1:] != k[:-1]) | (w[1:] != w[:-1])) + 1
        st = np.concatenate(([0], brk))
        cnt = np.diff(np.concatenate((st, [len(k)]))) if weights is None else np.add.reduceat(weights[o], st)
        pk, pw = k[st], w[st]
        kb = np.flatnonzero(pk[1:] != pk[:-1]) + 1
        ks = np.concatenate(([0], kb))
        self.keys = pk[ks]
        self.off = np.concatenate((ks, [len(pk)])).astype(np.int64)
        self.ids = pw.astype(np.int32)
        self.cnt = cnt.astype(np.float32)
        self.tot = np.add.reduceat(self.cnt.astype(np.float64), ks)

    def find(self, key):
        i = int(np.searchsorted(self.keys, key))
        return i if i < len(self.keys) and self.keys[i] == key else -1

    def pairs(self):
        return np.repeat(self.keys, np.diff(self.off)), self.ids

    def dense_into(self, i, out, weight):
        a, b = self.off[i], self.off[i + 1]
        out[self.ids[a:b]] += weight * self.cnt[a:b] / self.tot[i]


# ----------------------------------------------------------------------------------------------- K5
class KN:
    """Interpolated modified Kneser-Ney (Chen and Goodman), orders 1-5, over word ids."""

    def __init__(self, ids, V, order=5):
        self.V, self.order = V, order
        n = len(ids)
        self.T, self.disc, self.gam = {}, {}, {}
        L = order - 1
        keys = pack([ids[L - i - 1:n - i - 1] for i in range(L)])
        top = Table(keys, ids[L:])
        tabs = {order: top}
        for k in range(order - 1, 1, -1):  # continuation counts from the unique (k+1)-grams
            pk, pw = tabs[k + 1].pairs()
            tabs[k] = Table(pk & np.uint64((1 << (16 * (k - 1))) - 1), pw)
        _, pw2 = tabs[2].pairs()
        cc = np.bincount(pw2, minlength=V).astype(np.float64)
        self.p1 = (cc + 0.01) / (cc + 0.01).sum()
        for k, t in tabs.items():
            c = t.cnt
            n1, n2, n3, n4 = [max(1, int(np.sum(c == j))) for j in (1, 2, 3, 4)]
            Y = n1 / (n1 + 2 * n2)
            D = np.array([0, 1 - 2 * Y * n2 / n1, 2 - 3 * Y * n3 / n2, 3 - 4 * Y * n4 / n3])
            D = np.clip(D, 0, None)
            dc = D[np.minimum(c.astype(np.int64), 3)]
            per_ctx = np.repeat(np.arange(len(t.keys)), np.diff(t.off))
            self.disc[k] = ((c - dc) / t.tot[per_ctx]).astype(np.float64)
            g = np.bincount(per_ctx, weights=dc, minlength=len(t.keys))
            self.gam[k] = g / t.tot
            self.T[k] = t

    def dist(self, ctx):
        p = self.p1.copy()
        for k in range(2, self.order + 1):
            L = k - 1
            if len(ctx) < L:
                break
            i = self.T[k].find(pack1(ctx[::-1][:L]))
            if i < 0:
                break
            t = self.T[k]
            a, b = t.off[i], t.off[i + 1]
            p *= self.gam[k][i]
            p[t.ids[a:b]] += self.disc[k][a:b]
        return p


# ----------------------------------------------------------------------------------------------- vectors, classes
def embeddings(ids, V, d=64, nctx=1500):
    """Our own PPMI-SVD word vectors from the training text (window +-2); SCRIBE's vectors are not used."""
    freq = np.bincount(ids, minlength=V)
    order = np.argsort(-freq)
    ctx_words = [w for w in order if w not in (UNK, BOS)][:nctx]
    cmap = -np.ones(V, np.int64)
    cmap[ctx_words] = np.arange(len(ctx_words))
    M = np.zeros(V * nctx, np.float64)
    n = len(ids)
    for off in (-2, -1, 1, 2):
        a = ids[max(0, -off):n - max(0, off)]  # the word at t
        b = ids[max(0, off):n + min(0, off)]  # the word at t + off
        m = cmap[b] >= 0
        M += np.bincount(a[m].astype(np.int64) * nctx + cmap[b[m]], minlength=V * nctx)
    M = M.reshape(V, nctx)
    tot = M.sum()
    rs, cs = M.sum(1, keepdims=True), M.sum(0, keepdims=True)
    pc = cs ** 0.75 / (cs ** 0.75).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.log((M / tot) / ((rs / tot) * pc))
    P[~np.isfinite(P)] = 0
    P = np.maximum(P, 0).astype(np.float32)
    U, S, _ = np.linalg.svd(P, full_matrices=False)
    E = U[:, :d] * np.sqrt(S[:d])
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    E[UNK] = 0
    E[BOS] = 0
    return E.astype(np.float32)


def kmeans(E, K, seed, iters=12):
    rng = np.random.default_rng(seed)
    X = E[2:]
    C = X[rng.choice(len(X), K, replace=False)].copy()
    for _ in range(iters):
        a = np.argmax(X @ C.T, axis=1)
        for j in range(K):
            m = a == j
            if m.any():
                c = X[m].mean(0)
                C[j] = c / (np.linalg.norm(c) + 1e-9)
    lab = np.zeros(len(E), np.int32)
    lab[2:] = np.argmax(X @ C.T, axis=1)
    lab[UNK], lab[BOS] = K, K + 1
    return lab


# ----------------------------------------------------------------------------------------------- pattern bank (F1, F2, F4)
W_, C_, N_ = "W", "C", "N"


def all_patterns(L=4):
    out = [()]
    for _ in range(L):
        out = [p + (s,) for p in out for s in (W_, C_, N_)]
    return [p for p in out if any(s != N_ for s in p)]


def pat_vals(p, words, cls):
    """words[i], cls[i]: arrays (or scalars) for position i+1 back."""
    cols = []
    for i, s in enumerate(p):
        if s == W_:
            cols.append(words[i])
        elif s == C_:
            cols.append(32768 + cls[i])
        else:
            cols.append(65535 if np.isscalar(words[i]) else np.full(len(words[i]), 65535, np.int64))
    return cols


class Bank:
    def __init__(self, ids, cls, L=4):
        n = len(ids)
        self.L, self.cls = L, cls
        words = [ids[L - i - 1:n - i - 1].astype(np.int64) for i in range(L)]
        cl = [cls[w] for w in words]
        nxt = ids[L:]
        self.tab = {}
        for p in all_patterns(L):
            self.tab[p] = Table(pack(pat_vals(p, words, cl)), nxt)

    def keys(self, ctx):
        c = list(ctx[::-1][:self.L]) + [BOS] * self.L
        c = c[:self.L]
        return c, [int(self.cls[w]) for w in c]

    def lookup(self, p, words, cls):
        t = self.tab[p]
        return t, t.find(pack1(pat_vals(p, words, cls)))


class PatternMix:
    """A weighted mixture of pattern tables' maximum-likelihood distributions, renormalised over the patterns whose
    key was seen. Weights fit on held-out text (fit_weights)."""

    def __init__(self, bank, pats, V):
        self.bank, self.pats, self.V = bank, pats, V
        self.w = np.ones(len(pats)) / len(pats)

    def parts(self, ctx):
        words, cls = self.bank.keys(ctx)
        out = []
        for j, p in enumerate(self.pats):
            t, i = self.bank.lookup(p, words, cls)
            if i >= 0:
                out.append((j, t, i))
        return out

    def dist(self, ctx):
        parts = self.parts(ctx)
        if not parts:
            return None
        p = np.zeros(self.V)
        z = sum(self.w[j] for j, _, _ in parts)
        for j, t, i in parts:
            t.dense_into(i, p, self.w[j] / z)
        return p

    def fit(self, ctxs, ys, iters=20):
        Q, A = np.zeros((len(ys), len(self.pats))), np.zeros((len(ys), len(self.pats)), bool)
        for r, (ctx, y) in enumerate(zip(ctxs, ys)):
            for j, t, i in self.parts(ctx):
                a, b = t.off[i], t.off[i + 1]
                k = np.searchsorted(t.ids[a:b], y)
                Q[r, j] = t.cnt[a + k] / t.tot[i] if k < b - a and t.ids[a + k] == y else 0.0
                A[r, j] = True
        self.w = sleeping_em(Q, A)


def sleeping_em(Q, A, iters=30):
    """Mixture weights over experts that are not always available: an unavailable expert predicts the mixture of
    the available ones, which makes the update ordinary EM."""
    K = Q.shape[1]
    w = np.ones(K) / K
    rows = A.any(1)
    Q, A = Q[rows], A[rows]
    for _ in range(iters):
        wa = A * w
        P = (wa * Q).sum(1) / np.maximum(wa.sum(1), 1e-300)
        Qp = np.where(A, Q, P[:, None])
        R = w * Qp
        R /= np.maximum(R.sum(1, keepdims=True), 1e-300)
        w = R.mean(0) + 1e-6
        w /= w.sum()
    return w


class IB1:
    """Memory-based learning (IB1-IG; Daelemans and van den Bosch): the nearest stored examples under a distance whose
    feature weights are information gains (word and class at each of 4 positions). Neighbours are read from the
    pattern bank, nearest first, until k examples are gathered; farther buckets are down-weighted by exp(-beta*gap)."""

    def __init__(self, bank, ids, V, k=16):
        self.bank, self.V, self.k, self.beta = bank, V, k, 1.0
        y = ids[bank.L:]
        Hy = entropy_counts(np.bincount(y, minlength=V))
        ig_w, ig_c = [], []
        for i in range(bank.L):
            for s, dst in ((W_, ig_w), (C_, ig_c)):
                p = tuple(s if j == i else N_ for j in range(bank.L))
                dst.append(Hy - cond_entropy(bank.tab[p]))
        self.ig_w, self.ig_c = np.array(ig_w), np.array(ig_c)
        sims = []
        for p in all_patterns(bank.L):
            s = sum(self.ig_w[i] + self.ig_c[i] if v == W_ else self.ig_c[i] if v == C_ else 0 for i, v in enumerate(p))
            sims.append((s, p))
        self.order = [p for s, p in sorted(sims, key=lambda x: -x[0])]
        self.sim = {p: s for s, p in sims}

    def dist(self, ctx):
        words, cls = self.bank.keys(ctx)
        p, got, best = np.zeros(self.V), 0.0, None
        for pat in self.order:
            t, i = self.bank.lookup(pat, words, cls)
            if i < 0:
                continue
            if best is None:
                best = self.sim[pat]
            wgt = math.exp(-self.beta * (best - self.sim[pat]))
            a, b = t.off[i], t.off[i + 1]
            p[t.ids[a:b]] += wgt * t.cnt[a:b]
            got += t.tot[i]
            if got >= self.k:
                break
        if best is None:
            return None
        return p / p.sum()


def entropy_counts(c):
    c = c[c > 0].astype(np.float64)
    q = c / c.sum()
    return float(-(q * np.log2(q)).sum())


def cond_entropy(t):
    per = np.repeat(np.arange(len(t.keys)), np.diff(t.off))
    q = t.cnt / t.tot[per]
    h_ctx = np.bincount(per, weights=-(q * np.log2(q)), minlength=len(t.keys))
    return float((h_ctx * t.tot).sum() / t.tot.sum())


# ----------------------------------------------------------------------------------------------- F3, F5, F6, F7, F8
def ctx_feat(E, words, beta=(1.0, 0.6, 0.35, 0.2)):
    return sum(b * E[w] for b, w in zip(beta, words))


class SimHash:
    def __init__(self, ids, E, V, bits=(10, 14, 18, 22), seed=SEED):
        self.E, self.V, self.bits = E, V, bits
        self.R = np.random.default_rng(seed).standard_normal((E.shape[1], max(bits))).astype(np.float32)
        n, L = len(ids), 4
        f = ctx_feat(E, [ids[L - i - 1:n - i - 1] for i in range(L)])
        code = self._code(f)
        self.tabs = [Table(code >> np.uint64(max(bits) - b), ids[L:]) for b in bits]
        self.pats = list(range(len(bits)))
        self.w = np.ones(len(bits)) / len(bits)

    def _code(self, f):
        s = (f @ self.R) > 0
        return (s.astype(np.uint64) << np.arange(s.shape[1], dtype=np.uint64)).sum(1).astype(np.uint64) if s.ndim == 2 \
            else np.uint64(sum(int(x) << j for j, x in enumerate(s)))

    def parts(self, ctx):
        c = list(ctx[::-1][:4]) + [BOS] * 4
        code = self._code(ctx_feat(self.E, c[:4]))
        out = []
        for j, (b, t) in enumerate(zip(self.bits, self.tabs)):
            i = t.find(np.uint64(int(code) >> (max(self.bits) - b)))
            if i >= 0:
                out.append((j, t, i))
        return out

    dist = PatternMix.dist
    fit = PatternMix.fit


class Forest:
    """Random class forest: each tree keys the history by random class granularities at random positions."""

    def __init__(self, ids, E, V, ntrees=6, seed=SEED):
        rng = np.random.default_rng(seed)
        self.V, self.trees = V, []
        n, L = len(ids), 3
        words = [ids[L - i - 1:n - i - 1].astype(np.int64) for i in range(L)]
        for tr in range(ntrees):
            K = int(rng.choice([64, 128, 256, 512]))
            lab = kmeans(E, K, seed + 17 * tr)
            spec = [("W" if rng.random() < 0.5 else "C")] + [("C" if rng.random() < 0.7 else "N") for _ in range(L - 1)]
            cols = [w if s == "W" else (32768 + lab[w] if s == "C" else np.full(len(w), 65535, np.int64))
                    for s, w in zip(spec, words)]
            self.trees.append((lab, spec, Table(pack(cols), ids[L:])))

    def dist(self, ctx):
        c = (list(ctx[::-1][:3]) + [BOS] * 3)[:3]
        p, m = np.zeros(self.V), 0
        for lab, spec, t in self.trees:
            vals = [w if s == "W" else (32768 + int(lab[w]) if s == "C" else 65535) for s, w in zip(spec, c)]
            i = t.find(pack1(vals))
            if i >= 0:
                t.dense_into(i, p, 1.0)
                m += 1
        return p / m if m else None


class Indirect:
    """PAQ's indirect context: the last two words that followed the context (w-2, w-1) earlier, with w-1, as a key."""

    def __init__(self, ids, V):
        self.V = V
        last = {}
        k1, k2, nx = [], [], []
        for t in range(2, len(ids)):
            h = (int(ids[t - 2]), int(ids[t - 1]))
            f1, f2 = last.get(h, (65535, 65535))
            k1.append(pack1([ids[t - 1], f1, f2]))
            k2.append(pack1([ids[t - 1], f1]))
            nx.append(ids[t])
            last[h] = (int(ids[t]), f1)
        self.last = last
        nx = np.array(nx, np.int32)
        self.tabs = [Table(np.array(k1, np.uint64), nx), Table(np.array(k2, np.uint64), nx)]
        self.pats = [0, 1]
        self.w = np.array([0.5, 0.5])

    def _followers(self, ctx):
        h = (ctx[-2], ctx[-1]) if len(ctx) >= 2 else (BOS, ctx[-1] if ctx else BOS)
        f = []
        for t in range(len(ctx) - 1, 1, -1):  # the document so far overrides training
            if ctx[t - 2] == h[0] and ctx[t - 1] == h[1]:
                f.append(ctx[t])
                if len(f) == 2:
                    break
        g = list(self.last.get(h, (65535, 65535)))
        f = (f + g)[:2]
        return h, f

    def parts(self, ctx):
        h, f = self._followers(ctx)
        out = []
        for j, (t, key) in enumerate(zip(self.tabs, (pack1([h[1], f[0], f[1]]), pack1([h[1], f[0]])))):
            i = t.find(key)
            if i >= 0:
                out.append((j, t, i))
        return out

    dist = PatternMix.dist
    fit = PatternMix.fit


class CCAkNN:
    """Context states from CCA between the past (vectors of w-1..w-3) and the future (the vector of w0); the next word
    from the k nearest training states (route 2's learned linear map)."""

    def __init__(self, ids, E, V, d=32, store=600000, k=64, seed=SEED):
        self.E, self.V, self.k, self.tau = E, V, k, 0.05
        n, L, dE = len(ids), 3, E.shape[1]
        N, Dx = n - L, L * dE

        def X_of(r):
            return np.concatenate([E[ids[r + L - 1 - i]] for i in range(L)], 1).astype(np.float64)

        Sx, Sy = np.zeros(Dx), np.zeros(dE)
        Sxx, Syy, Sxy = np.zeros((Dx, Dx)), np.zeros((dE, dE)), np.zeros((Dx, dE))
        for a in range(0, N, 200000):
            r = np.arange(a, min(N, a + 200000))
            X, Y = X_of(r), E[ids[r + L]].astype(np.float64)
            Sx += X.sum(0); Sy += Y.sum(0); Sxx += X.T @ X; Syy += Y.T @ Y; Sxy += X.T @ Y
        mx, my = Sx / N, Sy / N
        Cxx = Sxx / N - np.outer(mx, mx) + 1e-3 * np.eye(Dx)
        Cyy = Syy / N - np.outer(my, my) + 1e-3 * np.eye(dE)
        Cxy = Sxy / N - np.outer(mx, my)
        Wx, Wy = inv_sqrt(Cxx), inv_sqrt(Cyy)
        Um, S, _ = np.linalg.svd(Wx @ Cxy @ Wy)
        self.mx, self.A = mx, (Wx @ Um[:, :d] * S[:d])
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(N, min(store, N), replace=False))
        Z = np.concatenate([((X_of(sel[a:a + 200000]) - mx) @ self.A) for a in range(0, len(sel), 200000)]).astype(np.float32)
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
        self.Z, self.nxt = Z, ids[sel + L]

    def dist(self, ctx):
        c = (list(ctx[::-1][:3]) + [BOS] * 3)[:3]
        x = np.concatenate([self.E[w] for w in c]).astype(np.float64) - self.mx
        z = (x @ self.A).astype(np.float32)
        z /= np.linalg.norm(z) + 1e-9
        s = self.Z @ z
        top = np.argpartition(-s, self.k)[:self.k]
        w = np.exp((s[top] - s[top].max()) / self.tau)
        p = np.bincount(self.nxt[top], weights=w, minlength=self.V)
        return p / p.sum()


def inv_sqrt(C):
    v, U = np.linalg.eigh(C)
    return U @ np.diag(1 / np.sqrt(np.maximum(v, 1e-9))) @ U.T


class SDM:
    """Sparse distributed memory with counters (Kanerva): M random hard locations in {-1,+1}^256; a context's
    address activates its L nearest locations, each holding next-word counters. long=True addresses by a decayed
    state of the whole past (a hyperdimensional bag, decay lam over 40 words) beside the last word."""

    def __init__(self, ids, E, V, long=False, M=8192, L=8, bits=256, lam=0.85, seed=SEED):
        rng = np.random.default_rng(seed + (7 if long else 0))
        self.E, self.V, self.L, self.long, self.lam = E, V, L, long, lam
        d = E.shape[1]
        self.P = rng.standard_normal((2 * d if long else 3 * d, bits)).astype(np.float32)
        self.H = np.sign(rng.standard_normal((bits, M))).astype(np.float32)
        n = len(ids)
        feats = self._train_feats(ids)
        locs = np.zeros((n - 3, L), np.int64)
        for a in range(0, n - 3, 2048):
            f = np.sign(feats[a:a + 2048] @ self.P)
            locs[a:a + 2048] = np.argpartition(-(f @ self.H), L, axis=1)[:, :L]
        nxt = np.repeat(ids[3:], L)
        self.tab = Table(locs.reshape(-1).astype(np.uint64), nxt)

    def _state(self, hist):
        h = hist[-40:]
        w = self.lam ** np.arange(len(h))[::-1]
        s = (w[:, None] * self.E[h]).sum(0)
        return s / (np.linalg.norm(s) + 1e-9)

    def _train_feats(self, ids):
        n, d = len(ids), self.E.shape[1]
        if not self.long:
            return np.concatenate([self.E[ids[3 - i - 1:n - i - 1]] * b for i, b in enumerate((1.0, 0.7, 0.5))], 1)
        S = np.zeros((n - 3, d), np.float32)
        for l in range(40):
            lo = 3 - 1 - l
            src = ids[max(lo, 0):n - 1 - l]
            S[max(0, -lo):] += (self.lam ** l) * self.E[src]
        S /= np.linalg.norm(S, axis=1, keepdims=True) + 1e-9
        return np.concatenate([self.E[ids[2:n - 1]], S], 1)

    def dist(self, ctx):
        c = list(ctx) if len(ctx) >= 3 else [BOS] * (3 - len(ctx)) + list(ctx)
        if self.long:
            f = np.concatenate([self.E[c[-1]], self._state(np.array(c))])
        else:
            f = np.concatenate([self.E[c[-1 - i]] * b for i, b in enumerate((1.0, 0.7, 0.5))])
        sc = np.sign(f @ self.P) @ self.H
        top = np.argpartition(-sc, self.L)[:self.L]
        p = np.zeros(self.V)
        for loc in top:
            i = self.tab.find(np.uint64(int(loc)))
            if i >= 0:
                self.tab.dense_into(i, p, 1.0)
        return p / p.sum() if p.sum() > 0 else None


# ----------------------------------------------------------------------------------------------- F9, F10
def fcm(X, C, m, seed, iters=15):
    """Fuzzy c-means (Bezdek) on unit vectors: soft memberships, centres renormalised to the sphere."""
    rng = np.random.default_rng(seed)
    Ce = X[rng.choice(len(X), C, replace=False)].copy()
    for _ in range(iters):
        d2 = np.maximum(2 - 2 * (X @ Ce.T), 1e-6)
        inv = d2 ** (-1.0 / (m - 1))
        U = inv / inv.sum(1, keepdims=True)
        Um = U ** m
        Ce = (Um.T @ X) / Um.sum(0)[:, None]
        Ce /= np.linalg.norm(Ce, axis=1, keepdims=True) + 1e-9
    return Ce.astype(np.float32)


class FuzzyRules:
    """F9: IF w-1 is A_i AND w-2 is A_j THEN next ~ P_ij. Fuzzy sets are fuzzy c-means prototypes of word vectors;
    each word belongs to its 3 nearest sets with FCM memberships (m = 2); firing = product (t-norm); consequents are
    next-word counts weighted by firing, learned by counting; inference is the firing-weighted average."""

    def __init__(self, ids, E, V, C=256, top=3, m=2.0, seed=SEED):
        self.V = V
        P = fcm(E[2:], C, m, seed)
        d2 = np.maximum(2 - 2 * (E @ P.T), 1e-6)
        mi = np.argpartition(d2, top, axis=1)[:, :top]
        dd = np.take_along_axis(d2, mi, 1)
        inv = dd ** (-1.0 / (m - 1))
        mu = inv / inv.sum(1, keepdims=True)
        mi[UNK], mi[BOS] = C, C + 1
        mu[UNK], mu[BOS] = 1.0 / top, 1.0 / top
        self.mi, self.mu, self.top = mi.astype(np.int64), mu.astype(np.float32), top
        n = len(ids)
        w1, w2, nx = ids[1:n - 1], ids[0:n - 2], ids[2:]
        keys, wts, nxs = [], [], []
        for a in range(top):
            for b in range(top):
                keys.append(self.mi[w1, a].astype(np.uint64) | (self.mi[w2, b].astype(np.uint64) << U16))
                wts.append(self.mu[w1, a] * self.mu[w2, b])
                nxs.append(nx)
        self.tab = Table(np.concatenate(keys), np.concatenate(nxs), np.concatenate(wts))

    def dist(self, ctx):
        c = (list(ctx[::-1][:2]) + [BOS] * 2)[:2]
        p, z = np.zeros(self.V), 0.0
        for a in range(self.top):
            for b in range(self.top):
                i = self.tab.find(np.uint64(int(self.mi[c[0], a]) | (int(self.mi[c[1], b]) << 16)))
                if i >= 0:
                    phi = float(self.mu[c[0], a] * self.mu[c[1], b])
                    self.tab.dense_into(i, p, phi)
                    z += phi
        return p / z if z > 0 else None


class FuzzyART:
    """F10: fuzzy ART (Carpenter, Grossberg and Rosen, 1991), a self-organising category learner. Input: the vectors
    of w-1 and w-2 (first 32 dimensions) scaled to [0, 1] and complement-coded. Categories form online on a sample of
    contexts (a new one whenever none matches at vigilance rho; fast learning). The categories then count next words
    over a larger sample, and a context reads its 3 best categories by the choice function."""

    def __init__(self, ids, E, V, rho=0.7, alpha=0.001, d=32, learn=150000, count=600000, maxcat=3000, seed=SEED):
        self.E, self.V, self.rho, self.alpha, self.d = E, V, rho, alpha, d
        n = len(ids)
        rng = np.random.default_rng(seed)
        W = []
        for t in np.sort(rng.choice(np.arange(2, n), min(learn, n - 2), replace=False)):
            I = self._input(ids[t - 1], ids[t - 2])
            if W:
                Wa = np.array(W)
                mn = np.minimum(I, Wa).sum(1)
                T = mn / (alpha + Wa.sum(1))
                for j in np.argsort(-T):
                    if mn[j] / I.sum() >= rho:
                        W[j] = np.minimum(I, W[j])
                        break
                else:
                    if len(W) < maxcat:
                        W.append(I.copy())
            else:
                W.append(I.copy())
        self.W = np.array(W, np.float32)
        sel = np.sort(rng.choice(np.arange(2, n), min(count, n - 2), replace=False))
        cat = np.zeros(len(sel), np.int64)
        for a in range(0, len(sel), 256):
            t = sel[a:a + 256]
            I = self._input(ids[t - 1], ids[t - 2])
            T = np.minimum(I[:, None, :], self.W[None]).sum(2) / (alpha + self.W.sum(1))
            cat[a:a + 256] = np.argmax(T, 1)
        self.tab = Table(cat.astype(np.uint64), ids[sel])
        self.ncat = len(self.W)

    def _input(self, w1, w2):
        f = np.concatenate([self.E[w1][..., :self.d], self.E[w2][..., :self.d]], -1)
        f = np.clip((f * 2.5 + 1) / 2, 0, 1)
        return np.concatenate([f, 1 - f], -1).astype(np.float32)

    def dist(self, ctx):
        c = (list(ctx[::-1][:2]) + [BOS] * 2)[:2]
        I = self._input(c[0], c[1])
        T = np.minimum(I, self.W).sum(1) / (self.alpha + self.W.sum(1))
        p, z = np.zeros(self.V), 0.0
        for j in np.argsort(-T)[:3]:
            i = self.tab.find(np.uint64(int(j)))
            if i >= 0:
                self.tab.dense_into(i, p, float(T[j]))
                z += float(T[j])
        return p / z if z > 0 else None


# ----------------------------------------------------------------------------------------------- build, fit
G = {}
NEEDS_BANK = {"F1", "F2", "F4"}


def shared():
    """Vectors and classes, computed once by prep and loaded by every arm job."""
    tr = np.load(f"{WORK}/train.npy")
    vocab = json.load(open(f"{WORK}/vocab.json"))
    V = len(vocab)
    if not os.path.exists(f"{WORK}/E.npy"):
        E = embeddings(tr, V)
        np.save(f"{WORK}/E.npy", E)
        np.save(f"{WORK}/cls.npy", kmeans(E, 512, SEED))
    G.update(tr=tr, vocab=vocab, V=V, E=np.load(f"{WORK}/E.npy"), cls=np.load(f"{WORK}/cls.npy"))
    G["tid"] = {t: i for i, t in enumerate(vocab)}


def build(names):
    tr, V, E, cls = G["tr"], G["V"], G["E"], G["cls"]
    t0 = time.time()
    G["K5"] = KN(tr, V)
    bank = Bank(tr, cls) if NEEDS_BANK & set(names) else None
    f1 = [tuple(C_ if i == j else W_ for i in range(L)) + (N_,) * (4 - L) for L in (2, 3, 4) for j in range(L)]
    f2 = [(N_, W_, N_, N_), (N_, W_, W_, N_), (W_, N_, W_, N_), (N_, N_, W_, N_), (W_, W_, N_, W_), (W_, N_, W_, W_),
          (N_, W_, W_, W_)]
    make = {"F1": lambda: PatternMix(bank, f1, V), "F2": lambda: PatternMix(bank, f2, V), "F3": lambda: SimHash(tr, E, V),
            "F4": lambda: IB1(bank, tr, V), "F5": lambda: Forest(tr, E, V), "F6": lambda: Indirect(tr, V),
            "F7": lambda: CCAkNN(tr, E, V), "F8a": lambda: SDM(tr, E, V, long=False), "F8b": lambda: SDM(tr, E, V, long=True),
            "F9": lambda: FuzzyRules(tr, E, V), "F10": lambda: FuzzyART(tr, E, V)}
    for a in names:
        if a != "K5":
            G[a] = make[a]()
            log(a, "built", round(time.time() - t0))
    if "F10" in names:
        G["F10_categories"] = G["F10"].ncat


def val_positions():
    va, vdoc = np.load(f"{WORK}/val.npy"), np.load(f"{WORK}/val_doc.npy")
    rng = np.random.default_rng(SEED)
    cand = np.flatnonzero(va != BOS)
    cand = cand[cand >= 1]
    n = 6000 if SMALL else 40000
    pos = np.sort(rng.choice(cand, min(n, len(cand)), replace=False))
    return va, vdoc, pos


def contexts(seq, pos):
    lastbos = np.maximum.accumulate(np.where(seq == BOS, np.arange(len(seq)), 0))
    return [[int(x) for x in seq[max(int(lastbos[t]), t - 60):t]] for t in pos]


def inner_fit(names, ctxs, ys):
    for a in ("F1", "F2", "F3", "F6"):
        if a in names:
            G[a].fit(ctxs, ys)
    for a, attr, grid in (("F4", "beta", (0.5, 1.0, 2.0)), ("F7", "tau", (0.02, 0.05, 0.1))):
        if a not in names:
            continue
        best = None
        for v in grid:
            setattr(G[a], attr, v)
            ll = 0.0
            for c, y in zip(ctxs[:1500], ys[:1500]):
                d = G[a].dist(c)
                ll += math.log(max(d[y], 1e-12)) if d is not None else 0.0
            best = max(best or (ll, v), (ll, v))
        setattr(G[a], attr, best[1])


def score(names, arm):
    va, vdoc, pos = val_positions()
    fitm = vdoc[pos] % 2 == 0
    allctx = contexts(va, pos)
    fidx = np.flatnonzero(fitm)[:4000]
    inner_fit(names, [allctx[i] for i in fidx], [int(va[pos[i]]) for i in fidx])
    Q = np.full((len(pos), len(names)), np.nan)
    for r, (c, t) in enumerate(zip(allctx, pos)):
        for j, a in enumerate(names):
            d = G[a].dist(c)
            if d is not None:
                Q[r, j] = d[int(va[t])]
    vocab = G["vocab"]
    nbytes = np.array([len(piece(vocab[va[t]], vocab[va[t - 1]] if va[t - 1] != BOS else None).encode()) for t in pos])
    A, Qz = ~np.isnan(Q), np.nan_to_num(Q)
    cfg = {"names": names}
    if arm == "ALL":
        cfg["w"] = dict(zip(names, sleeping_em(Qz[fitm], A[fitm]).tolist()))
    elif arm != "K5":
        q0, q1, av = Qz[fitm, 0], Qz[fitm, 1], A[fitm, 1]
        lam = 0.2
        for _ in range(40):
            r = lam * q1[av] / np.maximum((1 - lam) * q0[av] + lam * q1[av], 1e-300)
            lam = float(r.mean()) if av.any() else 0.0
        cfg["lambda"] = lam
    ev = ~fitm
    if arm == "ALL":
        w = np.array([cfg["w"][a] for a in names])
        wa = A[ev] * w
        qm = (wa * Qz[ev]).sum(1) / wa.sum(1)
    elif arm == "K5":
        qm = Qz[ev, 0]
    else:
        qm = np.where(A[ev, 1], (1 - cfg["lambda"]) * Qz[ev, 0] + cfg["lambda"] * Qz[ev, 1], Qz[ev, 0])
    bits, base, by, docs = -np.log2(np.maximum(qm, 1e-12)), -np.log2(np.maximum(Qz[ev, 0], 1e-12)), nbytes[ev], vdoc[pos][ev]
    ud = np.unique(docs)
    arr = np.array([(bits[docs == u].sum(), base[docs == u].sum(), by[docs == u].sum()) for u in ud])
    rng = np.random.default_rng(SEED)
    boots = [(lambda s: (s[0] - s[1]) / s[2])(arr[rng.integers(0, len(ud), len(ud))].sum(0)) for _ in range(2000)]
    cfg.update(bpb=float(bits.sum() / by.sum()), K5_bpb=float(base.sum() / by.sum()),
               minus_K5=float((bits.sum() - base.sum()) / by.sum()),
               ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
               coverage=float(A[ev, -1].mean()) if arm not in ("K5", "ALL") else 1.0,
               F4_beta=getattr(G.get("F4"), "beta", None), F7_tau=getattr(G.get("F7"), "tau", None))
    return cfg


def mixed(arm, ctx, cfg):
    p = G["K5"].dist(ctx)
    if arm == "K5":
        return p
    if arm == "ALL":
        w = cfg["w"]
        acc, z = w["K5"] * p, w["K5"]
        for a in cfg["names"][1:]:
            q = G[a].dist(ctx)
            if q is not None:
                acc = acc + w[a] * q
                z += w[a]
        return acc / z
    q = G[arm].dist(ctx)
    return p if q is None else (1 - cfg["lambda"]) * p + cfg["lambda"] * q


# ----------------------------------------------------------------------------------------------- generation
class RepeatBan:
    """The byte sampler's rule, lifted to words: a word is banned when any of its bytes, written in turn, would follow
    an 11-byte window with the byte that followed the same window earlier in the text (prompt included)."""
    W = 11

    def __init__(self, text):
        self.nx = {}
        self.text = b""
        self.add(text.encode("utf-8"))

    def add(self, b):
        old = len(self.text)
        self.text += b
        t = self.text
        for s in range(max(0, old - self.W), len(t) - self.W):
            self.nx.setdefault(t[s:s + self.W], set()).add(t[s + self.W])

    def banned(self, pb):
        w = self.text[-self.W:]
        local = set()
        for c in pb:
            if len(w) >= self.W:
                k = w[-self.W:]
                if c in self.nx.get(k, ()) or (k, c) in local:
                    return True
                local.add((k, c))
            w = w + bytes([c])
        return False


def generate_one(args):
    arm, idx, prompt, cfg = args
    vocab, tid = G["vocab"], G["tid"]
    rng = np.random.default_rng(SEED + idx)
    toks = TOKRE.findall(prompt)
    ctx = [BOS] + [tid.get(t, UNK) for t in toks]
    prev = toks[-1] if toks else None
    rep = RepeatBan(prompt)
    out = ""
    for _ in range(300):
        p = mixed(arm, ctx[-60:], cfg).copy()
        p[UNK] = p[BOS] = 0
        p /= p.sum()
        top = np.argpartition(-p, 3000)[:3000]
        top = top[np.argsort(-p[top])]
        cum = np.cumsum(p[top])
        cand = top[:int(np.searchsorted(cum, 0.995)) + 1]
        pieces = [piece(vocab[c], prev) for c in cand]
        keep = np.array([not rep.banned(s.encode("utf-8")) for s in pieces])
        if not keep.any():
            keep[:] = True
        c2, pr = cand[keep], p[cand[keep]]
        pr = pr / pr.sum()
        k = int(np.searchsorted(np.cumsum(pr), 0.9)) + 1
        pr = pr[:k] / pr[:k].sum()
        j = int(rng.choice(k, p=pr))
        w = int(c2[j])
        s = piece(vocab[w], prev)
        out += s
        rep.add(s.encode("utf-8"))
        ctx.append(w)
        prev = vocab[w]
        if len(out.encode("utf-8")) >= 600:
            break
    b = out.encode("utf-8")[:600]
    return idx, b.decode("utf-8", "ignore")


def cont_bits(args):
    arm, prompt, cont, cfg = args
    tid, vocab = G["tid"], G["vocab"]
    pt, ct = TOKRE.findall(prompt), TOKRE.findall(cont)
    ctx = [BOS] + [tid.get(t, UNK) for t in pt]
    bits, nb, prev = 0.0, 0, pt[-1] if pt else None
    for t in ct:
        y = tid.get(t, UNK)
        p = mixed(arm, ctx[-60:], cfg)
        bits += -math.log2(max(p[y], 1e-12))
        nb += len(piece(t, prev).encode())
        ctx.append(y)
        prev = t
    return bits, nb


def gen_and_scribe(arm, cfg):
    rows = [l.split("\t") for l in open(f"{ROOT}/bench/eval/gen_prompts_dev.tsv", encoding="utf-8").read().split("\n")[1:] if l]
    if SMALL:
        rows = rows[:20]
    prompts, conts = [unesc(r[2]) for r in rows], [unesc(r[3]) for r in rows]
    os.makedirs(f"{WORK}/gen", exist_ok=True)
    gf = f"{WORK}/gen/{arm}.gen.txt"
    if not os.path.exists(gf + ".cont.json"):
        outs = [generate_one((arm, i, p, cfg))[1] for i, p in enumerate(prompts)]
        with open(gf, "w", encoding="utf-8") as f:
            f.write(f"MODEL\tfuzzy_bake\t{arm}\n")
            for i, p in enumerate(prompts):
                f.write(f"PROMPT\t{i}\t{esc(p)}\nGEN\ttrie_T1.0\t{esc(outs[i])}\n")
        json.dump([cont_bits((arm, p, c, cfg)) for p, c in zip(prompts, conts)], open(gf + ".cont.json", "w"))
    md = f"{WORK}/gen/{arm}.scribe.md"
    if not os.path.exists(md):
        with open(md, "w") as f:
            subprocess.run([SCRIBE, "ladder", "--pack", f"enwik:{E8}:16:48", "--ref", f"enwik:{E8}:48:56", "--test",
                            f"enwik:{E8}:56:64", "--vectors", SCRIBE_VEC, "--profile", "v2", "--seed", "20261027",
                            "--gen-trie", gf, "--label", arm], stdout=f, stderr=subprocess.DEVNULL, check=True)
    gens = [unesc(l.split("\t", 2)[2]) for l in open(gf, encoding="utf-8").read().split("\n") if l.startswith("GEN\t")]
    cb = json.load(open(gf + ".cont.json"))
    trainb = open(f"{WORK}/train.txt", "rb").read()
    m = re.search(r"\| trie samples \(pooled\) \| (\d+) \| ([0-9.]+) \[([0-9.]+), ([0-9.]+)\] \| [^|]+\| ([0-9.]+) \[([0-9.]+), ([0-9.]+)\]",
                  open(md).read())
    return dict(scribe=float(m.group(2)), scribe_ci=[float(m.group(3)), float(m.group(4))], kept=int(m.group(1)),
                surface_S=float(m.group(5)), copy32=copy_share(gens, window_hashes(np.frombuffer(trainb, np.uint8), 32)),
                verbatim13=float(np.mean([has_run(g, ngram_hashes(G["tr"], 13)) for g in gens])) if gens else 0.0,
                cont_bpb=sum(b for b, _ in cb) / sum(n for _, n in cb))


def window_hashes(b, w):
    if len(b) < w:
        return np.zeros(0, np.uint64)
    h = np.full(len(b) - w + 1, 0xcbf29ce484222325, np.uint64)
    with np.errstate(over="ignore"):
        for j in range(w):
            h = (h ^ b[j:len(b) - w + 1 + j].astype(np.uint64)) * np.uint64(0x100000001b3)
    return np.unique(h)


def copy_share(gens, index):
    cov, tot = 0, 0
    for g in gens:
        b = np.frombuffer(g.encode("utf-8"), np.uint8)
        tot += len(b)
        if len(b) < 32:
            continue
        h = np.full(len(b) - 31, 0xcbf29ce484222325, np.uint64)
        with np.errstate(over="ignore"):
            for j in range(32):
                h = (h ^ b[j:len(b) - 31 + j].astype(np.uint64)) * np.uint64(0x100000001b3)
        i = np.searchsorted(index, h)
        hit = (i < len(index)) & (index[np.minimum(i, len(index) - 1)] == h)
        m = np.zeros(len(b), bool)
        for s in np.flatnonzero(hit):
            m[s:s + 32] = True
        cov += int(m.sum())
    return cov / max(tot, 1)


def ngram_hashes(ids, n):
    h = np.full(len(ids) - n + 1, 0xcbf29ce484222325, np.uint64)
    with np.errstate(over="ignore"):
        for j in range(n):
            h = (h ^ ids[j:len(ids) - n + 1 + j].astype(np.uint64)) * np.uint64(0x100000001b3)
    return np.unique(h)


def has_run(g, index, n=13):
    ids = np.array([G["tid"].get(t, UNK) for t in TOKRE.findall(g)], np.int64)
    if len(ids) < n:
        return False
    h = ngram_hashes(ids, n)
    i = np.searchsorted(index, h)
    return bool(((i < len(index)) & (index[np.minimum(i, len(index) - 1)] == h)).any())


def report():
    res = {}
    for a in ARMS + ["ALL"]:
        f = f"{WORK}/res_{a}.json"
        if os.path.exists(f):
            res[a] = json.load(open(f))
    k5 = res.get("K5", {}).get("gen", {})
    lines = ["# Fuzzy bake-off results", "", f"Data: {json.load(open(f'{WORK}/data.json'))}", "",
             "| Arm | val bits/byte | minus K5 [95%] | weight | coverage | SCRIBE v2 [95%] | minus K5 | surface S | copy32 | 13-word runs | cont bits/byte | screen |",
             "|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|"]
    for a, r in res.items():
        s, g = r["score"], r["gen"]
        wt = s.get("lambda", "")
        ds = g["scribe"] - k5.get("scribe", g["scribe"])
        ok = a not in ("K5",) and s["ci"][1] < 0 and s["minus_K5"] <= -0.01 and ds >= 0.02 and g["copy32"] <= k5.get("copy32", 1) + 0.02
        lines.append(f"| {a} | {s['bpb']:.4f} | {s['minus_K5']:+.4f} [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}] | "
                     f"{wt if wt == '' else f'{wt:.3f}'} | {s['coverage']:.3f} | {g['scribe']:.3f} [{g['scribe_ci'][0]:.3f}, {g['scribe_ci'][1]:.3f}] | "
                     f"{ds:+.3f} | {g['surface_S']:.3f} | {g['copy32']:.4f} | {g['verbatim13']:.3f} | {g['cont_bpb']:.4f} | {'PASS' if ok else '-'} |")
    if "ALL" in res:
        lines += ["", "ALL weights: " + json.dumps({k: round(v, 4) for k, v in res["ALL"]["score"]["w"].items()})]
    open(f"{WORK}/summary.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    t0 = time.time()
    if cmd == "prep":
        prep()
        shared()
        log("prep done", round(time.time() - t0))
    elif cmd == "arm":
        arm = sys.argv[2]
        out = f"{WORK}/res_{arm}.json"
        if os.path.exists(out):
            return
        shared()
        names = ["K5"] + (ARMS[1:] if arm == "ALL" else ([] if arm == "K5" else [arm]))
        build(names)
        cfg = score(names, arm)
        log(arm, "scored", json.dumps({k: cfg[k] for k in ("bpb", "minus_K5", "ci")}))
        gen = gen_and_scribe(arm, cfg)
        json.dump(dict(arm=arm, score=cfg, gen=gen, seconds=time.time() - t0, F10_categories=G.get("F10_categories")),
                  open(out, "w"), indent=1)
        log(arm, "done", json.dumps(gen), round(time.time() - t0), "s")
    elif cmd == "report":
        report()
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
