#!/usr/bin/env python3
"""Self-organising structure without a network (docs/GENERATION_PLAN.md §13.9). Odin Loch, 2026-11-01.

A neural network reshapes its whole structure to predict the data. Here a count model does the same by local moves,
each kept only if it lowers bits: words move between classes (the exchange algorithm), contexts are kept or dropped by
the bits they save net of their own cost (minimum description length), and categories split where they predict
wrongly (fuzzy ARTMAP's match tracking). Same data, vocabulary, sampler and SCRIBE ladder as bench/fuzzy/fuzzy_bake.py
(its prep must exist in FZ_WORK). Each job runs on one core.

Jobs (outputs in FZ_WORK/so/):
  so1x so1k   SO1: K5 + a class trigram and class-substitution contexts, with exchange classes / k-means classes
  so2p so2m   SO2: at the same stored-pair budget (30% of K5's), K5 pruned by count / contexts chosen by bits saved
  so3a so3k   SO3: K5 + categories by fuzzy ARTMAP (error-driven) / by k-means with the same number of categories
  so4         SO4: the whole learner (exchange classes, contexts chosen by bits, ARTMAP), alone and with K5
  curve1 curve2 curve4   SO4: the learning curve, K5 and the whole learner on the first 1 / 2 / 4 MB (8 MB is so4)
  so5         SO5: a shift to another domain (SO_SHIFT, plain text): fixed structure against re-organised structure
  report      FZ_WORK/so/summary.md
SO6 (writing) is the SCRIBE read of the generating jobs (so1x so1k so2p so2m so3a so3k so4).
"""
import json
import math
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fuzzy_bake as fb  # noqa: E402

UNK, BOS, SEED = fb.UNK, fb.BOS, fb.SEED
log = fb.log
SOW = f"{fb.WORK}/so"
G = fb.G


def xlogx(x):
    return x * np.log(np.maximum(x, 1e-300))


# ------------------------------------------------------------------------------------------- SO1: exchange classes
def exchange(tr, V, cls0, K, passes=6, min_count=3):
    """Martin, Liermann and Ney's exchange algorithm: move each word to the class that most raises the class-bigram
    likelihood. Returns the classes and the likelihood after each pass."""
    cls = cls0.copy()
    a, b = tr[:-1].astype(np.int64), tr[1:].astype(np.int64)
    uk, cnt = np.unique(a * V + b, return_counts=True)
    pa, pb, cnt = uk // V, uk % V, cnt.astype(np.float64)
    o = np.argsort(pa, kind="stable")
    out_off, out_b, out_c = np.searchsorted(pa[o], np.arange(V + 1)), pb[o], cnt[o]
    i2 = np.argsort(pb, kind="stable")
    in_off, in_a, in_c = np.searchsorted(pb[i2], np.arange(V + 1)), pa[i2], cnt[i2]
    KK = K + 2
    M = np.zeros((KK, KK))
    np.add.at(M, (cls[pa], cls[pb]), cnt)
    L, R = M.sum(1), M.sum(0)
    freq = np.bincount(tr, minlength=V)
    words = [int(w) for w in np.argsort(-freq) if freq[w] >= min_count and w not in (UNK, BOS)]
    ll = lambda: float(xlogx(M).sum() - xlogx(L).sum() - xlogx(R).sum())
    hist = [ll()]
    for ps in range(passes):
        moved = 0
        for w in words:
            ob, oc = out_b[out_off[w]:out_off[w + 1]], out_c[out_off[w]:out_off[w + 1]]
            ia, ic = in_a[in_off[w]:in_off[w + 1]], in_c[in_off[w]:in_off[w + 1]]
            so, si = ob != w, ia != w
            selfc = float(oc[~so].sum())
            rw = np.bincount(cls[ob[so]], weights=oc[so], minlength=KK)
            cw = np.bincount(cls[ia[si]], weights=ic[si], minlength=KK)
            nl, nr, a0 = float(oc.sum()), float(ic.sum()), int(cls[w])
            M[a0, :] -= rw
            M[:, a0] -= cw
            M[a0, a0] -= selfc
            L[a0] -= nl
            R[a0] -= nr
            F = xlogx(M)
            gain = ((xlogx(M[:K] + rw[None, :]) - F[:K]).sum(1) + (xlogx(M[:, :K] + cw[:, None]) - F[:, :K]).sum(0)
                    - (xlogx(L[:K] + nl) - xlogx(L[:K])) - (xlogx(R[:K] + nr) - xlogx(R[:K])))
            k = int(np.argmax(gain))
            M[k, :] += rw
            M[:, k] += cw
            M[k, k] += selfc
            L[k] += nl
            R[k] += nr
            moved += k != a0
            cls[w] = k
        hist.append(ll())
        log("exchange pass", ps, "moved", moved, "loglik", round(hist[-1]))
    return cls, hist


class ClassTri:
    """P(w | h) = P(c_t | c_t-1, c_t-2) P(w | c_t): Kneser-Ney over class ids, words within a class by frequency."""

    def __init__(self, tr, cls, V, K):
        self.cls = cls
        self.kn = fb.KN(cls[tr].astype(np.int32), K + 2, order=3)
        cnt = np.bincount(tr, minlength=V).astype(np.float64) + 0.01
        ctot = np.bincount(cls, weights=cnt, minlength=K + 2)
        self.pwc = cnt / ctot[cls]

    def dist(self, ctx):
        p = self.kn.dist([int(self.cls[w]) for w in ctx[-2:]])[self.cls] * self.pwc
        return p / p.sum()


class SubBank(fb.Bank):
    def __init__(self, ids, cls, pats, L=4):
        n = len(ids)
        self.L, self.cls = L, cls
        words = [ids[L - i - 1:n - i - 1].astype(np.int64) for i in range(L)]
        cl = [cls[w] for w in words]
        self.tab = {p: fb.Table(fb.pack(fb.pat_vals(p, words, cl)), ids[L:]) for p in pats}


F1_PATS = [tuple(fb.C_ if i == j else fb.W_ for i in range(L)) + (fb.N_,) * (4 - L) for L in (2, 3, 4) for j in range(L)]


# ------------------------------------------------------------------------------------------- SO2: contexts by bits
def filter_table(t, keep):
    nt = fb.Table.__new__(fb.Table)
    lens = np.diff(t.off)
    pm = np.repeat(keep, lens)
    nt.keys, nt.ids, nt.cnt, nt.tot = t.keys[keep], t.ids[pm], t.cnt[pm], t.tot[keep]
    nt.off = np.concatenate(([0], np.cumsum(lens[keep]))).astype(np.int64)
    return nt, pm


def pruned_kn(kn, budget):
    """K5 with its contexts kept by count (most frequent first, every order) until `budget` stored pairs."""
    rows = []
    for k, t in kn.T.items():
        lens = np.diff(t.off)
        rows.append(np.stack([t.tot, lens, np.full(len(t.tot), k), np.arange(len(t.tot))], 1))
    R = np.concatenate(rows)
    R = R[np.argsort(-R[:, 0], kind="stable")]
    keep_rows = R[np.cumsum(R[:, 1]) <= budget]
    p = fb.KN.__new__(fb.KN)
    p.V, p.order, p.p1 = kn.V, kn.order, kn.p1
    p.T, p.disc, p.gam = {}, {}, {}
    for k, t in kn.T.items():
        keep = np.zeros(len(t.keys), bool)
        keep[keep_rows[keep_rows[:, 2] == k, 3].astype(np.int64)] = True
        p.T[k], pm = filter_table(t, keep)
        p.disc[k], p.gam[k] = kn.disc[k][pm], kn.gam[k][keep]
    return p, int(keep_rows[:, 1].sum())


def kn_pairs(kn):
    return int(sum(len(t.ids) for t in kn.T.values()))


def mdl_select(bank, budget, p_uni, V):
    """Contexts kept by minimum description length: the bits saved by coding a context's next words with its own
    distribution instead of the unigram, minus the bits to store it (32 + k * (log2 V + 8) for k next words). The
    contexts with positive net saving are taken, the most saving per stored pair first, until `budget` pairs."""
    logpu = np.log2(p_uni)
    rows, pats = [], list(bank.tab)
    for j, p in enumerate(pats):
        t = bank.tab[p]
        per = np.repeat(np.arange(len(t.keys)), np.diff(t.off))
        q = t.cnt / t.tot[per]
        g = np.bincount(per, weights=t.cnt * (np.log2(q) - logpu[t.ids]), minlength=len(t.keys))
        k = np.diff(t.off)
        net = g - (32 + k * (math.log2(V) + 8))
        m = net > 0
        rows.append(np.stack([net[m] / k[m], k[m], np.full(m.sum(), j), np.flatnonzero(m)], 1))
    R = np.concatenate(rows)
    R = R[np.argsort(-R[:, 0], kind="stable")]
    R = R[np.cumsum(R[:, 1]) <= budget]
    sel = {}
    for j, p in enumerate(pats):
        keep = np.zeros(len(bank.tab[p].keys), bool)
        keep[R[R[:, 2] == j, 3].astype(np.int64)] = True
        sel[p] = keep
    return sel, int(R[:, 1].sum())


class FilteredBank(fb.Bank):
    def __init__(self, bank, sel):
        self.L, self.cls = bank.L, bank.cls
        self.tab = {p: filter_table(bank.tab[p], sel[p])[0] for p in sel}


class WithFloor:
    """A pattern mixture with a 1% unigram floor, so it always gives a distribution."""

    def __init__(self, pm, p_uni):
        self.pm, self.pu = pm, p_uni

    def dist(self, ctx):
        d = self.pm.dist(ctx)
        return self.pu.copy() if d is None else 0.99 * d + 0.01 * self.pu

    def fit(self, ctxs, ys):
        self.pm.fit(ctxs, ys)


# ------------------------------------------------------------------------------------------- SO3: ARTMAP
def art_input(E, w1, w2, d=32):
    f = np.concatenate([E[w1][..., :d], E[w2][..., :d]], -1)
    f = np.clip((f * 2.5 + 1) / 2, 0, 1)
    return np.concatenate([f, 1 - f], -1).astype(np.float32)


class CategoryCounts:
    """Categories (any assignment of contexts to category ids) holding next-word counts; a context reads its 3 best
    categories, weighted by score."""

    def __init__(self, V, cat, nxt):
        self.V = V
        self.tab = fb.Table(cat.astype(np.uint64), nxt)

    def from_scores(self, sc):
        p, z = np.zeros(self.V), 0.0
        for j in np.argsort(-sc)[:3]:
            i = self.tab.find(np.uint64(int(j)))
            if i >= 0 and sc[j] > 0:
                self.tab.dense_into(i, p, float(sc[j]))
                z += float(sc[j])
        return p / z if z > 0 else None


class ARTMAP:
    """Fuzzy ART with match tracking on the next word: a category that passes vigilance but predicts the true next word
    worse than the unigram (after 20 examples) is skipped, vigilance rises above its match, and the search goes on; if
    nothing fits, a new category is made. So categories split where they predict wrongly."""

    def __init__(self, ids, E, V, p_uni, rho=0.7, alpha=0.001, learn=150000, count=600000, maxcat=3000, min_n=20,
                 seed=SEED, error_driven=True):
        self.E, self.V, self.alpha = E, V, alpha
        n = len(ids)
        rng = np.random.default_rng(seed)
        Wa = np.zeros((maxcat, 128), np.float32)
        cnts, tots, J, splits = [], [], 0, 0
        for t in np.sort(rng.choice(np.arange(2, n), min(learn, n - 2), replace=False)):
            I, y = art_input(E, ids[t - 1], ids[t - 2]), int(ids[t])
            Isum = float(I.sum())
            placed = False
            if J:
                mn = np.minimum(I, Wa[:J]).sum(1)
                T = mn / (alpha + Wa[:J].sum(1))
                match = mn / Isum
                cand = np.flatnonzero(match >= rho)
                rho_cur = rho
                for j in cand[np.argsort(-T[cand])]:
                    if match[j] < rho_cur:
                        continue
                    if error_driven and tots[j] >= min_n and cnts[j].get(y, 0) / tots[j] < p_uni[y]:
                        rho_cur = match[j] + 1e-6  # match tracking
                        splits += 1
                        continue
                    Wa[j] = np.minimum(I, Wa[j])
                    cnts[j][y] = cnts[j].get(y, 0) + 1
                    tots[j] += 1
                    placed = True
                    break
            if not placed and J < maxcat:
                Wa[J] = I
                cnts.append({y: 1})
                tots.append(1)
                J += 1
        self.W, self.ncat, self.splits = Wa[:J].copy(), J, splits
        sel = np.sort(rng.choice(np.arange(2, n), min(count, n - 2), replace=False))
        cat = np.zeros(len(sel), np.int64)
        for a in range(0, len(sel), 64):
            t = sel[a:a + 64]
            I = art_input(E, ids[t - 1], ids[t - 2])
            cat[a:a + 64] = np.argmax(np.minimum(I[:, None, :], self.W[None]).sum(2) / (alpha + self.W.sum(1)), 1)
        self.cc = CategoryCounts(V, cat, ids[sel])

    def dist(self, ctx):
        c = (list(ctx[::-1][:2]) + [BOS] * 2)[:2]
        I = art_input(self.E, c[0], c[1])
        return self.cc.from_scores(np.minimum(I, self.W).sum(1) / (self.alpha + self.W.sum(1)))


class KMeansCats:
    """The similarity baseline: k-means (cosine) on the same context features, with K categories."""

    def __init__(self, ids, E, V, K, learn=150000, count=600000, seed=SEED, iters=12):
        self.E, self.V = E, V
        n = len(ids)
        rng = np.random.default_rng(seed)
        t = np.sort(rng.choice(np.arange(2, n), min(learn, n - 2), replace=False))
        X = self._feat(ids[t - 1], ids[t - 2])
        C = X[rng.choice(len(X), K, replace=False)].copy()
        for _ in range(iters):
            a = np.argmax(X @ C.T, 1)
            S = np.zeros_like(C)
            np.add.at(S, a, X)
            nz = np.linalg.norm(S, axis=1) > 0
            C[nz] = S[nz] / np.linalg.norm(S[nz], axis=1, keepdims=True)
        self.C = C
        sel = np.sort(rng.choice(np.arange(2, n), min(count, n - 2), replace=False))
        cat = np.concatenate([np.argmax(self._feat(ids[s - 1], ids[s - 2]) @ C.T, 1)
                              for s in np.array_split(sel, max(1, len(sel) // 4096))])
        self.cc = CategoryCounts(V, cat, ids[sel])

    def _feat(self, w1, w2):
        f = np.concatenate([self.E[w1][..., :32], self.E[w2][..., :32]], -1)
        return f / (np.linalg.norm(f, axis=-1, keepdims=True) + 1e-9)

    def dist(self, ctx):
        c = (list(ctx[::-1][:2]) + [BOS] * 2)[:2]
        return self.cc.from_scores(np.maximum(self.C @ self._feat(c[0], c[1]), 0))


# ------------------------------------------------------------------------------------------- mixtures, scoring
class Mix:
    """Experts mixed by weights fit on held-out text (sleeping-experts EM); `None` from an expert means unavailable."""

    def __init__(self, experts):
        self.ex = experts
        self.w = np.ones(len(experts)) / len(experts)

    def dist(self, ctx):
        acc, z = None, 0.0
        for w, e in zip(self.w, self.ex):
            d = e.dist(ctx)
            if d is not None:
                acc = w * d if acc is None else acc + w * d
                z += w
        return acc / z

    def fit(self, ctxs, ys):
        Q, A = np.zeros((len(ys), len(self.ex))), np.zeros((len(ys), len(self.ex)), bool)
        for r, (c, y) in enumerate(zip(ctxs, ys)):
            for j, e in enumerate(self.ex):
                d = e.dist(c)
                if d is not None:
                    Q[r, j], A[r, j] = d[y], True
        self.w = fb.sleeping_em(Q, A)


def val_split(n_eval=None):
    va, vdoc, pos = fb.val_positions()
    ctx = fb.contexts(va, pos)
    fitm = vdoc[pos] % 2 == 0
    fi, ev = np.flatnonzero(fitm)[:4000], np.flatnonzero(~fitm)
    if n_eval:
        ev = ev[:n_eval]
    vocab = G["vocab"]
    nb = np.array([len(fb.piece(vocab[va[t]], vocab[va[t - 1]] if va[t - 1] != BOS else None).encode()) for t in pos])
    return dict(fit_ctx=[ctx[i] for i in fi], fit_y=[int(va[pos[i]]) for i in fi], ev_ctx=[ctx[i] for i in ev],
                ev_y=np.array([int(va[pos[i]]) for i in ev]), ev_doc=vdoc[pos][ev], ev_bytes=nb[ev])


def bits_of(pred, ctxs, ys):
    return np.array([-math.log2(max(pred.dist(c)[y], 1e-12)) for c, y in zip(ctxs, ys)])


def per_doc(bits, vs):
    ud = np.unique(vs["ev_doc"])
    return {int(u): [float(bits[vs["ev_doc"] == u].sum()), int(vs["ev_bytes"][vs["ev_doc"] == u].sum())] for u in ud}


def paired(a, b, seed=SEED):
    """b - a in bits per byte, bootstrap over documents (per_doc dicts)."""
    ud = sorted(set(a) & set(b))
    arr = np.array([(a[u][0], b[u][0], a[u][1]) for u in ud])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(2000):
        s = arr[rng.integers(0, len(ud), len(ud))].sum(0)
        bs.append((s[1] - s[0]) / s[2])
    tot = arr.sum(0)
    return float((tot[1] - tot[0]) / tot[2]), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def finish(label, pred, vs, extra, generate=True):
    bits = bits_of(pred, vs["ev_ctx"], vs["ev_y"])
    res = dict(label=label, bpb=float(bits.sum() / vs["ev_bytes"].sum()), per_doc=per_doc(bits, vs), **extra)
    if generate:
        fb.mixed = lambda arm, ctx, cfg: pred.dist(ctx)  # the sampler and the continuation scorer read this predictor
        res["gen"] = fb.gen_and_scribe(f"so_{label}", {})
    json.dump(res, open(f"{SOW}/res_{label}.json", "w"), indent=1)
    log(label, "done", json.dumps({k: v for k, v in res.items() if k != "per_doc"}))


def k5_pred():
    class P:
        def dist(self, ctx):
            return G["K5"].dist(ctx)
    return P()


def uni(tr, V):
    c = np.bincount(tr, minlength=V).astype(np.float64) + 0.01
    return c / c.sum()


# ------------------------------------------------------------------------------------------- the jobs
def job_so1(which):
    tr, V, E, cls0 = G["tr"], G["V"], G["E"], G["cls"]
    G["K5"] = fb.KN(tr, V)
    vs = val_split()
    extra = {}
    if which == "x":
        cls, hist = exchange(tr, V, cls0, 512)
        extra["exchange_loglik"] = hist
    else:
        cls = cls0
    mix = Mix([k5_pred(), ClassTri(tr, cls, V, 512), fb.PatternMix(SubBank(tr, cls, F1_PATS), F1_PATS, V)])
    mix.ex[2].fit(vs["fit_ctx"], vs["fit_y"])
    mix.fit(vs["fit_ctx"], vs["fit_y"])
    extra["weights"] = mix.w.tolist()
    finish("so1" + which, mix, vs, extra)


def job_so2(which):
    tr, V, cls = G["tr"], G["V"], G["cls"]
    kn = fb.KN(tr, V)
    budget = int(0.3 * kn_pairs(kn))
    vs = val_split()
    extra = dict(K5_pairs=kn_pairs(kn), budget=budget)
    if which == "p":
        pred_kn, used = pruned_kn(kn, budget)
        G["K5"] = pred_kn
        pred = k5_pred()
    else:
        bank = SubBank(tr, cls, fb.all_patterns(4))
        sel, used = mdl_select(bank, budget, uni(tr, V), V)
        fbk = FilteredBank(bank, sel)
        extra["contexts_by_pattern"] = {"".join(p): int(sel[p].sum()) for p in sel if sel[p].sum()}
        pred = WithFloor(fb.PatternMix(fbk, list(sel), V), uni(tr, V))
        pred.fit(vs["fit_ctx"], vs["fit_y"])
    extra["pairs_used"] = used
    finish("so2" + which, pred, vs, extra)


def job_so3(which):
    tr, V, E = G["tr"], G["V"], G["E"]
    G["K5"] = fb.KN(tr, V)
    vs = val_split()
    art = ARTMAP(tr, E, V, uni(tr, V))
    extra = dict(categories=art.ncat, match_tracking_splits=art.splits)
    cat = art if which == "a" else KMeansCats(tr, E, V, art.ncat)
    mix = Mix([k5_pred(), cat])
    mix.fit(vs["fit_ctx"], vs["fit_y"])
    extra["weights"] = mix.w.tolist()
    finish("so3" + which, mix, vs, extra)


def whole_learner(tr, V, E, cls0, vs, passes=6):
    """SO4: exchange classes, then every pattern's contexts chosen by bits under 30% of K5's pairs, then ARTMAP."""
    kn = fb.KN(tr, V)
    budget = int(0.3 * kn_pairs(kn))
    cls, hist = exchange(tr, V, cls0, 512, passes=passes)
    bank = SubBank(tr, cls, fb.all_patterns(4))
    sel, used = mdl_select(bank, budget, uni(tr, V), V)
    pm = fb.PatternMix(FilteredBank(bank, sel), list(sel), V)
    pm.fit(vs["fit_ctx"], vs["fit_y"])
    art = ARTMAP(tr, E, V, uni(tr, V))
    alone = Mix([WithFloor(pm, uni(tr, V)), art])
    alone.fit(vs["fit_ctx"], vs["fit_y"])
    G["K5"] = kn
    withk = Mix([k5_pred(), pm, art])
    withk.fit(vs["fit_ctx"], vs["fit_y"])
    info = dict(K5_pairs=kn_pairs(kn), budget=budget, pairs_used=used, categories=art.ncat,
                match_tracking_splits=art.splits, exchange_loglik=hist, w_alone=alone.w.tolist(), w_withK5=withk.w.tolist())
    return alone, withk, info


def job_so4():
    tr, V, E, cls0 = G["tr"], G["V"], G["E"], G["cls"]
    vs = val_split()
    alone, withk, info = whole_learner(tr, V, E, cls0, vs)
    finish("so4", alone, vs, info)
    finish("so4k", withk, vs, info)


def job_curve(mb):
    tr, V, E, cls0 = G["tr"], G["V"], G["E"], G["cls"]
    bos = np.flatnonzero(tr == BOS)
    frac = mb / max(1e-9, json.load(open(f"{fb.WORK}/data.json"))["train_bytes"] / 2 ** 20)
    cut = int(bos[min(int(np.searchsorted(bos, int(len(tr) * frac))), len(bos) - 1)]) if frac < 1 else len(tr)
    sub = tr[:cut]
    E = fb.embeddings(sub, V)  # vectors and classes learned from the same prefix, so smaller sizes borrow nothing
    cls0 = fb.kmeans(E, 512, SEED)
    G["E"] = E
    vs = val_split(n_eval=10000)
    alone, withk, info = whole_learner(sub, V, E, cls0, vs, passes=4)
    k5 = k5_pred()
    for label, pred in ((f"curve{mb}_K5", k5), (f"curve{mb}_so4", alone), (f"curve{mb}_so4k", withk)):
        finish(label, pred, vs, dict(mb=mb, train_tokens=int(cut), **info), generate=False)


def job_so5():
    """A domain shift: the shift text (SO_SHIFT) arrives in 8 chunks. Each chunk is scored, then learned. FIXED keeps
    the wiki classes and the wiki's chosen contexts and only updates their counts; REORG re-runs 2 exchange passes and
    re-chooses the contexts by bits each time. K5 alone (updated counts) is shown beside them."""
    tr, V, E, cls0 = G["tr"], G["V"], G["E"], G["cls"]
    tid = G["tid"]
    text = open(os.environ["SO_SHIFT"], encoding="utf-8", errors="ignore").read()
    ids = np.array([tid.get(t, UNK) for t in fb.TOKRE.findall(text)], np.int32)
    chunks = np.array_split(ids, 8)
    vs = val_split(n_eval=2000)
    kn = fb.KN(tr, V)
    budget = int(0.3 * kn_pairs(kn))
    pats = fb.all_patterns(4)
    cls_fixed, _ = exchange(tr, V, cls0, 512, passes=4)
    bank = SubBank(tr, cls_fixed, pats)
    sel0, _ = mdl_select(bank, budget, uni(tr, V), V)
    keys0 = {p: bank.tab[p].keys[sel0[p]] for p in pats}
    pm = fb.PatternMix(FilteredBank(bank, sel0), pats, V)
    pm.fit(vs["fit_ctx"], vs["fit_y"])
    wts = pm.w.copy()
    systems = {}

    def make(data, cls, reselect):
        b = SubBank(data, cls, pats)
        if reselect:
            s, _ = mdl_select(b, budget, uni(data, V), V)
        else:
            s = {p: np.isin(b.tab[p].keys, keys0[p]) for p in pats}
        m = fb.PatternMix(FilteredBank(b, s), pats, V)
        m.w = wts
        return m

    G["K5"] = kn
    m0 = Mix([k5_pred(), pm])
    m0.fit(vs["fit_ctx"], vs["fit_y"])
    systems = dict(FIXED=Mix([k5_pred(), pm]), REORG=Mix([k5_pred(), pm]), K5=k5_pred())
    for s in ("FIXED", "REORG"):
        systems[s].w = m0.w.copy()
    cls_re = cls_fixed.copy()
    data = tr
    curve = {s: [] for s in systems}
    rng = np.random.default_rng(SEED)
    for i, ch in enumerate(chunks):
        pos = np.sort(rng.choice(np.arange(8, len(ch)), min(3000, len(ch) - 8), replace=False))
        ctxs = [[int(x) for x in ch[max(0, t - 60):t]] for t in pos]
        ys = ch[pos]
        nb = sum(len(fb.piece(G["vocab"][ch[t]], G["vocab"][ch[t - 1]]).encode()) for t in pos)
        for s, pred in systems.items():
            curve[s].append(float(bits_of(pred, ctxs, ys).sum() / nb))
        log("so5 chunk", i, {s: round(v[-1], 4) for s, v in curve.items()})
        data = np.concatenate([data, [BOS], ch]).astype(np.int32)
        G["K5"] = fb.KN(data, V)
        systems["FIXED"].ex[1] = make(data, cls_fixed, False)
        cls_re, _ = exchange(data, V, cls_re, 512, passes=2)
        systems["REORG"].ex[1] = make(data, cls_re, True)
    res = dict(label="so5", curve=curve, mean={s: float(np.mean(v)) for s, v in curve.items()},
               shift_tokens=int(len(ids)), shift_oov=float(np.mean(ids == UNK)))
    json.dump(res, open(f"{SOW}/res_so5.json", "w"), indent=1)
    log("so5 done", json.dumps(res["mean"]))


# ------------------------------------------------------------------------------------------- report
def report():
    R = {}
    for f in sorted(os.listdir(SOW)):
        if f.startswith("res_") and f.endswith(".json"):
            R[f[4:-5]] = json.load(open(f"{SOW}/{f}"))
    k5 = json.load(open(f"{fb.WORK}/res_K5.json")) if os.path.exists(f"{fb.WORK}/res_K5.json") else None
    L = ["# Self-organising structure: results (GENERATION_PLAN §13.9)", ""]

    def row(a, b, name):
        if a in R and b in R:
            d, ci = paired(R[a]["per_doc"], R[b]["per_doc"])
            L.append(f"| {name} | {R[a]['bpb']:.4f} | {R[b]['bpb']:.4f} | {d:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] |")

    L += ["## Bits per byte on held-out prose (paired over documents; negative favours the second)", "",
          "| Comparison | first | second | second − first [95%] |", "|---|---:|---:|---|"]
    row("so1k", "so1x", "SO1: k-means classes → exchange classes (both with K5)")
    row("so2p", "so2m", "SO2: K5 pruned by count → contexts chosen by bits (same pair budget)")
    row("so3k", "so3a", "SO3: k-means categories → ARTMAP categories (both with K5)")
    for mb in (1, 2, 4):
        row(f"curve{mb}_K5", f"curve{mb}_so4k", f"SO4 curve {mb} MB: K5 → whole learner + K5")
        row(f"curve{mb}_K5", f"curve{mb}_so4", f"SO4 curve {mb} MB: K5 → whole learner alone (30% of K5's pairs)")
    L += ["", "## Writing (SO6): SCRIBE v2 on GEN-A's 250 dev prompts", "",
          "| Arm | val bits/byte | SCRIBE v2 [95%] | surface S | copy32 | 13-word runs | cont bits/byte |", "|---|---:|---|---:|---:|---:|---:|"]
    if k5:
        g = k5["gen"]
        L.append(f"| K5 (bake-off) | {k5['score']['bpb']:.4f} | {g['scribe']:.3f} [{g['scribe_ci'][0]:.3f}, {g['scribe_ci'][1]:.3f}] | "
                 f"{g['surface_S']:.3f} | {g['copy32']:.4f} | {g['verbatim13']:.3f} | {g['cont_bpb']:.4f} |")
    for a, r in R.items():
        if "gen" in r:
            g = r["gen"]
            L.append(f"| {a} | {r['bpb']:.4f} | {g['scribe']:.3f} [{g['scribe_ci'][0]:.3f}, {g['scribe_ci'][1]:.3f}] | "
                     f"{g['surface_S']:.3f} | {g['copy32']:.4f} | {g['verbatim13']:.3f} | {g['cont_bpb']:.4f} |")
    if "so5" in R:
        L += ["", "## SO5: the domain shift (bits per byte on each chunk before it is learned)", "",
              "| System | " + " | ".join(f"c{i + 1}" for i in range(8)) + " | mean |", "|---|" + "---:|" * 9]
        for s, v in R["so5"]["curve"].items():
            L.append(f"| {s} | " + " | ".join(f"{x:.4f}" for x in v) + f" | {np.mean(v):.4f} |")
    L += ["", "## Details", ""]
    for a, r in R.items():
        L.append(f"- {a}: " + json.dumps({k: v for k, v in r.items() if k not in ('per_doc', 'gen', 'curve')})[:600])
    open(f"{SOW}/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    job = sys.argv[1] if len(sys.argv) > 1 else ""
    os.makedirs(SOW, exist_ok=True)
    if job == "report":
        return report()
    out = f"{SOW}/res_{'so5' if job == 'so5' else job}.json"
    if job.startswith("so") and os.path.exists(out) and job not in ("so4",):
        return
    fb.shared()
    t0 = time.time()
    if job in ("so1x", "so1k"):
        job_so1(job[-1])
    elif job in ("so2p", "so2m"):
        job_so2(job[-1])
    elif job in ("so3a", "so3k"):
        job_so3(job[-1])
    elif job == "so4":
        job_so4()
    elif job.startswith("curve"):
        job_curve(int(job[5:]))
    elif job == "so5":
        job_so5()
    else:
        print(__doc__)
        sys.exit(2)
    log(job, "finished", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
