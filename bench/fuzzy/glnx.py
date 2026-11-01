#!/usr/bin/env python3
"""GLNX: a gated linear network on top of CyphaLM's whole model (docs/GENERATION_PLAN.md §13.10). Odin Loch, 2026-11-01.

Odin: "Use my whole model. It's meant to be the first layer." Layer 1 is CyphaLM's default stack as `context8k` runs
it: the trie's 5 experts and match, the trie mixture, byte Kneser-Ney, InfiniGram at 5 caps, class, similar-context,
copy, the document parts, and the content mixer's chain after each stage. `cyphalm_trie context8k ... --dump-experts
FILE` writes each one's probability of the true byte, with its bucket, for every byte of the 1 MiB wiki holdout. Layer 2
learns online on top: gated linear mixers (Odin: linear content mixers worked best), each gate chosen by half-spaces of a
side-information vector (the experts' buckets, match lengths, recent losses, surprise, the previous byte's class, the
position in the word), weights on the simplex learned by exponentiated gradient on log loss. Every gate starts at the
content mixer's final output, so the network starts exactly where CyphaLM is.

The holdout's first half is where the content mixer fit its lambdas: layer 2 learns there too, then reads the second
half prequentially (predict, score, learn). C0 is the content mixer itself on the second half.

Usage: glnx.py DUMP [--fast]   (--fast: 64 KiB of each half, to profile and iterate)
"""
import json
import math
import sys
import time

import numpy as np

COLS = ["y", "uni", "o1", "bo2", "bo3", "bo4", "match_hit", "match_on", "trie_mix", "bkn", "bkn_b", "ig4", "ig16", "ig64",
        "ig512", "ig8192", "ig_b", "ig_n", "cls", "cls_b", "sim", "sim_b", "copy_hit", "copy_b", "copy_n", "trig", "trig_b",
        "cache", "cache_b", "topic", "topic_b", "chain_ig", "chain_cls", "chain_sim", "chain_copy", "chain"]
EXPERTS = ["uni", "o1", "bo2", "bo3", "bo4", "match_hit", "trie_mix", "bkn", "ig4", "ig16", "ig64", "ig512", "ig8192", "cls",
           "sim", "copy_hit", "trig", "cache", "topic", "chain_ig", "chain_cls", "chain_sim", "chain_copy", "chain"]
TRACK = ["trie_mix", "bkn", "ig8192", "cls", "sim", "cache", "chain"]  # experts whose recent loss is side information


def load(path, fast):
    D = np.fromfile(path, np.float32).reshape(-1, len(COLS)).astype(np.float64)
    H = len(D)
    half = H // 2
    if fast:
        return D[:65536], D[half:half + 65536], half
    return D[:half], D[half:], half


def byte_class(b):
    c = chr(int(b))
    return 0 if c.isspace() else 1 if c.isalpha() else 2 if c.isdigit() else 3 if c in "[]{}<>|&=" else 4


class GatedLin:
    """p = w[g] . q over the experts' probabilities of the true byte (all a linear mixture needs), w[g] on the simplex,
    g from m half-spaces of the side vector; exponentiated gradient on log loss."""

    def __init__(self, K, D, m, lr, init, seed):
        rng = np.random.default_rng(seed)
        self.H, self.th = rng.standard_normal((m, D)), rng.standard_normal(m) * 0.3
        self.w = np.tile(init / init.sum(), (2 ** m, 1))
        self.lr, self.pw, self.n = lr, 2 ** np.arange(m), np.zeros(2 ** m, int)

    def predict(self, q, s):
        self.g = int(((self.H @ s > self.th) * self.pw).sum()) if len(s) else 0
        self.q = q
        self.p = float(self.w[self.g] @ q)
        return self.p

    def learn(self):
        w = self.w[self.g] * np.exp(self.lr * np.minimum(self.q / max(self.p, 1e-300), 30.0))
        w = np.maximum(w / w.sum(), 1e-7)
        self.w[self.g] = w / w.sum()
        self.n[self.g] += 1


class OnlineChain:
    """The content mixer exactly as CyphaLM chains it (trie mixture, then Kneser-Ney, InfiniGram 8192, class,
    similar-context, copy, the document cache, each mixed into the running mixture with a lambda per bucket), but with
    every lambda learning online instead of frozen (Odin: don't freeze the mixer). Each stage's lambda follows the
    gradient of its own stage's log loss, as the frozen fit chose it, with a step that shrinks per bucket (AdaGrad)."""
    STAGES = [("bkn", "bkn_b"), ("ig8192", "ig_b"), ("cls", "cls_b"), ("sim", "sim_b"), ("copy_hit", "copy_b"),
              ("cache", "cache_b")]

    def __init__(self, lr=0.05, nb=64):
        self.lam = np.full((len(self.STAGES), nb), 0.05)
        self.g2 = np.full((len(self.STAGES), nb), 1e-3)
        self.lr = lr
        self.ix = [(COLS.index(e), COLS.index(b)) for e, b in self.STAGES]

    def predict(self, q, s, small, r=None):
        p = r[COLS.index("trie_mix")]
        self.trace = []
        for j, (ie, ib) in enumerate(self.ix):
            b = int(r[ib]) % self.lam.shape[1]
            pe = r[ie]
            pn = self.lam[j, b] * pe + (1 - self.lam[j, b]) * p
            self.trace.append((j, b, pe, p, pn))
            p = pn
        return p

    def learn(self):
        for j, b, pe, prev, pn in self.trace:
            g = (pe - prev) / max(pn, 1e-12)  # d log p_stage / d lambda
            self.g2[j, b] += g * g
            self.lam[j, b] = min(0.995, max(0.0, self.lam[j, b] + self.lr * g / math.sqrt(self.g2[j, b])))


class Net:
    """kind: 'one' (one layer, gated by a small context: InfiniGram bucket x previous byte class, as gate24 gates);
    'two' (8 neurons gated by half-spaces of the side vector, then a gated layer 2 over them and the content mixer)."""

    def __init__(self, kind, D, m=4, lr=0.05, neurons=8, seed=0):
        K = len(EXPERTS)
        start = np.full(K, 0.01)
        start[EXPERTS.index("chain")] = 1.0  # start at the content mixer
        self.kind = kind
        if kind == "one":
            self.units = [GatedLin(K, 0, 0, lr, start, seed + j) for j in range(16 * 5)]
        else:
            self.l1 = [GatedLin(K, D, m, lr, start, seed + 10 + j) for j in range(neurons)]
            s2 = np.full(neurons + 1, 0.01)
            s2[-1] = 1.0
            self.l2 = GatedLin(neurons + 1, D, m, lr, s2, seed + 99)

    def predict(self, q, s, small, r=None):
        if self.kind == "one":
            self.u = self.units[small]
            return self.u.predict(q, np.zeros(0))
        h = np.array([n.predict(q, s) for n in self.l1] + [q[-1]])
        return self.l2.predict(h, s)

    def learn(self):
        if self.kind == "one":
            self.u.learn()
            return
        for n in self.l1:
            n.learn()
        self.l2.learn()


def run(net, D, nsig, mu, sd, score, learn=True):
    E = np.maximum(D[:, [COLS.index(e) for e in EXPERTS]], 1e-9)
    ys = D[:, 0]
    ewma = np.full(len(TRACK), 2.0)
    tix = [EXPERTS.index(t) for t in TRACK]
    surprise, wordpos, prevc = 2.0, 0, 0
    bits = np.zeros(len(D))
    side_cols = [COLS.index(c) for c in ("bkn_b", "ig_b", "cls_b", "sim_b", "copy_b", "trig_b", "cache_b", "topic_b",
                                         "match_on")]
    for t in range(len(D)):
        q = E[t]
        r = D[t]
        s = np.concatenate([r[side_cols], [math.log1p(r[COLS.index("ig_n")]) / 9.0, r[COLS.index("copy_n")] / 32.0],
                            ewma, [surprise, min(wordpos, 12) / 12.0], np.eye(5)[prevc]])
        s = (s - mu) / sd
        small = min(int(r[COLS.index("ig_b")]), 15) * 5 + prevc
        p = net.predict(q, s, small, r)
        b = -math.log2(max(p, 1e-12))
        bits[t] = b
        if learn:
            net.learn()
        ewma = 0.95 * ewma + 0.05 * -np.log2(q[tix])
        surprise = 0.95 * surprise + 0.05 * b
        c = byte_class(ys[t])
        wordpos = 0 if c == 0 else wordpos + 1
        prevc = c
    return bits


def side_stats(D):
    """Mean and spread of the side vector over the first stream (for the half-space gates)."""
    side_cols = [COLS.index(c) for c in ("bkn_b", "ig_b", "cls_b", "sim_b", "copy_b", "trig_b", "cache_b", "topic_b", "match_on")]
    X = np.concatenate([D[:, side_cols], np.log1p(D[:, [COLS.index("ig_n")]]) / 9.0, D[:, [COLS.index("copy_n")]] / 32.0,
                        np.full((len(D), len(TRACK)), 2.0), np.full((len(D), 1), 2.0), np.full((len(D), 1), 0.5),
                        np.full((len(D), 5), 0.2)], 1)
    mu, sd = X.mean(0), X.std(0) + 1e-3
    sd[-(len(TRACK) + 7):] = 1.0  # the running features are left on their own scale
    mu[-(len(TRACK) + 7):] = np.concatenate([np.full(len(TRACK), 2.0), [2.0, 0.5], np.full(5, 0.2)])
    return mu, sd


def block_boot(a, b, block=4096, n=2000, seed=7):
    """b - a in bits per byte, bootstrap over 4 KiB blocks (as cyphalm_trie's block_boot)."""
    k = len(a) // block
    da = (b[:k * block] - a[:k * block]).reshape(k, block).sum(1)
    rng = np.random.default_rng(seed)
    bs = [da[rng.integers(0, k, k)].sum() / (k * block) for _ in range(n)]
    return float((b - a).mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


# ------------------------------------------------------------------------------------------------ D: on the online chain
# Odin (2026-11-01): "Try the experiment we just did with the online learning on the 2-layer one." The inputs are the raw
# experts plus the chain's stages as the online chain (C0o) computes them while it learns, not the frozen chain's.
RAW = ["uni", "o1", "bo2", "bo3", "bo4", "match_hit", "trie_mix", "bkn", "ig4", "ig16", "ig64", "ig512", "ig8192", "cls",
       "sim", "copy_hit", "trig", "cache", "topic"]
SELECT = [("ig_b", 16), ("bkn_b", 16), ("copy_b", 8), ("cls_b", 16), ("sim_b", 16), ("cache_b", 16), ("prev", 5),
          ("ign", 16)]  # each layer-1 neuron's small context (gate24-style), the same sizes as the buckets


def online_stages(D):
    """The online chain over all rows in order (first half then second): each row's 6 stage outputs, the last is final."""
    ch = OnlineChain()
    S = np.zeros((len(D), len(OnlineChain.STAGES)))
    for t in range(len(D)):
        r = D[t]
        ch.predict(None, None, None, r)
        S[t] = [tr[4] for tr in ch.trace]
        ch.learn()
    return S


class Net2:
    """Two layers over RAW + the online stages. sel='half': half-space gates on the side vector (as C2);
    sel='small': neuron j gated by its own small context SELECT[j], layer 2 by InfiniGram's bucket x the previous byte."""

    def __init__(self, K, D, sel, lr, neurons=8, seed=0):
        start = np.full(K, 0.01)
        start[-1] = 1.0  # start at the online chain's final stage
        self.sel = sel
        if sel == "half":
            self.l1 = [GatedLin(K, D, 4, lr, start, seed + j) for j in range(neurons)]
        else:
            self.l1 = [GatedLin(K, 0, 0, lr, start, seed + j) for j in range(len(SELECT))]
            self.tabs = [[GatedLin(K, 0, 0, lr, start, seed + 100 * j + g) for g in range(n)] for j, (_, n) in enumerate(SELECT)]
        n1 = len(self.l1)
        s2 = np.full(n1 + 1, 0.01)
        s2[-1] = 1.0
        self.l2 = GatedLin(n1 + 1, D, 4, lr, s2, seed + 99) if sel == "half" else None
        self.l2s = None if sel == "half" else [GatedLin(n1 + 1, 0, 0, lr, s2, seed + 999 + g) for g in range(80)]

    def predict(self, q, s, ctxs, small):
        if self.sel == "half":
            h = [n.predict(q, s) for n in self.l1]
            self.u2 = self.l2
            return self.u2.predict(np.array(h + [q[-1]]), s)
        self.units = [self.tabs[j][min(c, SELECT[j][1] - 1)] for j, c in enumerate(ctxs)]
        h = [u.predict(q, np.zeros(0)) for u in self.units]
        self.u2 = self.l2s[small]
        return self.u2.predict(np.array(h + [q[-1]]), np.zeros(0))

    def learn(self):
        for u in (self.l1 if self.sel == "half" else self.units):
            u.learn()
        self.u2.learn()


def run_d(net, D, S, mu, sd):
    E = np.maximum(np.concatenate([D[:, [COLS.index(e) for e in RAW]], S], 1), 1e-9)
    ys = D[:, 0]
    ewma = np.full(len(TRACK), 2.0)
    tix = [EXPERTS.index(t) for t in TRACK]
    E0 = np.maximum(D[:, [COLS.index(e) for e in EXPERTS]], 1e-9)
    surprise, wordpos, prevc = 2.0, 0, 0
    side_cols = [COLS.index(c) for c in ("bkn_b", "ig_b", "cls_b", "sim_b", "copy_b", "trig_b", "cache_b", "topic_b", "match_on")]
    bits = np.zeros(len(D))
    for t in range(len(D)):
        r = D[t]
        s = np.concatenate([r[side_cols], [math.log1p(r[COLS.index("ig_n")]) / 9.0, r[COLS.index("copy_n")] / 32.0],
                            ewma, [surprise, min(wordpos, 12) / 12.0], np.eye(5)[prevc]])
        s = (s - mu) / sd
        ctxs = [int(r[COLS.index("ig_b")]), int(r[COLS.index("bkn_b")]), int(r[COLS.index("copy_b")]), int(r[COLS.index("cls_b")]),
                int(r[COLS.index("sim_b")]), int(r[COLS.index("cache_b")]), prevc, min(15, int(math.log2(1 + r[COLS.index("ig_n")])))]
        small = min(int(r[COLS.index("ig_b")]), 15) * 5 + prevc
        p = net.predict(E[t], s, ctxs, small)
        b = -math.log2(max(p, 1e-12))
        bits[t] = b
        net.learn()
        ewma = 0.95 * ewma + 0.05 * -np.log2(E0[t, tix])
        surprise = 0.95 * surprise + 0.05 * b
        c = byte_class(ys[t])
        wordpos = 0 if c == 0 else wordpos + 1
        prevc = c
    return bits


def main_d(path, fast):
    A, B, half = load(path, fast)
    mu, sd = side_stats(A)
    S = online_stages(np.concatenate([A, B]))
    SA, SB = S[:len(A)], S[len(A):]
    base = -np.log2(np.maximum(B[:, COLS.index("chain")], 1e-12))
    online = -np.log2(np.maximum(SB[:, -1], 1e-12))
    res = {"C0": float(base.mean()), "D0_online_chain": float(online.mean())}
    K = len(RAW) + S.shape[1]
    arms = {"D1": dict(sel="small1"), "D2": dict(sel="half", lr=0.05), "D3": dict(sel="small", lr=0.05),
            "D4": dict(sel="small", lr=0.025), "D5": dict(sel="half", lr=0.025)}
    for name, cfg in arms.items():
        t0 = time.time()
        if cfg["sel"] == "small1":  # one online layer over the online chain (the C++ test's R2)
            units = [GatedLin(K, 0, 0, 0.05, np.r_[np.full(K - 1, 0.01), 1.0], j) for j in range(80)]

            class W:
                def predict(self, q, s, ctxs, small):
                    self.u = units[small]
                    return self.u.predict(q, np.zeros(0))

                def learn(self):
                    self.u.learn()
            net = W()
        else:
            net = Net2(K, len(mu), "half" if cfg["sel"] == "half" else "small", cfg["lr"])
        run_d(net, A, SA, mu, sd)
        bits = run_d(net, B, SB, mu, sd)
        d, ci = block_boot(base, bits)
        d2, ci2 = block_boot(online, bits)
        res[name] = dict(bpb=float(bits.mean()), minus_C0=d, ci=ci, minus_D0=d2, ci_D0=ci2, sec=round(time.time() - t0, 1), **cfg)
        print(name, json.dumps(res[name]), flush=True)
    res["rows"] = dict(first=len(A), second=len(B), fast=fast)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=1))
    json.dump(res, open(path + (".fast" if fast else "") + ".glnx_d.json", "w"), indent=1)


def main():
    path, fast = sys.argv[1], "--fast" in sys.argv
    if "--online-inputs" in sys.argv:
        return main_d(path, fast)
    A, B, half = load(path, fast)
    mu, sd = side_stats(A)
    D = len(mu)
    # Odin: more gates do not learn better (gate24's finding), and the mixer is not to be frozen
    arms = {"C0o": dict(kind="chain"), "C1": dict(kind="one"), "C2": dict(kind="two", m=4)}
    base = -np.log2(np.maximum(B[:, COLS.index("chain")], 1e-12))
    res = {"C0": dict(bpb=float(base.mean()), what="the content mixer as it is (lambdas fit on the first half)")}
    out = {}
    for name, cfg in arms.items():
        t0 = time.time()
        net = OnlineChain() if cfg["kind"] == "chain" else Net(cfg["kind"], D, m=cfg.get("m", 4), lr=cfg.get("lr", 0.05),
                                                                 neurons=cfg.get("neurons", 8))
        run(net, A, None, mu, sd, False)  # learn on the first half
        bits = run(net, B, None, mu, sd, True, learn=not cfg.get("frozen"))  # second half: predict, score (and learn)
        d, ci = block_boot(base, bits)
        res[name] = dict(bpb=float(bits.mean()), minus_C0=d, ci=ci, sec=round(time.time() - t0, 1), **cfg)
        out[name] = net
        print(name, json.dumps(res[name]), flush=True)
    # what layer 2 learned: the experts' mean weight in layer 1, over the gates in use (C2)
    net = out["C2"]
    W = np.array([(n.w * (n.n[:, None] + 1e-9)).sum(0) / max(n.n.sum(), 1) for n in net.l1]).mean(0)
    res["C2_layer1_mean_weights"] = {e: round(float(w), 4) for e, w in sorted(zip(EXPERTS, W), key=lambda x: -x[1])}
    res["C2_layer2_weights_by_gate_used"] = [[round(float(x), 3) for x in net.l2.w[g]] for g in np.argsort(-net.l2.n)[:4]]
    res["rows"] = dict(first=len(A), second=len(B), fast=fast)
    print(json.dumps(res, indent=1))
    json.dump(res, open(path + (".fast" if fast else "") + ".glnx.json", "w"), indent=1)


if __name__ == "__main__":
    main()
