#!/usr/bin/env python3
"""GLN into the content mixer, online on every part (docs/GENERATION_PLAN.md §13.15). Odin Loch, 2026-11-03.

Odin: "Have the GLN feed into the mixers as well, feed that into the online learning. So basically it's going to be
recurrent and online on every part."

Reads the preprocessing phase's dump (glnx_prep.sh: every expert's probability of each holdout byte, CyphaLM's whole
model) and GLNB's probability of each holdout byte (glnb.py ENWIK8 stream OUT.f32 [MIXER_P.f32]). Then, byte by byte:
  - the content mixer's chain, every lambda learning online (OPT: adagrad as cyphalm_trie --online-mix, logit
    (lambda = sigmoid(theta)), or eg: exponentiated gradient, the §13.15 fix, which no single byte can stall);
  - optionally GLNB as one more chain stage (lambda per InfiniGram bucket);
  - optionally the online final layer (cyphalm_trie --online-final: exponentiated gradient over every expert and chain
    stage, 80 weight sets by InfiniGram bucket x previous byte class), with GLNB as one more input.
Scores the holdout's second half; 95% intervals for differences from a block bootstrap (64 KiB blocks).

Usage: gln_mix.py DUMP GLN0.f32 [GLN1.f32]      (GLN1: GLNB with the mixer's feedback)
"""
import json
import math
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from glnx import COLS  # noqa: E402

STAGES = [("bkn", "bkn_b"), ("ig8192", "ig_b"), ("cls", "cls_b"), ("sim", "sim_b"), ("copy_hit", "copy_b"),
          ("cache", "cache_b")]
COPY_LO = [0, 3, 4, 6, 8, 12, 16, 32]
K_IN = [1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 18, 20, 22, 25, 27, 29]  # cyphalm_trie's kIn, before the chain


def stage(pe, bk, cur, opt, dead_below=0):
    """One chain stage: cur <- lam[b] pe + (1 - lam[b]) cur, lam learning online after each byte."""
    nb = int(bk.max()) + 1
    lam = np.full(nb, 0.05)
    acc = np.full(nb, 1e-3)
    th = np.full(nb, math.log(0.05 / 0.95))
    w = np.tile([0.05, 0.95], (nb, 1))
    dead = np.array([b < len(COPY_LO) and COPY_LO[b] < dead_below for b in range(nb)])
    lam[dead] = 0.0
    out = np.empty_like(cur)
    for k in range(len(cur)):
        b = bk[k]
        prev, e, l = cur[k], pe[k], lam[b]
        p = l * e + (1 - l) * prev
        out[k] = p
        if dead[b]:
            continue
        if opt == "eg":  # exponentiated gradient on (lam, 1 - lam), step 0.02: the §13.15 fix
            pp = max(p, 1e-300)
            w[b, 0] *= math.exp(0.02 * min(e / pp, 30.0))
            w[b, 1] *= math.exp(0.02 * min(prev / pp, 30.0))
            w[b] = np.maximum(w[b] / w[b].sum(), 1e-4)
            w[b] /= w[b].sum()
            lam[b] = w[b, 0]
            continue
        g = (e - prev) / max(p, 1e-12)
        if opt == "adagrad":
            acc[b] += g * g
            lam[b] = min(0.995, max(0.0, l + 0.05 * g / math.sqrt(acc[b])))
        else:
            gt = l * (1 - l) * g
            acc[b] += gt * gt
            th[b] = min(7.0, max(-9.0, th[b] + 0.1 * gt / math.sqrt(acc[b])))
            lam[b] = 1 / (1 + math.exp(-th[b]))
    return out


def chain(c, opt, copy_min, gln=None):
    cur = c["trie_mix"].copy()
    snaps = []
    for e, b in STAGES:
        cur = stage(c[e], c[b].astype(int), cur, opt, copy_min if e == "copy_hit" else 0)
        if e in ("ig8192", "cls", "sim", "copy_hit"):
            snaps.append(cur)
    if gln is not None:
        cur = stage(gln, c["ig_b"].astype(int), cur, opt)
    return cur, snaps


def final_layer(D, c, snaps, cur, gln=None):
    """cyphalm_trie --online-final, with GLNB as one more input when given."""
    cols = [D[:, j] for j in K_IN] + list(snaps) + [cur]
    if gln is not None:
        cols.insert(-1, gln)
    Q = np.maximum(np.stack(cols, 1), 1e-9)
    n = Q.shape[1]
    W = np.full((80, n), 0.01)
    W[:, -1] = 1.0
    W /= W.sum(1, keepdims=True)
    y = c["y"].astype(int)
    pb = np.concatenate([[32], y[:-1]])
    pcls = np.where(np.isin(pb, [9, 10, 11, 12, 13, 32]), 0,
                    np.where(((pb >= 65) & (pb <= 90)) | ((pb >= 97) & (pb <= 122)), 1,
                             np.where((pb >= 48) & (pb <= 57), 2, np.where(np.isin(pb, list(b"[]{}<>|&=")), 3, 4))))
    gate = np.minimum(c["ig_b"].astype(int), 15) * 5 + pcls
    out = np.empty(len(cur))
    for k in range(len(cur)):
        w = W[gate[k]]
        q = Q[k]
        p = float(w @ q)
        out[k] = p
        w *= np.exp(0.05 * np.minimum(q / max(p, 1e-300), 30.0))
        w /= w.sum()
        np.maximum(w, 1e-7, out=w)
        w /= w.sum()
    return out


def bits(p, half):
    return -np.log2(np.maximum(p[half:], 1e-12))


def diff_ci(a, b, block=65536, n=2000, seed=0):
    d = a - b
    nb = len(d) // block
    m = d[:nb * block].reshape(nb, block).mean(1)
    rng = np.random.default_rng(seed)
    bs = rng.choice(m, (n, nb)).mean(1)
    return [round(float(d.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def main():
    D = np.fromfile(sys.argv[1], np.float32).reshape(-1, len(COLS)).astype(np.float64)
    c = {n: D[:, COLS.index(n)] for n in COLS}
    half = len(D) // 2
    g0 = np.fromfile(sys.argv[2], np.float32).astype(np.float64)
    g1 = np.fromfile(sys.argv[3], np.float32).astype(np.float64) if len(sys.argv) > 3 else None
    res, B = {}, {}
    B["frozen"] = bits(c["chain"], half)
    B["gln_alone"] = bits(g0, half)
    for name, opt, copy_min, gln, fin in (("M0_chain_adagrad_clip", "adagrad", 8, None, False),
                                         ("M0_chain", "eg", 0, None, False),
                                         ("M1_chain_gln", "eg", 0, g0, False),
                                         ("M2_chain_final", "eg", 0, None, True),
                                         ("M3_chain_gln_final", "eg", 0, g0, True)) + (
            (("M4_chain_glnfb_final", "eg", 0, g1, True),) if g1 is not None else ()):
        cur, snaps = chain(c, opt, copy_min, gln)
        p = final_layer(D, c, snaps, cur, gln) if fin else cur
        B[name] = bits(p, half)
        print(name, round(float(B[name].mean()), 4), flush=True)
    for k, v in B.items():
        res[k] = round(float(v.mean()), 4)
    res["M1-M0"] = diff_ci(B["M1_chain_gln"], B["M0_chain"])
    res["M3-M2"] = diff_ci(B["M3_chain_gln_final"], B["M2_chain_final"])
    res["M2-M0"] = diff_ci(B["M2_chain_final"], B["M0_chain"])
    res["M0-frozen"] = diff_ci(B["M0_chain"], B["frozen"])
    if g1 is not None:
        res["M4-M3"] = diff_ci(B["M4_chain_glnfb_final"], B["M3_chain_gln_final"])
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
