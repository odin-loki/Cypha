#!/usr/bin/env python3
"""GLNB: a hybrid of a proper bitwise gated linear network with an LSTM inside it (Odin, 2026-11-01: "Make a hybrid model
of a proper bitwise GLN with an LSTM capability", fast mode, Python). Odin Loch, 2026-11-01. docs/GENERATION_PLAN.md §13.13.

Every byte is 8 binary predictions (most significant bit first, partial byte c0 as in PAQ).
- Base inputs: bitwise context models of orders 0-5 (hashed counts of the bit after the previous k bytes and the partial
  byte), counted from enwik8's first 7 MiB (CyphaLM's training text), then updated online.
- The LSTM: a byte-level LSTM (hidden 64) that learns online from scratch on the stream (truncated BPTT, Adam). Its
  recurrent state is part of every layer-1 neuron's gate, and its own probability of each bit is one more input.
- The GLN (Veness et al.): layer-1 neurons, each a gated geometric mixer (logistic mixing of the inputs' log-odds) whose
  weights are chosen by half-spaces of the side vector (the LSTM state and the inputs) for each bit position; a second
  layer of 4 neurons; a final neuron per bit position; an adaptive probability map (APM). Every neuron learns online
  from its own log loss. No backpropagation through the GLN.
Read as before: learn on 64 KiB of the holdout's first half, then predict, score and learn on 64 KiB of its second half,
the same bytes as glnx.py --fast (the content mixer there: frozen 1.7786, online 1.7557 bits per byte).
Usage: glnb.py [ENWIK8]
"""
import json
import math
import sys
import time

import numpy as np

TRAIN, HOLD, HALF = 7 * 2 ** 20, 2 ** 20, 2 ** 19
FAST = 65536
ORDERS = 6
TB = 22  # hash table bits per order


def stretch(p):
    return math.log(p / (1 - p))


def squash(x):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


M64 = 0xFFFFFFFFFFFFFFFF


def ctx_hash(h, k, c0):  # the same 64-bit arithmetic as pretrain's numpy keys
    return ((((h * 0x9E3779B97F4A7C15) & M64) ^ (c0 * 0x100000001B3) ^ ((k * 0xC2B2AE3D27D4EB4F) & M64)) >> 7) & ((1 << TB) - 1)


def history_hashes(b, k):
    """For each position t, a hash of the k bytes before it (vectorised; 0 for k = 0)."""
    n = len(b)
    h = np.zeros(n, np.uint64)
    with np.errstate(over="ignore"):
        for i in range(1, k + 1):
            prev = np.zeros(n, np.uint64)
            prev[i:] = b[:n - i].astype(np.uint64) + np.uint64(1)
            h = h * np.uint64(1099511628211) + prev
    return h


def pretrain(train):
    """Counts n1 (bit = 1) and n per hashed (order, previous bytes, partial byte) from the training text."""
    tabs = []
    b = train.astype(np.int64)
    for k in range(ORDERS):
        hk = history_hashes(train, k)
        n1 = np.zeros(1 << TB, np.float32)
        n = np.zeros(1 << TB, np.float32)
        for j in range(8):
            c0 = (1 << j) | (b >> (8 - j))
            bit = (b >> (7 - j)) & 1
            with np.errstate(over="ignore"):
                key = (((hk * np.uint64(0x9E3779B97F4A7C15)) ^ (c0.astype(np.uint64) * np.uint64(0x100000001B3))
                        ^ np.uint64((k * 0xC2B2AE3D27D4EB4F) & M64)) >> np.uint64(7)) & np.uint64((1 << TB) - 1)
            key = key.astype(np.int64)
            n1 += np.bincount(key, weights=bit, minlength=1 << TB).astype(np.float32)
            n += np.bincount(key, minlength=1 << TB).astype(np.float32)
        big = n > 255
        n1[big] *= 255 / n[big]
        n[big] = 255
        tabs.append((n1, n))
    return tabs


class LSTM:
    """A byte LSTM: embedding 16, hidden 64, softmax over 256; online truncated BPTT (window 20), Adam."""

    def __init__(self, H=64, Ein=16, seed=0, lr=0.003, T=20):
        rng = np.random.default_rng(seed)
        self.H, self.T, self.lr = H, T, lr
        self.P = dict(E=rng.standard_normal((256, Ein)) * 0.1, W=rng.standard_normal((4 * H, Ein + H)) * 0.1,
                      b=np.zeros(4 * H), Wo=rng.standard_normal((256, H)) * 0.1, bo=np.zeros(256))
        self.P["b"][H:2 * H] = 1.0  # forget gate bias
        self.m = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.t = 0
        self.h, self.c = np.zeros(H), np.zeros(H)
        self.buf = []
        self.prev = 32

    def dist(self):
        z = self.P["Wo"] @ self.h + self.P["bo"]
        z -= z.max()
        e = np.exp(z)
        return e / e.sum()

    def step(self, y):
        """Learn that the byte after self.prev is y, then advance the state with y."""
        P, H = self.P, self.H
        x = P["E"][self.prev]
        xh = np.concatenate([x, self.h])
        a = P["W"] @ xh + P["b"]
        i, f, o = (1 / (1 + np.exp(-a[:H])), 1 / (1 + np.exp(-a[H:2 * H])), 1 / (1 + np.exp(-a[2 * H:3 * H])))
        g = np.tanh(a[3 * H:])
        c = f * self.c + i * g
        h = o * np.tanh(c)
        self.buf.append((self.prev, xh, i, f, o, g, self.c, c, h, y))
        self.h, self.c, self.prev = h, c, y
        if len(self.buf) >= self.T:
            self.bptt()

    def bptt(self):
        P, H = self.P, self.H
        G = {k: np.zeros_like(v) for k, v in P.items()}
        dh_next, dc_next = np.zeros(H), np.zeros(H)
        for (prev, xh, i, f, o, g, c_prev, c, h, y) in reversed(self.buf):
            z = P["Wo"] @ h + P["bo"]
            z -= z.max()
            p = np.exp(z)
            p /= p.sum()
            p[y] -= 1
            G["Wo"] += np.outer(p, h)
            G["bo"] += p
            dh = P["Wo"].T @ p + dh_next
            tc = np.tanh(c)
            do = dh * tc * o * (1 - o)
            dc = dh * o * (1 - tc * tc) + dc_next
            di, df, dg = dc * g * i * (1 - i), dc * c_prev * f * (1 - f), dc * i * (1 - g * g)
            da = np.concatenate([di, df, do, dg])
            G["W"] += np.outer(da, xh)
            G["b"] += da
            dxh = P["W"].T @ da
            G["E"][prev] += dxh[:len(xh) - H]
            dh_next, dc_next = dxh[len(xh) - H:], dc * f
        self.t += 1
        for k in P:
            gk = np.clip(G[k] / len(self.buf), -5, 5)
            self.m[k] = 0.9 * self.m[k] + 0.1 * gk
            self.v[k] = 0.999 * self.v[k] + 0.001 * gk * gk
            P[k] -= self.lr * (self.m[k] / (1 - 0.9 ** self.t)) / (np.sqrt(self.v[k] / (1 - 0.999 ** self.t)) + 1e-8)
        self.buf = []


class Hybrid:
    def __init__(self, tabs, neurons=32, lstm=True, seed=0):
        rng = np.random.default_rng(seed)
        self.tabs, self.N, self.use_lstm = tabs, neurons, lstm
        self.K = ORDERS + (1 if lstm else 0) + 1  # inputs: orders, LSTM, bias
        self.D = (64 if lstm else 0) + self.K  # side vector: LSTM state + inputs
        self.H1 = rng.standard_normal((neurons, 4, self.D))
        self.W1 = np.full((neurons, 16, 8, self.K), 0.3)
        self.W1[..., -1] = 0.0
        self.H2 = rng.standard_normal((4, 2, self.D))
        self.W2 = np.full((4, 4, 8, neurons + 1), 1.0 / neurons)
        self.W3 = np.full((8, 5), 0.25)
        self.apm = np.tile(np.array([squash((i - 16) / 2.0) for i in range(33)]), (256, 1))
        self.lstm = LSTM(seed=seed) if lstm else None
        self.hist = [32] * 8

    def code_byte(self, y, learn=True):
        """Bits of byte y under the model, then learn it."""
        hist = self.hist
        hh = [0] * ORDERS
        for k in range(1, ORDERS):
            v = 0
            for i in range(1, k + 1):
                v = (v * 1099511628211 + hist[-i] + 1) & 0xFFFFFFFFFFFFFFFF
            hh[k] = v
        if self.lstm is not None:
            pl = self.lstm.dist()
            cum = np.concatenate([[0.0], np.cumsum(pl)])
            hstate = self.lstm.h
        bits, c0 = 0.0, 1
        for j in range(8):
            bit = (y >> (7 - j)) & 1
            keys = [ctx_hash(hh[k], k, c0) for k in range(ORDERS)]
            ps = []
            for k, key in enumerate(keys):
                n1, n = self.tabs[k][0][key], self.tabs[k][1][key]
                ps.append((n1 + 0.4) / (n + 0.8))
            if self.lstm is not None:
                lo, hi = (c0 - (1 << j)) << (8 - j), ((c0 - (1 << j)) + 1) << (8 - j)
                mid = (lo + hi) // 2
                tot = cum[hi] - cum[lo]
                ps.append(min(max((cum[hi] - cum[mid]) / max(tot, 1e-12), 1e-4), 1 - 1e-4))
            x = np.array([stretch(min(max(p, 1e-4), 1 - 1e-4)) for p in ps] + [1.0])
            z = np.concatenate([hstate, x]) if self.lstm is not None else x
            g1 = ((np.einsum("nmd,d->nm", self.H1, z) > 0) * (1 << np.arange(4))).sum(1)
            w1 = self.W1[np.arange(self.N), g1, j]
            s1 = np.clip(w1 @ x, -12, 12)
            p1 = 1 / (1 + np.exp(-s1))
            x2 = np.concatenate([s1, [1.0]])
            g2 = ((np.einsum("nmd,d->nm", self.H2, z) > 0) * (1 << np.arange(2))).sum(1)
            w2 = self.W2[np.arange(4), g2, j]
            s2 = np.clip(w2 @ x2, -12, 12)
            p2 = 1 / (1 + np.exp(-s2))
            x3 = np.concatenate([s2, [1.0]])
            s3 = float(np.clip(self.W3[j] @ x3, -12, 12))
            p3 = squash(s3)
            pos = (s3 + 8) * 2
            lo_i = int(min(max(math.floor(pos), 0), 31))
            fr = min(max(pos - lo_i, 0.0), 1.0)
            pa = self.apm[c0, lo_i] * (1 - fr) + self.apm[c0, lo_i + 1] * fr
            p = min(max(0.5 * p3 + 0.5 * pa, 1e-6), 1 - 1e-6)
            bits += -math.log2(p if bit else 1 - p)
            if learn:
                e1 = bit - p1
                self.W1[np.arange(self.N), g1, j] = np.clip(w1 + 0.002 * e1[:, None] * x[None, :], -8, 8)
                e2 = bit - p2
                self.W2[np.arange(4), g2, j] = np.clip(w2 + 0.002 * e2[:, None] * x2[None, :], -8, 8)
                self.W3[j] = np.clip(self.W3[j] + 0.002 * (bit - p3) * x3, -8, 8)
                self.apm[c0, lo_i] += 0.02 * (1 - fr) * (bit - self.apm[c0, lo_i])
                self.apm[c0, lo_i + 1] += 0.02 * fr * (bit - self.apm[c0, lo_i + 1])
                for k, key in enumerate(keys):
                    n1, n = self.tabs[k]
                    n1[key] += bit
                    n[key] += 1
                    if n[key] > 255:
                        n1[key] *= 0.5
                        n[key] *= 0.5
            c0 = (c0 << 1) | bit
        if self.lstm is not None:
            self.lstm.step(y)
        self.hist = hist[1:] + [y]
        return bits


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/root/cdata/enwik8"
    raw = np.fromfile(path, np.uint8, count=TRAIN + HOLD)
    train, hold = raw[:TRAIN], raw[TRAIN:]
    A, B = hold[:FAST], hold[HALF:HALF + FAST]
    t0 = time.time()
    base = pretrain(train)
    print("pretrained", round(time.time() - t0), "s", flush=True)
    res = {}
    for name, cfg in (("H16", dict(neurons=16)), ("H32", dict(neurons=32)), ("H64", dict(neurons=64)),
                      ("H32_noLSTM", dict(neurons=32, lstm=False))):
        t1 = time.time()
        tabs = [(n1.copy(), n.copy()) for n1, n in base]
        m = Hybrid(tabs, **cfg)
        for y in A:
            m.code_byte(int(y))
        bits = sum(m.code_byte(int(y)) for y in B)
        res[name] = dict(bpb=bits / len(B), sec=round(time.time() - t1), **cfg)
        print(name, json.dumps(res[name]), flush=True)
    res["reference"] = dict(content_mixer_frozen=1.7786, content_mixer_online=1.7557, bytes=len(B))
    json.dump(res, open("/root/work/glnx/glnb_fast.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
