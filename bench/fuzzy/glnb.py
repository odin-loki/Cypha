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
import os
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
        self.cache = self._forward(32)  # the state after reading a space; dist() predicts the first byte from it

    def dist(self):
        z = self.P["Wo"] @ self.h + self.P["bo"]
        z -= z.max()
        e = np.exp(z)
        return e / e.sum()

    def _forward(self, xb):
        """Read byte xb: the new state, and what BPTT needs to learn from it."""
        P, H = self.P, self.H
        xh = np.concatenate([P["E"][xb], self.h])
        a = P["W"] @ xh + P["b"]
        i, f, o = (1 / (1 + np.exp(-a[:H])), 1 / (1 + np.exp(-a[H:2 * H])), 1 / (1 + np.exp(-a[2 * H:3 * H])))
        g = np.tanh(a[3 * H:])
        c = f * self.c + i * g
        h = o * np.tanh(c)
        cache = (xb, xh, i, f, o, g, self.c, c, h)
        self.h, self.c = h, c
        return cache

    def step(self, y, learn=True):
        """The state has read every byte before y and predicted y: learn from that, then read y.
        (Until 2026-11-01 the state read y's predecessor only after predicting y, so it was one byte behind.)"""
        if learn:
            self.buf.append(self.cache + (y,))
            if len(self.buf) >= self.T:
                self.bptt()
        self.cache = self._forward(y)

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
    def __init__(self, tabs, neurons=32, lstm=True, seed=0, hs=4, extra=0):
        rng = np.random.default_rng(seed)
        self.tabs, self.N, self.use_lstm = tabs, neurons, lstm
        self.K = ORDERS + (1 if lstm else 0) + 1  # inputs: orders, LSTM, bias
        self.D = (64 if lstm else 0) + self.K + extra  # side vector: LSTM state + inputs + extra (the mixer's feedback)
        self.extra = np.zeros(extra)
        self.H1 = rng.standard_normal((neurons, hs, self.D))  # hs half-spaces per neuron: 2**hs contexts
        self.W1 = np.full((neurons, 2 ** hs, 8, self.K), 0.3)
        self.W1[..., -1] = 0.0
        self.H2 = rng.standard_normal((4, 2, self.D))
        self.W2 = np.full((4, 4, 8, neurons + 1), 1.0 / neurons)
        self.W3 = np.full((8, 5), 0.25)
        self.apm = np.tile(np.array([squash((i - 16) / 2.0) for i in range(33)]), (256, 1))
        self.lstm = LSTM(seed=seed) if lstm else None
        self.hist = [32] * 8

    def _ctx(self):
        """What every bit of the next byte reads: the history hashes, the LSTM's byte distribution and its state."""
        hist = self.hist
        hh = [0] * ORDERS
        for k in range(1, ORDERS):
            v = 0
            for i in range(1, k + 1):
                v = (v * 1099511628211 + hist[-i] + 1) & 0xFFFFFFFFFFFFFFFF
            hh[k] = v
        cum = hstate = None
        if self.lstm is not None:
            pl = self.lstm.dist()
            cum = np.concatenate([[0.0], np.cumsum(pl)])
            hstate = self.lstm.h
        return hh, cum, hstate

    def _bit(self, hh, cum, hstate, c0, j):
        """The probability that bit j (most significant first) is 1 after the partial byte c0, and what learning needs."""
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
        z = np.concatenate([hstate, x, self.extra]) if self.lstm is not None else np.concatenate([x, self.extra])
        g1 = ((np.einsum("nmd,d->nm", self.H1, z) > 0) * (1 << np.arange(self.H1.shape[1]))).sum(1)
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
        return p, (keys, x, g1, w1, p1, x2, g2, w2, p2, x3, p3, lo_i, fr)

    def code_byte(self, y, learn=True):
        """Bits of byte y under the model, then learn it (learn=False: only read it)."""
        hh, cum, hstate = self._ctx()
        bits, c0 = 0.0, 1
        for j in range(8):
            bit = (y >> (7 - j)) & 1
            p, (keys, x, g1, w1, p1, x2, g2, w2, p2, x3, p3, lo_i, fr) = self._bit(hh, cum, hstate, c0, j)
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
            self.lstm.step(y, learn)
        self.hist = self.hist[1:] + [y]
        return bits

    def byte_dist(self):
        """The model's probability of each of the 256 next bytes: 255 bit predictions, depth by depth."""
        hh, cum, hstate = self._ctx()
        P = np.ones(1)
        for j in range(8):
            p1 = np.array([self._bit(hh, cum, hstate, (1 << j) | r, j)[0] for r in range(1 << j)])
            P = np.stack([P * (1 - p1), P * p1], 1).reshape(-1)
        return P

    def snap(self):
        L = self.lstm
        return (self.hist[:],) + ((L.h, L.c, L.cache, L.buf[:]) if L is not None else ())

    def restore(self, st):
        self.hist = st[0][:]
        if self.lstm is not None:
            self.lstm.h, self.lstm.c, self.lstm.cache, self.lstm.buf = st[1], st[2], st[3], st[4][:]


def esc(s):
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")


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


def generate(m, prompt, rng, temp, nbytes=600, W=11, nucleus=0.9):
    """The byte sampler's rules (as fuzzy_bake.py's K5 arm and cyphalm_trie sample): no learning while writing; a byte
    is banned if it would follow an 11-byte window with the byte that followed the same window earlier (prompt
    included); temperature, then nucleus 0.9."""
    st = m.snap()
    pb = prompt.encode("utf-8")
    for y in pb:
        m.code_byte(y, learn=False)
    text, out = bytearray(pb), bytearray()
    nx = {}
    for s in range(len(text) - W):
        nx.setdefault(bytes(text[s:s + W]), set()).add(text[s + W])
    while len(out) < nbytes:
        P0 = m.byte_dist()
        P = P0 ** (1.0 / temp) if temp != 1.0 else P0.copy()
        key = bytes(text[-W:]) if len(text) >= W else None
        for c in nx.get(key, ()):
            P[c] = 0.0
        if P.sum() <= 0:
            P = P0.copy()
        P /= P.sum()
        order = np.argsort(-P)
        k = int(np.searchsorted(np.cumsum(P[order]), nucleus)) + 1
        keep = order[:k]
        y = int(rng.choice(keep, p=P[keep] / P[keep].sum()))
        if key is not None:
            nx.setdefault(key, set()).add(y)
        text.append(y)
        out.append(y)
        m.code_byte(y, learn=False)
    m.restore(st)
    return bytes(out).decode("utf-8", "ignore")


def gen_main(base, hold, arm, temp, nprompts, out):
    """Learn as the read does (64 KiB of the holdout's first half, then 64 KiB of its second half, scored), then write
    600 bytes for each of GEN-A's first NPROMPTS dev prompts, in the gen file format SCRIBE's ladder reads."""
    t0 = time.time()
    m = Hybrid(base, neurons=32, lstm=(arm != "H32_noLSTM"))
    for y in hold[:FAST]:
        m.code_byte(int(y))
    Bs = hold[HALF:HALF + FAST]
    bpb = sum(m.code_byte(int(y)) for y in Bs) / len(Bs)
    print(json.dumps(dict(arm=arm, bpb=bpb, sec=round(time.time() - t0))), flush=True)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rows = [l.split("\t") for l in open(f"{root}/bench/eval/gen_prompts_dev.tsv", encoding="utf-8").read().split("\n")[1:] if l]
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"MODEL\tglnb\t{arm}\n")
        for i, r in enumerate(rows[:nprompts]):
            p = unesc(r[2])
            g = generate(m, p, np.random.default_rng(20261101 + i), temp)
            f.write(f"PROMPT\t{i}\t{esc(p)}\nGEN\ttrie_T{temp}\t{esc(g)}\n")
            f.flush()
    print(json.dumps(dict(arm=arm, temp=temp, prompts=nprompts, out=out, sec=round(time.time() - t0))), flush=True)


def stream_main(base, hold, out, fb_path):
    """Learn and predict online over the whole 1 MiB holdout, in order, and write the model's probability of every byte
    (float32), for the content mixer to read as one more expert (GENERATION_PLAN §13.15). With MIXER_P (the mixer's
    probability of every holdout byte, float32), the mixer feeds back: two more side values gate every neuron, the
    mixer's recent loss and the GLN's recent loss against it (moving averages over about 32 bytes, from bytes already
    seen), so the GLN and the mixer form one recurrent loop."""
    fb = np.fromfile(fb_path, np.float32).astype(np.float64) if fb_path else None
    m = Hybrid(base, neurons=32, extra=2 if fb is not None else 0)
    P = np.empty(len(hold), np.float32)
    lm = lg = 8.0
    t0 = time.time()
    for k, y in enumerate(hold):
        if fb is not None:
            m.extra = np.array([(lm - 2.0) / 2.0, (lg - lm) / 2.0])
        b = m.code_byte(int(y))
        P[k] = 2.0 ** -b
        if fb is not None:
            lm += (-math.log2(max(fb[k], 1e-12)) - lm) / 32.0
            lg += (b - lg) / 32.0
        if k % 65536 == 65535:
            print(json.dumps(dict(bytes=k + 1, sec=round(time.time() - t0))), flush=True)
    P.tofile(out)
    h = len(hold) // 2
    print(json.dumps(dict(out=out, bpb_second_half=float(-np.log2(P[h:].astype(np.float64)).mean()),
                          feedback=bool(fb_path), sec=round(time.time() - t0))), flush=True)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/root/cdata/enwik8"
    raw = np.fromfile(path, np.uint8, count=TRAIN + HOLD)
    train, hold = raw[:TRAIN], raw[TRAIN:]
    A, B = hold[:FAST], hold[HALF:HALF + FAST]
    t0 = time.time()
    base = pretrain(train)
    print("pretrained", round(time.time() - t0), "s", flush=True)
    if len(sys.argv) > 3 and sys.argv[2] == "stream":  # glnb.py ENWIK8 stream OUT.f32 [MIXER_P.f32]
        return stream_main(base, hold, sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    if len(sys.argv) > 6 and sys.argv[2] == "gen":  # writing: glnb.py ENWIK8 gen ARM TEMP NPROMPTS OUT.gen.txt
        return gen_main(base, hold, sys.argv[3], float(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
    if len(sys.argv) > 4 and sys.argv[2] == "hs":  # context capacity: glnb.py ENWIK8 hs HALF_SPACES BYTES
        hs, size = int(sys.argv[3]), int(sys.argv[4])
        t1 = time.time()
        m = Hybrid(base, neurons=32, hs=hs)
        for y in hold[:size]:
            m.code_byte(int(y))
        Bs = hold[HALF:HALF + size]
        bits = sum(m.code_byte(int(y)) for y in Bs)
        r = dict(bpb=bits / len(Bs), sec=round(time.time() - t1), neurons=32, hs=hs, bytes=size)
        print(json.dumps(r), flush=True)
        out = os.environ.get("GLNB_OUT", "/root/work/glnx")  # GLNB_OUT: where the arm's JSON goes
        os.makedirs(out, exist_ok=True)
        json.dump(r, open(os.path.join(out, f"glnb_hs{hs}_{size}.json"), "w"))
        return
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
