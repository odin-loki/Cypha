#!/usr/bin/env python3
"""N0, the neural diagnostic (docs/GENERATION_PLAN.md section 13.3). Driven by bench/nn_diag/run_n0.sh.

The neural models here are instruments only. They are never part of CyphaLM, never a default, and their text is
never training text. They are deleted or archived after the report. N0 reads the GEN-A dev prompts only.

    nn_diag.py data     --docs DOCS.jsonl --dev F --out DIR --train-bytes N --val-bytes M
    nn_diag.py train    --arm m5|tr --dir DIR --threads T [--tiny]
    nn_diag.py valbpb   --arm m5|tr --dir DIR --threads T
    nn_diag.py serve    --arm m5|tr --dir DIR --threads T -- CMD...    answers CMD's distribution requests on its pipes
    nn_diag.py measures --dir DIR --arms K5,K8,M5,TR                    copy share, 12-word cap, invented words
    nn_diag.py summary  --dir DIR --arms K5,K8,M5,TR --out REPORT.md    tables, paired bootstraps, predictions N1-N7

Author: Odin Loch
"""
import argparse
import json
import math
import os
import re
import struct
import subprocess
import sys
import time

import numpy as np

V = 8192
P64 = np.uint64(0x100000001B3)


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# ------------------------------------------------------------------ text helpers

def unesc_tsv(s):  # gen_prompts_*.tsv: \t \n \\ escaped
    out, i = [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            out.append({"t": "\t", "n": "\n"}.get(s[i + 1], s[i + 1]))
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def unesc_gen(s):  # cyphalm's printable(): \n and \xHH; anything else literal (scribe's unescape_gen)
    b = s.encode("latin-1", "replace") if False else s.encode("utf-8", "surrogateescape")
    out, i = bytearray(), 0
    hexd = b"0123456789abcdef"
    while i < len(b):
        if b[i] == 92 and i + 1 < len(b) and b[i + 1] == 110:
            out.append(10)
            i += 2
        elif b[i] == 92 and i + 3 < len(b) and b[i + 1] == 120:
            h = lambda c: hexd.find(bytes([c])) if hexd.find(bytes([c])) >= 0 else 0
            out.append((h(b[i + 2]) << 4) | h(b[i + 3]))
            i += 4
        else:
            out.append(b[i])
            i += 1
    return bytes(out)


def read_gen(path):
    """[(prompt_id, prompt_bytes, {kind: gen_bytes})] in file order."""
    rows, cur = [], None
    for line in open(path, encoding="utf-8", errors="surrogateescape"):
        f = line.rstrip("\n").split("\t", 2)
        if f[0] == "PROMPT":
            cur = (int(f[1]), unesc_gen(f[2]), {})
            rows.append(cur)
        elif f[0] == "GEN":
            cur[2][f[1]] = unesc_gen(f[2])
    return rows


def roll_hash(buf, K):
    """64-bit polynomial hash of every K-byte window of buf (uint8 array)."""
    n = len(buf) - K + 1
    if n <= 0:
        return np.zeros(0, dtype=np.uint64)
    b = buf.astype(np.uint64)
    h = np.zeros(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(K):
            h = h * P64 + b[j:j + n] + np.uint64(1)
    return h


def member(x, H):
    """x in H for a sorted unique uint64 array H (np.isin re-sorts H on every call)."""
    if len(H) == 0 or len(x) == 0:
        return np.zeros(len(x), dtype=bool)
    i = np.minimum(np.searchsorted(H, x), len(H) - 1)
    return H[i] == x


def read_ids(path):
    return np.fromfile(path, dtype=np.uint16).astype(np.int64)


def load_tok_pieces(path):
    with open(path, "rb") as f:
        head = f.readline().split()
        n = int(head[3])
        pieces = []
        for _ in range(n):
            x = f.readline().split()
            pieces.append(bytes.fromhex(x[0].decode()) if x else b"")
    return pieces


# ------------------------------------------------------------------ data (W1)

def cmd_data(a):
    os.makedirs(a.out, exist_ok=True)
    K = 32
    spans = set()
    with open(a.dev, encoding="utf-8") as f:
        next(f)
        for line in f:
            c = unesc_tsv(line.rstrip("\n").split("\t")[3]).encode("utf-8")
            for i in range(len(c) - K + 1):
                spans.add(c[i:i + K])
    dev_h = np.unique(np.array([int(roll_hash(np.frombuffer(s, dtype=np.uint8), K)[0]) for s in spans], dtype=np.uint64))
    log("dev continuation spans: %d" % len(spans))
    need = a.train_bytes + a.val_bytes
    kept, dropped, total, pages = [], 0, 0, 0
    batch = []

    def flush(batch):
        nonlocal dropped, total
        buf = np.frombuffer(b"\n".join(batch), dtype=np.uint8)
        hits = np.nonzero(member(roll_hash(buf, K), dev_h))[0]
        bad = set()
        if len(hits):
            starts = np.cumsum([0] + [len(x) + 1 for x in batch[:-1]])
            for p in hits:
                s = bytes(buf[p:p + K])
                if s in spans:
                    bad.add(int(np.searchsorted(starts, p, side="right") - 1))
        for i, d in enumerate(batch):
            if i in bad:
                dropped += 1
            elif total < need:
                kept.append(d)
                total += len(d) + 1

    for line in open(a.docs, encoding="utf-8"):
        t = json.loads(line)["text"].replace("\n", " ").encode("utf-8")
        pages += 1
        batch.append(t)
        if sum(len(x) for x in batch) > 32 << 20:
            flush(batch)
            batch = []
            if total >= need:
                break
    if batch and total < need:
        flush(batch)
    text = b"\n".join(kept) + b"\n"
    if len(text) < need:
        sys.exit("not enough clean text: %d < %d" % (len(text), need))
    # Train: the first train_bytes, cut back to a page end; val: the next val_bytes, cut back to a page end.
    tcut = text.rindex(b"\n", 0, a.train_bytes) + 1
    vcut = text.rindex(b"\n", tcut, tcut + a.val_bytes) + 1
    open(os.path.join(a.out, "train.txt"), "wb").write(text[:tcut])
    open(os.path.join(a.out, "val.txt"), "wb").write(text[tcut:vcut])
    st = dict(pages_read=pages, pages_dropped=dropped, train_bytes=tcut, val_bytes=vcut - tcut,
              train_pages=text[:tcut].count(b"\n"), val_pages=text[tcut:vcut].count(b"\n"))
    json.dump(st, open(os.path.join(a.out, "data.json"), "w"), indent=1)
    log("data: %s" % st)


# ------------------------------------------------------------------ models

def torch_setup(threads):
    import torch
    torch.set_num_threads(threads)
    torch.manual_seed(20261101)
    return torch


def make_model(torch, arm):
    nn = torch.nn
    F = torch.nn.functional

    class M5(nn.Module):  # Bengio et al. 2003: 4 tokens -> embeddings 256 -> hidden 1,024 (tanh) -> softmax
        ctx = 4

        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(V, 256)
            self.hid = nn.Linear(4 * 256, 1024)
            self.out = nn.Linear(1024, V)

        def forward(self, x):  # x: (B, 4) -> logits (B, V)
            return self.out(torch.tanh(self.hid(self.emb(x).flatten(1))))

    class Block(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            self.h = h
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
            self.fc, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

        def forward(self, x):
            B, T, D = x.shape
            q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D)
            x = x + self.proj(y)
            return x + self.fc2(F.gelu(self.fc(self.ln2(x))))

    class TR(nn.Module):  # 6 layers, d 384, 6 heads, pre-LN, tied embeddings, context 256
        ctx = 256

        def __init__(self, L=6, d=384, h=6):
            super().__init__()
            self.tok = nn.Embedding(V, d)
            self.pos = nn.Embedding(256, d)
            self.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
            self.lnf = nn.LayerNorm(d)
            self.apply(self._init)
            for b in self.blocks:
                for p in (b.proj.weight, b.fc2.weight):
                    nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * L))

        @staticmethod
        def _init(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

        def forward(self, x):  # x: (B, T) -> logits (B, T, V)
            T = x.shape[1]
            h = self.tok(x) + self.pos(torch.arange(T))
            for b in self.blocks:
                h = b(h)
            return self.lnf(h) @ self.tok.weight.t()

    return M5() if arm == "m5" else TR()


HP = {  # AdamW, warmup then cosine to 10%; two passes over train for both nets
    "m5": dict(batch=2048, lr=1e-3, wd=0.01, warmup=500, evals=24),
    "tr": dict(batch=64, lr=1e-3, wd=0.1, warmup=200, evals=24),
}


def m5_ctx(ids, pos, pad=10):
    """(len(pos), 4) contexts ids[p-4..p-1], padded with the newline token before the stream starts."""
    idx = pos[:, None] + np.arange(-4, 0)[None, :]
    c = ids[np.clip(idx, 0, None)]
    c[idx < 0] = pad
    return c


def val_loss(torch, arm, model, vids, limit):
    """Mean nats per token on the first `limit` val tokens (checkpoint choice)."""
    F = torch.nn.functional
    model.eval()
    tot, n = 0.0, 0
    with torch.inference_mode():
        if arm == "m5":
            pos = np.arange(min(limit, len(vids)))
            for s in range(0, len(pos), 8192):
                p = pos[s:s + 8192]
                lg = model(torch.from_numpy(m5_ctx(vids, p)))
                tot += F.cross_entropy(lg, torch.from_numpy(vids[p]), reduction="sum").item()
                n += len(p)
        else:
            T = 256
            nw = min(limit, len(vids) - 1) // T
            for s in range(0, nw, 32):
                w = np.arange(s, min(nw, s + 32)) * T
                x = vids[w[:, None] + np.arange(T)[None, :]]
                y = vids[w[:, None] + np.arange(1, T + 1)[None, :]]
                lg = model(torch.from_numpy(x))
                tot += F.cross_entropy(lg.reshape(-1, V), torch.from_numpy(y).reshape(-1), reduction="sum").item()
                n += y.size
    model.train()
    return tot / max(n, 1)


def cmd_train(a):
    torch = torch_setup(a.threads)
    F = torch.nn.functional
    d = a.dir
    ids, vids = read_ids(os.path.join(d, "train.u16")), read_ids(os.path.join(d, "val.u16"))
    hp = dict(HP[a.arm])
    if a.tiny:
        hp.update(batch=max(8, hp["batch"] // 8), warmup=10, evals=3)
    model = make_model(torch, a.arm)
    nparam = sum(p.numel() for p in model.parameters())
    B = hp["batch"]
    if a.arm == "m5":
        units = np.arange(4, len(ids))          # one target per example
    else:
        T = 256
        units = np.arange((len(ids) - 1) // T)  # non-overlapping windows of 256 targets
    per_epoch = len(units) // B
    passes = 1 if a.tiny else 2
    steps = per_epoch * passes
    if a.tiny:
        steps = min(steps, a.tiny_steps)
    decay, other = [], []
    for n_, p in model.named_parameters():
        (decay if p.dim() >= 2 and "emb" not in n_ and "pos" not in n_ and n_ != "tok.weight" else other).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": hp["wd"]}, {"params": other, "weight_decay": 0.0}],
                            lr=hp["lr"], betas=(0.9, 0.95))
    last, best = os.path.join(d, a.arm + ".last.pt"), os.path.join(d, a.arm + ".best.pt")
    step, best_loss, hist = 0, float("inf"), []
    if os.path.exists(last):
        ck = torch.load(last, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, best_loss, hist = ck["step"], ck["best_loss"], ck["hist"]
        log("resumed %s at step %d" % (a.arm, step))
    eval_every = max(1, steps // hp["evals"])
    vlimit = 32768 if a.tiny else 262144
    log("train %s: %d params, %d train tokens, batch %d, %d steps (%d passes), eval every %d"
        % (a.arm, nparam, len(ids), B, steps, passes, eval_every))
    json.dump(dict(arm=a.arm, params=nparam, steps=steps, batch=B, hp=hp, train_tokens=int(len(ids))),
              open(os.path.join(d, a.arm + ".config.json"), "w"), indent=1)
    perm, perm_epoch = None, -1
    t0, tl, tn = time.time(), 0.0, 0
    model.train()
    while step < steps:
        ep, b = divmod(step, per_epoch)
        if ep != perm_epoch:
            perm = np.random.default_rng(20261101 + ep).permutation(units)
            perm_epoch = ep
        u = perm[b * B:(b + 1) * B]
        lr = hp["lr"] * (step + 1) / hp["warmup"] if step < hp["warmup"] else \
            hp["lr"] * (0.1 + 0.45 * (1 + math.cos(math.pi * (step - hp["warmup"]) / max(1, steps - hp["warmup"]))))
        for g in opt.param_groups:
            g["lr"] = lr
        if a.arm == "m5":
            x, y = torch.from_numpy(m5_ctx(ids, u)), torch.from_numpy(ids[u])
            loss = F.cross_entropy(model(x), y)
        else:
            w = u * 256
            x = torch.from_numpy(ids[w[:, None] + np.arange(256)[None, :]])
            y = torch.from_numpy(ids[w[:, None] + np.arange(1, 257)[None, :]])
            loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        tl, tn = tl + loss.item(), tn + 1
        if step % eval_every == 0 or step == steps:
            vl = val_loss(torch, a.arm, model, vids, vlimit)
            hist.append(dict(step=step, train=tl / tn, val=vl, lr=lr, s=time.time() - t0))
            tl, tn = 0.0, 0
            if vl < best_loss:
                best_loss = vl
                torch.save(model.state_dict(), best + ".tmp")
                os.replace(best + ".tmp", best)
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, best_loss=best_loss, hist=hist), last + ".tmp")
            os.replace(last + ".tmp", last)
            eta = (time.time() - t0) / max(1, len(hist)) * 0  # filled below
            rate = (time.time() - t0) / step if not hist[:-1] else None
            log("%s step %d/%d train %.4f val %.4f (best %.4f) lr %.2e %.0f s" % (
                a.arm, step, steps, hist[-1]["train"], vl, best_loss, lr, time.time() - t0))
            with open(os.path.join(d, a.arm + ".progress"), "w") as f:
                f.write("%s step %d/%d val %.4f best %.4f elapsed %.0f s\n" % (a.arm, step, steps, vl, best_loss, time.time() - t0))
    json.dump(hist, open(os.path.join(d, a.arm + ".hist.json"), "w"), indent=1)
    open(os.path.join(d, a.arm + ".trained"), "w").write("best val nats/token %.5f\n" % best_loss)


def load_best(torch, arm, d):
    m = make_model(torch, arm)
    m.load_state_dict(torch.load(os.path.join(d, arm + ".best.pt"), weights_only=True))
    m.eval()
    return m


def cmd_valbpb(a):
    """Bits per byte on the whole val text: TR with windows of 256 at stride 128 (each token scored once, with at
    least 128 tokens of context after the first window); M5 with its 4 tokens."""
    torch = torch_setup(a.threads)
    F = torch.nn.functional
    d = a.dir
    vids = read_ids(os.path.join(d, "val.u16"))
    nbytes = os.path.getsize(os.path.join(d, "val.txt"))
    model = load_best(torch, a.arm, d)
    nats = 0.0
    with torch.inference_mode():
        # token 0 has no history: scored by the arm's own empty-context guess is not defined; every arm skips it
        if a.arm == "m5":
            pos = np.arange(1, len(vids))
            for s in range(0, len(pos), 8192):
                p = pos[s:s + 8192]
                nats += F.cross_entropy(model(torch.from_numpy(m5_ctx(vids, p))), torch.from_numpy(vids[p]), reduction="sum").item()
        else:
            T, S = 256, 128
            starts = list(range(0, max(1, len(vids) - T), S))
            if starts[-1] + T < len(vids):
                starts.append(len(vids) - T)
            scored = 1  # positions [0, scored) done (token 0 skipped)
            for i in range(0, len(starts), 16):
                ws = starts[i:i + 16]
                x = torch.from_numpy(np.stack([vids[w:w + T] for w in ws]))
                lp = F.log_softmax(model(x), dim=-1)
                for j, w in enumerate(ws):
                    lo = max(scored, w + 1)
                    hi = w + T
                    if hi + 0 > len(vids) - 0:
                        hi = len(vids)
                    if lo >= hi:
                        continue
                    tgt = torch.from_numpy(vids[lo:hi])
                    nats -= lp[j, lo - w - 1:hi - w - 1].gather(1, tgt[:, None]).sum().item()
                    scored = hi
    bits = nats / math.log(2)
    out = dict(arm=a.arm, tokens=int(len(vids) - 1), bits=bits, bytes=nbytes, bpb=bits / nbytes)
    json.dump(out, open(os.path.join(d, a.arm.upper() + ".valbpb.json"), "w"), indent=1)
    log("valbpb %s" % out)


def cmd_serve(a):
    """Runs CMD (cyphalm_tok_gen n0-gen ... --pipe) and answers each request (int32 n, n uint32 ids) with the
    arm's next-token distribution (V float64)."""
    torch = torch_setup(a.threads)
    model = load_best(torch, a.arm, a.dir)
    ctx = model.ctx
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    rd, wr = p.stdout, p.stdin
    nreq = 0
    with torch.inference_mode():
        while True:
            hdr = rd.read(4)
            if len(hdr) < 4:
                break
            n = struct.unpack("<i", hdr)[0]
            h = np.frombuffer(rd.read(4 * n), dtype=np.uint32).astype(np.int64)
            if a.arm == "m5":
                x = torch.from_numpy(m5_ctx(h, np.array([len(h)])))
                lg = model(x)[0]
            else:
                x = torch.from_numpy(h[-ctx:][None, :])
                lg = model(x)[0, -1]
            q = torch.softmax(lg.double(), dim=-1).numpy()
            wr.write(q.astype("<f8").tobytes())
            wr.flush()
            nreq += 1
    rc = p.wait()
    log("serve %s: %d requests, exit %d" % (a.arm, nreq, rc))
    sys.exit(rc)


# ------------------------------------------------------------------ Q0 (§13.5): a yardstick only

def printable(b):
    o = []
    for c in b:
        if c == 10:
            o.append("\\n")
        elif 32 <= c < 127:
            o.append(chr(c))
        else:
            o.append("\\x%02x" % c)
    return "".join(o)


def cut600(b, n=600):
    if len(b) <= n:
        return b
    b = b[:n]
    while b and (b[-1] & 0xC0) == 0x80:  # do not end inside a UTF-8 character
        b = b[:-1]
    if b and b[-1] >= 0xC0:
        b = b[:-1]
    return b


def cmd_q0(a):
    """Qwen 3.5 9B by Ollama, raw completion (no template, thinking off), nucleus 0.9, no other sampler rule,
    seed 1000003 (k + 1) + 7919 t. Its text is scored and nothing else: never training text, tables or states."""
    import urllib.request
    assert "gen_prompts_test" not in a.dev
    prompts = []
    with open(a.dev, encoding="utf-8") as f:
        next(f)
        for line in f:
            prompts.append(unesc_tsv(line.rstrip("\n").split("\t")[2]))
    if a.limit >= 0:
        prompts = prompts[:a.limit]
    temps = [float(x) for x in a.temps.split(",")]
    os.makedirs(os.path.join(a.dir, "gen"), exist_ok=True)
    path = os.path.join(a.dir, "gen", "Q0.gen.txt")
    done = {}
    if os.path.exists(path + ".part"):
        for pid, _, g in read_gen(path + ".part"):
            if len(g) == len(temps):
                done[pid] = g
    t0 = time.time()
    with open(path + ".part", "w", encoding="utf-8") as out:
        out.write("MODEL\tq0\t%s\traw completion, top_p 0.9, top_k 0, no repeat penalty, cut at 600 bytes\n" % a.model)
        for k, pr in enumerate(prompts):
            out.write("PROMPT\t%d\t%s\n" % (k, printable(pr.encode("utf-8"))))
            for ti, T in enumerate(temps):
                kind = "trie_T%.1f" % T
                if k in done:
                    text = done[k][kind]
                else:
                    body = dict(model=a.model, prompt=pr, raw=True, stream=False, think=False,
                                options=dict(temperature=T, top_p=0.9, top_k=0, min_p=0.0, repeat_penalty=1.0,
                                             seed=1000003 * (k + 1) + 7919 * ti, num_predict=300, num_ctx=2048))
                    req = urllib.request.Request(a.url + "/api/generate", data=json.dumps(body).encode(),
                                                 headers={"Content-Type": "application/json"})
                    for attempt in range(5):
                        try:
                            r = json.loads(urllib.request.urlopen(req, timeout=600).read())
                            break
                        except Exception as e:
                            log("q0 retry %d: %s" % (attempt, e))
                            time.sleep(10)
                    else:
                        sys.exit("q0: the server does not answer")
                    text = cut600(r["response"].encode("utf-8"))
                out.write("GEN\t%s\t%s\n" % (kind, printable(text)))
            out.flush()
            if (k + 1) % 25 == 0:
                log("q0 %d/%d prompts (%.0f s)" % (k + 1, len(prompts), time.time() - t0))
    os.replace(path + ".part", path)


# ------------------------------------------------------------------ measures

def cmd_measures(a):
    d = a.dir
    train = open(os.path.join(d, "train.txt"), "rb").read()
    # 32-byte spans of the training text
    tb = np.frombuffer(train, dtype=np.uint8)
    H = np.unique(roll_hash(tb, 32))
    log("train spans: %d distinct" % len(H))
    # words: letter runs, lower case (cyphalm_tok_gen words); the training vocabulary
    seen = set(w.lower() for w in re.findall(rb"[A-Za-z]+", train))
    # 13-word windows (whitespace words) of the training text
    words = train.split()
    vocab = {}
    wid = np.fromiter((vocab.setdefault(w, len(vocab)) for w in words), dtype=np.int64, count=len(words))
    del words
    W = 13

    def wgram(ids):
        n = len(ids) - W + 1
        if n <= 0:
            return np.zeros(0, dtype=np.uint64)
        u = ids.astype(np.uint64)
        h = np.zeros(n, dtype=np.uint64)
        with np.errstate(over="ignore"):
            for j in range(W):
                h = h * np.uint64(0x9E3779B97F4A7C15) + u[j:j + n] + np.uint64(1)
        return h

    WG = np.unique(wgram(wid))
    log("train 13-word windows: %d distinct, vocabulary %d" % (len(WG), len(vocab)))
    res = {}
    for arm in a.arms.split(","):
        rows = read_gen(os.path.join(d, "gen", arm + ".gen.txt"))
        by = {}
        for pid, _, g in rows:
            for kind, text in g.items():
                e = by.setdefault(kind, dict(cov=0, n=0, words=0, inv=0, samples=0, cap=0, per=[]))
                buf = np.frombuffer(text, dtype=np.uint8)
                cov = np.zeros(len(buf), dtype=bool)
                if len(buf) >= 32:
                    hit = np.nonzero(member(roll_hash(buf, 32), H))[0]
                    for p in hit:
                        cov[p:p + 32] = True
                ws = [w.lower() for w in re.findall(rb"[A-Za-z]+", text)]
                inv = sum(1 for w in ws if w not in seen)
                sw = text.split()
                ids = np.array([vocab.get(w, -1 - i) for i, w in enumerate(sw)], dtype=np.int64)
                ok = ids >= 0
                capped = False
                if len(ids) >= W:
                    hw = wgram(np.where(ok, ids, 0))
                    valid = np.convolve(ok.astype(int), np.ones(W, dtype=int), "valid") == W
                    capped = bool(np.any(valid & member(hw, WG)))
                e["cov"] += int(cov.sum())
                e["n"] += len(buf)
                e["words"] += len(ws)
                e["inv"] += inv
                e["samples"] += 1
                e["cap"] += int(capped)
                e["per"].append((pid, int(cov.sum()), len(buf)))
        tot = dict(cov=0, n=0, words=0, inv=0, samples=0, cap=0)
        out = {}
        for kind, e in sorted(by.items()):
            for k in tot:
                tot[k] += e[k]
            out[kind] = dict(copy32=e["cov"] / max(1, e["n"]), invented=e["inv"] / max(1, e["words"]),
                             cap12=e["cap"] / max(1, e["samples"]), samples=e["samples"], gen_bytes=e["n"])
        out["pooled"] = dict(copy32=tot["cov"] / max(1, tot["n"]), invented=tot["inv"] / max(1, tot["words"]),
                             cap12=tot["cap"] / max(1, tot["samples"]), samples=tot["samples"], gen_bytes=tot["n"])
        res[arm] = out
        log("measures %s: %s" % (arm, out["pooled"]))
    json.dump(res, open(os.path.join(d, "measures.json"), "w"), indent=1)


# ------------------------------------------------------------------ summary

def rung_table(md):
    out = {}
    for line in md.splitlines():
        m = re.match(r"\| (.+?) \| (\d+) \| ([0-9.]+) \[([0-9.]+), ([0-9.]+)\] \| ([0-9.]+) \| ([0-9.]+) \[([0-9.]+), ([0-9.]+)\] \| (\d+) \|", line)
        if m and m.group(1) not in out:
            out[m.group(1)] = dict(n=int(m.group(2)), scribe=float(m.group(3)), lo=float(m.group(4)), hi=float(m.group(5)),
                                   S=float(m.group(7)), Slo=float(m.group(8)), Shi=float(m.group(9)), bytes=int(m.group(10)))
    return out


def per_sample(d, arm):
    """{(prompt_id, kind): v2 dict} from the ladder's export (ids name the GEN line)."""
    rows = read_gen(os.path.join(d, "gen", arm + ".gen.txt"))
    line_of = []
    for pid, _, g in rows:
        for kind in g:
            line_of.append((pid, kind))
    out = {}
    for line in open(os.path.join(d, "scribe", arm + ".export", "scores.jsonl"), encoding="utf-8"):
        j = json.loads(line)
        m = re.match(r".*/trie/g(\d+)$", j["id"])
        if j.get("system") == "trie" and m:
            out[line_of[int(m.group(1))]] = j["v2"]
    return out


def cont_bpb(d, arm, idx=None):
    rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(d, "gen", arm + ".gen.txt.cont.tsv"))][1:]
    rows = {int(r[0]): (int(r[1]), float(r[2])) for r in rows}
    return rows


def boot_ci(vals, B=2000, seed=20261029):
    rng = np.random.default_rng(seed)
    return np.percentile(vals, [2.5, 97.5]) if len(vals) else (float("nan"), float("nan"))


def paired_boot(sa, sb, B=2000, seed=20261029):
    """Pooled mean of B minus pooled mean of A, resampling prompts; sa/sb: {prompt: [scores]}."""
    P = sorted(set(sa) | set(sb))
    sumA = np.array([sum(sa.get(p, [])) for p in P]); nA = np.array([len(sa.get(p, [])) for p in P])
    sumB = np.array([sum(sb.get(p, [])) for p in P]); nB = np.array([len(sb.get(p, [])) for p in P])
    pt = sumB.sum() / nB.sum() - sumA.sum() / nA.sum()
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(P), len(P))
        bs.append(sumB[i].sum() / max(1, nB[i].sum()) - sumA[i].sum() / max(1, nA[i].sum()))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return pt, lo, hi


def cmd_summary(a):
    d = a.dir
    arms = a.arms.split(",")
    R = {}
    meas = json.load(open(os.path.join(d, "measures.json")))
    for arm in arms:
        md = open(os.path.join(d, "scribe", arm + ".scribe.md"), encoding="utf-8").read()
        rt = rung_table(md)
        ps = per_sample(d, arm)
        by_prompt, axes = {}, {}
        for (pid, kind), v in ps.items():
            by_prompt.setdefault(pid, []).append(v["scribe"])
            for ax in ("D", "DL", "DS", "DC", "DI"):
                axes.setdefault((kind, ax), []).append(v[ax])
                axes.setdefault(("pooled", ax), []).append(v[ax])
        cont, val = None, None
        if os.path.exists(os.path.join(d, "gen", arm + ".gen.txt.cont.tsv")):
            cb = cont_bpb(d, arm)
            nb = np.array([cb[k][0] for k in sorted(cb)]); bits = np.array([cb[k][1] for k in sorted(cb)])
            rng = np.random.default_rng(20261029)
            bs = []
            for _ in range(2000):
                i = rng.integers(0, len(nb), len(nb))
                bs.append(bits[i].sum() / nb[i].sum())
            cont = (bits.sum() / nb.sum(), *np.percentile(bs, [2.5, 97.5]))
        if os.path.exists(os.path.join(d, arm + ".valbpb.json")):
            val = json.load(open(os.path.join(d, arm + ".valbpb.json")))["bpb"]
        R[arm] = dict(rt=rt, by_prompt=by_prompt, axes={k: float(np.mean(v)) for k, v in axes.items()},
                      cont=cont, val=val, m=meas[arm])
    pooled = lambda arm: R[arm]["rt"].get("trie samples (pooled)", {})
    dsim = paired_boot(R["K5"]["by_prompt"], R["M5"]["by_prompt"]) if "M5" in R else None
    dstate = paired_boot(R["M5"]["by_prompt"], R["TR"]["by_prompt"]) if "TR" in R else None
    dk8 = paired_boot(R["K5"]["by_prompt"], R["K8"]["by_prompt"]) if "K8" in R else None
    L = []
    L.append("## Results\n")
    L.append("### The run\n")
    dj = json.load(open(os.path.join(d, "data.json")))
    ntr = os.path.getsize(os.path.join(d, "train.u16")) // 2
    L.append("- **Text.** %d pages read, %d dropped (a 32-byte span of a dev continuation). Train %s bytes (%d pages), val %s bytes (%d pages)." % (
        dj["pages_read"], dj["pages_dropped"], format(dj["train_bytes"], ","), dj["train_pages"], format(dj["val_bytes"], ","), dj["val_pages"]))
    L.append("- **Tokens.** Train %s tokens, %.3f bytes per token." % (format(ntr, ","), dj["train_bytes"] / ntr))
    for arm in ("m5", "tr"):
        cf, hf = os.path.join(d, arm + ".config.json"), os.path.join(d, arm + ".hist.json")
        if os.path.exists(cf) and os.path.exists(hf):
            c, h = json.load(open(cf)), json.load(open(hf))
            b = min(h, key=lambda x: x["val"])
            L.append("- **%s.** %s parameters, batch %d, %d steps, %.1f h. Best val loss %.4f nats per token at step %d (of %d checks); last %.4f." % (
                arm.upper(), format(c["params"], ","), c["batch"], c["steps"], h[-1]["s"] / 3600, b["val"], b["step"], len(h), h[-1]["val"]))
    L.append("")
    L.append("### Bits per byte\n")
    L.append("| Arm | Val (bits per byte) | GEN-A dev continuations [95%] |")
    L.append("|---|---:|---|")
    for arm in arms:
        c = R[arm]["cont"]
        if c is None or R[arm]["val"] is None:
            continue
        L.append("| %s | %.4f | %.4f [%.4f, %.4f] |" % (arm, R[arm]["val"], c[0], c[1], c[2]))
    L.append("\n### SCRIBE v2 on GEN-A dev (250 prompts)\n")
    L.append("| Arm | Pooled [95%] | T 0.8 [95%] | T 1.0 [95%] | Surface S pooled [95%] | Kept | Mean bytes |")
    L.append("|---|---|---|---|---|---:|---:|")
    for arm in arms:
        rt = R[arm]["rt"]
        f = lambda r: "%.3f [%.3f, %.3f]" % (rt[r]["scribe"], rt[r]["lo"], rt[r]["hi"]) if r in rt else "–"
        p = rt.get("trie samples (pooled)", {})
        L.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            arm, f("trie samples (pooled)"), f("trie samples T=0.8"), f("trie samples T=1.0"),
            "%.3f [%.3f, %.3f]" % (p["S"], p["Slo"], p["Shi"]) if p else "–", p.get("n", "–"), p.get("bytes", "–")))
    L.append("\n### SCRIBE's axes (mean over kept samples; lower is better)\n")
    L.append("| Arm | Rung | D | D_L | D_S | D_C | D_I |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for arm in arms:
        for kind in ("pooled", "trie_T0.8", "trie_T1.0"):
            ax = R[arm]["axes"]
            if (kind, "D") in ax:
                L.append("| %s | %s | %.2f | %.2f | %.2f | %.2f | %.2f |" % (
                    arm, kind.replace("trie_", ""), ax[(kind, "D")], ax[(kind, "DL")], ax[(kind, "DS")], ax[(kind, "DC")], ax[(kind, "DI")]))
    L.append("\n### Copying and words\n")
    L.append("| Arm | Copy share (32 B) pooled | T 0.8 | T 1.0 | Samples over the 12-word cap | Invented words |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for arm in arms:
        m = R[arm]["m"]
        L.append("| %s | %.4f | %.4f | %.4f | %.3f | %.4f |" % (
            arm, m["pooled"]["copy32"], m.get("trie_T0.8", {}).get("copy32", float("nan")),
            m.get("trie_T1.0", {}).get("copy32", float("nan")), m["pooled"]["cap12"], m["pooled"]["invented"]))
    L.append("\n### The two differences (pooled dev SCRIBE, paired bootstrap over prompts, 2,000 draws)\n")
    L.append("| Difference | Point | 95% |")
    L.append("|---|---:|---|")
    for name, x in (("Δsim = M5 − K5", dsim), ("Δstate = TR − M5", dstate), ("K8 − K5", dk8)):
        if x:
            L.append("| %s | %+.3f | [%+.3f, %+.3f] |" % (name, *x))
    # predictions
    S = {arm: pooled(arm).get("scribe", float("nan")) for arm in arms}
    ax = lambda arm, k: R[arm]["axes"].get(("pooled", k), float("nan"))
    pr = []
    if all(x in R for x in ("K5", "M5", "TR")):
        g1, g2 = R["K5"]["val"] - R["M5"]["val"], R["M5"]["val"] - R["TR"]["val"]
        pr.append(("N1", "Val bits per byte: M5 below K5 by ≥ 0.05; TR below M5 by ≥ 0.15",
                   "K5 − M5 = %.4f; M5 − TR = %.4f" % (g1, g2), g1 >= 0.05 and g2 >= 0.15))
    if "K8" in R:
        pr.append(("N2", "K8's SCRIBE within ±0.02 of K5's", "K8 − K5 = %+.3f" % (S["K8"] - S["K5"]), abs(S["K8"] - S["K5"]) <= 0.02))
    if "M5" in R:
        ds = (ax("K5", "DS") - ax("M5", "DS")) / ax("K5", "DS") if ax("K5", "DS") else float("nan")
        pr.append(("N3", "M5's SCRIBE ≥ K5's + 0.05, with D_S at least 15% lower",
                   "M5 − K5 = %+.3f; D_S lower by %.1f%%" % (S["M5"] - S["K5"], 100 * ds), S["M5"] - S["K5"] >= 0.05 and ds >= 0.15))
    if "TR" in R:
        dD = ax("M5", "D") - ax("TR", "D")
        dSC = (ax("M5", "DS") - ax("TR", "DS")) + (ax("M5", "DC") - ax("TR", "DC"))
        share = dSC / dD if dD > 0 else float("nan")
        pr.append(("N4", "TR's SCRIBE ≥ M5's + 0.10, mostly from D_S and D_C",
                   "TR − M5 = %+.3f; D_S + D_C give %.0f%% of the fall in D" % (S["TR"] - S["M5"], 100 * share),
                   S["TR"] - S["M5"] >= 0.10 and share > 0.5))
        pr.append(("N5", "TR's dev SCRIBE in [0.30, 0.60]", "TR = %.3f" % S["TR"], 0.30 <= S["TR"] <= 0.60))
        pr.append(("N6", "Δstate > Δsim", "Δstate %+.3f, Δsim %+.3f" % (dstate[0], dsim[0]), dstate[0] > dsim[0]))
        cs = {arm: R[arm]["m"]["pooled"]["copy32"] for arm in arms if arm in meas}
        pr.append(("N7", "The neural arms copy less than the count arms (32-byte copy share)",
                   "K5 %.4f, K8 %.4f, M5 %.4f, TR %.4f" % (cs["K5"], cs["K8"], cs["M5"], cs["TR"]),
                   max(cs["M5"], cs["TR"]) < min(cs["K5"], cs["K8"])))
    L.append("\n### Predictions\n")
    L.append("| # | Prediction | Result | Scored |")
    L.append("|---|---|---|---|")
    for k, p, r, h in pr:
        L.append("| %s | %s | %s | **%s** |" % (k, p, r, "hit" if h else "miss"))
    if dsim and dstate:
        big = ("Δsim", dsim) if dsim[0] >= dstate[0] else ("Δstate", dstate)
        excl = big[1][1] > 0 or big[1][2] < 0
        L.append("\n**What it settles (§13.3's rule).** The larger difference is %s (%+.3f [%+.3f, %+.3f]). Its interval %s 0, so %s." % (
            big[0], *big[1], "excludes" if excl else "includes",
            ("P1, similarity first, is ported first" if big[0] == "Δsim" else "P2, state first, is ported first") if excl
            else "the result is reported and the port waits for Odin"))
    L.append("\n### Five samples per arm (T 0.8, the first five dev prompts)\n")
    for arm in arms:
        rows = read_gen(os.path.join(d, "gen", arm + ".gen.txt"))[:5]
        L.append("**%s**\n" % arm)
        for pid, pr_, g in rows:
            t = g.get("trie_T0.8", b"").decode("utf-8", "replace")[:300].replace("\n", " / ").replace("|", "\\|")
            L.append("- *%s* → %s" % (pr_.decode("utf-8", "replace")[-80:].replace("|", "\\|"), t))
        L.append("")
    body = "\n".join(L) + "\n"
    open(os.path.join(d, "results.md"), "w").write(body)
    head = open(a.head, encoding="utf-8").read() if a.head and os.path.exists(a.head) else ""
    head = head.replace("{{DATE}}", time.strftime("%Y-%m-%d"))
    open(a.out, "w").write(head.replace("{{RESULTS}}", body) if "{{RESULTS}}" in head else head + body)
    log("summary written to %s" % a.out)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("data"); p.add_argument("--docs"); p.add_argument("--dev"); p.add_argument("--out")
    p.add_argument("--train-bytes", type=int); p.add_argument("--val-bytes", type=int)
    for name in ("train", "valbpb", "serve"):
        p = sp.add_parser(name); p.add_argument("--arm", choices=["m5", "tr"]); p.add_argument("--dir")
        p.add_argument("--threads", type=int, default=64)
        if name == "train":
            p.add_argument("--tiny", action="store_true"); p.add_argument("--tiny-steps", type=int, default=60)
        if name == "serve":
            p.add_argument("argv_cmd", nargs=argparse.REMAINDER)
    p = sp.add_parser("measures"); p.add_argument("--dir"); p.add_argument("--arms")
    p = sp.add_parser("summary"); p.add_argument("--dir"); p.add_argument("--arms"); p.add_argument("--out")
    p.add_argument("--head", default="")
    p = sp.add_parser("q0"); p.add_argument("--dir"); p.add_argument("--dev"); p.add_argument("--url")
    p.add_argument("--model", default="qwen3.5:9b"); p.add_argument("--temps", default="0.8,1.0")
    p.add_argument("--limit", type=int, default=-1)
    a = ap.parse_args()
    fn = {"data": cmd_data, "train": cmd_train, "valbpb": cmd_valbpb, "serve": cmd_serve,
          "measures": cmd_measures, "summary": cmd_summary, "q0": cmd_q0}[a.cmd]
    if a.cmd == "serve":
        a.cmd = a.argv_cmd
    fn(a)


if __name__ == "__main__":
    main()
