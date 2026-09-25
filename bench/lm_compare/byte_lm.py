#!/usr/bin/env python3
"""Byte-level neural baselines for the CyphaLM comparison (CPU, PyTorch).

Two model families, both over raw bytes (vocab 256) like CyphaLM:
  gpt   decoder-only Transformer (pre-LN, learned positions, tied embeddings)
  lstm  stacked LSTM with state carried across batches (truncated BPTT)

Subcommands:
  train  --arch gpt|lstm --budget-min 60 --out DIR   train for a wall-clock budget
  eval   --ckpt DIR/final.pt --text F --offset O --bytes N --dump out.f32 [--dynamic-lr LR]
  gen    --ckpt DIR/final.pt --prompts prompts.json --out gens.json
  bench  --ckpt DIR/final.pt                          latency / RSS / size

eval writes float32 natural-log next-byte distributions (256 per byte), the same
layout as `cyphalm_lm_quality --dump-dist`, so metrics.py scores all models
with one code path.
"""
import argparse
import json
import math
import os
import resource
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def rss_mb(field="VmRSS:"):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field):
                return int(line.split()[1]) / 1024.0
    return 0.0


# ---------------------------------------------------------------- models
class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)

    def attn(self, x, cache=None):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        if cache is not None:
            if cache.get("k") is not None:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        causal = cache is None or t > 1
        if cache is not None and t > 1 and k.shape[2] != t:
            raise ValueError("multi-token step only from an empty cache")
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return self.proj(y.transpose(1, 2).reshape(b, t, d))

    def forward(self, x, cache=None):
        x = x + self.attn(self.ln1(x), cache)
        return x + self.fc2(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, d=256, layers=6, heads=8, ctx=512):
        super().__init__()
        self.ctx = ctx
        self.emb = nn.Embedding(256, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.lnf = nn.LayerNorm(d)
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("fc2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, caches=None, start=0):
        t = idx.shape[1]
        x = self.emb(idx) + self.pos(torch.arange(start, start + t))
        for i, blk in enumerate(self.blocks):
            x = blk(x, None if caches is None else caches[i])
        return self.lnf(x) @ self.emb.weight.T


class LSTMLM(nn.Module):
    def __init__(self, d=512, layers=2, emb=64):
        super().__init__()
        self.emb = nn.Embedding(256, emb)
        self.rnn = nn.LSTM(emb, d, layers, batch_first=True)
        self.out = nn.Linear(d, 256)

    def forward(self, idx, state=None):
        y, state = self.rnn(self.emb(idx), state)
        return self.out(y), state


def build(cfg):
    if cfg["arch"] == "gpt":
        return GPT(cfg["d"], cfg["layers"], cfg["heads"], cfg["ctx"])
    return LSTMLM(cfg["d"], cfg["layers"], cfg.get("emb", 64))


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build(ck["cfg"])
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck["cfg"]


def autocast(cfg):
    return torch.autocast("cpu", dtype=torch.bfloat16, enabled=cfg.get("bf16", True))


# ---------------------------------------------------------------- training
def cmd_train(a):
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    data = np.memmap(a.data, dtype=np.uint8, mode="r")[: a.data_bytes]
    val = np.array(np.memmap(a.data, dtype=np.uint8, mode="r")[a.val_offset: a.val_offset + a.val_bytes])
    cfg = dict(arch=a.arch, d=a.d, layers=a.layers, heads=a.heads, ctx=a.ctx, emb=a.emb, bf16=not a.fp32)
    model = build(cfg)
    nparams = sum(p.numel() for p in model.parameters())
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=False)
    os.makedirs(a.out, exist_ok=True)
    log = {"cfg": cfg, "params": nparams, "batch": a.batch, "budget_min": a.budget_min, "budget_bytes": a.budget_bytes,
           "data_bytes": len(data), "points": []}
    budget = a.budget_min * 60.0
    marks = sorted(set(x * 60.0 for x in a.marks) | {budget})
    rng = np.random.default_rng(0)
    T = a.ctx
    # LSTM: B contiguous streams through the corpus, state carried (TBPTT).
    stream_pos = (np.arange(a.batch) * (len(data) // a.batch)).astype(np.int64)
    state = None
    spent, step, seen, mark_i = 0.0, 0, 0, 0
    rss_train = 0.0
    print(json.dumps({"params": nparams, "cfg": cfg}), flush=True)
    # --budget-bytes stops after a fixed amount of text (reproducible on a busy
    # machine); otherwise the wall-clock budget decides.
    def progress():
        return seen / a.budget_bytes if a.budget_bytes > 0 else spent / budget

    while progress() < 1.0:
        t0 = time.perf_counter()
        frac = progress()
        lr = a.lr * min(1.0, (step + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr
        if a.arch == "gpt":
            ix = rng.integers(0, len(data) - T - 1, a.batch)
            chunk = np.stack([data[i: i + T + 1] for i in ix])
        else:
            chunk = np.stack([data[p: p + T + 1] for p in stream_pos])
            stream_pos += T
            if stream_pos.max() + T + 1 >= len(data):
                stream_pos = (np.arange(a.batch) * (len(data) // a.batch) + rng.integers(0, 1 << 20)) % (len(data) - 10 * T)
                state = None
        x = torch.from_numpy(chunk[:, :-1].astype(np.int64))
        y = torch.from_numpy(chunk[:, 1:].astype(np.int64))
        with autocast(cfg):
            if a.arch == "gpt":
                logits = model(x)
            else:
                logits, state = model(x, state)
                state = tuple(s.detach() for s in state)
            loss = F.cross_entropy(logits.float().reshape(-1, 256), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        seen += x.numel()
        spent += time.perf_counter() - t0
        if step % 50 == 0:
            rss_train = max(rss_train, rss_mb())
            print(json.dumps({"step": step, "min": round(spent / 60, 2), "loss_bits": loss.item() / math.log(2),
                              "lr": lr, "bytes_seen": seen, "bytes_per_s": seen / spent}), flush=True)
        done = progress() >= 1.0
        if done or (a.budget_bytes <= 0 and mark_i < len(marks) and spent >= marks[mark_i]):
            vb = val_bits(model, cfg, val)
            pt = {"minutes": round(spent / 60, 2), "step": step, "bytes_seen": seen, "val_bits_per_byte": vb}
            log["points"].append(pt)
            print(json.dumps(pt), flush=True)
            name = "final.pt" if done else f"min{int(round(marks[mark_i] / 60))}.pt"
            torch.save({"cfg": cfg, "model": model.state_dict()}, os.path.join(a.out, name))
            mark_i += 1
            model.train()
    log.update(steps=step, bytes_seen=seen, train_seconds=spent, bytes_per_s=seen / spent,
               epochs=seen / len(data), peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               rss_train_mb=rss_train)
    json.dump(log, open(os.path.join(a.out, "train_log.json"), "w"), indent=1)


@torch.no_grad()
def val_bits(model, cfg, val):
    model.eval()
    lp = logprobs(model, cfg, val, batch=16)
    y = torch.from_numpy(val.astype(np.int64))
    return float(-lp[torch.arange(len(val)), y].mean() / math.log(2))


# ---------------------------------------------------------------- scoring
@torch.no_grad()
def logprobs(model, cfg, text, batch=16):
    """Natural-log next-byte distributions for every byte of text[0:]; byte 0 is
    predicted from an empty context (a zero byte), like CyphaLM after a reset."""
    n = len(text)
    seq = np.concatenate([[0], text]).astype(np.int64)  # seq[i] predicts text[i]
    out = torch.empty(n, 256)
    if cfg["arch"] == "lstm":
        state = None
        for s in range(0, n, 4096):
            x = torch.from_numpy(seq[s: min(n, s + 4096)])[None]
            with autocast(cfg):
                lg, state = model(x, state)
            out[s: s + x.shape[1]] = F.log_softmax(lg.float()[0], -1)
        return out
    T, half = cfg["ctx"], cfg["ctx"] // 2
    inp = seq[:n]
    # Windows of up to T inputs; each scores only new positions, so every
    # position after the first window sees at least T - half bytes of context.
    jobs, covered = [], 0
    while covered < n:
        end = min(n, covered + (T if covered == 0 else half))
        st = max(0, end - T)
        jobs.append((st, inp[st:end], covered - st))
        covered = end
    for i in range(0, len(jobs), batch):
        grp = jobs[i: i + batch]
        L = max(len(w) for _, w, _ in grp)
        x = torch.zeros(len(grp), L, dtype=torch.long)
        for j, (_, w, _) in enumerate(grp):
            x[j, : len(w)] = torch.from_numpy(w)
        with autocast(cfg):
            lg = model(x)
        lg = F.log_softmax(lg.float(), -1)
        for j, (st, w, lo) in enumerate(grp):
            out[st + lo: st + len(w)] = lg[j, lo: len(w)]
    return out


def dynamic_logprobs(model, cfg, text, lr, seg=128):
    """Dynamic evaluation (Krause et al. 2018): score a segment, then take one
    gradient step on it, so the model adapts to the text as it reads it (the
    neural analogue of CyphaLM's online learning)."""
    n = len(text)
    seq = np.concatenate([[0], text]).astype(np.int64)
    out = torch.empty(n, 256)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    state = None
    ctx = cfg.get("ctx", 512)
    for s in range(0, n, seg):
        e = min(n, s + seg)
        if cfg["arch"] == "lstm":
            x = torch.from_numpy(seq[s:e])[None]
            with autocast(cfg):
                lg, new_state = model(x, state)
            lg = lg.float()[0]
            state = tuple(t.detach() for t in new_state)
        else:
            b = max(0, e - ctx)
            x = torch.from_numpy(seq[b:e])[None]
            with autocast(cfg):
                lg = model(x).float()[0, s - b:]
        lp = F.log_softmax(lg, -1)
        out[s:e] = lp.detach()
        loss = F.nll_loss(lp, torch.from_numpy(text[s:e].astype(np.int64)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    return out


def cmd_eval(a):
    torch.set_num_threads(a.threads)
    model, cfg = load(a.ckpt)
    if a.text_file:
        text = np.frombuffer(open(a.text_file, "rb").read(), dtype=np.uint8)
    else:
        text = np.array(np.memmap(a.text, dtype=np.uint8, mode="r")[a.offset: a.offset + a.bytes])
    t0 = time.perf_counter()
    if a.dynamic_lr > 0:
        lp = dynamic_logprobs(model, cfg, text, a.dynamic_lr)
    else:
        lp = logprobs(model, cfg, text)
    secs = time.perf_counter() - t0
    lp.numpy().astype(np.float32).tofile(a.dump)
    y = torch.from_numpy(text.astype(np.int64))
    bits = float(-lp[torch.arange(len(text)), y].mean() / math.log(2))
    print(json.dumps({"bits_per_byte": bits, "bytes": len(text), "seconds": secs,
                      "ms_per_byte_batched": 1e3 * secs / len(text)}))


# ---------------------------------------------------------------- generation
class Stepper:
    """Single-stream incremental next-byte distributions (KV cache / LSTM state)."""

    def __init__(self, model, cfg):
        self.m, self.cfg = model, cfg
        self.hist = []
        self.reset()

    def reset(self):
        self.caches = [dict() for _ in range(self.cfg["layers"])] if self.cfg["arch"] == "gpt" else None
        self.state = None
        self.pos = 0

    @torch.no_grad()
    def feed(self, byts):
        """Feed bytes; return log-probs for the byte after them."""
        self.hist.extend(byts)
        cfg = self.cfg
        with autocast(cfg):
            if cfg["arch"] == "lstm":
                lg, self.state = self.m(torch.tensor([byts]), self.state)
                return F.log_softmax(lg.float()[0, -1], -1)
            if self.pos + len(byts) > cfg["ctx"]:
                # Window full: re-prime on the last half context (amortised).
                keep = self.hist[-(cfg["ctx"] // 2):]
                self.reset()
                lg = self.m(torch.tensor([keep]), self.caches, 0)
                self.pos = len(keep)
            else:
                lg = self.m(torch.tensor([byts]), self.caches, self.pos)
                self.pos += len(byts)
            return F.log_softmax(lg.float()[0, -1], -1)


def sample(lp, temp, min_p, gen):
    p = torch.softmax(lp / temp, -1)
    p[p < min_p * p.max()] = 0
    return int(torch.multinomial(p / p.sum(), 1, generator=gen))


def cmd_gen(a):
    torch.set_num_threads(a.threads)
    model, cfg = load(a.ckpt)
    prompts = json.load(open(a.prompts))
    res = []
    for i, pr in enumerate(prompts):
        gen = torch.Generator().manual_seed(a.seed + i)
        st = Stepper(model, cfg)
        pb = list(pr.encode("latin-1"))
        lp = st.feed([0] + pb)
        out = []
        t0 = time.perf_counter()
        for _ in range(a.max_bytes):
            b = sample(lp, a.temperature, a.min_p, gen)
            out.append(b)
            lp = st.feed([b])
        secs = time.perf_counter() - t0
        res.append({"prompt": pr, "completion": bytes(out).decode("latin-1"), "ms_per_byte": 1e3 * secs / a.max_bytes})
    json.dump(res, open(a.out, "w"), indent=1)


def cmd_bench(a):
    torch.set_num_threads(a.threads)
    rss0 = rss_mb()
    model, cfg = load(a.ckpt)
    if a.fp32:  # bf16 autocast costs more than it saves for one-token steps
        cfg = dict(cfg, bf16=False)
    nparams = sum(p.numel() for p in model.parameters())
    text = np.array(np.memmap(a.text, dtype=np.uint8, mode="r")[a.offset: a.offset + 4096])
    st = Stepper(model, cfg)
    st.feed([0] + list(text[:cfg.get("ctx", 512) // 2 - 1]))
    k = cfg.get("ctx", 512) // 2 - 1
    t0 = time.perf_counter()
    n = 0
    while n < a.steps:
        st.feed([int(text[k + n])])
        n += 1
    ms = 1e3 * (time.perf_counter() - t0) / n
    print(json.dumps({"params": nparams, "fp32_mb": nparams * 4 / 2**20, "ckpt_mb": os.path.getsize(a.ckpt) / 2**20,
                      "ms_per_distribution": ms, "threads": a.threads, "fp32": a.fp32,
                      "rss_mb": rss_mb(), "rss_model_mb": rss_mb() - rss0,
                      "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}))


def cmd_export(a):
    """Write a checkpoint for CyphaLM's native runtime (neural_expert.hpp): LSTM as BLM1, Transformer as BGT1."""
    model, cfg = load(a.ckpt)
    if cfg["arch"] == "gpt":  # BGT1 (native ByteGptExpert)
        with open(a.out, "wb") as f:
            f.write(b"BGT1")
            f.write(np.array([cfg["layers"], cfg["d"], cfg["heads"], cfg["ctx"]], dtype=np.uint32).tobytes())
            arrs = [model.emb.weight, model.pos.weight]
            for blk in model.blocks:
                arrs += [blk.ln1.weight, blk.ln1.bias, blk.qkv.weight, blk.proj.weight,
                         blk.ln2.weight, blk.ln2.bias, blk.fc.weight, blk.fc2.weight]
            arrs += [model.lnf.weight, model.lnf.bias]
            for t in arrs:
                f.write(t.detach().float().contiguous().numpy().tobytes())
        return
    rnn = model.rnn
    with open(a.out, "wb") as f:
        f.write(b"BLM1")
        f.write(np.array([rnn.num_layers, rnn.hidden_size, model.emb.embedding_dim], dtype=np.uint32).tobytes())
        arrs = [model.emb.weight]
        for l in range(rnn.num_layers):
            arrs += [getattr(rnn, f"weight_ih_l{l}"), getattr(rnn, f"weight_hh_l{l}"),
                     getattr(rnn, f"bias_ih_l{l}"), getattr(rnn, f"bias_hh_l{l}")]
        arrs += [model.out.weight, model.out.bias]
        for t in arrs:
            f.write(t.detach().float().contiguous().numpy().tobytes())


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train")
    t.add_argument("--arch", choices=["gpt", "lstm"], required=True)
    t.add_argument("--data", required=True)
    t.add_argument("--data-bytes", type=int, default=95_000_000)
    t.add_argument("--val-offset", type=int, default=99_000_000)
    t.add_argument("--val-bytes", type=int, default=65536)
    t.add_argument("--d", type=int, default=256)
    t.add_argument("--layers", type=int, default=6)
    t.add_argument("--heads", type=int, default=8)
    t.add_argument("--ctx", type=int, default=512)
    t.add_argument("--emb", type=int, default=64)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--warmup", type=int, default=200)
    t.add_argument("--budget-min", type=float, default=60)
    t.add_argument("--budget-bytes", type=int, default=0, help="stop after this many training bytes instead")
    t.add_argument("--marks", type=float, nargs="*", default=[5, 15, 30])
    t.add_argument("--fp32", action="store_true")
    t.add_argument("--threads", type=int, default=4)
    t.add_argument("--out", required=True)
    e = sp.add_parser("eval")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--text")
    e.add_argument("--text-file")
    e.add_argument("--offset", type=int, default=0)
    e.add_argument("--bytes", type=int, default=16384)
    e.add_argument("--dump", required=True)
    e.add_argument("--dynamic-lr", type=float, default=0.0)
    e.add_argument("--threads", type=int, default=4)
    g = sp.add_parser("gen")
    g.add_argument("--ckpt", required=True)
    g.add_argument("--prompts", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--max-bytes", type=int, default=400)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--min-p", type=float, default=0.1)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--threads", type=int, default=4)
    b = sp.add_parser("bench")
    b.add_argument("--ckpt", required=True)
    b.add_argument("--text", required=True)
    b.add_argument("--offset", type=int, default=96_000_000)
    b.add_argument("--steps", type=int, default=400)
    b.add_argument("--threads", type=int, default=4)
    b.add_argument("--fp32", action="store_true")
    x = sp.add_parser("export")
    x.add_argument("--ckpt", required=True)
    x.add_argument("--out", required=True)
    a = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval, "gen": cmd_gen, "bench": cmd_bench, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
