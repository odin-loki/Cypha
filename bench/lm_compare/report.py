#!/usr/bin/env python3
"""Markdown tables from the compare.py outputs in $WORK."""
import glob
import json
import os

import numpy as np

WORK = os.environ.get("WORK", "/home/user/ckpt/compare")
WINNER = os.environ.get("WINNER", f"{WORK}/cyphalm")
LIGHT = os.environ.get("LIGHT", "/home/user/ckpt/winner/winner_light.json")


def j(name, default=None):
    p = f"{WORK}/{name}"
    return json.load(open(p)) if os.path.exists(p) else default


E = j("evals.json", {})
B = j("bench.json", {})
G = j("gens.json", {})
J = j("judge.json", {})
C = j("curve.json", {})

LABEL = {
    "cyphalm/online": "CyphaLM (reads the text)",
    "cyphalm/frozen": "CyphaLM frozen",
    "cyphalm_light/online": "CyphaLM light* (reads the text)",
    "cyphalm_light/frozen": "CyphaLM light* frozen",
    "gpt/static": "Transformer",
    "gpt/dynamic": "Transformer + dynamic eval",
    "lstm/static": "LSTM",
    "lstm/dynamic": "LSTM + dynamic eval",
}
ROWS = [k for k in LABEL if any(x.startswith(k + "/") for x in E)]


def f(x, d=4):
    return "—" if x is None else f"{x:.{d}f}"


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def size_mb(paths):
    return sum(os.path.getsize(p) for p in paths if os.path.exists(p)) / 2**20


def main():
    p = []
    # ---- training cost
    rows = []
    tt = j("cyphalm/train_time.json", {})
    it = j("cyphalm/index_time.json", {})
    if tt:
        shards = sorted(glob.glob(f"{WINNER}/shard_*.hpbin"))
        rows.append(["CyphaLM (11 shards + ∞-gram index)", "11 × 2^20-slot hash tables + suffix array",
                     "95.0 MB × 1 pass", f"{(tt['wall_seconds'] + it.get('wall_seconds', 0)) / 60:.1f}",
                     f"{max(tt['peak_rss_mb'], it.get('peak_rss_mb', 0)):.0f}",
                     f"{size_mb(shards):.0f} + {size_mb([WINNER + '/enwik8_95m.igr']):.0f} index"])
    for arch, name in (("gpt", "Transformer"), ("lstm", "LSTM")):
        lg = j(f"nn/{arch}/train_log.json")
        if not lg:
            continue
        c = lg["cfg"]
        desc = (f"{lg['params'] / 1e6:.2f}M params: {c['layers']} layers, d {c['d']}, ctx {c['ctx']}" if arch == "gpt"
                else f"{lg['params'] / 1e6:.2f}M params: {c['layers']} × LSTM {c['d']}, TBPTT {c['ctx']}")
        rows.append([name, desc, f"{lg['bytes_seen'] / 1e6:.1f} MB ({lg['epochs']:.2f} epochs)",
                     f"{lg['train_seconds'] / 60:.1f}", f"{lg['peak_rss_mb']:.0f}",
                     f"{size_mb([f'{WORK}/nn/{arch}/final.pt']):.1f}"])
    p.append("### Training (4 cores, same 95 MB of enwik8)\n")
    p.append(md(["system", "model", "data read", "train min", "train peak RSS MB", "size on disk MB"], rows))

    # ---- held-out quality
    def g(k, t, m):
        return E.get(f"{k}/{t}", {}).get(m)

    rows = [[LABEL[k], f(g(k, "wiki", "bits_per_byte")), f(g(k, "alice", "bits_per_byte")),
             f(g(k, "lcet10", "bits_per_byte")), pct(g(k, "wiki", "top1")), pct(g(k, "wiki", "top5")),
             pct(g(k, "wiki", "ece_top1"))] for k in ROWS]
    p.append("\n### Held-out next-byte quality (bits/byte, lower is better; 16 KiB each)\n")
    p.append(md(["system", "wiki", "Alice", "lcet10", "top-1 wiki", "top-5 wiki", "ECE wiki"], rows))

    rows = []
    for k in ROWS:
        for t in ("wiki", "alice"):
            s = E.get(f"{k}/{t}")
            if not s:
                continue
            rows.append([LABEL[k], t, f(s["perplexity_per_byte"], 3), f(s["word_perplexity"], 0),
                         pct(s["next_word_greedy_acc"]), pct(s["word_first_byte_top1"]),
                         f(s["mean_entropy_bits"], 3), s["best_temperature"]])
    p.append("\n### LLM-style metrics\n")
    p.append(md(["system", "text", "byte perplexity", "word perplexity", "next-word greedy acc",
                 "word-start top-1", "entropy bits", "NLL-best temperature"], rows))

    # ---- in-context curve
    keys = list(E.get(f"{ROWS[0]}/wiki", {}).get("bits_by_position", {}).keys()) if ROWS else []
    for t in ("wiki", "alice"):
        rows = [[LABEL[k]] + [f(E.get(f"{k}/{t}", {}).get("bits_by_position", {}).get(b), 3) for b in keys]
                for k in ROWS]
        p.append(f"\n### Bits/byte by position in the held-out text ({t}): in-context learning\n")
        p.append(md(["system"] + [f"bytes {b}" for b in keys], rows))

    # ---- byte classes
    classes = ["lower", "upper", "space", "punct/markup", "digit", "non-ascii"]
    s0 = E.get(f"{ROWS[0]}/wiki", {}).get("bits_by_class", {}) if ROWS else {}
    rows = [[LABEL[k]] + [f(E.get(f"{k}/wiki", {}).get("bits_by_class", {}).get(c, {}).get("bits"), 3) for c in classes]
            for k in ROWS]
    p.append("\n### Bits/byte by byte class (wiki)\n")
    p.append(md(["system"] + [f"{c} ({pct(s0.get(c, {}).get('share'))})" for c in classes], rows))

    # ---- copy tests
    rows = []
    for k in ROWS:
        r = [LABEL[k]]
        for t in ("copy_wiki_2k", "copy_random_384"):
            s = E.get(f"{k}/{t}", {})
            r += [f(s.get("first_copy_bits"), 3), f(s.get("second_copy_bits"), 3)]
        rows.append(r)
    p.append("\n### Copying from context (a passage, then the same passage again)\n")
    p.append(md(["system", "wiki 2 KiB: 1st", "2nd", "random letters 384 B: 1st", "2nd"], rows))

    # ---- serving
    rows = []
    for k in ("cyphalm", "cyphalm_light"):
        b4, b1, s = B.get(f"{k}/t4"), B.get(f"{k}/t1"), E.get(f"{k}/online/wiki")
        if not (b4 and s):
            continue
        r = s["run"]
        gen = [x["ms_per_byte"] for x in G.get("cyphalm_bytes", [])] if k == "cyphalm" else []
        genw = [x["ms_per_byte"] for x in G.get("cyphalm", [])] if k == "cyphalm" else []
        rows.append([LABEL[f"{k}/online"].split(" (")[0], f(b4["ms_per_distribution"], 2),
                     f(b1["ms_per_distribution"], 2) if b1 else "—",
                     f(np.mean(gen), 2) if gen else "—", f(np.mean(genw), 2) if genw else "—",
                     f"{r['rss_anon_mb']:.0f} private + {r['rss_file_mb']:.0f} mapped"])
    for arch, name in (("gpt", "Transformer"), ("lstm", "LSTM")):
        b4, b1 = B.get(f"{arch}/t4/fp32"), B.get(f"{arch}/t1/fp32")
        if b4:
            gen = [x["ms_per_byte"] for x in G.get(arch, [])]
            rows.append([name + " (fp32 step; bf16: " + f(B[f"{arch}/t4"]["ms_per_distribution"], 1) + " ms)",
                         f(b4["ms_per_distribution"], 2), f(b1["ms_per_distribution"], 2) if b1 else "—",
                         f(np.mean(gen), 2) if gen else "—", "n/a",
                         f"{b4['rss_mb']:.0f} process ({b4['fp32_mb']:.0f} weights)"])
    p.append("\n### Serving cost (one stream)\n")
    p.append(md(["system", "ms / next-byte distribution, 4 threads", "1 thread", "generate ms/byte (byte sampling)",
                 "generate ms/byte (word lookahead)", "RSS MB"], rows))

    # ---- generation
    gens = [x for x in ("reference", "cyphalm", "cyphalm_bytes", "gpt", "lstm") if x in G]
    gl = {"reference": "true continuation", "cyphalm": "CyphaLM (word lookahead K 8)",
          "cyphalm_bytes": "CyphaLM (byte sampling)", "gpt": "Transformer", "lstm": "LSTM"}
    rows = []
    for gname in gens:
        r = [gl[gname]]
        for judge in ("cyphalm", "gpt", "lstm"):
            v = J.get(f"{gname}|{judge}")
            r.append(f(np.mean(v), 3) if v else "—")
        tx = J.get(f"{gname}|text", [])
        for m in ("d4", "copy12", "valid_words"):
            vals = [x[m] for x in tx if x[m] is not None]
            r.append(pct(np.mean(vals)) if vals else "—")
        rows.append(r)
    p.append("\n### Generation: 12 prompts (8 wiki, 4 Alice), 400 bytes, T 0.8, min-p 0.1\n")
    p.append(md(["generator", "judge CyphaLM", "judge Transformer", "judge LSTM", "distinct 4-grams",
                 "12-byte copies", "real words"], rows))

    # ---- learning curve
    if C:
        rows = []
        for arch, name in (("gpt", "Transformer"), ("lstm", "LSTM")):
            lg = j(f"nn/{arch}/train_log.json", {})
            mins = {"min5": 5, "min15": 15, "min30": 30, "final": 60}
            for ck, mn in mins.items():
                if f"{arch}/{ck}/wiki" in C:
                    rows.append([name, mn, f(C[f"{arch}/{ck}/wiki"]), f(C[f"{arch}/{ck}/alice"]), "—", "—"])
        tt = j("cyphalm/train_time.json", {})
        import re
        secs = [float(x) for x in re.findall(r"bpc, (\d+) s", tt.get("stdout", ""))]
        per_round = (sum(secs) / len(secs) / 60) if secs else None
        for k in (1, 4, 8, 11):
            if f"cyphalm_k{k}/online/wiki" in C:
                rows.append([f"CyphaLM {k} shard{'s' if k > 1 else ''}",
                             f"{per_round * -(-k // 4):.1f}" if per_round else "—",
                             f(C[f"cyphalm_k{k}/frozen/wiki"]), f(C[f"cyphalm_k{k}/frozen/alice"]),
                             f(C[f"cyphalm_k{k}/online/wiki"]), f(C[f"cyphalm_k{k}/online/alice"])])
        p.append("\n### Quality against training time\n")
        p.append(md(["system", "train min", "wiki (static/frozen)", "Alice (static/frozen)",
                     "wiki (reading)", "Alice (reading)"], rows))
        if tt:
            p.append("\nCyphaLM shard log:\n```\n" + tt["stdout"].strip() + "\n```")

    # ---- dynamic eval tuning
    for arch in ("gpt", "lstm"):
        t = E.get(f"tune_{arch}")
        if t:
            p.append(f"\n{arch} dynamic-eval SGD lr grid on enwik8 @97,000,000 (8 KiB, tuning only): "
                     + ", ".join(f"{k}: {v:.4f}" for k, v in t["grid"].items()) + f" → {t['best_lr']}")
    print("\n".join(p))


if __name__ == "__main__":
    main()
