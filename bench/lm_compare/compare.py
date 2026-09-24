#!/usr/bin/env python3
"""CyphaLM vs a byte LSTM and a byte Transformer, each trained for <= 1 hour
on the same 4-core CPU and the same 95 MB of enwik8.

  compare.py evals   distributions on held-out + copy tests -> WORK/dumps, WORK/evals.json
  compare.py gens    continuations from shared prompts       -> WORK/gens.json
  compare.py judge   every judge scores every continuation   -> WORK/judge.json
  compare.py report  tables                                  -> WORK/report.md

Environment: CYPHA_BUILD (cyphalm tools), WINNER (dir with winner.json),
NN (dir with gpt/ and lstm/ checkpoints), WORK (output dir).
"""
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

import metrics as M

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.environ.get("CYPHA_BUILD", "/home/user/cypha_build")
WINNER = os.environ.get("WINNER", "/home/user/ckpt/winner")
NN = os.environ.get("NN", "/home/user/ckpt/nn")
WORK = os.environ.get("WORK", "/home/user/ckpt/compare")
ENWIK8 = os.environ.get("ENWIK8", "/home/user/corpora/enwik8")
CANT = os.environ.get("CANT", "/home/user/corpora/cant")
PY = [sys.executable, os.path.join(HERE, "byte_lm.py")]

TEXTS = {  # name: (path, offset, bytes); all unseen in training (enwik8 < 95 MB)
    "wiki": (ENWIK8, 96_000_000, 16384),
    "alice": (f"{CANT}/alice29.txt", 20_000, 16384),
    "lcet10": (f"{CANT}/lcet10.txt", 50_000, 16384),
}
TUNE = (ENWIK8, 97_000_000, 8192)  # dynamic-eval learning rate is tuned here only


def slice_bytes(path, off, n):
    with open(path, "rb") as f:
        f.seek(off)
        return np.frombuffer(f.read(n), dtype=np.uint8)


def make_copy_texts():
    """Copy tests: a passage then the same passage again. The second copy is
    free for a model that can look it up (induction / in-context copying)."""
    os.makedirs(f"{WORK}/texts", exist_ok=True)
    wiki = slice_bytes(ENWIK8, 98_000_000, 2048).tobytes()
    rng = np.random.default_rng(7)
    rnd = bytes(rng.choice(np.frombuffer(b"abcdefghijklmnopqrstuvwxyz ", np.uint8), 384))
    out = {}
    for name, blob in (("copy_wiki_2k", wiki + wiki), ("copy_random_384", rnd + rnd)):
        p = f"{WORK}/texts/{name}.txt"
        open(p, "wb").write(blob)
        out[name] = (p, 0, len(blob))
    return out


def run(cmd, **kw):
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd}\n{r.stderr[-2000:]}")
    return r.stdout, time.perf_counter() - t0


def cypha_eval(manifest, path, off, n, dump, frozen):
    cmd = [f"{BUILD}/cyphalm_lm_quality", "--load", manifest, "--eval", path, "--eval-offset", str(off),
           "--eval-bytes", str(n), "--gen-bytes", "0", "--prompt-bytes", "0", "--dump-dist", dump]
    if frozen:
        cmd.append("--frozen-eval")
    out, _ = run(cmd)
    j = json.loads(out[out.index("{"):])
    e = j["eval"]
    return {"distribution_ms": e["distribution_ms"], "ms_per_byte": e["ms_per_byte"],
            "rss_mb": j.get("rss_mb_after_eval"), "rss_anon_mb": j.get("rss_anon_mb_after_eval"),
            "rss_file_mb": j.get("rss_file_mb_after_eval"), "rss_hwm_mb": j.get("rss_hwm_mb"),
            "harness_bits": e["nll_bits_per_byte"]}


def nn_eval(arch, path, off, n, dump, dyn_lr=0.0, text_file=None):
    cmd = PY + ["eval", "--ckpt", f"{NN}/{arch}/final.pt", "--dump", dump]
    cmd += ["--text-file", text_file] if text_file else ["--text", path, "--offset", str(off), "--bytes", str(n)]
    if dyn_lr > 0:
        cmd += ["--dynamic-lr", str(dyn_lr)]
    out, _ = run(cmd)
    return json.loads(out.strip().splitlines()[-1])


LIGHT = os.environ.get("LIGHT", "/home/user/ckpt/winner/winner_light.json")


def systems():
    s = {"cyphalm": ("cypha", f"{WINNER}/winner.json")}
    if os.path.exists(LIGHT):  # one model on 95 MB: ~2.5 h to train, outside the 1 h budget
        s["cyphalm_light"] = ("cypha", LIGHT)
    s.update(gpt=("nn", "gpt"), lstm=("nn", "lstm"))
    return s


def tune_dynamic(arch):
    path, off, n = TUNE
    text = slice_bytes(path, off, n)
    res = {}
    for lr in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0):
        d = f"{WORK}/dumps/tune_{arch}_{lr}.f32"
        nn_eval(arch, path, off, n, d, lr)
        res[lr] = M.score(M.load_dump(d, n), text)["bits_per_byte"]
        os.remove(d)
    static = f"{WORK}/dumps/tune_{arch}_static.f32"
    nn_eval(arch, path, off, n, static)
    res[0.0] = M.score(M.load_dump(static, n), text)["bits_per_byte"]
    os.remove(static)
    best = min((lr for lr in res if lr > 0), key=res.get)
    return best, res


def cmd_evals():
    os.makedirs(f"{WORK}/dumps", exist_ok=True)
    texts = dict(TEXTS)
    texts.update(make_copy_texts())
    res_path = f"{WORK}/evals.json"
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    for arch in ("gpt", "lstm"):
        key = f"tune_{arch}"
        if key not in res:
            best, grid = tune_dynamic(arch)
            res[key] = {"best_lr": best, "grid": {str(k): v for k, v in grid.items()}}
            json.dump(res, open(res_path, "w"), indent=1)
    for sysname, (kind, ref) in systems().items():
        modes = ("online", "frozen") if kind == "cypha" else ("static", "dynamic")
        for mode in modes:
            for tname, (path, off, n) in texts.items():
                key = f"{sysname}/{mode}/{tname}"
                if key in res:
                    continue
                dump = f"{WORK}/dumps/{sysname}_{mode}_{tname}.f32"
                if kind == "cypha":
                    extra = cypha_eval(ref, path, off, n, dump, mode == "frozen")
                else:
                    lr = res[f"tune_{ref}"]["best_lr"] if mode == "dynamic" else 0.0
                    extra = nn_eval(ref, path, off, n, dump, lr)
                text = slice_bytes(path, off, n)
                s = M.score(M.load_dump(dump, n), text)
                if tname.startswith("copy_"):
                    half = n // 2
                    bits = -M.load_dump(dump, n)[np.arange(n), text.astype(np.int64)] / M.LN2
                    s["first_copy_bits"] = float(bits[:half].mean())
                    s["second_copy_bits"] = float(bits[half:].mean())
                s["run"] = extra
                res[key] = s
                json.dump(res, open(res_path, "w"), indent=1)
                print(key, round(s["bits_per_byte"], 4), flush=True)
                if tname.startswith("copy_"):
                    os.remove(dump)  # held-out dumps are kept for re-scoring


# ---------------------------------------------------------------- generation
def prompts():
    """8 wiki prompts (held-out enwik8) + 4 Alice prompts, 256 bytes each, cut
    at a space; the true next 400 bytes are the reference."""
    out = []
    for k in range(8):
        off = 96_100_000 + k * 97_331
        blob = slice_bytes(ENWIK8, off, 1200).tobytes().decode("utf-8", "ignore")
        out.append(("wiki", blob))
    alice = open(f"{CANT}/alice29.txt", "rb").read().decode("latin-1")
    for k in range(4):
        out.append(("alice", alice[60_000 + k * 17_000: 60_000 + k * 17_000 + 1200]))
    res = []
    for dom, blob in out:
        blob = blob.encode("latin-1", "ignore").decode("latin-1")
        cut = blob.rfind(" ", 0, 256)
        res.append({"domain": dom, "prompt": blob[:cut + 1], "reference": blob[cut + 1: cut + 401]})
    return res


def cmd_gens():
    os.makedirs(WORK, exist_ok=True)
    ps = prompts()
    gpath = f"{WORK}/gens.json"
    gens = json.load(open(gpath)) if os.path.exists(gpath) else {}
    gens["reference"] = [{"prompt": p["prompt"], "completion": p["reference"], "domain": p["domain"]} for p in ps]
    json.dump(ps, open(f"{WORK}/prompts.json", "w"), indent=1)
    json.dump([p["prompt"] for p in ps], open(f"{WORK}/prompt_texts.json", "w"))
    for arch in ("gpt", "lstm"):
        if arch in gens:
            continue
        run(PY + ["gen", "--ckpt", f"{NN}/{arch}/final.pt", "--prompts", f"{WORK}/prompt_texts.json",
                  "--out", f"{WORK}/gen_{arch}.json", "--max-bytes", "400"])
        g = json.load(open(f"{WORK}/gen_{arch}.json"))
        for x, p in zip(g, ps):
            x["domain"] = p["domain"]
        gens[arch] = g
        json.dump(gens, open(gpath, "w"), indent=1)
    for name, k in (("cyphalm", 8), ("cyphalm_bytes", 0)):
        if name in gens:
            continue
        rows = []
        for i, p in enumerate(ps):
            cmd = [f"{BUILD}/cyphalm_generate", "--load", f"{WINNER}/winner.json", "--prompt", p["prompt"],
                   "--max-bytes", "400", "--seed", str(1 + i), "--word-candidates", str(k), "--latency"]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0:
                raise RuntimeError(f"cyphalm_generate failed: {r.stderr.decode('latin-1')[-2000:]}")
            out = r.stdout.decode("latin-1")
            comp = out.split("--- completion ---\n", 1)[1]
            comp = comp[:-1] if comp.endswith("\n") else comp
            lat = float(re.search(r"latency_ms=([\d.]+)", out).group(1))
            rows.append({"prompt": p["prompt"], "completion": comp, "domain": p["domain"],
                         "ms_per_byte": lat / max(1, len(comp))})
            print(name, i, flush=True)
        gens[name] = rows
        json.dump(gens, open(gpath, "w"), indent=1)


def judge_bits(judge, prompt, comp, tag):
    os.makedirs(f"{WORK}/judge", exist_ok=True)
    blob = (prompt + comp).encode("latin-1", "replace")
    tf = f"{WORK}/judge/{tag}.txt"
    open(tf, "wb").write(blob)
    dump = f"{WORK}/judge/{tag}.f32"
    n = len(blob)
    if judge == "cyphalm":
        cypha_eval(f"{WINNER}/winner.json", tf, 0, n, dump, frozen=True)
    else:
        nn_eval(judge, None, 0, n, dump, text_file=tf)
    lp = M.load_dump(dump, n)
    t = np.frombuffer(blob, np.uint8).astype(np.int64)
    bits = -lp[np.arange(n), t] / M.LN2
    os.remove(dump)
    os.remove(tf)
    k = len(prompt.encode("latin-1", "replace"))
    return float(bits[k:].mean())


def training_vocab():
    cache = f"{WORK}/vocab.json"
    if os.path.exists(cache):
        return set(json.load(open(cache)))
    from collections import Counter
    c = Counter()
    with open(ENWIK8, "rb") as f:
        left = 95_000_000
        while left > 0:
            blk = f.read(min(left, 8 << 20))
            left -= len(blk)
            c.update(w.lower() for w in re.findall(rb"[A-Za-z]+", blk))
    v = sorted(w.decode() for w, k in c.items() if k >= 3)
    json.dump(v, open(cache, "w"))
    return set(v)


def cmd_judge():
    gens = json.load(open(f"{WORK}/gens.json"))
    jpath = f"{WORK}/judge.json"
    res = json.load(open(jpath)) if os.path.exists(jpath) else {}
    vocab = training_vocab()
    for gname, rows in gens.items():
        for judge in ("cyphalm", "gpt", "lstm"):
            key = f"{gname}|{judge}"
            if key in res:
                continue
            res[key] = [judge_bits(judge, r["prompt"], r["completion"], f"{gname}_{judge}_{i}")
                        for i, r in enumerate(rows)]
            json.dump(res, open(jpath, "w"), indent=1)
            print(key, np.mean(res[key]), flush=True)
        key = f"{gname}|text"
        res[key] = [{"d4": M.distinct_ngrams(r["completion"]), "copy12": M.copy_rate(r["prompt"], r["completion"]),
                     "valid_words": M.word_validity(r["completion"], vocab)} for r in rows]
        json.dump(res, open(jpath, "w"), indent=1)


def cmd_bench_cypha():
    """CyphaLM next-byte distribution latency at 1 and 4 ensemble threads."""
    bpath = f"{WORK}/bench.json"
    res = json.load(open(bpath)) if os.path.exists(bpath) else {}
    for name, man in (("cyphalm", f"{WINNER}/winner.json"), ("cyphalm_light", LIGHT)):
        if not os.path.exists(man):
            continue
        for th in (4, 1):
            env = dict(os.environ, CYPHA_HP_ENSEMBLE_THREADS=str(th))
            dump = f"{WORK}/dumps/bench.f32"
            cmd = [f"{BUILD}/cyphalm_lm_quality", "--load", man, "--eval", ENWIK8, "--eval-offset", "96000000",
                   "--eval-bytes", "4096", "--gen-bytes", "0", "--prompt-bytes", "0", "--frozen-eval"]
            out, _ = run(cmd, env=env)
            e = json.loads(out[out.index("{"):])
            res[f"{name}/t{th}"] = {"ms_per_distribution": e["eval"]["distribution_ms"],
                                    "ms_per_byte_with_update": e["eval"]["ms_per_byte"],
                                    "rss_mb": e["rss_mb_after_eval"], "rss_anon_mb": e["rss_anon_mb_after_eval"]}
    json.dump(res, open(bpath, "w"), indent=1)


def cmd_bench():
    """Serving cost of the NN baselines (CyphaLM's comes from its eval runs)."""
    res = {}
    for arch in ("gpt", "lstm"):
        for th in (4, 1):
            for fp32 in (False, True):
                cmd = PY + ["bench", "--ckpt", f"{NN}/{arch}/final.pt", "--text", ENWIK8, "--threads", str(th)]
                out, _ = run(cmd + (["--fp32"] if fp32 else []))
                res[f"{arch}/t{th}" + ("/fp32" if fp32 else "")] = json.loads(out.strip().splitlines()[-1])
    old = json.load(open(f"{WORK}/bench.json")) if os.path.exists(f"{WORK}/bench.json") else {}
    old.update(res)
    json.dump(old, open(f"{WORK}/bench.json", "w"), indent=1)
    cmd_bench_cypha()


PER_SHARD = 95_000_000 // 11


def cmd_curve():
    """Quality against training time. NN: the checkpoints saved at 5/15/30/60
    min. CyphaLM: the first k shards with an index over the same k shards'
    bytes (shards train 4 at a time, so time grows in rounds of 4)."""
    cpath = f"{WORK}/curve.json"
    res = json.load(open(cpath)) if os.path.exists(cpath) else {}
    os.makedirs(f"{WORK}/dumps", exist_ok=True)
    for arch in ("gpt", "lstm"):
        for ck in ("min5", "min15", "min30", "final"):
            for tname in ("wiki", "alice"):
                key = f"{arch}/{ck}/{tname}"
                if key in res:
                    continue
                path, off, n = TEXTS[tname]
                dump = f"{WORK}/dumps/curve.f32"
                cmd = PY + ["eval", "--ckpt", f"{NN}/{arch}/{ck}.pt", "--dump", dump,
                            "--text", path, "--offset", str(off), "--bytes", str(n)]
                run(cmd)
                res[key] = M.score(M.load_dump(dump, n), slice_bytes(path, off, n))["bits_per_byte"]
                json.dump(res, open(cpath, "w"), indent=1)
    for k in (1, 4, 8, 11):
        man = f"{WINNER}/winner_k{k}.json"
        igr = "enwik8_95m.igr" if k == 11 else f"enwik8_k{k}.igr"
        if not os.path.exists(f"{WINNER}/{igr}"):
            run([f"{BUILD}/cyphalm_infinigram_build", "--text", ENWIK8, "--bytes", str(k * PER_SHARD),
                 "--out", f"{WINNER}/{igr}"])
        json.dump({"cyphalm_ensemble": 1, "members": [{"checkpoint": f"shard_{i}.json"} for i in range(k)],
                   "learning_rate": 0.01, "infinigram": igr}, open(man, "w"), indent=1)
        for mode in ("online", "frozen"):
            for tname in ("wiki", "alice"):
                key = f"cyphalm_k{k}/{mode}/{tname}"
                if key in res:
                    continue
                path, off, n = TEXTS[tname]
                dump = f"{WORK}/dumps/curve.f32"
                cypha_eval(man, path, off, n, dump, mode == "frozen")
                res[key] = M.score(M.load_dump(dump, n), slice_bytes(path, off, n))["bits_per_byte"]
                json.dump(res, open(cpath, "w"), indent=1)
                print(key, res[key], flush=True)


if __name__ == "__main__":
    {"evals": cmd_evals, "gens": cmd_gens, "judge": cmd_judge, "bench": cmd_bench, "bench_cypha": cmd_bench_cypha, "curve": cmd_curve}[sys.argv[1]]()
