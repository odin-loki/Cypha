#!/usr/bin/env python3
"""Offline replay of CyphaLM's mixing stages from per-byte component dumps.

`cyphalm_lm_quality --dump-components DIR` writes, for every scored byte,
what the served distribution was mixed from: each ensemble model's log P,
the ∞-gram query (longest match and reliable part), each neural expert's
log P and the served log P (layout: native/include/cypha/cyphalm/
component_dump.hpp). This module re-implements the stages of
HpSequenceBackend::next_byte_log_probs and their online weight updates
(native/src/cyphalm/hp_backend.cpp) over those arrays:

  mix_with_members_          geometric ensemble mix (or the context gate)
  update_ensemble_weights_   exponentiated gradient (update_gate_: AdaGrad)
  infinigram_mix_            model / longest match / reliable part per bucket
  neural_mix_                linear, log-linear or switch mix with the experts
  final_sharpen_             final temperature, fixed or learned per bucket

The experts' output-layer adaptation is not simulated: the dump holds their
actual log P, which already includes it (it depends only on the true
bytes). Members read the true bytes on their own and index counts depend
only on the context, so with the dumped settings the replay reproduces the
harness's eval.nll_bits_per_byte (`check`: float64 dumps to rounding,
float32 to about 1e-10 bits/byte). With other settings it screens mixing variants,
member and expert subsets and rates in seconds (`run`, `compare`). The
session cache is not replayed (the harness refuses to dump it).

  mixsim.py check DUMP [DUMP ...] [--tol T]
  mixsim.py check DUMP --set KEY=VALUE ... --against OTHER_DUMP
  mixsim.py run DUMP [--set KEY=VALUE ...] [--save-nll FILE.npy]
  mixsim.py compare DUMP [DUMP ...] --a KEY=VALUE ... --b KEY=VALUE ... [--block 1024]
  mixsim.py compare-dumps DUMP_A DUMP_B [--block 1024]

Settings (--set, --a, --b; unset = as dumped):
  models=0,2,3              ensemble models kept, 0 = the primary. A list other
                            than all of them in order starts at equal weights
  ensemble_start=w,w,...    start weights of the kept models (normalised)
  neural=0,1                experts kept; a list other than all of them starts
                            at the backend's start weights
  infinigram=0|1            drop the ∞-gram stage (its mode cannot change:
                            the reliable part depends on it)
  ensemble_gate=0|1, neural_mix=linear|log|switch, learn_mix=0|1
  ensemble_lr, infinigram_lr, neural_lr, final_temperature,
  final_temperature_lr      numbers
A stage whose mode or members change restarts at the backend's start
weights, as the C++ setters do; unchanged stages start from the dumped state.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

LN2 = math.log(2.0)
NN_BUCKETS = 16          # model confidence (8) x top-byte agreement with expert 0 (2)
FT_BUCKETS = 8           # top probability before sharpening
EG_BUCKETS = 8 * 2 * 4   # top probability x agrees with the pool x match length


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class Dump:
    """One --dump-components directory; the arrays are memory-mapped."""

    def __init__(self, path):
        self.path = path
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = m = json.load(f)
        if m.get("format") != "cyphalm_components" or m.get("version") != 1:
            raise ValueError(f"{path}: not a version-1 cyphalm_components dump")
        self.n = n = int(m["bytes"])
        self.dtype = np.dtype(m["dtype"]).newbyteorder("<")
        self.truth = self._array("truth.bin", np.uint8, (n,))
        self.models = self._array("models.bin", self.dtype, (n, m["models"], 256))
        self.served = self._array("served.bin", self.dtype, (n, 256))
        self.neural = (self._array("neural.bin", self.dtype, (n, m["neural"], 256)) if m["neural"]
                       else np.zeros((n, 0, 256)))
        self.ig_n = self.ig_total = self.ig_count = None
        if m["infinigram"]:
            self.ig_n = self._array("ig_n.bin", np.dtype("<i4"), (n, 2))
            self.ig_total = self._array("ig_total.bin", np.dtype("<u8"), (n, 2))
            self.ig_count = self._array("ig_count.bin", np.dtype("<u4"), (n, 2, 256))

    def _array(self, name, dtype, shape):
        p = os.path.join(self.path, name)
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        if os.path.getsize(p) != size:
            raise ValueError(f"{p}: {os.path.getsize(p)} bytes, expected {size} for shape {shape}")
        if size == 0:
            return np.zeros(shape, dtype=dtype)
        return np.memmap(p, dtype=dtype, mode="r", shape=shape)

    @property
    def harness_nll(self):
        """eval.nll_bits_per_byte as the harness reported it (None if absent)."""
        return self.meta.get("eval", {}).get("nll_bits_per_byte")

    def served_nll(self):
        """Per-byte NLL in bits of the served distribution, as dumped."""
        lp = self.served[np.arange(self.n), self.truth].astype(np.float64)
        return -lp / LN2


def settings(dump, **over):
    """The dump's settings with ``over`` applied (unknown keys raise)."""
    m = dump.meta
    s = {"models": list(range(m["models"])),
         "ensemble_start": None,
         "neural": list(range(m["neural"])),
         "infinigram": bool(m["infinigram"]),
         "infinigram_mode": m["infinigram_mode"],
         "ensemble_gate": bool(m["ensemble_gate"]),
         "neural_mix": m["neural_mix"],
         "learn_mix": bool(m["learn_mix"]),
         "ensemble_lr": m["ensemble_learning_rate"],
         "infinigram_lr": m["infinigram_learning_rate"],
         "neural_lr": m["neural_learning_rate"],
         "final_temperature": m["final_temperature"],
         "final_temperature_lr": m["final_temperature_lr"]}
    unknown = sorted(set(over) - set(s))
    if unknown:
        raise ValueError(f"unknown setting(s) {unknown}; known: {sorted(s)}")
    s.update(over)
    if s["infinigram_mode"] != m["infinigram_mode"]:
        raise ValueError("infinigram_mode cannot differ from the dump's (the reliable part was queried in "
                         f"{m['infinigram_mode']} mode); dump again with --infinigram-mode")
    if s["infinigram"] and not m["infinigram"]:
        raise ValueError("the dump has no ∞-gram stage")
    for key, count in (("models", m["models"]), ("neural", m["neural"])):
        if any(not 0 <= i < count for i in s[key]):
            raise ValueError(f"{key}: indices must be in [0, {count})")
    if not s["models"]:
        raise ValueError("models: keep at least one")
    if s["neural_mix"] not in ("linear", "log", "switch"):
        raise ValueError("neural_mix: expected linear, log or switch")
    if not s["final_temperature"] > 0 or not s["final_temperature_lr"] >= 0:
        raise ValueError("final_temperature must be > 0 and final_temperature_lr >= 0")
    return s


def _start(dump, s):
    """Start weights per stage: the dumped state where the stage is unchanged,
    else the backend's start values."""
    m, st = dump.meta, dump.meta["start"]
    all_models = s["models"] == list(range(m["models"]))
    M = len(s["models"])
    if s["ensemble_start"] is not None:
        w = [float(x) for x in s["ensemble_start"]]
        if len(w) != M or min(w) <= 0:
            raise ValueError("ensemble_start: one positive weight per kept model")
        ens = [x / sum(w) for x in w]
    elif all_models:
        ens = [float(x) for x in st["ensemble"]]
    else:
        ens = [1.0 / M] * M
    gate = None
    if s["ensemble_gate"] and M > 1:
        g = st["gate"]
        if m["ensemble_gate"] and all_models and s["ensemble_start"] is None and len(g) == (2 + 2 * EG_BUCKETS) * M:
            gate = {"b": list(g[:M]), "theta": list(g[M:M + EG_BUCKETS * M]), "g2": list(g[M + EG_BUCKETS * M:])}
        else:  # set_ensemble_gate: b at the ensemble weights, theta and the AdaGrad sums at 0
            gate = {"b": list(ens), "theta": [0.0] * (EG_BUCKETS * M), "g2": [0.0] * ((1 + EG_BUCKETS) * M)}
    ig = [list(map(float, w)) for w in st["infinigram"]] if s["infinigram"] else None
    K = len(s["neural"])
    nn_w = nn_a = nn_s = None
    if K:
        nb = NN_BUCKETS if s["neural_mix"] == "linear" else NN_BUCKETS * 4
        same = s["neural"] == list(range(m["neural"])) and s["neural_mix"] == m["neural_mix"]
        if same and len(st["neural"]) == nb * (K + 1):
            nn_w = [list(st["neural"][b * (K + 1):(b + 1) * (K + 1)]) for b in range(nb)]
            if s["neural_mix"] != "linear":
                nn_a = [list(st["neural_log"][b * (K + 1):(b + 1) * (K + 1)]) for b in range(nb)]
            if s["neural_mix"] == "switch":
                nn_s = list(st["neural_switch"])
        else:  # reset_neural_weights_
            nn_w = [[0.7] + [0.3 / K] * K for _ in range(nb)]
            if s["neural_mix"] != "linear":
                nn_a = [list(r) for r in nn_w]
            if s["neural_mix"] == "switch":
                nn_s = [0.5] * nb
    same_ft = (s["final_temperature"] == m["final_temperature"]
               and s["final_temperature_lr"] == m["final_temperature_lr"])
    ft = list(map(float, st["final_temperature"])) if same_ft else [float(s["final_temperature"])] * FT_BUCKETS
    return {"ensemble": ens, "gate": gate, "ig": ig, "nn_w": nn_w, "nn_a": nn_a, "nn_s": nn_s, "ft": ft}


class Result:
    """Per-byte NLL (bits), the served log P if kept, the final weights."""

    def __init__(self, nll, served, state, settings):
        self.nll = nll
        self.served = served
        self.state = state
        self.settings = settings

    @property
    def nll_bits_per_byte(self):
        return float(self.nll.mean())


def simulate(dump, keep_served=False, **over):
    """Replay the mixing stages over ``dump`` with its settings, changed by
    ``over`` (see ``settings``). Mirrors hp_backend.cpp operation by
    operation, including the order of the sums that matter."""
    s = settings(dump, **over)
    st = _start(dump, s)
    sel, nsel = list(s["models"]), list(s["neural"])
    M, K = len(sel), len(nsel)
    use_ig = s["infinigram"]
    longest16 = s["infinigram_mode"] == "longest16"
    nn_mode = s["neural_mix"]
    learn = s["learn_mix"]
    ens_eta, ig_eta, nn_eta = s["ensemble_lr"], s["infinigram_lr"], s["neural_lr"]
    ft_t, ft_eta = s["final_temperature"], s["final_temperature_lr"]
    ft_on = ft_t != 1.0 or ft_eta > 0.0
    w_ens, gate, igw = st["ensemble"], st["gate"], st["ig"]
    nn_w, nn_a, nn_s, ftemp = st["nn_w"], st["nn_a"], st["nn_s"], st["ft"]
    eg_bucket = [0] * M
    n = dump.n
    nll = np.empty(n)
    served = np.empty((n, 256)) if keep_served else None
    all_models = sel == list(range(dump.models.shape[1]))
    all_neural = nsel == list(range(dump.neural.shape[1]))
    for t in range(n):
        y = int(dump.truth[t])
        lps = np.array(dump.models[t] if all_models else dump.models[t, sel], dtype=np.float64)
        ig_lb = 0
        if use_ig:
            rn = int(dump.ig_n[t, 0])
            ig_lb = 0 if rn < 8 else 1 if rn < 16 else 2 if rn < 32 else 3
        # Ensemble: mix_with_members_ (members only: a lone model is served as scored).
        if M == 1:
            base = lps[0]
        else:
            if gate is not None:
                b = gate["b"]
                pool = b[0] * lps[0]
                for i in range(1, M):
                    pool += b[i] * lps[i]
                pool_top = int(np.argmax(pool))
                w = [0.0] * M
                for i in range(M):
                    ti = int(np.argmax(lps[i]))
                    conf = min(7, int(math.exp(lps[i][ti]) * 8.0))
                    eg_bucket[i] = (conf * 2 + (1 if ti == pool_top else 0)) * 4 + ig_lb
                    w[i] = b[i] + gate["theta"][eg_bucket[i] * M + i]
            else:
                w = w_ens
            mix = w[0] * lps[0]
            for i in range(1, M):
                mix += w[i] * lps[i]
            mx = mix.max()
            mix -= mx + math.log(np.exp(mix - mx).sum())
            base = mix
        ens_mix = base
        # ∞-gram: infinigram_mix_.
        if use_ig:
            p0 = np.exp(base)
            pmax = p0.max()
            counts = np.asarray(dump.ig_count[t], dtype=np.float64)
            parts = []
            for q in range(2):
                tot = counts[q].sum()
                parts.append(counts[q] / tot if tot > 0.0 else p0)
            p1, p2 = parts
            rtotal = int(dump.ig_total[t, 0])
            nb = 0 if rn == 0 else min(7, rn.bit_length())  # 1 + floor(log2 n)
            cb = 0 if rtotal <= 1 else 1 if rtotal <= 3 else 2 if rtotal <= 15 else 3
            hb = (0 if pmax < 0.3 else 2 if pmax < 0.6 else 4 if pmax < 0.9 else 6) + (
                1 if int(np.argmax(p0)) == int(np.argmax(p1)) else 0)
            ig_bucket = (nb * 4 + cb) * 8 + hb
            if longest16:
                d1 = int(np.count_nonzero(counts[0]))
                tb = 0 if d1 <= 1 else 1 if d1 == 2 else 2 if d1 <= 4 else 3
                det = 1 if int(np.count_nonzero(counts[1])) == 1 and int(dump.ig_total[t, 1]) >= 2 else 0
                ig_bucket = (((nb * 4 + cb) * 4 + tb) * 2 + det) * 8 + hb
            wi = igw[ig_bucket]
            base = np.log(np.maximum(wi[0] * p0 + wi[1] * p1 + wi[2] * p2, 1e-300))
        # Neural experts: neural_mix_.
        if K:
            lpn = np.array(dump.neural[t] if all_neural else dump.neural[t, nsel], dtype=np.float64)
            pin = np.exp(base)
            top_m = int(np.argmax(pin))
            conf = min(7, int(pin[top_m] * 8.0))
            nbk = conf * 2 + (1 if top_m == int(np.argmax(lpn[0])) else 0)
            if nn_mode != "linear":
                nbk = nbk * 4 + ig_lb
            nn_in = base
            if nn_mode != "log":
                wr = nn_w[nbk]
                plin = wr[0] * pin
                for i in range(K):
                    plin += wr[i + 1] * np.exp(lpn[i])
            if nn_mode == "linear":
                base = np.log(np.maximum(plin, 1e-300))
            else:
                ar = nn_a[nbk]
                z = ar[0] * base
                for i in range(K):
                    z += ar[i + 1] * lpn[i]
                mx = z.max()
                z -= mx + math.log(np.exp(z - mx).sum())
                plog = np.exp(z)
                if nn_mode == "log":
                    base = z
                else:
                    sw = nn_s[nbk]
                    base = np.log(np.maximum(sw * plin + (1.0 - sw) * plog, 1e-300))
        # Final temperature: final_sharpen_.
        if ft_on:
            fb = min(FT_BUCKETS - 1, int(math.exp(base.max()) * FT_BUCKETS))
            temp = ftemp[fb]
            x = base / temp
            mx = x.max()
            ft_in = base
            base = x - mx - math.log(np.exp(x - mx).sum())
        nll[t] = -base[y] / LN2
        if keep_served:
            served[t] = base
        if not learn:
            continue
        # Updates (consume_byte), each from its own stage's scoring.
        if K and nn_eta > 0.0:
            wr = nn_w[nbk]
            pe = [float(pin[y])] + [math.exp(lpn[i][y]) for i in range(K)]
            pm = 0.0
            for i in range(K + 1):
                pm += wr[i] * pe[i]
            p_lin = pm
            if nn_mode != "log":
                pm = max(pm, 1e-12)
                z = 0.0
                for i in range(K + 1):
                    g = _clamp(nn_eta * (pe[i] / pm - 1.0), -2.0, 2.0)
                    wr[i] = max(1e-4, wr[i] * math.exp(g))
                    z += wr[i]
                for i in range(K + 1):
                    wr[i] /= z
            if nn_mode != "linear":
                ar = nn_a[nbk]
                for i in range(K + 1):
                    src = nn_in if i == 0 else lpn[i - 1]
                    e = float(np.dot(plog, src))
                    ar[i] = _clamp(ar[i] + nn_eta * _clamp(src[y] - e, -2.0, 2.0), 0.0, 4.0)
            if nn_mode == "switch":
                pg = float(plog[y])
                p = max(sw * p_lin + (1.0 - sw) * pg, 1e-12)
                nn_s[nbk] = _clamp(sw + nn_eta * (p_lin - pg) / p, 0.01, 0.99)
        if M > 1 and ens_eta > 0.0:
            ens_p = np.exp(ens_mix)
            if gate is not None:
                for i in range(M):
                    g = _clamp(lps[i][y] - float(np.dot(ens_p, lps[i])), -20.0, 20.0)
                    ti = eg_bucket[i] * M + i
                    gate["g2"][i] += g * g
                    gate["g2"][M + ti] += g * g
                    if gate["g2"][i] > 0.0:
                        gate["b"][i] = _clamp(gate["b"][i] + ens_eta * g / math.sqrt(gate["g2"][i]), -4.0, 4.0)
                    if gate["g2"][M + ti] > 0.0:
                        gate["theta"][ti] = _clamp(
                            gate["theta"][ti] + ens_eta * g / math.sqrt(gate["g2"][M + ti]), -4.0, 4.0)
            else:
                w = list(w_ens)
                for i in range(M):
                    w[i] *= math.exp(ens_eta * _clamp(lps[i][y] - float(np.dot(ens_p, lps[i])), -20.0, 20.0))
                z = 0.0
                for i in range(M):
                    w[i] = max(w[i], 1e-4)
                    z += w[i]
                w_ens = [v / z for v in w]
        if ft_on and ft_eta > 0.0:
            p_out = np.exp(base)
            g = _clamp(ft_in[y] - float(np.dot(p_out, ft_in)), -20.0, 20.0)
            beta0 = 1.0 / ft_t
            beta = _clamp((1.0 / temp) * math.exp(ft_eta * g / temp), min(0.8, beta0), max(1.6, beta0))
            ftemp[fb] = 1.0 / beta
        if use_ig and ig_eta > 0.0:
            wi = igw[ig_bucket]
            py = (float(p0[y]), float(p1[y]), float(p2[y]))
            pm = wi[0] * py[0] + wi[1] * py[1] + wi[2] * py[2]
            z = 0.0
            for e in range(3):
                g = _clamp(ig_eta * (py[e] / max(pm, 1e-12) - 1.0), -2.0, 2.0)
                wi[e] = max(1e-4, wi[e] * math.exp(g))
                z += wi[e]
            for e in range(3):
                wi[e] /= z
    state = {"ensemble": w_ens, "gate": gate, "infinigram": igw, "neural": nn_w, "neural_log": nn_a,
             "neural_switch": nn_s, "final_temperature": ftemp}
    return Result(nll, served, state, s)


def block_bootstrap(a, b, block=1024, n_boot=10000, seed=0, level=0.95, lengths=None):
    """Paired block bootstrap of mean(a) - mean(b) over the same bytes.

    The per-byte differences are cut into contiguous blocks of ``block``
    bytes (``lengths``: the sizes of separate texts laid end to end; no
    block straddles two, and each text's last block may be shorter). Blocks
    are resampled with replacement and the byte-weighted mean difference is
    recomputed per resample. Returns the difference, the percentile interval
    at ``level``, a two-sided p (twice the smaller share of resamples on
    either side of 0) and the block count. Resampling blocks keeps the
    within-block dependence of neighbouring bytes, which per-byte resampling
    would ignore; with 16 KiB per text that is only 16 blocks, so small
    effects need several texts."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    lengths = [len(d)] if lengths is None else list(lengths)
    if sum(lengths) != len(d):
        raise ValueError("lengths must add up to the number of bytes")
    starts, sizes, pos = [], [], 0
    for n in lengths:
        for s in range(pos, pos + n, block):
            starts.append(s)
            sizes.append(min(block, pos + n - s))
        pos += n
    sums = np.add.reduceat(d, starts) if starts else np.zeros(0)
    sizes = np.asarray(sizes, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(starts), size=(n_boot, len(starts)))
    means = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    lo, hi = np.quantile(means, [(1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0])
    p = min(1.0, 2.0 * min(float((means >= 0.0).mean()), float((means <= 0.0).mean())))
    return {"delta": float(d.mean()), "lo": float(lo), "hi": float(hi), "level": level, "p_two_sided": p,
            "blocks": len(starts), "block_bytes": block, "bytes": int(len(d))}


def _parse_setting(kv):
    if "=" not in kv:
        raise ValueError(f"expected KEY=VALUE, got {kv!r}")
    k, v = kv.split("=", 1)
    if k in ("models", "neural"):
        return k, [int(x) for x in v.split(",") if x != ""]
    if k == "ensemble_start":
        return k, [float(x) for x in v.split(",")]
    if k in ("infinigram", "ensemble_gate", "learn_mix"):
        if v.lower() not in ("0", "1", "true", "false"):
            raise ValueError(f"{k}: expected 0 or 1")
        return k, v.lower() in ("1", "true")
    if k in ("neural_mix", "infinigram_mode"):
        return k, v
    return k, float(v)


def _settings_arg(items):
    return dict(_parse_setting(kv) for kv in (items or []))


def _default_tol(dump):
    return {"float64": 1e-9, "float32": 1e-6}.get(dump.meta["dtype"], 1e-2)


def cmd_check(args):
    over = _settings_arg(args.set)
    ref_dump = None
    if args.against:
        if len(args.dumps) != 1:
            raise ValueError("--against takes one dump to replay")
        ref_dump = Dump(args.against)
    ok = True
    for path in args.dumps:
        d = Dump(path)
        r = simulate(d, **over)
        tol = args.tol if args.tol is not None else _default_tol(d)
        ref = d if ref_dump is None else ref_dump
        if ref.n != d.n or not np.array_equal(np.asarray(ref.truth), np.asarray(d.truth)):
            raise ValueError(f"{args.against} did not score the same bytes as {path}")
        diff = None if ref.harness_nll is None else r.nll_bits_per_byte - ref.harness_nll
        good = diff is not None and abs(diff) <= tol
        ok &= good
        print(json.dumps({"dump": path, "settings": over, "against": ref.path, "bytes": d.n,
                          "dtype": d.meta["dtype"], "harness_nll_bits_per_byte": ref.harness_nll,
                          "simulated": r.nll_bits_per_byte, "diff": diff, "tol": tol,
                          "max_byte_diff_vs_served": float(np.abs(r.nll - ref.served_nll()).max()), "ok": good}))
    return 0 if ok else 1


def cmd_run(args):
    d = Dump(args.dump)
    r = simulate(d, **_settings_arg(args.set))
    if args.save_nll:
        np.save(args.save_nll, r.nll)
    print(json.dumps({"dump": args.dump, "bytes": d.n, "nll_bits_per_byte": r.nll_bits_per_byte,
                      "settings": r.settings}))
    return 0


def cmd_compare(args):
    sa, sb = _settings_arg(args.a), _settings_arg(args.b)
    per, na, nb, lengths = [], [], [], []
    for path in args.dumps:
        d = Dump(path)
        ra, rb = simulate(d, **sa), simulate(d, **sb)
        per.append({"dump": path, "bytes": d.n, "a": ra.nll_bits_per_byte, "b": rb.nll_bits_per_byte,
                    "delta": ra.nll_bits_per_byte - rb.nll_bits_per_byte})
        na.append(ra.nll)
        nb.append(rb.nll)
        lengths.append(d.n)
    boot = block_bootstrap(np.concatenate(na), np.concatenate(nb), block=args.block, n_boot=args.boot,
                           seed=args.seed, lengths=lengths)
    print(json.dumps({"a": sa, "b": sb, "per_dump": per, "pooled": boot}, indent=1))
    return 0


def cmd_compare_dumps(args):
    da, db = Dump(args.dump_a), Dump(args.dump_b)
    if da.n != db.n or not np.array_equal(np.asarray(da.truth), np.asarray(db.truth)):
        raise ValueError("the two dumps did not score the same bytes")
    a, b = da.served_nll(), db.served_nll()
    boot = block_bootstrap(a, b, block=args.block, n_boot=args.boot, seed=args.seed)
    print(json.dumps({"a": args.dump_a, "b": args.dump_b, "a_nll": float(a.mean()), "b_nll": float(b.mean()),
                      "pooled": boot}, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="replay with the dumped settings; compare with the harness's NLL")
    c.add_argument("dumps", nargs="+")
    c.add_argument("--tol", type=float, default=None,
                   help="bits/byte (default: 1e-9 for float64 dumps, 1e-6 float32, 1e-2 float16)")
    c.add_argument("--set", nargs="*", action="extend", default=[], metavar="KEY=VALUE",
                   help="replay with changed settings (with --against: the other run's settings)")
    c.add_argument("--against", metavar="DUMP",
                   help="compare with the NLL of this dump instead: a run on the same bytes with the "
                        "settings given by --set (predicts it from this dump)")
    c.set_defaults(fn=cmd_check)
    r = sub.add_parser("run", help="replay with changed settings")
    r.add_argument("dump")
    r.add_argument("--set", nargs="*", action="extend", default=[], metavar="KEY=VALUE")
    r.add_argument("--save-nll", metavar="FILE.npy", help="per-byte NLL in bits")
    r.set_defaults(fn=cmd_run)
    for name, fn in (("compare", cmd_compare), ("compare-dumps", cmd_compare_dumps)):
        p = sub.add_parser(name, help="two settings on the same dumps, paired block bootstrap" if name == "compare"
                           else "the served NLL of two dumps of the same bytes, paired block bootstrap")
        if name == "compare":
            p.add_argument("dumps", nargs="+")
            p.add_argument("--a", nargs="*", metavar="KEY=VALUE", default=[])
            p.add_argument("--b", nargs="*", metavar="KEY=VALUE", default=[])
        else:
            p.add_argument("dump_a")
            p.add_argument("dump_b")
        p.add_argument("--block", type=int, default=1024, help="bytes per bootstrap block (1024)")
        p.add_argument("--boot", type=int, default=10000, help="resamples (10000)")
        p.add_argument("--seed", type=int, default=0)
        p.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except ValueError as e:
        print(f"mixsim: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
