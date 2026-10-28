"""Meta-evaluation statistics for the SCRIBE paper (paper plan §5, §8, §9).

Author: Odin Loch

Reads per-text scores written by any metric and measures how well each metric
does on each test of docs/SCRIBE_PAPER_PLAN.md, with bootstrap uncertainty.
No metric is computed here.

Inputs
  texts    JSONL {"id", "set", "system", "text", "prompt"?, "ref"?, "meta"?}
  scores   JSONL {"id", "<metric>": value, ...}; several files are joined on id
  metrics  --metric NAME:+ (higher is better) or NAME:- (lower is better),
           optionally NAME:+:Label, or --metrics-config FILE.json holding
           {"metrics": [{"name", "dir": "+"|"-", "label"?}, ...]}
           (a plain list of those, or a {"name": "+"} mapping, also works)
  filters  --where KEY=VALUE or KEY!=VALUE on any field (dotted, e.g.
           meta.bucket=short); VALUE may list alternatives as a|b|c

Subcommands (meta.py <cmd> --help lists the options)
  ladder     control ladder: rung means, pairwise accuracy, PLAN 6.4 pass rule
  setladder  the same for set-level metrics (one value per subset of a rung)
  dose       dose-response of corruption families
  models     system means against a known order of systems
  pairs      MAUVE human evaluation: per-pair agreement, Bradley-Terry fit,
             system-level correlation with the Bradley-Terry scores
  corr       per-item human ratings (HANNA, OpenMEVA): item- and system-level
             correlation, paired bootstrap test against a reference metric
  length     correlation of each metric with text length
  selftest   synthetic checks with known answers

Output: Markdown to stdout or --out; every table also as rows of one CSV
(--csv, default next to --out) and a JSON file of the same name.

Conventions
  - Oriented score: a lower-is-better metric is negated before any accuracy
    or correlation, so for every metric an accuracy above 0.5 or a positive
    correlation means agreement with the expected order. Means and their
    intervals are printed in the metric's own units.
  - Missing values: NaN and non-finite scores are counted and left out,
    metric by metric.
  - Ties: a tied pair counts one half in pairwise accuracy (which makes it the
    AUROC); Spearman uses midranks; Kendall is tau-b. In `pairs` an exact
    metric tie is dropped and counted.
  - Bootstrap: percentile 95% intervals, 1,000 resamples by default. Each
    generator is seeded from --seed and the name and size of what is
    resampled, so a run is deterministic and metrics with the same complete
    cases see the same resamples (paired comparisons).
"""

import os

# Light CPU use: one BLAS thread unless the caller sets otherwise.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import hashlib
import json
import math
import re
import sys
import tempfile
import warnings
import zlib
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats

DEFAULT_SEED = 20261028
DEFAULT_BOOT = 1000
Z95 = 1.959963984540054
DIGITS = 3              # decimals in the Markdown tables (--digits); the CSV keeps full precision
MINUS = "−"

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", module="scipy")


# ---------------------------------------------------------------------------
# Formatting and output
# ---------------------------------------------------------------------------

def f3(x, nd=None):
    """Number to nd decimals (default --digits) with a typographic minus; n/a when undefined."""
    nd = DIGITS if nd is None else nd
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(x):
        return "n/a"
    s = f"{x:.{nd}f}"
    if s.startswith("-"):
        s = s[1:] if float(s) == 0 else MINUS + s[1:]
    return s


def fci(lo, hi):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "n/a"
    return f"[{f3(lo)}, {f3(hi)}]"


def fmci(m, lo, hi):
    return f"{f3(m)} {fci(lo, hi)}" if np.isfinite(lo) else f3(m)


def esc(s):
    return str(s).replace("|", "\\|")


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


class Report:
    """Collects Markdown lines and machine-readable rows."""

    def __init__(self, title):
        self.lines = [f"# {title}", ""]
        self.tables = {}
        self.notes = []

    def h(self, text, level=2):
        self.lines += [f"{'#' * level} {text}", ""]

    def p(self, text):
        self.lines += [text, ""]

    def note(self, text):
        self.notes.append(text)
        self.p(f"Note: {text}")

    def table(self, headers, rows):
        self.lines.append("| " + " | ".join(esc(h) for h in headers) + " |")
        self.lines.append("|" + "|".join("---" for _ in headers) + "|")
        for r in rows:
            self.lines.append("| " + " | ".join(esc(c) for c in r) + " |")
        self.lines.append("")

    def rows(self, name, rows):
        self.tables.setdefault(name, []).extend(rows)

    def markdown(self):
        return "\n".join(self.lines).rstrip() + "\n"


def write_outputs(rep, args, argv, quiet=False):
    md = rep.markdown()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md, encoding="utf-8")
    elif not quiet:
        sys.stdout.write(md)
    csv_path = args.csv or (str(Path(args.out).with_suffix(".csv")) if args.out else None)
    if not csv_path:
        return
    frames = []
    for name, r in rep.tables.items():
        if not r:
            continue
        df = pd.DataFrame(r).assign(table=name)
        for c in df.columns:
            # nullable integers, so counts stay integers when tables are stacked
            if pd.api.types.is_integer_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
                df[c] = df[c].astype("Int64")
        frames.append(df)
    if frames:
        df = pd.concat(frames, ignore_index=True, sort=False)
        df = df[["table"] + [c for c in df.columns if c != "table"]]
    else:
        df = pd.DataFrame({"table": []})
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    js = {"command": args.cmd, "argv": list(argv), "notes": rep.notes, "tables": rep.tables}
    Path(csv_path).with_suffix(".json").write_text(
        json.dumps(jsonable(js), indent=1, ensure_ascii=False), encoding="utf-8")


def warn(msg):
    print(f"meta.py: {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{ln}: bad JSON: {e}")
    return rows


def get_path(rec, dotted):
    cur = rec
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def load_records(paths, what="texts"):
    """JSONL records as a frame with columns id and _rec (the dict)."""
    recs = []
    for p in paths:
        recs.extend(read_jsonl(p))
    if not recs:
        raise SystemExit(f"no {what} in {paths}")
    ids = []
    for r in recs:
        if "id" not in r:
            raise SystemExit(f"a {what} record has no id: {str(r)[:120]}")
        ids.append(str(r["id"]))
    s = pd.Series(ids)
    if s.duplicated().any():
        raise SystemExit(f"duplicate {what} ids, e.g. {s[s.duplicated()].iloc[0]!r}")
    return pd.DataFrame({"id": ids, "_rec": recs})


def field(frame, key):
    return pd.Series([get_path(r, key) for r in frame["_rec"]], index=frame.index, dtype=object)


def numeric_frame(rows, skip=("id",)):
    """Numeric columns of a list of dicts (non-numeric fields are dropped).
    Nested objects are flattened to dotted names, e.g. {"v2": {"S": 0.4}} -> v2.S."""
    df = pd.json_normalize(rows)
    out = {}
    for c in df.columns:
        if c in skip:
            continue
        col = pd.to_numeric(df[c], errors="coerce")
        if col.notna().any() or df[c].isna().all():
            out[c] = col.astype(float)
    return pd.DataFrame(out, index=df.index)


def load_scores(paths):
    """Join score files on id. The same metric in two files must agree."""
    merged = None
    for p in paths:
        rows = read_jsonl(p)
        if not rows:
            warn(f"{p}: empty score file")
            continue
        if any("id" not in r for r in rows):
            raise SystemExit(f"{p}: a score line has no id")
        ids = pd.Index([str(r["id"]) for r in rows])
        if ids.duplicated().any():
            raise SystemExit(f"{p}: duplicate id {ids[ids.duplicated()][0]!r}")
        df = numeric_frame(rows)
        df.index = ids
        if merged is None:
            merged = df
            continue
        for c in merged.columns.intersection(df.columns):
            both = merged.index.intersection(df.index)
            a = merged.loc[both, c].to_numpy(float)
            b = df.loc[both, c].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            if not np.allclose(a[ok], b[ok], rtol=1e-9, atol=1e-12):
                raise SystemExit(f"metric {c!r} differs between score files ({p})")
        merged = merged.combine_first(df)
    if merged is None:
        raise SystemExit("no scores")
    return merged


@dataclass
class Metric:
    name: str
    sign: int
    label: str

    @property
    def direction(self):
        return "higher is better" if self.sign > 0 else "lower is better"

    @property
    def dir(self):
        return "+" if self.sign > 0 else "-"


_METRIC_RE = re.compile(r"^(.+?):([+-])(?::(.*))?$")


def _sign(v):
    if isinstance(v, bool):
        return 1 if v else -1
    s = str(v).strip().lower()
    if s in ("+", "+1", "1", "higher", "high", "max", "up"):
        return 1
    if s in ("-", "-1", "lower", "low", "min", "down"):
        return -1
    raise SystemExit(f"bad metric direction {v!r} (use + or -)")


def parse_metric_arg(s):
    m = _METRIC_RE.match(s)
    if not m:
        raise SystemExit(f"--metric {s!r}: expected NAME:+ or NAME:- (optionally NAME:+:Label)")
    return Metric(m[1], 1 if m[2] == "+" else -1, m[3] or m[1])


def parse_metrics(args, available):
    specs = []
    if getattr(args, "metrics_config", None):
        cfg = json.loads(Path(args.metrics_config).read_text(encoding="utf-8"))
        if isinstance(cfg, dict) and "metrics" in cfg:
            cfg = cfg["metrics"]
        if isinstance(cfg, dict):
            cfg = [{"name": k, **v} if isinstance(v, dict) else {"name": k, "dir": v}
                   for k, v in cfg.items()]
        for it in cfg:
            if isinstance(it, str):
                specs.append(parse_metric_arg(it))
                continue
            d = next((it[k] for k in ("dir", "direction", "higher_is_better") if k in it), None)
            if d is None:
                raise SystemExit(f"metrics config: no direction for {it.get('name')!r}")
            name = it["name"]
            specs.append(Metric(name, _sign(d), it.get("label") or it.get("display") or name))
    for s in getattr(args, "metric", None) or []:
        specs.append(parse_metric_arg(s))
    if not specs:
        cols = [c for c in available]
        warn("no --metric given; every numeric score column is taken as higher-is-better: "
             + ", ".join(cols))
        specs = [Metric(c, 1, c) for c in cols]
    names = [m.name for m in specs]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise SystemExit(f"metric listed twice: {sorted(dup)}")
    missing = [n for n in names if n not in available]
    if missing:
        raise SystemExit(f"metrics not in the score files: {missing}; available: {sorted(available)}")
    return specs


def apply_where(frame, wheres):
    keep = np.ones(len(frame), bool)
    for w in wheres or []:
        m = re.match(r"^([^!=]+)(!=|=)(.*)$", w)
        if not m:
            raise SystemExit(f"--where {w!r}: expected KEY=VALUE or KEY!=VALUE")
        key, op, val = m.groups()
        allowed = set(val.split("|"))
        vals = field(frame, key.strip()).map(lambda v: "" if v is None else str(v))
        hit = vals.isin(allowed).to_numpy()
        keep &= hit if op == "=" else ~hit
    return frame[keep].reset_index(drop=True)


def parse_systems(args, required=True):
    """Ordered (name, label) list, best first, from --systems or --systems-file."""
    items = []
    if getattr(args, "systems_file", None):
        for line in Path(args.systems_file).read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            name, _, label = line.rstrip("\n").partition("\t")
            items.append((name.strip(), label.strip() or name.strip()))
    elif getattr(args, "systems", None):
        s = args.systems
        parts = re.split(r"[;\n]", s) if (";" in s or "\n" in s) else s.split(",")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            name, eq, label = part.partition("=")
            items.append((name.strip(), label.strip() if eq else name.strip()))
    if required and len(items) < 2:
        raise SystemExit("give at least two systems, best first (--systems a,b,c or --systems-file)")
    names = [n for n, _ in items]
    if len(set(names)) != len(names):
        raise SystemExit("a system is listed twice in the order")
    return items


def text_frame(args):
    """Texts filtered by --where, with the requested metric columns joined on id."""
    texts = load_records(args.texts, "texts")
    texts = apply_where(texts, getattr(args, "where", None))
    if texts.empty:
        raise SystemExit("no texts left after --where")
    scores = load_scores(args.scores)
    metrics = parse_metrics(args, list(scores.columns))
    sub = scores.reindex(texts["id"])
    not_scored = int(sub.isna().all(axis=1).sum())
    for m in metrics:
        texts[m.name] = sub[m.name].to_numpy(float)
    return texts, metrics, not_scored


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def rng_for(seed, *tags):
    """Generator seeded from the run seed and the names of what is resampled."""
    return np.random.default_rng([int(seed)] + [zlib.crc32(str(t).encode("utf-8")) for t in tags])


def digest(idx):
    return hashlib.sha1(np.asarray(idx, np.int64).tobytes()).hexdigest()[:16]


def boot_counts(rng, n, B):
    """B x n resample counts: row b says how often each unit is drawn in resample b."""
    if n == 0:
        return np.zeros((B, 0))
    idx = rng.integers(0, n, size=(B, n))
    flat = (idx + (np.arange(B) * n)[:, None]).ravel()
    return np.bincount(flat, minlength=B * n).reshape(B, n).astype(float)


def boot_counts_strata(rng, strata, B):
    """Resample within each stratum, keeping its size."""
    strata = np.asarray(strata)
    W = np.zeros((B, len(strata)))
    for s in pd.unique(strata):
        idx = np.flatnonzero(strata == s)
        W[:, idx] = boot_counts(rng, idx.size, B)
    return W


def pct_ci(boot):
    b = np.asarray(boot, float)
    b = b[np.isfinite(b)]
    if b.size == 0:
        return math.nan, math.nan
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(lo), float(hi)


def auc(upper, lower):
    """P(upper > lower) + P(tie)/2 over all pairs (the AUROC)."""
    nu, nl = len(upper), len(lower)
    if nu == 0 or nl == 0:
        return math.nan
    r = stats.rankdata(np.concatenate([upper, lower]))
    return float((r[:nu].sum() - nu * (nu + 1) / 2) / (nu * nl))


def weighted_midranks(v, W):
    """Midranks of v within each resample (row of W), as rankdata gives on the
    resample written out with repeats."""
    v = np.asarray(v, float)
    order = np.argsort(v, kind="mergesort")
    vs = v[order]
    new = np.r_[True, vs[1:] != vs[:-1]]
    starts = np.flatnonzero(new)
    gid = np.cumsum(new) - 1
    G = np.add.reduceat(W[:, order], starts, axis=1)
    cum = np.cumsum(G, axis=1)
    grank = cum - G + (G + 1) / 2
    R = np.empty(W.shape)
    R[:, order] = grank[:, gid]
    return R


def auc_boot(xu, xl, Wu, Wl):
    if len(xu) == 0 or len(xl) == 0:
        return np.full(Wu.shape[0], np.nan)
    R = weighted_midranks(np.concatenate([xu, xl]), np.hstack([Wu, Wl]))
    nu, nl = Wu.sum(1), Wl.sum(1)
    ru = (Wu * R[:, :len(xu)]).sum(1)
    return (ru - nu * (nu + 1) / 2) / (nu * nl)


def wpearson(W, x, y):
    """Pearson correlation of 1-D x, y in each resample (rows of W), by
    weighted moments of the centred data (five matrix-vector products)."""
    x = np.asarray(x, float) - np.mean(x)
    y = np.asarray(y, float) - np.mean(y)
    N = W.sum(1)
    mx, my = W @ x / N, W @ y / N
    mxx, myy = W @ (x * x) / N, W @ (y * y) / N
    sxy = W @ (x * y) / N - mx * my
    sxx, syy = mxx - mx * mx, myy - my * my
    with np.errstate(all="ignore"):
        r = sxy / np.sqrt(sxx * syy)
    # a resample that drew only equal values has no correlation
    r[(sxx <= 1e-13 * mxx) | (syy <= 1e-13 * myy)] = np.nan
    return r


def rank_moments(W, R):
    """Second moment of the midranks R about their mean in each resample.
    Midranks of N draws always average (N + 1) / 2, ties or not."""
    N = W.sum(1)
    m = (N + 1) / 2
    return np.einsum("ij,ij,ij->i", W, R, R) / N - m * m


def rank_corr(W, Rx, Ry, vx=None, vy=None):
    """Spearman in each resample from midranks within the resample."""
    N = W.sum(1)
    m = (N + 1) / 2
    vx = rank_moments(W, Rx) if vx is None else vx
    vy = rank_moments(W, Ry) if vy is None else vy
    sxy = np.einsum("ij,ij,ij->i", W, Rx, Ry) / N - m * m
    with np.errstate(all="ignore"):
        r = sxy / np.sqrt(vx * vy)
    r[(vx <= 1e-9) | (vy <= 1e-9)] = np.nan       # all draws tied
    return r


def wspearman(W, x, y):
    return rank_corr(W, weighted_midranks(x, W), weighted_midranks(y, W))


class KendallBoot:
    """Bootstrap Kendall tau-b of many vector pairs over one matrix of counts.

    For counts w (a resample written out with repeats), tau-b of the resample is
        sum_ij w_i w_j sx_ij sy_ij / sqrt(sum_ij w_i w_j |sx_ij| * sum_ij w_i w_j |sy_ij|)
    with sx_ij = sign(x_i - x_j). Copies of one item are tied on both sides, so
    they drop out as tau-b drops tied pairs. The sums are quadratic forms in w,
    computed in blocks of rows; results are cached by key.
    """

    def __init__(self, W, block=256):
        self.W = np.ascontiguousarray(W, dtype=np.float32)
        self.block = block
        self.cache = {}

    def _quad(self, x, y, key):
        if key in self.cache:
            return self.cache[key]
        W, n = self.W, self.W.shape[1]
        out = np.zeros(W.shape[0])
        for s in range(0, n, self.block):
            e = min(n, s + self.block)
            sx = np.sign(x[s:e, None] - x[None, :])
            a = sx * np.sign(y[s:e, None] - y[None, :]) if y is not None else np.abs(sx)
            out += ((W[:, s:e] @ a.astype(np.float32)) * W).sum(1, dtype=np.float64)
        self.cache[key] = out
        return out

    def tau(self, x, y, kx, ky):
        num = self._quad(x, y, ("xy", kx, ky))
        dx = self._quad(x, None, ("x", kx))
        dy = self._quad(y, None, ("x", ky))
        with np.errstate(all="ignore"):
            t = num / np.sqrt(dx * dy)
        t[(dx <= 0) | (dy <= 0)] = np.nan
        return t


def corr3(x, y):
    """Pearson, Spearman and Kendall tau-b (NaN when undefined)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan, math.nan, math.nan
    return (float(np.corrcoef(x, y)[0, 1]), float(stats.spearmanr(x, y)[0]),
            float(stats.kendalltau(x, y)[0]))


def rowwise_pearson(A, Bm):
    A = np.asarray(A, float)
    Bm = np.broadcast_to(np.asarray(Bm, float), A.shape)
    a = A - A.mean(1, keepdims=True)
    b = Bm - Bm.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    with np.errstate(all="ignore"):
        r = (a * b).sum(1) / den
    r[~(den > 0)] = np.nan
    return r


def rowwise_spearman(A, Bm):
    A = np.asarray(A, float)
    Bm = np.broadcast_to(np.asarray(Bm, float), A.shape)
    return rowwise_pearson(stats.rankdata(A, axis=1), stats.rankdata(Bm, axis=1))


def rowwise_kendall(A, Bm):
    """Kendall tau-b per row of two B x k matrices (small k)."""
    A = np.asarray(A, float)
    Bm = np.broadcast_to(np.asarray(Bm, float), A.shape)
    sa = np.sign(A[:, :, None] - A[:, None, :])
    sb = np.sign(Bm[:, :, None] - Bm[:, None, :])
    den = np.sqrt(np.abs(sa).sum((1, 2)) * np.abs(sb).sum((1, 2)))
    with np.errstate(all="ignore"):
        t = (sa * sb).sum((1, 2)) / den
    t[~(den > 0) | np.isnan(A).any(1) | np.isnan(Bm).any(1)] = np.nan
    return t


def sys_corr(a, b):
    """Spearman and Kendall tau-b between two vectors of system values."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return math.nan, math.nan
    return float(stats.spearmanr(a, b)[0]), float(stats.kendalltau(a, b)[0])


def wilson(k, n, z=Z95):
    if n == 0:
        return math.nan, math.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z / den * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, c - h), min(1.0, c + h)


def bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values (NaN entries stay NaN)."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if ok.size == 0:
        return out
    q = p[ok]
    order = np.argsort(q)
    m = q.size
    adj = q[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def fit_bt(wins, ridge=0.01, tol=1e-10, max_iter=200):
    """Bradley-Terry strengths by penalised maximum likelihood (Newton).

    wins[i, j] is how often i beat j. Maximises
        sum_ij wins_ij log sigmoid(t_i - t_j) - ridge/2 * sum_i t_i^2.
    The likelihood only sees differences, so the optimum has mean zero; the
    ridge keeps the fit finite when a system never loses, and with hundreds of
    comparisons per system its pull is negligible. t stays centred and each
    Newton system is solved with 11'/k added, which leaves steps orthogonal to
    1 unchanged and keeps the solve well conditioned for any ridge >= 0. t is
    on the natural-log scale: P(i beats j) = sigmoid(t_i - t_j).
    """
    wins = np.asarray(wins, float)
    k = len(wins)
    N = wins + wins.T
    t = np.zeros(k)
    J = np.ones((k, k)) / k

    def loglik(t):
        d = t[:, None] - t[None, :]
        return float((wins * -np.logaddexp(0, -d)).sum() - ridge / 2 * (t * t).sum())

    cur = loglik(t)
    for _ in range(max_iter):
        d = t[:, None] - t[None, :]
        p = 1 / (1 + np.exp(-d))
        g = (wins - N * p).sum(1) - ridge * t
        h = N * p * (1 - p)
        H = h.copy()
        np.fill_diagonal(H, 0)
        np.fill_diagonal(H, -H.sum(1) - ridge)
        g -= g.mean()
        try:
            step = np.linalg.solve(J - H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(J - H, g, rcond=None)[0]
        step -= step.mean()
        s = 1.0
        while s > 1e-8:
            new = loglik(t + s * step)
            if new >= cur - 1e-12:
                break
            s /= 2
        t = t + s * step
        cur = loglik(t)
        if np.max(np.abs(s * step)) < tol:
            break
    return t


def fit_bt_mm(wins, max_iter=100000, tol=1e-13):
    """Zermelo / MM iteration (as in the MAUVE notebook), centred log strengths.
    Used only to check fit_bt."""
    wins = np.asarray(wins, float)
    N = wins + wins.T
    w = wins.sum(1)
    p = np.full(len(wins), 1 / len(wins))
    for _ in range(max_iter):
        q = w / (N / (p[:, None] + p[None, :])).sum(1)
        q /= q.sum()
        if np.abs(q - p).sum() < tol:
            p = q
            break
        p = q
    t = np.log(p)
    return t - t.mean()


# ---------------------------------------------------------------------------
# Groups of units (rungs, systems, strengths) and their pairwise accuracy
# ---------------------------------------------------------------------------

@dataclass
class Group:
    key: str
    label: str
    x: np.ndarray          # raw values, finite only
    missing: int
    sign: int
    W: np.ndarray          # B x n resample counts
    mean: float
    lo: float
    hi: float
    boot: np.ndarray = dc_field(repr=False)   # bootstrap means (raw units)

    @property
    def n(self):
        return int(self.x.size)

    @property
    def xo(self):
        return self.sign * self.x

    @property
    def olo(self):
        return self.lo if self.sign > 0 else -self.hi

    @property
    def ohi(self):
        return self.hi if self.sign > 0 else -self.lo


def make_group(key, label, raw, sign, B, seed, tag="group"):
    raw = np.asarray(raw, float)
    ok = np.isfinite(raw)
    x = raw[ok]
    W = boot_counts(rng_for(seed, tag, key, x.size), x.size, B)
    bm = W @ x / x.size if x.size else np.full(B, np.nan)
    lo, hi = pct_ci(bm)
    mean = float(x.mean()) if x.size else math.nan
    return Group(str(key), str(label), x, int((~ok).sum()), sign, W, mean, lo, hi, bm)


def pair_row(u, l, pass_acc=None):
    """Accuracy of upper > lower with its interval, and the separation test."""
    acc = auc(u.xo, l.xo)
    lo, hi = pct_ci(auc_boot(u.xo, l.xo, u.W, l.W))
    disjoint = bool(np.isfinite(u.olo) and np.isfinite(l.ohi) and u.olo > l.ohi)
    row = {"upper": u.key, "lower": l.key, "upper_label": u.label, "lower_label": l.label,
           "n_upper": u.n, "n_lower": l.n, "acc": acc, "acc_lo": lo, "acc_hi": hi,
           "disjoint": disjoint}
    if pass_acc is not None:
        row["pass"] = bool(np.isfinite(acc) and acc >= pass_acc and disjoint)
    return row


# ---------------------------------------------------------------------------
# ladder and setladder
# ---------------------------------------------------------------------------

def ladder_core(rep, frame, gcol, order, metrics, args, unit, tag):
    """Rung means, all pairwise accuracies, and the pass rule, per metric."""
    k = len(order)
    keys = frame[gcol].map(lambda v: "" if v is None else str(v)).to_numpy()
    absent = [n for n, _ in order if not (keys == n).any()]
    if absent:
        have = sorted(set(keys))[:30]
        raise SystemExit(f"systems not found in the {unit}: {absent}; present: {have}")
    adj_names = [f"{order[i][1]} > {order[i + 1][1]}" for i in range(k - 1)]
    summary = []
    for m in metrics:
        groups = [make_group(n, lab, frame.loc[keys == n, m.name].to_numpy(float), m.sign,
                             args.boot, args.seed, tag) for n, lab in order]
        rep.h(f"{m.label} ({m.direction})", 3)
        rep.table(["rung", "n", "missing", "mean [95% interval]"],
                  [[g.label, g.n, g.missing, fmci(g.mean, g.lo, g.hi)] for g in groups])
        rep.rows("rungs", [{"metric": m.name, "label": m.label, "dir": m.dir, "rung": g.key,
                            "rung_label": g.label, "position": i, "n": g.n, "missing": g.missing,
                            "mean": g.mean, "lo": g.lo, "hi": g.hi}
                           for i, g in enumerate(groups)])
        mat = {}
        rows = []
        for i in range(k):
            for j in range(i + 1, k):
                adjacent = j == i + 1
                r = pair_row(groups[i], groups[j], args.pass_acc if adjacent else None)
                r.update({"metric": m.name, "label": m.label, "adjacent": adjacent})
                mat[i, j] = r["acc"]
                rows.append(r)
        rep.rows("pairs", rows)
        adj = [r for r in rows if r["adjacent"]]
        rep.table(["upper > lower", "accuracy", "95% interval", "intervals disjoint", "verdict"],
                  [[f"{r['upper_label']} > {r['lower_label']}", f3(r["acc"]),
                    fci(r["acc_lo"], r["acc_hi"]), "yes" if r["disjoint"] else "no",
                    "**PASS**" if r["pass"] else "FAIL"] for r in adj])
        npass = sum(r["pass"] for r in adj)
        rep.p(f"Passed {npass} of {k - 1} adjacent pairs.")
        if k > 2:
            rep.p("All pairs (accuracy of row > column):")
            rep.table(["upper \\ lower"] + [g.label for g in groups[1:]],
                      [[groups[i].label] + [f3(mat[i, j]) if j > i else "" for j in range(1, k)]
                       for i in range(k - 1)])
        summary.append((m, adj, npass))
    rep.h("Summary", 2)
    rep.p(f"Adjacent pairs, accuracy and verdict (pass: accuracy >= {args.pass_acc} "
          f"and disjoint intervals in the right order).")
    rep.table(["metric"] + adj_names + ["passed"],
              [[m.label] + [f"{f3(r['acc'])} {'**PASS**' if r['pass'] else 'FAIL'}" for r in adj]
               + [f"{npass} of {k - 1}"] for m, adj, npass in summary])
    rep.rows("summary", [{"metric": m.name, "label": m.label, "dir": m.dir, "passed": npass,
                          "adjacent_pairs": k - 1} for m, adj, npass in summary])


def ladder_preamble(rep, args, unit, n_units):
    rep.p(f"{n_units} {unit}. Bootstrap: {args.boot} resamples, seed {args.seed}, percentile 95% "
          f"intervals. Accuracy is P(upper > lower) over all (upper, lower) pairs with ties "
          f"one half (the AUROC), computed on oriented scores (lower-is-better metrics negated). "
          f"\"Intervals disjoint\" means the rung-mean intervals do not overlap and are in "
          f"the right order. A pair passes when accuracy >= {args.pass_acc} and the intervals "
          f"are disjoint.")


def cmd_ladder(args):
    texts, metrics, not_scored = text_frame(args)
    order = parse_systems(args)
    names = {n for n, _ in order}
    gkeys = field(texts, args.group_key).map(lambda v: "" if v is None else str(v))
    absent = sorted(names - set(gkeys))
    if absent:
        raise SystemExit(f"systems not found in the texts: {absent}; present: {sorted(set(gkeys))[:30]}")
    texts = texts[gkeys.isin(names).to_numpy()].reset_index(drop=True)
    texts["_g"] = gkeys[gkeys.isin(names)].to_numpy()
    rep = Report("Control ladder")
    ladder_preamble(rep, args, "texts", len(texts))
    if not_scored:
        rep.note(f"{not_scored} texts have no score in any listed file")
    ladder_core(rep, texts, "_g", order, metrics, args, "texts", "group")
    return rep


def cmd_setladder(args):
    recs = []
    for p in args.values:
        recs.extend(read_jsonl(p))
    if not recs:
        raise SystemExit("no values")
    frame = pd.DataFrame({"_rec": recs})
    frame = apply_where(frame, args.where)
    nums = numeric_frame(list(frame["_rec"]), skip=("id", args.group_key, args.subset_key))
    metrics = parse_metrics(args, list(nums.columns))
    order = parse_systems(args)
    for m in metrics:
        frame[m.name] = nums[m.name].to_numpy(float)
    frame["_g"] = field(frame, args.group_key).map(lambda v: "" if v is None else str(v))
    absent = sorted({n for n, _ in order} - set(frame["_g"]))
    if absent:
        raise SystemExit(f"groups not found in the values: {absent}; present: {sorted(set(frame['_g']))[:30]}")
    frame = frame[frame["_g"].isin({n for n, _ in order})].reset_index(drop=True)
    rep = Report("Set-level control ladder")
    ladder_preamble(rep, args, "subsets", len(frame))
    rep.p("Units are subsets: accuracy is over all (upper subset, lower subset) pairs and the "
          "bootstrap resamples subsets within each rung.")
    ladder_core(rep, frame, "_g", order, metrics, args, "subsets", "setgroup")
    return rep


# ---------------------------------------------------------------------------
# dose
# ---------------------------------------------------------------------------

def cmd_dose(args):
    texts, metrics, not_scored = text_frame(args)
    fam = field(texts, args.family_key).map(lambda v: None if v is None else str(v))
    strength = pd.to_numeric(field(texts, args.strength_key), errors="coerce").to_numpy(float)
    base_names = set(args.base_family.split("|"))
    is_base_fam = fam.isna().to_numpy() | fam.isin(base_names).to_numpy()
    base = is_base_fam & (strength == 0)
    if args.families:
        families = [f.strip() for f in args.families.split(",") if f.strip()]
    else:
        families = sorted({f for f, b in zip(fam, is_base_fam) if not b})
    if not families:
        raise SystemExit(f"no corruption families under {args.family_key}")
    rep = Report("Dose-response")
    rep.p(f"{len(texts)} texts, families from `{args.family_key}`, strengths from "
          f"`{args.strength_key}`. A family without its own strength-0 texts uses the shared "
          f"base texts (strength 0, family {sorted(base_names)} or none): {int(base.sum())}. "
          f"Bootstrap: {args.boot} resamples within each strength, seed {args.seed}. Spearman "
          f"uses the oriented score, so −1 means quality falls perfectly with strength; a step "
          f"moves the right way when the oriented mean falls.")
    if not_scored:
        rep.note(f"{not_scored} texts have no score in any listed file")
    summary = {}
    for f in families:
        own = (fam == f).to_numpy() & np.isfinite(strength)
        rows = own | (base if not (own & (strength == 0)).any() else np.zeros_like(own))
        if not rows.any():
            rep.note(f"family {f!r} has no texts")
            continue
        levels = sorted(set(strength[rows]))
        rep.h(f"Family: {f}", 2)
        for m in metrics:
            vals = texts[m.name].to_numpy(float)
            groups = [make_group(f"{f}@{s:g}", f"{s:g}", vals[rows & (strength == s)], m.sign,
                                 args.boot, args.seed, "dose") for s in levels]
            # sample level: all complete texts of the family, resampled within strengths
            s_all = np.concatenate([np.full(g.n, s) for g, s in zip(groups, levels)])
            xo_all = np.concatenate([g.xo for g in groups])
            W_all = np.hstack([g.W for g in groups])
            rho_s = float(stats.spearmanr(s_all, xo_all)[0]) if len(xo_all) > 2 and np.ptp(xo_all) > 0 else math.nan
            lo_s, hi_s = pct_ci(wspearman(W_all, s_all, xo_all)) if len(xo_all) > 2 else (math.nan, math.nan)
            have = [i for i, g in enumerate(groups) if g.n > 0]
            mo = np.array([m.sign * groups[i].mean for i in have])
            lv = np.array([levels[i] for i in have])
            rho_m = float(stats.spearmanr(lv, mo)[0]) if len(mo) > 2 and np.ptp(mo) > 0 else math.nan
            steps = np.diff(mo)
            right = int((steps < 0).sum())
            nsteps = int(steps.size)
            rep.h(f"{m.label} ({m.direction}), {f}", 3)
            rep.table(["strength", "n", "missing", "mean [95% interval]"],
                      [[g.label, g.n, g.missing, fmci(g.mean, g.lo, g.hi)] for g in groups])
            rep.p(f"Spearman, samples: {f3(rho_s)} {fci(lo_s, hi_s)}. Spearman, means: "
                  f"{f3(rho_m)}. Steps the right way: {right} of {nsteps}.")
            rep.rows("dose_levels", [{"metric": m.name, "label": m.label, "dir": m.dir,
                                      "family": f, "strength": s, "n": g.n, "missing": g.missing,
                                      "mean": g.mean, "lo": g.lo, "hi": g.hi}
                                     for g, s in zip(groups, levels)])
            res = {"metric": m.name, "label": m.label, "dir": m.dir, "family": f,
                   "spearman_samples": rho_s, "spearman_samples_lo": lo_s,
                   "spearman_samples_hi": hi_s, "spearman_means": rho_m, "steps_right": right,
                   "steps": nsteps, "share_right": right / nsteps if nsteps else math.nan,
                   "monotone": bool(nsteps and right == nsteps)}
            rep.rows("dose", [res])
            summary[m.name, f] = res
    fams = [f for f in families if any((m.name, f) in summary for m in metrics)]
    rep.h("Summary", 2)
    rep.p(f"Spearman of the oriented means with strength (steps the right way). Last column: "
          f"families with Spearman on the means <= {f3(args.rho_threshold)}.")
    rows = []
    for m in metrics:
        cells, hits = [], 0
        for f in fams:
            r = summary.get((m.name, f))
            if r is None:
                cells.append("n/a")
                continue
            cells.append(f"{f3(r['spearman_means'])} ({r['steps_right']}/{r['steps']})")
            hits += bool(np.isfinite(r["spearman_means"]) and r["spearman_means"] <= args.rho_threshold)
        rows.append([m.label] + cells + [f"{hits} of {len(fams)}"])
    rep.table(["metric"] + fams + ["families hit"], rows)
    rep.p("Spearman at sample level [95% interval]:")
    rep.table(["metric"] + fams,
              [[m.label] + [fmci(summary[m.name, f]["spearman_samples"],
                                 summary[m.name, f]["spearman_samples_lo"],
                                 summary[m.name, f]["spearman_samples_hi"])
                            if (m.name, f) in summary else "n/a" for f in fams] for m in metrics])
    return rep


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def cmd_models(args):
    texts, metrics, not_scored = text_frame(args)
    order = parse_systems(args)
    k = len(order)
    gk = field(texts, args.group_key).map(lambda v: "" if v is None else str(v)).to_numpy()
    absent = [n for n, _ in order if not (gk == n).any()]
    if absent:
        raise SystemExit(f"systems not found: {absent}; present: {sorted(set(gk))[:30]}")
    known = np.arange(k, 0, -1, dtype=float)       # best system gets the highest value
    rep = Report("Model ladder")
    rep.p(f"{int(np.isin(gk, [n for n, _ in order]).sum())} texts in {k} systems, known order "
          f"best first: {', '.join(lab for _, lab in order)}. Spearman and Kendall (tau-b) are "
          f"between the oriented system means and the known order (1 = the known order). "
          f"Bootstrap: {args.boot} resamples of the texts within each system, seed {args.seed}; "
          f"P(perfect) is the share of resamples whose system means are in exactly the known "
          f"order. Adjacent accuracy is P(upper > lower) over text pairs, ties one half.")
    if not_scored:
        rep.note(f"{not_scored} texts have no score in any listed file")
    summary = []
    for m in metrics:
        vals = texts[m.name].to_numpy(float)
        groups = [make_group(n, lab, vals[gk == n], m.sign, args.boot, args.seed, "group")
                  for n, lab in order]
        mo = np.array([m.sign * g.mean for g in groups])
        rho, tau = sys_corr(known, mo)
        BM = m.sign * np.column_stack([g.boot for g in groups])
        rb, tb = rowwise_spearman(BM, known), rowwise_kendall(BM, known)
        ok = np.isfinite(BM).all(1)
        p_perfect = float(np.mean(np.all(BM[ok, :-1] > BM[ok, 1:], axis=1))) if ok.any() else math.nan
        perfect = bool(np.all(mo[:-1] > mo[1:]))
        adj = []
        for i in range(k - 1):
            r = pair_row(groups[i], groups[i + 1])
            r.update({"metric": m.name, "label": m.label})
            adj.append(r)
        res = {"metric": m.name, "label": m.label, "dir": m.dir, "spearman": rho,
               "spearman_lo": pct_ci(rb)[0], "spearman_hi": pct_ci(rb)[1], "kendall": tau,
               "kendall_lo": pct_ci(tb)[0], "kendall_hi": pct_ci(tb)[1],
               "p_perfect": p_perfect, "perfect": perfect}
        rep.h(f"{m.label} ({m.direction})", 3)
        rep.table(["system", "n", "missing", "mean [95% interval]"],
                  [[g.label, g.n, g.missing, fmci(g.mean, g.lo, g.hi)] for g in groups])
        rep.table(["upper > lower", "accuracy", "95% interval", "intervals disjoint"],
                  [[f"{r['upper_label']} > {r['lower_label']}", f3(r["acc"]),
                    fci(r["acc_lo"], r["acc_hi"]), "yes" if r["disjoint"] else "no"] for r in adj])
        rep.p(f"Spearman {fmci(rho, res['spearman_lo'], res['spearman_hi'])}, Kendall "
              f"{fmci(tau, res['kendall_lo'], res['kendall_hi'])}, in the known order: "
              f"{'yes' if perfect else 'no'}, P(perfect) {f3(p_perfect)}.")
        rep.rows("systems", [{"metric": m.name, "label": m.label, "dir": m.dir, "system": g.key,
                              "system_label": g.label, "position": i, "n": g.n,
                              "missing": g.missing, "mean": g.mean, "lo": g.lo, "hi": g.hi}
                             for i, g in enumerate(groups)])
        rep.rows("adjacent", adj)
        rep.rows("order", [res])
        summary.append((m, res, adj))
    rep.h("Summary", 2)
    rep.table(["metric", "Spearman [95% interval]", "Kendall [95% interval]", "in order",
               "P(perfect)"] + [f"{order[i][1]} > {order[i + 1][1]}" for i in range(k - 1)],
              [[m.label, fmci(r["spearman"], r["spearman_lo"], r["spearman_hi"]),
                fmci(r["kendall"], r["kendall_lo"], r["kendall_hi"]),
                "yes" if r["perfect"] else "no", f3(r["p_perfect"])]
               + [f3(a["acc"]) for a in adj] for m, r, adj in summary])
    return rep


# ---------------------------------------------------------------------------
# pairs (MAUVE human evaluation)
# ---------------------------------------------------------------------------

ANSWER_CODES = {"2a": (1, True), "1a": (1, False), "1b": (-1, False), "2b": (-1, True)}


def is_human_system(name, human):
    return str(name).strip().lower() in {h.strip().lower() for h in human.split("|")}


def cmd_pairs(args):
    texts = load_records(args.texts, "texts")
    scores = load_scores(args.scores)
    metrics = parse_metrics(args, list(scores.columns))
    sysmap = dict(zip(texts["id"], field(texts, args.system_key).map(str)))
    recs = read_jsonl(args.pairs)
    if args.where:                                   # --where filters the judgments
        recs = list(apply_where(pd.DataFrame({"_rec": recs}), args.where)["_rec"])
    if not recs:
        raise SystemExit("no pairs")
    a_ids = np.array([str(r.get("a")) for r in recs], dtype=object)
    b_ids = np.array([str(r.get("b")) for r in recs], dtype=object)
    unknown = sorted({t for t in np.r_[a_ids, b_ids] if t not in sysmap})
    if unknown:
        raise SystemExit(f"{len(unknown)} text ids of the pairs file are not in the texts, e.g. {unknown[:3]}")
    P = len(recs)
    questions = [q.strip() for q in args.questions.split(",") if q.strip()]

    # time filter (MAUVE paper: answers under 25 s of the estimated time are dropped)
    te = np.array([pd.to_numeric(r.get(args.time_key), errors="coerce") if r.get(args.time_key)
                   is not None else np.nan for r in recs], float)
    if np.isfinite(te).any() and args.min_time > 0:
        keep_time = ~(te <= args.min_time)      # a judgment with no time is kept
        time_note = (f"judgments with `{args.time_key}` <= {args.min_time} s dropped: "
                     f"{int((~keep_time).sum())} of {P}")
    else:
        keep_time = np.ones(P, bool)
        time_note = "no time filter"

    U = pd.unique(np.r_[a_ids, b_ids])
    pos = {t: i for i, t in enumerate(U)}
    ai = np.array([pos[t] for t in a_ids])
    bi = np.array([pos[t] for t in b_ids])
    usys = np.array([sysmap[t] for t in U], dtype=object)
    systems = sorted(set(usys))
    k = len(systems)
    sidx = {s: i for i, s in enumerate(systems)}
    su = np.array([sidx[s] for s in usys])
    sa, sb = su[ai], su[bi]
    S = np.zeros((len(U), k))
    S[np.arange(len(U)), su] = 1
    models = np.array([not is_human_system(s, args.human_system) for s in systems])
    sub = scores.reindex(U)
    V = {m.name: m.sign * sub[m.name].to_numpy(float) for m in metrics}

    # one set of pair resamples for every question and metric
    C = boot_counts(rng_for(args.seed, "pairs", P), P, args.boot)
    Mi = sparse.csr_matrix((np.ones(2 * P), (np.r_[np.arange(P), np.arange(P)], np.r_[ai, bi])),
                           shape=(P, len(U)))
    present = (Mi.T @ C.T).T > 0                     # B x texts: text drawn in the resample

    # metric system means over the unique texts of the pairs (all pairs, before the time filter)
    sys_means, sys_boot = {}, {}
    for m in metrics:
        v = V[m.name]
        ok = np.isfinite(v)
        v0 = np.where(ok, v, 0.0)
        cnt = S[ok].sum(0)
        sys_means[m.name] = np.where(cnt > 0, (S * v0[:, None]).sum(0) / np.maximum(cnt, 1), np.nan)
        Wt = (present & ok).astype(float)
        with np.errstate(all="ignore"):
            sys_boot[m.name] = (Wt * v0) @ S / (Wt @ S)

    rep = Report("MAUVE human evaluation: pairwise judgments")
    rep.p(f"{P} judged pairs over {len(U)} texts in {k} systems ({', '.join(systems)}). "
          f"{time_note}. Codes: 2a definitely A, 1a slightly A or tie, 1b slightly B, 2b "
          f"definitely B; a 1a or 2a is a win for A (as in the MAUVE paper). Metric agreement: the "
          f"metric prefers the text with the higher oriented score; exact metric ties and "
          f"missing scores are dropped; Wilson 95% intervals. Bradley-Terry: penalised maximum "
          f"likelihood (ridge {args.bt_ridge}), reported as 100 x log-strength centred at 0 "
          f"(the MAUVE paper's scale). System-level correlations are between the metric's mean "
          f"oriented score per system (over the unique texts of the pairs) and the BT scores, "
          f"over the models (human excluded, as in the MAUVE paper) and over all systems. "
          f"Bootstrap: {args.boot} resamples of the pairs, seed {args.seed}; each resample "
          f"refits BT and recomputes the system means over the texts it contains.")

    for q in questions:
        codes = [str(r.get(q, "")).strip().lower() for r in recs]
        side = np.array([ANSWER_CODES.get(c, (0, False))[0] for c in codes])
        definite = np.array([ANSWER_CODES.get(c, (0, False))[1] for c in codes])
        valid = (side != 0) & keep_time
        n_bad = int(((side == 0) & keep_time).sum())
        rep.h(f"Question {q}", 2)
        if n_bad:
            rep.note(f"{q}: {n_bad} judgments with an unknown answer code dropped")

        # (a) per-pair agreement
        arows = []
        for m in metrics:
            d = V[m.name][ai] - V[m.name][bi]
            fin = valid & np.isfinite(d)
            tie = fin & (d == 0)
            use = fin & ~tie
            agree = np.sign(d) == side
            n, kk = int(use.sum()), int((use & agree).sum())
            nd, kd = int((use & definite).sum()), int((use & definite & agree).sum())
            lo, hi = wilson(kk, n)
            lod, hid = wilson(kd, nd)
            arows.append({"question": q, "metric": m.name, "label": m.label, "dir": m.dir,
                          "n": n, "agree": kk, "rate": kk / n if n else math.nan, "lo": lo,
                          "hi": hi, "n_definite": nd, "agree_definite": kd,
                          "rate_definite": kd / nd if nd else math.nan, "lo_definite": lod,
                          "hi_definite": hid, "ties_dropped": int(tie.sum()),
                          "missing": int((valid & ~np.isfinite(d)).sum())})
        rep.p("Per-pair agreement with the people's choice:")
        rep.table(["metric", "n", "agreement [95% interval]", "n definite",
                   "definite only [95% interval]", "metric ties", "missing"],
                  [[r["label"], r["n"], fmci(r["rate"], r["lo"], r["hi"]), r["n_definite"],
                    fmci(r["rate_definite"], r["lo_definite"], r["hi_definite"]),
                    r["ties_dropped"], r["missing"]] for r in arows])
        rep.rows("agreement", arows)

        # (b) Bradley-Terry on wins per system (pairs within one system carry no information)
        win_s = np.where(side > 0, sa, sb)
        los_s = np.where(side > 0, sb, sa)
        vbt = valid & (sa != sb)
        wins = np.zeros((k, k))
        np.add.at(wins, (win_s[vbt], los_s[vbt]), 1)
        theta = fit_bt(wins, args.bt_ridge)
        vi = np.flatnonzero(vbt)
        E = sparse.csr_matrix((np.ones(vi.size), (vi, win_s[vi] * k + los_s[vi])), shape=(P, k * k))
        Wb = np.asarray((E.T @ C.T).T)
        theta_b = np.array([fit_bt(Wb[b].reshape(k, k), args.bt_ridge) for b in range(args.boot)])
        brows = []
        for i in np.argsort(-theta):
            lo, hi = pct_ci(100 * theta_b[:, i])
            n_cmp = wins[i].sum() + wins[:, i].sum()
            brows.append({"question": q, "system": systems[i], "wins": int(wins[i].sum()),
                          "comparisons": int(n_cmp),
                          "win_rate": wins[i].sum() / n_cmp if n_cmp else math.nan,
                          "bt": 100 * theta[i], "bt_lo": lo, "bt_hi": hi})
        rep.p("Bradley-Terry scores of the people's choices:")
        rep.table(["system", "wins", "comparisons", "win rate", "BT (x100) [95% interval]"],
                  [[r["system"], r["wins"], r["comparisons"], f3(r["win_rate"]),
                    fmci(r["bt"], r["bt_lo"], r["bt_hi"])] for r in brows])
        rep.rows("bradley_terry", brows)

        # (c) system-level correlation with the BT scores
        crows = []
        for m in metrics:
            row = {"question": q, "metric": m.name, "label": m.label, "dir": m.dir}
            for tag, sel in (("models", models), ("all", np.ones(k, bool))):
                if sel.sum() < 3:
                    continue
                rho, tau = sys_corr(theta[sel], sys_means[m.name][sel])
                A, Bm = theta_b[:, sel], sys_boot[m.name][:, sel]
                rb, tb = rowwise_spearman(A, Bm), rowwise_kendall(A, Bm)
                row.update({f"spearman_{tag}": rho, f"spearman_{tag}_lo": pct_ci(rb)[0],
                            f"spearman_{tag}_hi": pct_ci(rb)[1], f"kendall_{tag}": tau,
                            f"kendall_{tag}_lo": pct_ci(tb)[0], f"kendall_{tag}_hi": pct_ci(tb)[1],
                            f"n_systems_{tag}": int(sel.sum())})
            crows.append(row)
        g = lambda r, key: fmci(r.get(key, math.nan), r.get(key + "_lo", math.nan), r.get(key + "_hi", math.nan))
        rep.p(f"System level, metric means against BT scores ({int(models.sum())} models; "
              f"{k} systems with human):")
        rep.table(["metric", "Spearman, models", "Kendall, models", "Spearman, all", "Kendall, all"],
                  [[r["label"], g(r, "spearman_models"), g(r, "kendall_models"),
                    g(r, "spearman_all"), g(r, "kendall_all")] for r in crows])
        rep.rows("system_corr", crows)

    rep.rows("system_means", [{"metric": m.name, "label": m.label, "dir": m.dir, "system": s,
                               "n_texts": int(S[np.isfinite(V[m.name]), i].sum()),
                               "mean_oriented": sys_means[m.name][i]}
                              for m in metrics for i, s in enumerate(systems)])

    # agreement between workers on pairs judged more than once
    rep.h("Agreement between workers", 2)
    keys = [tuple(sorted((a, b))) for a, b in zip(a_ids, b_ids)]
    groups = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    rep_groups = [ix for ix in groups.values() if len(ix) > 1]
    workers = {str(r.get(args.worker_key)) for r in recs if r.get(args.worker_key) is not None}
    if not rep_groups:
        rep.p(f"No pair of texts was judged more than once ({P} judgments, {len(workers)} workers), "
              f"so agreement between workers cannot be measured on this file.")
    else:
        wrows = []
        for q in questions:
            num = den = 0
            fl_items = []
            for ix in rep_groups:
                prefs = []
                for i in ix:
                    s, _ = ANSWER_CODES.get(str(recs[i].get(q, "")).strip().lower(), (0, False))
                    if s == 0 or not keep_time[i]:
                        continue
                    prefs.append(a_ids[i] if s > 0 else b_ids[i])
                if len(prefs) < 2:
                    continue
                first = sum(p == keys[ix[0]][0] for p in prefs)
                second = len(prefs) - first
                num += first * (first - 1) / 2 + second * (second - 1) / 2
                den += len(prefs) * (len(prefs) - 1) / 2
                fl_items.append((first, second))
            if fl_items:
                arr = np.array(fl_items, float)
                n_i = arr.sum(1)
                p_i = (arr * (arr - 1)).sum(1) / (n_i * (n_i - 1))
                pc = arr.sum(0) / n_i.sum()
                pe = float((pc ** 2).sum())
                kappa = (p_i.mean() - pe) / (1 - pe) if pe < 1 else math.nan
            else:
                kappa = math.nan
            wrows.append({"question": q, "repeated_pairs": len(fl_items),
                          "judgment_pairs": den, "agreement": num / den if den else math.nan,
                          "fleiss_kappa": kappa})
        rep.p(f"{len(rep_groups)} pairs of texts were judged more than once. Agreement is the share "
              f"of pairs of judgments that prefer the same text; Fleiss' kappa for the binary choice.")
        rep.table(["question", "repeated pairs", "judgment pairs", "agreement", "Fleiss kappa"],
                  [[r["question"], r["repeated_pairs"], int(r["judgment_pairs"]), f3(r["agreement"]),
                    f3(r["fleiss_kappa"])] for r in wrows])
        rep.rows("worker_agreement", wrows)
    return rep


# ---------------------------------------------------------------------------
# corr (per-item human ratings)
# ---------------------------------------------------------------------------

def _ratings(v):
    """(mean, list of single ratings) from a stored rating: number, list or dict."""
    def num(x):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return math.nan
        return x if math.isfinite(x) else math.nan
    if v is None:
        return math.nan, []
    if isinstance(v, dict):
        raters = v.get("raters", v.get("ratings", v.get("scores")))
        raters = [num(x) for x in raters] if isinstance(raters, (list, tuple)) else []
        raters = [x for x in raters if math.isfinite(x)]
        mean = num(v["mean"]) if "mean" in v else (float(np.mean(raters)) if raters else math.nan)
        return mean, raters
    if isinstance(v, (list, tuple)):
        raters = [x for x in (num(x) for x in v) if math.isfinite(x)]
        return (float(np.mean(raters)) if raters else math.nan), raters
    return num(v), []


def human_table(texts, key, criteria):
    H = field(texts, key)
    if criteria:
        crits = criteria
    else:
        crits = sorted({c for h in H if isinstance(h, dict) for c, v in h.items()
                        if math.isfinite(_ratings(v)[0]) or _ratings(v)[1]})
    if not crits:
        raise SystemExit(f"no human ratings under {key}")
    out = {}
    for c in crits:
        means, raters = [], []
        for h in H:
            mu, rs = _ratings(h.get(c) if isinstance(h, dict) else None)
            means.append(mu)
            raters.append(rs)
        r = max((len(x) for x in raters), default=0)
        R = np.full((len(H), r), np.nan)
        for i, x in enumerate(raters):
            R[i, :len(x)] = x
        out[c] = (np.array(means, float), R)
    return out


def loo_ceiling(R, rng, draws=20):
    """Human ceiling: one rating of each item against the mean of its other
    ratings (leave one out), as Pearson, Spearman and Kendall over items.

    The held-out rating is drawn at random per item and the correlations are
    averaged over `draws` draws. With exchangeable rater positions this is the
    average over positions; unlike a by-position average it stays right when
    the ratings of an item are stored sorted (as OpenMEVA's are) or when a
    position is not one person across items (HANNA). Items need two ratings.
    """
    cnt = np.isfinite(R).sum(1)
    ok = cnt >= 2
    n = int(ok.sum())
    if n < 3:
        return math.nan, math.nan, math.nan, n
    Rk = R[ok]
    res = []
    for _ in range(draws):
        # pick the j-th finite rating of each item, j uniform over its ratings
        j = np.floor(rng.random(n) * cnt[ok]).astype(int)
        order = np.argsort(~np.isfinite(Rk), axis=1, kind="stable")   # finite ratings first
        col = order[np.arange(n), j]
        held = Rk[np.arange(n), col]
        rest = Rk.copy()
        rest[np.arange(n), col] = np.nan
        res.append(corr3(held, np.nanmean(rest, axis=1)))
    a = np.nanmean(np.array(res, float), axis=0)
    return float(a[0]), float(a[1]), float(a[2]), n


def cmd_corr(args):
    texts, metrics, not_scored = text_frame(args)
    criteria = [c.strip() for c in args.criteria.split(",")] if args.criteria else None
    human = human_table(texts, args.human_key, criteria)
    sys_all = field(texts, args.system_key).map(lambda v: "" if v is None else str(v)).to_numpy()
    is_h = np.array([is_human_system(s, args.human_system) for s in sys_all])
    variants = [("all", "all stories", np.ones(len(texts), bool))]
    if is_h.any():
        variants.append(("no_human", "without human-written stories", ~is_h))
    ref = args.reference or metrics[0].name
    if ref not in {m.name for m in metrics}:
        raise SystemExit(f"--reference {ref!r} is not among the metrics")
    ref_m = next(m for m in metrics if m.name == ref)
    B, seed = args.boot, args.seed
    X = {m.name: m.sign * texts[m.name].to_numpy(float) for m in metrics}

    rep = Report("Agreement with human ratings")
    rep.p(f"{len(texts)} texts, ratings under `{args.human_key}`, criteria: "
          f"{', '.join(human)}. Item level: Pearson, Spearman and Kendall tau-b between the "
          f"oriented metric and the mean rating; percentile intervals from {B} resamples of the "
          f"items (seed {seed}). Δ Kendall is the item-level Kendall minus that of the reference "
          f"metric ({ref_m.label}) on the same items; its p-value is a two-sided paired bootstrap "
          f"test (centred: p = (1 + #|d* − d| >= |d|) / (B + 1)), Benjamini-Hochberg adjusted "
          f"across metrics within each criterion. System level: Kendall and Spearman between "
          f"per-system mean ratings and per-system mean metric scores, intervals from resampling "
          f"items within systems. Human ceiling: one rating of each item, drawn at random, "
          f"against the mean of its other ratings (leave one out), averaged over 20 draws; it "
          f"compares one rater with k − 1, so it is a conservative reference for a metric "
          f"compared with the mean of all k.")
    if not_scored:
        rep.note(f"{not_scored} texts have no score in any listed file")

    crits = list(human)
    for vkey, vname, vmask in variants:
        w_cache, kb_cache, ry_cache, ws_cache = {}, {}, {}, {}

        def resample(idx):
            # one matrix of counts per item set, shared by every metric and criterion
            d = digest(idx)
            if d not in w_cache:
                W = boot_counts(rng_for(seed, "corr", vkey, d), idx.size, B)
                w_cache[d] = W
                kb_cache[d] = KendallBoot(W)
            return d, w_cache[d], kb_cache[d]

        item, sysr, tau_boot = {}, {}, {}
        for m in metrics:
            x = X[m.name]
            rx_cache = {}
            for c in crits:
                y = human[c][0]
                idx = np.flatnonzero(vmask & np.isfinite(x) & np.isfinite(y))
                xs, ys = x[idx], y[idx]
                pe, sp, kt = corr3(xs, ys)
                row = {"variant": vkey, "criterion": c, "metric": m.name, "label": m.label,
                       "dir": m.dir, "n": int(idx.size), "pearson": pe, "spearman": sp,
                       "kendall": kt}
                if idx.size > 2:
                    d, W, kb = resample(idx)
                    row["pearson_lo"], row["pearson_hi"] = pct_ci(wpearson(W, xs, ys))
                    if d not in rx_cache:
                        Rx = weighted_midranks(xs, W)
                        rx_cache[d] = (Rx, rank_moments(W, Rx))
                    if (c, d) not in ry_cache:
                        Ry = weighted_midranks(ys, W)
                        ry_cache[c, d] = (Ry, rank_moments(W, Ry))
                    (Rx, vx), (Ry, vy) = rx_cache[d], ry_cache[c, d]
                    row["spearman_lo"], row["spearman_hi"] = pct_ci(rank_corr(W, Rx, Ry, vx, vy))
                    tb = kb.tau(xs, ys, m.name, "human:" + c)
                    row["kendall_lo"], row["kendall_hi"] = pct_ci(tb)
                    tau_boot[m.name, c] = (idx, tb)
                item[m.name, c] = row

                # system level: means per system, items resampled within systems
                srow = {"variant": vkey, "criterion": c, "metric": m.name, "label": m.label,
                        "dir": m.dir}
                ss = sys_all[idx]
                names = sorted(set(ss))
                if len(names) >= 3:
                    si = np.searchsorted(np.array(names, dtype=object), ss)
                    Sm = np.zeros((idx.size, len(names)))
                    Sm[np.arange(idx.size), si] = 1
                    cnt = Sm.sum(0)
                    rho, tau = sys_corr(Sm.T @ ys / cnt, Sm.T @ xs / cnt)
                    ds = digest(idx)
                    if ds not in ws_cache:
                        Ws = boot_counts_strata(rng_for(seed, "corr-sys", vkey, ds), ss, B)
                        ws_cache[ds] = (Ws, Ws @ Sm)
                    Ws, wc = ws_cache[ds]
                    bx, by = (Ws * xs) @ Sm / wc, (Ws * ys) @ Sm / wc
                    rb, tbs = rowwise_spearman(by, bx), rowwise_kendall(by, bx)
                    srow.update({"n_systems": len(names), "kendall": tau,
                                 "kendall_lo": pct_ci(tbs)[0], "kendall_hi": pct_ci(tbs)[1],
                                 "spearman": rho, "spearman_lo": pct_ci(rb)[0],
                                 "spearman_hi": pct_ci(rb)[1]})
                sysr[m.name, c] = srow
            del rx_cache

        rep.h(f"{vname.capitalize()} ({int(vmask.sum())} texts)", 2)
        ceilings = {}
        for c in crits:
            y, R = human[c]
            item_rows = [item[m.name, c] for m in metrics]
            sys_rows = [sysr[m.name, c] for m in metrics]

            # paired bootstrap test of Kendall against the reference, on common items
            pvals = []
            for row in item_rows:
                mname = row["metric"]
                row["reference"] = ref
                if mname == ref or (mname, c) not in tau_boot or (ref, c) not in tau_boot:
                    row["d_kendall"], row["p"] = (0.0 if mname == ref else math.nan), math.nan
                    pvals.append(math.nan)
                    continue
                im, tm = tau_boot[mname, c]
                ir, tr = tau_boot[ref, c]
                if np.array_equal(im, ir):
                    d0 = row["kendall"] - item[ref, c]["kendall"]
                    db = tm - tr
                else:
                    idx = np.intersect1d(im, ir)
                    if idx.size < 3:
                        row["d_kendall"], row["p"] = math.nan, math.nan
                        pvals.append(math.nan)
                        continue
                    _, _, kb = resample(idx)
                    xm, xr, ys = X[mname][idx], X[ref][idx], y[idx]
                    d0 = corr3(xm, ys)[2] - corr3(xr, ys)[2]
                    db = kb.tau(xm, ys, mname, "human:" + c) - kb.tau(xr, ys, ref, "human:" + c)
                db = db[np.isfinite(db)]
                p = (1 + np.sum(np.abs(db - d0) >= abs(d0) - 1e-12)) / (db.size + 1) if db.size else math.nan
                row["d_kendall"], row["p"] = d0, float(min(1.0, p))
                pvals.append(row["p"])
            for row, pb in zip(item_rows, bh_adjust(pvals)):
                row["p_bh"] = pb

            ceil = loo_ceiling(R[vmask], rng_for(seed, "ceiling", vkey, c))
            ceilings[c] = ceil
            rep.h(f"{c}, {vname}", 3)
            rep.p(f"Human ceiling (leave one out, {ceil[3]} items with two or more ratings): "
                  f"Pearson {f3(ceil[0])}, Spearman {f3(ceil[1])}, Kendall {f3(ceil[2])}.")
            g = lambda r, key: fmci(r.get(key, math.nan), r.get(key + "_lo", math.nan), r.get(key + "_hi", math.nan))
            rep.table(["metric", "n", "Pearson", "Spearman", "Kendall", f"Δ Kendall vs {ref_m.label}",
                       "p", "p (BH)"],
                      [[r["label"], r["n"], g(r, "pearson"), g(r, "spearman"), g(r, "kendall"),
                        f3(r["d_kendall"]), f3(r["p"]), f3(r["p_bh"])] for r in item_rows])
            rep.table(["metric", "systems", "system Kendall", "system Spearman"],
                      [[r["label"], r.get("n_systems", 0), g(r, "kendall"), g(r, "spearman")]
                       for r in sys_rows])
            rep.rows("item", item_rows)
            rep.rows("system", sys_rows)
            rep.rows("ceiling", [{"variant": vkey, "criterion": c, "n_items": ceil[3],
                                  "raters_max": int(R.shape[1]), "pearson": ceil[0],
                                  "spearman": ceil[1], "kendall": ceil[2]}])
        rep.h(f"Summary, {vname}", 3)
        rep.p("Item-level Kendall tau-b:")
        rep.table(["metric"] + crits,
                  [[m.label] + [f3(item[m.name, c]["kendall"]) for c in crits] for m in metrics]
                  + [["human ceiling (LOO)"] + [f3(ceilings[c][2]) for c in crits]])
        rep.p("System-level Kendall tau-b:")
        rep.table(["metric"] + crits,
                  [[m.label] + [f3(sysr[m.name, c].get("kendall", math.nan)) for c in crits]
                   for m in metrics])
    return rep


# ---------------------------------------------------------------------------
# length
# ---------------------------------------------------------------------------

def cmd_length(args):
    texts, metrics, not_scored = text_frame(args)
    body = field(texts, "text").map(lambda t: "" if t is None else str(t))
    lengths = {"bytes": body.map(lambda t: len(t.encode("utf-8"))).to_numpy(float),
               "words": body.map(lambda t: len(t.split())).to_numpy(float)}
    sets = [("pooled", np.ones(len(texts), bool))]
    if args.by:
        gv = field(texts, args.by).map(lambda v: "" if v is None else str(v)).to_numpy()
        sets = [(g, gv == g) for g in sorted(set(gv))] + sets
    rep = Report("Correlation with length")
    rep.p(f"{len(texts)} texts. Pearson and Spearman between the oriented score (lower-is-better "
          f"metrics negated, so a positive value means longer texts score as better) and length "
          f"in bytes (UTF-8) and words (whitespace tokens). Intervals: {args.boot} resamples of "
          f"the texts, seed {args.seed}.")
    if not_scored:
        rep.note(f"{not_scored} texts have no score in any listed file")
    rows = []
    for m in metrics:
        x = m.sign * texts[m.name].to_numpy(float)
        for g, mask in sets:
            idx = np.flatnonzero(mask & np.isfinite(x))
            W = boot_counts(rng_for(args.seed, "length", g, digest(idx)), idx.size, args.boot)
            row = {"metric": m.name, "label": m.label, "dir": m.dir, "group": g, "n": int(idx.size),
                   "missing": int((mask & ~np.isfinite(x)).sum())}
            for ln, L in lengths.items():
                pe, sp, _ = corr3(x[idx], L[idx])
                row[f"pearson_{ln}"], row[f"spearman_{ln}"] = pe, sp
                if idx.size > 2:
                    row[f"pearson_{ln}_lo"], row[f"pearson_{ln}_hi"] = pct_ci(wpearson(W, x[idx], L[idx]))
                    row[f"spearman_{ln}_lo"], row[f"spearman_{ln}_hi"] = pct_ci(wspearman(W, x[idx], L[idx]))
            rows.append(row)
    g = lambda r, key: fmci(r.get(key, math.nan), r.get(key + "_lo", math.nan), r.get(key + "_hi", math.nan))
    rep.table(["metric", "set", "n", "missing", "Pearson, bytes", "Spearman, bytes",
               "Pearson, words", "Spearman, words"],
              [[r["label"], r["group"], r["n"], r["missing"], g(r, "pearson_bytes"),
                g(r, "spearman_bytes"), g(r, "pearson_words"), g(r, "spearman_words")] for r in rows])
    rep.rows("length", rows)
    return rep


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

def cmd_selftest(args):
    rng = np.random.default_rng(7)
    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond), detail))

    # pairwise accuracy: ties one half, equals the AUROC
    check("auc ties count one half", auc([1, 1], [1, 1]) == 0.5 and auc([1, 2], [1]) == 0.75
          and auc([2], [1]) == 1.0 and auc([0], [1]) == 0.0)
    try:
        from sklearn.metrics import roc_auc_score
        u, l = rng.integers(0, 5, 200), rng.integers(0, 4, 150)
        ref = roc_auc_score(np.r_[np.ones(200), np.zeros(150)], np.r_[u, l])
        check("auc equals sklearn roc_auc_score (ties)", abs(auc(u, l) - ref) < 1e-12, f"{auc(u, l):.6f} {ref:.6f}")
    except ImportError:
        check("auc equals sklearn roc_auc_score (ties)", True, "sklearn missing, skipped")

    # weighted statistics equal the plain ones on the resample written out
    x = rng.integers(0, 6, 60).astype(float)
    y = x + rng.integers(0, 4, 60)
    W = boot_counts(np.random.default_rng(1), 60, 5)
    ok_r = ok_p = ok_s = ok_k = ok_a = True
    kb = KendallBoot(W, block=17)
    tb, pb, sb = kb.tau(x, y, "x", "y"), wpearson(W, x, y), wspearman(W, x, y)
    ab = auc_boot(x[:30], y[30:], W[:, :30], W[:, 30:])
    for b in range(5):
        rep_idx = np.repeat(np.arange(60), W[b].astype(int))
        ok_r &= np.allclose(weighted_midranks(x, W)[b][rep_idx], stats.rankdata(x[rep_idx]))
        ok_p &= abs(pb[b] - np.corrcoef(x[rep_idx], y[rep_idx])[0, 1]) < 1e-10
        ok_s &= abs(sb[b] - stats.spearmanr(x[rep_idx], y[rep_idx])[0]) < 1e-10
        ok_k &= abs(tb[b] - stats.kendalltau(x[rep_idx], y[rep_idx])[0]) < 1e-5
        iu = np.repeat(np.arange(30), W[b, :30].astype(int))
        il = np.repeat(np.arange(30, 60), W[b, 30:].astype(int))
        ok_a &= abs(ab[b] - auc(x[iu], y[il])) < 1e-12
    check("weighted midranks = rankdata of the resample", ok_r)
    check("bootstrap Pearson = Pearson of the resample", ok_p)
    check("bootstrap Spearman = Spearman of the resample", ok_s)
    check("bootstrap Kendall tau-b = scipy tau-b of the resample", ok_k)
    check("bootstrap accuracy = accuracy of the resample", ok_a)

    # small helpers with known answers
    lo, hi = wilson(50, 100)
    check("Wilson 50/100 = [0.404, 0.596]", abs(lo - 0.40383) < 1e-4 and abs(hi - 0.59617) < 1e-4, f"{lo:.5f} {hi:.5f}")
    lo, hi = wilson(0, 10)
    check("Wilson 0/10 = [0, 0.278]", lo == 0 and abs(hi - 0.27753) < 1e-4, f"{hi:.5f}")
    adj = bh_adjust([0.01, 0.04, 0.03, 0.005, np.nan])
    check("Benjamini-Hochberg", np.allclose(adj[:4], [0.02, 0.04, 0.04, 0.02]) and np.isnan(adj[4]), str(adj))
    vals = rng.normal(3, 2, 400)
    g = make_group("g", "g", vals, 1, 2000, 5)
    se = vals.std(ddof=1) / 20
    check("bootstrap interval of a mean has width 2 x 1.96 x SE", abs((g.hi - g.lo) / (2 * Z95 * se) - 1) < 0.1,
          f"{(g.hi - g.lo) / (2 * Z95 * se):.3f}")
    g2 = make_group("g", "g", vals, 1, 2000, 5)
    check("deterministic resamples", np.array_equal(g.boot, g2.boot))
    # the pass rule needs both: accuracy 0.98 but one outlier puts the lower mean on top
    up = make_group("u", "u", 1 + rng.normal(0, 0.01, 50), 1, 1000, 5)
    low = make_group("l", "l", np.r_[np.zeros(49), 1e6], 1, 1000, 5)
    pr_ = pair_row(up, low, 0.95)
    check("pass rule: accuracy >= 0.95 with intervals in the wrong order fails",
          abs(pr_["acc"] - 0.98) < 1e-12 and not pr_["disjoint"] and not pr_["pass"])

    # Bradley-Terry recovers known strengths
    true = np.array([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0])
    k = len(true)
    i = rng.integers(0, k, 40000)
    j = (i + rng.integers(1, k, 40000)) % k
    win_i = rng.random(40000) < 1 / (1 + np.exp(-(true[i] - true[j])))
    wins = np.zeros((k, k))
    np.add.at(wins, (np.where(win_i, i, j), np.where(win_i, j, i)), 1)
    t_nr, t_mm = fit_bt(wins, ridge=0.0), fit_bt_mm(wins)
    check("BT recovers known strengths (max error < 0.05)", np.max(np.abs(t_nr - true)) < 0.05,
          f"max error {np.max(np.abs(t_nr - true)):.4f}")
    check("BT Newton (no ridge) = Zermelo MM", np.max(np.abs(t_nr - t_mm)) < 1e-6, f"{np.max(np.abs(t_nr - t_mm)):.2e}")
    t_r = fit_bt(wins, ridge=0.01)
    check("BT ridge 0.01 moves the fit by < 1e-4", np.max(np.abs(t_r - t_mm)) < 1e-4, f"{np.max(np.abs(t_r - t_mm)):.2e}")
    unbeaten = np.array([[0, 5, 5], [0, 0, 3], [0, 2, 0]], float)
    t_u = fit_bt(unbeaten, ridge=0.01)
    check("BT ridge keeps an unbeaten system finite, mean 0", np.all(np.isfinite(t_u)) and abs(t_u.mean()) < 1e-8
          and t_u[0] > t_u[1] > t_u[2], str(np.round(t_u, 3)))

    # ladder on synthetic data where the answer is known
    tmp = tempfile.TemporaryDirectory()
    td = Path(tmp.name)
    rungs = ["r0", "r1", "r2", "r3", "r4", "r5"]
    texts, scores = [], []
    for r, name in enumerate(rungs):
        q = 5.0 - r
        for t in range(300):
            tid = f"{name}-{t}"
            texts.append({"id": tid, "set": "syn", "system": name, "text": "w " * (5 + t % 50),
                          "meta": {"bucket": "short" if t % 2 else "standard"}})
            good = q + rng.normal(0, 0.05)
            scores.append({"id": tid, "good": good if not (r == 0 and t < 10) else None,
                           "rand": rng.normal(), "const": 1.0, "neg": -good, "noisy": q + rng.normal(0, 2.0)})
    with open(td / "texts.jsonl", "w") as f:
        f.writelines(json.dumps(t) + "\n" for t in texts)
    with open(td / "scores_a.jsonl", "w") as f:
        f.writelines(json.dumps({k: v for k, v in s.items() if k in ("id", "good", "rand")}) + "\n" for s in scores)
    with open(td / "scores_b.jsonl", "w") as f:
        f.writelines(json.dumps({k: v for k, v in s.items() if k in ("id", "const", "neg", "noisy")}) + "\n" for s in scores)
    (td / "metrics.json").write_text(json.dumps({"metrics": [
        {"name": "good", "dir": "+", "label": "Good"}, {"name": "rand", "dir": "+"},
        {"name": "const", "dir": "+"}, {"name": "neg", "dir": "-", "label": "Neg (lower)"},
        {"name": "noisy", "dir": "+"}]}))
    sysarg = ",".join(rungs)
    base = ["--texts", str(td / "texts.jsonl"), "--scores", str(td / "scores_a.jsonl"), str(td / "scores_b.jsonl")]
    main(["ladder", *base, "--metrics-config", str(td / "metrics.json"),
          "--systems", sysarg, "--out", str(td / "ladder.md")], quiet=True)
    df = pd.read_csv(td / "ladder.csv")
    is_true = lambda col: col.astype(str) == "True"
    adjp = df[(df.table == "pairs") & is_true(df.adjacent)]
    summ = df[df.table == "summary"].set_index("metric")
    rr = df[df.table == "rungs"]
    check("ladder: quality + small noise passes all 5 pairs", summ.loc["good", "passed"] == 5)
    check("ladder: direction - on the negated metric passes all 5", summ.loc["neg", "passed"] == 5)
    check("ladder: quality metric has accuracy 1 on every adjacent pair",
          np.allclose(adjp[adjp.metric == "good"]["acc"], 1) and np.allclose(adjp[adjp.metric == "neg"]["acc"], 1))
    check("ladder: random metric has accuracy near 0.5 and fails",
          adjp[adjp.metric == "rand"]["acc"].between(0.4, 0.6).all() and summ.loc["rand", "passed"] == 0)
    check("ladder: constant metric has accuracy exactly 0.5 (ties)",
          np.allclose(df[(df.table == "pairs") & (df.metric == "const")]["acc"], 0.5))
    check("ladder: 10 missing scores counted and excluded",
          rr[(rr.metric == "good") & (rr.rung == "r0")][["n", "missing"]].values.tolist() == [[290, 10]])
    full = df[(df.table == "pairs") & (df.metric == "good")]
    check("ladder: full matrix has all 15 pairs at accuracy 1", len(full) == 15 and np.allclose(full["acc"], 1))
    check("ladder: noisy metric passes no pair (overlap)", summ.loc["noisy", "passed"] == 0)
    main(["ladder", *base, "--metric", "neg:+", "--systems", sysarg, "--csv", str(td / "flip.csv")], quiet=True)
    flip = pd.read_csv(td / "flip.csv")
    fa = flip[(flip.table == "pairs") & is_true(flip.adjacent)]["acc"]
    check("ladder: direction + on a lower-is-better metric gives accuracy 0", np.allclose(fa, 0.0))
    main(["ladder", *base, "--metrics-config", str(td / "metrics.json"), "--systems", sysarg,
          "--csv", str(td / "again.csv")], quiet=True)
    check("ladder: two runs give identical output", (td / "again.csv").read_text() == (td / "ladder.csv").read_text())
    md = (td / "ladder.md").read_text()
    check("ladder: Markdown has the summary table", "| metric | r0 > r1 |" in md and "5 of 5" in md)
    main(["ladder", *base, "--metric", "good:+", "--systems", sysarg, "--where", "meta.bucket=short",
          "--where", "system!=r5", "--systems", "r0,r1,r2,r3,r4", "--csv", str(td / "where.csv")], quiet=True)
    wh = pd.read_csv(td / "where.csv")
    wr = wh[wh.table == "rungs"]
    check("ladder: --where filters on dotted fields", len(wr) == 5 and (wr["n"] + wr["missing"] == 150).all())

    # set-level ladder
    with open(td / "sets.jsonl", "w") as f:
        for r, name in enumerate(rungs):
            for s in range(20):
                f.write(json.dumps({"group": name, "subset": s, "mauve": 1 - r / 6 + rng.normal(0, 0.01),
                                    "zipf": rng.normal()}) + "\n")
    main(["setladder", "--values", str(td / "sets.jsonl"), "--metric", "mauve:+", "--metric", "zipf:+",
          "--systems", sysarg, "--csv", str(td / "sets.csv")], quiet=True)
    sl = pd.read_csv(td / "sets.csv")
    ss = sl[sl.table == "summary"].set_index("metric")
    check("setladder: set metric = quality passes all 5, random none",
          ss.loc["mauve", "passed"] == 5 and ss.loc["zipf", "passed"] == 0)

    # dose-response
    dt, ds = [], []
    for f_ in ("swap", "typo", "repeat"):
        for s in (10, 25, 50, 75, 100):
            for t in range(60):
                tid = f"{f_}-{s}-{t}"
                dt.append({"id": tid, "system": f_, "text": "x", "meta": {"family": f_, "strength": s}})
                ds.append({"id": tid, "down": -s / 100 + rng.normal(0, 0.05), "up": s / 100 + rng.normal(0, 0.05)})
    for t in range(60):
        dt.append({"id": f"base-{t}", "system": "human", "text": "x", "meta": {"family": "none", "strength": 0}})
        ds.append({"id": f"base-{t}", "down": rng.normal(0, 0.05), "up": rng.normal(0, 0.05)})
    with open(td / "dose_t.jsonl", "w") as f:
        f.writelines(json.dumps(t) + "\n" for t in dt)
    with open(td / "dose_s.jsonl", "w") as f:
        f.writelines(json.dumps(s) + "\n" for s in ds)
    main(["dose", "--texts", str(td / "dose_t.jsonl"), "--scores", str(td / "dose_s.jsonl"),
          "--metric", "down:+", "--metric", "up:+", "--csv", str(td / "dose.csv")], quiet=True)
    dd = pd.read_csv(td / "dose.csv")
    d1 = dd[dd.table == "dose"]
    check("dose: falling metric has Spearman(means) -1 and every step right",
          np.allclose(d1[d1.metric == "down"]["spearman_means"], -1) and (d1[d1.metric == "down"]["share_right"] == 1).all())
    check("dose: rising metric has Spearman(means) +1 and no step right",
          np.allclose(d1[d1.metric == "up"]["spearman_means"], 1) and (d1[d1.metric == "up"]["share_right"] == 0).all())
    lv = dd[(dd.table == "dose_levels") & (dd.metric == "down") & (dd.strength == 0)]
    check("dose: shared strength-0 texts attached to each family", len(lv) == 3 and (lv["n"] == 60).all())
    check("dose: sample Spearman strongly negative", (d1[d1.metric == "down"]["spearman_samples"] < -0.9).all())

    # model ladder
    mt, ms = [], []
    for r in range(5):
        for t in range(400):
            tid = f"m{r}-{t}"
            mt.append({"id": tid, "system": f"m{r}", "text": "x"})
            ms.append({"id": tid, "q": -r + rng.normal(0, 0.3), "rnd": rng.normal(), "ppl": np.exp(r + rng.normal(0, 0.3))})
    with open(td / "m_t.jsonl", "w") as f:
        f.writelines(json.dumps(t) + "\n" for t in mt)
    with open(td / "m_s.jsonl", "w") as f:
        f.writelines(json.dumps(s) + "\n" for s in ms)
    main(["models", "--texts", str(td / "m_t.jsonl"), "--scores", str(td / "m_s.jsonl"), "--metric", "q:+",
          "--metric", "rnd:+", "--metric", "ppl:-", "--systems", "m0,m1,m2,m3,m4", "--csv", str(td / "m.csv")], quiet=True)
    mo = pd.read_csv(td / "m.csv")
    od = mo[mo.table == "order"].set_index("metric")
    check("models: quality metric has Spearman 1, Kendall 1, P(perfect) > 0.99",
          np.isclose(od.loc["q", "spearman"], 1) and np.isclose(od.loc["q", "kendall"], 1) and od.loc["q", "p_perfect"] > 0.99,
          f"{od.loc['q', 'spearman']} {od.loc['q', 'kendall']} {od.loc['q', 'p_perfect']}")
    check("models: lower-is-better perplexity gets Kendall 1", np.isclose(od.loc["ppl", "kendall"], 1))
    check("models: random metric has P(perfect) < 0.5", od.loc["rnd", "p_perfect"] < 0.5, f"{od.loc['rnd', 'p_perfect']}")

    # pairs: people choose by a latent text quality; BT on per-system strengths
    syst = ["human", "s1", "s2", "s3", "s4", "s5"]
    strength = {"human": 1.0, "s1": 0.6, "s2": 0.2, "s3": -0.2, "s4": -0.6, "s5": -1.0}
    pt, psc, pp = [], [], []
    for n in range(3000):
        sa_, sb_ = rng.choice(syst, 2, replace=False)
        qa, qb = strength[sa_] + rng.normal(0, 0.5), strength[sb_] + rng.normal(0, 0.5)
        for side_, s_, q_ in (("a", sa_, qa), ("b", sb_, qb)):
            pt.append({"id": f"p{n}{side_}", "system": s_, "text": "x"})
            psc.append({"id": f"p{n}{side_}", "latent": q_, "rnd": rng.normal(), "flat": 0.0, "sysq": strength[s_]})
        win_a = rng.random() < 1 / (1 + np.exp(-2 * (qa - qb)))
        sure = abs(qa - qb) > 0.5                    # "definitely" when the difference is clear
        code = ("2a" if sure else "1a") if win_a else ("2b" if sure else "1b")
        pp.append({"pair_id": n, "a": f"p{n}a", "b": f"p{n}b", "q1": code, "q2": code, "q3": code,
                   "worker": f"w{n % 40}", "te": 30.0 if n % 10 else 5.0})
    pp.append(dict(pp[1], pair_id="dup"))
    with open(td / "p_t.jsonl", "w") as f:
        f.writelines(json.dumps(t) + "\n" for t in pt)
    with open(td / "p_s.jsonl", "w") as f:
        f.writelines(json.dumps(s) + "\n" for s in psc)
    with open(td / "pairs.jsonl", "w") as f:
        f.writelines(json.dumps(p) + "\n" for p in pp)
    main(["pairs", "--texts", str(td / "p_t.jsonl"), "--scores", str(td / "p_s.jsonl"), "--pairs", str(td / "pairs.jsonl"),
          "--metric", "latent:+", "--metric", "rnd:+", "--metric", "flat:+", "--metric", "sysq:+",
          "--questions", "q3", "--boot", "200", "--csv", str(td / "p.csv")], quiet=True)
    pr = pd.read_csv(td / "p.csv")
    ag = pr[pr.table == "agreement"].set_index("metric")
    check("pairs: latent-quality metric agrees > 0.7, definite answers more", ag.loc["latent", "rate"] > 0.7
          and ag.loc["latent", "rate_definite"] > ag.loc["latent", "rate"],
          f"{ag.loc['latent', 'rate']:.3f} {ag.loc['latent', 'rate_definite']:.3f}")
    check("pairs: random metric agrees near 0.5", abs(ag.loc["rnd", "rate"] - 0.5) < 0.05)
    check("pairs: a metric that always ties is dropped", ag.loc["flat", "n"] == 0 and ag.loc["flat", "ties_dropped"] > 2000)
    check("pairs: time filter drops the 300 judgments under 25 s", ag.loc["latent", "n"] == 2701, f"{ag.loc['latent', 'n']}")
    bt = pr[pr.table == "bradley_terry"].set_index("system")
    est = np.array([bt.loc[s, "bt"] for s in syst])
    check("pairs: BT order of systems recovered", np.all(np.diff(est) < 0), str(np.round(est, 1)))
    sc = pr[pr.table == "system_corr"].set_index("metric")
    check("pairs: system-level Spearman 1 for the system-strength metric",
          np.isclose(sc.loc["sysq", "spearman_all"], 1) and np.isclose(sc.loc["sysq", "spearman_models"], 1))
    check("pairs: repeated pair found for worker agreement",
          (pr.table == "worker_agreement").any() and pr[pr.table == "worker_agreement"]["agreement"].iloc[0] == 1)

    # corr: three raters around a true quality
    ct, cs = [], []
    n_items, sig = 1200, 1.0
    truth = rng.normal(0, 1, n_items)
    for it in range(n_items):
        s_ = "Human" if it % 8 == 0 else f"sys{it % 8}"
        tq = truth[it] + (0.8 if s_ == "Human" else 0.3 * (it % 8))
        rates = (tq + rng.normal(0, sig, 3)).round(3).tolist()
        ct.append({"id": f"c{it}", "system": s_, "text": "y" * (10 + it % 300),
                   "meta": {"human": {"CH": rates, "RE": float(np.mean(rates))}}})
        cs.append({"id": f"c{it}", "good": tq + rng.normal(0, 0.3), "bad": rng.normal(),
                   "good_copy": None, "inv": -(tq + rng.normal(0, 0.3)),
                   "len": float(len("y" * (10 + it % 300)))})
        cs[-1]["good_copy"] = cs[-1]["good"]
    with open(td / "c_t.jsonl", "w") as f:
        f.writelines(json.dumps(t) + "\n" for t in ct)
    with open(td / "c_s.jsonl", "w") as f:
        f.writelines(json.dumps(s) + "\n" for s in cs)
    main(["corr", "--texts", str(td / "c_t.jsonl"), "--scores", str(td / "c_s.jsonl"), "--metric", "good:+",
          "--metric", "bad:+", "--metric", "good_copy:+", "--metric", "inv:-", "--reference", "good",
          "--boot", "500", "--csv", str(td / "c.csv")], quiet=True)
    cr = pd.read_csv(td / "c.csv")
    it_ = cr[(cr.table == "item") & (cr.variant == "all") & (cr.criterion == "CH")].set_index("metric")
    check("corr: good metric has high item-level Kendall, random near 0",
          it_.loc["good", "kendall"] > 0.4 and abs(it_.loc["bad", "kendall"]) < 0.06,
          f"{it_.loc['good', 'kendall']:.3f} {it_.loc['bad', 'kendall']:.3f}")
    check("corr: paired test, random metric worse than reference (p < 0.01 after BH)",
          it_.loc["bad", "p_bh"] < 0.01 and it_.loc["bad", "d_kendall"] < 0)
    check("corr: identical metric has Δ 0 and p = 1", it_.loc["good_copy", "d_kendall"] == 0 and it_.loc["good_copy", "p"] == 1)
    check("corr: direction - gives the same sign as +", it_.loc["inv", "kendall"] > 0.4)
    ce = cr[(cr.table == "ceiling") & (cr.variant == "all") & (cr.criterion == "CH")].iloc[0]
    var_t = np.var(truth + np.array([(0.8 if i % 8 == 0 else 0.3 * (i % 8)) for i in range(n_items)]))
    theory = var_t / np.sqrt((var_t + sig ** 2) * (var_t + sig ** 2 / 2))
    check("corr: leave-one-out ceiling matches theory", abs(ce["pearson"] - theory) < 0.05, f"{ce['pearson']:.3f} vs {theory:.3f}")
    Rr = np.array([t["meta"]["human"]["CH"] for t in ct], float)
    c_raw = loo_ceiling(Rr, np.random.default_rng(2))[0]
    c_sorted = loo_ceiling(np.sort(Rr, axis=1), np.random.default_rng(2))[0]
    Rm = Rr.copy()
    Rm[::5, 2] = np.nan          # some items with two ratings only
    c_miss = loo_ceiling(Rm, np.random.default_rng(2))[0]
    check("corr: ceiling unchanged by sorted ratings or a missing rating", abs(c_raw - c_sorted) < 0.02
          and abs(c_raw - c_miss) < 0.03, f"{c_raw:.3f} {c_sorted:.3f} {c_miss:.3f}")
    nh = cr[(cr.table == "item") & (cr.variant == "no_human") & (cr.criterion == "CH")].set_index("metric")
    check("corr: without human-written stories drops them", nh.loc["good", "n"] == n_items - n_items // 8)
    check("corr: scalar ratings work (no ceiling)",
          cr[(cr.table == "ceiling") & (cr.criterion == "RE")]["pearson"].isna().all()
          and cr[(cr.table == "item") & (cr.criterion == "RE")]["kendall"].notna().any())
    sy = cr[(cr.table == "system") & (cr.variant == "all") & (cr.criterion == "CH")].set_index("metric")
    check("corr: system-level Kendall high for the good metric", sy.loc["good", "kendall"] > 0.7, f"{sy.loc['good', 'kendall']:.3f}")

    # length
    main(["length", "--texts", str(td / "c_t.jsonl"), "--scores", str(td / "c_s.jsonl"), "--metric", "len:+",
          "--metric", "bad:+", "--by", "system", "--csv", str(td / "l.csv")], quiet=True)
    le = pd.read_csv(td / "l.csv")
    pooled = le[(le.table == "length") & (le.group == "pooled")].set_index("metric")
    check("length: a metric equal to bytes has Pearson 1", abs(pooled.loc["len", "pearson_bytes"] - 1) < 1e-9)
    check("length: random metric near 0 with an interval covering 0",
          pooled.loc["bad", "pearson_bytes_lo"] < 0 < pooled.loc["bad", "pearson_bytes_hi"])
    tmp.cleanup()

    width = max(len(n) for n, _, _ in results)
    failed = 0
    for name, ok, detail in results:
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'}  {name.ljust(width)}  {detail}")
    print(f"{len(results) - failed} of {len(results)} checks passed")
    return failed


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(prog="meta.py", description="SCRIBE meta-evaluation statistics.",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, texts=True, metrics=True):
        if texts:
            sp.add_argument("--texts", nargs="+", action="extend", required=True, help="text JSONL files")
            sp.add_argument("--scores", nargs="+", action="extend", required=True, help="score JSONL files, joined on id")
        if metrics:
            sp.add_argument("--metric", action="append", help="NAME:+ or NAME:- (optionally NAME:+:Label); repeatable")
            sp.add_argument("--metrics-config", help="JSON list of metrics with directions and labels")
        sp.add_argument("--where", action="append", help="filter KEY=VALUE or KEY!=VALUE (dotted keys, a|b alternatives)")
        sp.add_argument("--boot", type=int, default=DEFAULT_BOOT, help="bootstrap resamples (default 1000)")
        sp.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"seed (default {DEFAULT_SEED})")
        sp.add_argument("--out", help="Markdown output file (default stdout)")
        sp.add_argument("--digits", type=int, default=3, help="decimals in the Markdown tables (default 3)")
        sp.add_argument("--csv", help="CSV output (a JSON file of the same name is written too); default next to --out")

    def order(sp, key_default):
        sp.add_argument("--systems", help="ordered systems, best first: a,b,c (or a;b;c), each name or name=Label")
        sp.add_argument("--systems-file", help="ordered systems, one per line: name or name<TAB>label")
        sp.add_argument("--group-key", default=key_default, help=f"field naming the system (default {key_default})")

    sp = sub.add_parser("ladder", help="control ladder with the PLAN 6.4 pass rule")
    common(sp)
    order(sp, "system")
    sp.add_argument("--pass-acc", type=float, default=0.95, help="accuracy needed to pass (default 0.95)")

    sp = sub.add_parser("setladder", help="control ladder for set-level metrics")
    sp.add_argument("--values", nargs="+", action="extend", required=True, help='JSONL {"group","subset",<metric>...}')
    common(sp, texts=False)
    order(sp, "group")
    sp.add_argument("--subset-key", default="subset")
    sp.add_argument("--pass-acc", type=float, default=0.95)

    sp = sub.add_parser("dose", help="dose-response of corruption families")
    common(sp)
    sp.add_argument("--family-key", default="meta.family")
    sp.add_argument("--strength-key", default="meta.strength")
    sp.add_argument("--families", help="families to report, comma separated (default: all)")
    sp.add_argument("--base-family", default="none|base|", help="family names of shared strength-0 texts")
    sp.add_argument("--rho-threshold", type=float, default=-0.9, help="Spearman on the means counted as a hit")

    sp = sub.add_parser("models", help="system means against a known order")
    common(sp)
    order(sp, "system")

    sp = sub.add_parser("pairs", help="MAUVE human evaluation (pairwise judgments)")
    common(sp)
    sp.add_argument("--pairs", required=True, help='JSONL {"pair_id","a","b","q1","q2","q3","worker","te"?}; '
                    '--where filters these judgments')
    sp.add_argument("--questions", default="q1,q2,q3")
    sp.add_argument("--system-key", default="system")
    sp.add_argument("--human-system", default="human", help="name of the human system (case-insensitive; a|b)")
    sp.add_argument("--time-key", default="te", help="field with the estimated answer time in seconds")
    sp.add_argument("--min-time", type=float, default=25.0,
                    help="drop judgments with time <= this (MAUVE paper: 25; 0 disables; only if the field exists)")
    sp.add_argument("--worker-key", default="worker")
    sp.add_argument("--bt-ridge", type=float, default=0.01, help="ridge of the Bradley-Terry fit (default 0.01)")

    sp = sub.add_parser("corr", help="item- and system-level correlation with human ratings")
    common(sp)
    sp.add_argument("--human-key", default="meta.human", help="field holding {criterion: rating(s)}")
    sp.add_argument("--criteria", help="criteria, comma separated (default: all found)")
    sp.add_argument("--reference", help="reference metric of the paired test (default: the first metric)")
    sp.add_argument("--system-key", default="system")
    sp.add_argument("--human-system", default="human", help="system of human-written texts (case-insensitive; a|b)")

    sp = sub.add_parser("length", help="correlation of each metric with length")
    common(sp)
    sp.add_argument("--by", help="also report each group of this field (e.g. meta.bucket)")

    sub.add_parser("selftest", help="synthetic checks with known answers")
    return p


COMMANDS = {"ladder": cmd_ladder, "setladder": cmd_setladder, "dose": cmd_dose, "models": cmd_models,
            "pairs": cmd_pairs, "corr": cmd_corr, "length": cmd_length}


def main(argv=None, quiet=False):
    global DIGITS
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if args.cmd == "selftest":
        return 1 if cmd_selftest(args) else 0
    DIGITS = args.digits
    if args.boot < 1:
        raise SystemExit("--boot must be at least 1")
    rep = COMMANDS[args.cmd](args)
    write_outputs(rep, args, argv, quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
