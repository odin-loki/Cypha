#!/usr/bin/env python3
"""bool_min.py: two-level Boolean minimisation (sum of products) with don't-cares, for GENERATION_PLAN §13.41 O3.

Written for Cypha from the textbook methods; no third-party code. Run with `python3 -I`.

  python3 -I scripts/bool_min.py selftest
  python3 -I scripts/bool_min.py pla FILE.pla [--espresso]     # Berkeley PLA (.i .o .ilb .ob .p, type fd)
  python3 -I scripts/bool_min.py cypha [--md]                  # the O3 tables (docs/reports/OPT_BOOLEAN.md)

Methods:
- Exact (up to about 16 inputs): Quine-McCluskey prime implicants over ON and don't-care minterms, essential primes,
  row and column dominance, then Petrick's method on a small cyclic core or an exact branch-and-bound set cover on a
  larger one. The cost is (products, literals), compared in that order.
- Heuristic (any size): Espresso's loop on cubes. The OFF-set is the complement of ON + DC (unate recursive
  paradigm); then EXPAND (raise literals while the cube misses the OFF-set, greedily covering most other cubes),
  IRREDUNDANT (drop cubes the rest and the don't-cares cover, by a tautology check of the cofactor), REDUCE (shrink
  each cube to the supercube of what only it covers), repeated while the cost falls.
Every result is checked against the function (all care minterms up to 20 inputs; by containment and tautology above).

Cubes are positional: two bits per variable (bit 2i: the variable may be 0; bit 2i+1: it may be 1). 11 is a free
variable, 01 the literal x', 10 the literal x, 00 an empty cube. Variable 0 is the least significant input bit.

Author: Odin Loch
"""
import itertools
import random
import sys

# ------------------------------------------------------------------------------------------------ cubes


class Space:
    """n Boolean inputs in positional cube notation."""

    def __init__(self, n):
        self.n = n
        self.full = (1 << (2 * n)) - 1
        self.m01 = int("01" * n, 2) if n else 0  # bit 2i set for each variable

    def lit(self, i, v):
        """The cube of one literal: x_i if v else x_i'."""
        return self.full & ~(3 << (2 * i)) | ((2 if v else 1) << (2 * i))

    def minterm(self, m):
        c = 0
        for i in range(self.n):
            c |= (2 if (m >> i) & 1 else 1) << (2 * i)
        return c

    def empty(self, c):
        lo, hi = c & self.m01, (c >> 1) & self.m01
        return (lo | hi) != self.m01

    def nlits(self, c):
        lo, hi = c & self.m01, (c >> 1) & self.m01
        return self.n - bin(lo & hi).count("1")

    def free(self, c):
        lo, hi = c & self.m01, (c >> 1) & self.m01
        return bin(lo & hi).count("1")

    def field(self, c, i):
        return (c >> (2 * i)) & 3

    def cofactor(self, c, p):
        """c's cofactor by cube p (None when they do not meet)."""
        x = c & p
        if self.empty(x):
            return None
        return c | (self.full ^ p)

    def to_str(self, c, names=None):
        out = []
        for i in range(self.n):
            f = self.field(c, i)
            if f == 3:
                continue
            nm = names[i] if names else "x%d" % i
            out.append(nm if f == 2 else nm + "'")
        return " ".join(out) if out else "1"

    def to_c(self, c, names):
        out = []
        for i in range(self.n):
            f = self.field(c, i)
            if f == 3:
                continue
            out.append(names[i] if f == 2 else "!" + names[i])
        return " && ".join(out) if out else "1"

    def pla_row(self, c):
        """PLA input part, variable n-1 first (the PLA's leftmost column is its first input)."""
        return "".join("-10"[0 if self.field(c, i) == 3 else (1 if self.field(c, i) == 2 else 2)] for i in reversed(range(self.n)))

    def minterms(self, c):
        """Every minterm of cube c (as integers)."""
        fixed, free = 0, []
        for i in range(self.n):
            f = self.field(c, i)
            if f == 3:
                free.append(i)
            elif f == 2:
                fixed |= 1 << i
            elif f == 0:
                return []
        out = []
        for bits in range(1 << len(free)):
            m = fixed
            for k, i in enumerate(free):
                if (bits >> k) & 1:
                    m |= 1 << i
            out.append(m)
        return out


def scc(sp, cover):
    """Single-cube containment: drop cubes inside another cube of the cover (and empty and repeated ones)."""
    cs = sorted(set(c for c in cover if not sp.empty(c)), key=lambda c: -sp.free(c))
    out = []
    for c in cs:
        if not any((d & c) == c for d in out):
            out.append(c)
    return out


def cost(sp, cover):
    return (len(cover), sum(sp.nlits(c) for c in cover))


# ------------------------------------------------------------------------------------------------ unate recursive paradigm


def _binate_var(sp, cover):
    """The most binate variable (most cubes with a literal, both polarities present first); None if all free."""
    best, bkey = None, None
    for i in range(sp.n):
        z = o = 0
        for c in cover:
            f = sp.field(c, i)
            if f == 1:
                z += 1
            elif f == 2:
                o += 1
        if z + o == 0:
            continue
        key = (1 if z and o else 0, z + o, -abs(z - o))
        if bkey is None or key > bkey:
            best, bkey = i, key
    return best, (bkey[0] == 1 if bkey else False)


def tautology(sp, cover):
    """Is the cover the whole space?"""
    if not cover:
        return False
    if any(c == sp.full for c in cover):
        return True
    vol = sum(1 << sp.free(c) for c in cover)
    if vol < (1 << sp.n):
        return False
    v, binate = _binate_var(sp, cover)
    if v is None:
        return True
    if not binate:
        # unate in every variable that splits... check all variables: a unate cover is a tautology only with the full cube
        if all(not (any(sp.field(c, i) == 1 for c in cover) and any(sp.field(c, i) == 2 for c in cover)) for i in range(sp.n)):
            return False
    for val in (0, 1):
        p = sp.lit(v, val)
        sub = [x for x in (sp.cofactor(c, p) for c in cover) if x is not None]
        if not tautology(sp, sub):
            return False
    return True


def complement(sp, cover):
    """A cover of the complement."""
    cover = scc(sp, cover)
    if not cover:
        return [sp.full]
    if any(c == sp.full for c in cover):
        return []
    if len(cover) == 1:
        c = cover[0]
        out = []
        for i in range(sp.n):
            f = sp.field(c, i)
            if f in (1, 2):
                out.append(sp.full & ~(3 << (2 * i)) | ((3 ^ f) << (2 * i)))
        return out
    v, _ = _binate_var(sp, cover)
    res = []
    parts = []
    for val in (0, 1):
        p = sp.lit(v, val)
        sub = [x for x in (sp.cofactor(c, p) for c in cover) if x is not None]
        parts.append([x & p for x in complement(sp, sub)])
    # merge: a cube in both halves (but for variable v) loses v
    mask = 3 << (2 * v)
    h1 = {c | mask: c for c in parts[1]}
    used1 = set()
    for c in parts[0]:
        k = c | mask
        if k in h1:
            res.append(k)
            used1.add(k)
        else:
            res.append(c)
    for c in parts[1]:
        if (c | mask) not in used1:
            res.append(c)
    return scc(sp, res)


def covers_cube(sp, cover, c):
    """Is cube c inside the union of cover?"""
    sub = [x for x in (sp.cofactor(d, c) for d in cover) if x is not None]
    return tautology(sp, sub)


# ------------------------------------------------------------------------------------------------ Espresso


def expand(sp, F, R):
    F = scc(sp, F)
    order = sorted(F, key=lambda c: (-sp.free(c), c))
    done = []
    covered = set()
    for c in order:
        if c in covered:
            continue
        cur = c
        while True:
            best, bkey = None, None
            for i in range(sp.n):
                f = sp.field(cur, i)
                if f == 3:
                    continue
                cand = cur | (3 << (2 * i))
                if any(not sp.empty(cand & r) for r in R):
                    continue
                # prefer the raise that swallows most other cubes, then the one that meets most of them
                key = (sum(1 for d in F if d != c and (cand & d) == d), sum(1 for d in F if not sp.empty(cand & d)), -i)
                if bkey is None or key > bkey:
                    best, bkey = cand, key
            if best is None:
                break
            cur = best
        for d in F:
            if (cur & d) == d:
                covered.add(d)
        done.append(cur)
    return scc(sp, done)


def irredundant(sp, F, D):
    F = list(F)
    # relatively essential cubes stay; the rest are dropped one at a time, smallest first, while still covered
    ess, red = [], []
    for k, c in enumerate(F):
        rest = F[:k] + F[k + 1:] + list(D)
        (red if covers_cube(sp, rest, c) else ess).append(c)
    keep = ess + red
    for c in sorted(red, key=lambda c: (sp.free(c), c)):
        rest = [d for d in keep if d != c] + list(D)
        if covers_cube(sp, rest, c):
            keep = [d for d in keep if d != c]
    return keep


def reduce_(sp, F, D):
    F = sorted(F, key=lambda c: (-sp.free(c), c))
    out = list(F)
    for k in range(len(out)):
        c = out[k]
        if c is None:
            continue
        rest = [d for j, d in enumerate(out) if j != k and d is not None] + list(D)
        sub = [x for x in (sp.cofactor(d, c) for d in rest) if x is not None]
        comp = complement(sp, sub)
        if not comp:
            out[k] = None  # redundant
            continue
        sup = 0
        for x in comp:
            sup |= x
        out[k] = c & sup
    return [c for c in out if c is not None]


def espresso(sp, F, D=(), R=None, max_iter=50):
    """Espresso's loop: F the ON-set cover, D the don't-care cover; returns a cover of F inside F + D."""
    F = scc(sp, F)
    D = scc(sp, D)
    if R is None:
        R = complement(sp, F + D)
    if not R:
        return [sp.full]
    F = expand(sp, F, R)
    F = irredundant(sp, F, D)
    best = list(F)
    for _ in range(max_iter):
        c0 = cost(sp, F)
        F = reduce_(sp, F, D)
        F = expand(sp, F, R)
        F = irredundant(sp, F, D)
        if cost(sp, F) < cost(sp, best):
            best = list(F)
        if cost(sp, F) >= c0:
            break
    return best


# ------------------------------------------------------------------------------------------------ exact: Quine-McCluskey


def primes_qm(n, on, dc):
    """Prime implicants as (value, dash mask) pairs over ON + DC minterms."""
    cur = set((m, 0) for m in set(on) | set(dc))
    primes = set()
    while cur:
        nxt = set()
        used = set()
        for v, mk in cur:
            for b in range(n):
                bit = 1 << b
                if mk & bit or v & bit:
                    continue
                o = (v | bit, mk)
                if o in cur:
                    nxt.add((v, mk | bit))
                    used.add((v, mk))
                    used.add(o)
        primes |= cur - used
        cur = nxt
    return sorted(primes)


def qm_cube(sp, v, mk):
    c = 0
    for i in range(sp.n):
        if (mk >> i) & 1:
            c |= 3 << (2 * i)
        else:
            c |= (2 if (v >> i) & 1 else 1) << (2 * i)
    return c


def _petrick(rows, cols_of, lits, limit=200000):
    """Petrick's method: expand the product of sums of each row's columns; all minimal (count, lits) column sets."""
    terms = {frozenset()}
    for r in rows:
        nt = set()
        for t in terms:
            if t & cols_of[r]:
                nt.add(t)
            else:
                for c in cols_of[r]:
                    nt.add(t | {c})
        # absorption: drop supersets
        srt = sorted(nt, key=len)
        keep = []
        for t in srt:
            if not any(k <= t for k in keep):
                keep.append(t)
        terms = set(keep)
        if len(terms) > limit:
            return None
    best = min((len(t), sum(lits[c] for c in t)) for t in terms)
    return [t for t in terms if (len(t), sum(lits[c] for c in t)) == best], best


def _bnb(rows, cols_of, rows_of, lits):
    """Exact minimum (count, lits) set cover by branch and bound."""
    best = [None, None]
    rows = list(rows)

    def lower_bound(unc):
        # a maximal set of rows with pairwise disjoint column sets: each needs its own column
        lb, taken = 0, set()
        for r in sorted(unc, key=lambda r: len(cols_of[r])):
            if not (cols_of[r] & taken):
                lb += 1
                taken |= cols_of[r]
        return lb

    def rec(unc, chosen, nl):
        if not unc:
            key = (len(chosen), nl)
            if best[0] is None or key < best[0]:
                best[0], best[1] = key, list(chosen)
            return
        if best[0] is not None and len(chosen) + lower_bound(unc) > best[0][0]:
            return
        r = min(unc, key=lambda r: (len(cols_of[r]), r))
        for c in sorted(cols_of[r], key=lambda c: (-len(rows_of[c] & unc), lits[c], c)):
            chosen.append(c)
            rec(unc - rows_of[c], chosen, nl + lits[c])
            chosen.pop()

    rec(frozenset(rows), [], 0)
    return best[1], best[0]


def exact(n, on, dc=(), petrick_rows=24, want_all=False):
    """Minimum sum of products of the function (ON minterms, DC minterms) by Quine-McCluskey + Petrick / set cover.
    Returns (cubes, info) with info: primes, essential, core rows, method."""
    sp = Space(n)
    on = sorted(set(on) - set())
    dcs = set(dc) - set(on)
    if not on:
        return [], {"primes": 0, "essential": 0, "core": 0, "method": "zero"}
    pr = primes_qm(n, on, dcs)
    cubes = [qm_cube(sp, v, mk) for v, mk in pr]
    lits = [sp.nlits(c) for c in cubes]
    onset = set(on)
    rows_of = []
    for v, mk in pr:
        ms = set()
        free = [b for b in range(n) if (mk >> b) & 1]
        for bits in range(1 << len(free)):
            m = v
            for k, b in enumerate(free):
                if (bits >> k) & 1:
                    m |= 1 << b
            if m in onset:
                ms.add(m)
        rows_of.append(frozenset(ms))
    cols_of = {r: set() for r in on}
    for c, ms in enumerate(rows_of):
        for r in ms:
            cols_of[r].add(c)
    chosen = set()
    unc = set(on)
    cols = set(c for c in range(len(cubes)) if rows_of[c])
    ness = 0
    # reduce: essential columns, row dominance, column dominance, until nothing changes
    changed = True
    while changed and unc:
        changed = False
        for r in sorted(unc):
            live = cols_of[r] & cols
            if len(live) == 1:
                c = next(iter(live))
                if c not in chosen:
                    chosen.add(c)
                    ness += 1
                unc -= rows_of[c]
                changed = True
                break
        if changed:
            continue
        # row dominance: a row whose live columns include another row's can go (covering the other covers it)
        lr = {r: frozenset(cols_of[r] & cols) for r in unc}
        rl = sorted(unc, key=lambda r: (len(lr[r]), r))
        for i, r in enumerate(rl):
            for s in rl[:i]:
                if s in unc and lr[s] <= lr[r] and r in unc:
                    unc.discard(r)
                    changed = True
                    break
        if changed:
            continue
        # column dominance: a column whose rows (among the uncovered) lie inside another's at no smaller cost goes
        cl = {c: frozenset(rows_of[c] & unc) for c in cols}
        for c in sorted(cols, key=lambda c: (len(cl[c]), -lits[c], c)):
            for d in cols:
                if d != c and cl[c] <= cl[d] and lits[d] <= lits[c] and (cl[c] < cl[d] or lits[d] < lits[c] or d < c):
                    cols.discard(c)
                    changed = True
                    break
            if changed:
                break
    method = "essential"
    if unc:
        crow = sorted(unc)
        ccols = {r: frozenset(cols_of[r] & cols) for r in crow}
        crows_of = {c: frozenset(rows_of[c] & unc) for c in cols}
        res = None
        if len(crow) <= petrick_rows:
            res = _petrick(crow, ccols, lits)
            method = "petrick"
        if res is None:
            pick, _ = _bnb(crow, ccols, crows_of, lits)
            method = "branch-and-bound"
            chosen |= set(pick)
        else:
            sols, _ = res
            pick = min(sols, key=lambda t: sorted(t))
            chosen |= set(pick)
    out = [cubes[c] for c in sorted(chosen)]
    return out, {"primes": len(cubes), "essential": ness, "core": len(unc), "method": method}


# ------------------------------------------------------------------------------------------------ checks and front end


def check(sp, on, dc, cover):
    """The cover is 1 on every ON minterm and 0 on every care minterm outside ON (enumerated; n <= 20)."""
    onset, dcset = set(on), set(dc)
    val = [False] * (1 << sp.n)
    for c in cover:
        for m in sp.minterms(c):
            val[m] = True
    for m in range(1 << sp.n):
        if m in dcset and m not in onset:
            continue
        if val[m] != (m in onset):
            return False
    return True


def minimise(n, on, dc=(), method="auto"):
    """The minimal SOP of a single-output function; method auto: exact up to 16 inputs, Espresso above."""
    sp = Space(n)
    on, dc = sorted(set(on)), sorted(set(dc) - set(on))
    if method == "auto":
        method = "exact" if n <= 16 else "espresso"
    if method == "exact":
        cov, info = exact(n, on, dc)
    else:
        F = [sp.minterm(m) for m in on]
        D = [sp.minterm(m) for m in dc]
        cov = espresso(sp, F, D)
        info = {"method": "espresso"}
    if n <= 20:
        assert check(sp, on, dc, cov), "minimised cover differs from the function"
    return cov, info


def table(n, f, valid=None):
    """ON and DC minterms of f over n input bits (input i = bit i of the minterm); valid(m) False: a don't-care."""
    on, dc = [], []
    for m in range(1 << n):
        if valid is not None and not valid(m):
            dc.append(m)
        elif f(m):
            on.append(m)
    return on, dc


def range_cubes(sp, lo, hi, nbits=None):
    """Cubes over the low nbits inputs (the rest free) covering the integers lo <= x < hi: aligned power-of-two blocks."""
    nbits = sp.n if nbits is None else nbits
    out = []
    x = lo
    while x < hi:
        k = 0
        while k < nbits and (x & ((1 << (k + 1)) - 1)) == 0 and x + (1 << (k + 1)) <= hi:
            k += 1
        c = sp.full
        for i in range(k, nbits):
            c &= sp.lit(i, (x >> i) & 1)
        out.append(c)
        x += 1 << k
    return out


def multi(n, nout, f, valid=None, method="auto"):
    """Each output bit of f (an integer function of the minterm) minimised alone. Returns the covers and a summary."""
    covs = []
    for o in range(nout):
        on, dc = table(n, lambda m: (f(m) >> o) & 1, valid)
        cov, info = minimise(n, on, dc, method)
        covs.append((cov, info, len(on), len(dc)))
    return covs


# ------------------------------------------------------------------------------------------------ PLA files


def read_pla(path):
    n = nout = None
    rows = []
    names, onames = None, None
    for ln in open(path):
        ln = ln.split("#")[0].strip()
        if not ln:
            continue
        if ln.startswith("."):
            k = ln.split()
            if k[0] == ".i":
                n = int(k[1])
            elif k[0] == ".o":
                nout = int(k[1])
            elif k[0] == ".ilb":
                names = k[1:]
            elif k[0] == ".ob":
                onames = k[1:]
            elif k[0] == ".e":
                break
            continue
        a, b = ln.split()
        rows.append((a, b))
    return n, nout, rows, names, onames


def pla_cube(sp, s):
    c = 0
    for k, ch in enumerate(s):
        i = sp.n - 1 - k
        c |= {"0": 1, "1": 2, "-": 3}[ch] << (2 * i)
    return c


def run_pla(path, force_espresso=False):
    n, nout, rows, names, onames = read_pla(path)
    sp = Space(n)
    print(".i %d\n.o %d" % (n, nout))
    if names:
        print(".ilb " + " ".join(names))
    out_rows = []
    for o in range(nout):
        F = [pla_cube(sp, a) for a, b in rows if b[o] == "1"]
        D = [pla_cube(sp, a) for a, b in rows if b[o] == "-"]
        if n <= 16 and not force_espresso:
            on = set()
            dc = set()
            for c in F:
                on |= set(sp.minterms(c))
            for c in D:
                dc |= set(sp.minterms(c))
            cov, _ = exact(n, sorted(on), sorted(dc - on))
        else:
            cov = espresso(sp, F, D)
        for c in cov:
            out_rows.append((sp.pla_row(c), "".join("1" if k == o else "0" for k in range(nout))))
    print(".p %d" % len(out_rows))
    for a, b in out_rows:
        print(a + " " + b)
    print(".e")


# ------------------------------------------------------------------------------------------------ selftest


def brute_min(n, on, dc):
    """The minimum (count, lits) over all prime covers, by trying every set of primes in size order (tiny n only)."""
    sp = Space(n)
    if not on:
        return (0, 0)
    pr = [qm_cube(sp, v, mk) for v, mk in primes_qm(n, on, dc)]
    onset = set(on)
    msets = [set(sp.minterms(c)) & onset for c in pr]
    for k in range(1, len(pr) + 1):
        best = None
        for comb in itertools.combinations(range(len(pr)), k):
            got = set()
            for c in comb:
                got |= msets[c]
            if got >= onset:
                key = (k, sum(sp.nlits(pr[c]) for c in comb))
                if best is None or key < best:
                    best = key
        if best is not None:
            return best
    return None


def selftest():
    fails = 0

    def ok(cond, what):
        nonlocal fails
        if not cond:
            fails += 1
            print("FAIL", what)

    # 1. Known functions.
    sp3 = Space(3)
    cov, info = exact(3, [0, 1, 2, 5, 6, 7])  # the cyclic function: no essential prime, two minimal covers
    ok(cost(sp3, cov) == (3, 6) and info["essential"] == 0 and info["method"] == "petrick", "cyclic f(0,1,2,5,6,7) %s %s" % (cost(sp3, cov), info))
    cov, _ = exact(3, [m for m in range(8) if bin(m).count("1") >= 2])  # majority: ab + bc + ac
    ok(cost(sp3, cov) == (3, 6), "majority")
    for n in range(2, 7):  # parity: no two minterms merge
        cov, _ = exact(n, [m for m in range(1 << n) if bin(m).count("1") % 2])
        ok(cost(Space(n), cov) == (1 << (n - 1), n << (n - 1)), "parity %d" % n)
    # BCD to seven segments, segment a, with 10-15 don't-care: A + C + BD + B'D' (A the MSB)
    on = [0, 2, 3, 5, 6, 7, 8, 9]
    cov, _ = exact(4, on, range(10, 16))
    ok(cost(Space(4), cov) == (4, 6), "7-seg a %s" % (cost(Space(4), cov),))
    # textbook: f = sum m(0,1,2,5,6,7,8,9,10,14): B'C' + CD' + A'BD = 3 products, 7 literals (B'D' is a prime but
    # redundant: B'C' and CD' cover its minterms)
    cov, _ = exact(4, [0, 1, 2, 5, 6, 7, 8, 9, 10, 14])
    ok(cost(Space(4), cov) == (3, 7) == brute_min(4, [0, 1, 2, 5, 6, 7, 8, 9, 10, 14], []), "textbook 4-var %s" % (cost(Space(4), cov),))
    # constant and empty functions
    ok(exact(4, list(range(16)))[0] == [Space(4).full], "tautology")
    ok(exact(4, [])[0] == [], "zero")
    ok(exact(4, [3], range(16))[0] == [Space(4).full], "all don't-care")

    # 2. Random functions: the exact method equals brute force; Espresso's covers are valid and never smaller.
    rng = random.Random(20261105)
    worse = 0
    trials = 0
    for t in range(300):
        n = rng.choice([2, 3, 4, 5])
        ms = list(range(1 << n))
        on = [m for m in ms if rng.random() < 0.4]
        dc = [m for m in ms if m not in on and rng.random() < 0.2]
        sp = Space(n)
        cov, _ = exact(n, on, dc)
        ok(check(sp, on, dc, cov), "exact valid %d %s %s" % (n, on, dc))
        if n <= 4:
            ok(cost(sp, cov) == brute_min(n, on, dc), "exact = brute %d %s %s: %s %s" % (n, on, dc, cost(sp, cov), brute_min(n, on, dc)))
        e = espresso(sp, [sp.minterm(m) for m in on], [sp.minterm(m) for m in dc]) if on else []
        ok(check(sp, on, dc, e), "espresso valid %d %s %s" % (n, on, dc))
        ok(cost(sp, e)[0] >= cost(sp, cov)[0], "espresso below exact?")
        worse += cost(sp, e)[0] > cost(sp, cov)[0]
        trials += 1
    print("selftest: espresso above the exact minimum on %d of %d random functions (2-5 inputs)" % (worse, trials))
    # Petrick and branch-and-bound agree on cyclic cores
    for t in range(60):
        n = rng.choice([4, 5, 6])
        on = [m for m in range(1 << n) if rng.random() < 0.5]
        if not on:
            continue
        a, _ = exact(n, on, petrick_rows=10 ** 9)
        b, _ = exact(n, on, petrick_rows=0)
        ok(cost(Space(n), a) == cost(Space(n), b), "petrick = bnb %d %s" % (n, on))

    # 3. Complement and tautology.
    for t in range(100):
        n = rng.choice([3, 4, 5, 6])
        sp = Space(n)
        F = []
        for _ in range(rng.randint(0, 6)):
            c = sp.full
            for i in range(n):
                r = rng.random()
                if r < 0.3:
                    c &= sp.lit(i, 0)
                elif r < 0.6:
                    c &= sp.lit(i, 1)
            F.append(c)
        R = complement(sp, F)
        fm = set()
        for c in F:
            fm |= set(sp.minterms(c))
        rm = set()
        for c in R:
            rm |= set(sp.minterms(c))
        ok(fm.isdisjoint(rm) and (fm | rm) == set(range(1 << n)), "complement %d" % n)
        ok(tautology(sp, F + R), "tautology of F + F'")
        ok(tautology(sp, F) == (len(fm) == 1 << n), "tautology %d" % n)

    # 4. Espresso on wide functions (above the exact range): sums of products with known minimal forms.
    for n, k in [(20, 5), (24, 6), (32, 4)]:
        sp = Space(n)
        rng2 = random.Random(n)
        vs = list(range(n))
        rng2.shuffle(vs)
        F = []
        for j in range(k):  # disjoint supports: each product is prime, and none can be dropped or merged
            c = sp.full
            for i in vs[j * 3:(j + 1) * 3]:
                c &= sp.lit(i, rng2.random() < 0.5)
            F.append(c)
        # hand Espresso the products split into smaller cubes (each on one more variable, both ways)
        G = []
        for j, c in enumerate(F):
            x = vs[-1 - j]
            G += [c & sp.lit(x, 0), c & sp.lit(x, 1)]
        e = espresso(sp, G)
        ok(sorted(e) == sorted(F), "espresso %d inputs recovers %d products (got %d)" % (n, k, len(e)))
    # 18-input function with don't-cares: x0 x1 + x2' x3 where every minterm with x17 set is a don't-care
    sp = Space(18)
    F = [sp.lit(0, 1) & sp.lit(1, 1) & sp.lit(17, 0), sp.lit(2, 0) & sp.lit(3, 1) & sp.lit(17, 0)]
    D = [sp.lit(17, 1)]
    e = espresso(sp, F, D)
    ok(cost(sp, e) == (2, 4), "espresso with don't-cares %s" % (cost(sp, e),))

    print("selftest:", "pass" if fails == 0 else "%d failures" % fails)
    return 1 if fails else 0


# ------------------------------------------------------------------------------------------------ main


def main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        return 2
    if argv[1] == "selftest":
        return selftest()
    if argv[1] == "pla":
        run_pla(argv[2], "--espresso" in argv)
        return 0
    if argv[1] == "cypha":
        return cypha_tables("--md" in argv)
    print("unknown command " + argv[1])
    return 2


def cypha_tables(md):
    """The O3 tables are defined in opt_boolean_tables (kept beside this file)."""
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opt_boolean_tables.py")
    spec = importlib.util.spec_from_file_location("opt_boolean_tables", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run(sys.modules[__name__], md)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
