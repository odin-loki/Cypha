#!/usr/bin/env python3
"""opt_boolean_tables.py: the Boolean decisions O3 found in Cypha's hot loops (GENERATION_PLAN §13.41 O3), as truth
tables with don't-cares, minimised by bool_min.py. Run through `python3 -I scripts/bool_min.py cypha [--md]`.

Each entry: where it is, how often it runs, its inputs (input i = bit i of the row), its outputs (an integer per row;
each output bit is minimised alone), the rows that cannot occur (don't-cares), and the code's current form.
Integer-valued inputs (lengths, counts) are given as their bits; thresholds on real numbers as one input per
threshold, with the impossible orders (x >= 0.8 but not x >= 0.5) as don't-cares.

Author: Odin Loch
"""

CTRL_SPACE = {9, 10, 11, 12, 13, 32}  # C locale isspace


def byte_class(b):  # full_mixer.hpp fx::byte_class, content_mixer.hpp OnlineFinal::byte_class (C locale)
    if b in CTRL_SPACE:
        return 0
    if 65 <= b <= 90 or 97 <= b <= 122:
        return 1
    if 48 <= b <= 57:
        return 2
    if b != 0 and chr(b) in "[]{}<>|&=":
        return 3
    return 4


def copy_bucket(n):
    return 0 if n == 0 else 1 if n < 4 else 2 if n < 6 else 3 if n < 8 else 4 if n < 12 else 5 if n < 16 else 6 if n < 32 else 7


def len_bucket(n):
    return 0 if n < 4 else 1 if n < 8 else 2 if n < 16 else 3 if n < 32 else 4 if n < 64 else 5 if n < 128 else 6


def bits(m, i):
    return (m >> i) & 1


def sim_bucket(m):
    # inputs: T1..T4 (top >= .5, .65, .8, .9), A1..A3 (agree >= .5, .8, .99), L1 (live >= 1), L2 (live >= 2),
    # L8 (live >= 8), W (word > 0)
    T = [bits(m, i) for i in range(4)]
    A = [bits(m, 4 + i) for i in range(3)]
    L1, L2, L8, W = bits(m, 7), bits(m, 8), bits(m, 9), bits(m, 10)
    if not L1:
        return 0
    sb, ab, lb = sum(T), sum(A), L2 + L8
    return 1 + ((sb * 4 + ab) * 3 + lb) * 2 + W


def sim_valid(m):
    T = [bits(m, i) for i in range(4)]
    A = [bits(m, 4 + i) for i in range(3)]
    L = [bits(m, 7), bits(m, 8), bits(m, 9)]
    mono = lambda v: all(v[i] >= v[i + 1] for i in range(len(v) - 1))
    return mono(T) and mono(A) and mono(L)


def count_bucket5(m):
    # t's low 4 bits and H = any of bits 4..63 (count_bucket reads t only through these)
    t, H = m & 15, bits(m, 4)
    if H:
        return 3
    return 0 if t <= 1 else 1 if t <= 3 else 2 if t <= 15 else 3


def cls_bucket(m):
    # ClassContext::bucket: order (0..3, two bits) * 4 + count bucket (the 5 inputs above)
    return (m >> 5) * 4 + count_bucket5(m & 31)


def cst(m):
    # FullMixer's copy state per bit: H (copy predicts), M (the partial byte agrees with copy's byte), E (copy's bit)
    H, M, E = bits(m, 0), bits(m, 1), bits(m, 2)
    return 0 if not H else (2 + E if M else 1)


def fact_pref(m):
    # planner make_plan: F1 = f.wd == 1, B1 = b.wd == 1, EQ = f.wd == b.wd, FT = f.text, BT = b.text
    F1, B1, EQ, FT, BT = (bits(m, i) for i in range(5))
    return int((F1 and not B1) or (EQ and FT and not BT))


def fact_valid(m):
    F1, B1, EQ = bits(m, 0), bits(m, 1), bits(m, 2)
    if F1 and B1:
        return EQ == 1
    if F1 != B1:
        return EQ == 0
    return True


def sent_end(m):
    # K5's gate: t == "." || t == "!" || t == "?" || t == "\n"; inputs t[0]'s 8 bits and S1 = (t.size() == 1)
    return int(bits(m, 8) == 1 and (m & 255) in (ord("."), ord("!"), ord("?"), 10))


MARKUP = set(b"[]{}|=<>&*#")


def markup(m):  # K5's gate and the planner's write-on: string_view("[]{}|=<>&*#").find(c) != npos
    return int(m in MARKUP)


def bitpos(m):  # OnlineNgram::dist: while ((c0 >> (j + 1)) != 0) ++j, c0 in 1..255
    return max(m.bit_length() - 1, 0)


def mlen(m):  # MetaMixer::byte_begin: lg = floor(log2(ig_n + 1)) capped at 15, ig_n in 0..8192
    lg, n = 0, m + 1
    while n > 1 and lg < 15:
        n >>= 1
        lg += 1
    return lg


def eq8(m):  # a == b over two bytes (copy_pred's scan, OnlineNgram::claim's slot check)
    return int((m & 255) == (m >> 8))


ENTRIES = [
    # name, where, owner, runs, n, nout, f, valid, current form, implementation
    ("byte_class", "online/full_mixer.hpp:121 fx::byte_class; online/content_mixer.hpp:332", "M2 (owned)",
     "4 per byte", 8, 3, byte_class, None,
     "isspace, isalpha, isdigit, strchr: up to 4 library calls and 4 tests", "256-byte table (patch)"),
    ("copy_bucket", "lm/copy_expert.hpp:35", "free", "3-5 per byte", 7, 3, copy_bucket,
     lambda n: n == 0 or 3 <= n <= 64,
     "a chain of 7 compares", "66-entry table with a clamp (done)"),
    ("len_bucket", "tools/lm/cyphalm_trie.cpp:877 (InfiniGram bucket)", "context8k driver (owned)",
     "10 per byte", 14, 3, len_bucket, lambda n: n <= 8192,
     "a chain of 6 compares", "n < 4 ? 0 : min(6, bit_width(n) - 2) (patch)"),
    ("count_bucket", "tools/lm/cyphalm_trie.cpp:878", "context8k driver (owned)", "10 per byte", 5, 2, count_bucket5, None,
     "a chain of 3 compares on a 64-bit count", "(t > 1) + (t > 3) + (t > 15) (patch)"),
    ("sim_bucket", "lm/similar_context.hpp:618", "free", "2 per byte", 11, 7, sim_bucket, sim_valid,
     "three chains of 4, 3 and 2 compares, then arithmetic", "sums of compare results (done)"),
    ("cls_bucket", "lm/class_context.hpp:135", "free", "3 per byte", 7, 4, cls_bucket, None,
     "a chain of 3 compares, then order * 4 + it", "order * 4 + (t > 1) + (t > 3) + (t > 15) (done)"),
    ("copy_state", "online/full_mixer.hpp:600 (cst)", "M2 (owned)", "8 per byte", 3, 2, cst, None,
     "two nested ifs", "unchanged: already minimal, and the branch skips a log"),
    ("bit_position", "online/online_ngram.hpp:162", "O5 (owned)", "255 per byte", 8, 3, bitpos, lambda c: c >= 1,
     "a loop of up to 8 shifts and tests", "bit_width(c0) - 1 (patch)"),
    ("match_log2", "online/meta_mixer.hpp:265 (mlen_)", "M2 (owned)", "1 per byte", 14, 4, mlen, lambda n: n <= 8192,
     "a loop of up to 15 shifts and tests", "min(15, bit_width(ig_n + 1) - 1) (patch)"),
    ("byte_equal", "lm/copy_expert.hpp:25 (copy_pred's window scan); online/online_ngram.hpp:230 (claim)", "free / O5",
     "16,384 per byte x 3 calls (copy); 6 per bit (claim)", 16, 1, eq8, None,
     "one compare and branch per window position",
     "copy: SSE2 compares into a 64-bit mask, walked from the top bit, positions that cannot win skipped (done); claim: masks (patch)"),
    ("sentence_end", "lm/word_kn.hpp:1318, 1373, 1469 (K5's gate)", "M1 (owned)", "1 per word", 9, 1, sent_end, None,
     "4 string compares", "size 1 and a 256-bit set (patch)"),
    ("markup", "lm/word_kn.hpp:1321, 1472; lm/planner/writer.hpp:655", "M1 / free", "1 per written byte per draft", 8, 1,
     markup, None, "string_view::find over 11 bytes", "256-bit set (done in writer.hpp; patch in word_kn.hpp)"),
    ("fact_preference", "lm/planner/writer.hpp:514 (make_plan)", "free", "per fact of the main entity, once per prompt",
     5, 1, fact_pref, fact_valid, "(F1 && !B1) || (EQ && FT && !BT)", "unchanged: already minimal"),
]


def cand_order(m):
    # similar_context.hpp's top-k order: G = (dot x > dot y), E = (dot x == dot y), L = (slot x < slot y)
    G, E, L = bits(m, 0), bits(m, 1), bits(m, 2)
    return int(G or (E and L))


ENTRIES.append(("candidate_order", "lm/similar_context.hpp:901 (search's partial_sort)", "free",
                "about 2,300 candidates x log 32 per byte", 3, 1, cand_order, lambda m: not (bits(m, 0) and bits(m, 1)),
                "x.first > y.first || (x.first == y.first && x.second < y.second), inside a heap",
                "one 64-bit compare of packed keys, in a sorted buffer (done)"))

# Functions of a 32-bit count, given as ranges per output bit: [lo, hi) where the bit is 1. Minimised by Espresso
# from cubes (aligned blocks of the ranges) and checked at the ranges' edges and at random counts.
WIDE = [
    ("used_count_choice", "lm/packed_trie.hpp:114 (used_count, Mode::Synth)", "free", "per follower of 3 nodes per training byte",
     32, [[(65, 1 << 32)], [(2, 65)]],
     "if (c == 0) 0; if (c < 2) 0; if (c <= 64) c; else the ladder (outputs: bit 1 ladder, bit 0 c itself)",
     "both values made, one kept by a mask; the ladder by counted compares (done)"),
]


def run_wide(bm, rows):
    import random
    rng = random.Random(1)
    for name, where, owner, runs, n, ranges, current, impl in WIDE:
        sp = bm.Space(n)
        per, tot_p, tot_l = [], 0, 0
        for rs in ranges:
            F = []
            for lo, hi in rs:
                F += bm.range_cubes(sp, lo, hi)
            cov = bm.espresso(sp, F)
            inside = lambda x: any(lo <= x < hi for lo, hi in rs)
            edges = [v for lo, hi in rs for v in (lo - 1, lo, hi - 1, hi)] + [rng.getrandbits(n) for _ in range(20000)]
            for x in edges:
                if 0 <= x < (1 << n):
                    m = sp.minterm(x)
                    assert any((c & m) == m for c in cov) == inside(x), name
            p, l = bm.cost(sp, cov)
            per.append("%d/%d" % (p, l))
            tot_p += p
            tot_l += l
        rows.append((name, where, owner, runs, n, "2^%d" % n, 0, "-", " ".join(per), "%d/%d" % (tot_p, tot_l), "espresso",
                     current, impl))


def run(bm, md=False):
    rows = []
    for name, where, owner, runs, n, nout, f, valid, current, impl in ENTRIES:
        sp = bm.Space(n)
        tot_p = tot_l = canon_p = canon_l = 0
        on_n = dc_n = 0
        per = []
        meth = set()
        for o in range(nout):
            on, dc = bm.table(n, (lambda o: lambda m: (f(m) >> o) & 1)(o), valid)
            on_n += len(on)
            dc_n = len(dc)
            canon_p += len(on)
            canon_l += len(on) * n
            if n <= 16 and len(on) + len(dc) <= 4096:
                cov, info = bm.exact(n, on, dc)
                meth.add(info["method"] if info["method"] != "essential" else "QM")
            else:
                # dense and wide: Espresso from cubes (runs of equal value), never from minterms
                F = bm.scc(sp, [sp.minterm(m) for m in on]) if len(on) <= 512 else _runs(bm, sp, on)
                D = _runs(bm, sp, dc) if dc else []
                cov = bm.espresso(sp, F, D)
                meth.add("espresso")
            assert bm.check(sp, on, dc, cov), name
            p, l = bm.cost(sp, cov)
            per.append("%d/%d" % (p, l))
            tot_p += p
            tot_l += l
        rows.append((name, where, owner, runs, n, 1 << n, dc_n, "%d/%d" % (canon_p, canon_l), " ".join(per),
                     "%d/%d" % (tot_p, tot_l), "+".join(sorted(meth)), current, impl))
    run_wide(bm, rows)
    if md:
        print("| Function | Where | Owner | Runs | Inputs | Rows | Don't-care rows | Canonical SOP (products/literals) | "
              "Minimal SOP per output bit | Minimal SOP total | Method | Current code | O3 form |")
        print("|---|---|---|---|---:|---:|---:|---:|---|---:|---|---|---|")
        for r in rows:
            print("| " + " | ".join(str(x) for x in r) + " |")
    else:
        for r in rows:
            print("\t".join(str(x) for x in r))
    return 0


def _runs(bm, sp, ms):
    """Cubes for a set of minterms given as runs of consecutive integers (aligned blocks)."""
    ms = sorted(ms)
    out = []
    i = 0
    while i < len(ms):
        j = i
        while j + 1 < len(ms) and ms[j + 1] == ms[j] + 1:
            j += 1
        out += bm.range_cubes(sp, ms[i], ms[j] + 1)
        i = j + 1
    return out
