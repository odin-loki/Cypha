#!/usr/bin/env python3
"""DK2 (docs/reports/DK2_DONT_KNOW_TRAINING.md): the don't-know message and tag in cyphalm_trie qa rows.

For each set (dev, test) and chat model (trie, trie_ig):
- echo: answers with a control token in them, any "<|" but the don't-know tag's "<|unknown " (frames: "<|entity=" or
  "<|about ");
- refusals (grade 1) and tagged ones; agree: the message's entity ("nothing about X in what") equals the tag's
  (entity="Y"), normalised (accents folded, punctuation and quotes dropped, a leading article dropped), as a share of
  all refusals (the DK2 rule's R3) and of tagged ones;
- whether the message and the tag name a span of the question (at least 5 bytes, word-bounded).
Grading only: it reads rows files and the T-QA question files; it answers nothing.

usage: dk2_measures.py ROWS.tsv DEV_TSV TEST_TSV [LABEL]
"""
import math, re, sys, unicodedata

ECHO = re.compile(r"<\|(?!unknown )")
FRAME = re.compile(r"<\|(entity=|about )")

def unprintable(a):  # undo cyphalm_trie's printable(): "\\n" and "\\xNN" back to bytes, then UTF-8
    b = re.sub(rb"\\x([0-9a-f]{2})", lambda m: bytes([int(m.group(1), 16)]), a.encode("utf-8"))
    return b.replace(b"\\n", b"\n").decode("utf-8", "replace")

def norm(s):
    s = unicodedata.normalize("NFKD", s).lower()
    s = "".join(c for c in s if c.isalnum() or c == " ")
    return " ".join(s.split())

def strip_article(s):
    for a in ("the ", "a ", "an ", "The ", "A ", "An "):
        if s.startswith(a):
            return s[len(a):]
    return s

def msg_entity(a):  # as cyphalm_trie's msg_entity (A1d)
    p = a.find("nothing about ")
    if p < 0:
        return ""
    e = a.find(" in what", p + 14)
    return "" if e < 0 else a[p + 14:e]

def tag_entity(a):
    t = a.find("<|unknown")
    if t < 0:
        return ""
    p = a.find('entity="', t)
    if p < 0:
        return ""
    e = a.find('"', p + 8)
    return a[p + 8:] if e < 0 else a[p + 8:e]

def names_in(name, q):  # as cyphalm_trie's names_in: normalised, at least 5 bytes, a span of the question
    n = norm(strip_article(name))
    return len(n) >= 5 and (" " + n + " ") in (" " + norm(q) + " ")

def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)

def load_q(path):
    qs = []
    with open(path, encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        iq = head.index("question") if "question" in head else 0
        for line in f:
            c = line.rstrip("\n").split("\t")
            qs.append(c[iq] if iq < len(c) else "")
    return qs

def main():
    rows, devp, testp = sys.argv[1], sys.argv[2], sys.argv[3]
    label = sys.argv[4] if len(sys.argv) > 4 else rows
    qs = {"dev": load_q(devp), "test": load_q(testp)}
    st = {}
    with open(rows, encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        ix = {h: i for i, h in enumerate(head)}
        for line in f:
            c = line.rstrip("\n").split("\t")
            if c[ix["form"]] != "chat":
                continue
            s, lm = c[ix["set"]], c[ix["lm"]]
            a = unprintable(c[ix["answer"]]) if ix["answer"] < len(c) else ""
            q = qs[s][int(c[ix["row"]])] if s in qs and int(c[ix["row"]]) < len(qs[s]) else ""
            d = st.setdefault((s, lm), dict(n=0, echo=0, frame=0, refused=0, ragree=0, tagged=0, msg=0, agree=0, msg_q=0,
                                            tag_q=0, both_q=0))
            d["n"] += 1
            if ECHO.search(a):
                d["echo"] += 1
            if FRAME.search(a):
                d["frame"] += 1
            m, t = msg_entity(a), tag_entity(a)
            same = bool(m and t and norm(strip_article(m)) == norm(strip_article(t)))
            if c[ix["grade"]] == "1":
                d["refused"] += 1
                d["ragree"] += same
            if "<|unknown" in a:
                d["tagged"] += 1
                if m:
                    d["msg"] += 1
                d["agree"] += same
                if m and names_in(m, q):
                    d["msg_q"] += 1
                if t and names_in(t, q):
                    d["tag_q"] += 1
                if m and t and names_in(m, q) and names_in(t, q) and norm(strip_article(m)) == norm(strip_article(t)):
                    d["both_q"] += 1
    for (s, lm), d in sorted(st.items()):
        lo, hi = wilson(d["echo"], d["n"])
        alo, ahi = wilson(d["agree"], d["tagged"])
        rlo, rhi = wilson(d["ragree"], d["refused"])
        print(f"DK2 {label} {s} chat/{lm} answers {d['n']} echo {d['echo']} ({d['echo']/max(1,d['n']):.3f} [{lo:.3f}, {hi:.3f}])"
              f" frame_echo {d['frame']} refused {d['refused']}"
              f" agree_of_refused {d['ragree']} ({d['ragree']/max(1,d['refused']):.3f} [{rlo:.3f}, {rhi:.3f}])"
              f" tagged {d['tagged']} message_has_entity {d['msg']}"
              f" message_eq_tag {d['agree']} ({d['agree']/max(1,d['tagged']):.3f} [{alo:.3f}, {ahi:.3f}])"
              f" message_in_question {d['msg_q']} tag_in_question {d['tag_q']} both_same_and_in_question {d['both_q']}")

if __name__ == "__main__":
    main()
