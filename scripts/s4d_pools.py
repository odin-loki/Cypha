#!/usr/bin/env python3
"""S4d's pools (docs/reports/S4_UNSEEN_AND_FRONTIER.md, "S4d"), written as dump pages for pattern_learn --signals.

Every pool is written in the form of s4b_dolly_pages.py's Dolly pages: <page>, a title, and the text in <text>, escaped
as the dump escapes text (& < > "). Each subcommand writes a dev file and a test file (or one file).

  oasst READY.messages.jsonl.gz DEV.xml TEST.xml
      The own-voice pool O: OpenAssistant oasst1 (2023-04-12, "ready" messages, Apache-2.0). One assistant reply per
      conversation tree: among the English replies to the tree's root prompt that are not deleted, passed review, are
      not synthetic and keep at least 256 bytes after rendering, the one with the lowest rank (no rank last), then the
      earliest, then the smallest message id. Markdown is rendered to plain lines (render_md below). A tree goes to dev
      when the first byte of SHA-256(message_tree_id) is even, else to test. Title "OASST <first 8 hex of the tree id>".
  dolly DOLLY.jsonl DEV.xml TEST.xml
      The own-voice Dolly pool Y: brainstorming and creative-writing responses of at least 256 bytes that S4b's pool D
      did not take (D took every 10th qualifying response). Dev: even line numbers; test: odd. Title "Dolly NNNNN".
  plain2 ENWIK8 OUT.xml
      The fresh control P2: S4c's renderer (s4c_plain_pages.py) on the article pages from byte 50,000,000 that S4c's P
      did not take and that S4d's fresh stream F does not read: qualifying index q (0-based) with q % 4 == 2. P took
      q % 4 == 0; F reads q % 4 in {1, 3}.
  ns ENWIK8 DEV.xml TEST.xml
      The new-page-type pool L2: the dump's namespace pages (a title with a colon followed by a non-space: Wikipedia,
      Template, Image, Category, Portal, Help ...) whose body starts at or after byte 9,437,184, not redirects. No S4
      signal and no reader has scored them (pattern_learn's page list skips namespaces). The body is copied unchanged;
      the title gets a space after its first colon so that the page list keeps it. Dev: even index in dump order; test:
      odd.

Data is untrusted: it is only parsed as JSON or bytes and written escaped. Run with python3 -I.
"""
import gzip
import hashlib
import html
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this repository's scripts (s4c's renderer)

CALIB_FROM = 9_437_184
PLAIN_FROM = 50_000_000


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def page(title: str, body_escaped: str, user: str) -> str:
    return ("  <page>\n    <title>%s</title>\n    <id>0</id>\n    <revision>\n      <id>0</id>\n"
            "      <timestamp>2006-03-01T00:00:00Z</timestamp>\n      <contributor>\n        <username>%s</username>\n"
            "        <id>0</id>\n      </contributor>\n      <text xml:space=\"preserve\">%s</text>\n    </revision>\n"
            "  </page>\n" % (esc(title), user, body_escaped))


def render_md(text: str) -> str:
    """Markdown to plain lines: fenced code removed; headings, tables and images dropped; links to their words; URLs,
    inline-code ticks, emphasis markers and HTML tags removed; "*" and "+" list items written as "- item"."""
    out, fence = [], False
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        s = line.strip()
        if s.startswith("```") or s.startswith("~~~"):
            fence = not fence
            continue
        if fence or not s:
            continue
        if re.match(r"^#{1,6}(\s|$)", s) or s.startswith("|"):
            continue
        s = re.sub(r"^(>\s*)+", "", s)
        s = re.sub(r"^[*+]\s+", "- ", s)
        s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)
        s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
        s = re.sub(r"(?:https?|ftp)://\S+", "", s)
        s = re.sub(r"</?[a-zA-Z][^>]*>", "", s)
        s = s.replace("`", "")
        s = re.sub(r"(\*\*|__)(.+?)\1", r"\2", s)
        s = re.sub(r"(?<![\w*])\*(?!\s)([^*]+?)(?<!\s)\*(?![\w*])", r"\1", s)
        s = s.strip()
        if s and s != "-":
            out.append(s)
    return "\n".join(out)


def oasst(src: str, dev_out: str, test_out: str) -> int:
    msgs = [json.loads(line) for line in gzip.open(src, "rt", encoding="utf-8")]
    roots = {m["message_id"] for m in msgs if m.get("parent_id") is None}
    best = {}
    for m in msgs:
        if m.get("role") != "assistant" or m.get("parent_id") not in roots or m.get("lang") != "en":
            continue
        if m.get("deleted") or not m.get("review_result") or m.get("synthetic"):
            continue
        body = render_md(m.get("text") or "")
        if len(body.encode("utf-8")) < 256:
            continue
        rank = m.get("rank")
        key = (rank if rank is not None else 1_000_000, m.get("created_date") or "", m["message_id"])
        t = m["message_tree_id"]
        if t not in best or key < best[t][0]:
            best[t] = (key, body)
    n = [0, 0]
    with open(dev_out, "w", encoding="utf-8", newline="\n") as fd, open(test_out, "w", encoding="utf-8", newline="\n") as ft:
        for f in (fd, ft):
            f.write("<mediawiki>\n")
        for t in sorted(best):
            side = hashlib.sha256(t.encode()).digest()[0] % 2  # 0 dev, 1 test
            (fd, ft)[side].write(page("OASST " + t.replace("-", "")[:8], esc(best[t][1]), "oasst"))
            n[side] += 1
        for f in (fd, ft):
            f.write("</mediawiki>\n")
    print("oasst trees with a reply %d: dev %d test %d (messages %d, roots %d)" % (len(best), n[0], n[1], len(msgs), len(roots)))
    return 0


def dolly(src: str, dev_out: str, test_out: str) -> int:
    n = [0, 0]
    qualifying = 0
    with open(src, encoding="utf-8") as f, open(dev_out, "w", encoding="utf-8", newline="\n") as fd, \
            open(test_out, "w", encoding="utf-8", newline="\n") as ft:
        for o in (fd, ft):
            o.write("<mediawiki>\n")
        for line_no, line in enumerate(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            resp = rec.get("response", "")
            if len(resp.encode("utf-8")) < 256:
                continue
            qualifying += 1
            if (qualifying - 1) % 10 == 0:  # S4b's D
                continue
            if rec.get("category") not in ("brainstorming", "creative_writing"):
                continue
            side = line_no % 2  # 0 dev, 1 test
            (fd, ft)[side].write(page("Dolly %05d" % line_no, esc(resp), "dolly"))
            n[side] += 1
        for o in (fd, ft):
            o.write("</mediawiki>\n")
    print("dolly own-voice pages: dev %d test %d (of %d qualifying responses; D's every 10th left out)" % (n[0], n[1], qualifying))
    return 0


def pages_of(data: bytes):
    """(title, body start, body end, body bytes) of every page with a text, in dump order."""
    pos = 0
    while True:
        pb = data.find(b"<page>", pos)
        if pb < 0:
            return
        pe = data.find(b"</page>", pb)
        if pe < 0:
            return
        pos = pe + 7
        tb, te = data.find(b"<title>", pb), data.find(b"</title>", pb)
        xb = data.find(b"<text", te)
        if tb < 0 or te < 0 or te > pe or xb < 0 or xb > pe:
            continue
        xg = data.find(b">", xb)
        if xg < 0 or xg > pe or data[xg - 1:xg] == b"/":
            continue
        xe = data.find(b"</text>", xg)
        if xe < 0 or xe > pe:
            continue
        yield html.unescape(data[tb + 7:te].decode("utf-8", "replace")), xg + 1, xe, data[xg + 1:xe]


def plain2(src: str, out_path: str) -> int:
    import s4c_plain_pages as s4c
    data = open(src, "rb").read()
    q = kept = 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as out:
        out.write("<mediawiki>\n")
        for title, b, _, raw in pages_of(data):
            if b < PLAIN_FROM:
                continue
            body = raw.decode("utf-8", "replace")
            if body[:40].strip().lower().startswith("#redirect") or s4c.namespace_title(title) or s4c.skipped_title(title):
                continue
            q += 1
            if (q - 1) % 4 != 2:
                continue
            kept += 1
            out.write(page(title, esc(s4c.render(body)), "plain"))
        out.write("</mediawiki>\n")
    print("plain2 pages %d of %d qualifying pages from byte %d (q %% 4 == 2)" % (kept, q, PLAIN_FROM))
    return 0


def ns(src: str, dev_out: str, test_out: str) -> int:
    data = open(src, "rb").read()
    n = [0, 0]
    by_ns = {}
    k = 0
    with open(dev_out, "w", encoding="utf-8", newline="\n") as fd, open(test_out, "w", encoding="utf-8", newline="\n") as ft:
        for o in (fd, ft):
            o.write("<mediawiki>\n")
        for title, b, _, raw in pages_of(data):
            c = title.find(":")
            if c < 0 or c + 1 >= len(title) or title[c + 1] == " " or b < CALIB_FROM:
                continue
            if raw[:40].strip().lower().startswith(b"#redirect"):
                continue
            side = k % 2
            k += 1
            (fd, ft)[side].write(page(title[:c + 1] + " " + title[c + 1:], raw.decode("utf-8", "replace"), "ns"))
            n[side] += 1
            by_ns[title[:c]] = by_ns.get(title[:c], 0) + 1
        for o in (fd, ft):
            o.write("</mediawiki>\n")
    top = sorted(by_ns.items(), key=lambda x: -x[1])[:8]
    print("namespace pages: dev %d test %d | %s" % (n[0], n[1], ", ".join("%s %d" % t for t in top)))
    return 0


def main() -> int:
    a = sys.argv[1:]
    cmds = {"oasst": (oasst, 3), "dolly": (dolly, 3), "plain2": (plain2, 2), "ns": (ns, 3)}
    if not a or a[0] not in cmds or len(a) - 1 != cmds[a[0]][1]:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    return cmds[a[0]][0](*a[1:])


if __name__ == "__main__":
    sys.exit(main())
