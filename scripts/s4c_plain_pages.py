#!/usr/bin/env python3
"""S4c's control pool P (docs/reports/S4_UNSEEN_AND_FRONTIER.md, "S4c"): wiki pages with their markup removed.

Article pages of the dump whose body starts at or after byte 50,000,000 (read by no S4b set), not redirects, namespace
pages, lists or disambiguation pages, every 4th in dump order. Each is written as a page in the same form as
s4b_dolly_pages.py's Dolly pages: its title, and its text with the markup removed by this renderer (separate from
pattern_learn's prose reader). Templates, tables, references, comments, tags, categories, images and interwiki links are
removed; links become their shown words; headings are dropped; list items are written as "- item"; indents lose their
markers. If an out-of-scope signal scores these pages unlike their sources, it reads markup, not content.

usage: s4c_plain_pages.py ENWIK8 OUT.xml
"""
import html
import re
import sys

FROM = 50_000_000
EVERY = 4
NAMESPACES = re.compile(r"^(category|image|file|media|template|wikipedia|help|portal|user|talk|special|mediawiki|wiktionary|"
                        r"[a-z]{2,3}(-[a-z]+)?|simple|zh-min-nan|be-x-old|roa-rup|map-bms|nds-nl|bat-smg)\s*:", re.I)


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def skipped_title(title: str) -> bool:
    t = title.lower()
    return t.startswith("list of ") or "(disambiguation)" in t


def namespace_title(title: str) -> bool:  # pattern_learn's rule: a colon followed by a non-space
    c = title.find(":")
    return c != -1 and c + 1 < len(title) and title[c + 1] != " "


def link_text(m: re.Match) -> str:
    inner = m.group(1)
    target = inner.split("|", 1)[0].strip()
    if NAMESPACES.match(target.lstrip(":")) and not target.startswith(":"):
        return ""
    shown = inner.rsplit("|", 1)[-1] if "|" in inner else target
    return shown + (m.group(2) or "")


def render(text: str) -> str:
    t = html.unescape(text)  # the dump's XML escapes
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"<ref[^>]*/>", "", t, flags=re.I)
    t = re.sub(r"<ref[^>]*>.*?</ref>", "", t, flags=re.I | re.S)
    t = re.sub(r"<(math|gallery|timeline|imagemap|hiero|score)[^>]*>.*?</\1>", "", t, flags=re.I | re.S)
    while True:  # innermost templates and tables first, until none is left
        u = re.sub(r"\{\{[^{}]*\}\}", "", t)
        u = re.sub(r"\{\|(?:(?!\{\||\|\}).)*\|\}", "", u, flags=re.S)
        if u == t:
            break
        t = u
    while True:  # innermost links first (image captions hold links)
        u = re.sub(r"\[\[([^\[\]]*)\]\]([a-z]*)", link_text, t)
        if u == t:
            break
        t = u
    t = re.sub(r"\[(?:https?|ftp|news|mailto):[^\s\]]*\s+([^\]]*)\]", r"\1", t)
    t = re.sub(r"\[(?:https?|ftp|news|mailto):[^\]]*\]", "", t)
    t = re.sub(r"(?:https?|ftp)://\S+", "", t)
    t = re.sub(r"'{2,}", "", t)
    t = re.sub(r"__[A-Z]+__", "", t)
    t = re.sub(r"</?[a-zA-Z][^>]*>", "", t)
    t = html.unescape(t)  # HTML entities (&nbsp;, &mdash;) written escaped in the dump
    out = []
    for line in t.split("\n"):
        s = line.strip()
        if not s or re.match(r"^=+.*=+$", s) or s[0] in "|!{}":
            continue
        if s[0] in "*#":
            s = "- " + s.lstrip("*#").strip()
        elif s[0] in ":;":
            s = s.lstrip(":;").strip()
        if s and s != "-":
            out.append(s)
    return "\n".join(out)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    data = open(sys.argv[1], "rb").read()
    pos = qualifying = kept = 0
    with open(sys.argv[2], "w", encoding="utf-8", newline="\n") as out:
        out.write("<mediawiki>\n")
        while True:
            pb = data.find(b"<page>", pos)
            if pb < 0:
                break
            pe = data.find(b"</page>", pb)
            if pe < 0:
                break
            pos = pe + 7
            tb, te = data.find(b"<title>", pb), data.find(b"</title>", pb)
            xb = data.find(b"<text", te)
            if tb < 0 or te < 0 or te > pe or xb < 0 or xb > pe:
                continue
            xg = data.find(b">", xb)
            if xg < 0 or xg > pe or data[xg - 1:xg] == b"/":
                continue
            xe = data.find(b"</text>", xg)
            if xe < 0 or xe > pe or xg + 1 < FROM:
                continue
            title = html.unescape(data[tb + 7:te].decode("utf-8", "replace"))
            body = data[xg + 1:xe].decode("utf-8", "replace")
            if body[:40].strip().lower().startswith("#redirect") or namespace_title(title) or skipped_title(title):
                continue
            qualifying += 1
            if (qualifying - 1) % EVERY:
                continue
            kept += 1
            out.write("  <page>\n    <title>%s</title>\n    <id>0</id>\n    <revision>\n      <id>0</id>\n"
                      "      <timestamp>2006-03-01T00:00:00Z</timestamp>\n      <contributor>\n        <username>plain</username>\n"
                      "        <id>0</id>\n      </contributor>\n      <text xml:space=\"preserve\">%s</text>\n    </revision>\n"
                      "  </page>\n" % (esc(title), esc(render(body))))
        out.write("</mediawiki>\n")
    print("plain pages %d of %d qualifying pages from byte %d" % (kept, qualifying, FROM))
    return 0


if __name__ == "__main__":
    sys.exit(main())
