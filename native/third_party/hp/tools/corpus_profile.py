#!/usr/bin/env python3
"""Profile a Wikipedia XML dump slice (enwik8 / enwik9 / 1 MB stub).

Counts the axes the prize winners actually model: articles, markup,
tables, links, templates, digits, math/pre/nowiki, headers, UTF-8.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

MARKERS = (
    (b"<page>", "page_open"),
    (b"</page>", "page_close"),
    (b"<title>", "title_open"),
    (b"{|", "table"),
    (b"[[Category:", "category"),
    (b"[[", "wikilink"),
    (b"{{", "template"),
    (b"<ref", "ref"),
    (b"http://", "http"),
    (b"https://", "https"),
    (b"<math", "math"),
    (b"<pre", "pre"),
    (b"<nowiki", "nowiki"),
    (b"&amp;", "entity"),
    (b"&lt;", "entity"),
    (b"&gt;", "entity"),
    (b"&quot;", "entity"),
    (b"&nbsp;", "entity"),
    (b"#REDIRECT", "redirect"),
    (b"#redirect", "redirect"),
    (b"<comment>", "comment"),
    (b"<revision>", "revision"),
    (b"Infobox", "infobox"),
    (b"{{cite", "cite"),
    (b"{{Cite", "cite"),
)

NEEDLE_MAX = max(len(n) for n, _ in MARKERS)


def profile(path: Path, limit: int | None = None) -> dict:
    counts: collections.Counter[str] = collections.Counter()
    n = 0
    digit_runs = 0
    in_digit = False
    utf8_hi = 0
    page_sizes: list[int] = []
    bytes_in_page = 0
    in_page = False
    sample_titles: list[str] = []
    capturing_title = False
    title_buf = bytearray()
    leftover = b""

    def consume(buf: bytes, start: int) -> None:
        nonlocal n, in_digit, digit_runs, utf8_hi
        nonlocal bytes_in_page, in_page, capturing_title, title_buf
        i = start
        while i < len(buf):
            b = buf[i]
            n += 1
            if in_page:
                bytes_in_page += 1
            if 48 <= b <= 57:
                if not in_digit:
                    digit_runs += 1
                    in_digit = True
                counts["digit"] += 1
            else:
                in_digit = False
            if b >= 0x80:
                utf8_hi += 1
            if 65 <= b <= 90:
                counts["upper"] += 1
            elif 97 <= b <= 122:
                counts["lower"] += 1
            elif b == 10:
                counts["newline"] += 1
            if capturing_title:
                if b == 0x3C:
                    capturing_title = False
                    if title_buf and len(sample_titles) < 16:
                        sample_titles.append(title_buf.decode("utf-8", "replace"))
                    title_buf = bytearray()
                elif len(title_buf) < 160:
                    title_buf.append(b)
            i += 1

    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            if limit is not None:
                remain = limit - n
                if remain <= 0:
                    break
                if len(leftover) + len(chunk) > remain + NEEDLE_MAX:
                    chunk = chunk[: max(0, remain + NEEDLE_MAX - len(leftover))]
            buf = leftover + chunk
            # Count markers whose start lies in the new (non-overlap) region.
            new_start = len(leftover)
            for needle, name in MARKERS:
                pos = 0
                while True:
                    j = buf.find(needle, pos)
                    if j < 0:
                        break
                    if j >= new_start or pos == 0 and j < new_start:
                        # accept if the match begins at/after previous leftover
                        # except we must include matches that started in leftover
                        # and finished in chunk — those have j < new_start.
                        if j + len(needle) <= len(buf):
                            if j >= new_start or (j < new_start and j + len(needle) > new_start):
                                counts[name] += 1
                                if name == "page_open":
                                    if in_page and bytes_in_page:
                                        page_sizes.append(bytes_in_page)
                                    in_page = True
                                    bytes_in_page = 0
                                elif name == "page_close":
                                    in_page = False
                                    page_sizes.append(bytes_in_page)
                                    bytes_in_page = 0
                                elif name == "title_open":
                                    capturing_title = True
                                    title_buf = bytearray()
                    pos = j + 1
            # Consume only the new bytes for per-byte stats.
            consume(buf, new_start)
            leftover = buf[-(NEEDLE_MAX - 1) :] if len(buf) >= NEEDLE_MAX else buf
            # Those leftover bytes were already consumed; do not consume again.
            leftover_for_search = leftover
            leftover = leftover_for_search
            if limit is not None and n >= limit:
                break

    file_n = path.stat().st_size if limit is None else min(path.stat().st_size, limit)
    return {
        "path": str(path),
        "bytes": file_n,
        "pages_open": counts["page_open"],
        "pages_close": counts["page_close"],
        "tables": counts["table"],
        "wikilinks": counts["wikilink"],
        "templates": counts["template"],
        "refs": counts["ref"],
        "http": counts["http"] + counts["https"],
        "math": counts["math"],
        "pre": counts["pre"],
        "nowiki": counts["nowiki"],
        "entities": counts["entity"],
        "categories": counts["category"],
        "redirects": counts["redirect"],
        "infobox": counts["infobox"],
        "revisions": counts["revision"],
        "comments": counts["comment"],
        "cites": counts["cite"],
        "digits": counts["digit"],
        "digit_runs": digit_runs,
        "utf8_hi": utf8_hi,
        "upper": counts["upper"],
        "lower": counts["lower"],
        "newlines": counts["newline"],
        "page_sizes": page_sizes,
        "sample_titles": sample_titles,
        "scanned": n,
    }


def summarize(st: dict) -> str:
    n = max(st["bytes"], 1)
    sizes = st["page_sizes"]
    lines = [
        f"file          {st['path']}",
        f"bytes         {st['bytes']:,}   scanned {st['scanned']:,}",
        f"pages         open {st['pages_open']:,}   close {st['pages_close']:,}",
        f"revisions     {st['revisions']:,}",
        f"tables {{|    {st['tables']:,}    {1000 * st['tables'] / n:.3f} /kB",
        f"wikilinks [[  {st['wikilinks']:,}    {1000 * st['wikilinks'] / n:.3f} /kB",
        f"templates {{  {st['templates']:,}    {1000 * st['templates'] / n:.3f} /kB",
        f"cites         {st['cites']:,}",
        f"infobox       {st['infobox']:,}",
        f"categories    {st['categories']:,}",
        f"redirects     {st['redirects']:,}",
        f"<ref          {st['refs']:,}",
        f"http(s)       {st['http']:,}",
        f"<math         {st['math']:,}",
        f"<pre          {st['pre']:,}",
        f"<nowiki       {st['nowiki']:,}",
        f"entities      {st['entities']:,}",
        f"digits        {st['digits']:,}  ({100 * st['digits'] / n:.2f}%)  runs {st['digit_runs']:,}",
        f"utf8 hi-bit   {st['utf8_hi']:,}  ({100 * st['utf8_hi'] / n:.3f}%)",
        f"letters A-Z   {st['upper']:,}   a-z {st['lower']:,}   A/a {st['upper'] / max(st['lower'], 1):.3f}",
        f"newlines      {st['newlines']:,}",
    ]
    if sizes:
        sizes_sorted = sorted(sizes)
        lines.append(
            f"page bytes    n={len(sizes)}  min={sizes_sorted[0]:,}  "
            f"p50={sizes_sorted[len(sizes)//2]:,}  "
            f"p90={sizes_sorted[int(len(sizes)*0.9)]:,}  "
            f"max={sizes_sorted[-1]:,}  "
            f"mean={sum(sizes)//len(sizes):,}"
        )
    if st["sample_titles"]:
        lines.append("sample titles:")
        for t in st["sample_titles"][:12]:
            lines.append(f"  - {t}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    print(summarize(profile(Path(args.path), args.limit)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
