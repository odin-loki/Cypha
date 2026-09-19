#!/usr/bin/env python3
"""Split a MediaWiki XML dump into <page> records and emit a reordered stream.

Restore is by sorting titles (starlit) or by shipping the permutation.
This tool is for lab tests on enwik8 slices — it does not pack a decoder.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def split_pages(data: bytes) -> tuple[bytes, list[bytes], bytes]:
    open_tag = b"<page>"
    close_tag = b"</page>"
    pages: list[bytes] = []
    pos = data.find(open_tag)
    if pos < 0:
        return data, [], b""
    header = data[:pos]
    while pos >= 0:
        end = data.find(close_tag, pos)
        if end < 0:
            pages.append(data[pos:])
            return header, pages, b""
        end += len(close_tag)
        pages.append(data[pos:end])
        nxt = data.find(open_tag, end)
        if nxt < 0:
            return header, pages, data[end:]
        if nxt > end:
            # interstitial (should be rare / whitespace)
            pages[-1] += data[end:nxt]
        pos = nxt
    return header, pages, b""


def page_title(page: bytes) -> bytes:
    a = page.find(b"<title>")
    b = page.find(b"</title>", a + 7) if a >= 0 else -1
    if a < 0 or b < 0:
        return b""
    return page[a + 7 : b]


def is_redirect(page: bytes) -> bool:
    t = page.lower()
    return b"#redirect" in t


def classify(page: bytes) -> str:
    low = page.lower()
    if b"#redirect" in low:
        return "redirect"
    if b"{{disambig" in low or b"{{disambiguation" in low:
        return "disambig"
    if b"[[image:" in low or b"[[file:" in low or b"<title>image:" in low:
        return "image"
    if b"{{infobox" in low:
        return "infobox"
    if b"{|" in page:
        return "table"
    return "article"


def reorder(pages: list[bytes], mode: str) -> list[bytes]:
    if mode == "identity":
        return list(pages)
    if mode == "title":
        return sorted(pages, key=page_title)
    if mode == "title-rev":
        return sorted(pages, key=page_title, reverse=True)
    if mode == "redirects-last":
        body = [p for p in pages if not is_redirect(p)]
        red = [p for p in pages if is_redirect(p)]
        return body + red
    if mode == "fx2-manual":
        # Approximate fx2 manual-sort: images / disambig / redirects to the end.
        order = {"article": 0, "infobox": 0, "table": 1, "image": 2, "disambig": 3, "redirect": 4}
        return sorted(pages, key=lambda p: (order.get(classify(p), 0), page_title(p)))
    if mode == "size":
        return sorted(pages, key=lambda p: (-len(p), page_title(p)))
    raise SystemExit(f"unknown mode {mode}")


def stats(pages: list[bytes]) -> str:
    c: dict[str, int] = {}
    for p in pages:
        c[classify(p)] = c.get(classify(p), 0) + 1
    return " ".join(f"{k}={v}" for k, v in sorted(c.items()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--mode", default="redirects-last")
    ap.add_argument("--limit-bytes", type=int, default=None)
    ap.add_argument("--restore-check", action="store_true")
    args = ap.parse_args()
    data = Path(args.input).read_bytes()
    if args.limit_bytes is not None:
        data = data[: args.limit_bytes]
    header, pages, footer = split_pages(data)
    print(f"in={len(data):,} header={len(header):,} pages={len(pages)} footer={len(footer):,}")
    print(f"classes {stats(pages)}")
    new_pages = reorder(pages, args.mode)
    out = header + b"".join(new_pages) + footer
    Path(args.output).write_bytes(out)
    print(f"out={len(out):,} sha256={hashlib.sha256(out).hexdigest()[:16]}")
    if args.restore_check:
        # starlit restore: sort pages by title (enwik is already title-sorted,
        # so identity should match after title-sort of a title-sorted dump).
        restored = header + b"".join(sorted(new_pages, key=page_title)) + footer
        ok = restored == data
        print(f"title-sort restore {'PASS' if ok else 'FAIL'} ({len(restored):,})")
        if not ok:
            # fallback: original order is not pure title order (XML wrapper)
            orig = header + b"".join(pages) + footer
            print(f"roundtrip identity {orig == data}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
