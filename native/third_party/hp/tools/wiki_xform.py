#!/usr/bin/env python3
"""Reversible entity fold for enwik XML (H1.5 lab).

Maps common XML entities to unused C0 bytes so models see one symbol
instead of 4–6. Inverse is exact. Raw 0x10–0x15 in the source are
escaped as 0x15 + byte so the map is bijective.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ESC = 0x15
MAP = (
    (b"&amp;nbsp;", bytes([0x14])),
    (b"&nbsp;", bytes([0x14])),
    (b"&quot;", bytes([0x13])),
    (b"&lt;", bytes([0x10])),
    (b"&gt;", bytes([0x11])),
    (b"&amp;", bytes([0x12])),
)
INV = {
    0x10: b"&lt;",
    0x11: b"&gt;",
    0x12: b"&amp;",
    0x13: b"&quot;",
    0x14: b"&amp;nbsp;",
}


def encode(data: bytes) -> bytes:
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        b = data[i]
        if 0x10 <= b <= 0x15:
            out.append(ESC)
            out.append(b)
            i += 1
            continue
        hit = False
        for pat, tok in MAP:
            if data.startswith(pat, i):
                out.extend(tok)
                i += len(pat)
                hit = True
                break
        if not hit:
            out.append(b)
            i += 1
    return bytes(out)


def decode(data: bytes) -> bytes:
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        b = data[i]
        if b == ESC and i + 1 < n:
            out.append(data[i + 1])
            i += 2
            continue
        if b in INV:
            out.extend(INV[b])
            i += 1
            continue
        out.append(b)
        i += 1
    return bytes(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("e", "d", "check"))
    ap.add_argument("inp")
    ap.add_argument("out", nargs="?")
    args = ap.parse_args()
    raw = Path(args.inp).read_bytes()
    if args.mode == "e":
        enc = encode(raw)
        Path(args.out).write_bytes(enc)
        print(f"{len(raw):,} -> {len(enc):,}  sha {hashlib.sha256(enc).hexdigest()[:16]}")
        return 0
    if args.mode == "d":
        dec = decode(raw)
        Path(args.out).write_bytes(dec)
        print(f"{len(raw):,} -> {len(dec):,}")
        return 0
    enc = encode(raw)
    back = decode(enc)
    ok = back == raw
    print(f"roundtrip {'PASS' if ok else 'FAIL'}  {len(raw):,} -> {len(enc):,} -> {len(back):,}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
