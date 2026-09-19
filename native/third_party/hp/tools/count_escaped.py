#!/usr/bin/env python3
from pathlib import Path
import sys

KEYS = [
    b"&lt;ref",
    b"&lt;math",
    b"&lt;pre",
    b"&lt;nowiki",
    b"&lt;gallery",
    b"[[Category:",
    b"#REDIRECT",
    b"#redirect",
    b"{{Infobox",
    b"{{infobox",
    b"{|",
    b"[[",
]


def main() -> None:
    path = Path(sys.argv[1])
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    d = path.read_bytes()
    if limit:
        d = d[:limit]
    low = d.lower()
    print(path, "bytes", len(d))
    for k in KEYS:
        n = low.count(k.lower())
        print(f"  {k.decode()}  {n}")


if __name__ == "__main__":
    main()
