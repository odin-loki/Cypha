#!/usr/bin/env python3
"""Print bpc from Tee-Object JSON dumps (UTF-8 or UTF-16)."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8", errors="ignore")
    return json.loads(text)


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    paths = sorted(root.glob("*.json")) if root.is_dir() else [root]
    for p in paths:
        j = load(p)
        print(
            f"{p.name}: bpc={j.get('bpc')} cell={j.get('cell_variant')} "
            f"mode={j.get('mode')} n_train={j.get('n_train')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
