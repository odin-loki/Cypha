#!/usr/bin/env python3
"""Summarize Tee-Object JSON dumps under bench/results/math_open/."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def last_json(text: str) -> dict | None:
    i = text.rfind("{")
    if i < 0:
        return None
    depth = 0
    for j, ch in enumerate(text[i:]):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[i : i + j + 1])
    return None


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "bench/results/math_open")
    rows = []
    for p in sorted(root.glob("*.json")):
        if p.name == "summary.json":
            continue
        raw = p.read_bytes()
        if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
            text = raw.decode("utf-16")
        else:
            text = raw.decode("utf-8", errors="ignore")
        # Prefer whole-file JSON when Tee-Object wrote a single object.
        obj = None
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = last_json(text)
        if not isinstance(obj, dict):
            print(f"{p.name}: NO_JSON")
            continue
        mi = obj.get("math_integration") or {}
        nav = mi.get("navigation_config") or {}
        ip = obj.get("intelligence_profile") or {}
        pc = obj.get("profile_completeness") or {}
        row = {
            "file": p.name,
            "bpc": obj.get("bpc"),
            "n_train": obj.get("n_train"),
            "kappa": mi.get("kappa", pc.get("kappa", ip.get("kappa"))),
            "eigen": nav.get("use_eigenvalue_d_eff"),
            "kappa_target": nav.get("kappa_lambda_target"),
            "method": ip.get("lstm_hidden_d_eff_method"),
        }
        rows.append(row)
        print(
            f"{row['file']}: n={row['n_train']} bpc={row['bpc']} kappa={row['kappa']} "
            f"eigen={row['eigen']} target={row['kappa_target']} method={row['method']}"
        )
    out = root / "summary.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
