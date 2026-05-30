#!/usr/bin/env python3
"""Merge converge winners (tabular/vision) with profiled_medium regression."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO / "cypha_bench" / "config" / "everyday_profile.json"
MEDIUM = REPO / "config" / "profiled_medium.json"


def main() -> None:
    current = json.loads(PROFILE.read_text(encoding="utf-8"))
    medium = json.loads(MEDIUM.read_text(encoding="utf-8"))
    reg = dict(medium["regression_difregressor"])
    merged = {
        **current,
        "source": "converge winners + profiled_medium regression hybrid",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "regimes": {
            **current.get("regimes", {}),
            "regression": reg,
        },
        "architecture": {
            **current.get("architecture", {}),
            "replay_ratio": 0.15,
            "ood_sigma": 15.0,
        },
    }
    PROFILE.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"Wrote {PROFILE}")


if __name__ == "__main__":
    main()
