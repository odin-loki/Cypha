#!/usr/bin/env python3
"""Evaluate each SOM upgrade vs baseline; print keep/revert summary."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results" / "baseline.json"
UPGRADES = ["none", "U2", "U1", "U3", "U4", "U5", "U6", "all"]
SEEDS = 3


def run(upgrade: str, out: Path) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "benchmark_baseline.py"),
        "--dataset",
        "classification",
        "--seeds",
        str(SEEDS),
        "--upgrade",
        upgrade,
        "--output",
        str(out),
    ]
    if upgrade != "none" and BASELINE.exists():
        cmd.extend(["--compare", str(BASELINE)])
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    with open(out, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    results = {}
    for u in UPGRADES:
        out = ROOT / "results" / f"som_{u.lower()}.json"
        print(f"\n=== {u} ===", flush=True)
        results[u] = run(u, out)
    summary = []
    base_acc = results["none"]["classification"]["accuracy_mean"]
    base_train = results["none"]["classification"]["train_ms_mean"]
    for u in UPGRADES:
        if u == "none":
            continue
        m = results[u]["classification"]
        acc = m["accuracy_mean"]
        train = m["train_ms_mean"]
        checks = results[u].get("checks", {})
        ok = checks.get("all_pass", acc >= base_acc - 0.005 and train <= base_train * 1.25)
        verdict = "KEEP" if ok else "REVERT"
        summary.append({
            "upgrade": u,
            "accuracy": acc,
            "delta_acc": acc - base_acc,
            "train_ms": train,
            "verdict": verdict,
            "checks": checks,
        })
        print(f"{u}: acc={acc:.4f} ({acc-base_acc:+.4f}) train={train:.2f}ms -> {verdict}")

    report = ROOT / "results" / "som_upgrade_summary.json"
    with open(report, "w", encoding="utf-8") as f:
        json.dump({"baseline_acc": base_acc, "summary": summary}, f, indent=2)
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
