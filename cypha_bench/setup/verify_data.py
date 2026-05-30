#!/usr/bin/env python3
"""
Verify cypha_bench datasets are present and valid.

Usage:
    python cypha_bench/setup/verify_data.py
    python cypha_bench/setup/verify_data.py --skip-missing
    python cypha_bench/setup/verify_data.py --only mnist gutenberg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(BENCH_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT.parent))

from cypha_bench.common.paths import DATA_DIR

REQUIRED: dict[str, list[str]] = {
    "mnist": [
        "mnist/train-images-idx3-ubyte",
        "mnist/train-labels-idx1-ubyte",
        "mnist/t10k-images-idx3-ubyte",
        "mnist/t10k-labels-idx1-ubyte",
    ],
    "wikitext2": [
        "wikitext2/wikitext-2/wiki.train.tokens",
    ],
    "nsl_kdd": [
        "nsl_kdd/KDDTrain+.txt",
        "nsl_kdd/KDDTest+.txt",
    ],
    "ecg5000": [
        "ecg5000/ECG5000_TRAIN.txt",
        "ecg5000/ECG5000_TEST.txt",
    ],
    "canterbury": [
        "canterbury/alice29.txt",
    ],
    "gutenberg": [
        "gutenberg/moby_dick.txt",
        "gutenberg/sherlock_holmes.txt",
        "gutenberg/alice.txt",
    ],
    "chess": [
        "chess/Kasparov.pgn",
    ],
}


def _check_paths(rel_paths: list[str]) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    present: list[str] = []
    for rel in rel_paths:
        full = DATA_DIR / rel
        if full.exists() and full.stat().st_size > 0:
            present.append(rel)
        else:
            missing.append(rel)
    return present, missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify cypha_bench datasets.")
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Report missing files but exit 0 (warn only).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(REQUIRED.keys()),
        help="Verify only selected dataset groups.",
    )
    args = parser.parse_args()

    groups = args.only if args.only else sorted(REQUIRED.keys())
    all_missing: list[str] = []
    all_present: list[str] = []

    print(f"Checking data under: {DATA_DIR}\n")
    for group in groups:
        present, missing = _check_paths(REQUIRED[group])
        all_present.extend(present)
        all_missing.extend(missing)
        status = "OK" if not missing else f"MISSING {len(missing)}"
        print(f"  [{group}] {status}")
        for p in missing:
            print(f"    - {p}")

    print(f"\nPresent: {len(all_present)}  Missing: {len(all_missing)}")

    if all_missing:
        print("\nMISSING FILES:")
        for p in all_missing:
            print(f"  data/{p}")
        print("\nRun: python cypha_bench/setup/acquire_data.py")
        if args.skip_missing:
            print("(Continuing with --skip-missing.)")
            return 0
        return 1

    print("All required data files present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
