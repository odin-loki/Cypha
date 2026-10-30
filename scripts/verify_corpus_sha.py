#!/usr/bin/env python3
"""Check corpus files against published SHA-256 sums and record every digest.

RedPajama lists (Together, v1.0.0):

  https://data.together.xyz/redpajama-data-1T/v1.0.0/sha256/<source>_SHA256SUMS.txt

Wikimedia dump SHA-1 lists are used for the Wikisource and Wikibooks bz2 files.
Files with no published sum still get a SHA-256 in the local manifest.

Resumable. A file already in ``sha256/checked.jsonl`` with the same size is skipped.

Usage:
  python scripts/verify_corpus_sha.py
  python scripts/verify_corpus_sha.py --root D:\\cypha_llm\\corpus --jobs 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SUM_URLS = [
    "https://data.together.xyz/redpajama-data-1T/v1.0.0/sha256/arxiv_SHA256SUMS.txt",
    "https://data.together.xyz/redpajama-data-1T/v1.0.0/sha256/c4_SHA256SUMS.txt",
    "https://data.together.xyz/redpajama-data-1T/v1.0.0/sha256/common_crawl_SHA256SUMS.txt",
    "https://data.together.xyz/redpajama-data-1T/v1.0.0/sha256/github_SHA256SUMS.txt",
    "https://data.together.xyz/redpajama-data-1T/v1.0.0/sha256/stackexchange_SHA256SUMS.txt",
    "https://data.together.xyz/redpajama-data-1T/v1.0.0/sha256/wikipedia_SHA256SUMS.txt",
]

WIKI_SHA1 = [
    ("https://dumps.wikimedia.org/enwikisource/latest/enwikisource-latest-sha1sums.txt",
     "enwikisource-latest-pages-articles.xml.bz2"),
    ("https://dumps.wikimedia.org/enwikibooks/latest/enwikibooks-latest-sha1sums.txt",
     "enwikibooks-latest-pages-articles.xml.bz2"),
    ("https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-sha1sums.txt",
     "enwiki-latest-pages-articles-multistream.xml.bz2"),
]


def default_root() -> Path:
    if os.name == "nt":
        return Path(r"D:\cypha_llm\corpus")
    return Path("/mnt/d/cypha_llm/corpus")


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "cypha-corpus-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        dest.write_bytes(resp.read())


def parse_sums(path: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        digest, name = parts[0], parts[-1].lstrip("*")
        if len(digest) not in (40, 64):
            continue
        found[Path(name).name] = digest.lower()
    return found


def hash_file(path_s: str) -> tuple[str, int, str, str]:
    sha256 = hashlib.sha256()
    sha1 = hashlib.sha1()
    size = 0
    with open(path_s, "rb") as handle:
        while True:
            block = handle.read(8 << 20)
            if not block:
                break
            sha256.update(block)
            sha1.update(block)
            size += len(block)
    return path_s, size, sha256.hexdigest(), sha1.hexdigest()


def load_done(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        done[row["path"]] = row
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description="SHA-check the corpus on D:")
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    root: Path = args.root
    official = root / "sha256" / "official"
    official.mkdir(parents=True, exist_ok=True)
    for url in SUM_URLS:
        name = url.rsplit("/", 1)[-1]
        print(f"sum {name}", flush=True)
        fetch(url, official / name)
    expected: dict[str, str] = {}
    for path in sorted(official.glob("*SHA256SUMS.txt")):
        got = parse_sums(path)
        print(f"  {path.name} entries {len(got)}", flush=True)
        overlap = set(expected) & set(got)
        if overlap:
            print(f"  duplicate names {len(overlap)}", flush=True)
        expected.update(got)

    sha1_expected: dict[str, str] = {}
    for url, want_name in WIKI_SHA1:
        dest = official / url.rsplit("/", 1)[-1]
        try:
            fetch(url, dest)
        except Exception as exc:  # noqa: BLE001
            print(f"sha1 list failed {url} {exc}", flush=True)
            continue
        parsed = parse_sums(dest)
        if want_name in parsed:
            sha1_expected[want_name] = parsed[want_name]
            print(f"  sha1 {want_name} present", flush=True)
        else:
            print(f"  sha1 {want_name} missing from {dest.name}", flush=True)

    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "sha256"]
        for name in filenames:
            files.append(Path(dirpath) / name)
    files.sort(key=lambda p: p.stat().st_size, reverse=True)

    by_name: dict[str, list[Path]] = {}
    for path in files:
        by_name.setdefault(path.name, []).append(path)
    local_names = set(by_name)
    missing = sorted(set(expected) - local_names)
    extra = sorted(name for name in local_names if name not in expected and not name.endswith(".part"))
    print(f"official {len(expected)} local_files {len(files)} missing_official {len(missing)} no_official_sum {len(extra)}", flush=True)
    buckets: dict[str, int] = {}
    for name in missing:
        key = name.split("_", 1)[0]
        if name.startswith("c4-"):
            key = "c4"
        elif name.startswith("filtered_"):
            key = "github"
        elif "common_crawl" in name or name.startswith("cc_"):
            key = "common_crawl"
        elif "wiki" in name:
            key = "wikipedia"
        buckets[key] = buckets.get(key, 0) + 1
    print("missing_by_source " + " ".join(f"{k}={v}" for k, v in sorted(buckets.items())), flush=True)
    for name in missing:
        if not name.startswith("c4-"):
            print(f"  not_on_disk {name}", flush=True)
    miss_path = root / "sha256" / "missing_official.txt"
    miss_path.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")

    checked_path = root / "sha256" / "checked.jsonl"
    done = load_done(checked_path)
    pending: list[Path] = []
    reused = 0
    for path in files:
        row = done.get(str(path))
        if row and row.get("size") == path.stat().st_size and row.get("sha256"):
            reused += 1
            continue
        pending.append(path)
    print(f"reuse {reused} to_hash {len(pending)}", flush=True)

    out = checked_path.open("a", encoding="utf-8")
    counts = {"ok": 0, "mismatch": 0, "no-official-sum": 0, "partial": 0}
    hashed_bytes = 0
    t0 = time.monotonic()
    mismatches: list[str] = []

    def take(path_s: str, size: int, sha256: str, sha1: str) -> None:
        nonlocal hashed_bytes
        path = Path(path_s)
        hashed_bytes += size
        partial = path.name.endswith(".part")
        want = expected.get(path.name)
        if partial:
            status = "partial"
        elif want:
            status = "ok" if sha256 == want else "mismatch"
        else:
            status = "no-official-sum"
        sha1_want = None
        for remote_name, digest in sha1_expected.items():
            if "wikisource" in remote_name and "wikisource" in path.name:
                sha1_want = digest
            elif "wikibooks" in remote_name and "wikibooks" in path.name:
                sha1_want = digest
            elif "multistream" in remote_name and "multistream" in path.name:
                sha1_want = digest
        sha1_status = None
        if sha1_want and not partial:
            sha1_status = "ok" if sha1 == sha1_want else "mismatch"
            if sha1_status == "mismatch":
                status = "mismatch"
            elif status == "no-official-sum":
                status = "sha1-ok"
        row = {
            "path": path_s,
            "size": size,
            "sha256": sha256,
            "sha1": sha1,
            "expected_sha256": want,
            "expected_sha1": sha1_want,
            "sha1_status": sha1_status,
            "status": status,
        }
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        if row["status"] == "mismatch":
            mismatches.append(path_s)
            print(f"MISMATCH {path_s}", flush=True)
        out.write(json.dumps(row) + "\n")
        out.flush()
        elapsed = max(time.monotonic() - t0, 0.001)
        print(
            f"hashed {hashed_bytes / (1024 ** 3):.2f} GB  {hashed_bytes / elapsed / (1024 ** 2):.0f} MB/s  {path.name}",
            flush=True,
        )

    # Count reused rows too.
    for path in files:
        row = done.get(str(path))
        if row and row.get("size") == path.stat().st_size and row.get("sha256"):
            counts[row.get("status", "ok")] = counts.get(row.get("status", "ok"), 0) + 1
            if row.get("status") == "mismatch":
                mismatches.append(str(path))

    if pending:
        with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futures = [pool.submit(hash_file, str(path)) for path in pending]
            for fut in as_completed(futures):
                take(*fut.result())
    out.close()

    report = root / "sha256" / "report.txt"
    lines = [
        f"official_entries {len(expected)}",
        f"missing_official {len(missing)}",
        f"local_files {len(files)}",
        f"hashed_now_bytes {hashed_bytes}",
        f"mismatches {len(mismatches)}",
    ]
    for key in sorted(counts):
        lines.append(f"{key} {counts[key]}")
    for path_s in mismatches:
        lines.append(f"MISMATCH {path_s}")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
