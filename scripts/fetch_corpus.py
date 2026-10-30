#!/usr/bin/env python3
"""Resumable corpus fetch for a ~1 TB text mix on D:.

Stop with Ctrl+C. Run the same command again and finished files are skipped.
Partial files stay as ``*.part`` and continue with an HTTP Range request.

Categories and budgets (bytes on disk, compressed files count as stored):

  books       40 GB   Gutenberg texts, Wikisource, Wikibooks
  code       200 GB   RedPajama GitHub slice
  articles   580 GB   C4, RedPajama Common Crawl, English Wikipedia, a WET sample
  papers      90 GB   RedPajama arXiv
  structured  80 GB   Stack Exchange
  labelled    20 GB   Dolly, OpenAssistant, AG News, SQuAD, IMDB

The RedPajama book slice is not used. It was removed for copyright.

Legal book text is tens of gigabytes, not hundreds. The terabyte is web text,
code, papers, and reference dumps. Quotas are caps, not promises that every
source is that large.

Usage:
  python scripts/fetch_corpus.py --check
  python scripts/fetch_corpus.py
  python scripts/fetch_corpus.py --root D:\\cypha_llm\\corpus
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

GB = 1024 ** 3

BUDGET = {
    "books": 40 * GB,
    "code": 200 * GB,
    "articles": 580 * GB,
    "papers": 90 * GB,
    "structured": 80 * GB,
    "labelled": 20 * GB,
}

# Seed files and list-of-URLs. Checked with HEAD or a short GET on 2026-10-25.
DIRECT = [
    ("articles", "wiki/enwiki-pages-articles-multistream.xml.bz2",
     "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2"),
    ("books", "wiki/enwikisource-pages-articles.xml.bz2",
     "https://dumps.wikimedia.org/enwikisource/latest/enwikisource-latest-pages-articles.xml.bz2"),
    ("books", "wiki/enwikibooks-pages-articles.xml.bz2",
     "https://dumps.wikimedia.org/enwikibooks/latest/enwikibooks-latest-pages-articles.xml.bz2"),
    ("labelled", "dolly/databricks-dolly-15k.jsonl",
     "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"),
    ("labelled", "oasst/train-00000-of-00001.parquet",
     "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/data/train-00000-of-00001-b42a775f407cee45.parquet"),
    ("labelled", "ag_news/train-00000-of-00001.parquet",
     "https://huggingface.co/datasets/fancyzhx/ag_news/resolve/main/data/train-00000-of-00001.parquet"),
    ("labelled", "squad/train-00000-of-00001.parquet",
     "https://huggingface.co/datasets/rajpurkar/squad/resolve/main/plain_text/train-00000-of-00001.parquet"),
    ("labelled", "imdb/train-00000-of-00001.parquet",
     "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text/train-00000-of-00001.parquet"),
]

LISTS = [
    ("papers", "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T/resolve/main/urls/arxiv.txt?download=true"),
    ("code", "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T/resolve/main/urls/github.txt?download=true"),
    ("articles", "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T/resolve/main/urls/c4.txt?download=true"),
    ("articles", "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T/resolve/main/urls/common_crawl.txt?download=true"),
    ("structured", "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T/resolve/main/urls/stackexchange.txt?download=true"),
]

# Newest crawl whose wet.paths.gz returned 200 on 2026-10-25. A 50 GB cap
# lives inside the articles budget; the list itself is far larger.
WET_PATHS = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-43/wet.paths.gz"
WET_CAP = 50 * GB
GUTENBERG_CATALOG = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
GUTENBERG_DELAY = 2.0


def default_root() -> Path:
    if os.name == "nt":
        return Path(r"D:\cypha_llm\corpus")
    return Path("/mnt/d/cypha_llm/corpus")


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"done": {}, "skipped": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def used_bytes(root: Path, category: str) -> int:
    folder = root / category
    if not folder.exists():
        return 0
    total = 0
    for p in folder.rglob("*"):
        if p.is_file() and not p.name.endswith(".part"):
            total += p.stat().st_size
    return total


def request(url: str, headers: dict | None = None, timeout: int = 60):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "cypha-corpus-fetch/1.0"})
    return urllib.request.urlopen(req, timeout=timeout)


def probe(url: str) -> tuple[int, str]:
    try:
        with request(url, {"User-Agent": "cypha-corpus-fetch/1.0", "Range": "bytes=0-199"}) as resp:
            resp.read(200)
            return resp.status, resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        return exc.code, ""
    except Exception as exc:  # noqa: BLE001
        return 0, str(exc)


def format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds > 99 * 24 * 3600:
        return "--:--:--"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


class Progress:
    """One updating line: bar, bytes against the 1 TB budget, speed, time left."""

    def __init__(self, target: int, already: int) -> None:
        self.target = target
        self.base = already
        self.session = 0
        self.t0 = time.monotonic()
        self.label = ""
        self.active = 0
        self.lock = threading.Lock()

    def add(self, nbytes: int, label: str) -> None:
        with self.lock:
            self.session += nbytes
            self.label = label
            self.draw()

    def begin(self, label: str) -> None:
        with self.lock:
            self.active += 1
            self.label = label
            self.draw()

    def end(self) -> None:
        with self.lock:
            self.active = max(0, self.active - 1)
            self.draw()

    def draw(self) -> None:
        have = self.base + self.session
        frac = min(1.0, have / self.target) if self.target else 0.0
        elapsed = max(time.monotonic() - self.t0, 0.001)
        speed = self.session / elapsed
        left = format_eta((self.target - have) / speed if speed > 1 else None)
        width = 20
        filled = int(width * frac)
        bar = "#" * filled + "-" * (width - filled)
        name = self.label[-24:]
        line = (
            f"[{bar}] {have / GB:6.2f}/{self.target / GB:.0f} GB"
            f"  {speed / (1024 * 1024):5.1f} MB/s  left {left}  {name}"
        )
        cols = shutil.get_terminal_size(fallback=(80, 24)).columns
        if cols < 40 or cols > 160:
            cols = 78
        visible = line[: cols - 1].ljust(cols - 1)
        sys.stdout.write("\r" + visible)
        sys.stdout.flush()

    def finish_file(self) -> None:
        with self.lock:
            self.draw()


def download(url: str, dest: Path, progress: Progress | None = None, timeout: int = 120,
             accept_gzip: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    part = dest.with_suffix(dest.suffix + ".part")
    # A gzip body cannot be continued with a byte range of the decoded file.
    if accept_gzip and part.exists():
        part.unlink()
    existing = 0 if accept_gzip else (part.stat().st_size if part.exists() else 0)
    headers = {"User-Agent": "cypha-corpus-fetch/1.0"}
    if accept_gzip:
        headers["Accept-Encoding"] = "gzip"
    elif existing:
        headers["Range"] = f"bytes={existing}-"
    label = dest.name
    if progress is not None:
        progress.begin(label)
    try:
        with request(url, headers, timeout) as resp:
            code = getattr(resp, "status", 200)
            if code == 416 and part.exists():
                part.replace(dest)
                return
            mode = "ab" if code == 206 and existing else "wb"
            total = None
            content_range = resp.headers.get("Content-Range")
            if content_range and "/" in content_range:
                total_s = content_range.rsplit("/", 1)[-1]
                if total_s.isdigit():
                    total = int(total_s)
            elif resp.headers.get("Content-Length", "").isdigit() and code != 206:
                total = int(resp.headers["Content-Length"])
            stream = resp
            if accept_gzip and "gzip" in resp.headers.get("Content-Encoding", "").lower():
                stream = gzip.GzipFile(fileobj=resp)
                total = None
            got = existing if mode == "ab" else 0
            with part.open(mode) as out:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    got += len(chunk)
                    if progress is not None:
                        progress.add(len(chunk), label)
        if total is not None and got != total:
            raise OSError(f"short download {got} of {total} for {dest.name}")
        part.replace(dest)
    finally:
        if progress is not None:
            progress.end()


def fetch_text(url: str) -> str:
    with request(url) as resp:
        return resp.read().decode("utf-8", errors="replace")


def under_budget(root: Path, category: str, extra_cap: int | None = None) -> bool:
    used = used_bytes(root, category)
    if used >= BUDGET[category]:
        return False
    if extra_cap is not None:
        return used < extra_cap
    return True


def take(root: Path, category: str, url: str, dest: Path, state: dict, state_path: Path,
         progress: Progress | None = None) -> None:
    key = url
    if key in state["done"]:
        return
    if not under_budget(root, category):
        return
    download(url, dest, progress)
    state["done"][key] = {"path": str(dest), "bytes": dest.stat().st_size if dest.exists() else 0}
    save_state(state_path, state)


def gutenberg(root: Path, state: dict, state_path: Path, progress: Progress | None = None) -> None:
    catalog = root / "books" / "pg_catalog.csv"
    take(root, "books", GUTENBERG_CATALOG, catalog, state, state_path, progress)
    if not catalog.exists():
        return
    with catalog.open(encoding="utf-8", errors="replace", newline="") as handle:
        rows = csv.DictReader(handle)
        field = "Text#" if rows.fieldnames and "Text#" in rows.fieldnames else None
        for row in rows:
            if not under_budget(root, "books"):
                return
            raw = row.get(field) if field else None
            if not raw:
                continue
            book_id = raw.strip()
            if not book_id.isdigit():
                continue
            url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
            dest = root / "books" / "gutenberg" / f"pg{book_id}.txt"
            if url in state["done"] or url in state["skipped"] or dest.exists():
                continue
            time.sleep(GUTENBERG_DELAY)
            try:
                # Gutenberg answers 406 unless the client accepts the gzip variant.
                download(url, dest, progress, accept_gzip=True)
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    state["skipped"][url] = 404
                    save_state(state_path, state)
                    continue
                raise
            state["done"][url] = {"path": str(dest), "bytes": dest.stat().st_size}
            save_state(state_path, state)


def from_list(root: Path, category: str, list_url: str, state: dict, state_path: Path,
              progress: Progress | None = None, cap: int | None = None) -> None:
    text = fetch_text(list_url)
    for line in text.splitlines():
        url = line.strip()
        if not url.startswith("http"):
            continue
        if not under_budget(root, category, cap):
            return
        name = url.rstrip("/").split("/")[-1]
        sub = "wet" if "commoncrawl.org" in url else url.split("/")[-2] if "/" in url[8:] else "files"
        dest = root / category / sub / name
        take(root, category, url, dest, state, state_path, progress)


def wet(root: Path, state: dict, state_path: Path, progress: Progress | None = None) -> None:
    listing = root / "articles" / "wet.paths.gz"
    take(root, "articles", WET_PATHS, listing, state, state_path, progress)
    if not listing.exists():
        return
    with gzip.open(listing, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            path = line.strip()
            if not path or path.startswith("#"):
                continue
            if used_bytes(root, "articles") >= BUDGET["articles"]:
                return
            wet_used = sum(
                p.stat().st_size
                for p in (root / "articles" / "wet").glob("*")
                if p.is_file() and not p.name.endswith(".part")
            ) if (root / "articles" / "wet").exists() else 0
            if wet_used >= WET_CAP:
                return
            url = "https://data.commoncrawl.org/" + path.lstrip("/")
            dest = root / "articles" / "wet" / Path(path).name
            take(root, "articles", url, dest, state, state_path, progress)


def check() -> int:
    ok = True
    seeds = [url for _, _, url in DIRECT] + [url for _, url in LISTS]
    seeds += [WET_PATHS, GUTENBERG_CATALOG, "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"]
    for url in seeds:
        status, info = probe(url)
        good = status in (200, 206)
        ok = ok and good
        print(f"{'ok' if good else 'FAIL'} {status} {info[:40]} {url}")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Stoppable corpus download toward ~1 TB on D:")
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--check", action="store_true", help="Probe seed URLs and exit")
    parser.add_argument("--jobs", type=int, default=8, help="Files downloading at once")
    args = parser.parse_args()
    if args.check:
        return check()
    root: Path = args.root
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    state = load_state(state_path)
    target = sum(BUDGET.values())
    already = sum(used_bytes(root, category) for category in BUDGET)
    progress = Progress(target, already)
    print(f"downloading to {root}", flush=True)
    print(f"target {target / GB:.0f} GB   already {already / GB:.2f} GB", flush=True)
    print("one file at a time", flush=True)
    try:
        take(root, "books", "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
             root / "books" / "gutenberg" / "pg1342.txt", state, state_path, progress)
        for category, list_url in LISTS:
            from_list(root, category, list_url, state, state_path, progress)
        for category, rel, url in DIRECT:
            if rel.endswith("enwiki-pages-articles-multistream.xml.bz2"):
                continue
            take(root, category, url, root / category / rel, state, state_path, progress)
        gutenberg(root, state, state_path, progress)
        wet(root, state, state_path, progress)
        for category, rel, url in DIRECT:
            if not rel.endswith("enwiki-pages-articles-multistream.xml.bz2"):
                continue
            take(root, category, url, root / category / rel, state, state_path, progress)
    except KeyboardInterrupt:
        save_state(state_path, state)
        print("\nstopped. run the same command to continue.", file=sys.stderr)
        return 130
    save_state(state_path, state)
    print("pass finished")
    for category, budget in BUDGET.items():
        have = used_bytes(root, category)
        print(f"{category} {have / GB:.1f} GB / {budget / GB:.0f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
