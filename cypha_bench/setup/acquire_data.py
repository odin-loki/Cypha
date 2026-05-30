#!/usr/bin/env python3
"""
Cross-platform dataset downloader for cypha_bench.

Run from anywhere:
    python cypha_bench/setup/acquire_data.py

Skips files that already exist. Uses requests with urllib fallback.
"""

from __future__ import annotations

import gzip
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

try:
    import requests
except ImportError:
    requests = None

BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(BENCH_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT.parent))

from cypha_bench.common.paths import DATA_DIR

MNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

def _ensure_dirs() -> None:
    for name in (
        "mnist",
        "wikitext2",
        "nsl_kdd",
        "ecg5000",
        "canterbury",
        "gutenberg",
        "chess",
        "financial",
        "uci",
    ):
        (DATA_DIR / name).mkdir(parents=True, exist_ok=True)


def _download_url(url: str, dest: Path, force: bool = False) -> bool:
    if dest.exists() and not force:
        print(f"  skip (exists): {dest.relative_to(BENCH_ROOT)}")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  download: {url}")
    print(f"       -> {dest.relative_to(BENCH_ROOT)}")

    data: bytes
    if requests is not None:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
        data = resp.content
    else:
        with urlopen(url, timeout=120) as resp:
            data = resp.read()

    dest.write_bytes(data)
    return True


def _valid_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path, "r") as zf:
            return zf.testzip() is None
    except zipfile.BadZipFile:
        return False


def _gunzip_inplace(gz_path: Path) -> None:
    out_path = gz_path.with_suffix("")
    if out_path.exists():
        print(f"  skip (exists): {out_path.relative_to(BENCH_ROOT)}")
        return
    with gzip.open(gz_path, "rb") as f_in:
        out_path.write_bytes(f_in.read())
    print(f"  extracted: {out_path.relative_to(BENCH_ROOT)}")


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            target = dest_dir / member
            if target.exists():
                continue
            zf.extract(member, dest_dir)
    print(f"  extracted zip -> {dest_dir.relative_to(BENCH_ROOT)}")


def _extract_tar_gz(tar_path: Path, dest_dir: Path) -> None:
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(dest_dir, filter="data")
    print(f"  extracted tar -> {dest_dir.relative_to(BENCH_ROOT)}")


def acquire_mnist() -> None:
    print("\n[MNIST]")
    mnist_dir = DATA_DIR / "mnist"
    for fname in MNIST_FILES:
        url = MNIST_BASE + fname
        gz_path = mnist_dir / fname
        _download_url(url, gz_path)
        _gunzip_inplace(gz_path)


def acquire_wikitext2() -> None:
    print("\n[WikiText-2]")
    wt_dir = DATA_DIR / "wikitext2"
    zip_path = wt_dir / "wikitext-2-v1.zip"
    token_dir = wt_dir / "wikitext-2"
    if token_dir.exists() and (token_dir / "wiki.train.tokens").exists():
        print("  skip (exists): wikitext-2/wiki.train.tokens")
        return
    urls = [
        "https://blog.salesforceairesearch.com/wp-content/uploads/2019/09/wikitext-2-v1.zip",
        "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip",
    ]
    if zip_path.exists() and not _valid_zip(zip_path):
        print("  removing corrupt zip")
        zip_path.unlink()
    if not zip_path.exists() or not _valid_zip(zip_path):
        for url in urls:
            try:
                _download_url(url, zip_path, force=True)
                if _valid_zip(zip_path):
                    break
                print(f"  invalid zip from {url}, trying next mirror")
                zip_path.unlink(missing_ok=True)
            except Exception as exc:
                print(f"  failed {url}: {exc}")
    if zip_path.exists() and _valid_zip(zip_path):
        _extract_zip(zip_path, wt_dir)


def acquire_nsl_kdd() -> None:
    print("\n[NSL-KDD]")
    kdd_dir = DATA_DIR / "nsl_kdd"
    base = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
    _download_url(base + "KDDTrain+.txt", kdd_dir / "KDDTrain+.txt")
    _download_url(base + "KDDTest+.txt", kdd_dir / "KDDTest+.txt")


def acquire_ecg5000() -> None:
    print("\n[ECG5000]")
    ecg_dir = DATA_DIR / "ecg5000"
    train = ecg_dir / "ECG5000_TRAIN.txt"
    if train.exists():
        print(f"  skip (exists): {train.relative_to(BENCH_ROOT)}")
        return
    zip_path = ecg_dir / "ECG5000.zip"
    try:
        _download_url(
            "http://www.timeseriesclassification.com/Downloads/ECG5000.zip",
            zip_path,
        )
        if zip_path.exists() and _valid_zip(zip_path):
            _extract_zip(zip_path, ecg_dir)
        else:
            print("  ECG5000 zip unavailable — domains will use synthetic fallback")
            zip_path.unlink(missing_ok=True)
    except Exception as exc:
        print(f"  ECG5000 skipped: {exc}")


def acquire_canterbury() -> None:
    print("\n[Canterbury Corpus]")
    cb_dir = DATA_DIR / "canterbury"
    tar_path = cb_dir / "cantrbry.tar.gz"
    if _download_url(
        "http://corpus.canterbury.ac.nz/resources/cantrbry.tar.gz",
        tar_path,
    ) or not (cb_dir / "alice29.txt").exists():
        if tar_path.exists():
            _extract_tar_gz(tar_path, cb_dir)


def acquire_gutenberg() -> None:
    print("\n[Gutenberg]")
    gut_dir = DATA_DIR / "gutenberg"
    books = [
        (
            "moby_dick.txt",
            "https://www.gutenberg.org/files/2701/2701-0.txt",
        ),
        (
            "sherlock_holmes.txt",
            "https://www.gutenberg.org/files/1661/1661-0.txt",
        ),
        (
            "alice.txt",
            "https://www.gutenberg.org/files/11/11-0.txt",
        ),
    ]
    for fname, url in books:
        _download_url(url, gut_dir / fname)


def acquire_chess() -> None:
    print("\n[Chess PGN]")
    chess_dir = DATA_DIR / "chess"
    zip_path = chess_dir / "Kasparov.zip"
    pgn_path = chess_dir / "Kasparov.pgn"

    if pgn_path.exists():
        print(f"  skip (exists): {pgn_path.relative_to(BENCH_ROOT)}")
        return

    if _download_url(
        "https://www.pgnmentor.com/players/Kasparov.zip",
        zip_path,
    ) or not pgn_path.exists():
        if zip_path.exists():
            _extract_zip(zip_path, chess_dir)
            extracted = chess_dir / "Kasparov.pgn"
            if extracted.exists() and not pgn_path.exists():
                shutil.move(str(extracted), str(pgn_path))


def main() -> None:
    print(f"Data directory: {DATA_DIR}")
    _ensure_dirs()

    steps = [
        acquire_mnist,
        acquire_wikitext2,
        acquire_nsl_kdd,
        acquire_ecg5000,
        acquire_canterbury,
        acquire_gutenberg,
        acquire_chess,
    ]
    for step in steps:
        try:
            step()
        except Exception as exc:
            print(f"  WARNING: {step.__name__} failed: {exc}")

    print("\nData acquisition complete.")
    print("20 Newsgroups will be auto-downloaded by sklearn on first run (~14MB).")
    print("Financial data will be pulled live by yfinance on first run (tiny).")


if __name__ == "__main__":
    main()
