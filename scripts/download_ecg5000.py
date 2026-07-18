#!/usr/bin/env python3
"""Download UCR ECG5000 (.ts from Zenodo) and convert to loader .txt format.

Writes:
  bench/data/ecg5000/ECG5000_{TRAIN,TEST}.ts
  bench/data/ecg5000/ECG5000_{TRAIN,TEST}.txt

.txt layout matches native load_ecg_file: ``<label> <v0> <v1> ... <v139>``.
Source: Zenodo record 11186692 (UCR / timeseriesclassification.com archive).
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

ZENODO_API = "https://zenodo.org/api/records/11186692"


def ts_to_ucr_txt(ts_path: Path, txt_path: Path) -> int:
    lines_out: list[str] = []
    for raw in ts_path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if not s or s.startswith("@") or s.startswith("#"):
            continue
        if ":" not in s:
            continue
        body, label = s.rsplit(":", 1)
        vals = [v.strip() for v in body.split(",") if v.strip()]
        if len(vals) < 2:
            continue
        lines_out.append(label.strip() + " " + " ".join(vals))
    txt_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return len(lines_out)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "bench" / "data" / "ecg5000"
    out.mkdir(parents=True, exist_ok=True)
    rec = json.load(urllib.request.urlopen(ZENODO_API))
    for f in rec["files"]:
        key = f["key"]
        url = f["links"]["self"]
        dest = out / key
        print(f"GET {key}")
        dest.write_bytes(urllib.request.urlopen(url).read())
        print(f"  wrote {dest} ({dest.stat().st_size} bytes)")
    n_tr = ts_to_ucr_txt(out / "ECG5000_TRAIN.ts", out / "ECG5000_TRAIN.txt")
    n_te = ts_to_ucr_txt(out / "ECG5000_TEST.ts", out / "ECG5000_TEST.txt")
    print(f"converted TRAIN={n_tr} TEST={n_te} -> {out}")
    print("Run: cypha_bench_run --domain-tag d10  (expect data_source=ecg5000)")


if __name__ == "__main__":
    main()
