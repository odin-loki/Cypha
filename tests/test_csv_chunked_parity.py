"""CSVDataset.from_file chunked vs full-buffer parity."""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from cypha_studio.core.dataset import CSVDataset


def _write_csv(path: Path, n: int) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["f0", "f1", "f2", "y"])
        for i in range(n):
            w.writerow([i * 0.1, i * 0.2, i * 0.3, str(i % 4)])


def test_csv_chunked_matches_full_load():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "d.csv"
        _write_csv(p, 55)
        a = CSVDataset.from_file(p)
        for chunk in (7, 20, 1000):
            b = CSVDataset.from_file(p, read_chunk_rows=chunk)
            np.testing.assert_array_equal(a.X, b.X)
            np.testing.assert_array_equal(a.y, b.y)
            assert a.feature_names == b.feature_names
