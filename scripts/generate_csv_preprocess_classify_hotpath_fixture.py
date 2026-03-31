# Emit ``parity_fixtures/csv_preprocess_classify_hotpath/`` from
# ``studio_trainer_preprocess_classify_hotpath/`` (same goldens; rows from ``train.csv``).
from __future__ import annotations

import copy
import csv
import json
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "parity_fixtures" / "studio_trainer_preprocess_classify_hotpath"
_OUT = _ROOT / "parity_fixtures" / "csv_preprocess_classify_hotpath"


def _fmt_float(x: float) -> str:
    return format(x, ".17g")


def main() -> int:
    _OUT.mkdir(parents=True, exist_ok=True)
    required = ("sidecar.json", "preprocessor.json", "f_field.json", "before.cypha")
    for name in required:
        p = _SRC / name
        if not p.is_file():
            print(f"missing {p}; run scripts/generate_studio_trainer_preprocess_classify_hotpath_fixture.py", file=sys.stderr)
            return 1
        shutil.copy2(p, _OUT / name)

    data = json.loads((_SRC / "sidecar.json").read_text(encoding="utf-8"))
    steps = data["steps"]
    d_raw = int(data["d_raw"])

    train_path = _OUT / "train.csv"
    with train_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"f{i}" for i in range(d_raw)] + ["label"])
        for st in steps:
            xs = [_fmt_float(float(v)) for v in st["x_raw"]]
            w.writerow(xs + [st["label"]])

    new_side = copy.deepcopy(data)
    del new_side["steps"]
    new_side["csv"] = "train.csv"
    new_side["csv_spec"] = {
        "has_header": True,
        "delimiter": ",",
        "target_col_name": "label",
        "feature_col_names": [f"f{i}" for i in range(d_raw)],
    }
    prev = new_side.get("description", "")
    new_side["description"] = (prev + " CSV ingest path: train.csv + csv_spec → native load_csv_dense (csv_preprocess_classify_hotpath).").strip()

    (_OUT / "sidecar.json").write_text(json.dumps(new_side, indent=2), encoding="utf-8")
    print(f"Wrote {_OUT}/ (sidecar.json, train.csv, copied assets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
