#!/usr/bin/env python3
"""
Emit ``parity_fixtures/studio_trainer_preprocess_gh_classify_hotpath/`` for
``preprocess_train_classify_parity``.

Copies ``parity_fixtures/studio_trainer_gh_classify_hotpath/`` (``before.cypha``,
``f_field.json``, numeric sidecar) and rewrites ``steps`` to use ``x_raw`` (same
vectors as GH ``x``). Adds an **identity** ``Preprocessor`` (``scale=False``, no
PCA) so ``transform_one(x_raw)`` reproduces the GH latent rows. Goldens stay
identical to the GH CTest so native parity is inherited.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cypha_studio.core.dataset import Preprocessor

_OUT = _ROOT / "parity_fixtures" / "studio_trainer_preprocess_gh_classify_hotpath"
_GH = _ROOT / "parity_fixtures" / "studio_trainer_gh_classify_hotpath"


def main() -> None:
    gh_side = _GH / "sidecar.json"
    if not gh_side.is_file():
        raise SystemExit(f"missing {gh_side} — run generate_studio_trainer_gh_classify_hotpath_fixture.py")

    j = json.loads(gh_side.read_text(encoding="utf-8"))
    d_in = int(j["d_in"])
    rows = [np.asarray(st["x"], dtype=np.float64) for st in j["steps"]]
    X_fit = np.stack(rows, axis=0)
    pre = Preprocessor(scale=False, pca_dim=None)
    pre.fit(X_fit)
    if int(pre.output_dim or 0) != d_in:
        raise SystemExit("identity preprocessor output_dim mismatch")

    steps_out = [{"x_raw": st["x"], "label": st["label"]} for st in j["steps"]]

    j_out = dict(j)
    j_out["description"] = (
        "GH hotpath goldens (studio_trainer_gh_classify_hotpath) via identity Preprocessor + x_raw steps"
    )
    j_out["d_raw"] = d_in
    j_out["steps"] = steps_out

    _OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(_GH / "before.cypha", _OUT / "before.cypha")
    shutil.copy(_GH / "f_field.json", _OUT / "f_field.json")
    (_OUT / "preprocessor.json").write_text(json.dumps(pre.save_state(), indent=2), encoding="utf-8")
    (_OUT / "sidecar.json").write_text(json.dumps(j_out, indent=2), encoding="utf-8")
    print(f"Wrote {_OUT} (n_steps={len(steps_out)}, use_gh; goldens from studio_trainer_gh_classify_hotpath)")


if __name__ == "__main__":
    main()
