#!/usr/bin/env python3
"""
Regenerate parity_fixtures/ for native-port regression testing.

Run from repo root:
  python scripts/generate_parity_fixtures.py

Commits should include updated manifest.json, reference.cypha, expected.npz
when inference numerics change intentionally.
"""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from Cypha import (
    CyphaDIF,
    VectorEncoder,
    _BESSEL_TABLES_OK,
    _EPS,
    _REPLAY_CAP,
    _REPLAY_RATIO,
    _ALIGN_EVERY,
    _softmax_batch,
    cypha_load_binary,
    cypha_save_binary,
)

_SEED_TRAIN = 424242
_SEED_EVAL = 424243
_INPUT_DIM = 8
_FIELD_DIM = 24
_N_TRAIN = 120
_FIXTURE_DIR = _ROOT / "parity_fixtures"


def _train_reference_clf() -> CyphaDIF:
    rng = np.random.default_rng(_SEED_TRAIN)
    offs = {
        "0": np.array([2.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
        "1": np.array([0, 2.0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
        "2": np.array([0, 0, 2.0, 0, 0, 0, 0, 0], dtype=np.float64),
    }
    clf = CyphaDIF(
        encoder=VectorEncoder(_INPUT_DIM),
        field_dim=_FIELD_DIM,
        rng=np.random.default_rng(_SEED_TRAIN),
    )
    for _ in range(_N_TRAIN):
        lbl = str(rng.integers(0, 3))
        x = rng.normal(0, 0.35, _INPUT_DIM) + offs[lbl]
        clf.train_step(x, lbl)
    return clf


def _write_train_step_vector_sidecar(fixture_dir: Path, state_rt: dict, x_row: np.ndarray) -> None:
    """One `train_step` loss for `native/tools/train_step_vector_parity.cpp`."""
    clf = CyphaDIF(
        encoder=VectorEncoder(_INPUT_DIM),
        field_dim=_FIELD_DIM,
        rng=np.random.default_rng(0),
    )
    clf.load_state(state_rt)
    x_ts = np.ascontiguousarray(x_row, dtype=np.float64).copy()
    ts_before = int(clf._total_steps)
    hp = {
        "world_lr": float(clf.world_lr),
        "delta_lr": float(clf.delta_lr),
        "ood_sigma": float(clf.ood_sigma),
        "enc_lr": float(clf.enc_lr),
        "replay_ratio": float(_REPLAY_RATIO),
        "replay_cap": int(_REPLAY_CAP),
        "align_every": int(_ALIGN_EVERY),
        "temp_recalib_every": 0,
    }
    loss_ts = float(clf.train_step(x_ts, "0"))
    ts_dir = fixture_dir / "train_step_vector"
    ts_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "x": x_ts.tolist(),
        "label": "0",
        "expected_loss": loss_ts,
        "total_steps_before": ts_before,
        **hp,
    }
    (ts_dir / "sidecar.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    if not _BESSEL_TABLES_OK:
        raise SystemExit(
            "Cypha Bessel lookup tables are unavailable (need scipy or bessel_ratios.npz next to Cypha.py). "
            "Run: python scripts/export_bessel_ratios_npz.py  (requires scipy once), or pip install scipy."
        )
    _FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    trained = _train_reference_clf()
    cypha_path = _FIXTURE_DIR / "reference.cypha"
    cypha_save_binary(trained.save_state(), str(cypha_path))
    # Expectations must use the same bytes as reference.cypha (round-trip).
    state_rt = cypha_load_binary(str(cypha_path))
    clf = CyphaDIF(
        encoder=VectorEncoder(_INPUT_DIM),
        field_dim=_FIELD_DIM,
        rng=np.random.default_rng(0),
    )
    clf.load_state(state_rt)

    rng = np.random.default_rng(_SEED_EVAL)
    N = 7
    x_input = rng.normal(0, 0.4, (N, _INPUT_DIM)).astype(np.float64)
    # Mix in structured points near each class
    x_input[0] = np.array([1.8, 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    x_input[1] = np.array([0.1, 1.7, 0.2, 0, 0, 0, 0, 0], dtype=np.float64)
    x_input[2] = np.array([0, 0.1, 1.9, 0, 0, 0, 0, 0], dtype=np.float64)

    H = clf.batch_encode([x_input[i] for i in range(N)])
    LLR, labels = clf.score_matrix(H, use_field=True)
    T = float(clf.temperature)
    probs = _softmax_batch(LLR / (T + _EPS))
    gates = clf.world_gate_vector(H, use_field=True)
    best_idx = probs.argmax(axis=1)
    conf_batch = probs[np.arange(N), best_idx] * gates

    serial_conf = np.empty(N, dtype=np.float64)
    pred_idx = np.empty(N, dtype=np.int32)
    label_to_i = {lb: i for i, lb in enumerate(labels)}
    for i in range(N):
        pred, conf = clf.infer(x_input[i])
        serial_conf[i] = conf
        pred_idx[i] = label_to_i[pred]

    np.savez_compressed(
        _FIXTURE_DIR / "expected.npz",
        x_input=x_input,
        H=H,
        llr=LLR,
        probs=probs,
        gates=gates,
        conf_batch=conf_batch,
        pred_idx=pred_idx,
        serial_conf=serial_conf,
        temperature=np.array([T], dtype=np.float64),
        eps=np.array([_EPS], dtype=np.float64),
    )

    _write_train_step_vector_sidecar(_FIXTURE_DIR, state_rt, x_input[0])

    # Native M1 harness (see native/README.md): F_field is runtime state not in .cypha v3.
    F_field = np.ascontiguousarray(clf.memory.world.F_field, dtype=np.float64)
    xs_list = [x_input[i] for i in range(N)]
    bif_rows = clf.batch_infer_full(xs_list, use_field=True)
    bif_entropy = np.array([float(r["entropy"]) for r in bif_rows], dtype=np.float64)
    bif_conf = np.array([float(r["confidence"]) for r in bif_rows], dtype=np.float64)
    np.testing.assert_allclose(bif_conf, conf_batch, rtol=0, atol=1e-12)

    n_bin = struct.pack(
        "<8sIIIIIdd",
        b"CYPHNP01",
        2,
        int(N),
        int(_INPUT_DIM),
        int(len(labels)),
        int(_FIELD_DIM),
        float(T),
        float(_EPS),
    )
    (_FIXTURE_DIR / "native_parity.bin").write_bytes(
        n_bin
        + F_field.tobytes(order="C")
        + np.ascontiguousarray(x_input, dtype=np.float64).tobytes(order="C")
        + np.ascontiguousarray(LLR, dtype=np.float64).tobytes(order="C")
        + np.ascontiguousarray(probs, dtype=np.float64).tobytes(order="C")
        + np.ascontiguousarray(gates, dtype=np.float64).tobytes(order="C")
        + bif_entropy.tobytes(order="C")
        + bif_conf.tobytes(order="C")
    )

    manifest = {
        "fixture_schema": 2,
        "generator": "scripts/generate_parity_fixtures.py",
        "seed_train": _SEED_TRAIN,
        "seed_eval": _SEED_EVAL,
        "n_train_steps": _N_TRAIN,
        "model": {
            "type": "CyphaDIF",
            "encoder": "VectorEncoder",
            "input_dim": _INPUT_DIM,
            "field_dim": _FIELD_DIM,
        },
        "labels": labels,
        "cypha_version_note": "Must match Cypha.py _CYPHA_VERSION after load",
    }
    (_FIXTURE_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    train_hp = {
        "world_lr": float(trained.world_lr),
        "delta_lr": float(trained.delta_lr),
        "ood_sigma": float(trained.ood_sigma),
        "enc_lr": float(trained.enc_lr),
        "replay_ratio": float(_REPLAY_RATIO),
        "replay_cap": int(_REPLAY_CAP),
        "align_every": int(_ALIGN_EVERY),
        "temp_recalib_every": 0,
    }
    (_FIXTURE_DIR / "train_hparams.json").write_text(
        json.dumps(train_hp, indent=2), encoding="utf-8"
    )

    readme = _FIXTURE_DIR / "README.md"
    readme.write_text(
        "# Parity fixtures\n\n"
        "Generated by `python scripts/generate_parity_fixtures.py`.\n\n"
        "- `reference.cypha` — `cypha_save_binary` snapshot (includes Tier-1 `ctx_*`, `field_W_T`, `w_inject` when feat_dim≠field_dim)\n"
        "- `expected.npz` — numeric targets for `score_matrix` / `infer` / `batch_infer`\n"
        "- `native_parity.bin` — `F_field` + tensors for the C++ `cypha_parity` tool (v2: + `batch_infer_full` entropy & confidence tail)\n"
        "- `manifest.json` — `fixture_schema`, seeds, geometry, label order\n"
        "- `train_hparams.json` — training LRs + `align_every` / `temp_recalib_every` for `cypha_rest` /update\n"
        "- `train_step_vector/sidecar.json` — one `dif_train_step_vector` loss vs native `train_step_vector_parity`\n"
        "- `preprocessor_fit/` + `preprocessor_fit_no_scale/` — `design.json` + `expected_preprocessor.json` + `probe.json` for native `PreprocessorState::fit_from_design_matrix` (scale on/off + PCA; from `generate_preprocessor_fit_fixture.py`)\n"
        "- `csv_ingest/` — `cases.json` + CSVs + expected JSON for native `load_csv_dense` vs `CSVDataset.from_file` (names/indices, multiline quotes, `generate_csv_ingest_fixture.py`)\n"
        "- `dif_regressor_train_step/` — `before.cypha` + `f_field.json` + `sidecar.json` for native `DIFRegressor` (cold hash + warm LLR routing + `replay_u01`; `generate_dif_regressor_train_step_fixture.py`)\n"
        "- `batch_llr/sidecar.json` — `x_input` + `llr` slice for native `batch_llr_parity` (also from `generate_batch_llr_fixture.py`)\n"
        "- `quantile_dif_train/` — `before.cypha` + `f_field.json` + `sidecar.json` for native `quantile_dif_train_parity` (from `generate_quantile_dif_train_fixture.py`)\n"
        "- `dif_train_replay/` — same tool + `replay_u01` recorded stream for `replay_ratio>0` (from `generate_dif_train_replay_fixture.py`)\n"
        "- `studio_trainer_classify_hotpath/` — `Trainer.fit`-order online loop + `enc_lr>0` + `replay_u01` (from `generate_studio_trainer_classify_hotpath_fixture.py`)\n"
        "- `studio_trainer_gh_classify_hotpath/` — `Trainer.fit` order + `gh_train_step` + threaded `chi`/`psi` (from `generate_studio_trainer_gh_classify_hotpath_fixture.py`)\n"
        "- `studio_trainer_preprocess_classify_hotpath/` — `Preprocessor` + `train_step` on transformed rows (from `generate_studio_trainer_preprocess_classify_hotpath_fixture.py`)\n"
        "- `csv_preprocess_classify_hotpath/` — same goldens via `train.csv` + `load_csv_dense` (`generate_csv_preprocess_classify_hotpath_fixture.py`; run preprocess hotpath generator first)\n"
        "- `studio_trainer_preprocess_gh_classify_hotpath/` — identity `Preprocessor` + GH goldens from `studio_trainer_gh_classify_hotpath/` (`generate_studio_trainer_preprocess_gh_classify_hotpath_fixture.py`; run GH generator first)\n"
        "- `mke_train_step/` — one `MKERegressor.train_step` vs `mke_train_step_parity` (from `generate_mke_train_step_fixture.py`)\n"
        "- `mke_train_extended/` — multi-step `MKERegressor.train_step` + `replay_u01` / `replay_warmup` (from `generate_mke_train_extended_fixture.py`)\n"
        "- `regression_head.json` — optional scalar MoE targets per class label; "
        "see `docs/port/schemas/regression_head.schema.json`\n",
        encoding="utf-8",
    )

    print(
        f"Wrote {_FIXTURE_DIR}/ (reference.cypha, expected.npz, native_parity.bin, manifest.json, "
        "train_hparams.json, train_step_vector/, batch_llr/, quantile_dif_train/, dif_train_replay/, "
        "studio_trainer_classify_hotpath/, studio_trainer_gh_classify_hotpath/, "
        "studio_trainer_preprocess_classify_hotpath/, csv_preprocess_classify_hotpath/, studio_trainer_preprocess_gh_classify_hotpath/, "
        "mke_train_step/, mke_train_extended/)"
    )

    import subprocess

    sub_env = {**os.environ, "PYTHONPATH": str(_ROOT)}

    batch_script = _ROOT / "scripts" / "generate_batch_llr_fixture.py"
    if batch_script.is_file():
        subprocess.run([sys.executable, str(batch_script)], check=True, cwd=str(_ROOT), env=sub_env)

    pre_fit_script = _ROOT / "scripts" / "generate_preprocessor_fit_fixture.py"
    if pre_fit_script.is_file():
        subprocess.run([sys.executable, str(pre_fit_script)], check=True, cwd=str(_ROOT), env=sub_env)

    csv_script = _ROOT / "scripts" / "generate_csv_ingest_fixture.py"
    if csv_script.is_file():
        subprocess.run([sys.executable, str(csv_script)], check=True, cwd=str(_ROOT), env=sub_env)

    dif_reg_script = _ROOT / "scripts" / "generate_dif_regressor_train_step_fixture.py"
    if dif_reg_script.is_file():
        subprocess.run([sys.executable, str(dif_reg_script)], check=True, cwd=str(_ROOT), env=sub_env)

    q_script = _ROOT / "scripts" / "generate_quantile_dif_train_fixture.py"
    if q_script.is_file():
        subprocess.run([sys.executable, str(q_script)], check=True, cwd=str(_ROOT), env=sub_env)

    dr_script = _ROOT / "scripts" / "generate_dif_train_replay_fixture.py"
    if dr_script.is_file():
        subprocess.run([sys.executable, str(dr_script)], check=True, cwd=str(_ROOT), env=sub_env)

    sthp_script = _ROOT / "scripts" / "generate_studio_trainer_classify_hotpath_fixture.py"
    if sthp_script.is_file():
        subprocess.run([sys.executable, str(sthp_script)], check=True, cwd=str(_ROOT), env=sub_env)

    stgh_script = _ROOT / "scripts" / "generate_studio_trainer_gh_classify_hotpath_fixture.py"
    if stgh_script.is_file():
        subprocess.run([sys.executable, str(stgh_script)], check=True, cwd=str(_ROOT), env=sub_env)

    stpr_script = _ROOT / "scripts" / "generate_studio_trainer_preprocess_classify_hotpath_fixture.py"
    if stpr_script.is_file():
        subprocess.run([sys.executable, str(stpr_script)], check=True, cwd=str(_ROOT), env=sub_env)

    csv_prep_script = _ROOT / "scripts" / "generate_csv_preprocess_classify_hotpath_fixture.py"
    if csv_prep_script.is_file():
        subprocess.run([sys.executable, str(csv_prep_script)], check=True, cwd=str(_ROOT), env=sub_env)

    stprgh_script = _ROOT / "scripts" / "generate_studio_trainer_preprocess_gh_classify_hotpath_fixture.py"
    if stprgh_script.is_file():
        subprocess.run([sys.executable, str(stprgh_script)], check=True, cwd=str(_ROOT), env=sub_env)

    mke_script = _ROOT / "scripts" / "generate_mke_train_step_fixture.py"
    if mke_script.is_file():
        subprocess.run([sys.executable, str(mke_script)], check=True, cwd=str(_ROOT), env=sub_env)

    mke_ext_script = _ROOT / "scripts" / "generate_mke_train_extended_fixture.py"
    if mke_ext_script.is_file():
        subprocess.run([sys.executable, str(mke_ext_script)], check=True, cwd=str(_ROOT), env=sub_env)


if __name__ == "__main__":
    main()
