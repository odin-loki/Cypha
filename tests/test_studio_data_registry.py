"""Studio core: CSV dataset loader and model registry (no Qt)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def test_csv_dataset_from_file_stats(tmp_path):
    from cypha_studio.core.dataset import CSVDataset

    p = tmp_path / "tiny.csv"
    p.write_text(
        "f0,f1,f2,target\n"
        "0.0,0.1,0.2,cat_a\n"
        "1.0,1.1,1.2,cat_b\n"
        "2.0,2.1,2.2,cat_a\n",
        encoding="utf-8",
    )
    ds = CSVDataset.from_file(str(p))
    assert ds.n_samples == 3
    assert ds.n_features == 3
    st = ds.stats()
    assert st.n_classes >= 2
    assert st.n_samples == 3
    assert "cat_a" in st.class_counts


def _registry_register_executable() -> Path | None:
    env = os.environ.get("REGISTRY_REGISTER_BIN", "").strip()
    if env:
        p = Path(env)
        if p.is_file():
            return p
    if sys.platform == "win32":
        candidates = [
            _ROOT / "native" / "build-mingw-w64" / "registry_register.exe",
            _ROOT / "native" / "build" / "Release" / "registry_register.exe",
            _ROOT / "native" / "build" / "Debug" / "registry_register.exe",
            _ROOT / "native" / "build" / "registry_register.exe",
        ]
    else:
        candidates = [_ROOT / "native" / "build" / "registry_register"]
    for p in candidates:
        if p.is_file():
            return p
    return None


def test_native_registry_register_tool_visible_to_model_registry(tmp_path):
    """Native ``registry_register`` copies artifacts into the same tree ``ModelRegistry`` scans."""
    exe = _registry_register_executable()
    if exe is None:
        pytest.skip("registry_register not built (set REGISTRY_REGISTER_BIN or build native/)")
    fix = _ROOT / "parity_fixtures"
    if not (fix / "reference.cypha").is_file() or not (fix / "registry_register" / "card.json").is_file():
        pytest.skip("parity fixtures missing")

    root = tmp_path / "reg_root"
    name, version = "native_tool_smoke", "9.9.9"
    cmd = [
        str(exe),
        str(root),
        name,
        version,
        str(fix / "reference.cypha"),
        str(fix / "registry_register" / "card.json"),
        "--overwrite",
        "--and-verify",
    ]
    subprocess.run(cmd, check=True, cwd=str(_ROOT))
    from cypha_studio.core.registry import ModelRegistry

    reg = ModelRegistry(str(root))
    assert reg.exists(name, version)
    assert reg.registered_entry_count() >= 1


def test_registry_register_list_load_state_roundtrip(tmp_path):
    """Disk round-trip; reconstructed model uses a fresh RNG so we compare `save_state` arrays."""
    from Cypha import CyphaDIF, VectorEncoder
    from cypha_studio.core.registry import ModelCard, ModelRegistry

    rng = np.random.default_rng(7)
    clf = CyphaDIF(VectorEncoder(4), field_dim=40, rng=rng)
    for i in range(24):
        clf.train_step(rng.standard_normal(4), str(i % 2))

    reg = ModelRegistry(str(tmp_path))
    card = ModelCard(
        name="pytest-roundtrip",
        version="0.0.1",
        task="classification",
        model_type="CyphaDIF",
        encoder_type="VectorEncoder",
        input_dim=4,
    )
    reg.register(clf, card)
    assert reg.list_model_names() == ["pytest-roundtrip"]
    assert reg.registered_versions("pytest-roundtrip") == ["0.0.1"]
    assert reg.registered_entry_count() == 1
    assert list(reg.iter_registered_pairs()) == [("pytest-roundtrip", "0.0.1")]
    listed = reg.list_models()
    assert len(listed) == 1
    assert listed[0].name == "pytest-roundtrip"
    assert listed[0].version == "0.0.1"

    loaded, pre, c2 = reg.load("pytest-roundtrip", "0.0.1")
    assert c2.name == "pytest-roundtrip"
    assert pre is None
    s0, s1 = clf.save_state(), loaded.save_state()
    assert set(s0.keys()) == set(s1.keys())
    for k in s0:
        a, b = s0[k], s1[k]
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_registry_dif_regressor_roundtrip(tmp_path):
    """``DIFRegressor`` + ``cypha_save_binary`` round-trip via ``ModelRegistry`` (state dict parity)."""
    from Cypha import DIFRegressor, VectorEncoder
    from cypha_studio.core.registry import ModelCard, ModelRegistry

    rng = np.random.default_rng(11)
    reg_m = DIFRegressor(VectorEncoder(3), field_dim=32, n_experts=4, rng=rng)
    for i in range(48):
        reg_m.train_step(rng.standard_normal(3), float(rng.normal(0, 0.5)))

    rdir = ModelRegistry(str(tmp_path))
    card = ModelCard(
        name="difreg-rt",
        version="1.0.0",
        task="regression",
        model_type="DIFRegressor",
        encoder_type="VectorEncoder",
        input_dim=3,
        field_dim=32,
    )
    rdir.register(reg_m, card)
    loaded, pre, c2 = rdir.load("difreg-rt", "1.0.0")
    assert pre is None and c2.model_type == "DIFRegressor"
    s0, s1 = reg_m.save_state(), loaded.save_state()

    def _assert_state_leaves_equal(x, y, path: str) -> None:
        if isinstance(x, dict):
            assert set(x.keys()) == set(y.keys()), path
            for kk in x:
                _assert_state_leaves_equal(x[kk], y[kk], f"{path}.{kk}")
        elif isinstance(x, np.ndarray):
            np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-6, err_msg=path)
        else:
            assert x == y, path

    _assert_state_leaves_equal(s0, s1, "difreg")

    xq = rng.standard_normal(3)
    y0, u0 = reg_m.predict(xq)
    y1, u1 = loaded.predict(xq)
    np.testing.assert_allclose(y0, y1, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(u0, u1, rtol=1e-5, atol=1e-5)


def _assert_nested_state_equal(x, y, path: str = "root") -> None:
    if isinstance(x, dict):
        assert set(x.keys()) == set(y.keys()), path
        for kk in x:
            _assert_nested_state_equal(x[kk], y[kk], f"{path}.{kk}")
    elif isinstance(x, np.ndarray):
        np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-6, err_msg=path)
    else:
        assert x == y, path


def test_registry_rff_regressor_roundtrip(tmp_path):
    from Cypha import RFFRegressor
    from cypha_studio.core.registry import ModelCard, ModelRegistry

    rng = np.random.default_rng(3)
    X = rng.standard_normal((40, 3))
    y = X[:, 0] * 0.5 + rng.normal(0, 0.1, size=40)
    reg = RFFRegressor(D=48, seed=9)
    reg.fit(X, y)
    reg.train_step(rng.standard_normal(3), float(rng.normal()))

    rdir = ModelRegistry(str(tmp_path))
    card = ModelCard(
        name="rff-rt",
        version="0.1.0",
        task="regression",
        model_type="RFFRegressor",
        encoder_type="RFFEncoder",
        input_dim=3,
    )
    rdir.register(reg, card)
    loaded, pre, c2 = rdir.load("rff-rt", "0.1.0")
    assert pre is None and c2.model_type == "RFFRegressor"
    _assert_nested_state_equal(reg.save_state(), loaded.save_state(), "rff")

    xq = rng.standard_normal((5, 3))
    np.testing.assert_allclose(reg.predict(xq), loaded.predict(xq), rtol=1e-5, atol=1e-5)
    y0, v0 = reg.predict_with_uncertainty(xq)
    y1, v1 = loaded.predict_with_uncertainty(xq)
    np.testing.assert_allclose(y0, y1, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(v0, v1, rtol=1e-5, atol=1e-5)


def test_registry_two_stage_dif_regressor_roundtrip(tmp_path):
    from Cypha import TwoStageDIFRegressor
    from cypha_studio.core.registry import ModelCard, ModelRegistry

    rng = np.random.default_rng(5)
    X = rng.standard_normal((60, 2))
    y = np.sin(X[:, 0]) + 0.1 * rng.standard_normal(60)
    reg = TwoStageDIFRegressor(K=4, D=32, seed=11)
    reg.fit(X, y, field_dim=48)

    rdir = ModelRegistry(str(tmp_path))
    card = ModelCard(
        name="tsd-rt",
        version="1.0.0",
        task="regression",
        model_type="TwoStageDIF",
        encoder_type="VectorEncoder",
        input_dim=2,
        field_dim=48,
    )
    rdir.register(reg, card)
    loaded, pre, c2 = rdir.load("tsd-rt", "1.0.0")
    assert pre is None and c2.model_type == "TwoStageDIF"
    _assert_nested_state_equal(reg.save_state(), loaded.save_state(), "tsd")

    xq = rng.standard_normal((8, 2))
    np.testing.assert_allclose(reg.predict(xq), loaded.predict(xq), rtol=1e-5, atol=1e-5)


def test_registry_mke_regressor_roundtrip(tmp_path):
    from Cypha import MKERegressor
    from cypha_studio.core.registry import ModelCard, ModelRegistry

    rng = np.random.default_rng(13)
    X = rng.standard_normal((120, 3))
    y = X.sum(axis=1) + 0.05 * rng.standard_normal(120)
    mke = MKERegressor.from_data(
        X[:100], y_seed=y[:100], K=4, D=48, field_dim=56, rng_seed=42,
    )
    for i in range(40):
        mke.train_step(X[i % len(X)], float(y[i % len(y)]))

    rdir = ModelRegistry(str(tmp_path))
    card = ModelCard(
        name="mke-rt",
        version="2.0.0",
        task="regression",
        model_type="MKE",
        encoder_type="RFFEncoder",
        input_dim=3,
        n_classes=4,
        field_dim=56,
    )
    rdir.register(mke, card)
    loaded, pre, c2 = rdir.load("mke-rt", "2.0.0")
    assert pre is None and c2.model_type == "MKE"
    _assert_nested_state_equal(mke.save_state(), loaded.save_state(), "mke")

    xq = rng.standard_normal(3)
    y0, u0 = mke.predict(xq)
    y1, u1 = loaded.predict(xq)
    np.testing.assert_allclose(y0, y1, rtol=1e-5, atol=1e-5)
    assert abs(float(u0) - float(u1)) < 1e-5
