"""
C++ / Python benchmark parity demonstration.

This script answers the question: "Does the C++ get the same benchmark results as Python?"

Architecture note
-----------------
The cypha_bench suite trains and evaluates models *entirely in Python* using
BenchClassifier → CyphaDIF.  The C++ port (cypha_rest) is an **inference-only**
REST server — it has no online training path.

What this means for benchmarks:
  - You cannot run cypha_bench directly through cypha_rest.
  - However, for any model trained in Python and saved as a .cypha file, the
    C++ server and the Python runtime will produce **identical** label + confidence
    predictions for the same input.  This is proven by the parity test suite.

This script demonstrates end-to-end parity for a D01-style classification task:
  1. Train a CyphaDIF model in Python (online, like cypha_bench does).
  2. Save the model as a .cypha file.
  3. Load the model into cypha_rest (native C++ REST server) if available.
  4. Run the test set through both Python and C++ and compare predictions.

Usage
-----
  # Python-only check (always works):
  python scripts/cpp_bench_parity_demo.py

  # With C++ server (requires native build):
  CYPHA_REST_BIN=native/build/cypha_rest python scripts/cpp_bench_parity_demo.py

Prerequisites for C++ path
---------------------------
  cmake -S native -B native/build -DCYPHA_BUILD_QT=OFF
  cmake --build native/build --config Release
  # binary appears at native/build/cypha_rest (Linux) or native/build/Release/cypha_rest.exe (Windows)

Parity tolerance
----------------
  Labels must be identical.  Confidence values must agree within 1e-5 (floating-
  point rounding between Python float64 and C++ double).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Bootstrap repo root
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from Cypha import CyphaDIF, VectorEncoder, cypha_save_binary  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: generate a D01-style synthetic dataset
# ─────────────────────────────────────────────────────────────────────────────

rng_np = np.random.default_rng(42)
X, y = make_classification(
    n_samples=2000, n_features=10, n_informative=5, n_redundant=2, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: train a Python model (identical to what cypha_bench does)
# ─────────────────────────────────────────────────────────────────────────────

print("Training Python CyphaDIF model …")
clf = CyphaDIF(encoder=VectorEncoder(10), rng=rng_np)
for xi, yi in zip(X_train, y_train):
    clf.train_step(xi, str(yi))

# Evaluate in Python
py_labels = []
py_confs = []
for xi in X_test:
    label, conf = clf.infer(xi)
    py_labels.append(label)
    py_confs.append(conf)

py_acc = sum(str(yt) == lp for yt, lp in zip(y_test, py_labels)) / len(y_test)
print(f"  Python accuracy: {py_acc:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: save model
# ─────────────────────────────────────────────────────────────────────────────

with tempfile.TemporaryDirectory() as tmpdir:
    model_path = Path(tmpdir) / "parity_demo.cypha"
    cypha_save_binary(clf.save_state(), str(model_path))
    print(f"  Model saved: {model_path.name} ({model_path.stat().st_size:,} bytes)")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: try to run cypha_rest (C++ server)
    # ─────────────────────────────────────────────────────────────────────────

    rest_bin = os.environ.get("CYPHA_REST_BIN", "")
    if not rest_bin:
        # Try common build locations
        candidates = [
            _REPO / "native" / "build" / "cypha_rest",
            _REPO / "native" / "build" / "Release" / "cypha_rest.exe",
            _REPO / "native" / "build" / "cypha_rest.exe",
        ]
        for c in candidates:
            if c.exists():
                rest_bin = str(c)
                break

    if not rest_bin:
        print("\nC++ cypha_rest binary not found.")
        print("Build it with:")
        print("  cmake -S native -B native/build -DCYPHA_BUILD_QT=OFF")
        print("  cmake --build native/build --config Release")
        print("Then re-run with: CYPHA_REST_BIN=<path> python scripts/cpp_bench_parity_demo.py")
        print("\nParity is still proven by the 20+ CTest + pytest parity fixtures:")
        print("  pytest tests/ -k 'native'")
        print("\nPython-only result:")
        print(f"  Accuracy: {py_acc:.4f}  |  Experts: {len(clf.memory._classes)}")
        sys.exit(0)

    # -------------------------------------------------------------------------
    # Launch cypha_rest on a free port
    # -------------------------------------------------------------------------
    import socket

    def _free_port() -> int:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    port = _free_port()
    registry_dir = Path(tmpdir) / "registry"
    registry_dir.mkdir()

    srv = subprocess.Popen(
        [rest_bin, "--port", str(port), "--registry", str(registry_dir)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        import urllib.request

        # Wait for the server to start
        deadline = time.time() + 10.0
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                break
            except Exception:
                time.sleep(0.2)
        else:
            print("ERROR: cypha_rest did not start within 10 s")
            sys.exit(1)

        # Register and load the model
        reg_body = json.dumps(
            {"name": "parity_demo", "version": "1", "model_cypha": str(model_path)}
        ).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/register",
            data=reg_body,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)

        load_body = json.dumps({"name": "parity_demo", "version": "1"}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/load",
            data=load_body,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)

        # Run test set through C++
        cpp_labels = []
        cpp_confs = []
        for xi in X_test:
            body = json.dumps({"input": xi.tolist()}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/predict",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            resp = json.loads(urllib.request.urlopen(req, timeout=5).read())
            cpp_labels.append(resp["label"])
            cpp_confs.append(resp["confidence"])

        cpp_acc = sum(str(yt) == lc for yt, lc in zip(y_test, cpp_labels)) / len(y_test)

        # ─────────────────────────────────────────────────────────────────────
        # Step 5: compare
        # ─────────────────────────────────────────────────────────────────────

        label_match = sum(lp == lc for lp, lc in zip(py_labels, cpp_labels))
        conf_diffs = [abs(fp - fc) for fp, fc in zip(py_confs, cpp_confs)]
        max_conf_diff = max(conf_diffs)
        mean_conf_diff = float(np.mean(conf_diffs))

        print(f"\n{'='*60}")
        print("PARITY RESULT")
        print(f"{'='*60}")
        print(f"  Test samples      : {len(X_test)}")
        print(f"  Python accuracy   : {py_acc:.4f}")
        print(f"  C++ accuracy      : {cpp_acc:.4f}")
        print(f"  Label match       : {label_match}/{len(X_test)} ({label_match/len(X_test):.2%})")
        print(f"  Max conf. diff    : {max_conf_diff:.2e}")
        print(f"  Mean conf. diff   : {mean_conf_diff:.2e}")
        print(f"  Parity tolerance  : 1e-5")

        if label_match == len(X_test) and max_conf_diff < 1e-5:
            print("\n✅ PERFECT PARITY — C++ and Python produce identical predictions.")
        else:
            mismatches = [(i, py_labels[i], cpp_labels[i]) for i in range(len(X_test)) if py_labels[i] != cpp_labels[i]]
            print(f"\n❌ PARITY FAILURE — {len(mismatches)} label mismatches, max conf diff {max_conf_diff:.2e}")
            for i, pl, cl in mismatches[:5]:
                print(f"     sample {i}: Python={pl}, C++={cl}")

    finally:
        srv.terminate()
        srv.wait()
