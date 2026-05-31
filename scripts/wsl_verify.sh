#!/usr/bin/env bash
# Cypha — WSL verification: tests, benchmark (optional), cProfile hotspots.
# Usage (from repo root, in WSL or via: wsl bash scripts/wsl_verify.sh)
# Optional: RUN_NATIVE=1 — after pytest, build native/build, ctest, and run test_cypha_rest_smoke with CYPHA_REST_BIN.
# Optional: CYPHA_BUILD_QT=1 with RUN_NATIVE=1 — pass -DCYPHA_BUILD_QT=ON (needs qt6-base-dev); ctest runs native_qt_stub_load_reference; then pytest tests/test_qt_stub_native.py.
# Optional: PYTEST_MARK='not slow' — pass -m to the main pytest invocation (e.g. skip tests/test_cypha_studio_runner.py @slow).
# Optional: FULL_STUDIO_DEPS=1 — after requirements-verify.txt, also cypha_studio/requirements.txt + pytest-qt (CI-like PySide6 + qtbot; keeps httpx/pytest from verify).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"

echo "== Cypha WSL verify =="
echo "ROOT=$ROOT"

VENV="${VENV:-$ROOT/.venv-wsl}"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Creating venv: $VENV"
  python3 -m venv "$VENV"
fi
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
"$PY" -V

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  "$PIP" install -q -U pip
  "$PIP" install -q -r requirements-verify.txt
  if [[ "${FULL_STUDIO_DEPS:-0}" == "1" ]]; then
    "$PIP" install -q -r cypha_studio/requirements.txt
    "$PIP" install -q pytest-qt
  fi
  "$PIP" install -q pytest pytest-cov || true
fi

echo ""
echo "---- test_cypha.py (Cypha.py unit suite) ----"
"$PY" test_cypha.py

echo ""
echo "---- cypha_studio/test_cypha_studio.py ----"
"$PY" cypha_studio/test_cypha_studio.py

echo ""
echo "---- pytest tests/ ----"
if [[ ! -f parity_fixtures/manifest.json ]]; then
  echo "Generating parity_fixtures/ (first run)..."
  "$PY" scripts/generate_parity_fixtures.py
fi
if [[ -n "${PYTEST_MARK:-}" ]]; then
  "$PY" -m pytest -m "$PYTEST_MARK" tests/ -q --tb=line
else
  "$PY" -m pytest tests/ -q --tb=line
fi

echo ""
echo "---- M6: export_experiment_schema_sql.py (stdout smoke) ----"
"$PY" scripts/export_experiment_schema_sql.py >/dev/null

# Optional: build Linux cypha_rest and run REST subprocess tests (not skipped).
# Requires: cmake, g++.  Example: RUN_NATIVE=1 bash scripts/wsl_verify.sh
if [[ "${RUN_NATIVE:-0}" == "1" ]]; then
  echo ""
  echo "---- RUN_NATIVE=1: native/build + ctest + cypha_rest smoke (tests/test_cypha_rest_smoke.py) ----"
  echo "      For M6 CTest native_experiment_db_smoke: sudo apt-get install -y libsqlite3-dev"
  NATIVE_CMAKE_EXTRA=()
  if [[ "${CYPHA_BUILD_QT:-0}" == "1" ]]; then
    NATIVE_CMAKE_EXTRA+=(-DCYPHA_BUILD_QT=ON)
    echo "      CYPHA_BUILD_QT=1: install qt6-base-dev for cypha_qt_stub + CTest native_qt_stub_load_reference"
  fi
  cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release "${NATIVE_CMAKE_EXTRA[@]}"
  cmake --build native/build -j"$(nproc)"
  ctest --test-dir native/build --output-on-failure
  export CYPHA_REST_BIN="$ROOT/native/build/cypha_rest"
  "$PY" -m pytest tests/test_cypha_rest_smoke.py -q --tb=line
  if [[ "${CYPHA_BUILD_QT:-0}" == "1" ]]; then
    "$PY" -m pytest tests/test_qt_stub_native.py -q --tb=line
  fi
  echo "      M1: reference.cypha has empty Tier-1 for cypha_parity (PORT_CONTRACT §4)."
else
  echo ""
  echo "Tip: Windows MinGW cypha_rest.exe — PowerShell: native/scripts/build_cypha_rest_mingw_wsl.ps1 [-RunPytest]"
  echo "      Rebuild native after parity fixture bumps (e.g. native_parity.bin v2) so ctest cypha_parity matches."
  echo "      CTest also covers mke_train_step / mke_train_extended (mke_train_step_parity); bump those fixtures together with native when router math changes."
fi

if [[ "${RUN_BENCHMARK:-0}" == "1" ]]; then
  echo ""
  echo "---- benchmark.py (full suite; can take several minutes) ----"
  BENCH_LOG="${BENCHMARK_LOG:-artifacts/profiles/benchmark_baseline.txt}"
  "$PY" benchmark.py 2>&1 | tee "$BENCH_LOG"
  echo "Benchmark log: $BENCH_LOG"
fi

if [[ "${SKIP_PROFILE:-0}" != "1" ]]; then
  echo ""
  echo "---- cProfile: test_cypha.py -> profile_stats.cprof + profile_hotspots.txt ----"
  "$PY" -m cProfile -o profile_stats.cprof -s cumtime test_cypha.py 2>&1 | tee profile_test_cypha_run.log
  "$PY" scripts/print_profile_hotspots.py profile_stats.cprof -n 40 | tee profile_hotspots.txt
  echo "Profiling: profile_hotspots.txt, profile_stats.cprof"
else
  echo ""
  echo "---- SKIP_PROFILE=1: skipping cProfile ----"
fi

echo ""
echo "Done."
