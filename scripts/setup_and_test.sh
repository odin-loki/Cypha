#!/usr/bin/env bash
# Unix/WSL: create .venv-wsl, install verify deps, generate parity fixtures if missing, pytest + legacy suites.
# Optional: FULL_STUDIO_DEPS=1 — after requirements-verify.txt, also cypha_studio/requirements.txt + pytest-qt (CI-like GUI).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VENV="${VENV:-$ROOT/.venv-wsl}"
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install -q -U pip
"$VENV/bin/pip" install -q -r requirements-verify.txt
if [[ "${FULL_STUDIO_DEPS:-0}" == "1" ]]; then
  "$VENV/bin/pip" install -q -r cypha_studio/requirements.txt
  "$VENV/bin/pip" install -q pytest-qt
fi
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
if [[ ! -f parity_fixtures/manifest.json ]]; then
  echo "Generating parity_fixtures/ ..."
  "$VENV/bin/python" scripts/generate_parity_fixtures.py
fi
echo "== pytest =="
"$VENV/bin/python" -m pytest tests/ -v --tb=short
echo "== test_cypha.py =="
"$VENV/bin/python" test_cypha.py
echo "== cypha_studio tests =="
"$VENV/bin/python" cypha_studio/test_cypha_studio.py
echo "All done."
echo "Optional: native + CTest — see native/README.md (e.g. apt install libsqlite3-dev; RUN_NATIVE=1 bash scripts/wsl_verify.sh)."
