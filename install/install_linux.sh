#!/usr/bin/env bash
# Cypha — Linux / WSL installer (Python stack + optional native build)
#
# Usage:
#   bash install/install_linux.sh [OPTIONS]
#
# Options:
#   --studio      Install PySide6 + pyqtgraph + pytest-qt (full Studio GUI deps)
#   --native      Build the C++ native core (requires cmake, ninja, gcc/g++)
#   --qt          Add -DCYPHA_BUILD_QT=ON to native build (requires qt6-base-dev)
#   --cuda        Add -DCYPHA_ENABLE_CUDA=ON (requires nvidia-cuda-toolkit)
#   --skip-tests  Skip the pytest health-check at the end
#   --help        Show this help message
#
# Examples:
#   # Headless Python only
#   bash install/install_linux.sh
#
#   # Full Studio + native core with Qt
#   bash install/install_linux.sh --studio --native --qt
#
#   # Full native with CUDA (GPU box)
#   bash install/install_linux.sh --native --cuda
#
# Run from the repository root:
#   cd /path/to/Cypha
#   bash install/install_linux.sh [OPTIONS]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OPT_STUDIO=0
OPT_NATIVE=0
OPT_QT=0
OPT_CUDA=0
OPT_SKIP_TESTS=0

for arg in "$@"; do
    case "$arg" in
        --studio)     OPT_STUDIO=1 ;;
        --native)     OPT_NATIVE=1 ;;
        --qt)         OPT_QT=1 ;;
        --cuda)       OPT_CUDA=1 ;;
        --skip-tests) OPT_SKIP_TESTS=1 ;;
        --help)
            sed -n '3,30p' "$0"
            exit 0
            ;;
        *) echo "Unknown option: $arg (use --help)"; exit 1 ;;
    esac
done

step() { echo; echo "==> $*"; }
ok()   { echo "    OK: $*"; }
warn() { echo "    WARN: $*"; }
die()  { echo "    ERROR: $*" >&2; exit 1; }

# ─── 0. Prerequisites ────────────────────────────────────────────────────────
step "Checking prerequisites"

command -v python3 >/dev/null 2>&1 || die "python3 not found. Install Python 3.11+."
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
python3 -c "import sys; assert sys.version_info >= (3,11), 'need 3.11+'" 2>/dev/null \
    || die "Python 3.11+ required (found $PY_VER)."
ok "Python $PY_VER"

if [[ $OPT_NATIVE -eq 1 ]]; then
    command -v cmake >/dev/null 2>&1 || die "cmake not found. Install: sudo apt-get install -y cmake ninja-build"
    command -v ninja >/dev/null 2>&1 || warn "ninja not found; CMake will fall back to make (slower)."
    command -v g++   >/dev/null 2>&1 || die "g++ not found. Install: sudo apt-get install -y build-essential"
    ok "Build tools available"
fi

if [[ $OPT_QT -eq 1 ]]; then
    dpkg -s qt6-base-dev >/dev/null 2>&1 \
        || die "qt6-base-dev not installed. Run: sudo apt-get install -y qt6-base-dev"
    ok "qt6-base-dev present"
fi

if [[ $OPT_CUDA -eq 1 ]]; then
    command -v nvcc >/dev/null 2>&1 || die "nvcc not found. Install NVIDIA CUDA toolkit."
    ok "nvcc present"
fi

# ─── 1. Virtual environment ──────────────────────────────────────────────────
step "Creating virtual environment (.venv)"

if [[ ! -d .venv ]]; then
    python3 -m venv .venv
    ok "Created .venv"
else
    ok ".venv already exists — skipping creation"
fi

# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip

# ─── 2. Python dependencies ──────────────────────────────────────────────────
step "Installing Python dependencies"

pip install --quiet -r requirements-verify.txt
ok "Core verify dependencies installed"

if [[ $OPT_STUDIO -eq 1 ]]; then
    pip install --quiet -r cypha_studio/requirements.txt
    pip install --quiet pytest-qt
    ok "Studio (PySide6 + pyqtgraph + pytest-qt) installed"
fi

for pkg in cypha_bench cypha_lm cypha_som; do
    if [[ -f "$pkg/requirements.txt" ]]; then
        pip install --quiet -r "$pkg/requirements.txt"
        ok "$pkg dependencies installed"
    fi
done

if [[ -f cypha_lm/pyproject.toml ]]; then
    pip install --quiet -e cypha_lm
    ok "cypha_lm installed (editable)"
fi

# ─── 3. Native C++ build (optional) ─────────────────────────────────────────
if [[ $OPT_NATIVE -eq 1 ]]; then
    step "Building native C++ core"

    # System SQLite (optional — enables experiment_db_smoke CTest)
    if ! dpkg -s libsqlite3-dev >/dev/null 2>&1; then
        warn "libsqlite3-dev not found. CMake will use the bundled SQLite amalgamation."
    fi

    CMAKE_ARGS=(
        -S native
        -B native/build-install
        -DCMAKE_BUILD_TYPE=Release
        --preset wsl-gcc-release
    )

    if [[ $OPT_QT -eq 1 ]]; then
        CMAKE_ARGS+=(-DCYPHA_BUILD_QT=ON)
    fi

    if [[ $OPT_CUDA -eq 1 ]]; then
        CMAKE_ARGS+=(-DCYPHA_ENABLE_CUDA=ON)
        # Default to architecture detection; override with CMAKE_CUDA_ARCHITECTURES env var
        if [[ -n "${CMAKE_CUDA_ARCHITECTURES:-}" ]]; then
            CMAKE_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES")
        fi
    fi

    cmake "${CMAKE_ARGS[@]}"
    cmake --build native/build-install -j"$(nproc)"
    ok "Native build complete"

    step "Running native CTest"
    ctest --test-dir native/build-install --output-on-failure
    ok "All native CTests passed"
else
    warn "Skipping native build (--native not specified)."
fi

# ─── 4. Smoke test ──────────────────────────────────────────────────────────
if [[ $OPT_SKIP_TESTS -eq 0 ]]; then
    step "Running pytest health check"
    export QT_QPA_PLATFORM=offscreen
    python -m pytest tests/test_lm_api.py cypha_lm/model/tests/test_cypha_lm.py tests/ -m "not slow" -q --tb=short --ignore=tests/test_gui_qtbot.py
    ok "All tests passed"
else
    warn "Skipping tests (--skip-tests specified)"
fi

# ─── Done ────────────────────────────────────────────────────────────────────
step "Installation complete"
cat <<'EOF'

Activate the environment:
    source .venv/bin/activate

Run the Python Studio GUI:
    python cypha_studio/main.py

Run the headless REST server:
    python -m uvicorn cypha_studio.server.api:app --host 0.0.0.0 --port 8765

Generate + load a demo CyphaLM checkpoint:
    python scripts/generate_demo_lm_checkpoint.py
    export CYPHA_LM_CHECKPOINT=examples/demo_cyphalm/demo

Run the native REST server (if --native was used):
    ./native/build-install/cypha_rest --model parity_fixtures/reference.cypha

Run the native Qt shell (if --native --qt was used):
    ./native/build-install/qt/cypha_qt_shell

Run the multi-domain benchmark:
    python benchmark.py

Run tests:
    pytest tests/ -m "not slow"

EOF
