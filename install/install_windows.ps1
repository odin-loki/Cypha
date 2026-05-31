<#
.SYNOPSIS
    Cypha — Windows installer (Python stack + optional native build)

.DESCRIPTION
    Sets up Cypha on Windows:
      1. Creates a Python virtual environment (.venv)
      2. Installs Python dependencies (verify, studio, bench, lm, som)
      3. Optionally builds the native C++ core (cypha_core, cypha_rest,
         cypha_qt_shell) via MSVC or MinGW cross-compile from WSL
      4. Runs the pytest gate to confirm the installation is healthy

.PARAMETER Studio
    Install PySide6 + pyqtgraph + pytest-qt (full Studio GUI deps).
    Without this flag only the headless/API dependencies are installed.

.PARAMETER Native
    Build the C++ native core after Python setup. Requires Visual Studio 2022
    (MSVC) or WSL with GCC + CMake installed. Skipped by default.

.PARAMETER Qt
    When combined with -Native, adds -DCYPHA_BUILD_QT=ON to the CMake
    configure step. Requires Qt6 (msvc2022_64 kit) on your PATH.
    Qt prefix can be set via the QT_PREFIX env variable, e.g.:
        $env:QT_PREFIX = "C:\Qt\6.7.0\msvc2022_64"

.PARAMETER SkipTests
    Skip the pytest health-check at the end.

.EXAMPLE
    # Headless Python install, no GUI, no native build
    powershell -ExecutionPolicy Bypass -File install\install_windows.ps1

.EXAMPLE
    # Full Studio + native build with Qt shell
    powershell -ExecutionPolicy Bypass -File install\install_windows.ps1 -Studio -Native -Qt

.NOTES
    Run from the repository root:
        cd C:\path\to\Cypha
        powershell -ExecutionPolicy Bypass -File install\install_windows.ps1 [flags]
#>

[CmdletBinding()]
param(
    [switch]$Studio,
    [switch]$Native,
    [switch]$Qt,
    [switch]$SkipTests
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot

function Step { param([string]$msg) Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Ok   { param([string]$msg) Write-Host "    OK: $msg" -ForegroundColor Green }
function Warn { param([string]$msg) Write-Host "    WARN: $msg" -ForegroundColor Yellow }
function Die  { param([string]$msg) Write-Host "    ERROR: $msg" -ForegroundColor Red; exit 1 }

# ─── 0. Prerequisites ───────────────────────────────────────────────────────
Step "Checking prerequisites"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Die "Python not found. Install Python 3.11+ from https://python.org and add to PATH."
}
$pyVer = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ([version]$pyVer -lt [version]"3.11") {
    Die "Python 3.11+ required (found $pyVer)."
}
Ok "Python $pyVer"

# ─── 1. Virtual environment ──────────────────────────────────────────────────
Step "Creating virtual environment (.venv)"
Set-Location $RepoRoot

if (-not (Test-Path ".venv")) {
    python -m venv .venv
    Ok "Created .venv"
} else {
    Ok ".venv already exists — skipping creation"
}

$pip = ".venv\Scripts\pip.exe"
$python = ".venv\Scripts\python.exe"

# Upgrade pip silently
& $pip install --quiet --upgrade pip

# ─── 2. Python dependencies ──────────────────────────────────────────────────
Step "Installing Python dependencies"

& $pip install --quiet -r requirements-verify.txt
Ok "Core verify dependencies installed"

if ($Studio) {
    & $pip install --quiet -r cypha_studio\requirements.txt
    & $pip install --quiet pytest-qt
    Ok "Studio (PySide6 + pyqtgraph + pytest-qt) installed"
}

# Install satellite packages in editable mode if present
foreach ($pkg in @("cypha_bench", "cypha_lm", "cypha_som")) {
    $reqFile = "$pkg\requirements.txt"
    if (Test-Path $reqFile) {
        & $pip install --quiet -r $reqFile
        Ok "$pkg dependencies installed"
    }
}

if (Test-Path "cypha_lm\pyproject.toml") {
    & $pip install --quiet -e cypha_lm
    Ok "cypha_lm installed (editable)"
}

# ─── 3. Native C++ build (optional) ─────────────────────────────────────────
if ($Native) {
    Step "Building native C++ core"

    $cmakeArgs = @(
        "--preset", "windows-msvc-release"
    )
    if ($Qt) {
        $qtPrefix = $env:QT_PREFIX
        if (-not $qtPrefix) {
            Warn "QT_PREFIX not set. CMake will try to find Qt6 automatically."
        } else {
            $cmakeArgs += "-DCMAKE_PREFIX_PATH=$qtPrefix"
        }
        $cmakeArgs += "-DCYPHA_BUILD_QT=ON"
    }

    Push-Location native
    try {
        cmake @cmakeArgs
        cmake --build --preset windows-msvc-release-build
        Ok "Native build complete"

        if ($Qt) {
            $shellBin = "build-windows-msvc\Release\cypha_qt_shell.exe"
            if (Test-Path $shellBin) {
                windeployqt $shellBin --release --no-translations 2>$null
                Ok "windeployqt completed"
            }
        }
    } finally {
        Pop-Location
    }
} else {
    Warn "Skipping native build (-Native not specified). To build C++ core, re-run with -Native."
}

# ─── 4. Smoke test ──────────────────────────────────────────────────────────
if (-not $SkipTests) {
    Step "Running pytest health check"
    $env:QT_QPA_PLATFORM = "offscreen"
    & $python -m pytest tests/ -m "not slow" -q --tb=short
    if ($LASTEXITCODE -ne 0) {
        Die "pytest failed. See output above."
    }
    Ok "All tests passed"
} else {
    Warn "Skipping tests (-SkipTests specified)"
}

# ─── Done ────────────────────────────────────────────────────────────────────
Step "Installation complete"
Write-Host @"

Activate the environment:
    .venv\Scripts\activate

Run the Python Studio GUI:
    python cypha_studio\main.py

Run the headless REST server:
    python -m uvicorn cypha_studio.server.api:app --host 0.0.0.0 --port 8765

Run the native Qt shell (if -Native -Qt was used):
    powershell -ExecutionPolicy Bypass -File scripts\run_cypha_qt_windows.ps1

Run tests:
    pytest tests\ -m "not slow"

"@ -ForegroundColor Green
