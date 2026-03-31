# Windows: venv + verify deps + parity fixtures + pytest + legacy test scripts.
# Optional: -Studio — after requirements-verify.txt, also cypha_studio/requirements.txt + pytest-qt (CI-like GUI).
param(
    [switch]$Studio
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Venv = Join-Path $Root ".venv-win"
$Py = Join-Path $Venv "Scripts\python.exe"
$Pip = Join-Path $Venv "Scripts\pip.exe"

if (-not (Test-Path $Py)) {
    py -3 -m venv $Venv
}
& $Pip install -q -U pip
& $Pip install -q -r requirements-verify.txt
if ($Studio) {
    & $Pip install -q -r (Join-Path $Root "cypha_studio\requirements.txt")
    & $Pip install -q pytest-qt
}
$env:QT_QPA_PLATFORM = if ($env:QT_QPA_PLATFORM) { $env:QT_QPA_PLATFORM } else { "offscreen" }

$Manifest = Join-Path $Root "parity_fixtures\manifest.json"
if (-not (Test-Path $Manifest)) {
    Write-Host "Generating parity_fixtures/ ..."
    & $Py (Join-Path $Root "scripts\generate_parity_fixtures.py")
}

Write-Host "== pytest =="
& $Py -m pytest (Join-Path $Root "tests") -v --tb=short

Write-Host "== test_cypha.py =="
& $Py (Join-Path $Root "test_cypha.py")

Write-Host "== cypha_studio tests =="
& $Py (Join-Path $Root "cypha_studio\test_cypha_studio.py")

Write-Host "All done."
Write-Host "Optional: native build + CYPHA_REST_BIN — see native/README.md (vcpkg sqlite3 for experiment_db_smoke on MSVC)."
