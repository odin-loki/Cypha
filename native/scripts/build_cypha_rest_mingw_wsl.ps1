# Cross-compile native/*.exe via MinGW on WSL (from Windows PowerShell).
# Default: cypha_rest only (fast). Use -AllTargets for full tree (parity tools, experiment_db_smoke, …).
# Usage (repo root):
#   powershell -File native/scripts/build_cypha_rest_mingw_wsl.ps1
#   powershell -File native/scripts/build_cypha_rest_mingw_wsl.ps1 -AllTargets
#   powershell -File native/scripts/build_cypha_rest_mingw_wsl.ps1 -RunPytest
#
# Requires: WSL with cmake, g++-mingw-w64-x86-64, make. Uses single-quoted bash -lc
# so PowerShell does not expand $(nproc).

param(
    [switch]$AllTargets,
    [switch]$RunPytest
)

$ErrorActionPreference = "Stop"

function Convert-WindowsPathToWsl {
    param([string]$Path)
    $p = (Resolve-Path $Path).Path
    if ($p -match '^([A-Za-z]):[\\/]+(.+)$') {
        $drive = $matches[1].ToLower()
        $rest = $matches[2] -replace '\\', '/'
        return "/mnt/$drive/$rest"
    }
    throw "Cannot map path to WSL: $p"
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$wslRoot = Convert-WindowsPathToWsl $repoRoot
$tc = "$wslRoot/native/toolchains/mingw-w64-x86_64.cmake"

# One line so `bash -lc` receives a single argument reliably.
if ($AllTargets) {
    $bashCmd = "set -e; cd '$wslRoot' && cmake -S native -B native/build-mingw-w64 -DCMAKE_TOOLCHAIN_FILE='$tc' -DCMAKE_BUILD_TYPE=Release && cmake --build native/build-mingw-w64 -j8"
} else {
    $bashCmd = "set -e; cd '$wslRoot' && cmake -S native -B native/build-mingw-w64 -DCMAKE_TOOLCHAIN_FILE='$tc' -DCMAKE_BUILD_TYPE=Release && cmake --build native/build-mingw-w64 -j8 --target cypha_rest"
}
wsl -e bash -lc $bashCmd

$exe = Join-Path $repoRoot "native\build-mingw-w64\cypha_rest.exe"
if (-not (Test-Path $exe)) { throw "Build finished but missing $exe" }
Write-Host "Built: $exe"
if ($AllTargets) { Write-Host "All MinGW targets under native\build-mingw-w64\ (run ctest from WSL if desired)." }

if ($RunPytest) {
    $env:CYPHA_REST_BIN = $exe
    Set-Location $repoRoot
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m pytest tests/test_cypha_rest_smoke.py -v --tb=short
    } else {
        python -m pytest tests/test_cypha_rest_smoke.py -v --tb=short
    }
}
