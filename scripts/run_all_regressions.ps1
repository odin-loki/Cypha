# Native regression checks: cmake build + ctest -R native_.
# Usage (repo root): powershell -ExecutionPolicy Bypass -File scripts/run_all_regressions.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$BuildDir = Join-Path $Root "native\build"
if (-not $env:QT_QPA_PLATFORM) {
    $env:QT_QPA_PLATFORM = "offscreen"
}

Write-Host "== cmake configure =="
cmake -S (Join-Path $Root "native") -B $BuildDir -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== cmake build =="
cmake --build $BuildDir --parallel
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== ctest -R native_ =="
ctest --test-dir $BuildDir -R native_ --output-on-failure
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Native regression OK (ctest -R native_)."
