# Windows: native cmake build + ctest -R native_.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$BuildDir = Join-Path $Root "native\build"
$env:QT_QPA_PLATFORM = if ($env:QT_QPA_PLATFORM) { $env:QT_QPA_PLATFORM } else { "offscreen" }

Write-Host "== cmake configure =="
cmake -S (Join-Path $Root "native") -B $BuildDir -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== cmake build =="
cmake --build $BuildDir --parallel
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== ctest -R native_ =="
ctest --test-dir $BuildDir -R native_ --output-on-failure
exit $LASTEXITCODE
