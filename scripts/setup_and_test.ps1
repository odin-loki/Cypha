# Windows: native cmake build + ctest -R native_.
param(
    # Default resolved below via Get-DefaultNativeBuildDir (outside the OneDrive-synced
    # repo tree). Pass -BuildDir to override, e.g. -BuildDir native\build for the old behavior.
    [string]$BuildDir = ""
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "lib\NativeBenchCommon.ps1")

$Root = Get-CyphaRepoRoot -ScriptRoot $PSScriptRoot
Set-Location $Root

if (-not $BuildDir) {
    $BuildDir = Get-DefaultNativeBuildDir
} elseif (-not [System.IO.Path]::IsPathRooted($BuildDir)) {
    $BuildDir = Join-Path $Root $BuildDir
}
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
