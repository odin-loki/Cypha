# Package Windows x86_64 release ZIP from a native MSVC CMake build directory.
#
# Usage (from repo root):
#   pwsh -File scripts/package_release_windows.ps1 -Version 2.3.24 -BuildDir native/build-msvc-release
#
# Looks for *.exe in <BuildDir>, <BuildDir>/Release, and one level of nested Release folders
# (Visual Studio multi-config layout).

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Version,
    [Parameter(Mandatory = $true)][string]$BuildDir,
    [string]$OutDir = "dist"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not [IO.Path]::IsPathRooted($BuildDir)) {
    $BuildDir = Join-Path $RepoRoot $BuildDir
}
if (-not [IO.Path]::IsPathRooted($OutDir)) {
    $OutDir = Join-Path $RepoRoot $OutDir
}
if (-not (Test-Path $BuildDir)) {
    throw "Build directory not found: $BuildDir"
}

$Staging = Join-Path $OutDir "cypha-$Version-windows-x86_64"
$Archive = Join-Path $OutDir "cypha-$Version-windows-x86_64.zip"

# Production binaries (added to user PATH by install.ps1)
$Binaries = @(
    "cypha_rest.exe",
    "cypha_bench_run.exe",
    "cypha_bench_report.exe",
    "cypha_diagnostics_run.exe",
    "cypha_tune_run.exe",
    "cyphalm_bench_native.exe",
    "cypha_baseline_lock.exe",
    "baseline_lock_validate.exe",
    "registry_register.exe",
    "create_model_smoke.exe"
)

# Dev / research golden tools (bin/dev/, not on PATH)
$DevBinaries = @(
    "score_batch_golden.exe",
    "multilabel_dif_golden.exe",
    "merge_from_golden.exe",
    "similarity_index_golden.exe",
    "embed_table_golden.exe",
    "retrieval_golden.exe",
    "som_golden.exe",
    "kernel_llr_golden.exe",
    "gh_infer_deliberation_golden.exe",
    "cyphalm_checkpoint_golden.exe"
)

function Find-BuiltExe {
    param([string]$Name)
    $candidates = @(
        (Join-Path $BuildDir $Name),
        (Join-Path $BuildDir "Release\$Name"),
        (Join-Path $BuildDir "RelWithDebInfo\$Name")
    )
    Get-ChildItem -Path $BuildDir -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        $candidates += (Join-Path $_.FullName $Name)
        $candidates += (Join-Path $_.FullName "Release\$Name")
    }
    foreach ($c in $candidates) {
        if (Test-Path -LiteralPath $c) { return (Resolve-Path -LiteralPath $c).Path }
    }
    return $null
}

if (Test-Path $Staging) { Remove-Item $Staging -Recurse -Force }
New-Item -ItemType Directory -Force -Path (Join-Path $Staging "bin\dev") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Staging "share\demo_fixtures") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Staging "share\examples") | Out-Null
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Set-Content -Path (Join-Path $Staging "VERSION") -Value $Version -Encoding ascii -NoNewline

foreach ($bin in $Binaries) {
    $src = Find-BuiltExe $bin
    if (-not $src) {
        throw "Required release binary missing from build dir: $bin (looked under $BuildDir)"
    }
    Copy-Item -LiteralPath $src -Destination (Join-Path $Staging "bin\$bin") -Force
    Write-Host "  + bin/$bin"
}

foreach ($bin in $DevBinaries) {
    $src = Find-BuiltExe $bin
    if (-not $src) {
        throw "Required dev release binary missing from build dir: $bin (looked under $BuildDir)"
    }
    Copy-Item -LiteralPath $src -Destination (Join-Path $Staging "bin\dev\$bin") -Force
    Write-Host "  + bin/dev/$bin"
}

Copy-Item (Join-Path $RepoRoot "packaging\install_release_windows.ps1") (Join-Path $Staging "install.ps1") -Force

@"
Cypha $Version - Windows x86_64 native tools (MSVC, full C++ framework)
=======================================================================

Quick install (adds %LOCALAPPDATA%\Cypha\$Version\bin to user PATH):
  powershell -ExecutionPolicy Bypass -File install.ps1

Run native REST (classifier + CyphaLM + CyphaDIF routes):
  cypha_rest.exe --listen 127.0.0.1:8099 --cypha share\demo_fixtures\reference.cypha ^
    --f-field-json share\demo_fixtures\f_field.json

CyphaDIF REST routes (POST JSON):
  /dif/retrieve   - ranked database hits (input, database, top_k, optional label)
  /dif/generate   - latent samples (mode: langevin | from_observation | retrieval_augmented)

Run native bench domains (d01-d17):
  cypha_bench_run.exe --domain 17

Rebuild bench report from saved tables:
  cypha_bench_report.exe --output .\bench_report

Run native diagnostics (phases 1-4 orchestrator):
  cypha_diagnostics_run.exe --fixtures share\demo_fixtures\..\..\fixtures

Run CyphaLM bench CLI:
  cyphalm_bench_native.exe --mode hybrid --profile d17 --n-train 5000 --n-eval 500 --threads 1

Dev golden tools (not on PATH): bin\dev\*_golden.exe

Qt shell: build cypha_qt_shell from native/ with MSVC + Qt 6 (see docs/native/qt/README.md
and packaging/build_windows_bundle.ps1).

These binaries are built with MSVC (Visual Studio). Redistributable VC++ runtime may be
required on clean machines (usual Windows Update / VS redist).
"@ | Set-Content -Path (Join-Path $Staging "README.txt") -Encoding utf8

foreach ($f in @("reference.cypha", "f_field.json", "train_hparams.json")) {
    $src = Join-Path $RepoRoot "fixtures\$f"
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $Staging "share\demo_fixtures\$f") -Force
    }
}

$demo = Join-Path $RepoRoot "examples\demo_cyphalm"
if (Test-Path $demo) {
    Copy-Item $demo (Join-Path $Staging "share\examples\demo_cyphalm") -Recurse -Force
}

if (Test-Path $Archive) { Remove-Item $Archive -Force }
Add-Type -AssemblyName System.IO.Compression.FileSystem
[IO.Compression.ZipFile]::CreateFromDirectory($Staging, $Archive, [IO.Compression.CompressionLevel]::Optimal, $true)
Write-Host "Created $Archive ($((Get-Item $Archive).Length) bytes)"
