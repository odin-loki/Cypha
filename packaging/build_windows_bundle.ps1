<#
.SYNOPSIS
    Package cypha_qt_shell as a self-contained Windows folder (windeployqt).

.DESCRIPTION
    Builds on docs/FUTURE.md section 3 and native/qt/README.md:
      1. cmake --build (Release, CYPHA_BUILD_QT=ON) for cypha_qt_shell + cypha_rest
      2. Copies executables into an output folder
      3. Runs windeployqt to pull required Qt DLLs alongside the exe

    MinGW cross-builds from WSL cannot produce a Qt shell — run this script natively
    on Windows with Qt 6 installed (MSVC or MinGW kit).

.PARAMETER Version
    Version string written to VERSION in the output folder (default: dev).

.PARAMETER BuildDir
    CMake build directory (default: native\build-qt-release).

.PARAMETER OutDir
    Destination folder for the packaged distribution.
    Default: dist\cypha-<Version>-windows-qt

.PARAMETER QtBinDir
    Directory containing windeployqt.exe (default: auto-detect from PATH or C:\Qt).

.PARAMETER SkipBuild
    Skip cmake configure/build; package from an existing build tree.

.PARAMETER WithFixtures
    Copy fixtures\reference.cypha + f_field.json into share\demo_fixtures.

.EXAMPLE
    # From repo root (Qt 6 on PATH):
    powershell -ExecutionPolicy Bypass -File packaging\build_windows_bundle.ps1 -Version 2.2.8 -WithFixtures

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File packaging\build_windows_bundle.ps1 `
      -BuildDir native\build `
      -QtBinDir "C:\Qt\6.11.0\msvc2022_64\bin" `
      -WithFixtures
#>

[CmdletBinding()]
param(
    [string]$Version   = "dev",
    [string]$BuildDir  = "native\build-qt-release",
    [string]$OutDir    = "",
    [string]$QtBinDir  = "",
    [switch]$SkipBuild,
    [switch]$WithFixtures
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BuildDir = Join-Path $RepoRoot $BuildDir
if ($OutDir -eq "") {
    $OutDir = Join-Path $RepoRoot "dist\cypha-$Version-windows-qt"
} else {
    $OutDir = Join-Path $RepoRoot $OutDir
}

function Find-BuiltExe {
    param([string]$Name)
    $candidates = @(
        (Join-Path $BuildDir $Name),
        (Join-Path $BuildDir "qt\$Name")
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    return $null
}

function Find-WinDeployQt {
    if ($QtBinDir -ne "") {
        $wdq = Join-Path $QtBinDir "windeployqt.exe"
        if (-not (Test-Path $wdq)) {
            throw "windeployqt.exe not found at $wdq"
        }
        return $wdq
    }
    $cmd = Get-Command "windeployqt.exe" -ErrorAction SilentlyContinue
    if ($null -ne $cmd) { return $cmd.Source }
    foreach ($root in @("C:\Qt", "$env:USERPROFILE\Qt")) {
        if (-not (Test-Path $root)) { continue }
        $found = Get-ChildItem -Path $root -Filter "windeployqt.exe" -Recurse -ErrorAction SilentlyContinue |
                 Sort-Object FullName -Descending | Select-Object -First 1
        if ($null -ne $found) { return $found.FullName }
    }
    throw @"
windeployqt.exe not found on PATH or in C:\Qt.
Install Qt 6 from https://www.qt.io/download and add its bin\ to PATH,
or pass -QtBinDir 'C:\Qt\6.x.x\msvc2022_64\bin'.
"@
}

if (-not $SkipBuild) {
    Write-Host "==> Configure Release build (CYPHA_BUILD_QT=ON)"
    cmake -S (Join-Path $RepoRoot "native") -B $BuildDir `
        -DCMAKE_BUILD_TYPE=Release `
        -DCYPHA_BUILD_QT=ON
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

    Write-Host "==> Build cypha_qt_shell + cypha_rest"
    cmake --build $BuildDir --config Release --target cypha_qt_shell cypha_rest
    if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }
}

$ShellExe = Find-BuiltExe "cypha_qt_shell.exe"
if ($null -eq $ShellExe) {
    throw "cypha_qt_shell.exe not found under $BuildDir. Build with -DCYPHA_BUILD_QT=ON first."
}
$RestExe = Find-BuiltExe "cypha_rest.exe"

Write-Host "Found: $ShellExe"
if ($null -ne $RestExe) { Write-Host "Found: $RestExe" }

$WinDeployQt = Find-WinDeployQt
Write-Host "windeployqt: $WinDeployQt"

if (Test-Path $OutDir) {
    Write-Host "Cleaning existing output: $OutDir"
    Remove-Item -Recurse -Force $OutDir
}
New-Item -ItemType Directory -Path $OutDir | Out-Null

Copy-Item $ShellExe -Destination $OutDir
Write-Host "Copied cypha_qt_shell.exe"

if ($null -ne $RestExe) {
    Copy-Item $RestExe -Destination $OutDir
    Write-Host "Copied cypha_rest.exe (sidecar)"
}

$ShellDest = Join-Path $OutDir "cypha_qt_shell.exe"
Write-Host "==> windeployqt (bundle Qt DLLs)"
& $WinDeployQt `
    --release `
    --no-translations `
    --no-system-d3d-compiler `
    --no-opengl-sw `
    $ShellDest
if ($LASTEXITCODE -ne 0) { throw "windeployqt failed with exit code $LASTEXITCODE" }

Set-Content -Path (Join-Path $OutDir "VERSION") -Value $Version -NoNewline

if ($WithFixtures) {
    $DemoDir = Join-Path $OutDir "share\demo_fixtures"
    New-Item -ItemType Directory -Path $DemoDir -Force | Out-Null
    $FixturesDir = Join-Path $RepoRoot "fixtures"
    foreach ($f in @("reference.cypha", "f_field.json", "train_hparams.json")) {
        $src = Join-Path $FixturesDir $f
        if (Test-Path $src) {
            Copy-Item $src -Destination $DemoDir
            Write-Host "Copied $f -> share\demo_fixtures\"
        }
    }
}

Write-Host ""
Write-Host "Package ready: $OutDir" -ForegroundColor Green
Write-Host ""
Write-Host "Run:"
Write-Host "  $OutDir\cypha_qt_shell.exe"
Write-Host ""
if ($WithFixtures) {
    Write-Host "Smoke test:"
    Write-Host "  $OutDir\cypha_qt_shell.exe --smoke share\demo_fixtures\reference.cypha"
}
