<#
.SYNOPSIS
    Package cypha_qt_shell.exe as a self-contained Windows folder.

.DESCRIPTION
    1. Runs cmake --install to copy cypha_qt_shell.exe (and cypha_rest.exe if built)
       into an output folder.
    2. Runs windeployqt to copy all required Qt DLLs alongside the exe.
    3. Optionally copies the parity fixtures for a quick smoke test.

.PARAMETER BuildDir
    Path to the CMake build directory that contains cypha_qt_shell.exe.
    Default: native\build (relative to repo root).

.PARAMETER OutDir
    Destination folder for the packaged distribution.
    Default: native\dist\cypha_qt_shell_windows

.PARAMETER QtBinDir
    Directory containing windeployqt.exe and Qt DLLs.
    Default: auto-detected from PATH.

.PARAMETER WithFixtures
    If set, copies parity_fixtures\reference.cypha and parity_fixtures\f_field.json
    into the package for a quick demo.

.EXAMPLE
    # Build first (from repo root, with Qt 6 installed on Windows):
    cmake -S native -B native\build -DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_QT=ON
    cmake --build native\build --target cypha_qt_shell
    # Then package:
    powershell -ExecutionPolicy Bypass -File native\scripts\package_windows_qt.ps1 -WithFixtures
#>

[CmdletBinding()]
param(
    [string]$BuildDir  = "native\build",
    [string]$OutDir    = "native\dist\cypha_qt_shell_windows",
    [string]$QtBinDir  = "",
    [switch]$WithFixtures
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Resolve repo root ──────────────────────────────────────────────────────
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot  = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path

$BuildDir = Join-Path $RepoRoot $BuildDir
$OutDir   = Join-Path $RepoRoot $OutDir

# ── Locate cypha_qt_shell.exe ──────────────────────────────────────────────
$candidates = @(
    (Join-Path $BuildDir "qt\cypha_qt_shell.exe"),
    (Join-Path $BuildDir "cypha_qt_shell.exe")
)
$ShellExe = $null
foreach ($c in $candidates) {
    if (Test-Path $c) { $ShellExe = $c; break }
}
if ($null -eq $ShellExe) {
    Write-Error "cypha_qt_shell.exe not found under $BuildDir.`nBuild first: cmake --build $BuildDir --target cypha_qt_shell"
}
Write-Host "Found: $ShellExe"

# ── Locate windeployqt.exe ─────────────────────────────────────────────────
$WinDeployQt = $null
if ($QtBinDir -ne "") {
    $WinDeployQt = Join-Path $QtBinDir "windeployqt.exe"
    if (-not (Test-Path $WinDeployQt)) {
        Write-Error "windeployqt.exe not found at $WinDeployQt"
    }
} else {
    $WinDeployQt = Get-Command "windeployqt.exe" -ErrorAction SilentlyContinue |
                   Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue
    if ($null -eq $WinDeployQt) {
        # Try common Qt install locations
        $qtRoots = @("C:\Qt", "$env:USERPROFILE\Qt")
        foreach ($r in $qtRoots) {
            $found = Get-ChildItem -Path $r -Filter "windeployqt.exe" -Recurse -ErrorAction SilentlyContinue |
                     Sort-Object FullName -Descending | Select-Object -First 1
            if ($found) { $WinDeployQt = $found.FullName; break }
        }
    }
    if ($null -eq $WinDeployQt) {
        Write-Error ("windeployqt.exe not found on PATH or in C:\Qt.`n" +
            "Install Qt 6 from https://www.qt.io/download and add its bin/ to PATH,`n" +
            "or pass -QtBinDir 'C:\Qt\6.x.x\msvc2022_64\bin'.")
    }
}
Write-Host "windeployqt: $WinDeployQt"

# ── Create output directory and copy exe ──────────────────────────────────
if (Test-Path $OutDir) {
    Write-Host "Cleaning existing output: $OutDir"
    Remove-Item -Recurse -Force $OutDir
}
New-Item -ItemType Directory -Path $OutDir | Out-Null

Copy-Item $ShellExe -Destination $OutDir
Write-Host "Copied cypha_qt_shell.exe -> $OutDir"

# Copy cypha_rest.exe if present
$RestExe = Join-Path $BuildDir "cypha_rest.exe"
if (-not (Test-Path $RestExe)) {
    $RestExe = Join-Path $BuildDir "qt\cypha_rest.exe"
}
if (Test-Path $RestExe) {
    Copy-Item $RestExe -Destination $OutDir
    Write-Host "Copied cypha_rest.exe -> $OutDir"
} else {
    Write-Host "cypha_rest.exe not found — skipping (optional)"
}

# ── Run windeployqt ────────────────────────────────────────────────────────
$ShellExeDest = Join-Path $OutDir "cypha_qt_shell.exe"
Write-Host "Running windeployqt..."
& $WinDeployQt `
    --no-translations `
    --no-system-d3d-compiler `
    --no-opengl-sw `
    $ShellExeDest

if ($LASTEXITCODE -ne 0) {
    Write-Error "windeployqt failed with exit code $LASTEXITCODE"
}
Write-Host "windeployqt complete."

# ── Optional: copy parity fixtures for demo ──────────────────────────────
if ($WithFixtures) {
    $FixturesDir = Join-Path $RepoRoot "parity_fixtures"
    $DemoDir     = Join-Path $OutDir "demo_fixtures"
    New-Item -ItemType Directory -Path $DemoDir -Force | Out-Null
    foreach ($f in @("reference.cypha", "f_field.json", "train_hparams.json")) {
        $src = Join-Path $FixturesDir $f
        if (Test-Path $src) {
            Copy-Item $src -Destination $DemoDir
            Write-Host "Copied $f -> demo_fixtures\"
        }
    }
}

# ── Summary ────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Package ready: $OutDir"
Write-Host ""
Write-Host "To run:"
Write-Host "   $OutDir\cypha_qt_shell.exe"
Write-Host ""
Write-Host "To smoke test (headless):"
if ($WithFixtures) {
    Write-Host "   $OutDir\cypha_qt_shell.exe --smoke demo_fixtures\reference.cypha demo_fixtures\f_field.json"
} else {
    Write-Host "   $OutDir\cypha_qt_shell.exe --smoke <path\to\reference.cypha> <path\to\f_field.json>"
}
