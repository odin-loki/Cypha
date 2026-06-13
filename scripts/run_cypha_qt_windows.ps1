#Requires -Version 5.1
<#
.SYNOPSIS
  Launch native cypha_rest and the Cypha Qt shell (Windows).

.DESCRIPTION
  Default layout matches a Visual Studio generator build:
    native\build-windows-msvc\Release\cypha_rest.exe
    native\build-windows-msvc\qt\Release\cypha_qt_shell.exe

  Configure example (Qt 6 MSVC kit; adjust path if yours differs):
    cd native
    cmake -G "Visual Studio 18 2026" -A x64 -B build-windows-msvc `
      -DCYPHA_BUILD_QT=ON -DCMAKE_PREFIX_PATH="C:/Qt/6.11.0/msvc2022_64"
    cmake --build build-windows-msvc --config Release --target cypha_rest cypha_qt_shell
    & "C:\Qt\6.11.0\msvc2022_64\bin\windeployqt.exe" --release `
      build-windows-msvc\qt\Release\cypha_qt_shell.exe

.PARAMETER NoServer
  Do not start cypha_rest (shell only; use native features or point REST URL at another server).

.PARAMETER BuildDir
  CMake binary directory (default: native\build-windows-msvc under repo root).

.PARAMETER Model
  .cypha file passed to cypha_rest --cypha (default: fixtures\reference.cypha).

.PARAMETER Listen
  cypha_rest --listen value (default: 127.0.0.1:8099).

.PARAMETER QtBin
  Optional: directory containing Qt DLLs if you did not run windeployqt (e.g. C:\Qt\6.11.0\msvc2022_64\bin).
#>
[CmdletBinding()]
param(
  [switch] $NoServer,
  [string] $BuildDir = "",
  [string] $Model = "",
  [string] $Listen = "127.0.0.1:8099",
  [string] $QtBin = ""
)

$ErrorActionPreference = "Stop"
# PSScriptRoot = ...\Cypha\scripts ; repo root = parent
$RepoRoot = Split-Path $PSScriptRoot -Parent
if (-not (Test-Path (Join-Path $RepoRoot "native\CMakeLists.txt"))) {
  Write-Error "Could not find native\CMakeLists.txt - run this script from the Cypha repo (scripts\run_cypha_qt_windows.ps1)."
}

if (-not $BuildDir) {
  $BuildDir = Join-Path $RepoRoot "native\build-windows-msvc"
}

$RestExe = Join-Path $BuildDir "Release\cypha_rest.exe"
$ShellExe = Join-Path $BuildDir "qt\Release\cypha_qt_shell.exe"

if (-not $Model) {
  $Model = Join-Path $RepoRoot "fixtures\reference.cypha"
}

if (-not (Test-Path $RestExe)) {
  Write-Error "Missing $RestExe - configure and build native (see script header)."
}
if (-not (Test-Path $ShellExe)) {
  Write-Error "Missing $ShellExe - build with -DCYPHA_BUILD_QT=ON and Qt CMAKE_PREFIX_PATH."
}
if (-not (Test-Path $Model)) {
  Write-Error "Model not found: $Model"
}

if ($QtBin -and (Test-Path $QtBin)) {
  $env:PATH = $QtBin + [System.IO.Path]::PathSeparator + $env:PATH
}

if (-not $NoServer) {
  Write-Host "Starting cypha_rest: $Listen  model: $Model"
  Start-Process -FilePath $RestExe -ArgumentList @("--listen", $Listen, "--cypha", $Model) -WorkingDirectory $RepoRoot
  Start-Sleep -Seconds 1
  $baseUrl = if ($Listen -match ":") { "http://$Listen" } else { "http://127.0.0.1:8099" }
  Write-Host "REST base URL (paste in shell if needed): $baseUrl"
} else {
  Write-Host "Skipping cypha_rest (-NoServer)."
}

Write-Host "Starting cypha_qt_shell..."
Start-Process -FilePath $ShellExe -WorkingDirectory (Split-Path $ShellExe -Parent)

Write-Host 'Done. Close this window when finished (server may still run in another console window).'
