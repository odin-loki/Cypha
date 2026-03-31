#Requires -Version 5.1
<#
.SYNOPSIS
  Configure and build native targets inside WSL (Linux ELF in native/build-wsl), then run ctest from that environment.

.DESCRIPTION
  Uses GNU toolchains in WSL2. Artifacts live under native/build-wsl on the Windows filesystem (/mnt/c/...).
  Use a Windows path with forward slashes for wslpath (the script normalizes backslashes).

.PARAMETER BuildType
  CMake CMAKE_BUILD_TYPE (default Release).

.PARAMETER CtestRegex
  If set, passed to ctest -R (e.g. native_experiment_db). Empty runs all tests.

.PARAMETER SkipTests
  Build only; do not run ctest.

.PARAMETER ConfigureOnly
  Run cmake .. only (no build, no tests).
#>
param(
  [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
  [string]$BuildType = "Release",
  [string]$CtestRegex = "",
  [switch]$SkipTests,
  [switch]$ConfigureOnly
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$winFwd = $RepoRoot -replace "\\", "/"
if ($winFwd -notmatch "^[A-Za-z]:") {
  throw "Repo path must be drive-based: $RepoRoot"
}

$unixRepo = (wsl wslpath -u $winFwd).Trim()
if ([string]::IsNullOrWhiteSpace($unixRepo)) {
  throw "wslpath -u failed for $winFwd"
}

$ctestSuffix = if ($CtestRegex -ne "") { " -R $CtestRegex" } else { "" }

# Single-line bash avoids CRLF from Windows line endings breaking `set -o pipefail`.
if ($ConfigureOnly) {
  $bashBody = "set -euo pipefail && cd '$unixRepo/native' && mkdir -p build-wsl && cd build-wsl && cmake .. -DCMAKE_BUILD_TYPE=$BuildType"
}
elseif ($SkipTests) {
  $bashBody = "set -euo pipefail && cd '$unixRepo/native' && mkdir -p build-wsl && cd build-wsl && cmake .. -DCMAKE_BUILD_TYPE=$BuildType && cmake --build . --parallel `$(nproc)"
}
else {
  $bashBody = "set -euo pipefail && cd '$unixRepo/native' && mkdir -p build-wsl && cd build-wsl && cmake .. -DCMAKE_BUILD_TYPE=$BuildType && cmake --build . --parallel `$(nproc) && ctest --output-on-failure$ctestSuffix"
}

wsl -e bash -lc "$bashBody"
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Host "WSL native build directory: $unixRepo/native/build-wsl"
