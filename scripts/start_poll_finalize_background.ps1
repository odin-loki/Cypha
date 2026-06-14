# Phase 18: start poll_and_finalize_overnight.ps1 in the background after manually
# launching production overnight (e.g. run_production_overnight.ps1).
#
# Usage:
#   pwsh -File scripts/start_poll_finalize_background.ps1
#   pwsh -File scripts/start_poll_finalize_background.ps1 -BuildDir native/build
#   pwsh -File scripts/start_poll_finalize_background.ps1 -LogFile bench/results/poll_finalize.log
param(
    [string]$BuildDir = "native/build",
    [string]$LogFile = "bench/results/poll_finalize.log"
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$pollScript = Join-Path $PSScriptRoot "poll_and_finalize_overnight.ps1"

if (-not (Test-Path $pollScript)) {
    throw "missing $pollScript"
}

if ([System.IO.Path]::IsPathRooted($LogFile)) {
    $resolvedLog = $LogFile
} else {
    $resolvedLog = Join-Path $root $LogFile
}

$logDir = Split-Path $resolvedLog -Parent
if ($logDir) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

$pwsh = Get-Command pwsh -ErrorAction SilentlyContinue
if ($pwsh) {
    $exe = $pwsh.Source
    $argList = @(
        "-NoProfile",
        "-File", $pollScript,
        "-BuildDir", $BuildDir,
        "-LogFile", $resolvedLog
    )
} else {
    $exe = "powershell.exe"
    $argList = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $pollScript,
        "-BuildDir", $BuildDir,
        "-LogFile", $resolvedLog
    )
}

$proc = Start-Process -FilePath $exe -ArgumentList $argList -WorkingDirectory $root -PassThru -WindowStyle Hidden
if (-not $proc) {
    throw "failed to start poll_and_finalize_overnight.ps1"
}

Write-Host "Started poll_and_finalize_overnight.ps1 in background (PID $($proc.Id))" -ForegroundColor Green
Write-Host "  build: $BuildDir" -ForegroundColor DarkGray
Write-Host "  log:   $resolvedLog" -ForegroundColor DarkGray
Write-Host "Tail the log or run watch_production_overnight.ps1 while overnight processes run." -ForegroundColor Yellow
