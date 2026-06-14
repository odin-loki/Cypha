# Phase 13/14: production overnight tier — 300k train / 2000 eval, real WikiText/gutenberg.
# Chains D17 + d21 + cell sweep + baseline-lock refresh with status=production.
# On success, runs finalize_production_overnight.ps1 (validate -Production + d27/d28 bench).
# Usage:
#   pwsh -File scripts/run_production_overnight.ps1
#   pwsh -File scripts/run_production_overnight.ps1 -BuildDir native/build -SkipCellSweep
param(
    [string]$BuildDir = "native/build",
    [int]$Threads = 1,
    [switch]$SkipCellSweep
)

$ErrorActionPreference = "Stop"

$root = Split-Path $PSScriptRoot -Parent
$resultsDir = Join-Path $root "bench/results"
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $resultsDir "production_overnight_$timestamp.log"

Write-Host "== Cypha production overnight (300k train / 2000 eval) ==" -ForegroundColor Cyan
Write-Host "Log: $logPath" -ForegroundColor Yellow
Write-Host "ETA: plan 8-24 hours wall-clock (D17 + D21 + 28-variant cell sweep @ 300k, single thread). Run overnight and tail this log." -ForegroundColor Yellow

$allScript = Join-Path $PSScriptRoot "run_overnight_all.ps1"
$invokeArgs = @{
    BuildDir     = $BuildDir
    Threads      = $Threads
    Production   = $true
}
if ($SkipCellSweep) {
    $invokeArgs.SkipCellSweep = $true
}

Start-Transcript -Path $logPath | Out-Null
try {
    & $allScript @invokeArgs
    if ($LASTEXITCODE -ne 0) {
        throw "production overnight failed exit=$LASTEXITCODE"
    }
} finally {
    Stop-Transcript | Out-Null
}

$finalizeScript = Join-Path $PSScriptRoot "finalize_production_overnight.ps1"
Write-Host "== finalize production overnight ==" -ForegroundColor Cyan
& $finalizeScript -BuildDir $BuildDir
if ($LASTEXITCODE -ne 0) {
    throw "finalize_production_overnight failed exit=$LASTEXITCODE"
}

Write-Host "Done. Updated bench/BASELINE_LOCK.json (status=production). Log: $logPath" -ForegroundColor Green
