# Phase 13/14: production overnight tier — 300k train / 2000 eval, real WikiText/gutenberg.
# Chains D17 + d21 + cell sweep + baseline-lock refresh with status=production.
# On success, runs finalize_production_overnight.ps1 (validate -Production + d27/d28 bench),
# then commit_production_lock.ps1 -DryRun (preview only). To git-commit the lock after
# validation, run commit_production_lock.ps1 -Force manually (or poll_and_finalize_overnight.ps1 -Force).
# Stderr progress from native tools ([cyphalm] train steps, [cell_sweep] variant starts) is
# captured in the transcript log alongside Write-Host output — tail the log to confirm liveness.
# Usage:
#   pwsh -File scripts/run_production_overnight.ps1
#   pwsh -File scripts/run_production_overnight.ps1 -BuildDir native/build -SkipCellSweep
param(
    [string]$BuildDir = "",
    [int]$Threads = 1,
    [switch]$SkipCellSweep,
    [switch]$MathIntegration
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "lib\NativeBenchCommon.ps1")
$BuildDir = Get-DefaultNativeBuildDir -Override $BuildDir

$root = Split-Path $PSScriptRoot -Parent
$resultsDir = Join-Path $root "bench/results"
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $resultsDir "production_overnight_$timestamp.log"

Write-Host "== Cypha production overnight (300k train / 2000 eval) ==" -ForegroundColor Cyan
Write-Host "Log: $logPath" -ForegroundColor Yellow
Write-Host "ETA: plan 8-24 hours wall-clock (D17 + D21 + 28-variant cell sweep @ 300k, single thread). Run overnight and tail this log." -ForegroundColor Yellow
Write-Host "Progress: stderr lines [cyphalm] and [cell_sweep] appear in the transcript log every ~10k train steps / per variant." -ForegroundColor Yellow

$allScript = Join-Path $PSScriptRoot "run_overnight_all.ps1"
$invokeArgs = @{
    BuildDir     = $BuildDir
    Threads      = $Threads
    Production   = $true
}
if ($SkipCellSweep) {
    $invokeArgs.SkipCellSweep = $true
}
if ($MathIntegration) {
    $invokeArgs.MathIntegration = $true
    $env:CYPHA_OVERNIGHT_MATH_INTEGRATION = "1"
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

$commitScript = Join-Path $PSScriptRoot "commit_production_lock.ps1"
Write-Host "== commit preview (manual step: commit_production_lock.ps1 -Force) ==" -ForegroundColor Cyan
& $commitScript -BuildDir $BuildDir -DryRun
if ($LASTEXITCODE -ne 0) {
    Write-Host "commit preview exited $LASTEXITCODE (lock may not be ready to commit yet)" -ForegroundColor Yellow
}

Write-Host "Done. Updated bench/BASELINE_LOCK.json (status=production). Log: $logPath" -ForegroundColor Green
