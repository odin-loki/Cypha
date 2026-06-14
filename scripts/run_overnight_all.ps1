# Phase 9: D17 hybrid + D21 RPSM overnight automation, optional cell sweep,
#          merge BPC into bench/BASELINE_LOCK.json via cypha_baseline_lock.
# Usage:
#   powershell -File scripts/run_overnight_all.ps1
#   powershell -File scripts/run_overnight_all.ps1 -Fast -SkipCellSweep -NTrain 500
param(
    [string]$BuildDir = "native/build",
    [int]$NTrain = 300000,
    [int]$NEval = 2000,
    [int]$Threads = 1,
    [switch]$SkipCellSweep,
    [switch]$Fast
)

$ErrorActionPreference = "Stop"

$root = Split-Path $PSScriptRoot -Parent
$effectiveNTrain = $NTrain
$effectiveNEval = $NEval
if ($Fast) {
    if ($NTrain -eq 300000) { $effectiveNTrain = 200 }
    if ($NEval -eq 2000) { $effectiveNEval = 64 }
}

$d17Script = Join-Path $PSScriptRoot "run_d17_overnight.ps1"
$rpsmScript = Join-Path $PSScriptRoot "run_rpsm_overnight.ps1"
$lockScript = Join-Path $PSScriptRoot "update_baseline_lock.ps1"

Write-Host "== Phase 9 overnight automation (fast=$Fast n_train=$effectiveNTrain n_eval=$effectiveNEval) ==" -ForegroundColor Cyan

$overnightArgs = @{
    BuildDir = $BuildDir
    NTrain   = $effectiveNTrain
    NEval    = $effectiveNEval
    Threads  = $Threads
}
if ($Fast) {
    $overnightArgs.Fast = $true
}

Write-Host "== D17 hybrid overnight ==" -ForegroundColor Cyan
& $d17Script @overnightArgs
if ($LASTEXITCODE -ne 0) {
    throw "run_d17_overnight failed exit=$LASTEXITCODE"
}

Write-Host "== D21 RPSM overnight ==" -ForegroundColor Cyan
& $rpsmScript @overnightArgs
if ($LASTEXITCODE -ne 0) {
    throw "run_rpsm_overnight failed exit=$LASTEXITCODE"
}

if (-not $SkipCellSweep) {
    Write-Host "== cell hypothesis overnight sweep ==" -ForegroundColor Cyan
    & $d17Script @overnightArgs -CellSweep
    if ($LASTEXITCODE -ne 0) {
        throw "cell sweep overnight failed exit=$LASTEXITCODE"
    }
}

$lockArgs = @{
    BuildDir = $BuildDir
    NTrain   = $effectiveNTrain
    NEval    = $effectiveNEval
    Threads  = $Threads
}
if ($Fast) {
    $lockArgs.Fast = $true
}

Write-Host "== baseline lock: d17 ==" -ForegroundColor Cyan
& $lockScript -Run d17 @lockArgs
if ($LASTEXITCODE -ne 0) {
    throw "cypha_baseline_lock d17 failed exit=$LASTEXITCODE"
}

Write-Host "== baseline lock: d21 ==" -ForegroundColor Cyan
& $lockScript -Run d21 @lockArgs
if ($LASTEXITCODE -ne 0) {
    throw "cypha_baseline_lock d21 failed exit=$LASTEXITCODE"
}

if (-not $SkipCellSweep) {
    Write-Host "== baseline lock: cell-sweep ==" -ForegroundColor Cyan
    & $lockScript -Run cell-sweep @lockArgs
    if ($LASTEXITCODE -ne 0) {
        throw "cypha_baseline_lock cell-sweep failed exit=$LASTEXITCODE"
    }
}

Write-Host "Done. Updated bench/BASELINE_LOCK.json" -ForegroundColor Green
