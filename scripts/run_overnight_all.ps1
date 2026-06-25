# Phase 9: D17 hybrid + D21 RPSM overnight automation, optional cell sweep,
#          merge BPC into bench/BASELINE_LOCK.json via cypha_baseline_lock.
# Usage:
#   powershell -File scripts/run_overnight_all.ps1
#   powershell -File scripts/run_overnight_all.ps1 -Fast -SkipCellSweep -NTrain 500
#   powershell -File scripts/run_overnight_all.ps1 -Medium  # 5k train, real WikiText/gutenberg
#   powershell -File scripts/run_overnight_all.ps1 -Production  # 300k train, status=production in lock
param(
    [string]$BuildDir = "native/build",
    [int]$NTrain = 300000,
    [int]$NEval = 2000,
    [int]$Threads = 1,
    [switch]$SkipCellSweep,
    [switch]$Fast,
    [switch]$Medium,
    [switch]$Production,
    [switch]$MathIntegration
)

$ErrorActionPreference = "Stop"

$root = Split-Path $PSScriptRoot -Parent
$tierCount = @($Fast, $Medium, $Production | Where-Object { $_ }).Count
if ($tierCount -gt 1) {
    throw "cannot combine -Fast, -Medium, and -Production"
}

$effectiveNTrain = $NTrain
$effectiveNEval = $NEval
if ($Fast) {
    if ($NTrain -eq 300000) { $effectiveNTrain = 200 }
    if ($NEval -eq 2000) { $effectiveNEval = 64 }
} elseif ($Medium) {
    if ($NTrain -eq 300000) { $effectiveNTrain = 5000 }
    if ($NEval -eq 2000) { $effectiveNEval = 256 }
} elseif ($Production) {
    if ($NTrain -eq 300000) { $effectiveNTrain = 300000 }
    if ($NEval -eq 2000) { $effectiveNEval = 2000 }
}

$d17Script = Join-Path $PSScriptRoot "run_d17_overnight.ps1"
$rpsmScript = Join-Path $PSScriptRoot "run_rpsm_overnight.ps1"
$lockScript = Join-Path $PSScriptRoot "update_baseline_lock.ps1"

Write-Host "== Phase 9 overnight automation (fast=$Fast medium=$Medium production=$Production n_train=$effectiveNTrain n_eval=$effectiveNEval) ==" -ForegroundColor Cyan

$overnightArgs = @{
    BuildDir = $BuildDir
    NTrain   = $effectiveNTrain
    NEval    = $effectiveNEval
    Threads  = $Threads
}
if ($Fast) {
    $overnightArgs.Fast = $true
}
if ($Medium) {
    $overnightArgs.Medium = $true
}
if ($Production) {
    $overnightArgs.Production = $true
}
if ($MathIntegration) {
    $overnightArgs.MathIntegration = $true
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
if ($Medium) {
    $lockArgs.Medium = $true
}
if ($Production) {
    $lockArgs.Production = $true
}

if ($Production) {
    $lockArgs.Production = $true
}

if ($MathIntegration) {
    Write-Host "== baseline lock: d17-math ==" -ForegroundColor Cyan
    & $lockScript -Run d17-math @lockArgs
    if ($LASTEXITCODE -ne 0) {
        throw "cypha_baseline_lock d17-math failed exit=$LASTEXITCODE"
    }
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
    if ($MathIntegration) {
        & $lockScript -Run cell-sweep @lockArgs -OutputDir bench/results/cell_sweep -MathIntegration
    } else {
        & $lockScript -Run cell-sweep @lockArgs -OutputDir bench/results/cell_sweep
    }
    if ($LASTEXITCODE -ne 0) {
        throw "cypha_baseline_lock cell-sweep failed exit=$LASTEXITCODE"
    }
}

Write-Host "Done. Updated bench/BASELINE_LOCK.json" -ForegroundColor Green
