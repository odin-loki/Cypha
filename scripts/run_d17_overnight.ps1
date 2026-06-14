# D17 WikiText-2 overnight run — 300k train token budget (see bench/config/d17_wikitext_overnight_profile.json).
# Usage:
#   pwsh -File scripts/run_d17_overnight.ps1
#   pwsh -File scripts/run_d17_overnight.ps1 -BuildDir native/build -NTrain 500
param(
    [string]$BuildDir = "native/build",
    [int]$NTrain = 300000,
    [int]$NEval = 2000,
    [int]$Threads = 1,
    [string]$Profile = "d17",
    [string]$Mode = "hybrid",
    [switch]$CellSweep
)

$ErrorActionPreference = "Stop"

$root = Split-Path $PSScriptRoot -Parent
$exe = Join-Path $root (Join-Path $BuildDir "cyphalm_bench_native.exe")
if (-not (Test-Path $exe)) {
    $exe = Join-Path $root (Join-Path $BuildDir "cyphalm_bench_native")
}
if (-not (Test-Path $exe)) {
    throw "missing cyphalm_bench_native under $BuildDir (build native first)"
}

$env:CYPHA_BENCH_FULL_CORPUS = "1"
$env:CYPHA_BENCH_OVERNIGHT = "1"
$env:CYPHA_BENCH_FULL_N_TRAIN = "$NTrain"

Push-Location (Join-Path $root $BuildDir)
try {
    if ($CellSweep) {
        $sweepExe = Join-Path (Get-Location) "cypha_cell_hypothesis_sweep.exe"
        if (-not (Test-Path $sweepExe)) {
            $sweepExe = Join-Path (Get-Location) "cypha_cell_hypothesis_sweep"
        }
        if (-not (Test-Path $sweepExe)) {
            throw "missing cypha_cell_hypothesis_sweep in $BuildDir"
        }
        Write-Host "== cell hypothesis overnight sweep (28 variants, n_train=$NTrain) ==" -ForegroundColor Cyan
        & $sweepExe --overnight-sweep --profile $Profile --n-train $NTrain --n-eval $NEval --threads $Threads
    } else {
        Write-Host "== D17 overnight bench (profile=$Profile mode=$Mode n_train=$NTrain) ==" -ForegroundColor Cyan
        & $exe --profile $Profile --mode $Mode --overnight --n-train $NTrain --n-eval $NEval --threads $Threads
    }
    if ($LASTEXITCODE -ne 0) {
        throw "overnight run failed exit=$LASTEXITCODE"
    }
} finally {
    Pop-Location
}

Write-Host "Done. Results under bench/results or repo results/ (cell sweep)." -ForegroundColor Green
