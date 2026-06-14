# D21 RPSM overnight run — 300k train token budget (see bench/config/d21_rpsm_profile.json).
# Usage:
#   pwsh -File scripts/run_rpsm_overnight.ps1
#   pwsh -File scripts/run_rpsm_overnight.ps1 -BuildDir native/build -NTrain 500
#   pwsh -File scripts/run_rpsm_overnight.ps1 -Fast  # synthetic corpus if WikiText missing
param(
    [string]$BuildDir = "native/build",
    [int]$NTrain = 300000,
    [int]$NEval = 2000,
    [int]$Threads = 1,
    [string]$Profile = "d21",
    [string]$Mode = "rpsm",
    [switch]$Fast
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
if ($Fast -or $NTrain -ne 300000) {
    $env:CYPHA_BENCH_FAST = "1"
}

Push-Location (Join-Path $root $BuildDir)
try {
    Write-Host "== D21 RPSM overnight bench (profile=$Profile mode=$Mode n_train=$NTrain) ==" -ForegroundColor Cyan
    & $exe --profile $Profile --mode $Mode --overnight --n-train $NTrain --n-eval $NEval --threads $Threads
    if ($LASTEXITCODE -ne 0) {
        throw "overnight run failed exit=$LASTEXITCODE"
    }
} finally {
    Pop-Location
}

Write-Host "Done. Results under bench/results or repo results/." -ForegroundColor Green
