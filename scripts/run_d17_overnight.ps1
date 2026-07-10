# D17 WikiText-2 overnight run — 300k train token budget (see bench/config/d17_wikitext_overnight_profile.json).
# Usage:
#   pwsh -File scripts/run_d17_overnight.ps1
#   pwsh -File scripts/run_d17_overnight.ps1 -BuildDir native/build -NTrain 500
#   pwsh -File scripts/run_d17_overnight.ps1 -Fast  # synthetic corpus if WikiText missing
#   pwsh -File scripts/run_d17_overnight.ps1 -Medium  # 5k train, real WikiText/gutenberg
#   pwsh -File scripts/run_d17_overnight.ps1 -Production  # 300k train, status=production in lock
#   pwsh -File scripts/run_d17_overnight.ps1 -MathIntegration  # hybrid + profile-guided math loss
#   $env:CYPHA_OVERNIGHT_MATH_INTEGRATION = "1"; pwsh -File scripts/run_d17_overnight.ps1
param(
    [string]$BuildDir = "native/build",
    [int]$NTrain = 300000,
    [int]$NEval = 2000,
    [int]$Threads = 1,
    [string]$Profile = "d17",
    [string]$Mode = "hybrid",
    [switch]$CellSweep,
    [switch]$Fast,
    [switch]$Medium,
    [switch]$Production,
    [switch]$MathIntegration
)

$ErrorActionPreference = "Stop"

$tierCount = @($Fast, $Medium, $Production | Where-Object { $_ }).Count
if ($tierCount -gt 1) {
    throw "cannot combine -Fast, -Medium, and -Production"
}

$root = Split-Path $PSScriptRoot -Parent
$resultsDir = Join-Path $root "bench/results"
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $resultsDir "overnight_d17_$timestamp.log"
Write-Host "Log: $logPath" -ForegroundColor Yellow

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
if ($Fast) {
    $env:CYPHA_BENCH_FAST = "1"
} elseif ($NTrain -ne 300000 -and -not $Medium -and -not $Production) {
    $env:CYPHA_BENCH_FAST = "1"
}

$useMathIntegration = $MathIntegration -or ($env:CYPHA_OVERNIGHT_MATH_INTEGRATION -eq "1")

function Invoke-NativeWithProgressLog {
    param(
        [string]$Exe,
        [string[]]$NativeArgs,
        [string]$LogPath
    )
    # Native tools log progress on stderr ([cyphalm], [cell_sweep]). With 2>&1 that becomes
    # ErrorRecord output; ErrorActionPreference Stop would abort on the first progress line.
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Exe @NativeArgs 2>&1 | Tee-Object -FilePath $LogPath -Append
    } finally {
        $ErrorActionPreference = $prevEap
    }
}

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
        $sweepArgs = @(
            "--overnight-sweep"
            "--profile", $Profile
            "--n-train", $NTrain
            "--n-eval", $NEval
            "--threads", $Threads
        )
        if ($NTrain -ge 5000) {
            $cellSweepOut = Join-Path $root "bench\results\cell_sweep"
            $sweepArgs += @("--output-dir", $cellSweepOut)
        }
        if ($useMathIntegration) {
            $sweepArgs += @("--intelligence-profile", "--math-integration")
        }
        Invoke-NativeWithProgressLog -Exe $sweepExe -NativeArgs $sweepArgs -LogPath $logPath
    } else {
        $benchArgs = @(
            "--profile", $Profile,
            "--mode", $Mode,
            "--overnight",
            "--n-train", $NTrain,
            "--n-eval", $NEval,
            "--threads", $Threads
        )
        if ($useMathIntegration) {
            $benchArgs += @("--math-integration", "--intelligence-profile")
        }
        $mathNote = if ($useMathIntegration) { " math-integration" } else { "" }
        Write-Host "== D17 overnight bench (profile=$Profile mode=$Mode n_train=$NTrain$mathNote) ==" -ForegroundColor Cyan
        Invoke-NativeWithProgressLog -Exe $exe -NativeArgs $benchArgs -LogPath $logPath
    }
    if ($LASTEXITCODE -ne 0) {
        throw "overnight run failed exit=$LASTEXITCODE"
    }
} finally {
    Pop-Location
}

Write-Host "Done. Results under bench/results (cell sweep: bench/results/cell_sweep). Log: $logPath" -ForegroundColor Green
