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
    [string]$BuildDir = "",
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
. (Join-Path $PSScriptRoot "lib\NativeBenchCommon.ps1")

$tierCount = @($Fast, $Medium, $Production | Where-Object { $_ }).Count
if ($tierCount -gt 1) {
    throw "cannot combine -Fast, -Medium, and -Production"
}

$root = Get-CyphaRepoRoot -ScriptRoot $PSScriptRoot
$BuildDir = Get-DefaultNativeBuildDir -Override $BuildDir
$buildAbs = Resolve-NativeBuildDir -RepoRoot $root -BuildDir $BuildDir
$resultsDir = Join-Path $root "bench/results"
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $resultsDir "overnight_d17_$timestamp.log"
Write-Host "Log: $logPath" -ForegroundColor Yellow

$exe = Resolve-NativeExePath -BuildDir $buildAbs -Stem "cyphalm_bench_native"
if (-not $exe) {
    throw "missing cyphalm_bench_native under $buildAbs (build native first)"
}
"$(Get-Date -Format o) starting $exe n_train=$NTrain n_eval=$NEval mode=$Mode profile=$Profile build=$buildAbs" | Out-File -FilePath $logPath -Encoding utf8

$env:CYPHA_BENCH_FULL_CORPUS = "1"
$env:CYPHA_BENCH_OVERNIGHT = "1"
$env:CYPHA_BENCH_FULL_N_TRAIN = "$NTrain"
if ($Fast) {
    $env:CYPHA_BENCH_FAST = "1"
} elseif ($NTrain -ne 300000 -and -not $Medium -and -not $Production) {
    $env:CYPHA_BENCH_FAST = "1"
}

$useMathIntegration = $MathIntegration -or ($env:CYPHA_OVERNIGHT_MATH_INTEGRATION -eq "1")

Push-Location $buildAbs
try {
    if ($CellSweep) {
        $sweepExe = Resolve-NativeExePath -BuildDir $buildAbs -Stem "cypha_cell_hypothesis_sweep"
        if (-not $sweepExe) {
            throw "missing cypha_cell_hypothesis_sweep in $buildAbs"
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
