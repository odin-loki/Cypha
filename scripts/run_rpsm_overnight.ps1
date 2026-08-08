# D21 RPSM overnight run — 300k train token budget (see bench/config/d21_rpsm_profile.json).
# Usage:
#   pwsh -File scripts/run_rpsm_overnight.ps1
#   pwsh -File scripts/run_rpsm_overnight.ps1 -BuildDir native/build -NTrain 500
#   pwsh -File scripts/run_rpsm_overnight.ps1 -Fast  # synthetic corpus if WikiText missing
#   pwsh -File scripts/run_rpsm_overnight.ps1 -Medium  # 5k train, real WikiText/gutenberg
#   pwsh -File scripts/run_rpsm_overnight.ps1 -Production  # 300k train, status=production in lock
param(
    [string]$BuildDir = "native/build",
    [int]$NTrain = 300000,
    [int]$NEval = 2000,
    [int]$Threads = 1,
    [string]$Profile = "d21",
    [string]$Mode = "rpsm",
    [switch]$Fast,
    [switch]$Medium,
    [switch]$Production
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "lib\NativeBenchCommon.ps1")

$tierCount = @($Fast, $Medium, $Production | Where-Object { $_ }).Count
if ($tierCount -gt 1) {
    throw "cannot combine -Fast, -Medium, and -Production"
}

$root = Get-CyphaRepoRoot -ScriptRoot $PSScriptRoot
$buildAbs = Resolve-NativeBuildDir -RepoRoot $root -BuildDir $BuildDir
$exe = Resolve-NativeExePath -BuildDir $buildAbs -Stem "cyphalm_bench_native"
if (-not $exe) {
    throw "missing cyphalm_bench_native under $buildAbs (build native first)"
}

$resultsDir = Join-Path $root "bench/results"
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $resultsDir "overnight_d21_$timestamp.log"
Write-Host "Log: $logPath" -ForegroundColor Yellow

$env:CYPHA_BENCH_FULL_CORPUS = "1"
$env:CYPHA_BENCH_OVERNIGHT = "1"
$env:CYPHA_BENCH_FULL_N_TRAIN = "$NTrain"
if ($Fast) {
    $env:CYPHA_BENCH_FAST = "1"
} elseif ($NTrain -ne 300000 -and -not $Medium -and -not $Production) {
    $env:CYPHA_BENCH_FAST = "1"
}

# NOTE: the bench binary's final JSON result line must go through the PowerShell
# pipeline (not raw inherited-console stdout) or Start-Transcript can silently drop
# it when this script is invoked several `&`-levels deep (run_production_overnight.ps1
# -> run_overnight_all.ps1 -> run_rpsm_overnight.ps1), as happened on 2026-06-28 (D21
# printed nothing between its header and "Done."). Piping through Tee-Object — the
# same pattern run_d17_overnight.ps1 already uses — forces the output through the
# pipeline so Transcript reliably captures it, and also gives D21 its own log file.
# Native tools may log progress on stderr; with 2>&1 that becomes ErrorRecord output,
# so relax ErrorActionPreference for the duration of the call (matches
# Invoke-NativeWithProgressLog in run_d17_overnight.ps1).
Push-Location $buildAbs
try {
    Write-Host "== D21 RPSM overnight bench (profile=$Profile mode=$Mode n_train=$NTrain) ==" -ForegroundColor Cyan
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $exe --profile $Profile --mode $Mode --overnight --n-train $NTrain --n-eval $NEval --threads $Threads 2>&1 |
            Tee-Object -FilePath $logPath -Append
    } finally {
        $ErrorActionPreference = $prevEap
    }
    if ($LASTEXITCODE -ne 0) {
        throw "overnight run failed exit=$LASTEXITCODE"
    }
} finally {
    Pop-Location
}

Write-Host "Done. Results under bench/results or repo results/. Log: $logPath" -ForegroundColor Green
