# Wait for cypha_cell_hypothesis_sweep to exit, then refresh cell_sweep_results in BASELINE_LOCK.
# Usage: powershell -File scripts/wait_cell_sweep_and_lock.ps1 [-AutoCommit]
param(
    [string]$BuildDir = "",
    [int]$PollSeconds = 60,
    [switch]$AutoCommit
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "lib\NativeBenchCommon.ps1")

$root = Get-CyphaRepoRoot -ScriptRoot $PSScriptRoot
$BuildDir = Get-DefaultNativeBuildDir -Override $BuildDir
$resultsDir = Join-Path $root "bench\results"
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null
$logPath = Join-Path $resultsDir ("cell_sweep_wait_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

function Write-Log([string]$Msg) {
    $line = "$(Get-Date -Format o) $Msg"
    Write-Host $line
    $line | Out-File -FilePath $logPath -Append -Encoding utf8
}

Write-Log "waiting for cypha_cell_hypothesis_sweep (poll=${PollSeconds}s)"
while (Get-Process -Name "cypha_cell_hypothesis_sweep" -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds $PollSeconds
}
Write-Log "cell sweep finished; checking artifacts"

$sweepDir = Join-Path $root "bench\results\cell_sweep"
$variantCount = 0
if (Test-Path $sweepDir) {
    $variantCount = @(Get-ChildItem -Path $sweepDir -Filter "variant_*.json" -ErrorAction SilentlyContinue).Count
}
$manifestPath = Join-Path $sweepDir "manifest.json"
$hasManifest = Test-Path $manifestPath
Write-Log "cell_sweep artifacts: variant_json=$variantCount manifest=$hasManifest"

if ($variantCount -lt 28 -and -not $hasManifest) {
    Write-Log "FAIL cell sweep incomplete (variants=$variantCount, expected >=28 or manifest.json)"
    exit 1
}

Write-Log "updating lock"

$updateScript = Join-Path $PSScriptRoot "update_baseline_lock.ps1"
& $updateScript -Run "cell-sweep" -Production -MathIntegration -BuildDir $BuildDir
if ($LASTEXITCODE -ne 0) {
    Write-Log "FAIL update_baseline_lock cell-sweep exit=$LASTEXITCODE"
    exit $LASTEXITCODE
}

$validateScript = Join-Path $PSScriptRoot "validate_baseline_lock.ps1"
& $validateScript -Production
if ($LASTEXITCODE -ne 0) {
    Write-Log "FAIL validate_baseline_lock -Production exit=$LASTEXITCODE"
    exit $LASTEXITCODE
}

if ($AutoCommit) {
    Push-Location $root
    try {
        git add bench/BASELINE_LOCK.json
        git commit -m "bench: lock production cell sweep results (n_train=300000)"
        if ($LASTEXITCODE -ne 0) {
            Write-Log "WARN git commit exit=$LASTEXITCODE (nothing to commit?)"
        } else {
            Write-Log "committed cell sweep lock"
            git push origin HEAD
            if ($LASTEXITCODE -ne 0) {
                Write-Log "WARN git push exit=$LASTEXITCODE"
            }
        }
    } finally {
        Pop-Location
    }
}

$validateAll = Join-Path $PSScriptRoot "cypha_native_validate_all.ps1"
if (Test-Path $validateAll) {
    Write-Log "running cypha_native_validate_all.ps1 -SkipBuild (CYPHA_VALIDATE_PRODUCTION=1)"
    $env:CYPHA_VALIDATE_PRODUCTION = "1"
    $env:CYPHA_VALIDATE_OVERNIGHT_COMPLETE = "1"
    & $validateAll -SkipBuild
    if ($LASTEXITCODE -ne 0) {
        Write-Log "WARN cypha_native_validate_all exit=$LASTEXITCODE"
    } else {
        Write-Log "cypha_native_validate_all OK"
    }
}

Write-Log "OK log=$logPath"