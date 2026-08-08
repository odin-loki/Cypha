# Phase 18: full production-complete validation - lock tier + finalize pipeline + d31/d30 bench gates.
# Usage:
#   pwsh -File scripts/validate_production_complete.ps1
#   pwsh -File scripts/validate_production_complete.ps1 -BuildDir native/build -LockFile bench/BASELINE_LOCK.json
#   pwsh -File scripts/validate_production_complete.ps1 -AllowPending   # smoke: pass when n_train < 300k
param(
    [string]$BuildDir = "native/build",
    [string]$LockFile = "",
    [switch]$AllowPending
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "lib\NativeBenchCommon.ps1")
$root = Get-CyphaRepoRoot -ScriptRoot $PSScriptRoot
$buildAbs = Resolve-NativeBuildDir -RepoRoot $root -BuildDir $BuildDir
$PRODUCTION_N_TRAIN_MIN = 300000
$PRODUCTION_STATUSES = @("production", "completed")

if (-not $LockFile) {
    $LockFile = Join-Path $root "bench\BASELINE_LOCK.json"
} elseif (-not [System.IO.Path]::IsPathRooted($LockFile)) {
    $LockFile = Join-Path $root $LockFile
}

$validateScript = Join-Path $PSScriptRoot "validate_baseline_lock.ps1"
$finalizeScript = Join-Path $PSScriptRoot "finalize_production_overnight.ps1"
$benchExe = Resolve-NativeExePath -BuildDir $buildAbs -Stem "cypha_bench_run"
if (-not $benchExe) {
    throw "missing cypha_bench_run under $buildAbs (build native first)"
}

function Test-DomainTagExists {
    param([string]$Tag)
    $profiles = Get-ChildItem -Path (Join-Path $root "bench\config") -Filter "${Tag}_*_profile.json" -ErrorAction SilentlyContinue
    if ($profiles) { return $true }
    $indexPath = Join-Path $root "bench\config\profiles_index.json"
    if (Test-Path $indexPath) {
        try {
            $index = Get-Content $indexPath -Raw | ConvertFrom-Json
            foreach ($prop in $index.PSObject.Properties) {
                if ($prop.Value.domain -eq $Tag) { return $true }
            }
        } catch { }
    }
    return $false
}

function Show-LockSummary {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        Write-Host "validate_production_complete: lock file not found: $Path" -ForegroundColor Red
        return
    }

    try {
        $lock = Get-Content $Path -Raw | ConvertFrom-Json
    } catch {
        Write-Host "validate_production_complete: invalid lock JSON: $($_.Exception.Message)" -ForegroundColor Red
        return
    }

    Write-Host ""
    Write-Host "== BASELINE_LOCK summary ==" -ForegroundColor Cyan
    foreach ($name in @("overnight_results", "rpsm_results", "cell_sweep_results")) {
        if ($lock.PSObject.Properties.Name -notcontains $name -or $null -eq $lock.$name) {
            Write-Host ("  {0,-22} (absent)" -f $name) -ForegroundColor DarkGray
            continue
        }
        $section = $lock.$name
        $nTrain = if ($section.PSObject.Properties.Name -contains "n_train") { $section.n_train } else { "?" }
        $status = if ($section.PSObject.Properties.Name -contains "status") { $section.status } else { "?" }
        $bpc = if ($section.PSObject.Properties.Name -contains "bpc") { $section.bpc } else { "?" }
        Write-Host ("  {0,-22} n_train={1,-8} status={2,-12} bpc={3}" -f $name, $nTrain, $status, $bpc)
    }
}

function Test-ProductionCompleteLock {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return @{ Ok = $false; Reason = "lock file not found: $Path" }
    }

    try {
        $lock = Get-Content $Path -Raw | ConvertFrom-Json
    } catch {
        return @{ Ok = $false; Reason = "invalid lock JSON: $($_.Exception.Message)" }
    }

    if ($lock.PSObject.Properties.Name -notcontains "overnight_results" -or $null -eq $lock.overnight_results) {
        return @{ Ok = $false; Reason = "overnight_results missing" }
    }

    $overnight = $lock.overnight_results
    $nTrain = if ($overnight.PSObject.Properties.Name -contains "n_train") { [int]$overnight.n_train } else { 0 }
    $status = if ($overnight.PSObject.Properties.Name -contains "status") { [string]$overnight.status } else { "" }

    if ($nTrain -lt $PRODUCTION_N_TRAIN_MIN) {
        return @{
            Ok = $false
            Pending = $true
            Reason = "overnight_results n_train=$nTrain < $PRODUCTION_N_TRAIN_MIN (pending_production)"
            NTrain = $nTrain
            Status = $status
        }
    }

    if ($PRODUCTION_STATUSES -notcontains $status) {
        return @{
            Ok = $false
            Pending = $true
            Reason = "overnight_results status '$status' invalid for production complete (n_train=$nTrain; expected $($PRODUCTION_STATUSES -join ', '))"
            NTrain = $nTrain
            Status = $status
        }
    }

    return @{
        Ok = $true
        Pending = $false
        Reason = "production complete (n_train=$nTrain status=$status)"
        NTrain = $nTrain
        Status = $status
    }
}

Write-Host "== validate production complete ==" -ForegroundColor Cyan
Write-Host "  lock:  $LockFile"
Write-Host "  build: $BuildDir"
if ($AllowPending) {
    Write-Host "  mode:  -AllowPending (smoke - pending_production passes)" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "== validate_baseline_lock -Production ==" -ForegroundColor Cyan
& $validateScript -LockFile $LockFile -Production
if ($LASTEXITCODE -ne 0) {
    Show-LockSummary -Path $LockFile
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "== finalize_production_overnight ==" -ForegroundColor Cyan
& $finalizeScript -BuildDir $BuildDir -LockFile $LockFile
if ($LASTEXITCODE -ne 0) {
    Show-LockSummary -Path $LockFile
    exit $LASTEXITCODE
}

if (Test-DomainTagExists -Tag "d31") {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d31 ==" -ForegroundColor Cyan
    Push-Location $root
    try {
        & $benchExe --domain-tag d31
        if ($LASTEXITCODE -ne 0) {
            Show-LockSummary -Path $LockFile
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
} else {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d31 (skipped - domain not present) ==" -ForegroundColor DarkGray
}

if (Test-DomainTagExists -Tag "d30") {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d30 ==" -ForegroundColor Cyan
    Push-Location $root
    try {
        & $benchExe --domain-tag d30
        if ($LASTEXITCODE -ne 0) {
            Show-LockSummary -Path $LockFile
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
} else {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d30 (skipped - domain not present) ==" -ForegroundColor DarkGray
}

$check = Test-ProductionCompleteLock -Path $LockFile
Show-LockSummary -Path $LockFile
Write-Host ""
if ($check.Ok) {
    Write-Host "validate_production_complete: OK - $($check.Reason)" -ForegroundColor Green
    exit 0
}

if ($AllowPending -and $check.Pending) {
    Write-Host "validate_production_complete: OK (pending) - $($check.Reason)" -ForegroundColor Yellow
    exit 0
}

Write-Host "validate_production_complete: FAIL - $($check.Reason)" -ForegroundColor Red
exit 1
