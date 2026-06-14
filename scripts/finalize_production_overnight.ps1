# Phase 14: post-production-overnight validation — lock check + d27/d28 bench domains.
# Usage:
#   pwsh -File scripts/finalize_production_overnight.ps1
#   pwsh -File scripts/finalize_production_overnight.ps1 -BuildDir native/build -LockFile bench/BASELINE_LOCK.json
param(
    [string]$BuildDir = "native/build",
    [string]$LockFile = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent

if (-not $LockFile) {
    $LockFile = Join-Path $root "bench\BASELINE_LOCK.json"
} elseif (-not [System.IO.Path]::IsPathRooted($LockFile)) {
    $LockFile = Join-Path $root $LockFile
}

$validateScript = Join-Path $PSScriptRoot "validate_baseline_lock.ps1"
$benchExe = Join-Path $root (Join-Path $BuildDir "cypha_bench_run.exe")
if (-not (Test-Path $benchExe)) {
    $benchExe = Join-Path $root (Join-Path $BuildDir "cypha_bench_run")
}
if (-not (Test-Path $benchExe)) {
    throw "missing cypha_bench_run under $BuildDir (build native first)"
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
        Write-Host "finalize: lock file not found: $Path" -ForegroundColor Red
        return
    }

    try {
        $lock = Get-Content $Path -Raw | ConvertFrom-Json
    } catch {
        Write-Host "finalize: invalid lock JSON: $($_.Exception.Message)" -ForegroundColor Red
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

Write-Host "== finalize production overnight ==" -ForegroundColor Cyan
Write-Host "  lock:  $LockFile"
Write-Host "  build: $BuildDir"

Write-Host ""
Write-Host "== validate_baseline_lock -Production ==" -ForegroundColor Cyan
& $validateScript -LockFile $LockFile -Production
$validateCode = $LASTEXITCODE
if ($validateCode -ne 0) {
    Show-LockSummary -Path $LockFile
    exit $validateCode
}

Write-Host ""
Write-Host "== cypha_bench_run --domain-tag d27 ==" -ForegroundColor Cyan
Push-Location $root
try {
    & $benchExe --domain-tag d27
    if ($LASTEXITCODE -ne 0) {
        Show-LockSummary -Path $LockFile
        exit $LASTEXITCODE
    }
} finally {
    Pop-Location
}

if (Test-DomainTagExists -Tag "d28") {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d28 ==" -ForegroundColor Cyan
    Push-Location $root
    try {
        & $benchExe --domain-tag d28
        if ($LASTEXITCODE -ne 0) {
            Show-LockSummary -Path $LockFile
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
} else {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d28 (skipped — domain not present) ==" -ForegroundColor DarkGray
}

Show-LockSummary -Path $LockFile
Write-Host ""
Write-Host "finalize_production_overnight: OK" -ForegroundColor Green
exit 0
