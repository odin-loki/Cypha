# Phase 14/23: post-production-overnight validation - lock check + d27/d28 bench domains.
# Phase 23: after validate_baseline_lock -Production, when overnight_results.n_train < 300000
# and cypha_baseline_lock exists under BuildDir, best-effort refresh via
# update_baseline_lock.ps1 -Run all -Production -BuildDir (warn on fail, continue).
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
    foreach ($name in @("overnight_results", "rpsm_results", "cell_sweep_results", "math_integration_results")) {
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

$PRODUCTION_N_TRAIN_MIN = 300000
try {
    if (Test-Path $LockFile) {
        $lockForRefresh = Get-Content $LockFile -Raw | ConvertFrom-Json
        $overnightNTrain = $null
        if ($lockForRefresh.PSObject.Properties.Name -contains "overnight_results" -and $null -ne $lockForRefresh.overnight_results) {
            $overnightSec = $lockForRefresh.overnight_results
            if ($overnightSec.PSObject.Properties.Name -contains "n_train") {
                $overnightNTrain = [int]$overnightSec.n_train
            }
        }
        if ($null -ne $overnightNTrain -and $overnightNTrain -lt $PRODUCTION_N_TRAIN_MIN) {
            $lockExe = Join-Path $root (Join-Path $BuildDir "cypha_baseline_lock.exe")
            if (-not (Test-Path $lockExe)) {
                $lockExe = Join-Path $root (Join-Path $BuildDir "cypha_baseline_lock")
            }
            if (Test-Path $lockExe) {
                Write-Host ""
                Write-Host "== update_baseline_lock.ps1 -Run all -Production (overnight n_train=$overnightNTrain < $PRODUCTION_N_TRAIN_MIN) ==" -ForegroundColor Cyan
                $updateScript = Join-Path $PSScriptRoot "update_baseline_lock.ps1"
                & $updateScript -Run all -Production -BuildDir $BuildDir
                if ($LASTEXITCODE -ne 0) {
                    Write-Host "finalize: WARN update_baseline_lock failed exit=$LASTEXITCODE (best-effort; continuing)" -ForegroundColor Yellow
                }
            } else {
                Write-Host ""
                Write-Host "finalize: WARN cypha_baseline_lock not found under $BuildDir (skip lock refresh)" -ForegroundColor Yellow
            }
        }
    }
} catch {
    Write-Host ""
    Write-Host "finalize: WARN lock refresh check failed: $($_.Exception.Message)" -ForegroundColor Yellow
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
    Write-Host "== cypha_bench_run --domain-tag d28 (skipped - domain not present) ==" -ForegroundColor DarkGray
}

if (Test-DomainTagExists -Tag "d42") {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d42 (math integration production) ==" -ForegroundColor Cyan
    Push-Location $root
    try {
        & $benchExe --domain-tag d42
        if ($LASTEXITCODE -ne 0) {
            Show-LockSummary -Path $LockFile
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
} else {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d42 (skipped - domain not present) ==" -ForegroundColor DarkGray
}

if (Test-DomainTagExists -Tag "d53") {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d53 (production preset ship lock) ==" -ForegroundColor Cyan
    Push-Location $root
    try {
        & $benchExe --domain-tag d53
        if ($LASTEXITCODE -ne 0) {
            Show-LockSummary -Path $LockFile
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
} else {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d53 (skipped - domain not present) ==" -ForegroundColor DarkGray
}

if (Test-DomainTagExists -Tag "d54") {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d54 (production math certificate) ==" -ForegroundColor Cyan
    Push-Location $root
    try {
        & $benchExe --domain-tag d54
        if ($LASTEXITCODE -ne 0) {
            Show-LockSummary -Path $LockFile
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
} else {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d54 (skipped - domain not present) ==" -ForegroundColor DarkGray
}

if (Test-DomainTagExists -Tag "d56") {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d56 (cell sweep math integration) ==" -ForegroundColor Cyan
    Push-Location $root
    try {
        & $benchExe --domain-tag d56
        if ($LASTEXITCODE -ne 0) {
            Show-LockSummary -Path $LockFile
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
} else {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d56 (skipped - domain not present) ==" -ForegroundColor DarkGray
}

if (Test-DomainTagExists -Tag "d57") {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d57 (production cell sweep math certificate) ==" -ForegroundColor Cyan
    Push-Location $root
    try {
        & $benchExe --domain-tag d57
        if ($LASTEXITCODE -ne 0) {
            Show-LockSummary -Path $LockFile
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
} else {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d57 (skipped - domain not present) ==" -ForegroundColor DarkGray
}

if (Test-DomainTagExists -Tag "d58") {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d58 (production overnight math complete) ==" -ForegroundColor Cyan
    Push-Location $root
    try {
        & $benchExe --domain-tag d58
        if ($LASTEXITCODE -ne 0) {
            Show-LockSummary -Path $LockFile
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
} else {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d58 (skipped - domain not present) ==" -ForegroundColor DarkGray
}

if (Test-DomainTagExists -Tag "d59") {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d59 (kernel blend floor grid) ==" -ForegroundColor Cyan
    Push-Location $root
    try {
        & $benchExe --domain-tag d59
        if ($LASTEXITCODE -ne 0) {
            Show-LockSummary -Path $LockFile
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
}

if (Test-DomainTagExists -Tag "d60") {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d60 (excess grad margin grid) ==" -ForegroundColor Cyan
    Push-Location $root
    try {
        & $benchExe --domain-tag d60
        if ($LASTEXITCODE -ne 0) {
            Show-LockSummary -Path $LockFile
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
}

if (Test-DomainTagExists -Tag "d61") {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d61 (excess grad scale grid) ==" -ForegroundColor Cyan
    Push-Location $root
    try {
        & $benchExe --domain-tag d61
        if ($LASTEXITCODE -ne 0) {
            Show-LockSummary -Path $LockFile
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
}

if (Test-DomainTagExists -Tag "d62") {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d62 (math ablation stack complete) ==" -ForegroundColor Cyan
    Push-Location $root
    try {
        & $benchExe --domain-tag d62
        if ($LASTEXITCODE -ne 0) {
            Show-LockSummary -Path $LockFile
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
}

if (Test-DomainTagExists -Tag "d63") {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d63 (reu forget blend grid) ==" -ForegroundColor Cyan
    Push-Location $root
    try {
        & $benchExe --domain-tag d63
        if ($LASTEXITCODE -ne 0) {
            Show-LockSummary -Path $LockFile
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
}

if (Test-DomainTagExists -Tag "d64") {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d64 (kappa trajectory window grid) ==" -ForegroundColor Cyan
    Push-Location $root
    try {
        & $benchExe --domain-tag d64
        if ($LASTEXITCODE -ne 0) {
            Show-LockSummary -Path $LockFile
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
}

if (Test-DomainTagExists -Tag "d65") {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d65 (navigation loss warmup grid) ==" -ForegroundColor Cyan
    Push-Location $root
    try {
        & $benchExe --domain-tag d65
        if ($LASTEXITCODE -ne 0) {
            Show-LockSummary -Path $LockFile
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
}

if (Test-DomainTagExists -Tag "d66") {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d66 (free energy beta grid) ==" -ForegroundColor Cyan
    Push-Location $root
    try {
        & $benchExe --domain-tag d66
        if ($LASTEXITCODE -ne 0) {
            Show-LockSummary -Path $LockFile
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
}

if (Test-DomainTagExists -Tag "d67") {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d67 (kernel blend grid) ==" -ForegroundColor Cyan
    Push-Location $root
    try {
        & $benchExe --domain-tag d67
        if ($LASTEXITCODE -ne 0) {
            Show-LockSummary -Path $LockFile
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
}

if (Test-DomainTagExists -Tag "d68") {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d68 (kernel m grid) ==" -ForegroundColor Cyan
    Push-Location $root
    try {
        & $benchExe --domain-tag d68
        if ($LASTEXITCODE -ne 0) {
            Show-LockSummary -Path $LockFile
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
}

if (Test-DomainTagExists -Tag "d69") {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d69 (hybrid blend logit grid) ==" -ForegroundColor Cyan
    Push-Location $root
    try {
        & $benchExe --domain-tag d69
        if ($LASTEXITCODE -ne 0) {
            Show-LockSummary -Path $LockFile
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
}

if (Test-DomainTagExists -Tag "d70") {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d70 (mdl forget max norm grid) ==" -ForegroundColor Cyan
    Push-Location $root
    try {
        & $benchExe --domain-tag d70
        if ($LASTEXITCODE -ne 0) {
            Show-LockSummary -Path $LockFile
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
}

if (Test-DomainTagExists -Tag "d71") {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d71 (kernel lr scale grid) ==" -ForegroundColor Cyan
    Push-Location $root
    try {
        & $benchExe --domain-tag d71
        if ($LASTEXITCODE -ne 0) {
            Show-LockSummary -Path $LockFile
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
}

if (Test-DomainTagExists -Tag "d72") {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d72 (alpha init grid) ==" -ForegroundColor Cyan
    Push-Location $root
    try {
        & $benchExe --domain-tag d72
        if ($LASTEXITCODE -ne 0) {
            Show-LockSummary -Path $LockFile
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
}

if (Test-DomainTagExists -Tag "d73") {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d73 (hybrid blend lr grid) ==" -ForegroundColor Cyan
    Push-Location $root
    try {
        & $benchExe --domain-tag d73
        if ($LASTEXITCODE -ne 0) {
            Show-LockSummary -Path $LockFile
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
}

if (Test-DomainTagExists -Tag "d74") {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d74 (n experts grid) ==" -ForegroundColor Cyan
    Push-Location $root
    try {
        & $benchExe --domain-tag d74
        if ($LASTEXITCODE -ne 0) {
            Show-LockSummary -Path $LockFile
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
}

if (Test-DomainTagExists -Tag "d75") {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d75 (max memory slots grid) ==" -ForegroundColor Cyan
    Push-Location $root
    try {
        & $benchExe --domain-tag d75
        if ($LASTEXITCODE -ne 0) {
            Show-LockSummary -Path $LockFile
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
}

if (Test-DomainTagExists -Tag "d76") {
    Write-Host ""
    Write-Host "== cypha_bench_run --domain-tag d76 (compress interval grid) ==" -ForegroundColor Cyan
    Push-Location $root
    try {
        & $benchExe --domain-tag d76
        if ($LASTEXITCODE -ne 0) {
            Show-LockSummary -Path $LockFile
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
}

Show-LockSummary -Path $LockFile
Write-Host ""
Write-Host "finalize_production_overnight: OK" -ForegroundColor Green
exit 0
