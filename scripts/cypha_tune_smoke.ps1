# Run cypha_tune_run on all native tune smoke sweep configs.
param(
    [string]$BuildDir = "C:\Temp\cypha_full_cpp_build",
    [int]$MaxCells = 4,
    [switch]$DryRun,
    [switch]$Write,
    [switch]$SkipMissingExe
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$tuneExe = Join-Path $BuildDir "cypha_tune_run.exe"
$configDir = Join-Path $root "cypha_bench\config"
$outDir = Join-Path $root "cypha_bench\artifacts\tuning"

$configs = @(
    "cyphalm_hybrid_lstm_tune_smoke.json",
    "cyphalm_d17_phase1c_tune_smoke.json",
    "cypha_branch_a_encoder_tune_smoke.json"
)

if (-not (Test-Path $tuneExe)) {
    if ($SkipMissingExe) {
        Write-Host "SKIP cypha_tune_smoke: missing $tuneExe" -ForegroundColor Yellow
        exit 0
    }
    throw "missing $tuneExe"
}

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$results = [ordered]@{}
$allOk = $true

Write-Host "Cypha tune smoke" -ForegroundColor Cyan
Write-Host "  build:     $BuildDir"
Write-Host "  max_cells: $MaxCells"
Write-Host "  dry_run:   $DryRun"
Write-Host ""

foreach ($cfgName in $configs) {
    $cfgPath = Join-Path $configDir $cfgName
    if (-not (Test-Path $cfgPath)) {
        $results[$cfgName] = @{ Ok = $false; Detail = "missing config" }
        $allOk = $false
        Write-Host "[$cfgName] FAIL - missing $cfgPath" -ForegroundColor Red
        continue
    }

    $sweepId = [System.IO.Path]::GetFileNameWithoutExtension($cfgName)
    $outPath = Join-Path $outDir ($sweepId + "_smoke_results.json")
    $tuneArgs = @("--config", $cfgPath, "--max-cells", $MaxCells)
    if ($DryRun) {
        $tuneArgs += "--dry-run"
    } elseif ($Write) {
        $tuneArgs += @("--write", "--out", $outPath)
    }

    Write-Host "== $cfgName ==" -ForegroundColor Yellow
    Push-Location $root
    try {
        & $tuneExe @tuneArgs
        $code = $LASTEXITCODE
    } finally {
        Pop-Location
    }

    if ($code -eq 0) {
        $detail = if ($DryRun) { "dry-run ok" } elseif ($Write) { "wrote $outPath" } else { "ok" }
        $results[$cfgName] = @{ Ok = $true; Detail = $detail }
        Write-Host "[$cfgName] PASS - $detail" -ForegroundColor Green
    } else {
        $results[$cfgName] = @{ Ok = $false; Detail = "exit $code" }
        $allOk = $false
        Write-Host "[$cfgName] FAIL - exit $code" -ForegroundColor Red
    }
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "TUNE SMOKE SUMMARY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
foreach ($entry in $results.GetEnumerator()) {
    $icon = if ($entry.Value.Ok) { "PASS" } else { "FAIL" }
    $color = if ($entry.Value.Ok) { "Green" } else { "Red" }
    $detail = if ($entry.Value.Detail) { " - $($entry.Value.Detail)" } else { "" }
    Write-Host ("  [{0}] {1}{2}" -f $icon, $entry.Key, $detail) -ForegroundColor $color
}
Write-Host "========================================" -ForegroundColor Cyan

if ($allOk) {
    Write-Host "OK cypha_tune_smoke" -ForegroundColor Green
    exit 0
}
Write-Host "FAILED cypha_tune_smoke" -ForegroundColor Red
exit 1
