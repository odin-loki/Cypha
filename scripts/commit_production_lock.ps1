# Phase 15: commit production BASELINE_LOCK.json after overnight validation.
# Runs finalize_production_overnight.ps1 first; commits only when validation passes and
# overnight_results.n_train >= 300000. Never pushes automatically.
#
# Usage:
#   pwsh -File scripts/commit_production_lock.ps1
#   pwsh -File scripts/commit_production_lock.ps1 -DryRun
#   pwsh -File scripts/commit_production_lock.ps1 -BuildDir C:\Temp\cypha_full_cpp_build -Force
param(
    [string]$BuildDir = "native/build",
    [string]$LockFile = "",
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent

if (-not $LockFile) {
    $LockFile = Join-Path $root "bench\BASELINE_LOCK.json"
} elseif (-not [System.IO.Path]::IsPathRooted($LockFile)) {
    $LockFile = Join-Path $root $LockFile
}

$lockRel = $LockFile
if ($lockRel.StartsWith($root)) {
    $lockRel = $lockRel.Substring($root.Length).TrimStart('\', '/')
}

$finalizeScript = Join-Path $PSScriptRoot "finalize_production_overnight.ps1"
if (-not (Test-Path $finalizeScript)) {
    throw "missing $finalizeScript"
}

Write-Host "== commit production lock ==" -ForegroundColor Cyan
Write-Host "  lock:  $LockFile"
Write-Host "  build: $BuildDir"
Write-Host ""

Write-Host "== finalize_production_overnight.ps1 ==" -ForegroundColor Cyan
& $finalizeScript -BuildDir $BuildDir -LockFile $LockFile
if ($LASTEXITCODE -ne 0) {
    Write-Host "commit_production_lock: validation failed (exit $LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
}

if (-not (Test-Path $LockFile)) {
    Write-Host "commit_production_lock: lock file not found: $LockFile" -ForegroundColor Red
    exit 1
}

try {
    $lock = Get-Content $LockFile -Raw | ConvertFrom-Json
} catch {
    Write-Host "commit_production_lock: invalid lock JSON: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

$nTrain = $null
if ($lock.PSObject.Properties.Name -contains "overnight_results" -and $null -ne $lock.overnight_results) {
    $section = $lock.overnight_results
    if ($section.PSObject.Properties.Name -contains "n_train") {
        $nTrain = [int]$section.n_train
    }
}

if ($null -eq $nTrain -or $nTrain -lt 300000) {
    $shown = if ($null -eq $nTrain) { "?" } else { $nTrain }
    Write-Host "commit_production_lock: overnight_results.n_train=$shown < 300000 - not ready to commit" -ForegroundColor Yellow
    exit 1
}

$status = if ($section.PSObject.Properties.Name -contains "status") { $section.status } else { "?" }
$bpc = if ($section.PSObject.Properties.Name -contains "bpc") { $section.bpc } else { "?" }

Write-Host ""
Write-Host "== git diff $lockRel ==" -ForegroundColor Cyan
Push-Location $root
try {
    git diff -- $lockRel
} finally {
    Pop-Location
}

$commitMessage = "bench: lock production overnight results (n_train=$nTrain)`n`nValidated via finalize_production_overnight.ps1 (status=$status, bpc=$bpc)."

Write-Host ""
Write-Host "== suggested commit message ==" -ForegroundColor Cyan
Write-Host $commitMessage
Write-Host ""

if ($DryRun) {
    Write-Host "Dry run - skipped git add/commit." -ForegroundColor Yellow
    exit 0
}

if (-not $Force) {
    Write-Host "Preview only - pass -Force to git add and commit (never pushes)." -ForegroundColor Yellow
    exit 0
}

Push-Location $root
try {
    git add -- $lockRel
    if ($LASTEXITCODE -ne 0) {
        throw "git add failed exit=$LASTEXITCODE"
    }
    git commit -m $commitMessage
    if ($LASTEXITCODE -ne 0) {
        throw "git commit failed exit=$LASTEXITCODE"
    }
} finally {
    Pop-Location
}

Write-Host "commit_production_lock: committed $lockRel (not pushed)" -ForegroundColor Green
exit 0
