# Phase 20: remove repo-root d##_smoke.json / d##_*_smoke.json CTest spill files.
# Does not touch native/build* smoke JSON or bench/BASELINE_LOCK.json.
#
# Usage:
#   pwsh -File scripts/cleanup_repo_smoke_artifacts.ps1 -DryRun
#   pwsh -File scripts/cleanup_repo_smoke_artifacts.ps1 -Force
param(
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$lockRel = "bench\BASELINE_LOCK.json"
$lockPath = Join-Path $root $lockRel
$smokePattern = '^d\d{2}(_.*)?_smoke\.json$'

if ($Force -and $DryRun) {
    throw "use -DryRun or -Force, not both"
}

if (-not $Force -and -not $DryRun) {
    $DryRun = $true
}

$candidates = Get-ChildItem -Path $root -File -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -match $smokePattern
    }

$toRemove = @()
foreach ($file in $candidates) {
    if ($file.FullName -eq $lockPath) {
        Write-Host "skip (protected lock): $($file.Name)" -ForegroundColor Yellow
        continue
    }
    $toRemove += $file
}

Write-Host "== cleanup repo-root smoke artifacts ==" -ForegroundColor Cyan
if ($DryRun) {
    Write-Host "  mode: DryRun (list only; pass -Force to remove)" -ForegroundColor Yellow
} else {
    Write-Host "  mode: Force (remove listed files)" -ForegroundColor Yellow
}
Write-Host "  root: $root" -ForegroundColor DarkGray
Write-Host ""

if (-not $toRemove -or $toRemove.Count -eq 0) {
    Write-Host "no repo-root smoke artifacts matched $smokePattern" -ForegroundColor Green
    exit 0
}

foreach ($file in $toRemove) {
    if ($DryRun) {
        Write-Host "  would remove: $($file.Name)" -ForegroundColor DarkGray
    } else {
        Remove-Item -LiteralPath $file.FullName -Force
        Write-Host "  removed: $($file.Name)" -ForegroundColor Green
    }
}

Write-Host ""
if ($DryRun) {
    Write-Host "DryRun complete ($($toRemove.Count) file(s)). Re-run with -Force to remove." -ForegroundColor Yellow
} else {
    Write-Host "cleanup_repo_smoke_artifacts: removed $($toRemove.Count) file(s)." -ForegroundColor Green
}

exit 0
