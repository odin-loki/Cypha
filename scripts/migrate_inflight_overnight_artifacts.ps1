# Phase 23: merge in-flight build_p13 overnight cell-sweep spill from repo-root results/
# into bench/results/cell_sweep/ (focused variant of migrate_legacy_results.ps1).
#
# Copies summary.csv, manifest.json, and variant_*.json when the spill path exists.
# Merge policy: never overwrite a destination file that is newer than the source.
# Never deletes or modifies bench/BASELINE_LOCK.json.
#
# Usage:
#   pwsh -File scripts/migrate_inflight_overnight_artifacts.ps1
#   pwsh -File scripts/migrate_inflight_overnight_artifacts.ps1 -DryRun
#   pwsh -File scripts/migrate_inflight_overnight_artifacts.ps1 -Force
param(
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$spillDir = Join-Path $root "results"
$targetDir = Join-Path $root "bench\results\cell_sweep"
$lockRel = "bench\BASELINE_LOCK.json"
$lockPath = Join-Path $root $lockRel

if ($Force -and $DryRun) {
    Write-Host "migrate_inflight_overnight_artifacts: use -DryRun or -Force, not both." -ForegroundColor Red
    exit 1
}

if (-not $Force -and -not $DryRun) {
    $DryRun = $true
}

function Get-SpillArtifacts {
    param([string]$Dir)

    if (-not (Test-Path $Dir)) { return @() }

    $files = [System.Collections.Generic.List[System.IO.FileInfo]]::new()
    $summary = Join-Path $Dir "summary.csv"
    if (Test-Path $summary) { $files.Add((Get-Item $summary)) }
    $manifest = Join-Path $Dir "manifest.json"
    if (Test-Path $manifest) { $files.Add((Get-Item $manifest)) }
    foreach ($variant in (Get-ChildItem -Path $Dir -Filter "variant_*.json" -File -ErrorAction SilentlyContinue)) {
        if ($variant.FullName -eq $lockPath) {
            Write-Host "  [skip] $($variant.Name) (protected lock)" -ForegroundColor Yellow
            continue
        }
        $files.Add($variant)
    }
    return $files.ToArray()
}

function Test-ShouldCopy {
    param([System.IO.FileInfo]$Source, [string]$DestPath)

    if ($Source.FullName -eq $lockPath) { return $false }
    if (-not (Test-Path $DestPath)) { return $true }
    $dest = Get-Item $DestPath
    return $Source.LastWriteTimeUtc -gt $dest.LastWriteTimeUtc
}

$artifacts = Get-SpillArtifacts -Dir $spillDir

if ($artifacts.Count -eq 0) {
    if (Test-Path $spillDir) {
        Write-Host "migrate_inflight_overnight_artifacts: $spillDir exists but no cell-sweep artifacts found; nothing to do." -ForegroundColor DarkGray
    } else {
        Write-Host "migrate_inflight_overnight_artifacts: no repo-root results/ spill; nothing to do." -ForegroundColor DarkGray
    }
    exit 0
}

Write-Host "== migrate in-flight overnight cell-sweep spill ==" -ForegroundColor Cyan
Write-Host "  source: $spillDir"
Write-Host "  target: $targetDir"
if ($DryRun) {
    Write-Host "  mode:   DryRun (plan only)" -ForegroundColor Yellow
} else {
    Write-Host "  mode:   Force (copy newer/missing files)" -ForegroundColor Yellow
}
Write-Host ""

if (-not $DryRun) {
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
}

$migrated = @()
$skipped = @()
$planned = @()

foreach ($src in $artifacts) {
    if ($src.FullName -eq $lockPath) {
        Write-Host "  [skip] $($src.Name) (protected lock)" -ForegroundColor Yellow
        continue
    }

    $destPath = Join-Path $targetDir $src.Name
    if (Test-ShouldCopy -Source $src -DestPath $destPath) {
        if ($DryRun) {
            $planned += $src.Name
            Write-Host "  [copy] $($src.Name)" -ForegroundColor Yellow
        } else {
            Copy-Item -Path $src.FullName -Destination $destPath -Force
            $migrated += $src.Name
            Write-Host "  [copied] $($src.Name)" -ForegroundColor Green
        }
    } else {
        $skipped += $src.Name
        Write-Host "  [skip] $($src.Name) (destination newer)" -ForegroundColor DarkGray
    }
}

Write-Host ""
if ($DryRun) {
    Write-Host "DryRun plan: $($planned.Count) would copy, $($skipped.Count) would skip." -ForegroundColor Yellow
    if ($planned.Count -gt 0) {
        Write-Host "Apply: pwsh -File scripts/migrate_inflight_overnight_artifacts.ps1 -Force" -ForegroundColor Cyan
    }
    exit 0
}

if ($migrated.Count -gt 0) {
    Write-Host "Migrated $($migrated.Count) file(s): $($migrated -join ', ')" -ForegroundColor Green
} else {
    Write-Host "No files copied (all destinations newer or up to date)." -ForegroundColor DarkGray
}
if ($skipped.Count -gt 0) {
    Write-Host "Skipped $($skipped.Count) file(s): $($skipped -join ', ')" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "Repo-root results/ spill kept. For full legacy cleanup see migrate_legacy_results.ps1." -ForegroundColor DarkGray
exit 0
