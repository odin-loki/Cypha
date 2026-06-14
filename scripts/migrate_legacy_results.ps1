# Migrate legacy repo-root results/ cell-sweep artifacts to bench/results/cell_sweep/.
#
# Copies summary.csv, manifest.json, and variant_*.json when the legacy path exists.
# Merge policy: never overwrite a destination file that is newer than the source.
#
# Usage:
#   pwsh -File scripts/migrate_legacy_results.ps1
#   pwsh -File scripts/migrate_legacy_results.ps1 -DryRun
#   pwsh -File scripts/migrate_legacy_results.ps1 -RemoveLegacy
param(
    [switch]$DryRun,
    [switch]$RemoveLegacy
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$legacyDir = Join-Path $root "results"
$targetDir = Join-Path $root "bench\results\cell_sweep"

function Get-LegacyArtifacts {
    param([string]$Dir)
    if (-not (Test-Path $Dir)) { return @() }
    $files = [System.Collections.Generic.List[System.IO.FileInfo]]::new()
    $summary = Join-Path $Dir "summary.csv"
    if (Test-Path $summary) { $files.Add((Get-Item $summary)) }
    $manifest = Join-Path $Dir "manifest.json"
    if (Test-Path $manifest) { $files.Add((Get-Item $manifest)) }
    foreach ($variant in (Get-ChildItem -Path $Dir -Filter "variant_*.json" -File -ErrorAction SilentlyContinue)) {
        $files.Add($variant)
    }
    return $files.ToArray()
}

function Test-ShouldCopy {
    param([System.IO.FileInfo]$Source, [string]$DestPath)
    if (-not (Test-Path $DestPath)) { return $true }
    $dest = Get-Item $DestPath
    return $Source.LastWriteTimeUtc -gt $dest.LastWriteTimeUtc
}

$artifacts = Get-LegacyArtifacts -Dir $legacyDir
if ($artifacts.Count -eq 0) {
    if (Test-Path $legacyDir) {
        Write-Host "migrate_legacy_results: $legacyDir exists but no summary.csv or variant_*.json found; nothing to do." -ForegroundColor DarkGray
    } else {
        Write-Host "migrate_legacy_results: no legacy results/ at repo root; nothing to do." -ForegroundColor DarkGray
    }
    exit 0
}

Write-Host "== migrate legacy cell-sweep results ==" -ForegroundColor Cyan
Write-Host "  source: $legacyDir"
Write-Host "  target: $targetDir"
if ($DryRun) { Write-Host "  mode:   DryRun (plan only)" -ForegroundColor Yellow }
Write-Host ""

if (-not $DryRun) {
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
}

$migrated = @()
$skipped = @()
$planned = @()

foreach ($src in $artifacts) {
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

if ($RemoveLegacy) {
    $missing = @()
    foreach ($src in $artifacts) {
        $destPath = Join-Path $targetDir $src.Name
        if (-not (Test-Path $destPath)) {
            $missing += $src.Name
        }
    }
    if ($missing.Count -gt 0) {
        Write-Host ""
        Write-Host "-RemoveLegacy: bench/results/cell_sweep missing $($missing -join ', '); legacy results/ kept." -ForegroundColor Yellow
        exit 1
    }
    Remove-Item -Path $legacyDir -Recurse -Force
    Write-Host ""
    Write-Host "Removed legacy $legacyDir" -ForegroundColor Green
}

exit 0
