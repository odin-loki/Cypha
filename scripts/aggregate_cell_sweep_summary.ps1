# Aggregate cell-sweep variant JSON artifacts into summary.csv vs locked B0/B1/B2 baselines.
# Read-only on bench/results/cell_sweep (and optional repo-root results/ spill).
# Never touches bench/BASELINE_LOCK.json or overnight processes.
#
# Usage:
#   pwsh -File scripts/aggregate_cell_sweep_summary.ps1
#   pwsh -File scripts/aggregate_cell_sweep_summary.ps1 -DryRun
#   pwsh -File scripts/aggregate_cell_sweep_summary.ps1 -OutputDir bench/results/cell_sweep
param(
    [switch]$DryRun,
    [string]$OutputDir = "bench/results/cell_sweep",
    [string]$SpillDir = "results"
)

$ErrorActionPreference = "Stop"

$BaselineB0 = 3.478
$BaselineB1 = 2.979
$BaselineB2 = 2.873

$root = Split-Path $PSScriptRoot -Parent
$primaryDir = Join-Path $root ($OutputDir -replace '/', '\')
$spillPath = Join-Path $root ($SpillDir -replace '/', '\')
$lockPath = Join-Path $root "bench\BASELINE_LOCK.json"

function Get-ScanDirs {
    $dirs = [System.Collections.Generic.List[string]]::new()

    if (Test-Path $primaryDir) {
        $dirs.Add($primaryDir)
    }

    if (Test-Path $spillPath) {
        $spillVariants = @(Get-ChildItem -Path $spillPath -Filter "variant_*.json" -File -ErrorAction SilentlyContinue)
        if ($spillVariants.Count -gt 0) {
            $dirs.Add($spillPath)
        }
    }

    return $dirs.ToArray()
}

function Get-VariantArtifacts {
    param([string[]]$Dirs)

    $byId = @{}

    foreach ($dir in $Dirs) {
        foreach ($file in (Get-ChildItem -Path $dir -Filter "variant_*.json" -File -ErrorAction SilentlyContinue)) {
            if ($file.FullName -eq $lockPath) { continue }

            $variantId = $file.BaseName -replace '^variant_', ''
            if ($variantId -eq '') { continue }

            $existing = $byId[$variantId]
            if ($null -ne $existing -and $existing.LastWriteTimeUtc -ge $file.LastWriteTimeUtc) {
                continue
            }

            $byId[$variantId] = $file
        }
    }

    return $byId
}

function Read-VariantRow {
    param(
        [string]$VariantId,
        [System.IO.FileInfo]$File
    )

    try {
        $json = Get-Content -Path $File.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        Write-Host "  [warn] $($File.Name): could not parse JSON ($($_.Exception.Message))" -ForegroundColor Yellow
        return $null
    }

    $id = if ($json.id) { [string]$json.id } else { $VariantId }
    if (-not $json.PSObject.Properties.Name -contains 'bpc' -or $null -eq $json.bpc) {
        Write-Host "  [warn] $($File.Name): missing bpc" -ForegroundColor Yellow
        return $null
    }

    $bpc = [double]$json.bpc
    return [pscustomobject]@{
        variant = $id
        bpc     = $bpc
        vs_B0   = $bpc - $BaselineB0
        vs_B1   = $bpc - $BaselineB1
        vs_B2   = $bpc - $BaselineB2
        source  = $File.FullName
    }
}

function Format-CsvNumber {
    param([double]$Value)
    return ('{0:F6}' -f $Value)
}

$scanDirs = Get-ScanDirs
$artifactMap = Get-VariantArtifacts -Dirs $scanDirs

Write-Host "== aggregate cell-sweep summary ==" -ForegroundColor Cyan
Write-Host "  baselines: B0=$BaselineB0 B1=$BaselineB1 B2=$BaselineB2"
Write-Host "  scan dirs: $(if ($scanDirs.Count -gt 0) { ($scanDirs -join '; ') } else { '(none)' })"
if ($DryRun) {
    Write-Host "  mode:      DryRun (writes summary.csv, no other side effects)" -ForegroundColor Yellow
}

$rows = [System.Collections.Generic.List[object]]::new()
foreach ($variantId in ($artifactMap.Keys | Sort-Object)) {
    $file = $artifactMap[$variantId]
    $row = Read-VariantRow -VariantId $variantId -File $file
    if ($null -ne $row) {
        $rows.Add($row)
    }
}

$outDir = $primaryDir
if (-not (Test-Path $outDir)) {
    if ($DryRun) {
        Write-Host "  [info] output dir missing; would create $outDir" -ForegroundColor Yellow
    } else {
        New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    }
}

$summaryPath = Join-Path $outDir "summary.csv"
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add('variant,bpc,vs_B0,vs_B1,vs_B2') | Out-Null

foreach ($row in ($rows | Sort-Object variant)) {
    $lines.Add(
        ('{0},{1},{2},{3},{4}' -f $row.variant,
            (Format-CsvNumber $row.bpc),
            (Format-CsvNumber $row.vs_B0),
            (Format-CsvNumber $row.vs_B1),
            (Format-CsvNumber $row.vs_B2))
    ) | Out-Null
}

$csvText = ($lines -join "`n") + "`n"
$utf8NoBom = New-Object System.Text.UTF8Encoding $false

if ($DryRun -and -not (Test-Path $outDir)) {
    Write-Host "  [dry-run] $($rows.Count) row(s); CSV preview:" -ForegroundColor Yellow
    Write-Host $csvText
} else {
    [System.IO.File]::WriteAllText($summaryPath, $csvText, $utf8NoBom)
    Write-Host "  wrote:     $summaryPath ($($rows.Count) row(s))" -ForegroundColor Green
}

$hash = $null
if (Test-Path $summaryPath) {
    $hash = (Get-FileHash -Path $summaryPath -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Host "  sha256:    $hash"
} else {
    $hashBytes = [System.Text.Encoding]::UTF8.GetBytes($csvText)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $hash = ([BitConverter]::ToString($sha.ComputeHash($hashBytes)) -replace '-', '').ToLowerInvariant()
    } finally {
        $sha.Dispose()
    }
    Write-Host "  sha256:    $hash (preview; file not written)"
}

if ($rows.Count -eq 0) {
    Write-Host "  note:      no variant JSON found; empty summary.csv header only" -ForegroundColor DarkGray
} elseif ($rows.Count -lt 25) {
    Write-Host "  note:      partial sweep ($($rows.Count)/25+ variants); missing IDs are omitted" -ForegroundColor DarkYellow
}

Write-Host ""
Write-Host "rows=$($rows.Count) hash=$hash"
