# Real-data profiling pass — D03 tabular bench + tune smoke over bench/data/iris.csv.
# Writes docs/reports/REAL_DATA_PROFILE_2026-07-17.md (or -ReportPath override).
# Uses scratch CYPHA_REPO_ROOT so BASELINE_* / bench/report in the real tree stay untouched.
param(
    [string]$BuildDir = "",
    [string]$ReportPath = "",
    [switch]$SkipBuild,
    [switch]$SkipTune
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "lib\NativeBenchCommon.ps1")

$root = Get-CyphaRepoRoot -ScriptRoot $PSScriptRoot
$BuildDir = if ($BuildDir) { $BuildDir } else { Join-Path $root "native\build_realprof" }
$ReportPath = if ($ReportPath) { $ReportPath } else {
    Join-Path $root "docs\reports\REAL_DATA_PROFILE_2026-07-17.md"
}

$benchExe = Join-Path $BuildDir "cypha_bench_run.exe"
$tuneExe = Join-Path $BuildDir "cypha_tune_run.exe"
$irisCsv = Join-Path $root "bench\data\iris.csv"
$tuneCfg = Join-Path $root "bench\config\real_data_profile_tune_smoke.json"

if (-not (Test-Path $irisCsv)) {
    throw "missing sample CSV: $irisCsv"
}

if (-not $SkipBuild) {
    Write-Host "Configuring native/build_realprof..." -ForegroundColor Cyan
    cmake -S (Join-Path $root "native") -B $BuildDir `
        -DCMAKE_BUILD_TYPE=Release `
        -DCYPHA_BUILD_EXPERIMENT_DB=OFF `
        -G Ninja
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

    Write-Host "Building cypha_bench_run + cypha_tune_run..." -ForegroundColor Cyan
    cmake --build $BuildDir --target cypha_bench_run cypha_tune_run
    if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }
}

foreach ($exe in @($benchExe, $tuneExe)) {
    if (-not (Test-Path $exe)) {
        throw "missing $exe (run without -SkipBuild or set -BuildDir)"
    }
}

$scratchRoot = Join-Path $env:TEMP "cypha_real_data_profile_$([Guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Force -Path (Join-Path $scratchRoot "bench\data") | Out-Null
Copy-Item $irisCsv (Join-Path $scratchRoot "bench\data\iris.csv") -Force
New-Item -ItemType Directory -Force -Path (Join-Path $scratchRoot "bench\config") | Out-Null
Copy-Item $tuneCfg (Join-Path $scratchRoot "bench\config\real_data_profile_tune_smoke.json") -Force
$everydayProfile = Join-Path $root "bench\config\everyday_profile.json"
if (-not (Test-Path $everydayProfile)) { throw "missing $everydayProfile" }
Copy-Item $everydayProfile (Join-Path $scratchRoot "bench\config\everyday_profile.json") -Force

$prevRepoRoot = $env:CYPHA_REPO_ROOT
$prevFast = $env:CYPHA_BENCH_FAST
$env:CYPHA_REPO_ROOT = $scratchRoot
$env:CYPHA_BENCH_FAST = "1"
$env:CYPHA_BENCH_RUN_BIN = $benchExe
$env:CYPHA_TUNE_RUN_BIN = $tuneExe

function Get-D03Metrics {
    param([string]$TablePath)
    if (-not (Test-Path $TablePath)) { return $null }
    $j = Get-Content $TablePath -Raw | ConvertFrom-Json
    $datasets = $null
    if ($j.experiments -and $j.experiments.datasets) {
        $datasets = $j.experiments.datasets
    } elseif ($j.datasets) {
        $datasets = $j.datasets
    }
    $iris = $null
    $wine = $null
    if ($datasets) {
        foreach ($ds in $datasets) {
            if ($ds.dataset -eq "iris") { $iris = $ds }
            if ($ds.dataset -eq "wine") { $wine = $ds }
        }
    }
    return [ordered]@{
        IrisAccuracy = if ($iris -and $iris.cypha_scores) { $iris.cypha_scores.accuracy } else { $null }
        WineAccuracy = if ($wine -and $wine.cypha_scores) { $wine.cypha_scores.accuracy } else { $null }
        IrisSource   = if ($iris) { $iris.data_source } else { $null }
        WineSource   = if ($wine) { $wine.data_source } else { $null }
    }
}

try {
    Write-Host "Running cypha_bench_run --domain 3 (CYPHA_BENCH_FAST=1)..." -ForegroundColor Yellow
    $benchSw = [System.Diagnostics.Stopwatch]::StartNew()
    Push-Location $scratchRoot
    try {
        Invoke-NativeWithProgressLog -Exe $benchExe -NativeArgs @("--domain", "3")
        if ($LASTEXITCODE -ne 0) { throw "cypha_bench_run exit $LASTEXITCODE" }
    } finally {
        Pop-Location
    }
    $benchSw.Stop()
    $benchSec = [math]::Round($benchSw.Elapsed.TotalSeconds, 3)

    $d03Table = Join-Path $scratchRoot "bench\report\tables\d03.json"
    $benchMetrics = Get-D03Metrics -TablePath $d03Table

    $tuneSec = $null
    $tuneCells = $null
    if (-not $SkipTune) {
        $tuneOut = Join-Path $scratchRoot "bench\artifacts\tuning\real_data_profile_tune_smoke_results.json"
        New-Item -ItemType Directory -Force -Path (Split-Path $tuneOut -Parent) | Out-Null
        Write-Host "Running cypha_tune_run (2-cell D03 smoke)..." -ForegroundColor Yellow
        $tuneSw = [System.Diagnostics.Stopwatch]::StartNew()
        Push-Location $scratchRoot
        try {
            Invoke-NativeWithProgressLog -Exe $tuneExe -NativeArgs @(
                "--config", $tuneCfg, "--max-cells", "2", "--write", "--out", $tuneOut
            )
            if ($LASTEXITCODE -ne 0) { throw "cypha_tune_run exit $LASTEXITCODE" }
        } finally {
            Pop-Location
        }
        $tuneSw.Stop()
        $tuneSec = [math]::Round($tuneSw.Elapsed.TotalSeconds, 3)
        if (Test-Path $tuneOut) {
            $tuneJson = Get-Content $tuneOut -Raw | ConvertFrom-Json
            $tuneCells = @($tuneJson.runs | ForEach-Object { $_.cell_id })
        }
    }

    $irisRows = (Import-Csv $irisCsv).Count
    $generated = (Get-Date -Format "yyyy-MM-dd HH:mm:ss K")

    $md = @"
# Real-data profiling pass - 2026-07-17

**Scope:** Bill of Work section 7 housekeeping - log bench/tune timing + metrics on a real CSV under ``bench/data/``. Did not touch ``build_math``, ``build_deff``, ``BASELINE_*``, or overnight.

## Sample data

| File | Rows | Loader |
|------|------|--------|
| ``bench/data/iris.csv`` | $irisRows | ``load_tabular_dataset("iris")`` -> ``data_source=csv`` when present |

Wine in D03 still uses synthetic fallback (no ``wine.csv`` in tree); iris confirms the CSV ingest path on real UCI measurements.

## Build

``````powershell
cmake -S native -B native/build_realprof -DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_EXPERIMENT_DB=OFF -G Ninja
cmake --build native/build_realprof --target cypha_bench_run cypha_tune_run
``````

## Bench run (`cypha_bench_run --domain 3`)

| Setting | Value |
|---------|-------|
| ``CYPHA_BENCH_FAST`` | ``1`` |
| ``CYPHA_REPO_ROOT`` | scratch temp tree (no writes to repo ``bench/BASELINE_*``) |
| Wall time | **${benchSec}s** |

### D03 metrics (from ``bench/report/tables/d03.json``)

| Dataset | Source | Accuracy |
|---------|--------|----------|
| iris | $($benchMetrics.IrisSource) | $($benchMetrics.IrisAccuracy) |
| wine | $($benchMetrics.WineSource) | $($benchMetrics.WineAccuracy) |

## Tune smoke (`cypha_tune_run`)

| Setting | Value |
|---------|-------|
| Config | ``bench/config/real_data_profile_tune_smoke.json`` |
| Cells | $(if ($tuneCells) { ($tuneCells -join ", ") } else { "skipped" }) |
| Wall time | $(if ($null -ne $tuneSec) { "**${tuneSec}s**" } else { "skipped (-SkipTune)" }) |

## Reproduce

``````powershell
scripts/run_real_data_profile.ps1
# or: scripts/run_real_data_profile.ps1 -SkipBuild -BuildDir native/build_realprof
``````

Generated: $generated
"@

    New-Item -ItemType Directory -Force -Path (Split-Path $ReportPath -Parent) | Out-Null
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($ReportPath, $md, $utf8NoBom)
    Write-Host "Wrote $ReportPath" -ForegroundColor Green
}
finally {
    if ($null -ne $prevRepoRoot) { $env:CYPHA_REPO_ROOT = $prevRepoRoot } else { Remove-Item Env:CYPHA_REPO_ROOT -ErrorAction SilentlyContinue }
    if ($null -ne $prevFast) { $env:CYPHA_BENCH_FAST = $prevFast } else { Remove-Item Env:CYPHA_BENCH_FAST -ErrorAction SilentlyContinue }
    Remove-Item Env:CYPHA_BENCH_RUN_BIN -ErrorAction SilentlyContinue
    Remove-Item Env:CYPHA_TUNE_RUN_BIN -ErrorAction SilentlyContinue
    if (Test-Path $scratchRoot) {
        Remove-Item -Recurse -Force $scratchRoot -ErrorAction SilentlyContinue
    }
}
