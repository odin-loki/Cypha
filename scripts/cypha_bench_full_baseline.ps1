# Full native bench baseline lock (v2.2): Release build, all domains without FAST, CyphaLM BPC sweep.
param(
    [string]$BuildDir = "C:\Temp\cypha_full_cpp_build",
    [switch]$SkipBuild,
    [switch]$Medium,
    [switch]$SkipLmBench
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$stamp = Get-Date -Format "yyyyMMdd"

$isMedium = $Medium -or ($env:CYPHA_BENCH_MEDIUM -eq "1")
$lmNTrain = if ($isMedium) { 50000 } else { 300000 }
$lmNEval = 2000
$lmProfiles = if ($isMedium) { @("d04", "d17") } else { @("d17") }

function BinPath {
    param([string]$Stem)
    Join-Path $BuildDir "$Stem.exe"
}

function Get-JsonFile {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return $null }
    Get-Content $Path -Raw | ConvertFrom-Json
}

function Extract-BaselineMetrics {
    param(
        [string]$TablesDir,
        [hashtable]$LmResults = @{}
    )
    $m = [ordered]@{
        generated_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd HH:mm:ss UTC")
        mode          = if ($isMedium) { "medium" } else { "full" }
        lm_n_train    = $lmNTrain
        lm_n_eval     = $lmNEval
    }

    $d01 = Get-JsonFile (Join-Path $TablesDir "d01.json")
    if ($d01 -and $d01.experiments.tasks) {
        foreach ($task in $d01.experiments.tasks) {
            $key = "d01.$($task.dataset).accuracy"
            $m[$key] = [double]$task.cypha_scores.accuracy
        }
    }

    $d02 = Get-JsonFile (Join-Path $TablesDir "d02.json")
    if ($d02 -and $d02.experiments.cypha_scores) {
        $m["d02.rmse"] = [double]$d02.experiments.cypha_scores.rmse
        $m["d02.r2"] = [double]$d02.experiments.cypha_scores.r2
    }

    $d03 = Get-JsonFile (Join-Path $TablesDir "d03.json")
    if ($d03 -and $d03.experiments.datasets) {
        foreach ($ds in $d03.experiments.datasets) {
            $key = "d03.$($ds.dataset).accuracy"
            $m[$key] = [double]$ds.cypha_scores.accuracy
        }
    }

    foreach ($dom in @("d04", "d05", "d17")) {
        $dj = Get-JsonFile (Join-Path $TablesDir "$dom.json")
        if ($dj -and $null -ne $dj.experiments.bpc) {
            $m["$dom.bpc"] = [double]$dj.experiments.bpc
        } elseif ($dj -and $dj.experiments.cypha_scores -and $null -ne $dj.experiments.cypha_scores.rmse) {
            $m["$dom.rmse"] = [double]$dj.experiments.cypha_scores.rmse
        }
    }

    $d08 = Get-JsonFile (Join-Path $TablesDir "d08.json")
    if ($d08 -and $d08.experiments.experiments) {
        foreach ($exp in $d08.experiments.experiments) {
            if ($exp.encoding) {
                $m["d08.$($exp.encoding).accuracy"] = [double]$exp.cypha_scores.accuracy
            }
        }
    }

    foreach ($entry in $LmResults.GetEnumerator()) {
        $m[$entry.Key] = [double]$entry.Value
    }

    return $m
}

function Compare-BaselineMetrics {
    param(
        [hashtable]$Current,
        [hashtable]$Previous
    )
    $skip = @("generated_utc", "mode", "lm_n_train", "lm_n_eval")
    $rows = @()
    foreach ($key in $Current.Keys) {
        if ($skip -contains $key) { continue }
        $cur = $Current[$key]
        if ($null -eq $cur) { continue }
        $row = [ordered]@{
            metric = $key
            current = $cur
            previous = $null
            delta = $null
            pct_change = $null
        }
        if ($Previous -and $Previous.Contains($key) -and $null -ne $Previous[$key]) {
            $prev = [double]$Previous[$key]
            $row.previous = $prev
            $row.delta = [math]::Round($cur - $prev, 6)
            if ([math]::Abs($prev) -gt 1e-12) {
                $row.pct_change = [math]::Round(100.0 * ($cur - $prev) / $prev, 3)
            }
        }
        $rows += [pscustomobject]$row
    }
    return $rows
}

Write-Host "Cypha full native bench baseline (v2.2)" -ForegroundColor Cyan
Write-Host "  repo:      $root"
Write-Host "  build:     $BuildDir"
Write-Host "  mode:      $(if ($isMedium) { 'MEDIUM (50k LM)' } else { 'FULL (300k d17 LM)' })"
Write-Host "  CYPHA_BENCH_FAST: (unset - full domain scale)"
Write-Host ""

Remove-Item env:CYPHA_BENCH_FAST -ErrorAction SilentlyContinue
if ($isMedium) {
    $env:CYPHA_BENCH_MEDIUM = "1"
} else {
    Remove-Item env:CYPHA_BENCH_MEDIUM -ErrorAction SilentlyContinue
}
$env:CYPHA_REPO_ROOT = $root

# --- Build ---
if (-not $SkipBuild) {
    Write-Host "== CMake configure (Release) ==" -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
    cmake -S (Join-Path $root "native") -B $BuildDir `
        -DCMAKE_BUILD_TYPE=Release `
        -DCYPHA_BUILD_QT=OFF `
        -DCYPHA_BUILD_EXPERIMENT_DB=ON `
        -G Ninja
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

    Write-Host "== CMake build (bench targets) ==" -ForegroundColor Yellow
    cmake --build $BuildDir --target cypha_bench_run cyphalm_bench_native --parallel
    if ($LASTEXITCODE -ne 0) { throw "cmake --build failed" }
} elseif (-not (Test-Path $BuildDir)) {
    throw "BuildDir missing: $BuildDir"
}

$benchExe = BinPath "cypha_bench_run"
$lmExe = BinPath "cyphalm_bench_native"
if (-not (Test-Path $benchExe)) { throw "missing $benchExe" }
if (-not $SkipLmBench -and -not (Test-Path $lmExe)) { throw "missing $lmExe" }

# --- Full domain bench (no FAST) ---
Write-Host ""
Write-Host "== cypha_bench_run --from-domain 1 (no CYPHA_BENCH_FAST) ==" -ForegroundColor Yellow
Push-Location $root
try {
    & $benchExe --from-domain 1
    if ($LASTEXITCODE -ne 0) { throw "cypha_bench_run --from-domain 1 failed (exit $LASTEXITCODE)" }

    Write-Host ""
    Write-Host "== cypha_bench_run --report-only ==" -ForegroundColor Yellow
    & $benchExe --report-only
    if ($LASTEXITCODE -ne 0) { throw "cypha_bench_run --report-only failed (exit $LASTEXITCODE)" }
} finally {
    Pop-Location
}

# --- CyphaLM native BPC bench ---
$lmResults = @{}
if (-not $SkipLmBench) {
    foreach ($profile in $lmProfiles) {
        Write-Host ""
        Write-Host "== cyphalm_bench_native hybrid $profile n=$lmNTrain ==" -ForegroundColor Yellow
        $lmLog = Join-Path $env:TEMP "cyphalm_bench_${profile}_${lmNTrain}.log"
        Push-Location $root
        try {
            & $lmExe --mode hybrid --profile $profile --n-train $lmNTrain --n-eval $lmNEval --threads 1 `
                *> $lmLog
            if ($LASTEXITCODE -ne 0) {
                Get-Content $lmLog -ErrorAction SilentlyContinue | Select-Object -Last 20
                throw "cyphalm_bench_native $profile failed (exit $LASTEXITCODE)"
            }
            $lmJson = Get-Content $lmLog -Raw | ConvertFrom-Json
            $lmResults["cyphalm.$profile.bpc"] = [double]$lmJson.bpc
            Write-Host ("  bpc={0:N4} corpus={1}" -f $lmJson.bpc, $lmJson.corpus)
        } finally {
            Pop-Location
        }
    }
}

# --- Snapshot tables + metrics ---
$tablesSrc = Join-Path $root "bench\report\tables"
$baselineRoot = Join-Path $root "bench\report\baselines"
$baselineDir = Join-Path $baselineRoot "native_$stamp"
New-Item -ItemType Directory -Force -Path $baselineDir | Out-Null

Write-Host ""
Write-Host "== Saving baseline snapshot: $baselineDir ==" -ForegroundColor Yellow
Copy-Item -Path (Join-Path $tablesSrc "*.json") -Destination $baselineDir -Force
$summarySrc = Join-Path $root "bench\report\summary.json"
if (Test-Path $summarySrc) {
    Copy-Item $summarySrc (Join-Path $baselineDir "summary.json") -Force
}
$manifestSrc = Join-Path $root "bench\report\figures\figures_manifest.json"
if (Test-Path $manifestSrc) {
    Copy-Item $manifestSrc (Join-Path $baselineDir "figures_manifest.json") -Force
}

$metrics = Extract-BaselineMetrics -TablesDir $tablesSrc -LmResults $lmResults
$metricsPath = Join-Path $baselineDir "baseline_metrics.json"
$metrics | ConvertTo-Json -Depth 6 | Set-Content -Path $metricsPath -Encoding UTF8

$latestPath = Join-Path $baselineRoot "latest_metrics.json"
$previous = $null
if (Test-Path $latestPath) {
    $prevJson = Get-Content $latestPath -Raw | ConvertFrom-Json
    $previous = [ordered]@{}
    $prevJson.PSObject.Properties | ForEach-Object { $previous[$_.Name] = $_.Value }
}

$comparison = Compare-BaselineMetrics -Current $metrics -Previous $previous
$comparisonPath = Join-Path $baselineDir "baseline_comparison.json"
$comparison | ConvertTo-Json -Depth 4 | Set-Content -Path $comparisonPath -Encoding UTF8
Copy-Item $metricsPath $latestPath -Force

# --- Summary ---
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "BASELINE METRICS ($stamp)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
foreach ($row in ($comparison | Sort-Object metric)) {
    $line = "  $($row.metric): $($row.current)"
    if ($null -ne $row.previous) {
        $sign = if ($row.delta -ge 0) { "+" } else { "" }
        $line += "  (prev $($row.previous), ${sign}$($row.delta)"
        if ($null -ne $row.pct_change) { $line += ", $($row.pct_change)%)" } else { $line += ")" }
    }
    Write-Host $line
}
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Saved: $baselineDir" -ForegroundColor Green
Write-Host "Metrics: $metricsPath" -ForegroundColor Green
if ($previous) {
    Write-Host "Compared vs previous: $latestPath (pre-run snapshot)" -ForegroundColor Green
} else {
    Write-Host "No previous baseline - first lock recorded." -ForegroundColor Yellow
}

exit 0
