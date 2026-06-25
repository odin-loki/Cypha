# Math integration bench report (Phase 26 prep): hybrid baseline vs --math-integration.
param(
    [string]$BuildDir = (Join-Path (Split-Path $PSScriptRoot -Parent) "native\build_math"),
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$reportPath = Join-Path $root "bench\MATH_INTEGRATION_REPORT.md"
$nTrain = 400
$nEval = 80

function BinPath {
    param([string]$Stem)
    Join-Path $BuildDir "$Stem.exe"
}

function Parse-BenchJson {
    param([string]$Stdout)
    if ([string]::IsNullOrWhiteSpace($Stdout)) { return $null }
    try {
        return $Stdout.Trim() | ConvertFrom-Json
    } catch {
        return $null
    }
}

function Extract-Metrics {
    param(
        [object]$Json,
        [string]$Label
    )
    $bpc = $null
    $kappa = $null
    $allComplete = $null
    if ($null -eq $Json) {
        return [ordered]@{
            label = $Label
            bpc = $bpc
            kappa = $kappa
            all_complete = $allComplete
            exit_note = "no JSON stdout"
        }
    }
    if ($null -ne $Json.bpc) { $bpc = [double]$Json.bpc }
    if ($Json.profile_completeness) {
        $allComplete = [bool]$Json.profile_completeness.all_complete
        if ($null -ne $Json.profile_completeness.kappa) {
            $kappa = [double]$Json.profile_completeness.kappa
        }
    }
    if ($null -eq $kappa -and $Json.math_integration -and $null -ne $Json.math_integration.kappa) {
        $kappa = [double]$Json.math_integration.kappa
    }
    if ($null -eq $kappa -and $Json.intelligence_profile -and $Json.intelligence_profile.kappa) {
        $kappa = [double]$Json.intelligence_profile.kappa
    }
    return [ordered]@{
        label = $Label
        bpc = $bpc
        kappa = $kappa
        all_complete = $allComplete
        exit_note = "ok"
    }
}

Write-Host "Cypha math integration bench (Phase 26 prep)" -ForegroundColor Cyan
Write-Host "  repo:   $root"
Write-Host "  build:  $BuildDir"
Write-Host "  report: $reportPath"
Write-Host ""

if (-not $SkipBuild) {
    Write-Host "== CMake configure (Release) ==" -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
    cmake -S (Join-Path $root "native") -B $BuildDir `
        -DCMAKE_BUILD_TYPE=Release `
        -DCYPHA_BUILD_QT=OFF `
        -G Ninja
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

    Write-Host "== CMake build (cyphalm_bench_native) ==" -ForegroundColor Yellow
    cmake --build $BuildDir --target cyphalm_bench_native --parallel
    if ($LASTEXITCODE -ne 0) { throw "cmake --build failed" }
} elseif (-not (Test-Path $BuildDir)) {
    throw "BuildDir missing: $BuildDir"
}

$benchExe = BinPath "cyphalm_bench_native"
if (-not (Test-Path $benchExe)) {
    throw "missing $benchExe"
}

$env:CYPHA_BENCH_FAST = "1"
$commonArgs = @(
    "--profile", "d17",
    "--mode", "hybrid",
    "--n-train", "$nTrain",
    "--n-eval", "$nEval",
    "--intelligence-profile"
)

Write-Host "== hybrid baseline (--intelligence-profile) ==" -ForegroundColor Yellow
$baselineOut = & $benchExe @commonArgs 2>&1 | Out-String
$baselineCode = $LASTEXITCODE
$baselineJson = Parse-BenchJson -Stdout $baselineOut
$baseline = Extract-Metrics -Json $baselineJson -Label "hybrid_baseline"

Write-Host "== hybrid + --math-integration ==" -ForegroundColor Yellow
$mathOut = & $benchExe @commonArgs "--math-integration" 2>&1 | Out-String
$mathCode = $LASTEXITCODE
$mathJson = Parse-BenchJson -Stdout $mathOut
$math = Extract-Metrics -Json $mathJson -Label "math_integration"

$stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd HH:mm:ss UTC")
$lines = @(
    "# Math Integration Bench Report",
    "",
    "Generated: $stamp",
    "",
    "Build: ``$BuildDir``",
    "",
    "Settings: ``CYPHA_BENCH_FAST=1``, profile **d17**, mode **hybrid**, ``n_train=$nTrain``, ``n_eval=$nEval``.",
    "",
    "## Results",
    "",
    "| Run | Exit | BPC | κ | profile_completeness.all_complete |",
    "|-----|------|-----|---|-----------------------------------|"
)

function Fmt-Num($v) {
    if ($null -eq $v) { return "n/a" }
    return ("{0:N6}" -f $v)
}
function Fmt-Bool($v) {
    if ($null -eq $v) { return "n/a" }
    if ($v) { return "true" }
    return "false"
}

$lines += "| hybrid baseline | $baselineCode | $(Fmt-Num $baseline.bpc) | $(Fmt-Num $baseline.kappa) | $(Fmt-Bool $baseline.all_complete) |"
$lines += "| hybrid + math-integration | $mathCode | $(Fmt-Num $math.bpc) | $(Fmt-Num $math.kappa) | $(Fmt-Bool $math.all_complete) |"
$lines += ""
$lines += "## Notes"
$lines += ""
$lines += "- Baseline: ``cyphalm_bench_native --intelligence-profile``"
$lines += "- Math integration: adds ``--math-integration`` (profile-guided 7-stat navigation loss during train)"
$lines += "- Validation gate: ``cypha_bench_run --domain-tag d40`` (``math_integration_ready`` when exit 0 + complete profile + finite BPC)"

$lines -join "`n" | Set-Content -Path $reportPath -Encoding UTF8
Write-Host ""
Write-Host "Wrote $reportPath" -ForegroundColor Green

if ($baselineCode -ne 0 -and $mathCode -ne 0) {
    Write-Host "Warning: both bench runs exited non-zero" -ForegroundColor Yellow
}
