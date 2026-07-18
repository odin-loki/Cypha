# Large-context / large-data Cypha profiling suite.
#
# Axes swept:
#   1) PGM capacity (slot count N) — pgm_cell_bench --scale large|xl
#   2) Train data budget — sample-efficiency tiers + H23/hybrid BPC
#   3) Sequence length — needle-haystack depths
#   4) Optional train-step phase trace (CYPHA_PERF_TRACE) at a mid/large budget
#
# Usage:
#   powershell -File scripts/run_large_context_profile.ps1 -Tier Medium
#   powershell -File scripts/run_large_context_profile.ps1 -Tier Large -BuildDir native/build-pgm
#   powershell -File scripts/run_large_context_profile.ps1 -Tier XL -SkipNeedle   # longest
#   powershell -File scripts/run_large_context_profile.ps1 -Tier Medium -PgmOnly
#
# Tiers (approx wall time @ ~130 chars/s, single-threaded):
#   Medium : PGM large + BPC tiers to 100k (hybrid+H23) + haystack to 2k   (~1–2 h)
#   Large  : + 300k BPC + haystack to 4k + perf trace @ 40k               (~3–5 h)
#   XL     : + PGM xl + haystack to 8k + hidden-dim 256 @ 40k             (~6–10 h)

param(
    [ValidateSet("Medium", "Large", "XL")]
    [string]$Tier = "Medium",
    [string]$BuildDir = "native/build-pgm",
    [switch]$SkipBuild,
    [switch]$SkipPgm,
    [switch]$SkipBpc,
    [switch]$SkipNeedle,
    [switch]$SkipPerfTrace,
    [switch]$PgmOnly,
    [switch]$BpcOnly,
    [int]$Threads = 1
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "lib\NativeBenchCommon.ps1")
$root = Get-CyphaRepoRoot -ScriptRoot $PSScriptRoot

$resultsRoot = Join-Path $root "bench\results\large_context_profile"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = Join-Path $resultsRoot "${Tier}_$timestamp"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$logPath = Join-Path $runDir "suite.log"
Write-Host "Large-context profile tier=$Tier" -ForegroundColor Cyan
Write-Host "Artifacts: $runDir" -ForegroundColor Yellow
Write-Host "Log: $logPath" -ForegroundColor Yellow

function Write-SuiteLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $Message
    Add-Content -Path $logPath -Value $line
    Write-Host $line
}

function Resolve-NativeExe {
    param([string]$Name)
    $candidates = @(
        (Join-Path $root (Join-Path $BuildDir "$Name.exe")),
        (Join-Path $root (Join-Path $BuildDir $Name)),
        (Join-Path $root (Join-Path $BuildDir "Release\$Name.exe")),
        (Join-Path $root (Join-Path $BuildDir "RelWithDebInfo\$Name.exe"))
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    return $null
}

if ($PgmOnly) {
    $SkipBpc = $true
    $SkipNeedle = $true
    $SkipPerfTrace = $true
}
if ($BpcOnly) {
    $SkipPgm = $true
    $SkipNeedle = $true
    $SkipPerfTrace = $true
}

# --- Tier budgets -----------------------------------------------------------
$pgmScale = "large"
$pgmSteps = 20000
$bpcTiers = "10000,40000,100000"
$nEval = 2000
$haystack = "512,1024,2048"
$needleEpochs = 5
$perfTraceNTrain = 0
$hiddenDimSweep = @()

switch ($Tier) {
    "Medium" {
        $pgmScale = "large"
        $pgmSteps = 20000
        $bpcTiers = "10000,40000,100000"
        $haystack = "512,1024,2048"
        $needleEpochs = 5
        $perfTraceNTrain = 0
    }
    "Large" {
        $pgmScale = "large"
        $pgmSteps = 30000
        $bpcTiers = "10000,40000,100000,300000"
        $haystack = "1024,2048,4096"
        $needleEpochs = 5
        $perfTraceNTrain = 40000
    }
    "XL" {
        $pgmScale = "xl"
        $pgmSteps = 30000
        $bpcTiers = "10000,40000,100000,300000"
        $haystack = "1024,2048,4096,8192"
        $needleEpochs = 5
        $perfTraceNTrain = 40000
        $hiddenDimSweep = @(256)
    }
}

# --- Build ------------------------------------------------------------------
if (-not $SkipBuild) {
    Write-SuiteLog "Configuring/building native tools under $BuildDir"
    $buildAbs = Join-Path $root $BuildDir
    if (-not (Test-Path (Join-Path $buildAbs "CMakeCache.txt"))) {
        cmake -S (Join-Path $root "native") -B $buildAbs -DCMAKE_BUILD_TYPE=Release
        if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }
    }
    $targets = @(
        "pgm_cell_bench",
        "cyphalm_bench_native",
        "cyphalm_sample_efficiency_curve",
        "cyphalm_needle_haystack"
    )
    foreach ($t in $targets) {
        Write-SuiteLog "cmake --build --target $t"
        cmake --build $buildAbs --config Release --target $t
        if ($LASTEXITCODE -ne 0) { throw "build failed for $t" }
    }
}

$pgmExe = Resolve-NativeExe "pgm_cell_bench"
$benchExe = Resolve-NativeExe "cyphalm_bench_native"
$curveExe = Resolve-NativeExe "cyphalm_sample_efficiency_curve"
$needleExe = Resolve-NativeExe "cyphalm_needle_haystack"

if (-not $SkipPgm -and -not $pgmExe) { throw "missing pgm_cell_bench under $BuildDir" }
if (-not $SkipBpc -and (-not $benchExe -or -not $curveExe)) {
    throw "missing cyphalm_bench_native or cyphalm_sample_efficiency_curve under $BuildDir"
}
if (-not $SkipNeedle -and -not $needleExe) { throw "missing cyphalm_needle_haystack under $BuildDir" }

# Full WikiText for real data budgets (never FAST for this suite).
Remove-Item Env:CYPHA_BENCH_FAST -ErrorAction SilentlyContinue
$env:CYPHA_BENCH_FULL_CORPUS = "1"
$env:CYPHA_BENCH_OVERNIGHT = "1"

$manifest = [ordered]@{
    tier = $Tier
    timestamp = $timestamp
    build_dir = $BuildDir
    pgm_scale = $pgmScale
    pgm_steps = $pgmSteps
    bpc_tiers = $bpcTiers
    n_eval = $nEval
    haystack_chars = $haystack
    needle_epochs = $needleEpochs
    perf_trace_n_train = $perfTraceNTrain
    hidden_dim_sweep = $hiddenDimSweep
    artifacts = @()
}
$manifestPath = Join-Path $runDir "manifest.json"

function Save-Manifest {
    ($manifest | ConvertTo-Json -Depth 6) | Set-Content -Path $manifestPath -Encoding utf8
}

# --- 1) PGM capacity microbench --------------------------------------------
if (-not $SkipPgm) {
    Write-SuiteLog "Phase 1: PGM capacity microbench scale=$pgmScale steps=$pgmSteps"
    $out = Join-Path $runDir "pgm_cell_bench_${pgmScale}.txt"
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $pgmExe --scale $pgmScale --steps $pgmSteps --warmup 500 2>&1 |
            Tee-Object -FilePath $out
    } finally {
        $ErrorActionPreference = $prevEap
    }
    if ($LASTEXITCODE -ne 0) { throw "pgm_cell_bench failed exit=$LASTEXITCODE" }
    $manifest.artifacts += @("pgm_cell_bench_${pgmScale}.txt")
    Save-Manifest
    Write-SuiteLog "Phase 1 done -> $out"
}

# --- 2) Sample-efficiency / BPC at large data ------------------------------
if (-not $SkipBpc) {
    $exeDir = Split-Path $benchExe -Parent
    foreach ($variant in @(@{name = "hybrid"; cell = ""}, @{name = "H23"; cell = "H23"})) {
        Write-SuiteLog "Phase 2: sample-efficiency $($variant.name) tiers=$bpcTiers"
        $out = Join-Path $runDir "sample_efficiency_$($variant.name).json"
        $args = @(
            "--exe-dir", $exeDir,
            "--out", $out,
            "--profile", "d17",
            "--mode", "hybrid",
            "--tiers", $bpcTiers,
            "--n-eval", "$nEval",
            "--bench-seed", "42",
            "--write-table"
        )
        if ($variant.cell) {
            $args += @("--cell-variant", $variant.cell)
        }
        Invoke-NativeWithProgressLog -Exe $curveExe -NativeArgs $args -LogPath $logPath
        if ($LASTEXITCODE -ne 0) {
            throw "sample_efficiency $($variant.name) failed exit=$LASTEXITCODE"
        }
        $manifest.artifacts += @("sample_efficiency_$($variant.name).json")
        Save-Manifest
        Write-SuiteLog "Phase 2 $($variant.name) done -> $out"
    }
}

# --- 3) Needle-haystack (sequence length) ----------------------------------
if (-not $SkipNeedle) {
    Write-SuiteLog "Phase 3: needle-haystack depths=$haystack epochs=$needleEpochs"
    $out = Join-Path $runDir "needle_haystack.json"
    $args = @(
        "--out", $out,
        "--haystack-chars", $haystack,
        "--train-epochs", "$needleEpochs",
        "--write-table"
    )
    Invoke-NativeWithProgressLog -Exe $needleExe -NativeArgs $args -LogPath $logPath
    if ($LASTEXITCODE -ne 0) { throw "needle_haystack failed exit=$LASTEXITCODE" }
    $manifest.artifacts += @("needle_haystack.json")
    Save-Manifest
    Write-SuiteLog "Phase 3 done -> $out"
}

# --- 4) Train-step phase profile at larger budget --------------------------
if (-not $SkipPerfTrace -and $perfTraceNTrain -gt 0) {
    Write-SuiteLog "Phase 4: CYPHA_PERF_TRACE @ n_train=$perfTraceNTrain"
    $env:CYPHA_PERF_TRACE = "1"
    $outJson = Join-Path $runDir "perf_trace_${perfTraceNTrain}.json"
    $outLog = Join-Path $runDir "perf_trace_${perfTraceNTrain}.log"
    $args = @(
        "--profile", "d17",
        "--mode", "hybrid",
        "--n-train", "$perfTraceNTrain",
        "--n-eval", "500",
        "--threads", "1",
        "--bench-seed", "42"
    )
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $raw = & $benchExe @args 2>&1 | Tee-Object -FilePath $outLog
    } finally {
        $ErrorActionPreference = $prevEap
        Remove-Item Env:CYPHA_PERF_TRACE -ErrorAction SilentlyContinue
    }
    if ($LASTEXITCODE -ne 0) { throw "perf_trace failed exit=$LASTEXITCODE" }
    $text = ($raw | ForEach-Object { "$_" }) -join "`n"
    $jsonStart = $text.LastIndexOf("`n{")
    if ($jsonStart -lt 0) { $jsonStart = $text.IndexOf("{") } else { $jsonStart += 1 }
    if ($jsonStart -ge 0) {
        Set-Content -Path $outJson -Value $text.Substring($jsonStart) -Encoding utf8
        $manifest.artifacts += @("perf_trace_${perfTraceNTrain}.json")
    }
    $manifest.artifacts += @("perf_trace_${perfTraceNTrain}.log")
    Save-Manifest
    Write-SuiteLog "Phase 4 done"
}

# --- 5) Hidden-dim scale spot check (XL) -----------------------------------
foreach ($h in $hiddenDimSweep) {
    Write-SuiteLog "Phase 5: hidden-dim=$h @ n_train=40000"
    $out = Join-Path $runDir "bpc_hybrid_h${h}_40000.json"
    $args = @(
        "--profile", "d17",
        "--mode", "hybrid",
        "--lstm-hidden", "$h",
        "--n-train", "40000",
        "--n-eval", "$nEval",
        "--threads", "$Threads",
        "--bench-seed", "42",
        "--intelligence-profile"
    )
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $raw = & $benchExe @args 2>&1 | Tee-Object -FilePath (Join-Path $runDir "bpc_hybrid_h${h}_40000.log")
    } finally {
        $ErrorActionPreference = $prevEap
    }
    if ($LASTEXITCODE -ne 0) { throw "hidden-dim $h failed" }
    $text = ($raw | ForEach-Object { "$_" }) -join "`n"
    $jsonStart = $text.LastIndexOf("`n{")
    if ($jsonStart -lt 0) { $jsonStart = $text.IndexOf("{") } else { $jsonStart += 1 }
    if ($jsonStart -ge 0) {
        Set-Content -Path $out -Value $text.Substring($jsonStart) -Encoding utf8
        $manifest.artifacts += @("bpc_hybrid_h${h}_40000.json")
    }
    Save-Manifest
}

$manifest.completed_at = (Get-Date -Format "o")
Save-Manifest
Write-SuiteLog "SUITE COMPLETE tier=$Tier -> $runDir"
Write-Host ""
Write-Host "Done. Results in: $runDir" -ForegroundColor Green
Write-Host "Manifest: $manifestPath"
