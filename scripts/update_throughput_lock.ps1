# Throughput-lock scaffold (Perf §0.4). Local smoke only — does not hard-fail CI.
# Writes bench/THROUGHPUT_LOCK.json with chars/sec (wall Measure-Command) + score_matrix µs/row.
param(
    [string]$BuildDir = "",
    [string]$OutFile = "",
    [int]$NTrain = 2048,
    [int]$NEval = 64,
    [switch]$SmokeOnly
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
if (-not $OutFile) {
    $OutFile = Join-Path $root "bench\THROUGHPUT_LOCK.json"
}
if (-not $BuildDir) {
    foreach ($cand in @(
            (Join-Path $root "native\build-windows-msvc\Release"),
            (Join-Path $root "native\build-windows-msvc"),
            (Join-Path $root "native\build\Release"),
            (Join-Path $root "native\build")
        )) {
        if (Test-Path (Join-Path $cand "cyphalm_bench_native.exe")) { $BuildDir = $cand; break }
        if (Test-Path (Join-Path $cand "cyphalm_bench_native")) { $BuildDir = $cand; break }
    }
}
if (-not $BuildDir) {
    Write-Host "update_throughput_lock: no build dir with cyphalm_bench_native; writing schema stub only" -ForegroundColor Yellow
}

$benchExe = $null
$smokeExe = $null
if ($BuildDir) {
    foreach ($name in @("cyphalm_bench_native.exe", "cyphalm_bench_native")) {
        $p = Join-Path $BuildDir $name
        if (Test-Path $p) { $benchExe = $p; break }
    }
    foreach ($name in @("throughput_lock_smoke.exe", "throughput_lock_smoke")) {
        $p = Join-Path $BuildDir $name
        if (Test-Path $p) { $smokeExe = $p; break }
    }
}

$scoreUsPerRow = $null
$deviceInfo = "unknown"
if ($smokeExe) {
    $smokeOut = & $smokeExe 2>&1 | Out-String
    if ($smokeOut -match "score_matrix_us_per_row=([0-9.]+)") {
        $scoreUsPerRow = [double]$Matches[1]
    }
    if ($smokeOut -match "device=([^\r\n]+)") {
        $deviceInfo = $Matches[1].Trim()
    }
}

$charsPerSec = $null
$wallMs = $null
if ($benchExe -and -not $SmokeOnly) {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $benchExe --mode hybrid --profile d17 --n-train $NTrain --n-eval $NEval --threads 1 --bench-seed 42
    if ($LASTEXITCODE -ne 0) {
        Write-Host "update_throughput_lock: bench exited $LASTEXITCODE (recording null chars/sec)" -ForegroundColor Yellow
    } else {
        $sw.Stop()
        $wallMs = [math]::Round($sw.Elapsed.TotalMilliseconds, 3)
        if ($wallMs -gt 0) {
            $charsPerSec = [math]::Round(($NTrain / ($wallMs / 1000.0)), 3)
        }
    }
}

$lock = [ordered]@{
    schema_version     = 1
    status             = "scaffold"
    measured_at_utc    = (Get-Date).ToUniversalTime().ToString("o")
    toolchain_note     = "MSVC Release recommended; MinGW not a gate"
    device_info        = $deviceInfo
    d17_slice          = [ordered]@{
        n_train       = $NTrain
        n_eval        = $NEval
        threads       = 1
        bench_seed    = 42
        wall_ms       = $wallMs
        chars_per_sec = $charsPerSec
    }
    score_matrix       = [ordered]@{
        n               = 32
        d               = 64
        K               = 16
        us_per_row      = $scoreUsPerRow
    }
    ci_hard_fail       = $false
    notes              = "Wave 1 scaffold - do not gate CI until medians stabilize"
}

$dir = Split-Path $OutFile -Parent
if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
($lock | ConvertTo-Json -Depth 6) | Set-Content -Path $OutFile -Encoding utf8
Write-Host "update_throughput_lock: wrote $OutFile"
Write-Host "  chars_per_sec=$charsPerSec  score_matrix_us_per_row=$scoreUsPerRow"
exit 0
