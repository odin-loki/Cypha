# Safe native sweep: never kills processes; never overwrites slot exe while running.
param(
    [string]$BuildDir = "C:\Temp\cypha_native_build6",
    [int]$NTrain = 300000,
    [int]$NEval = 2000,
    [string]$Profile = "d17",
    [string]$SlotExe = "C:\Temp\cyphalm_bench_slot1.exe",
    [string]$Results = "docs\native\CYPHALM_NATIVE_BENCH_RESULTS.jsonl",
    [int]$LogEvery = 50000,
    [string[]]$Modes = @("hybrid", "char_lstm", "ssm", "ssm_gria", "context_bank", "spectral")
)

$ErrorActionPreference = "Stop"
$srcExe = Join-Path $BuildDir "cyphalm_bench_native.exe"
if (-not (Test-Path $srcExe)) { throw "missing $srcExe" }
if (-not (Test-Path $SlotExe)) { Copy-Item $srcExe $SlotExe -Force }

$lock = "C:\Temp\cyphalm_bench.lock"
if (Test-Path $lock) { throw "lock exists: $lock" }
New-Item -ItemType File -Path $lock -Force | Out-Null

$env:CYPHALM_TRAIN_LOG_EVERY = "$LogEvery"
$modes = $Modes
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not (Test-Path (Join-Path $root "native"))) { $root = Split-Path $PSScriptRoot -Parent }
$outPath = Join-Path $root $Results

Push-Location $BuildDir
try {
    foreach ($mode in $modes) {
        Write-Host ("=== " + $mode + " n=" + $NTrain + " ===") -ForegroundColor Cyan
        $logFile = "C:\Temp\cyphalm_bench_" + $mode + "_${NTrain}.log"
        $proc = Start-Process -FilePath $SlotExe -ArgumentList @(
            "--mode", $mode, "--profile", $Profile,
            "--n-train", "$NTrain", "--n-eval", "$NEval", "--threads", "1"
        ) -NoNewWindow -Wait -PassThru -RedirectStandardOutput $logFile -RedirectStandardError ($logFile + ".err")
        if ($proc.ExitCode -ne 0) {
            Get-Content ($logFile + ".err") -ErrorAction SilentlyContinue | Select-Object -Last 5
            throw ("bench failed mode=" + $mode + " exit=" + $proc.ExitCode)
        }
        $json = Get-Content $logFile -Raw
        Write-Host $json
        $line = $json | ConvertFrom-Json | ConvertTo-Json -Compress
        $meta = @{
            timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
            build = $BuildDir
            slot = $SlotExe
        }
        $record = ($meta | ConvertTo-Json -Compress).TrimEnd('}')
        $record = $record + "," + ($line.TrimStart('{'))
        Add-Content -Path $outPath -Value $record
    }
} finally {
    Pop-Location
    Remove-Item $lock -Force -ErrorAction SilentlyContinue
}

Write-Host ("Done. Results: " + $outPath)
