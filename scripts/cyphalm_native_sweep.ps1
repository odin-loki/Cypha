# Native CyphaLM benchmark sweep - run ONE instance only (Windows linker lock).
param(
    [string]$BuildDir = "C:\Temp\cypha_native_build6",
    [int]$NTrain = 300000,
    [int]$NEval = 2000,
    [string]$Profile = "d17",
    [string]$Results = "docs\native\CYPHALM_NATIVE_BENCH_RESULTS.jsonl",
    [string]$SlotExe = "C:\Temp\cyphalm_bench_slot0.exe"
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "lib\NativeBenchCommon.ps1")

$srcExe = Join-Path $BuildDir "cyphalm_bench_native.exe"
if (-not (Test-Path $srcExe)) { throw "missing $srcExe" }
Copy-Item $srcExe $SlotExe -Force

# Do not kill other cyphalm processes here — use dedicated slot exe only.

$lock = "C:\Temp\cyphalm_bench.lock"
if (Test-Path $lock) { throw "lock exists: $lock" }
New-Item -ItemType File -Path $lock -Force | Out-Null

$modes = @("hybrid", "char_lstm", "ssm", "ssm_gria", "context_bank", "spectral")
# Bug fix: this used to double up Split-Path (scripts -> repo -> grandparent), landing
# one directory too high before a Test-Path fallback quietly recovered. Resolve once,
# correctly, via the shared helper (same "$PSScriptRoot -> Split-Path -Parent" pattern
# used by the working overnight scripts).
$root = Get-CyphaRepoRoot -ScriptRoot $PSScriptRoot
$outPath = Join-Path $root $Results

Push-Location $BuildDir
try {
    foreach ($mode in $modes) {
        Write-Host ("=== " + $mode + " n=" + $NTrain + " ===") -ForegroundColor Cyan
        $json = & $SlotExe --mode $mode --profile $Profile --n-train $NTrain --n-eval $NEval --threads 1
        if ($LASTEXITCODE -ne 0) { throw ("bench failed mode=" + $mode + " exit=" + $LASTEXITCODE) }
        Write-Host $json
        $line = $json | ConvertFrom-Json | ConvertTo-Json -Compress
        $meta = @{ timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss"); build = $BuildDir; slot = $SlotExe }
        $record = ($meta | ConvertTo-Json -Compress).TrimEnd('}')
        $record = $record + "," + ($line.TrimStart('{'))
        Add-Content -Path $outPath -Value $record
    }
} finally {
    Pop-Location
    Remove-Item $lock -Force -ErrorAction SilentlyContinue
}

Write-Host ("Done. Results: " + $outPath)
