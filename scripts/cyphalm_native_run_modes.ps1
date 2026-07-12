# Sequential 300k bench for build6 — cmd isolation (stderr progress logs won't abort PowerShell).
param(
    [string]$BuildDir = "C:\Temp\cypha_native_build6",
    [int]$NTrain = 300000,
    [int]$NEval = 2000,
    [string]$Profile = "d17",
    [string[]]$ModeList = @("hybrid", "char_lstm", "ssm", "ssm_gria", "context_bank", "spectral")
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "lib\NativeBenchCommon.ps1")

$srcExe = Join-Path $BuildDir "cyphalm_bench_native.exe"
$runExe = "C:\Temp\cyphalm_bench_build6_run.exe"
if (-not (Test-Path $srcExe)) { throw "missing $srcExe" }
Copy-Item $srcExe $runExe -Force

$root = Get-CyphaRepoRoot -ScriptRoot $PSScriptRoot
$outPath = Join-Path $root "docs\native\CYPHALM_NATIVE_BENCH_RESULTS.jsonl"

foreach ($mode in $ModeList) {
    Write-Host "=== $mode @ $NTrain ===" -ForegroundColor Cyan
    $outJson = "C:\Temp\bench6_${mode}_${NTrain}.json"
    $outErr = "${outJson}.err"
    if (Test-Path $outJson) { Remove-Item $outJson -Force }
    if (Test-Path $outErr) { Remove-Item $outErr -Force }
    $cmd = "set CYPHALM_TRAIN_LOG_EVERY=100000&& cd /d $BuildDir&& `"$runExe`" --mode $mode --profile $Profile --n-train $NTrain --n-eval $NEval --threads 1 > `"$outJson`" 2> `"$outErr`""
    cmd /c $cmd
    if ($LASTEXITCODE -ne 0) {
        if (Test-Path $outErr) { Get-Content $outErr -Tail 5 }
        throw "mode=$mode exit=$LASTEXITCODE"
    }
    if (-not (Test-Path $outJson) -or (Get-Item $outJson).Length -lt 10) {
        throw "mode=$mode produced empty output"
    }
    $json = Get-Content $outJson -Raw
    Write-Host $json
    $line = ($json | ConvertFrom-Json | ConvertTo-Json -Compress)
    $meta = @{ timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss"); build = $BuildDir; exe = $runExe }
    $record = ($meta | ConvertTo-Json -Compress).TrimEnd('}')
    $record = $record + "," + ($line.TrimStart('{'))
    Add-Content -Path $outPath -Value $record
}

Write-Host "Done. Appended to $outPath"
