# One-shot native CyphaLM validation: build, CTest, pytest, REST LM smoke, optional 40k bench.
param(
    [string]$BuildDir = "C:\Temp\cypha_native_build6",
    [switch]$SkipBuild,
    [switch]$Run40kHybrid
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent

if (-not $SkipBuild) {
    cmake -S (Join-Path $root "native") -B $BuildDir -DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_EXPERIMENT_DB=OFF -G Ninja
    cmake --build $BuildDir --target cyphalm_bench_native cyphalm_parity cyphalm_checkpoint_parity cypha_rest --parallel
}

ctest --test-dir $BuildDir -R native_cyphalm --output-on-failure
if ($LASTEXITCODE -ne 0) { throw "CTest failed" }

$env:CYPHALM_PARITY_BIN = Join-Path $BuildDir "cyphalm_parity.exe"
$env:CYPHALM_CHECKPOINT_PARITY_BIN = Join-Path $BuildDir "cyphalm_checkpoint_parity.exe"
$env:CYPHALM_BENCH_NATIVE_BIN = Join-Path $BuildDir "cyphalm_bench_native.exe"
$env:CYPHA_REST_BIN = Join-Path $BuildDir "cypha_rest.exe"

python -m pytest (Join-Path $root "tests\test_cyphalm_native_parity.py") (Join-Path $root "tests\test_cyphalm_rest_lm_smoke.py") -q
if ($LASTEXITCODE -ne 0) { throw "pytest failed" }

python (Join-Path $root "native\scripts\smoke_cyphalm_rest_lm.py")
if ($LASTEXITCODE -ne 0) { throw "REST LM smoke failed" }

if ($Run40kHybrid) {
    $runExe = "C:\Temp\cyphalm_validate_hybrid40k.exe"
    Copy-Item (Join-Path $BuildDir "cyphalm_bench_native.exe") $runExe -Force
    $outJson = "C:\Temp\validate_hybrid40k.json"
    $outErr = "${outJson}.err"
    cmd /c "set CYPHALM_TRAIN_LOG_EVERY=100000&& cd /d $BuildDir&& `"$runExe`" --mode hybrid --profile d17 --n-train 40000 --n-eval 2000 --threads 1 > `"$outJson`" 2> `"$outErr`""
    if ($LASTEXITCODE -ne 0) { throw "40k hybrid bench failed" }
    Get-Content $outJson -Raw | Write-Host
}

Write-Host "OK cyphalm_native_validate" -ForegroundColor Green
