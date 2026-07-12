# One-shot native CyphaLM validation: build, CTest, REST LM smoke, optional 40k bench.
param(
    [string]$BuildDir = "C:\Temp\cypha_native_build6",
    [switch]$SkipBuild,
    [switch]$Run40kHybrid
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "lib\NativeBenchCommon.ps1")

$root = Get-CyphaRepoRoot -ScriptRoot $PSScriptRoot

if (-not $SkipBuild) {
    cmake -S (Join-Path $root "native") -B $BuildDir -DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_EXPERIMENT_DB=OFF -G Ninja
    cmake --build $BuildDir --target cyphalm_bench_native cyphalm_parity cyphalm_checkpoint_parity cypha_rest --parallel
}

ctest --test-dir $BuildDir -R native_cyphalm --output-on-failure
if ($LASTEXITCODE -ne 0) { throw "CTest failed" }

$restExe = Join-Path $BuildDir "cypha_rest.exe"
if (-not (Test-Path $restExe)) { throw "missing $restExe" }

Write-Host "== cypha_rest /health smoke ==" -ForegroundColor Yellow
$refCypha = Join-Path $root "fixtures\reference.cypha"
$fField = Join-Path $root "fixtures\f_field.json"
$restArgs = @("--listen", "127.0.0.1:18765", "--cypha", $refCypha, "--f-field-json", $fField)
$restProc = Start-Process -FilePath $restExe -ArgumentList $restArgs -PassThru -WindowStyle Hidden
try {
    Start-Sleep -Seconds 2
    $health = curl.exe -s -o NUL -w "%{http_code}" "http://127.0.0.1:18765/health"
    if ($health -ne "200") { throw "cypha_rest /health returned $health" }
    Write-Host "cypha_rest /health: OK"
} finally {
    if (-not $restProc.HasExited) {
        Stop-Process -Id $restProc.Id -Force -ErrorAction SilentlyContinue
    }
}

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
