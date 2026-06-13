# Smoke-test MinGW cypha_rest.exe (Windows).
# Run from repo root: powershell -File native/scripts/smoke_cypha_rest_mingw.ps1
# With regression sidecar: powershell -File native/scripts/smoke_cypha_rest_mingw.ps1 -WithRegression
param([switch]$WithRegression)
$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$bd = Join-Path $root "native/build-mingw-w64"
$exe = Join-Path $bd "cypha_rest.exe"
if (-not (Test-Path $exe)) { throw "Missing $exe; build MinGW first (see build_cypha_rest_mingw_wsl.ps1)." }

$args = @(
    "--listen", "127.0.0.1:18099",
    "--cypha", (Join-Path $root "fixtures/reference.cypha"),
    "--f-field-json", (Join-Path $root "fixtures/f_field.json")
)
if ($WithRegression) {
    $reg = Join-Path $root "fixtures/regression_head.json"
    if (-not (Test-Path $reg)) { throw "Missing $reg" }
    $args += @("--regression-json", $reg)
}

$p = Start-Process -FilePath $exe -ArgumentList $args -PassThru -WindowStyle Hidden
try {
    Start-Sleep -Seconds 2
    $r = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:18099/health"
    Write-Host $r.Content
    if ($r.StatusCode -ne 200) { exit 1 }

    $predBody = '{"input":[0,0,0,0,0,0,0,0],"use_gh":true,"return_explanation":false}'
    $pred = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:18099/predict" -Body $predBody -ContentType "application/json"
    if (-not $pred.label) { throw "/predict missing label" }
    if ($WithRegression) {
        if ($null -eq $pred.regression_val) { throw "expected regression_val with --regression-json" }
        Write-Host ("regression_val={0} uncertainty={1}" -f $pred.regression_val, $pred.uncertainty)
    }
} finally {
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
}
Write-Host "smoke_cypha_rest_mingw: OK"
