# CTest: multi-model cypha_rest (registry preload + model field on /predict).
# Expects a built cypha_rest.exe under native/build_intel or native/build_verify_phase_a.
param(
  [string]$BuildDir = ""
)
$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ($BuildDir -eq "") {
  foreach ($cand in @("build_intel", "build_verify_phase_a", "build-mingw-w64")) {
    $p = Join-Path $root "native/$cand/cypha_rest.exe"
    if (Test-Path $p) { $BuildDir = Join-Path $root "native/$cand"; break }
  }
}
if ($BuildDir -eq "" -or -not (Test-Path (Join-Path $BuildDir "cypha_rest.exe"))) {
  throw "cypha_rest.exe not found; build native first."
}

$exe = Join-Path $BuildDir "cypha_rest.exe"
$regRoot = Join-Path $BuildDir "rest_multi_model_registry"
$regExe = Join-Path $BuildDir "registry_register.exe"
$fix = Join-Path $root "fixtures"
$cypha = Join-Path $fix "reference.cypha"
$ff = Join-Path $fix "f_field.json"
$card = Join-Path $fix "registry_register/card.json"

if (Test-Path $regRoot) { Remove-Item -Recurse -Force $regRoot }
New-Item -ItemType Directory -Path $regRoot | Out-Null

& $regExe $regRoot "native_reg_smoke" "0.0.1" $cypha $card --overwrite --and-verify
if ($LASTEXITCODE -ne 0) { throw "registry_register failed" }
Copy-Item $ff (Join-Path $regRoot "native_reg_smoke/0.0.1/f_field.json")

$port = 18101
$modelInReg = Join-Path $regRoot "native_reg_smoke/0.0.1/model.cypha"
$args = @(
  "--listen", "127.0.0.1:$port",
  "--cypha", $modelInReg,
  "--f-field-json", $ff,
  "--registry", $regRoot,
  "--preload-registry"
)
$p = Start-Process -FilePath $exe -ArgumentList $args -PassThru -WindowStyle Hidden
try {
  Start-Sleep -Seconds 2
  $models = Invoke-RestMethod -Uri "http://127.0.0.1:$port/models"
  if ($models.models.Count -lt 1) { throw "GET /models empty" }
  $row = $models.models[0]
  if (-not $row.loaded) { throw "expected loaded=true on preloaded model" }
  if (-not $models.active_model) { throw "expected active_model set" }

  $predBody = '{"input":[0,0,0,0,0,0,0,0],"use_gh":true}'
  $predDefault = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$port/predict" -Body $predBody -ContentType "application/json"
  if (-not $predDefault.label) { throw "/predict default missing label" }

  $predNamed = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$port/predict" `
    -Body ('{"model":"native_reg_smoke/0.0.1","input":[0,0,0,0,0,0,0,0],"use_gh":true}') `
    -ContentType "application/json"
  if (-not $predNamed.label) { throw "/predict with model field missing label" }
} finally {
  Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
}
Write-Host "test_cypha_rest_multi_model: OK"
