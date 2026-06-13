# CTest: validate cypha_rest JSON key shapes (PORT_CONTRACT §3).
param(
  [Parameter(Mandatory = $true)]
  [string]$BuildDir
)
$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$exe = Join-Path $BuildDir "cypha_rest.exe"
$regExe = Join-Path $BuildDir "registry_register.exe"
$schemaExe = Join-Path $BuildDir "rest_schema_contract.exe"
foreach ($p in @($exe, $regExe, $schemaExe)) {
  if (-not (Test-Path $p)) { throw "Missing $p" }
}

$regRoot = Join-Path $BuildDir "rest_schema_registry"
$fix = Join-Path $root "fixtures"
$cypha = Join-Path $fix "reference.cypha"
$ff = Join-Path $fix "f_field.json"
$card = Join-Path $fix "registry_register/card.json"

if (Test-Path $regRoot) { Remove-Item -Recurse -Force $regRoot }
& $regExe $regRoot "native_schema_smoke" "0.0.1" $cypha $card --overwrite --and-verify
if ($LASTEXITCODE -ne 0) { throw "registry_register failed" }
Copy-Item $ff (Join-Path $regRoot "native_schema_smoke/0.0.1/f_field.json")

$port = 18103
$modelInReg = Join-Path $regRoot "native_schema_smoke/0.0.1/model.cypha"
$args = @(
  "--listen", "127.0.0.1:$port",
  "--cypha", $modelInReg,
  "--f-field-json", $ff,
  "--registry", $regRoot,
  "--preload-registry"
)
$p = Start-Process -FilePath $exe -ArgumentList $args -PassThru -WindowStyle Hidden -WorkingDirectory $BuildDir
try {
  Start-Sleep -Seconds 2
  & $schemaExe --base-url "http://127.0.0.1:$port/"
  if ($LASTEXITCODE -ne 0) { throw "rest_schema_contract failed" }
} finally {
  Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
}
Write-Host "test_rest_schema_contract: OK"
