# Smoke: cypha_rest serves Studio Web UI at GET / (HTML).
param(
  [Parameter(Mandatory = $true)]
  [string]$BuildDir
)
$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$exe = Join-Path $BuildDir "cypha_rest.exe"
if (-not (Test-Path $exe)) {
  $exe = Join-Path $BuildDir "cypha_rest"
}
if (-not (Test-Path $exe)) { throw "Missing cypha_rest in $BuildDir" }

$staticSrc = Join-Path $root "native\tools\static"
$staticDst = Join-Path $BuildDir "static"
if (Test-Path $staticDst) { Remove-Item -Recurse -Force $staticDst }
Copy-Item -Recurse -Force $staticSrc $staticDst

$args = @(
  "--listen", "127.0.0.1:18100",
  "--cypha", (Join-Path $root "fixtures\reference.cypha"),
  "--f-field-json", (Join-Path $root "fixtures\f_field.json")
)

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $exe
$psi.Arguments = ($args -join ' ')
$psi.WorkingDirectory = $BuildDir
$psi.UseShellExecute = $false
$psi.EnvironmentVariables["CYPHA_REST_STATIC_DIR"] = $staticSrc
$p = [System.Diagnostics.Process]::Start($psi)
try {
  Start-Sleep -Seconds 2
  $r = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:18100/"
  if ($r.StatusCode -ne 200) { throw "GET / status $($r.StatusCode)" }
  if ($r.Content -notmatch "<html") { throw "GET / did not return HTML" }
  if ($r.Content -notmatch "Cypha Studio") { throw "GET / missing Cypha Studio title" }

  $js = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:18100/ui/app.js"
  if ($js.StatusCode -ne 200) { throw "GET /ui/app.js status $($js.StatusCode)" }
  if ($js.Content -notmatch "refreshHealth") { throw "GET /ui/app.js unexpected body" }
} finally {
  Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
}
Write-Host "test_cypha_rest_ui_smoke: OK"
