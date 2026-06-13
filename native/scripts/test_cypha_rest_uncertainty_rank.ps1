# CTest: GET /uncertainty-rank (active learning — entropy rank over feature rows).
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
$fix = Join-Path $root "fixtures"
$cypha = Join-Path $fix "reference.cypha"
$ff = Join-Path $fix "f_field.json"

$port = 18102
$args = @(
  "--listen", "127.0.0.1:$port",
  "--cypha", $cypha,
  "--f-field-json", $ff
)
$p = Start-Process -FilePath $exe -ArgumentList $args -PassThru -WindowStyle Hidden
try {
  Start-Sleep -Seconds 2
  $bodyPath = Join-Path $env:TEMP "cypha_uncertainty_rank_body.json"
  @'
{"rows":[[0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0]],"top_n":2}
'@ | Set-Content -NoNewline -Encoding utf8 $bodyPath
  $resp = curl.exe -s -w "`n%{http_code}" -X POST "http://127.0.0.1:$port/uncertainty-rank" `
    -H "Content-Type: application/json" --data-binary "@$bodyPath"
  Remove-Item -Force $bodyPath -ErrorAction SilentlyContinue
  $lines = $resp -split "`n"
  $code = $lines[-1]
  $content = ($lines[0..($lines.Length - 2)] -join "`n")
  if ($code -ne "200") { throw "unexpected status $code body=$content" }
  $j = $content | ConvertFrom-Json
  if ($j.indices.Count -ne 2) { throw "expected top_n=2 indices, got $($j.indices.Count)" }
  if ($j.entropies.Count -ne 2) { throw "expected 2 entropies" }
  if ($j.entropies[0] -lt $j.entropies[1]) { throw "indices should be sorted by descending entropy" }
} finally {
  Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
}
Write-Host "test_cypha_rest_uncertainty_rank: OK"
