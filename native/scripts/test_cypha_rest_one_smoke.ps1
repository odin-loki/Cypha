# CTest: One Cypha REST smoke — /health, /sample, /retrieve; optional /predict_next.
# Requires cypha_rest + fixtures/reference.cypha (no registry_register).
param(
  [Parameter(Mandatory = $true)]
  [string]$BuildDir,
  [string]$RestExe = ""
)
$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ($RestExe) {
  $exe = $RestExe
} else {
  $exe = Join-Path $BuildDir "cypha_rest.exe"
  if (-not (Test-Path $exe)) {
    $exe = Join-Path $BuildDir "cypha_rest"
  }
}
if (-not (Test-Path $exe)) { throw "Missing cypha_rest in $BuildDir" }

$cypha = Join-Path $root "fixtures\reference.cypha"
$ff = Join-Path $root "fixtures\f_field.json"
$regression = Join-Path $root "fixtures\regression_head.json"
$sidecar = Join-Path $root "fixtures\retrieval\sidecar.json"
foreach ($p in @($cypha, $ff)) {
  if (-not (Test-Path $p)) { throw "Missing $p" }
}
$regressionOk = Test-Path $regression

$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
$listener.Start()
try {
  $port = $listener.LocalEndpoint.Port
} finally {
  $listener.Stop()
}
$listenAddr = "127.0.0.1:$port"
$baseUrl = "http://$listenAddr"

$restArgs = @(
  "--listen", $listenAddr,
  "--cypha", $cypha,
  "--f-field-json", $ff
)
if ($regressionOk) {
  $restArgs += @("--regression-json", $regression)
}

$demoBase = Join-Path $root "examples\demo_cyphalm\demo"
$sequenceOk = (Test-Path "$demoBase.json") -and (Test-Path "$demoBase.npz")
if ($sequenceOk) {
  $restArgs += @("--sequence-checkpoint", $demoBase)
}

$p = Start-Process -FilePath $exe -ArgumentList $restArgs -PassThru -WindowStyle Hidden -WorkingDirectory $BuildDir
try {
  $deadline = (Get-Date).AddSeconds(20)
  $healthy = $false
  while ((Get-Date) -lt $deadline) {
    if ($p.HasExited) {
      throw "cypha_rest exited early (code $($p.ExitCode))"
    }
    try {
      $healthProbe = Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/health" -TimeoutSec 2
      if ($healthProbe.StatusCode -eq 200) {
        $healthy = $true
        break
      }
    } catch { }
    Start-Sleep -Milliseconds 200
  }
  if (-not $healthy) { throw "cypha_rest health timeout" }

  $health = ($healthProbe.Content | ConvertFrom-Json)
  if ($health.model -ne "Cypha") {
    throw "/health model=$($health.model), expected Cypha"
  }

  $predictPayload = @{
    input = @(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  } | ConvertTo-Json -Compress
  $predictResp = Invoke-WebRequest -UseBasicParsing -Method Post -Uri "$baseUrl/predict" `
    -Body $predictPayload -ContentType "application/json" -TimeoutSec 15
  if ($predictResp.StatusCode -ne 200) {
    throw "POST /predict status $($predictResp.StatusCode)"
  }
  $predict = $predictResp.Content | ConvertFrom-Json
  if (-not $predict.label) { throw "/predict missing label" }
  if ($regressionOk) {
    if ($null -eq $predict.regression_val) {
      throw "/predict regression_val null with regression_head.json present"
    }
    Write-Host "/predict regression_val=$($predict.regression_val)"
  } else {
    Write-Host "/predict regress skip (no regression_head.json)"
  }

  $samplePayload = @{
    mode      = "langevin"
    input     = @(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    n_samples = 2
    n_steps   = 4
    seed      = 42
  } | ConvertTo-Json -Compress
  $sampleResp = Invoke-WebRequest -UseBasicParsing -Method Post -Uri "$baseUrl/sample" `
    -Body $samplePayload -ContentType "application/json" -TimeoutSec 15
  if ($sampleResp.StatusCode -ne 200) {
    throw "POST /sample status $($sampleResp.StatusCode)"
  }
  $sample = $sampleResp.Content | ConvertFrom-Json
  if ($sample.space -ne "latent") { throw "/sample space=$($sample.space), expected latent" }
  if (-not $sample.samples -or $sample.samples.Count -lt 1) {
    throw "/sample returned no samples"
  }

  if (Test-Path $sidecar) {
    $sc = Get-Content $sidecar -Raw | ConvertFrom-Json
    $case = $sc.cases[0]
    $inputArr = @()
    foreach ($v in $case.query_x) { $inputArr += [double]$v }
    $dbArr = @()
    foreach ($row in $case.database_x) {
      $r = @()
      foreach ($v in $row) { $r += [double]$v }
      $dbArr += ,$r
    }
    $retrievePayload = @{
      input    = $inputArr
      database = $dbArr
      top_k    = [int]$case.top_k
    } | ConvertTo-Json -Compress -Depth 20
  } else {
    $retrievePayload = @{
      input    = @(1.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
      database = @(
        @(1.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        @(0.1, 1.7, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0)
      )
      top_k    = 2
    } | ConvertTo-Json -Compress -Depth 20
  }
  $retrieveResp = Invoke-WebRequest -UseBasicParsing -Method Post -Uri "$baseUrl/retrieve" `
    -Body $retrievePayload -ContentType "application/json" -TimeoutSec 15
  if ($retrieveResp.StatusCode -ne 200) {
    throw "POST /retrieve status $($retrieveResp.StatusCode)"
  }
  $retrieve = $retrieveResp.Content | ConvertFrom-Json
  if (-not $retrieve.hits) { throw "/retrieve missing hits" }

  if ($sequenceOk) {
    if (-not $health.sequence_loaded) {
      throw "/health sequence_loaded=false after --sequence-checkpoint"
    }
    $pnPayload = '{"token_id":1}' 
    $pnResp = Invoke-WebRequest -UseBasicParsing -Method Post -Uri "$baseUrl/predict_next" `
      -Body $pnPayload -ContentType "application/json" -TimeoutSec 15
    if ($pnResp.StatusCode -ne 200) {
      throw "POST /predict_next status $($pnResp.StatusCode)"
    }
    $pn = $pnResp.Content | ConvertFrom-Json
    if (-not $pn.log_probs) { throw "/predict_next missing log_probs" }
  }
} finally {
  if (-not $p.HasExited) {
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    $p.WaitForExit(5000) | Out-Null
  }
}
Write-Host "test_cypha_rest_one_smoke: OK"
