# Stream CyphaLM generation (SSE) from native cypha_rest (port 8099).
# Requires CyphaLM at startup: --cyphalm-checkpoint or CYPHALM_CHECKPOINT.
$body = Get-Content -Raw "$PSScriptRoot\lm_generate_body.json" | ConvertFrom-Json
$body.stream = $true
$json = $body | ConvertTo-Json -Compress
Invoke-WebRequest -Uri "http://127.0.0.1:8099/generate/stream" -Method POST -ContentType "application/json" -Body $json


