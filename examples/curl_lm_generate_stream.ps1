# Stream CyphaLM generation (SSE) — FastAPI only; requires CyphaLM loaded.
# Load first: POST /lm/load with {"checkpoint_path": "path/to/ckpt"}
$body = Get-Content -Raw "$PSScriptRoot\lm_generate_body.json" | ConvertFrom-Json
$body.stream = $true
$json = $body | ConvertTo-Json -Compress
Invoke-WebRequest -Uri "http://127.0.0.1:7749/generate/stream" -Method POST -ContentType "application/json" -Body $json
