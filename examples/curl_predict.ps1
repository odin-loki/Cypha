# One-liner curl against the /predict endpoint using cypha_predict_body.json
# Adjust port and input vector to match your model's latent dim.
$body = Get-Content "$PSScriptRoot\cypha_predict_body.json" -Raw
Invoke-RestMethod -Method POST -Uri "http://127.0.0.1:8099/predict" `
    -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 5

