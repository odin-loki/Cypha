# Example load against a live uvicorn (trusted network only).
# Requires ApacheBench on PATH (e.g. Apache httpd "ab.exe").
#
#   $env:CYPHA_API_HOST = "127.0.0.1"
#   $env:CYPHA_API_PORT = "7749"
#   .\scripts\loadtest_ab_predict_example.ps1

$hostName = if ($env:CYPHA_API_HOST) { $env:CYPHA_API_HOST } else { "127.0.0.1" }
$port = if ($env:CYPHA_API_PORT) { $env:CYPHA_API_PORT } else { "7749" }
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$body = Join-Path $root "examples\cypha_predict_body.json"
$url = "http://${hostName}:${port}/predict"

& ab -n 500 -c 10 -T "application/json" -p $body $url
