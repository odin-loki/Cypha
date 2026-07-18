# Demo Cypha's unique AI surface: online train + OOD + uncertainty-rank (REST).
# Requires: cypha_rest built; optional model path.
param(
    [string]$RestUrl = "http://127.0.0.1:8765",
    [string]$BuildDir = "native/build_ewc_d16"
)

$ErrorActionPreference = "Stop"
Write-Host "== Cypha capability demo =="
Write-Host "REST: $RestUrl"
Write-Host "Endpoints to exercise manually or via curl:"
Write-Host "  GET  $RestUrl/health"
Write-Host "  POST $RestUrl/train          (online step)"
Write-Host "  POST $RestUrl/predict        (with epistemic)"
Write-Host "  GET/POST $RestUrl/uncertainty-rank"
Write-Host "  GET  $RestUrl/intelligence/report"
Write-Host ""
Write-Host "Product claim (honest):"
Write-Host "  - Online adaptation without full retrain"
Write-Host "  - OOD / epistemic uncertainty surfaces"
Write-Host "  - Zero forgetting via per-task model files (D16F)"
Write-Host "  - Shared-model CL: EWC optional; task-sticky via CYPHA_D16_TASK_STICKY=1"
Write-Host ""
try {
    $r = Invoke-WebRequest -Uri "$RestUrl/health" -UseBasicParsing -TimeoutSec 2
    Write-Host "health: $($r.StatusCode) $($r.Content)"
} catch {
    Write-Host "REST not running. Start: $BuildDir\cypha_rest.exe"
    exit 0
}
