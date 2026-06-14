# One-shot migrate + remove legacy repo-root results/.
#
# Runs migrate_legacy_results.ps1, then -RemoveLegacy when migration succeeds.
#
# Usage:
#   pwsh -File scripts/cleanup_legacy_results.ps1
#   pwsh -File scripts/cleanup_legacy_results.ps1 -DryRun
param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$migrateScript = Join-Path $PSScriptRoot "migrate_legacy_results.ps1"
if (-not (Test-Path $migrateScript)) {
    throw "missing $migrateScript"
}

Write-Host "== cleanup legacy results ==" -ForegroundColor Cyan
if ($DryRun) { Write-Host "  mode: DryRun (migrate plan only; no removal)" -ForegroundColor Yellow }
Write-Host ""

$migrateArgs = @{}
if ($DryRun) { $migrateArgs["DryRun"] = $true }

& $migrateScript @migrateArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "cleanup_legacy_results: migration failed (exit $LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
}

if ($DryRun) {
    Write-Host ""
    Write-Host "DryRun complete. Re-run without -DryRun to migrate and remove legacy results/." -ForegroundColor Yellow
    exit 0
}

& $migrateScript -RemoveLegacy
if ($LASTEXITCODE -ne 0) {
    Write-Host "cleanup_legacy_results: legacy removal failed (exit $LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "cleanup_legacy_results: migration and legacy removal complete." -ForegroundColor Green
exit 0
