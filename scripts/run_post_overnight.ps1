# Phase 22: post-overnight maintainer pipeline - poll/finalize/commit + production verify.
# Chains poll_and_finalize_overnight.ps1 (-Once pre-check, then -Force) and verify_production_pipeline.ps1.
# BuildDir auto-detect (from running run_production_overnight.ps1) is handled inside poll_and_finalize_overnight.ps1
# when -BuildDir is the default native/build.
# Does not call gh; authenticate manually after success:
#   gh auth login
#   gh auth status
# Then: pwsh -File scripts/publish_release.ps1 -Tag v2.3.22
#
# Usage:
#   pwsh -File scripts/run_post_overnight.ps1
#   pwsh -File scripts/run_post_overnight.ps1 -BuildDir native/build_p13
#   pwsh -File scripts/run_post_overnight.ps1 -SkipPoll
#   pwsh -File scripts/run_post_overnight.ps1 -AllowPending
param(
    [string]$BuildDir = "native/build",
    [switch]$SkipPoll,
    [switch]$AllowPending
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$pollScript = Join-Path $PSScriptRoot "poll_and_finalize_overnight.ps1"
$verifyScript = Join-Path $PSScriptRoot "verify_production_pipeline.ps1"

Write-Host "== run post overnight (Phase 22) ==" -ForegroundColor Cyan
Write-Host "  build: $BuildDir" -ForegroundColor DarkGray
if ($SkipPoll) {
    Write-Host "  poll:  skipped (-SkipPoll)" -ForegroundColor DarkGray
} else {
    Write-Host "  poll:  -Once pre-check, then -Force finalize+commit" -ForegroundColor DarkGray
}
if ($AllowPending) {
    Write-Host "  verify: -AllowPending" -ForegroundColor DarkGray
}

$exitCode = 0

if (-not $SkipPoll) {
    Write-Host ""
    Write-Host "== poll_and_finalize_overnight.ps1 -Once (pre-check) ==" -ForegroundColor Cyan
    & $pollScript -BuildDir $BuildDir -Once
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "run_post_overnight: overnight processes still running (-Once pre-check)" -ForegroundColor Yellow
        Write-Host "  watch: pwsh -File scripts/watch_production_overnight.ps1 -Once" -ForegroundColor DarkGray
        Write-Host "  retry when done: pwsh -File scripts/run_post_overnight.ps1" -ForegroundColor DarkGray
        exit 1
    }

    Write-Host ""
    Write-Host "== poll_and_finalize_overnight.ps1 -Force ==" -ForegroundColor Cyan
    & $pollScript -BuildDir $BuildDir -Force
    if ($LASTEXITCODE -ne 0) {
        Write-Host "run_post_overnight: FAIL (poll_and_finalize_overnight -Force)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
} else {
    Write-Host ""
    Write-Host "run_post_overnight: -SkipPoll (assuming finalize/commit already done)" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "== verify_production_pipeline.ps1 ==" -ForegroundColor Cyan
$verifyArgs = @{
    BuildDir = $BuildDir
}
if ($AllowPending) {
    $verifyArgs.AllowPending = $true
}
& $verifyScript @verifyArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "run_post_overnight: FAIL (verify_production_pipeline)" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "run_post_overnight: OK" -ForegroundColor Green
Write-Host "Next (manual): gh auth login; gh auth status" -ForegroundColor Yellow
Write-Host "Then publish:  pwsh -File scripts/publish_release.ps1 -Tag v2.3.22" -ForegroundColor Yellow
exit 0
