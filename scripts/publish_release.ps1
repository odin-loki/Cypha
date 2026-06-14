# Publish a GitHub Release for a tag using maintainer notes from create_release_notes.ps1.
# Phase 20 (v2.3.20, prep): d34 repo smoke hygiene validation, CYPHA_VALIDATE_REPO_SMOKE_HYGIENE, 112 CTests when d34 merged.
# Phase 19 (v2.3.19, shipped): d33 release publish validation, verify_release_publish.ps1,
# CYPHA_VALIDATE_RELEASE_PUBLISH, 111 CTests.
# Phase 18 (v2.3.18, shipped): d32 production complete, validate_production_complete.ps1,
# start_poll_finalize_background.ps1, CYPHA_VALIDATE_PRODUCTION_COMPLETE, gh auth preflight, 110 CTests.
# Phase 17 (v2.3.17, shipped): d31 post-overnight pipeline, poll_and_finalize_overnight.ps1,
# cleanup_legacy_results.ps1, CYPHA_VALIDATE_POST_OVERNIGHT_PIPELINE, 109 CTests.
# Phase 16 (v2.3.16, shipped): d30 artifact path hygiene, migrate_legacy_results.ps1, overnight progress logging, 108 CTests.
# Phase 15 (v2.3.15, shipped): d29 release readiness, commit_production_lock.ps1, watch_production_overnight.ps1, 107 CTests.
# Phase 14 (v2.3.14): d28 overnight completion, finalize_production_overnight.ps1, 106 CTests.
# Phase 13 (v2.3.13): production overnight tier (300k), run_production_overnight.ps1, d27, validate -Production, 104 CTests.
# Phase 12 (v2.3.12): medium overnight tier, d26, validate_baseline_lock.ps1, publish -DryRun, corpus_and_d25 CI job.
# Phase 11 (v2.3.11): WikiText download, gutenberg fallback, corpus_smoke, d25, 101 CTests.
# Phase 10 (v2.3.10): d24 production lock, cypha_baseline_lock --run all, hybrid EWC bias/W_slow.
# Usage:
#   pwsh -File scripts/publish_release.ps1
#   pwsh -File scripts/publish_release.ps1 -Tag v2.3.14 -Draft
#   pwsh -File scripts/publish_release.ps1 -Tag v2.3.14 -DryRun          # notes to stdout + temp file; no gh
#   pwsh -File scripts/publish_release.ps1 -Tag v2.3.14 -NotesOnly       # alias for -DryRun
param(
  [string]$Tag = "v2.3.20",
  [switch]$Draft,
  [Alias("NotesOnly")]
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Push-Location $root
try {
  $notesScript = Join-Path $PSScriptRoot "create_release_notes.ps1"
  if (-not (Test-Path $notesScript)) {
    throw "missing $notesScript"
  }

  $notesFile = Join-Path $env:TEMP "cypha_release_notes_$($Tag -replace '[^a-zA-Z0-9._-]','_').md"
  $notesRunner = Get-Command pwsh -ErrorAction SilentlyContinue
  if ($notesRunner) {
    & pwsh -NoProfile -File $notesScript -Tag $Tag | Out-File -FilePath $notesFile -Encoding utf8
  } else {
    & powershell -NoProfile -ExecutionPolicy Bypass -File $notesScript -Tag $Tag | Out-File -FilePath $notesFile -Encoding utf8
  }

  if ($DryRun) {
    Write-Host "== release notes preview ($Tag) ==" -ForegroundColor Cyan
    Get-Content -Path $notesFile -Raw | Write-Host
    Write-Host "Notes written to: $notesFile" -ForegroundColor Green
    Write-Host "Dry run - skipped gh release create." -ForegroundColor Yellow
    exit 0
  }

  $gh = Get-Command gh -ErrorAction SilentlyContinue
  if (-not $gh) {
    Write-Warning "GitHub CLI (gh) not found on PATH. Install from https://cli.github.com/ and run 'gh auth login'."
    exit 2
  }

  $prevEap = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  $auth = gh auth status 2>&1
  $authOk = ($LASTEXITCODE -eq 0)
  $ErrorActionPreference = $prevEap
  if (-not $authOk) {
    Write-Host "ERROR: GitHub CLI is not authenticated (gh auth status failed)." -ForegroundColor Red
    Write-Host "Run:  gh auth login" -ForegroundColor Yellow
    Write-Host "Then verify with:  gh auth status" -ForegroundColor DarkGray
    if ($auth) {
      Write-Host ($auth | Out-String) -ForegroundColor DarkGray
    }
    exit 1
  }

  Write-Host "Authenticated. Existing releases:" -ForegroundColor Cyan
  gh release list --limit 5
  if ($LASTEXITCODE -ne 0) {
    Write-Warning "gh release list failed (non-fatal); continuing with release create."
  }

  $ghArgs = @(
    "release", "create", $Tag,
    "--title", "Cypha $Tag",
    "--notes-file", $notesFile
  )
  if ($Draft) {
    $ghArgs += "--draft"
  }

  Write-Host "Creating GitHub release $Tag ..." -ForegroundColor Cyan
  & gh @ghArgs
  if ($LASTEXITCODE -ne 0) {
    throw "gh release create failed exit=$LASTEXITCODE"
  }

  Write-Host "Release $Tag published. Attach Linux/Windows bundles from CI or local packaging if needed." -ForegroundColor Green
} finally {
  Pop-Location
}
