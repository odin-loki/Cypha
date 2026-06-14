# Publish a GitHub Release for a tag using maintainer notes from create_release_notes.ps1.
# Phase 10 (v2.3.10+): highlights from create_release_notes.ps1 — d24 production lock,
# cypha_baseline_lock --run all, hybrid EWC bias/W_slow, ci_federated_tls_windows.ps1, 99 CTests.
# Usage:
#   pwsh -File scripts/publish_release.ps1
#   pwsh -File scripts/publish_release.ps1 -Tag v2.3.10 -Draft
param(
  [string]$Tag = "v2.3.10",
  [switch]$Draft
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Push-Location $root
try {
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
    Write-Warning "gh is not authenticated. Run 'gh auth login' then retry."
    Write-Warning "After auth: verify with 'gh auth status' and list existing releases via 'gh release list'."
    Write-Warning ($auth | Out-String)
    exit 2
  }

  Write-Host "Authenticated. Existing releases:" -ForegroundColor Cyan
  gh release list --limit 5
  if ($LASTEXITCODE -ne 0) {
    Write-Warning "gh release list failed (non-fatal); continuing with release create."
  }

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
