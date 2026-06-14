# Stub: emit markdown release notes for a tag (local maintainer helper).
# Usage: pwsh -File scripts/create_release_notes.ps1 -Tag v2.2.8
param(
  [Parameter(Mandatory = $true)]
  [string]$Tag
)

$ErrorActionPreference = "Stop"
$ver = $Tag -replace '^v', ''

Write-Output "## Cypha $Tag"
Write-Output ""
Write-Output "Native C++ release **$ver** — prebuilt CLI + Linux AppImage (see packaging/README.md)."
Write-Output ""
Write-Output "### Changes since previous tag"
Write-Output ""

try {
  $prev = git describe --tags --abbrev=0 "$Tag^" 2>$null
  if ($LASTEXITCODE -eq 0 -and $prev) {
    git log --oneline "$prev..$Tag"
  } else {
    git log --oneline -20
  }
} catch {
  Write-Output "_Run from repo root with git history available._"
}

Write-Output ""
Write-Output "### Assets"
Write-Output "- cypha-$ver-linux-x86_64.tar.gz"
Write-Output "- cypha-$ver-linux-x86_64.AppImage"
Write-Output "- cypha-$ver-windows-x86_64.zip"
