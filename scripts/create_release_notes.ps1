# Emit markdown release notes for a tag (local maintainer helper).
# Usage: pwsh -File scripts/create_release_notes.ps1 -Tag v2.3.5
param(
  [Parameter(Mandatory = $true)]
  [string]$Tag
)

$ErrorActionPreference = "Stop"
$ver = $Tag -replace '^v', ''

$highlights = @{
  "2.3.6" = @(
    "Cell hypothesis **28-variant overnight sweep**: ``cypha_cell_hypothesis_sweep --overnight-sweep`` writes ``results/variant_*.json`` + ``summary.csv``.",
    "Bench domain **d20** overnight smoke; CTest ``native_cell_hypothesis_overnight_smoke`` (3 variants @ n_train=200).",
    "``scripts/run_d17_overnight.ps1`` — D17 WikiText 300k runner (``CYPHA_BENCH_OVERNIGHT=1``).",
    "Intelligence Stats **Phase 6** docs; CI gate **85 CTests**."
  )
  "2.3.5" = @(
    "Qt CyphaLM generate tab: **epistemic halt** checkbox (Paper IV r_eu gate; matches REST ``/generate``).",
    "D17 WikiText **overnight** profile @ 300k tokens: ``d17_wikitext_overnight_profile.json``, ``--overnight`` / ``CYPHA_BENCH_OVERNIGHT=1``.",
    "CTest ``native_d17_wikitext_overnight_smoke`` (500-train wiring check).",
    "Intelligence Stats **Phase 5** docs; CI gate **81 CTests**."
  )
  "2.3.4" = @(
    "Intelligence Stats Phase 4: EWC, curriculum, epistemic halt on REST ``/generate``, federated merge stub.",
    "RPSM Option B scaffold, cell hypothesis H02–H14, D17 full WikiText profile.",
    "Native-only runtime (P7); **80 CTests** CI gate."
  )
}

Write-Output "## Cypha $Tag"
Write-Output ""
Write-Output "Native C++ release **$ver** — prebuilt CLI + Linux AppImage (see [packaging/README.md](../packaging/README.md))."
Write-Output ""

if ($highlights.ContainsKey($ver)) {
  Write-Output "### Highlights"
  Write-Output ""
  foreach ($line in $highlights[$ver]) {
    Write-Output "- $line"
  }
  Write-Output ""
}

Write-Output "### Changes since previous tag"
Write-Output ""

try {
  $prev = git describe --tags --abbrev=0 "$Tag^" 2>$null
  if ($LASTEXITCODE -eq 0 -and $prev) {
    git log --oneline "$prev..$Tag"
  } else {
    git log --oneline -30
  }
} catch {
  Write-Output "_Run from repo root with git history available._"
}

Write-Output ""
Write-Output "### Validation"
Write-Output ""
Write-Output '```powershell'
Write-Output "cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_QT=ON"
Write-Output "cmake --build native/build --config Release"
Write-Output 'ctest --test-dir native/build -R "native_" --output-on-failure'
Write-Output '```'
Write-Output ""

Write-Output "### Assets"
Write-Output "- cypha-$ver-linux-x86_64.tar.gz"
Write-Output "- cypha-$ver-linux-x86_64.AppImage"
Write-Output "- cypha-$ver-windows-x86_64.zip"
Write-Output ""
Write-Output "_Publish via GitHub Releases requires ``gh auth login``._"
