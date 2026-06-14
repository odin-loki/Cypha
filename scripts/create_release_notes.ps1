# Emit markdown release notes for a tag (local maintainer helper).
# Usage: pwsh -File scripts/create_release_notes.ps1 -Tag v2.3.5
param(
  [Parameter(Mandatory = $true)]
  [string]$Tag
)

$ErrorActionPreference = "Stop"
$ver = $Tag -replace '^v', ''

$highlights = @{
  "2.3.10" = @(
    "Intelligence Stats **Phase 10**: hybrid EWC **GRIA bias** + SSM **W_slow** layer-0 Fisher (extends Phase 9 U/V + W_fast + α).",
    "Bench domain **d24** production lock validation — ``cypha_bench_run --domain-tag d24``; profile ``bench/config/d24_production_lock_profile.json``.",
    "``cypha_baseline_lock --run all`` (d17 + d21 + cell-sweep); CTest ``native_d24_production_lock_smoke``.",
    "Windows federated TLS mirror: ``scripts/ci_federated_tls_windows.ps1`` (vcpkg / ``OPENSSL_ROOT_DIR``).",
    "CI gate **99 CTests**; release notes v2.3.10 Phase 10 template."
  )
  "2.3.9" = @(
    "Intelligence Stats **Phase 9**: hybrid EWC **weight** Fisher on GRIA **U**/**V** + SSM **W_fast**; EWC anchor/Fisher in CyphaLM ``checkpoint.json``.",
    "``scripts/run_overnight_all.ps1`` — unified overnight runner (D17 + d21 RPSM + cell sweep + baseline-lock refresh).",
    "Bench **d23** overnight lock validation; CTest ``native_d23_overnight_lock_smoke``.",
    "CI gate **98 CTests**; optional federated TLS job (``scripts/ci_federated_tls_linux.sh``)."
  )
  "2.3.8" = @(
    "Intelligence Stats **Phase 8**: bench domain **d22** cross-profile (d18 intelligence + d16 EWC probe + d20 cell sweep smoke).",
    "``bench/config/d22_intelligence_cross_profile.json`` + combined report ``bench/report/tables/d22_intelligence_cross_profile.json``.",
    "CTest ``native_d22_cross_smoke`` via ``cypha_bench_run --domain-tag d22``.",
    "CI gate **94 CTests**; release notes v2.3.8 template."
  )
  "2.3.7" = @(
    "Intelligence Stats **Phase 7**: baseline lock ``bench/BASELINE_LOCK.json`` (D17 hybrid **2.873 BPC** @ 300k).",
    "``scripts/publish_release.ps1`` - local ``gh release create`` wrapper (graceful fail if ``gh`` not authed).",
    "CTest ``native_overnight_mini_smoke`` (800-train overnight wiring, ``CYPHA_BENCH_FAST=1``).",
    "CI gate **93 CTests**; optional federated TLS job; release workflow Phase 7 asset notes."
  )
  "2.3.6" = @(
    "Cell hypothesis **28-variant overnight sweep**: ``cypha_cell_hypothesis_sweep --overnight-sweep`` writes ``results/variant_*.json`` + ``summary.csv``.",
    "Bench domain **d20** overnight smoke; CTest ``native_cell_hypothesis_overnight_smoke`` (3 variants @ n_train=200).",
    "``scripts/run_d17_overnight.ps1`` - D17 WikiText 300k runner (``CYPHA_BENCH_OVERNIGHT=1``).",
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
    "RPSM Option B scaffold, cell hypothesis H02-H14, D17 full WikiText profile.",
    "Native-only runtime (P7); **80 CTests** CI gate."
  )
}

Write-Output "## Cypha $Tag"
Write-Output ""
Write-Output "Native C++ release **$ver** - prebuilt CLI + Linux AppImage (see [packaging/README.md](../packaging/README.md))."
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
