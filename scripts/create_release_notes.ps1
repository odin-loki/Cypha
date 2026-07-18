# Emit markdown release notes for a tag (local maintainer helper).
# Usage: pwsh -File scripts/create_release_notes.ps1 -Tag v2.3.5
param(
  [Parameter(Mandatory = $true)]
  [string]$Tag
)

$ErrorActionPreference = "Stop"
$ver = $Tag -replace '^v', ''

$highlights = @{
  "2.3.25" = @(
    "One Cypha cutover: single public type cypha::Cypha (classify + regress + latent sample + sequence)",
    "REST: /sample /retrieve /sequence/*; health model_type=Cypha; primary predict/update/sample via Cypha",
    "Living sequence spine PGM→Wy (U06); hybrid D17 kept historical; PGM cell checkpoint round-trip",
    "Qt Studio: studio_cypha_ owns classify/sample/sequence; Sample latents UI",
    "Smokes: native_one_cypha_smoke, native_cypha_rest_one_smoke, pgm checkpoint round-trip; CI green"
  )
  "2.3.24" = @(
    "Intelligence Stats **Phase 24** (shipped): bench **d38** production overnight completion certificate - full 300k cross-section + 28-variant cell sweep; profile ``bench/config/d38_overnight_certificate_profile.json``; CTest ``native_d38_overnight_certificate_smoke``.",
    "``poll_and_finalize_overnight.ps1 -AutoCommit`` - post-finalize lock commit when ``n_train >= 300000``; watch variant **STALL_WARNING** detector.",
    "``CYPHA_VALIDATE_OVERNIGHT_CERTIFICATE=1`` - runs d38 when profile exists.",
    "CI gate **116 CTests**; full 300k production overnight **in progress** (maintainer workflow only)."
  )
  "2.3.23" = @(
    "Intelligence Stats **Phase 23** (shipped): bench **d37** overnight lock refresh validation - post-overnight baseline lock update toolchain; profile ``bench/config/d37_lock_refresh_profile.json``; CTest ``native_d37_lock_refresh_smoke``.",
    "``scripts/migrate_inflight_overnight_artifacts.ps1`` - merge in-flight overnight cell-sweep spill from repo-root ``results/`` into ``bench/results/cell_sweep/``.",
    "``finalize_production_overnight.ps1`` best-effort ``update_baseline_lock -Production``; ``publish_release.ps1 -NotesPath`` for offline gh.",
    "CI gate **115 CTests**; full 300k production overnight **in progress** (maintainer workflow only)."
  )
  "2.3.22" = @(
    "Intelligence Stats **Phase 22** (shipped): bench **d36** production pipeline E2E validation - full maintainer overnight-to-publish toolchain; profile ``bench/config/d36_pipeline_e2e_profile.json``; CTest ``native_d36_pipeline_e2e_smoke``.",
    "``scripts/run_post_overnight.ps1`` - poll/finalize/commit wrapper + ``verify_production_pipeline.ps1``; watch **effective_n_train** from latest variant JSON.",
    "``CYPHA_VALIDATE_PIPELINE_E2E=1`` - runs d36 when profile exists.",
    "CI gate **114 CTests**; full 300k production overnight **in progress** (maintainer workflow only)."
  )
  "2.3.21" = @(
    "Intelligence Stats **Phase 21** (shipped): bench **d35** lock commit pipeline validation - post-overnight commit toolchain; profile ``bench/config/d35_lock_commit_pipeline_profile.json``; CTest ``native_d35_lock_commit_pipeline_smoke``.",
    "``scripts/verify_production_pipeline.ps1`` - unified maintainer smoke gate (production complete + release publish + repo smoke cleanup + optional d35).",
    "Overnight watch **25/28 variant progress**; poll dedupe + heartbeat log fix; ``CYPHA_VALIDATE_LOCK_COMMIT_PIPELINE=1``.",
    "CI gate **113 CTests**; full 300k production overnight **in progress** (maintainer workflow only)."
  )
  "2.3.20" = @(
    "Intelligence Stats **Phase 20** (shipped): bench **d34** repo smoke hygiene validation - repo-root ``d*_smoke.json`` leak detection + ``.gitignore`` patterns; profile ``bench/config/d34_repo_smoke_hygiene_profile.json``; CTest ``native_d34_repo_smoke_hygiene_smoke``.",
    "``scripts/cleanup_repo_smoke_artifacts.ps1`` - remove repo-root smoke JSON spill files from local CTest runs (``-DryRun`` / ``-Force``).",
    "Poll heartbeat logging + manual restart docs; ``cypha_native_validate_all.ps1`` env **`CYPHA_VALIDATE_REPO_SMOKE_HYGIENE=1`**.",
    "CI gate **112 CTests**; full 300k production overnight **in progress** (maintainer workflow only)."
  )
  "2.3.19" = @(
    "Intelligence Stats **Phase 19** (shipped): bench **d33** release publish validation - publish script presence + production/overnight-complete gates; profile ``bench/config/d33_release_publish_profile.json``; CTest ``native_d33_release_publish_smoke``.",
    "``scripts/verify_release_publish.ps1`` - production-complete gate + d33 + ``publish_release.ps1 -DryRun`` preview; poll/finalize **BuildDir auto-detect** from running overnight.",
    "``cypha_native_validate_all.ps1`` env **`CYPHA_VALIDATE_RELEASE_PUBLISH=1`** - runs d33 when profile exists.",
    "CI gate **111 CTests**; full 300k production overnight **in progress** (maintainer workflow only)."
  )
  "2.3.18" = @(
    "Intelligence Stats **Phase 18** (shipped): bench **d32** production complete validation - full production tier gate when ``overnight_results.n_train >= 300000``; profile ``bench/config/d32_production_complete_profile.json``; CTest ``native_d32_production_complete_smoke``.",
    "``scripts/validate_production_complete.ps1`` - unified post-overnight validator (baseline lock + finalize + d31/d30); ``scripts/start_poll_finalize_background.ps1``; cell sweep ``overnight_progress.log`` sidecar.",
    "``publish_release.ps1`` ``gh auth`` preflight; ``cypha_native_validate_all.ps1`` env **`CYPHA_VALIDATE_PRODUCTION_COMPLETE=1`** - runs d32 when profile exists.",
    "CI gate **110 CTests**; full 300k production overnight **in progress** (maintainer workflow only)."
  )
  "2.3.17" = @(
    "Intelligence Stats **Phase 17** (shipped): bench **d31** post-overnight pipeline validation - d27->d30 chain + pipeline script presence gate; profile ``bench/config/d31_post_overnight_pipeline_profile.json``; CTest ``native_d31_post_overnight_pipeline_smoke``.",
    "``scripts/poll_and_finalize_overnight.ps1`` - poll until overnight processes exit, then ``finalize_production_overnight.ps1`` + ``commit_production_lock.ps1`` (``-DryRun`` preview; ``-Force`` to commit; never pushes).",
    "``scripts/cleanup_legacy_results.ps1`` + ``migrate_legacy_results.ps1 -ArchiveLegacy`` - one-shot legacy ``results/`` migrate + remove or archive.",
    "``cypha_native_validate_all.ps1`` env **`CYPHA_VALIDATE_POST_OVERNIGHT_PIPELINE=1`** - runs d31 when profile exists.",
    "CI gate **109 CTests**; full 300k production overnight **in progress** (maintainer workflow only)."
  )
  "2.3.16" = @(
    "Intelligence Stats **Phase 16** (shipped): bench **d30** artifact path hygiene validation - legacy repo-root ``results/`` path detection in ``cell_sweep_results.artifact_path``, verifies ``bench/results/.gitkeep``; profile ``bench/config/d30_artifact_hygiene_profile.json``; CTest ``native_d30_artifact_hygiene_smoke``.",
    "``scripts/migrate_legacy_results.ps1`` - merge repo-root ``results/`` cell-sweep artifacts into ``bench/results/cell_sweep/`` (``-DryRun`` / ``-RemoveLegacy``).",
    "**Overnight progress logging** - stderr ``[cyphalm]`` / ``[cell_sweep]`` (full sweep only); ``run_d17_overnight.ps1`` tees to ``bench/results/overnight_d17_<timestamp>.log``.",
    "``cypha_native_validate_all.ps1`` env **`CYPHA_VALIDATE_ARTIFACT_HYGIENE=1`** - runs d30 when profile exists.",
    "CI gate **108 CTests**; full 300k production overnight **in progress** (maintainer workflow only)."
  )
  "2.3.15" = @(
    "Intelligence Stats **Phase 15** (shipped): bench **d29** release readiness validation - schema + production tier (d27) + overnight-complete (d28) + release script presence; profile ``bench/config/d29_release_readiness_profile.json``; CTest ``native_d29_release_readiness_smoke``.",
    "``cypha_native_validate_all.ps1`` env **`CYPHA_VALIDATE_OVERNIGHT_COMPLETE=1`** - runs **`cypha_bench_run --domain-tag d28`** after baseline lock validate; **`CYPHA_VALIDATE_RELEASE_READINESS=1`** - runs d29; **`CYPHA_VALIDATE_PRODUCTION=1`** - production lock validate; **`CYPHA_STRICT_TEST_COUNT=1`** - fail when count â‰  **107**.",
    "``scripts/commit_production_lock.ps1`` - chains ``finalize_production_overnight.ps1``, then stage/commit updated ``bench/BASELINE_LOCK.json`` after 300k overnight (``-DryRun`` / ``-Force``; maintainer-only; never pushes).",
    "``scripts/watch_production_overnight.ps1`` - stall-aware production overnight log/process watcher.",
    "CI gate **107 CTests**; full 300k production overnight **not** run in CI (maintainer workflow; **in progress**); ``gh auth login`` required for GitHub Release publish."
  )
  "2.3.14" = @(
    "Intelligence Stats **Phase 14**: **baseline lock status validator fix** - ``validate_baseline_lock.ps1`` and ``baseline_lock_validate`` accept ``medium_smoke`` and ``production`` (fixes production/medium overnight lock validation).",
    "**Cell sweep artifact path** - default overnight output ``bench/results/cell_sweep`` (``bench_paths::results_dir()``); wired through ``cypha_baseline_lock --output-dir``, ``update_baseline_lock.ps1``, and ``run_overnight_all.ps1``.",
    "Bench domain **d28** unified overnight completion validation - cross-check ``overnight_results``, ``rpsm_results``, and ``cell_sweep_results`` for matching ``n_train``/``n_eval``; profile ``bench/config/d28_overnight_complete_profile.json``; CTest ``native_d28_overnight_complete_smoke``.",
    "``scripts/finalize_production_overnight.ps1`` - post-overnight gate: ``validate_baseline_lock.ps1 -Production``, d27 + d28 bench domains, lock section summary; chained from ``run_production_overnight.ps1`` on success.",
    "CI gate **106 CTests**; full 300k production overnight **not** run in CI (maintainer workflow only)."
  )
  "2.3.13" = @(
    "Intelligence Stats **Phase 13**: **production overnight tier** - 300k train / 2000 eval (``-Production`` on overnight scripts, ``cypha_baseline_lock --production``, ``status=production`` in lock JSON).",
    "Dedicated maintainer runner ``scripts/run_production_overnight.ps1`` - chains ``run_overnight_all.ps1 -Production``, logs to ``bench/results/production_overnight_<timestamp>.log``.",
    "Bench domain **d27** production overnight lock validation - ``cypha_bench_run --domain-tag d27``; profile ``bench/config/d27_production_lock_profile.json``; CTest ``native_d27_production_lock_smoke``.",
    "``scripts/validate_baseline_lock.ps1 -Production`` and ``baseline_lock_validate --production`` - when ``overnight_results.n_train >= 300000``, require ``status=production`` or ``completed`` and BPC within **0.05** of d17 hybrid **2.873** pin; ``CYPHA_VALIDATE_PRODUCTION=1`` in ``cypha_native_validate_all.ps1``.",
    "``scripts/monitor_overnight.ps1`` - lightweight poll of ``bench/BASELINE_LOCK.json`` ``run_at`` / status while overnight jobs run.",
    "CI gate **104 CTests**; full 300k production overnight **not** run in CI (maintainer workflow only)."
  )
  "2.3.12" = @(
    "Intelligence Stats **Phase 12**: **medium overnight** profile - between mini smoke and full 300k (``CYPHA_BENCH_MEDIUM_OVERNIGHT=1``; profile ``bench/config/d26_medium_overnight_profile.json``).",
    "Bench domain **d26** medium overnight lock validation - ``cypha_bench_run --domain-tag d26``; CTest ``native_d26_medium_overnight_smoke``.",
    "``cypha_baseline_lock --validate`` baseline lock validator - schema + BPC drift check against ``bench/BASELINE_LOCK.json``; CTest ``native_baseline_lock_validator_smoke``.",
    "CI optional ``corpus_and_d25`` job (WikiText fetch + ``native_corpus_smoke`` / ``native_d25_corpus_smoke``); ``publish_release.ps1 -DryRun`` / ``-NotesOnly`` preview.",
    "CI gate **103 CTests**; release notes v2.3.12 Phase 12 template."
  )
  "2.3.11" = @(
    "Intelligence Stats **Phase 11**: WikiText-2 download (``scripts/download_wikitext2.ps1``, ``scripts/download_wikitext2.sh``); gutenberg fallback for d17/d21 when WikiText absent.",
    "``corpus_smoke`` CLI + bench **d25** corpus readiness validation; CTest ``native_d25_corpus_smoke``, ``native_corpus_smoke``.",
    "Overnight ``-Fast`` propagates ``CYPHA_BENCH_FAST=1`` through ``run_d17_overnight.ps1``, ``run_rpsm_overnight.ps1``, ``run_overnight_all.ps1``, ``update_baseline_lock.ps1``.",
    "CI gate **101 CTests**; release notes v2.3.11 Phase 11 template."
  )
  "2.3.10" = @(
    "Intelligence Stats **Phase 10**: hybrid EWC **GRIA bias** + SSM **W_slow** layer-0 Fisher (extends Phase 9 U/V + W_fast + Î±).",
    "Bench domain **d24** production lock validation - ``cypha_bench_run --domain-tag d24``; profile ``bench/config/d24_production_lock_profile.json``.",
    "``cypha_baseline_lock --run all`` (d17 + d21 + cell-sweep); CTest ``native_d24_production_lock_smoke``.",
    "Windows federated TLS mirror: ``scripts/ci_federated_tls_windows.ps1`` (vcpkg / ``OPENSSL_ROOT_DIR``).",
    "CI gate **99 CTests**; release notes v2.3.10 Phase 10 template."
  )
  "2.3.9" = @(
    "Intelligence Stats **Phase 9**: hybrid EWC **weight** Fisher on GRIA **U**/**V** + SSM **W_fast**; EWC anchor/Fisher in Cypha ``checkpoint.json``.",
    "``scripts/run_overnight_all.ps1`` - unified overnight runner (D17 + d21 RPSM + cell sweep + baseline-lock refresh).",
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
    "Qt Cypha generate tab: **epistemic halt** checkbox (Paper IV r_eu gate; matches REST ``/generate``).",
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
