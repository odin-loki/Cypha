# Scripts index

Native build, validation, and release helpers for **One Cypha** (`cypha::Cypha`). Narrative "when to run what" lives in **[`docs/README.md`](../docs/README.md)**. **Fixtures / native / schema cadence:** **[`docs/verify/MAINTENANCE.md`](../docs/verify/MAINTENANCE.md)**.

> **Living pin:** D17 hybrid GRIA+LSTM L2 + Wave2 BPTT **2.664 BPC** in `bench/BASELINE_LOCK.json` is the production lock / overnight validation target. Prior L1 pin **2.873** is archived. Living sequence default is **Hybrid** -- [`docs/reports/ONE_CYPHA_CUTOVER.md`](../docs/reports/ONE_CYPHA_CUTOVER.md).

## Validation gates (primary)

| Script | Purpose | Typical output |
|--------|---------|----------------|
| `cypha_native_validate_all.ps1` | Windows full gate: rebuild + CTest (`-R native_`, 214 tests) + bench d01/d04/d17/forecast + fig01–09 + tune dry-run + REST smoke (`-SkipBuild` after rebuild) | console |
| `ci_native_linux.sh` | Linux/WSL mirror of CI **`build_and_test`**: cmake + `ctest -R native_` | console |
| `ci_federated_tls_linux.sh` | Linux/WSL mirror of optional CI **`federated_tls`**: `-DCYPHA_ENABLE_OPENSSL=ON` + `ctest -R native_federated_tls` | console |
| `ci_federated_tls_windows.ps1` | Windows mirror of optional CI **`federated_tls`**: OpenSSL via vcpkg / `OPENSSL_ROOT_DIR`, `ctest -R native_federated_tls_smoke` | console |
| `cyphalm_native_validate.ps1` | Cypha sequence CTest subset + checkpoint smoke | console |
| `wsl_verify.sh` | WSL: native build + CTest + optional REST smoke (`RUN_NATIVE=1`) | console |
| `build_native_wsl.ps1` | WSL CMake build in `native/build-wsl` + optional ctest | console |

```powershell
# Windows — full production gate
powershell -File scripts\cypha_native_validate_all.ps1

# Windows — optional federated TLS smoke (OpenSSL via vcpkg or OPENSSL_ROOT_DIR)
pwsh -File scripts\ci_federated_tls_windows.ps1
```

```bash
# Linux / WSL — CI mirror
bash scripts/ci_native_linux.sh
ctest --test-dir native/build -R native_ --output-on-failure
```

## Release packaging

| Script | Purpose |
|--------|---------|
| `package_release_linux.sh` | Bundle Linux release tarball |
| `package_release_windows.ps1` | Bundle Windows MSVC release zip (`.sh` wraps to PowerShell) |
| `native/scripts/package_windows_qt.ps1` | Windows Qt deployment helper |
| `publish_release.ps1` | Local `gh release create` (`-DryRun` / `-NotesOnly` preview without gh) |

## Bench / tune / diagnostics

| Script / binary | Purpose | Output |
|-----------------|---------|--------|
| **`cypha_bench_run`** (native) | Multi-domain benchmark (d01–d17 + d25 corpus readiness + d26 medium overnight + d27 production overnight + cross-domain) | stdout / JSON |
| **`cypha_bench_report`** (native) | Regenerate bench report figures from JSON artifacts | disk |
| **`cypha_tune_run`** (native) | Sweep JSON → per-cell native bench | disk |
| **`cypha_diagnostics_run`** (native) | Phases 1–4 validation orchestrator (parity exes + inline checks) | stdout |
| `cypha_bench_full_baseline.ps1` | Capture baseline bench profile | `artifacts/profiles/` |
| `cypha_tune_smoke.ps1` | Dry-run tune sweep smoke | console |
| `cyphalm_native_run_modes.ps1` | Cypha sequence mode matrix | console |
| `cyphalm_native_sweep.ps1` / `cyphalm_native_sweep_safe.ps1` | Cypha sequence config sweeps | disk |
| `download_wikitext2.ps1` | Download WikiText-2 raw into `bench/data/wikitext2/wikitext-2/` (PowerShell 5+) | disk |
| `download_wikitext2.sh` | Bash equivalent for Linux/CI | disk |
| `run_unified_context_tournament.ps1` | U01–U10 unified-context BPC tournament vs B2 (40k screen, 300k crown) | `bench/results/unified_context_tournament/` |
| `run_large_context_profile.ps1` | Large-context / large-data profiling: PGM N-scale + BPC tiers (hybrid/H23) + needle-haystack (`-Tier Medium\|Large\|XL`) | `bench/results/large_context_profile/` |
| `run_d17_overnight.ps1` | D17 WikiText 300k overnight (optional `-CellSweep`, `-Fast`, `-Medium`, `-Production`) | disk |
| `run_rpsm_overnight.ps1` | RPSM d21 overnight bench (optional `-Fast`, `-Medium`, `-Production`) | disk |
| `run_overnight_all.ps1` | D17 + d21 + cell sweep + `update_baseline_lock.ps1` merge (passes `-Fast`, `-Medium`, or `-Production` to child scripts) | `bench/BASELINE_LOCK.json` |
| `run_production_overnight.ps1` | Dedicated 300k production overnight wrapper — chains `run_overnight_all.ps1 -Production`, logs to `bench/results/production_overnight_<timestamp>.log`, then `finalize_production_overnight.ps1` + `commit_production_lock.ps1 -DryRun` preview | disk |
| `finalize_production_overnight.ps1` | Post-overnight gate: `validate_baseline_lock.ps1 -Production`, best-effort `update_baseline_lock.ps1 -Run all -Production` when `overnight_results.n_train < 300000` and `cypha_baseline_lock` exists (Phase 23), `cypha_bench_run --domain-tag d27` (+ d28 when present); prints lock section summary (`n_train`, `status`, `bpc`) | console |
| `commit_production_lock.ps1` | Phase 15: chains `finalize_production_overnight.ps1`, shows lock diff + suggested message; `-DryRun` preview, `-Force` to commit (never pushes) | console |
| `monitor_overnight.ps1` | Poll `bench/BASELINE_LOCK.json` sections; optional `-LogFile` (default: latest `bench/results/production_overnight_*.log`, last 3 lines per poll) | console |
| `watch_production_overnight.ps1` | Production overnight watcher — latest log size growth + last line, running `run_production_overnight.ps1` / `cyphalm_bench_native` / `cypha_cell_hypothesis_sweep` PIDs, cell sweep `variant_*.json` progress from whichever of `bench/results/cell_sweep` or repo-root `results/` has more variants (`done/28` + `effective_n_train` while overnight running; latest variant mtime + `manifest.json` `n_train`; highlights in-flight spill path), lock section summary; hints `poll_and_finalize_overnight.ps1` when processes exit; `-Once`, `-ProcessId`, `-ProductionLogFile`; stall warn after 30m without log growth or variant progress (**STALL_WARNING**); optional `-LogFile` append for stall warnings (Phase 24) | console |
| `poll_and_finalize_overnight.ps1` | Phase 17/19/20/21/24: poll until overnight processes exit, then `finalize_production_overnight.ps1` + `commit_production_lock.ps1` (DryRun preview by default; `-Force` to git commit; **`-AutoCommit`** runs `-Force` when lock `n_train >= 300000` after finalize, logs **AUTO_COMMIT** to `-LogFile`); `-Once` exits 1 if still running; optional `-LogFile` append with per-cycle **HEARTBEAT**; **auto-detects `-BuildDir`** from running `run_production_overnight.ps1` when default `native/build`; poll query errors are logged and retried | console |
| `migrate_inflight_overnight_artifacts.ps1` | Phase 23: merge in-flight `build_p13` overnight spill from repo-root `results/` into `bench/results/cell_sweep/`; never touches `bench/BASELINE_LOCK.json`; `-DryRun` preview (default), `-Force` copy | disk |
| `cleanup_repo_smoke_artifacts.ps1` | Phase 20 (shipped): remove repo-root `d##_smoke.json` / `d##_*_smoke.json` CTest spill files (not under `native/build*`); never touches `bench/BASELINE_LOCK.json`; `-DryRun` lists, `-Force` removes | disk |
| `verify_production_pipeline.ps1` | Phase 21 (shipped): unified maintainer smoke - production complete + release publish + repo smoke cleanup preview + optional d35 (`-AllowPending` when lock below 300k); default tag `v2.3.25` or `git describe` | console |
| `run_post_overnight.ps1` | Phase 22 (shipped): post-overnight wrapper - `poll_and_finalize_overnight.ps1 -Once` pre-check then `-Force`, `migrate_inflight_overnight_artifacts.ps1 -DryRun` preview, then `verify_production_pipeline.ps1`; `-SkipPoll`, `-SkipMigrate`, `-AllowPending`; documents `gh auth login` after success | console |
| `start_poll_finalize_background.ps1` | Phase 18/19/21/24: start `poll_and_finalize_overnight.ps1` in background (`-BuildDir`, `-LogFile` default `bench/results/poll_finalize.log`, optional **`-AutoCommit`** passthrough) after manual production overnight start; kills existing `poll_and_finalize_overnight.ps1` PIDs before spawn (dedupe); **same BuildDir auto-detect** when default | console |
| `update_baseline_lock.ps1` | Wrapper for `cypha_baseline_lock` (`-Run d17\|d21\|cell-sweep\|all`; `-Fast` sets `CYPHA_BENCH_FAST=1`; `-Medium` → `--medium`; `-Production` → `--production`) | lock JSON |
| `validate_baseline_lock.ps1` | Validate `bench/BASELINE_LOCK.json` schema + d17 pin (`-LockFile`, `-Strict`, `-Production`) | console |
| `validate_production_complete.ps1` | Phase 18: full production-complete gate — `validate_baseline_lock.ps1 -Production`, `finalize_production_overnight.ps1`, `cypha_bench_run --domain-tag d31` + d30; requires `overnight_results.n_train >= 300000` and `status=production|completed` (`-AllowPending` for smoke) | console |
| `verify_release_publish.ps1` | Phase 19 (shipped): release publish smoke - production complete + d33 bench + `publish_release.ps1 -DryRun` preview (no `gh` call) | console |
| `publish_release.ps1` | Local `gh release create` wrapper (`-DryRun` / `-NotesOnly` preview without gh; `gh auth` preflight) | console |
| `wsl_bench_gpu.sh` | WSL GPU bench helper | console |

## Corpus smoke (Phase 11)

| Binary | Purpose | CTest |
|--------|---------|-------|
| **`corpus_smoke`** (native) | Probe `load_bench_corpus("d17"|"d21", …)` — WikiText-2 or gutenberg fallback | `native_corpus_smoke` |
| **`cypha_bench_run --domain-tag d25`** | Corpus readiness validation; writes `bench/report/tables/d25_corpus_readiness.json` | `native_d25_corpus_smoke` |

## Forecast data (event-forecasting framework)

| Script | Purpose |
|--------|---------|
| **`fetch_forecast_data.ps1`** | Ensure `bench/data/forecast/` sample aliases; `-Bulk` fetches public GDELT + CoW dyadic MID + VIEWS API (UCDP GED fatalities) snapshots; `-Repair` fixes corrupted CSVs |
| **`cypha_bench_run --domain-tag forecast`** | Full `ForecastPipeline` bench on forecast sample/bulk data |

## Medium overnight + baseline lock validator (Phase 12 — shipped everywhere)

| Script / binary | Purpose | CTest |
|-----------------|---------|-------|
| **`-Medium`** on overnight scripts | 5k train / 256 eval, real WikiText or gutenberg (no `CYPHA_BENCH_FAST`) | manual |
| **`cypha_bench_run --domain-tag d26`** | Medium overnight lock validation; profile `bench/config/d26_medium_overnight_profile.json` | `native_d26_medium_overnight_smoke` |
| **`validate_baseline_lock.ps1`** | PS validator for `bench/BASELINE_LOCK.json` (`-Strict` rejects `fast_smoke`-only overnight) | manual |
| **`baseline_lock_validate`** (native) | C++ schema + d17 pin validator | `native_baseline_lock_validate_smoke` |
| **`publish_release.ps1 -DryRun`** | Preview release notes without calling `gh` | manual |

Optional CI job **`corpus_and_d25`** (`continue-on-error`): `bash scripts/download_wikitext2.sh` + `ctest -R "native_corpus_smoke|native_d25_corpus_smoke"`.

## Production overnight tier (Phase 13)

| Script / binary | Purpose | CTest |
|-----------------|---------|-------|
| **`-Production`** on overnight scripts | 300k train / 2000 eval, real WikiText or gutenberg (`status=production`; mutually exclusive with `-Fast`/`-Medium`) | manual |
| **`run_production_overnight.ps1`** | Dedicated production wrapper — `run_overnight_all.ps1 -Production`, logs to `bench/results/production_overnight_<timestamp>.log`, then **`finalize_production_overnight.ps1`** + **`commit_production_lock.ps1 -DryRun`** preview | manual |
| **`finalize_production_overnight.ps1`** | Post-overnight validation: **`validate_baseline_lock.ps1 -Production`**, **`cypha_bench_run --domain-tag d27`** (+ d28 when profile exists); lock section summary | manual |
| **`monitor_overnight.ps1 -LogFile`** | Poll lock JSON + tail production overnight log (auto-picks latest `bench/results/production_overnight_*.log` when `-LogFile` omitted) | manual |
| **`watch_production_overnight.ps1`** | Watch production run — log byte growth, last line, process PIDs (`run_production_overnight.ps1`, `cyphalm_bench_native`, `cypha_cell_hypothesis_sweep`), cell sweep `variant_*.json` count (`done/28` while overnight running; latest variant mtime + `manifest.json` `n_train`; legacy `results/` fallback), lock sections; hints **`poll_and_finalize_overnight.ps1`** when processes disappear; `-Once` snapshot; warns if log stalled 30m+; notes legacy `results/summary.csv` | manual |
| **`cypha_bench_run --domain-tag d27`** | Production overnight lock validation; profile `bench/config/d27_production_lock_profile.json` | `native_d27_production_lock_smoke` |
| **`validate_baseline_lock.ps1 -Production`** | When `overnight_results.n_train >= 300000`, require `status=production` or `completed`, BPC within 0.05 of `d17_hybrid_baseline.bpc` pin | manual |
| **`baseline_lock_validate --production`** (native) | C++ production-tier validator | `native_baseline_lock_validate_smoke` |

Full 300k production overnight is **not** run in CI. Blocking gate **214 `native_` CTests** (dynamic tally via `Get-ExpectedNativeTestCount` / `ctest -N -R native_`). Optional `CYPHA_VALIDATE_PRODUCTION=1` on `cypha_native_validate_all.ps1` runs `validate_baseline_lock.ps1 -Production`. Optional `CYPHA_VALIDATE_OVERNIGHT_COMPLETE=1` runs `cypha_bench_run --domain-tag d28` after baseline lock validate.

## Overnight completion + finalize (Phase 14 — shipped)

| Script / binary | Purpose | CTest |
|-----------------|---------|-------|
| **`cypha_bench_run --domain-tag d28`** | Unified overnight completion validation — cross-check `overnight_results`, `rpsm_results`, `cell_sweep_results` share `n_train`/`n_eval`; profile `bench/config/d28_overnight_complete_profile.json` | `native_d28_overnight_complete_smoke` |
| **`finalize_production_overnight.ps1`** | Post-overnight gate: `validate_baseline_lock.ps1 -Production`, d27 + d28 bench domains, lock section summary | manual (chained from `run_production_overnight.ps1`) |
| Status validator fix | `validate_baseline_lock.ps1` / `baseline_lock_validate` accept `medium_smoke` and `production` | `native_baseline_lock_validate_production_status` |
| Cell sweep artifact path | Default overnight output `bench/results/cell_sweep` via `bench_paths::results_dir()` | manual |
| `migrate_legacy_results.ps1` | Copy repo-root `results/` (`summary.csv`, `variant_*.json`, `manifest.json`) into `bench/results/cell_sweep/`; merge without overwriting newer dest files; `-DryRun` plan only; prints `-RemoveLegacy` / `-ArchiveLegacy` hints when destination has all files; `-RemoveLegacy` deletes legacy `results/`; `-ArchiveLegacy` moves legacy `results/` to `bench/results/legacy_archive_<timestamp>/` | disk |
| `cleanup_legacy_results.ps1` | Thin wrapper: `migrate_legacy_results.ps1` then `-RemoveLegacy` on success; `-DryRun` previews migration only | disk |

## Release readiness + production lock commit (Phase 15)

| Script / env | Purpose | CTest |
|--------------|---------|-------|
| **`CYPHA_VALIDATE_OVERNIGHT_COMPLETE=1`** | On `cypha_native_validate_all.ps1`, run **`cypha_bench_run --domain-tag d28`** after baseline lock validate | manual |
| **`CYPHA_VALIDATE_RELEASE_READINESS=1`** | On `cypha_native_validate_all.ps1`, run **`cypha_bench_run --domain-tag d29`** when profile exists; graceful skip if not merged | manual |
| **`CYPHA_STRICT_TEST_COUNT=1`** | Fail `ctest_native` when parsed `native_` count != expected (dynamic via `Get-ExpectedNativeTestCount`; currently **214**); default warns only | manual | |
| **`commit_production_lock.ps1`** | Run **`finalize_production_overnight.ps1`**, then preview/commit lock when **`overnight_results.n_train >= 300000`** (`-DryRun` preview; **`-Force`** required to commit; never pushes) | manual |
| **`cypha_bench_run --domain-tag d29`** | Release-readiness validation (when shipped); profile `bench/config/d29_release_readiness_profile.json` | `native_d29_release_readiness_smoke` *(when merged)* |

```powershell
# Optional extended validation (after -SkipBuild rebuild)
$env:CYPHA_VALIDATE_PRODUCTION = "1"
$env:CYPHA_VALIDATE_OVERNIGHT_COMPLETE = "1"
$env:CYPHA_VALIDATE_RELEASE_READINESS = "1"
$env:CYPHA_VALIDATE_ARTIFACT_HYGIENE = "1"
pwsh -File scripts\cypha_native_validate_all.ps1 -SkipBuild

# Preview production lock commit (diff + message; no git write)
pwsh -File scripts\commit_production_lock.ps1 -DryRun

# Commit lock after successful finalize (-Force required; does not push)
pwsh -File scripts\commit_production_lock.ps1 -Force
```

Blocking gate **214 `native_` CTests** (dynamic tally; see [Validation gates](#validation-gates-primary)).

## Artifact path hygiene + legacy migration (Phase 16 — shipped)

| Script / env | Purpose | CTest |
|--------------|---------|-------|
| **`migrate_legacy_results.ps1`** | Merge repo-root **`results/`** cell-sweep artifacts into **`bench/results/cell_sweep/`**; never overwrites newer destination files; **`-DryRun`** prints plan only; after migration prints **`-RemoveLegacy`** / **`-ArchiveLegacy`** hints when destination has all files; **`-RemoveLegacy`** deletes legacy **`results/`**; **`-ArchiveLegacy`** moves legacy **`results/`** to **`bench/results/legacy_archive_<timestamp>/`** | manual |
| **`cleanup_legacy_results.ps1`** | One-shot migrate + **`-RemoveLegacy`** wrapper; **`-DryRun`** previews migration only (no removal) | manual |
| **`CYPHA_VALIDATE_ARTIFACT_HYGIENE=1`** | On `cypha_native_validate_all.ps1`, run **`cypha_bench_run --domain-tag d30`** when profile exists | manual |
| **`CYPHA_STRICT_TEST_COUNT=1`** | Fail `ctest_native` when parsed `native_` count != expected (dynamic via `Get-ExpectedNativeTestCount`; currently **214**); default warns only | manual | |
| **`cypha_bench_run --domain-tag d30`** | Artifact path hygiene validation; profile `bench/config/d30_artifact_hygiene_profile.json` | `native_d30_artifact_hygiene_smoke` |

```powershell
# Preview legacy results/ migration (no writes)
pwsh -File scripts\migrate_legacy_results.ps1 -DryRun

# Migrate legacy cell-sweep artifacts (prints -RemoveLegacy / -ArchiveLegacy hints when ready)
pwsh -File scripts\migrate_legacy_results.ps1

# Migrate, then remove repo-root results/
pwsh -File scripts\migrate_legacy_results.ps1 -RemoveLegacy

# Migrate, then archive repo-root results/ to bench/results/legacy_archive_<timestamp>/
pwsh -File scripts\migrate_legacy_results.ps1 -ArchiveLegacy

# One-shot migrate + remove (preview with -DryRun)
pwsh -File scripts\cleanup_legacy_results.ps1 -DryRun
pwsh -File scripts\cleanup_legacy_results.ps1

# Optional extended validation (after -SkipBuild rebuild)
$env:CYPHA_VALIDATE_ARTIFACT_HYGIENE = "1"
pwsh -File scripts\cypha_native_validate_all.ps1 -SkipBuild
```

## Overnight finalize + lock commit automation (Phase 17 — shipped)

| Script / env | Purpose | CTest |
|--------------|---------|-------|
| **`poll_and_finalize_overnight.ps1`** | Poll until `run_production_overnight.ps1` / `cyphalm_bench_native` / `cypha_cell_hypothesis_sweep` processes exit; then **`finalize_production_overnight.ps1`** + **`commit_production_lock.ps1`** (DryRun commit preview by default; **`-Force`** for git add/commit, never pushes) | manual |
| **`cleanup_legacy_results.ps1`** | One-shot **`migrate_legacy_results.ps1`** + **`RemoveLegacy`**; **`-DryRun`** previews migration only | manual |
| **`CYPHA_VALIDATE_POST_OVERNIGHT_PIPELINE=1`** | On `cypha_native_validate_all.ps1`, run **`cypha_bench_run --domain-tag d31`** when profile exists | manual |
| **`cypha_bench_run --domain-tag d31`** | Post-overnight pipeline validation — d27→d30 chain + script presence; profile `bench/config/d31_post_overnight_pipeline_profile.json` | `native_d31_post_overnight_pipeline_smoke` |
| **`watch_production_overnight.ps1`** | Stall-aware watcher — also tracks `cypha_cell_hypothesis_sweep`; prints hint to run **`poll_and_finalize_overnight.ps1`** when processes disappear between polls | manual |
| **`run_production_overnight.ps1`** | On success: **`finalize_production_overnight.ps1`** then **`commit_production_lock.ps1 -DryRun`** (preview only; manual **`commit_production_lock.ps1 -Force`** or **`poll_and_finalize_overnight.ps1 -Force`** to commit) | manual |

Blocking gate **214 `native_` CTests** (dynamic tally; see [Validation gates](#validation-gates-primary)).

```powershell
# Start production overnight (long-running)
pwsh -File scripts\run_production_overnight.ps1

# Watch log growth + processes (hints poll_and_finalize when run finishes)
pwsh -File scripts\watch_production_overnight.ps1

# Poll until processes exit, finalize, commit preview (or -Force to commit)
pwsh -File scripts\poll_and_finalize_overnight.ps1

# Single check — exit 1 if overnight processes still running
pwsh -File scripts\poll_and_finalize_overnight.ps1 -Once

# After validation passes, git-commit lock locally (does not push)
pwsh -File scripts\poll_and_finalize_overnight.ps1 -Force
```

## Production-complete validation (Phase 18 — shipped)

| Script / binary | Purpose | CTest |
|-----------------|---------|-------|
| **`start_poll_finalize_background.ps1`** | Launch **`poll_and_finalize_overnight.ps1`** in the background via `Start-Process` after manually starting production overnight; kills existing poll PIDs first (dedupe); **`-LogFile`** defaults to **`bench/results/poll_finalize.log`** | manual |
| **`poll_and_finalize_overnight.ps1 -LogFile`** | Append session transcript to log (start banner + transcript + exit footer per run); omit **`-LogFile`** for console-only | manual |
| **`validate_production_complete.ps1`** | Maintainer gate: **`validate_baseline_lock.ps1 -Production`**, **`finalize_production_overnight.ps1`**, **`cypha_bench_run --domain-tag d31`** + d30; requires `overnight_results.n_train >= 300000` and `status=production\|completed` (`-AllowPending` for smoke when lock is below production tier) | manual |
| **`cypha_bench_run --domain-tag d32`** | Production-complete lock validation; profile `bench/config/d32_production_complete_profile.json` | `native_d32_production_complete_smoke` |
| **`CYPHA_VALIDATE_PRODUCTION_COMPLETE=1`** | Optional extended validation on `cypha_native_validate_all.ps1` — runs d32 bench domain after baseline lock validate | manual |

```powershell
# Start production overnight (long-running; separate terminal)
pwsh -File scripts\run_production_overnight.ps1

# Detached poll → finalize → commit preview (default log bench/results/poll_finalize.log)
pwsh -File scripts\start_poll_finalize_background.ps1

# Custom build dir + log path
pwsh -File scripts\start_poll_finalize_background.ps1 -BuildDir native/build -LogFile bench/results/poll_finalize.log

# Foreground poll with append log (same -LogFile behavior)
pwsh -File scripts\poll_and_finalize_overnight.ps1 -LogFile bench/results/poll_finalize.log

# Full production-complete gate (fails when lock below 300k unless -AllowPending)
pwsh -File scripts\validate_production_complete.ps1 -BuildDir native\build

# Smoke / CI lock — pass when n_train < 300k (pending_production)
pwsh -File scripts\validate_production_complete.ps1 -AllowPending

# Optional extended validation (after -SkipBuild rebuild)
$env:CYPHA_VALIDATE_PRODUCTION_COMPLETE = "1"
pwsh -File scripts\cypha_native_validate_all.ps1 -SkipBuild
```

Blocking gate **214 `native_` CTests** (dynamic tally; see [Validation gates](#validation-gates-primary)).

## Release publish smoke (Phase 19 — shipped)

| Script / binary | Purpose | CTest |
|-----------------|---------|-------|
| **`verify_release_publish.ps1`** | Maintainer smoke gate - chains **`validate_production_complete.ps1`** (auto **`-AllowPending`** when lock `n_train < 300000`), **`cypha_bench_run --domain-tag d33`** when exe + profile exist, **`publish_release.ps1 -DryRun`** for latest tag (`v2.3.25` default or `git describe --tags --abbrev=0`); does **not** call `gh` - run **`gh auth login`** manually before real publish | manual |
| **`poll_and_finalize_overnight.ps1`** | When **`-BuildDir`** is default **`native/build`** and overnight processes are running, auto-detects BuildDir from **`run_production_overnight.ps1`** command line (e.g. **`native/build_p13`**) | manual |
| **`start_poll_finalize_background.ps1`** | Same BuildDir auto-detect when default before spawning background poll; **kills existing `poll_and_finalize_overnight.ps1` PIDs** before starting a new background poll (dedupe) | manual |
| **`cypha_bench_run --domain-tag d33`** | Release publish validation; profile `bench/config/d33_release_publish_profile.json` | `native_d33_release_publish_smoke` |
| **`CYPHA_VALIDATE_RELEASE_PUBLISH=1`** | Optional extended validation on `cypha_native_validate_all.ps1` — runs d33 after production complete step | manual |

```powershell
# Full release publish smoke (no gh — DryRun notes preview only)
pwsh -File scripts\verify_release_publish.ps1

# Explicit build dir + tag
pwsh -File scripts\verify_release_publish.ps1 -BuildDir native\build -Tag v2.3.25

# Force -AllowPending even when lock is at 300k
pwsh -File scripts\verify_release_publish.ps1 -AllowPending

# After smoke passes — authenticate and publish manually
gh auth login
gh auth status
pwsh -File scripts\publish_release.ps1 -Tag v2.3.25

# Poll with auto-detected BuildDir (when overnight uses native/build_p13)
pwsh -File scripts\poll_and_finalize_overnight.ps1
pwsh -File scripts\start_poll_finalize_background.ps1
```

Blocking gate **214 `native_` CTests** (dynamic tally; see [Validation gates](#validation-gates-primary)).

## Repo smoke hygiene (Phase 20 - shipped)

| Script / binary | Purpose | CTest |
|-----------------|---------|-------|
| **`cleanup_repo_smoke_artifacts.ps1`** | Remove repo-root **`d##_smoke.json`** / **`d##_*_smoke.json`** spill files left by local CTest runs; skips **`native/build*`** and never touches **`bench/BASELINE_LOCK.json`**; **`-DryRun`** lists candidates (default when neither flag passed); **`-Force`** deletes | manual |
| **`cypha_bench_run --domain-tag d34`** | Repo smoke hygiene validation; profile `bench/config/d34_repo_smoke_hygiene_profile.json` | `native_d34_repo_smoke_hygiene_smoke` |
| **`CYPHA_VALIDATE_REPO_SMOKE_HYGIENE=1`** | Optional extended validation on `cypha_native_validate_all.ps1` — runs d34 after release publish step | manual |
| **`poll_and_finalize_overnight.ps1 -LogFile`** | Each poll cycle appends **HEARTBEAT** line: timestamp, overnight process count, lock `overnight_results.n_train`; query failures log **ERROR** and retry (does not treat failed query as "processes exited") | manual |
| **`start_poll_finalize_background.ps1`** | Same poll/finalize behavior when spawned in background; **dedupes** by killing existing `poll_and_finalize_overnight.ps1` PIDs before spawn; re-run manually if the background poll process exits (crash, reboot, or poll errors) while overnight is still running | manual |

```powershell
# Preview repo-root smoke JSON cleanup (default without -Force)
pwsh -File scripts\cleanup_repo_smoke_artifacts.ps1 -DryRun

# Remove repo-root d##_smoke.json artifacts
pwsh -File scripts\cleanup_repo_smoke_artifacts.ps1 -Force

# Foreground poll with heartbeat log (restart manually if poll process dies)
pwsh -File scripts\poll_and_finalize_overnight.ps1 -LogFile bench/results/poll_finalize.log

# Background poll (kills prior poll PIDs; restart manually after crash/reboot if overnight still running)
pwsh -File scripts\start_poll_finalize_background.ps1
```

## Overnight watch + poll dedupe (Phase 21 -- shipped)

| Script / env | Purpose | CTest |
|--------------|---------|-------|
| **`watch_production_overnight.ps1`** | Cell sweep progress from **`variant_*.json`** count under **`bench/results/cell_sweep`** (fallback repo-root **`results/`** when primary empty); shows **`done/28`** while overnight processes run; latest variant mtime + **`manifest.json`** **`n_train`**; still tails **`overnight_progress.log`** when present | manual |
| **`poll_and_finalize_overnight.ps1`** | Comment typo fix; **`Write-PollHeartbeat`** / **`Write-PollError`** use **`$script:resolvedLogFile`** so **`-LogFile`** append is reliable inside functions | manual |
| **`start_poll_finalize_background.ps1`** | Before **`Start-Process`**, kills existing **`poll_and_finalize_overnight.ps1`** PIDs and logs killed PID list (dedupe duplicate background polls) | manual |

```powershell
# Watch overnight with cell sweep variant progress (done/28 while running)
pwsh -File scripts\watch_production_overnight.ps1 -Once

# Background poll dedupes existing poll processes before starting
pwsh -File scripts\start_poll_finalize_background.ps1
```

## Lock commit pipeline (Phase 21 - shipped)

| Script / binary | Purpose | CTest |
|-----------------|---------|-------|
| **`verify_production_pipeline.ps1`** | Unified maintainer smoke gate - chains **`validate_production_complete.ps1`** (auto **`-AllowPending`** when lock `n_train < 300000`), **`verify_release_publish.ps1`** logic, **`cleanup_repo_smoke_artifacts.ps1 -DryRun`**, optional **`cypha_bench_run --domain-tag d35`** | manual |
| **`cypha_bench_run --domain-tag d35`** | Lock commit pipeline validation; profile `bench/config/d35_lock_commit_pipeline_profile.json` | `native_d35_lock_commit_pipeline_smoke` |
| **`CYPHA_VALIDATE_LOCK_COMMIT_PIPELINE=1`** | Optional extended validation on `cypha_native_validate_all.ps1` - runs d35 after repo smoke hygiene step | manual |

```powershell
# Full production pipeline smoke (no gh - DryRun notes preview only)
pwsh -File scripts\verify_production_pipeline.ps1 -AllowPending

# Optional extended validation (after -SkipBuild rebuild)
$env:CYPHA_VALIDATE_LOCK_COMMIT_PIPELINE = "1"
pwsh -File scripts\cypha_native_validate_all.ps1 -SkipBuild
```

Blocking gate **214 `native_` CTests** (dynamic tally; see [Validation gates](#validation-gates-primary)).

## Production pipeline E2E (Phase 22 - shipped)

| Script / binary | Purpose | CTest |
|-----------------|---------|-------|
| **`run_post_overnight.ps1`** | Maintainer post-overnight wrapper - **`poll_and_finalize_overnight.ps1 -Once`** pre-check (exit 1 if processes still running), then **`-Force`** finalize+commit; then **`verify_production_pipeline.ps1`**; **`-SkipPoll`** when finalize/commit already done; **`-AllowPending`** for smoke verify; documents **`gh auth login`** after success | manual |
| **`cypha_bench_run --domain-tag d36`** | Production pipeline E2E validation; profile `bench/config/d36_pipeline_e2e_profile.json` | `native_d36_pipeline_e2e_smoke` |
| **`watch_production_overnight.ps1`** | Cell sweep progress shows **`effective_n_train`** in progress line - when overnight is running and **`manifest.json`** **`n_train < 300000`**, reads **`n_train`** from latest **`variant_*.json`** (manifest may lag during production sweep) | manual |
| **`verify_production_pipeline.ps1`** | Default release tag **`v2.3.25`** (or **`git describe --tags --abbrev=0`** when available) | manual |
| **`CYPHA_VALIDATE_PIPELINE_E2E=1`** | Optional extended validation on `cypha_native_validate_all.ps1` - runs d36 after lock commit pipeline step | manual |

```powershell
# After production overnight finishes (or -SkipPoll if already finalized/committed)
pwsh -File scripts\run_post_overnight.ps1

# Skip poll when finalize+commit already done; smoke verify only
pwsh -File scripts\run_post_overnight.ps1 -SkipPoll -AllowPending

# Watch with effective_n_train during overnight
pwsh -File scripts\watch_production_overnight.ps1 -Once

# Optional extended validation (after -SkipBuild rebuild)
$env:CYPHA_VALIDATE_PIPELINE_E2E = "1"
pwsh -File scripts\cypha_native_validate_all.ps1 -SkipBuild
```

Blocking gate **214 `native_` CTests** (dynamic tally; see [Validation gates](#validation-gates-primary)).

## In-flight overnight artifact migration (Phase 23 - shipped)

| Script | Purpose | CTest |
|--------|---------|-------|
| **`migrate_inflight_overnight_artifacts.ps1`** | Merge repo-root **`results/`** cell-sweep spill (e.g. in-flight **`build_p13`** overnight) into **`bench/results/cell_sweep/`**; merge without overwriting newer dest files; never touches **`bench/BASELINE_LOCK.json`**; **`-DryRun`** preview (default), **`-Force`** copy | manual |
| **`cypha_bench_run --domain-tag d37`** | Overnight lock refresh validation; profile `bench/config/d37_lock_refresh_profile.json` | `native_d37_lock_refresh_smoke` |
| **`CYPHA_VALIDATE_LOCK_REFRESH=1`** | Optional extended validation on `cypha_native_validate_all.ps1` - runs d37 after pipeline E2E (d36) step | manual |
| **`publish_release.ps1 -NotesPath`** | Offline release notes file for **`gh release create`** when not piping from **`-DryRun`** | manual |
| **`finalize_production_overnight.ps1`** | After **`validate_baseline_lock.ps1 -Production`**, when **`overnight_results.n_train < 300000`** and **`cypha_baseline_lock`** exists under **`-BuildDir`**, best-effort **`update_baseline_lock.ps1 -Run all -Production`** (warn on fail, continue to d27/d28) | manual |
| **`run_post_overnight.ps1`** | Chains poll/finalize, then **`migrate_inflight_overnight_artifacts.ps1 -DryRun`** preview (unless **`-SkipMigrate`**), then **`verify_production_pipeline.ps1`** | manual |
| **`watch_production_overnight.ps1`** | Cell sweep progress counts **`variant_*.json`** in both **`bench/results/cell_sweep`** and repo-root **`results/`**; uses whichever directory has more variants as the progress source | manual |

```powershell
# Preview in-flight spill merge (default without -Force)
pwsh -File scripts\migrate_inflight_overnight_artifacts.ps1

# Apply merge after preview
pwsh -File scripts\migrate_inflight_overnight_artifacts.ps1 -Force

# Post-overnight pipeline includes spill preview unless skipped
pwsh -File scripts\run_post_overnight.ps1
pwsh -File scripts\run_post_overnight.ps1 -SkipMigrate

# Watch progress during build_p13 overnight (uses dir with more variants)
pwsh -File scripts\watch_production_overnight.ps1 -Once

# Optional extended validation (after -SkipBuild rebuild)
$env:CYPHA_VALIDATE_LOCK_REFRESH = "1"
pwsh -File scripts\cypha_native_validate_all.ps1 -SkipBuild
```

Blocking gate **214 `native_` CTests** (dynamic tally; see [Validation gates](#validation-gates-primary)).

## Auto-commit + variant stall detector (Phase 24 - prep)

| Script | Purpose | CTest |
|--------|---------|-------|
| **`poll_and_finalize_overnight.ps1 -AutoCommit`** | After successful finalize, logs **AUTO_COMMIT** to **`-LogFile`**, then runs **`commit_production_lock.ps1 -Force`** when lock **`overnight_results.n_train >= 300000`**; otherwise **`-DryRun`** preview. Default (no switch) remains DryRun only | manual |
| **`start_poll_finalize_background.ps1 -AutoCommit`** | Passthrough **`-AutoCommit`** to background **`poll_and_finalize_overnight.ps1`** | manual |
| **`watch_production_overnight.ps1`** | Tracks cell-sweep **`variant_*.json`** count across polls; while overnight processes run, emits **STALL_WARNING** to console (and optional **`-LogFile`** append) when variant count unchanged for **`-StallMinutes`** (default 30). Production overnight log override: **`-ProductionLogFile`** | manual |
| **`cypha_bench_run --domain-tag d38`** | Production overnight completion certificate; profile `bench/config/d38_overnight_certificate_profile.json` | `native_d38_overnight_certificate_smoke` *(when merged)* |
| **`CYPHA_VALIDATE_OVERNIGHT_CERTIFICATE=1`** | Optional extended validation on `cypha_native_validate_all.ps1` - runs d38 after lock refresh (d37) step | manual |

```powershell
# Background poll with auto-commit when production threshold met
pwsh -File scripts\start_poll_finalize_background.ps1 -AutoCommit

# Foreground poll with auto-commit + heartbeat log
pwsh -File scripts\poll_and_finalize_overnight.ps1 -AutoCommit -LogFile bench/results/poll_finalize.log

# Watch with variant stall warnings appended to log
pwsh -File scripts\watch_production_overnight.ps1 -LogFile bench/results/watch_stall.log

# Optional extended validation (after -SkipBuild rebuild)
$env:CYPHA_VALIDATE_OVERNIGHT_CERTIFICATE = "1"
pwsh -File scripts\cypha_native_validate_all.ps1 -SkipBuild
```

Blocking gate **214 `native_` CTests** (dynamic tally via `Get-ExpectedNativeTestCount` / `ctest -N -R native_`).

## Kernel LLR / XOR profiling

| Binary | Purpose | Output |
|--------|---------|--------|
| **`xor_kernel_bench`** (native) | XOR linear vs Nyström kernel via `dif_train_step_vector` + kernel update | stdout JSON; CTest `native_xor_kernel_bench_smoke` |
| **`kernel_llr_parity`** (native) | Kernel LLR vs `fixtures/kernel_llr/` | CTest `native_kernel_llr` |

## REST / Qt helpers

| Script | Purpose |
|--------|---------|
| `run_cypha_qt_windows.ps1` | Launch `cypha_rest` + Qt shell on Windows |
| `native/scripts/build_cypha_rest_mingw_wsl.ps1` | MinGW cross-build `cypha_rest.exe` in WSL |
| `native/scripts/smoke_cypha_rest_mingw.ps1` | Subprocess REST smoke vs parity fixtures |
| `loadtest_ab_predict_example.sh` / `.ps1` | Example `ab` against live `/predict` |

## Parity fixtures

**`fixtures/`** are **frozen committed goldens**. Do not regenerate from removed Python generators. After intentional contract changes, update sidecars manually or extend native parity tools under `native/tools/`, then run **`ctest -R native_<name>`**. See [`fixtures/README.md`](../fixtures/README.md) and [`docs/verify/MAINTENANCE.md`](../docs/verify/MAINTENANCE.md).

| Native parity binary | Fixture dir | CTest |
|---------------------|-------------|-------|
| `cypha_parity` | root (`reference.cypha`, `native_parity.bin`) | `native_parity` |
| `batch_llr_parity` | `batch_llr/` | `native_batch_llr` |
| `quantile_dif_train_parity` | `quantile_dif_train/` | `native_quantile_dif_train` |
| `memory_train_parity` | `memory_train/` | `native_memory_train` |
| `preprocessor_parity` | `preprocessor/` | `native_preprocessor` |
| `preprocessor_fit_parity` | `preprocessor_fit/`, `preprocessor_fit_no_scale/` | `native_preprocessor_fit` |
| `regression_m4_parity` | `regression_m4/` | `native_regression_m4` |
| *(see `native/README.md`)* | *(24+ dirs)* | `ctest -R native_` |

Override binary paths with **`CYPHA_*_BIN`** env vars (see [`native/README.md`](../native/README.md)).

> **Note on `.sh` scripts:** `wsl_verify.sh`, `ci_native_linux.sh`, and release packagers use **`eol=lf`** in **`.gitattributes`**. On Unix, **`chmod +x scripts/*.sh`** if needed.
