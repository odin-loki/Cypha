# Finalize prep — post-H22 production overnight (2026-07-17)

**Author:** Odin Loch (agent docs)  
**Scope:** Document the exact post-sweep command and prerequisites. **Do not run finalize while H19–H22 are in flight.**  
**Repo HEAD at prep:** `0f502e7`  
**Overnight state @ prep:** H18 @ 21/25 healthy — see [`OVERNIGHT_HEALTH_2026-07-17.md`](OVERNIGHT_HEALTH_2026-07-17.md) §8.

---

## Script confirmed

`scripts/poll_and_finalize_overnight.ps1` **exists** in the repo. It:

1. Polls every 60 s until no overnight-related processes remain (`cyphalm_bench_native`, `cypha_cell_hypothesis_sweep`, or a `run_production_overnight.ps1` shell).
2. Runs `scripts/finalize_production_overnight.ps1` (lock validation + d27/d28/d42/d53+ production gates).
3. Runs `scripts/commit_production_lock.ps1` (DryRun preview by default; **`-Force`** only when invoked via `-AutoCommit` and `overnight_results.n_train >= 300000`).

Related helpers: `scripts/start_poll_finalize_background.ps1` (detached poll watcher), `scripts/watch_production_overnight.ps1` (health monitor; hints this script when processes disappear).

---

## Exact post-H22 command

Run from **repo root** after H22 child exits, artifact flush completes, and poll heartbeats show `processes=0`:

```powershell
pwsh -File scripts/poll_and_finalize_overnight.ps1 `
  -BuildDir native/build_math `
  -AutoCommit `
  -LogFile bench/results/poll_finalize.log
```

**Why `-BuildDir native/build_math`:** The live production cell-sweep runs from `native/build_math` (math-integration overnight). After processes exit, BuildDir auto-detect from a running `run_production_overnight.ps1` no longer applies — pass the build dir explicitly.

**What `-AutoCommit` does:** After successful finalize, runs `commit_production_lock.ps1 -Force` when `bench/BASELINE_LOCK.json` → `overnight_results.n_train >= 300000`; otherwise DryRun preview only. **Never pushes.**

### One-shot status check (no finalize)

```powershell
pwsh -File scripts/poll_and_finalize_overnight.ps1 -Once
```

Exit **0** = no overnight processes detected; exit **1** = still running or poll query failed.

### Preview only (no git commit)

```powershell
pwsh -File scripts/poll_and_finalize_overnight.ps1 -BuildDir native/build_math
```

Omit `-AutoCommit` / `-Force` → finalize runs, commit step is **DryRun**.

### Force commit (manual override)

```powershell
pwsh -File scripts/poll_and_finalize_overnight.ps1 -BuildDir native/build_math -Force
```

Use only when validation passed and `n_train >= 300000` but AutoCommit threshold logic is insufficient.

---

## Prerequisites checklist

| # | Prerequisite | How to verify |
|---|--------------|---------------|
| 1 | **H22 complete** — last progress line `variant=H22 25/25` in `bench/results/cell_sweep/overnight_progress.log` | Read log tail |
| 2 | **Artifact flush** — `write_overnight_artifacts` wrote `bench/results/cell_sweep/variant_H*.json` (incl. H15–H22) | Check mtimes under `bench/results/cell_sweep/` |
| 3 | **No overnight processes** — poll heartbeats `processes=0` | `poll_finalize.log` or `-Once` |
| 4 | **Binaries built** — `native/build_math/cypha_bench_run` (and `cypha_baseline_lock` for lock refresh) exist | `Test-Path native/build_math/cypha_bench_run.exe` |
| 5 | **Lock present** — `bench/BASELINE_LOCK.json` readable; `overnight_results.n_train=300000` at production tier | Do **not** hand-edit; finalize refreshes |
| 6 | **Git ready** — working tree allows committing lock diff (no conflicting staged lock edits) | `git status bench/BASELINE_LOCK.json` |
| 7 | **Do not kill overnight** — let H19–H22 finish naturally | See overnight health doc |

---

## Post-finalize follow-ups (not part of poll script)

After lock commit lands:

1. **`scripts/aggregate_cell_sweep_summary.ps1`** — populate `bench/results/cell_sweep/summary.csv` from variant JSON ([`CELL_SWEEP_SUMMARY_TOOL_2026-07-17.md`](CELL_SWEEP_SUMMARY_TOOL_2026-07-17.md)).
2. **d38 merge** — production overnight completion certificate (115→116 CTests); gates were `pending_production` pre-lock.
3. **Manual push** — poll/commit scripts never push; push lock commit when ready.

---

## OFF-LIMITS (this session)

- Do **not** run finalize for real while sweep is in flight.
- Do **not** kill overnight processes.
- Do **not** hand-edit `bench/BASELINE_LOCK.json`.

---

## Cross-links

- Wave-2 landings since closeout: [`PRODUCT_ADJUST_WAVE2_2026-07-17.md`](PRODUCT_ADJUST_WAVE2_2026-07-17.md)
- Closeout summary: [`PRODUCT_ADJUST_CLOSEOUT_2026-07-17.md`](PRODUCT_ADJUST_CLOSEOUT_2026-07-17.md)
- Overnight health: [`OVERNIGHT_HEALTH_2026-07-17.md`](OVERNIGHT_HEALTH_2026-07-17.md)
- Dev plan finalize chain: [`DEV_PLAN_2026-07-11.md`](DEV_PLAN_2026-07-11.md) § overnight recovery
