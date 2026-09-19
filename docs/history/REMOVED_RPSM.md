# RPSM removal (2026-09)

**Status:** retired — not part of the product build  
**Superseded by:** [hp CyphaLM](LEGACY_LLM.md#what-replaced-it) (`HpSequenceBackend` / `apply_hp_production_recipe`)  
**Git archaeology:** full source, tests, and CMake targets remain on `main` and in PR #1 history before the removal commit on branch `cursor/hp-llm-integration-6ad7`. Use `git log --all -- native/src/rpsm/` or browse pre-removal commits to recover files.

## What RPSM was

**RPSM** (Recursive Predictive State Model) was a CyphaLM sequence layer built on:

- **Option A — batched LLR GEMM:** `PsiMatrices` + `rpsm_score_matrix_batched` in the DIF inference path (`infer_cpu.cpp`), toggled by `CYPHA_USE_RPSM_LLR` (default ON when RPSM shipped).
- **Option B — sequence layer:** `RpsmSequenceLayer` with hierarchical `W_up`/`W_down`, profile-guided loss, and `--mode rpsm` on `cyphalm_bench_native`.

It was researched as a successor to Hybrid GRIA+LSTM for WikiText BPC (bench **d21**, overnight lock section `rpsm_results` in `bench/BASELINE_LOCK.json`). The track was marked **STOP** in July 2026 (`docs/research/upgrades/README.md`, `docs/archive/plans/RPSM_UPGRADE_PLAN.md`) but code remained until hp became the production LLM path.

## Why it was removed

- User/product decision: RPSM is **unused**; hp is the living CyphaLM algorithm (light default + champ opt-in).
- Keeping RPSM in the default build inflated CI time (`native_cyphalm_bench_rpsm_smoke` ~45 min, `native_d21_rpsm_smoke`) and contradicted docs claiming hp as production.
- `CYPHA_USE_RPSM_LLR` bypassed `cypha::accel::score_matrix`, making CUDA audit docs stale.

## Where it lived (removed paths)

| Area | Paths |
|------|-------|
| Core sources | `native/src/rpsm/psi_matrices.cpp`, `rpsm_sequence_layer.cpp` |
| Headers | `native/include/cypha/rpsm/psi_matrices.hpp`, `rpsm_sequence_layer.hpp` |
| Inference hook | `use_rpsm_llr_from_env()`, `rpsm_score_matrix_batched` in `native/src/infer_cpu.cpp` |
| Config | `ContextMode::Rpsm`, `BenchMode::Rpsm`, `rpsm_*` fields in `cyphalm_config.hpp` / `.cpp` |
| Bench profiles | `bench/config/profiles/cyphalm_d21_rpsm.json`, `cyphalm_d21_rpsm_small.json`, `bench/config/d21_rpsm_profile.json` |
| Overnight script | `scripts/run_rpsm_overnight.ps1` |
| CTests (examples) | `native_rpsm_sequence_smoke`, `native_rpsm_batched_llr_smoke`, `native_cyphalm_bench_rpsm_smoke`, `native_d21_rpsm_smoke`, `native_score_matrix_parallel_parity`, `native_rpsm_*` regression suite |
| CMake | RPSM sources on `cypha_core`, RPSM compile defs, targets in `native/CMakeLists.txt`, `CyphaRegression.cmake`, `LegacyCyphalm.cmake`, `CyphaCTest.cmake` |

## What replaced it

| Former role | Replacement |
|-------------|-------------|
| d21 overnight / lock `rpsm_results` | hp profile `bench/config/profiles/cyphalm_d21_hp.json`; `cypha_baseline_lock --run d21` writes `mode=hp` (lock key `rpsm_results` kept for schema compat) |
| `--mode rpsm` CLI | Retired; `rpsm` aliases map to `Hp` / `Hybrid` (hp) in config parser |
| d21 bench domain | `run_d21_hp_overnight_smoke` in `bench_domains.cpp` |
| LLR scoring default | `score_matrix_use_field` → `cypha::accel::score_matrix` (no RPSM early-return) |

## Lock file compatibility

`bench/BASELINE_LOCK.json` still uses the section name **`rpsm_results`** for historical locks. Validators accept `mode: hp` (current) or legacy `mode: rpsm`. Do not interpret the key name as “RPSM is still live.”

## Related archive material (unchanged)

Research specs and closeout reports were already archived before code removal:

- `docs/research/upgrades/RPSM_COMBINED_SPEC.md`, `RPSM_IMPLEMENTATION.md`
- `docs/archive/plans/RPSM_UPGRADE_PLAN.md`
- `docs/archive/reports/RPSM_SMALL_TIER_GATE_2026-07-18.md`, `PARALLEL_SCORE_2026-07-17.md`

See also [`LEGACY_LLM.md`](LEGACY_LLM.md) for the Hybrid GRIA+LSTM stack gated behind `CYPHA_BUILD_LEGACY_CYPHALM`.
