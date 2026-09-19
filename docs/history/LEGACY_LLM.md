# Legacy CyphaLM stacks (Hybrid GRIA+LSTM and friends)

**Status:** gated off by default — not the product LLM path  
**Living algorithm:** **hp** — see [`docs/native/CYPHALM_TIER2_MODEL.md`](../native/CYPHALM_TIER2_MODEL.md) and [`MODEL_CARD.md`](../../MODEL_CARD.md)  
**Git archaeology:** Hybrid/LSTM/SSM sources remain in the tree on `main`; default `cypha_core` no longer compiles them. Enable `-DCYPHA_BUILD_LEGACY_CYPHALM=ON` to rebuild legacy tools/CTests. Pre-hp default-build behavior is recoverable from git history on branch `cursor/hp-llm-integration-6ad7` and earlier `main` commits.

## Lineage (short)

| Era | Algorithm | Typical BPC pin | Notes |
|-----|-----------|-----------------|-------|
| 2026-06 | Hybrid GRIA+LSTM (L1) | **2.873** @ 300k WikiText | First hybrid production default |
| 2026-08 | Hybrid L2 + Wave2 BPTT | **2.664** @ 300k | `apply_hybrid_production_recipe`; lock in `bench/BASELINE_LOCK.json` |
| 2026-07–08 | RPSM (d21) | research only | See [`REMOVED_RPSM.md`](REMOVED_RPSM.md) — code removed 2026-09 |
| 2026-09+ | **hp** (CyphaLM) | not locked in BASELINE_LOCK yet | Light `HP_SLOT_MAX=24` (~1.6 GB); champ opt-in `CYPHA_HP_PROFILE=champ` |

Historical BPC numbers in the lock file and paper draft are **not** hp metrics. Do not invent new BPC locks for hp without a measured overnight run.

## What was gated (`CYPHA_BUILD_LEGACY_CYPHALM=OFF`, default)

CMake option defined in `native/cmake/LegacyCyphalm.cmake`. When OFF (default):

- **Excluded from `cypha_core`:** GRIA, char-LSTM, SSM, compressive memory, PGM cell paths used only by hybrid training, and related train-step code paths.
- **Skipped executables / CTests:** `stacked_lstm_smoke`, `cyphalm_ssm_golden`, `cyphalm_char_lstm_golden`, `ewc_hybrid_smoke`, `navigation_loss_hybrid_smoke`, `pgm_cell_smoke`, `native_ewc_weights_smoke`, and other targets listed in `CYPHA_LEGACY_CYPHALM_EXE_TARGETS`.
- **Production entry:** `Cypha::init_default_sequence` → `apply_hp_production_recipe()` → `HpSequenceBackend`.

When ON (maintainer / archaeology):

```bash
cmake -S native -B native/build-legacy -DCYPHA_BUILD_LEGACY_CYPHALM=ON
cmake --build native/build-legacy
```

## CLI / config aliases (living tree)

`hybrid`, `hybrid_gria_lstm`, and retired `rpsm` profile strings map to **hp** in `cyphalm_config.cpp`. They exist so old scripts and JSON profiles fail soft, not to claim Hybrid LSTM is still default.

## Where sources still live

Headers and legacy implementation files remain under `native/include/cypha/cyphalm/` and `native/src/cyphalm/` (e.g. GRIA, LSTM, SSM modules). They are reference / optional-build only.

## Removed vs gated

| Subsystem | Disposition |
|-----------|-------------|
| **RPSM** | **Removed** from tree — see [`REMOVED_RPSM.md`](REMOVED_RPSM.md) |
| **Hybrid GRIA+LSTM / SSM / CharLSTM** | **Gated** behind `CYPHA_BUILD_LEGACY_CYPHALM` |
| **Cell hypothesis sweep** | Maintainer overnight tooling; lock `cell_sweep_results.status=historical` |
| **PGM → Wy (U06)** | Opt-in research; not hp default |

## Docs and pins

- Living hp docs: `docs/native/CYPHALM_TIER2_MODEL.md`, `MODEL_CARD.md`, root `README.md` § Sequence / LLM.
- Historical hybrid pins: `bench/BASELINE_LOCK.json` (`d17_hybrid_baseline`, `overnight_results`) — comparison only.
- Archive reports: `docs/archive/reports/BASELINE_PIN_CANONICAL_2026-07-17.md`, cell-sweep summaries under `docs/archive/reports/`.
