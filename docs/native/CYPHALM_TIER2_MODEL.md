# CyphaLM Native — hp context mixer (Tier 2)

**Status:** production LLM path (2026-09-18)  
**Library target:** `cypha_lm_native` (INTERFACE → `cypha_core`)  
**Algorithm source:** [odin-loki/CompressionAlgorithm](https://github.com/odin-loki/CompressionAlgorithm) `hp/` tree (vendored under `native/third_party/hp/`)

## Overview

Cypha's sequence / LLM algorithm is the **hp** integer-exact Hutter Prize context-mixing compressor. The previous Hybrid GRIA+LSTM stack is **not** the production path; its sources remain in the tree for reference but are excluded from the default `cypha_core` build. **RPSM** was fully removed in 2026-09.

**Superseded / removed (history):** [`docs/history/LEGACY_LLM.md`](../../history/LEGACY_LLM.md) · [`docs/history/REMOVED_RPSM.md`](../../history/REMOVED_RPSM.md) · [`docs/history/REMOVED_HP_SKUS.md`](../../history/REMOVED_HP_SKUS.md)

```
token (byte 0..255) ─► HpSequenceBackend ─► hp::Predictor
                              │
                              ├─ per-bit: experts → mixer → APM → log_prob
                              └─ next-byte log_probs ─► CyphaLMModel::predict_next
```

Public API (`CyphaLMModel`, `Cypha::init_default_sequence`, `predict_next`, `generate`, BPC eval) is unchanged; the implementation delegates to `hp::Predictor` via `HpSequenceBackend`.

### RAM

Superseded (2026-09-23): `HpSequenceBackend` holds one predictor, full-vocab
scoring walks the bit tree with undo on it (legacy 256-clone only with
`CYPHA_HP_LEGACY_BYTE_LOGPROBS=1`), tables are demand-zero, context slots are
packed to 16 bits and checkpoint tables are mapped from the `.hpbin`. Served
footprints and the RAM / quality frontier:
[`CYPHALM_LM_QUALITY_REPORT.md`, RAM](../reports/CYPHALM_LM_QUALITY_REPORT.md#ram).

## Context mode

| Enum | Alias | Meaning |
|------|-------|---------|
| `Hp` | `hp`, `hybrid`, `gate24`, `champ`, … | **Production** — gate24 (v78 + `HP_SLOT_MAX=24`) |
| Others | `char_lstm`, `ssm_gria`, … | Legacy research enums; map to hp or no-op stubs |

Configure with `apply_hp_production_recipe()` (the only hp recipe).

## hp knobs (`CyphaLMConfig`)

| Field | Default | hp flag |
|-------|---------|---------|
| `hp_table_bits` | 22 | `--mem` (table size; 22 ≈ 4 MiB) |
| `hp_slot_max` | 24 | Requested slot cap; compile-time `HP_SLOT_MAX=24` is authoritative |
| `hp_mixer_lr` | 2 | mixer learning rate |
| `hp_gria` | true | GRIA alpha gating |
| `vocab_size` | 256 | byte tokens (must be ≤ 256) |

Lossy and serve-time fields (`hp_lossy_tier`, `hp_cm_drop`, the table caps,
`hp_match_drop`, `hp_frozen_scoring`, `hp_tree_prune`,
`hp_serve_mixer_lr_scale`, `hp_ensemble_learning_rate`) and their env vars:
[`CYPHALM_LM_QUALITY_REPORT.md`, Reference](../reports/CYPHALM_LM_QUALITY_REPORT.md#reference).

### Compile profile (gate24 only)

| Build | CMake | `HP_SLOT_MAX` | v78 flags | enwik8.8mb BPC (measured) |
|-------|-------|---------------|-----------|---------------------------|
| **gate24 (default)** | (none required) | 24 | baked into the vendored tree ([strip](../reports/CYPHALM_HP_GATE24_STRIP.md)) | observe **1.611729**, archive **1.611759** |

Removed by choice: **light** (~1.72 BPC, 0/78 flags) and **champ** (`SLOT_MAX=35`, ~15 GB RSS). See [`REMOVED_HP_SKUS.md`](../../history/REMOVED_HP_SKUS.md).

CMake: `-DCYPHA_HP_XSIMD=OFF` disables hp xsimd mixer dots on hosts without SSE4.1; default ON.

## Metrics — do not mix

| Metric | What it measures |
|--------|------------------|
| **Cypha BPC** | `-log2 P(next_byte)` from `eval_bpc` / `predict_next` on a token stream |
| **hp archive size** | Compressed bytes from `hp c` CLI on a raw file — **not comparable** to Cypha BPC without explicit labeling |

## Tests

```bash
cd native/build-wsl-gcc
./hp_llm_smoke
./hp_roundtrip_smoke
./cyphalm_model_golden --mode hp
# PR CI fast gate (Linux): scripts/ci_native_fast.sh  (-LE cypha_slow)
# macOS CI: scripts/ci_native_hp_smoke.sh
ctest -R 'native_hp|native_cyphalm_model_golden' --output-on-failure
```

### CI platforms (CyphaLM gate24)

| Platform | Job | Gate |
|----------|-----|------|
| Linux | `Build and test (Linux)` | `scripts/ci_native_hp_smoke.sh` |
| Windows | `Build (Windows MSVC)` | compile + artifact check |
| macOS | `Build and test (macOS)` | `scripts/ci_native_hp_smoke.sh` |

Optional: `Native slow CTest (optional)` runs `scripts/ci_native_fast.sh` (`-LE cypha_slow`). Bench/lock/forecast/d21–d76 smokes are labeled `cypha_slow` and are not in the PR gate.

## Checkpoints

`save_cyphalm_model(model, BASE)` writes `BASE.json` (algorithm `hp`, the
full `CyphaLMConfig`) and `BASE.hpbin` (all predictor state, `HPCP` format
v3). `load_cyphalm_model` restores the trained model exactly (v1/v2 files
convert on load) and also accepts an ensemble manifest. Formats, fields and
env vars: [`CYPHALM_LM_QUALITY_REPORT.md`, Reference](../reports/CYPHALM_LM_QUALITY_REPORT.md#reference).

## Third party

See `native/third_party/hp/THIRD_PARTY.md` for provenance and xsimd license.
