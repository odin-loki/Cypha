# CyphaLM Native — hp context mixer (Tier 2)

**Status:** production LLM path (2026-09-18)  
**Library target:** `cypha_lm_native` (INTERFACE → `cypha_core`)  
**Algorithm source:** [odin-loki/CompressionAlgorithm](https://github.com/odin-loki/CompressionAlgorithm) `hp/` tree (vendored under `native/third_party/hp/`)

## Overview

Cypha's sequence / LLM algorithm is the **hp** integer-exact Hutter Prize context-mixing compressor. The previous Hybrid GRIA+LSTM stack is **not** the production path; its sources remain in the tree for reference but are excluded from the default `cypha_core` build.

```
token (byte 0..255) ─► HpSequenceBackend ─► hp::Predictor
                              │
                              ├─ per-bit: experts → mixer → APM → log_prob
                              └─ next-byte log_probs ─► CyphaLMModel::predict_next
```

Public API (`CyphaLMModel`, `Cypha::init_default_sequence`, `predict_next`, `generate`, BPC eval) is unchanged; the implementation delegates to `hp::Predictor` via `HpSequenceBackend`.

## Context mode

| Enum | Alias | Meaning |
|------|-------|---------|
| `Hp` | `hp`, `hybrid`, `hybrid_gria_lstm` | **Production** — hp RAM-speed (`HP_SLOT_MAX=24`) |
| `HpChamp` | `hp_champ`, `champ` | **Research** — full slot cap (`HP_SLOT_MAX=35`; requires champ build) |
| Others | `char_lstm`, `ssm_gria`, … | Legacy research enums; map to hp or no-op stubs |

Configure with `apply_hp_production_recipe()` (default) or `apply_hp_champ_recipe()` (research).

## hp knobs (`CyphaLMConfig`)

| Field | Default | hp flag |
|-------|---------|---------|
| `hp_table_bits` | 22 | `--mem` (table size; 22 ≈ 4 MiB) |
| `hp_slot_max` | 24 | Requested slot cap; compile-time `HP_SLOT_MAX` is authoritative |
| `hp_mixer_lr` | 2 | mixer learning rate |
| `hp_gria` | true | GRIA alpha gating |
| `vocab_size` | 256 | byte tokens (must be ≤ 256) |

### CMake / memory profiles

| Build | CMake | `HP_SLOT_MAX` | Lab RSS @ mem 22 (harness) |
|-------|-------|---------------|----------------------------|
| **Production (default)** | (none) | 24 | ~1.6 GB |
| **Champ / research** | `-DCYPHA_HP_CHAMP_BUILD=ON` | 35 | ~15 GB |

Lab numbers from `native/third_party/hp/tools/hp_harness.sh` (RECORD H34: Pearson +0.96 vs SLOT_MAX=35, +895 B on 8 MB gate).

CMake: `-DCYPHA_HP_XSIMD=ON` enables hp xsimd mixer dots (SSE4.1); default OFF for portability.

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
ctest -R 'native_hp|native_cyphalm_model_golden' --output-on-failure
```

## Checkpoints

`save_cyphalm_model` / `load_cyphalm_model` persist **config** (algorithm=`hp`). hp predictor tables are session-local (online adaptation); re-run `train_sequence` on a corpus to warm tables after load.

## Third party

See `native/third_party/hp/THIRD_PARTY.md` for provenance and xsimd license.
