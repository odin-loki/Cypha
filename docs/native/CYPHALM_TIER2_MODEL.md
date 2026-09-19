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

### RAM hotspot (documented, not optimized)

`HpSequenceBackend` keeps two `hp::Predictor` heap instances (`pred_` for live state, `scratch_` for lookahead). At gate24 table sizes, full-vocab scoring uses legacy 256-clone (bit-tree checkpoint pool OOMs at v78 scale).

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

### Compile profile (gate24 only)

| Build | CMake | `HP_SLOT_MAX` | v78 flags | enwik8.8mb BPC (measured) |
|-------|-------|---------------|-----------|---------------------------|
| **gate24 (default)** | (none required) | 24 | ON (`v78_flags.ps1`) | observe **1.611729**, archive **1.611759** |

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

`save_cyphalm_model` / `load_cyphalm_model` persist **config** (algorithm=`hp`). hp predictor tables are session-local (online adaptation); re-run `train_sequence` on a corpus to warm tables after load.

## Third party

See `native/third_party/hp/THIRD_PARTY.md` for provenance and xsimd license.
