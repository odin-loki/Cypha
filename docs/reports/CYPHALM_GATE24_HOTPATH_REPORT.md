# CyphaLM gate24 hot-path profile + micro-opts

**Date:** 2026-09-20  
**Branch:** `cursor/gate24-hotpath-checkpoint-ci-6c23`  
**Harness:** `native/tools/cyphalm_hp_hotpath_profile.cpp`, `native/tests/regression/hp_gate24_ci_gate.cpp`

---

## Executive summary

Profiled the gate24 `hp::Predictor` adapt/serve paths on top of **delta-undo bit-tree inference** ([#7](https://github.com/odin-loki/Cypha/pull/7)). Applied **safe micro-optimizations** in `HpSequenceBackend` (reused `log_probs_buf_`). Added **binary HPCP v1 checkpoint** (`.hpbin`) for train-once / serve-many, plus **CI gates** for compress-faithful BPC and measured latency ceilings.

| Path | Hotspot (measured, table_bits=16, post #7) | Change |
|------|-------------------------------------------|--------|
| `observe_next_byte` / `eval_bpc` | **~42 µs/byte** (compress-equivalent) | Unchanged |
| `log_prob_byte` | **~23–26 ms** (`copy_state_from` + 8 bits) | Reused `scratch_` |
| `next_byte_log_probs(32)` | **~46 ms** (delta-undo bit-tree) | Was ~3.7 s pre-#7 (legacy clone) |
| `next_byte_log_probs(256)` | **~228 ms** (delta-undo bit-tree) | Was ~22 s pre-#7 |

**RAM:** `VmHWM` construct ~761 MB → after profile ~1.27 GB @ mem 16 (single `scratch_` fork + main predictor).

---

## Hot-path breakdown

Gate24 production now uses **delta-undo MSB bit-tree** for full-vocab scoring (default `next_byte_log_probs`). Legacy 256-fork path: `CYPHA_HP_LEGACY_BYTE_LOGPROBS=1`. Per bit-tree call:

1. **Experts + ctx chain** — inside each `predict()` (dominant CPU)
2. **Mixer + APM + hedge** — per bit inside DFS branches
3. **Undo stack** — one `scratch_` fork; `UndoRecorderScope` + `pop_frame` per branch (replaces 256× `clone_from`)

### Measured latencies (2026-09-20, Linux KVM, 4 vCPU, gate24, `hp_table_bits=16`, rebased on #7)

```json
{
  "observe_next_byte_us": 41.6,
  "log_prob_byte_us": 22929,
  "next_byte_log_probs32_ms": 46,
  "next_byte_log_probs256_us": 227780,
  "vm_hwm_kb_construct": 760916,
  "vm_hwm_kb_after_serve": 1265392
}
```

**Interpretation:** Training/BPC (`observe_stream_bits`) remains fast (~24k bytes/s at mem 16). Full-vocab REST `predict_next` is now **sub-second** at vocab 256 via delta-undo (was seconds pre-#7).

---

## Micro-optimizations (this PR)

### `HpSequenceBackend` (`hp_backend.cpp`)

| Opt | Before | After |
|-----|--------|-------|
| `next_byte_log_probs` output | Fresh `std::vector` each call | **Reused** `log_probs_buf_` (explicit copy on return) |
| Serve path | Legacy 256× `clone_from` (gate24) | **Delta-undo bit-tree** (from #7; this PR keeps buffer reuse) |

No change to hp integer math, mixer weights, or BPC semantics.

---

## Binary checkpoint (HPCP v1)

| File | Role |
|------|------|
| `{base}.json` | Config metadata + `train_step_count` |
| `{base}.hpbin` | Magic `HPCP` v1 — full `hp::Predictor` tables + parser state |

Implementation: `hp/blob_io.hpp`, `hp/checkpoint.hpp` (serialized ContextModels, mixer, match, GRIA, …). Round-trip smoke: `native_hp_checkpoint_roundtrip_smoke`.

---

## CI gates (lightweight)

| Test | Gate |
|------|------|
| `native_hp_gate24_ci_gate` | Fixture BPC **6.53989 ± 0.05** (compress-equivalent, 16-token pattern) |
| | `log_prob_byte` **≤ 50 ms** median-of-5 (measured ~25 ms KVM / ~38.7 ms GHA + slack) |
| | `next_byte_log_probs(32)` **≤ 75 ms** median-of-3 (measured ~46 ms + slack) |
| `native_hp_checkpoint_roundtrip_smoke` | BPC identical before/after `.hpbin` load |

PR script: `scripts/ci_native_hp_smoke.sh` (regex `native_hp_*`).

---

## How to reproduce

```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build native/build --target cyphalm_hp_hotpath_profile hp_gate24_ci_gate hp_checkpoint_roundtrip_smoke -j$(nproc)
native/build/cyphalm_hp_hotpath_profile --max-bytes 4096 --table-bits 16
native/build/hp_gate24_ci_gate
native/build/hp_checkpoint_roundtrip_smoke
```

---

## Related

- [`CYPHALM_HP_ALGORITHM_PROFILE.md`](CYPHALM_HP_ALGORITHM_PROFILE.md) — enwik gate24 BPC bar  
- [`CYPHALM_TRAIN_SCALE.md`](CYPHALM_TRAIN_SCALE.md) — shard/undo roadmap  
- [#7 delta-undo](https://github.com/odin-loki/Cypha/pull/7) — bit-tree undo inference (merged)
- [#9 lossy LLM](https://github.com/odin-loki/Cypha/pull/9) — `prune_cold_slots` / serve compact (merged)
- [`CYPHALM_LOSSY_LLM_PLAN.md`](CYPHALM_LOSSY_LLM_PLAN.md) — lossy roadmap
