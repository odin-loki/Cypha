# CyphaLM gate24 hot-path profile + micro-opts

**Date:** 2026-09-20  
**Branch:** `cursor/gate24-hotpath-checkpoint-ci-6c23`  
**Harness:** `native/tools/cyphalm_hp_hotpath_profile.cpp`, `native/tests/regression/hp_gate24_ci_gate.cpp`

---

## Executive summary

Profiled the gate24 `hp::Predictor` adapt/serve paths and applied **safe micro-optimizations** in `HpSequenceBackend` (no undo stack). Added **binary HPCP v1 checkpoint** (`.hpbin`) for train-once / serve-many, plus **CI gates** for compress-faithful BPC and measured latency ceilings.

| Path | Hotspot (measured, table_bits=16) | Change |
|------|-----------------------------------|--------|
| `observe_next_byte` / `eval_bpc` | **~47 µs/bit-byte** (compress-equivalent) | Unchanged (already fast) |
| `log_prob_byte` | **~27–29 ms** (single assign_from + 8 bits) | Lazy `scratch_` alloc |
| `next_byte_log_probs(32)` | **~3.7 s** (32× `clone_from` legacy path) | Reused `log_probs_buf_`; legacy path still clone-bound |
| `next_byte_log_probs(256)` | **~7.15 s** (gate24 legacy path) | Same assign_from reuse; still serve-bound |

**RAM:** lazy `scratch_` defers second full predictor until first serve-path call (`VmHWM` construct ~509 MB → after profile ~1.27 GB @ mem 16).

---

## Hot-path breakdown

Gate24 production uses the **legacy 256-assign** path for full-vocab scoring (`CYPHA_HP_GATE24` — bit-tree DFS checkpoint pool OOM at v78 table sizes). Per call:

1. **Experts + ctx chain** — inside each `predict()` (dominant CPU in upstream hp)
2. **Mixer + APM + hedge** — per bit inside `byte_log_prob` / `observe_next_byte`
3. **assign_from / clone_from** — Cypha adapter overhead on serve path (addressed by scratch reuse)

### Measured latencies (2026-09-20, Linux KVM, 4 vCPU, gate24, `hp_table_bits=16`)

```json
{
  "observe_next_byte_us": 47.2,
  "log_prob_byte_us": 26171,
  "next_byte_log_probs256_us": 21982207,
  "next_byte_log_probs32_ms": 3700,
  "vm_hwm_kb_construct": 508636,
  "vm_hwm_kb_after_serve": 1265356
}
```

**Interpretation:** Training/BPC (`observe_stream_bits`) is **~21k bytes/s** at mem 16 — suitable for CI. Full-vocab REST `predict_next` remains **seconds per call** until delta-undo lands ([#7](https://github.com/odin-loki/Cypha/pull/7); see also [`CYPHALM_LOSSY_LLM_PLAN.md`](CYPHALM_LOSSY_LLM_PLAN.md) Phase 1).

---

## Micro-optimizations (this PR)

### `HpSequenceBackend` (`hp_backend.cpp`)

| Opt | Before | After |
|-----|--------|-------|
| `scratch_` lifetime | Constructed with `pred_` always | **Lazy** on first serve-path call (~509 MB → defer 2nd predictor) |
| `next_byte_log_probs` output | Fresh `std::vector` each call | **Reused** `log_probs_buf_` (explicit copy on return) |
| Legacy `assign_from` reuse | — | **Rejected** — breaks `hp_bit_tree_smoke` parity vs `clone_from` (Δ≈2.59 nats); needs hp investigation |

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
| | `log_prob_byte` **≤ 35 ms** median-of-5 (measured ~26.2 ms + slack) |
| | `next_byte_log_probs(32)` **≤ 5000 ms** median-of-3 (measured ~3.7 s + slack) |
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
- [#7 delta-undo](https://github.com/odin-loki/Cypha/pull/7) — next latency win for full-vocab `predict_next`
- [`CYPHALM_LOSSY_LLM_PLAN.md`](CYPHALM_LOSSY_LLM_PLAN.md) — undo stack roadmap
