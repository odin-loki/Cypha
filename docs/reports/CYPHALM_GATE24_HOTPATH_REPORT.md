# CyphaLM gate24 hot-path profile + micro-opts

**Date:** 2026-09-20  
**Branch:** `cursor/gate24-hotpath-checkpoint-ci-6c23`  
**Harness:** `native/tools/cyphalm_hp_hotpath_profile.cpp`, `native/tests/regression/hp_gate24_ci_gate.cpp`

---

## Executive summary

Profiled the gate24 `hp::Predictor` adapt/serve paths on **delta-undo bit-tree** ([#7](https://github.com/odin-loki/Cypha/pull/7)) and **single-predictor serve** ([#12](https://github.com/odin-loki/Cypha/pull/12)). Applied **safe micro-optimizations** in `HpSequenceBackend` (reused `log_probs_buf_`). Added **binary HPCP v1 checkpoint** (`.hpbin`) for train-once / serve-many, plus **CI gates** for compress-faithful BPC and measured latency ceilings.

| Path | Hotspot (measured, table_bits=16, post #7+#12) | Change |
|------|-----------------------------------------------|--------|
| `observe_next_byte` / `eval_bpc` | **~36 µs/byte** (compress-equivalent) | Unchanged |
| `log_prob_byte` | **~3.5 ms** (undo on live `pred_`) | Single-pred from #12 |
| `next_byte_log_probs(32)` | **~20 ms** (delta-undo bit-tree on `pred_`) | Was ~3.7 s pre-#7 |
| `next_byte_log_probs(256)` | **~155 ms** (delta-undo bit-tree) | Was ~22 s pre-#7 |

**RAM:** `VmHWM` construct ~508 MB @ mem 16 (one `pred_`, no standing scratch twin — see [`GATE24_POST_UNDO_BENCH.md`](GATE24_POST_UNDO_BENCH.md)).

---

## Hot-path breakdown

Gate24 production now uses **delta-undo MSB bit-tree** for full-vocab scoring (default `next_byte_log_probs`). Legacy 256-fork path: `CYPHA_HP_LEGACY_BYTE_LOGPROBS=1`. Per bit-tree call:

1. **Experts + ctx chain** — inside each `predict()` (dominant CPU)
2. **Mixer + APM + hedge** — per bit inside DFS branches
3. **Undo stack** — delta undo on live `pred_`; `UndoRecorderScope` + `pop_frame` per branch (replaces 256× `clone_from`)

### Measured latencies (2026-09-20, Linux KVM, 4 vCPU, gate24, `hp_table_bits=16`, post #7+#12)

```json
{
  "observe_next_byte_us": 35.7,
  "log_prob_byte_us": 3855,
  "next_byte_log_probs32_ms": 20,
  "next_byte_log_probs256_us": 155212,
  "vm_hwm_kb_construct": 508368,
  "vm_hwm_kb_after_serve": 760968
}
```

**Interpretation:** Training/BPC (`observe_stream_bits`) remains fast. Full-vocab REST `predict_next` is **sub-second** at vocab 256 (see [`GATE24_POST_UNDO_BENCH.md`](GATE24_POST_UNDO_BENCH.md) for mem22 bench).

---

## Micro-optimizations (this PR)

### `HpSequenceBackend` (`hp_backend.cpp`)

| Opt | Before | After |
|-----|--------|-------|
| `next_byte_log_probs` output | Fresh `std::vector` each call | **Reused** `log_probs_buf_` (explicit copy on return) |
| Serve path | Legacy 256× fork (gate24) | **Single-pred undo bit-tree** (#7+#12; this PR keeps `log_probs_buf_` reuse) |

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
| | `log_prob_byte` **≤ 7.5 ms** median-of-5 (measured ~5.9 ms GHA macOS + slack) |
| | `next_byte_log_probs(32)` **≤ 30 ms** median-of-3 (measured ~20 ms + slack) |
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
- [#12 single-pred serve](https://github.com/odin-loki/Cypha/pull/12) — undo on live `pred_` (merged)
- [#9 lossy LLM](https://github.com/odin-loki/Cypha/pull/9) — `prune_cold_slots` (merged)
- [`GATE24_POST_UNDO_BENCH.md`](GATE24_POST_UNDO_BENCH.md) — mem22 serve latency + RSS bench
- [`CYPHALM_LOSSY_LLM_PLAN.md`](CYPHALM_LOSSY_LLM_PLAN.md) — lossy roadmap
