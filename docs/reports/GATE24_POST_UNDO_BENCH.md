# gate24 post-undo serve benchmark

**Date:** 2026-09-20  
**Harness:** `native/tools/hp_inference_bench`  
**Profile:** gate24 (`v78_flags.ps1` + `HP_SLOT_MAX=24`, `table_bits=22`)

---

## Executive summary

After PR #7 (delta undo) and single-predictor serve (this PR), gate24 full-vocab scoring is **~293× faster** than the documented pre-undo checkpoint-tree baseline, with **~50% lower construct RSS** (one `hp::Predictor` instead of live+scratch twin).

---

## Measured latency + RSS (this run)

Warmup: **65,536** `consume_byte` steps (synthetic PRNG seed 42). Timed: **5** iterations after 2 warmup calls.

| Path | Latency (µs/call) | Latency (ms/call) |
|------|-------------------|-------------------|
| `predict_next` | **163,664** | **0.16** |
| `next_byte_log_probs` (default bit-tree) | **162,038** | **0.16** |
| `next_byte_log_probs_bit_tree` | **161,083** | **0.16** |
| `serve_greedy_next_byte` | **3,229** | **0.003** |

| RSS metric | kB | GiB |
|------------|-----|-----|
| VmRSS after model construct | **1,564,100** | **~1.49** |
| VmRSS after 64 KB warmup | **1,564,484** | **~1.49** |
| VmHWM after latency bench | **3,124,020** | **~2.98** |

Machine JSON: [`GATE24_POST_UNDO_BENCH.json`](GATE24_POST_UNDO_BENCH.json)

---

## Comparison to documented pre-undo baseline

Sources (not re-run here — cited from repo docs at PR #7 merge):

| Metric | Pre-undo (documented) | Post-undo + single-pred (measured) | Δ |
|--------|----------------------|-------------------------------------|---|
| `predict_next` @ mem22 | **~48,000,000 µs** (~48 s) — CHANGELOG PR #7 checkpoint-tree baseline | **163,664 µs** (~0.16 s) | **~293× faster** |
| Construct VmRSS class | **~4.3 GB** (dual `pred_` + 9 DFS checkpoints) — CHANGELOG | **1.56 GB** after construct | **~−49%** vs post-undo dual-scratch (3.12 GB); **~−64%** vs 4.3 GB class |
| `legacy` 256-clone @ 64 KB | **23,551 ms** — [`CYPHALM_BPC_GAP_REPORT.md`](CYPHALM_BPC_GAP_REPORT.md) | *(not re-run — opt-in only)* | — |
| Checkpoint DFS bit-tree @ 64 KB | **71,112 ms** — BPC gap report | **161 ms** (this run) | **~442× faster** |

---

## Single-predictor serve

**Status: implemented.**

`HpSequenceBackend` now holds **one** live `pred_`. Bit-tree DFS, `log_prob_byte`, `serve_greedy_next_byte`, and `serve_sample_next_byte` run speculative updates on `pred_` with `hp::PredictorUndoStack` delta undo — no standing `scratch_` twin.

| Concern | Result |
|---------|--------|
| Bit-tree correctness | `hp_bit_tree_smoke` PASS (max Δ vs legacy = 0) |
| Undo round-trip | `hp_undo_smoke` PASS |
| Serve/train split | `cyphalm_serve_smoke` PASS |
| RSS | Construct drops from ~3.1 GB (dual predictor) to **~1.5 GB** |

`hp_serve_compact` / `compact_for_serve()` remain as API hints but are **no-ops** when single-predictor is active (nothing to drop).

---

## Reproduce

```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build --target hp_inference_bench hp_bit_tree_smoke hp_undo_smoke cyphalm_serve_smoke -j$(nproc)

./native/build/hp_bit_tree_smoke
./native/build/hp_undo_smoke
./native/build/cyphalm_serve_smoke
./native/build/hp_inference_bench --table-bits 22 --warmup-bytes 65536 --iters 5 --json
```

Legacy 256-clone path (slow): `CYPHA_HP_LEGACY_BYTE_LOGPROBS=1` or `next_byte_log_probs_legacy()`.

---

*All post-undo numbers in the latency/RSS table are measured on the cloud-agent VM; pre-undo figures are cited from in-repo documentation only.*
