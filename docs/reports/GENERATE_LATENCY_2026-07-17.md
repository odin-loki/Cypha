# CyphaLM `/generate` latency profile (2026-07-17)

**Scope:** Profile CyphaLM REST `/generate` → `generate_decode` → `predict_next` (D17 hybrid). Infer-only; not `train_step`.

**OFF-LIMITS:** `build_math`, `build_deff`, `BASELINE_*`, overnight.

**Build:** `native/build_perf_gen` (Ninja, Release, MinGW 13.2.0). Tool: `infer_latency_bench`.

**Prior context:** Train D17 floor already reached (Parts 1–7). RPSM `score_matrix` ~48% win shipped in `4d3afa2` (separate DIF path).

## Hot path

| Surface | Call chain |
|---------|------------|
| `POST /generate` | `handle_generate` → `generate_decode` → `predict_next` per token |
| `POST /lm/predict_next` | `predict_next` (same core; JSON wrap) |

`generate_decode` for prompt length *P* and *N* new tokens runs `predict_next` about *(P−1)+N* times (`consume_prompt` + decode loop). Bench `generate_1tok` with prompt `{1,2,3}` ≈ **3×** `predict_next` wall (observed).

## Quiet baseline (`infer_latency_bench`, same session)

Fixture: D17 hybrid synthetic (short train then decode). Quieter of two back-to-back runs:

| Metric | µs | derived |
|--------|-----|---------|
| `predict_next` | **618.74** (alt 651.40) | ~**1615 tok/s** |
| `generate` 1 tok | **1529.19** (alt 1706.06) | ~**654 tok/s** end-to-end for that microbench |

Later 3-run medians under load were **~1642 / ~3405 µs** (2–3× worse) — host noise; do not use for A/B.

## Allocation / dead-subsystem audit (generate-only)

Already cleaned on shared `predict_next` (Parts 3–7 / dead-work audit):

- Vocab `log_g` / `log_l` / LSTM h,c scratch reuse
- N-gram embed out-param scratch
- D17 DIF predict skip + Part 7b skip dead `last_dif_out_.mean` zero-fill
- Compressive memory store is interval-gated (not every step)

Remaining per-step work on D17 hybrid generate is **live compute**: SSM step, GRIA + n-gram fusion, LSTM forward, hybrid blend, `fill_top_k` (vocab partial_sort for REST top-k).

**Trialled (not shipped):** in-place hybrid `blend_log_probs` scratch + swap into `out.log_probs`. Bit-identical intent; wall A/B drowned in host noise (no reliable win). Reverted — docs-only floor this pass.

## Floor

CyphaLM generate is dominated by `predict_next` (~0.6–1.2 ms/tok when quiet on this host). Further micro-opts need a quieter machine or instruction-level counters; no safe free allocation skip found beyond prior Parts.

DIF REST `score_matrix` remains the proven infer win (~48% in `4d3afa2`).

## Reproduce

```text
cmake -S native -B native/build_perf_gen -DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_EXPERIMENT_DB=OFF -G Ninja
cmake --build native/build_perf_gen -j8 --target infer_latency_bench
native/build_perf_gen/infer_latency_bench
```
