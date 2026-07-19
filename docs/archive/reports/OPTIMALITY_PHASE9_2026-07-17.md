# Optimality Phase 9 — Runtime Criticality Monitor (read-only) (2026-07-17)

**Build:** `native/build_opt_p9` (Ninja, Release, MinGW 13.2.0)  
**Scope:** Read-only telemetry first; no inference-path mutation. Did not touch `build_math`, `build_deff`, `BASELINE_*`, overnight scripts.

## Shipped

| Component | Location | Notes |
|-----------|----------|-------|
| **`CriticalityField`** | `native/include/cypha/intelligence/criticality_vector.hpp` | Per-gauge: `value`, `tier` (T1/T2), `target_or_bound`, `cadence` (hot/mid/cold), `distance`, `available` |
| **`CriticalityVector`** | same | `fields[]` + `mid_enabled` gate |
| **Builder** | `native/src/intelligence/criticality_vector.cpp` | Hot fields from profiler + optional session extras; mid estimators behind `CriticalityBuildOptions::enable_mid` |
| **`IntelligenceProfiler::criticality_vector()`** | `intelligence_profiler.hpp/.cpp` | Read-only accessor |
| **`LmIntelligenceMonitor::hot_criticality_input()`** | `lm_intelligence_monitor.hpp/.cpp` | Hot gauges from token history |
| **`SoftWorldMonitor::criticality_hot_input()`** | `soft_world_monitor.hpp` | Paper V maturation → hot extras |
| **REST** | `intelligence_rest_routes.cpp` | `GET /intelligence/criticality?mid=1&step=N`; also embedded in `/intelligence/profile` and `/intelligence/report` |
| **CTest** | `native_criticality_vector_p9_smoke` | Asserts inference byte-identical with monitor on vs off |

## Hot fields (every step)

| Field | Tier | Target / bound | Source |
|-------|------|----------------|--------|
| `alpha` | T2 | 0.5 | Profiler NIG mean (GRIA axis) |
| `routing_entropy` | T2 | 0.8 | Optional `CriticalityHotInput` (MoE gate softmax) |
| `dead_expert_fraction` | T2 | 0.0 | Optional `CriticalityHotInput` |
| `anomaly_score` | T2 | 0.5 | Session / LM monitor epistemic ratio |
| `drift_score` | T2 | 0.0 | LM monitor α drift; soft-world drift proxy |
| `nig_confidence` | T2 | 0.82 | NIG epistemic inverse or session override |
| `effective_sample_size` | T2 | 8.0 | NIG update counts or token history length |
| `ood_rate` | T1 | ≤ 0.05 | Session OOD flag rate |

## Mid fields (gated: `enable_mid && step % 64 == 0`)

| Field | Tier | Target / bound | Status |
|-------|------|----------------|--------|
| `spectral_radius` | T2 | 1.0 | Optional input (power-iteration hook) |
| `kernel_frobenius_error` | T1 | ≤ 0.01 | Optional input (`rff_features` estimator) |
| `forgetting_canary` | T1 | ≤ 0.001 | Optional input (frozen-canary proxy) |

Mid fields are omitted from the vector unless the gate is open **and** the caller supplies estimator values.

## REST field shape

```json
{
  "criticality_vector": {
    "fields": [
      {
        "name": "alpha",
        "value": 0.52,
        "tier": "T2",
        "target_or_bound": 0.5,
        "cadence": "hot",
        "distance": 0.02,
        "available": true
      }
    ],
    "mid_enabled": false
  },
  "source": "intelligence_profiler"
}
```

Dedicated route: `GET /intelligence/criticality` (optional `?mid=1&step=0`).

## Read-only guarantee

`criticality_vector_p9_smoke`:

1. Runs `infer_at_h` on `fixtures/reference.cypha` + first `native_parity.bin` row.
2. Builds profiler + criticality vector (hot + mid-gated).
3. Re-runs inference — **label, confidence, and LLR vector are `memcmp`-identical**.

No goldens touched.

## Files touched (Phase 9)

- `native/include/cypha/intelligence/criticality_vector.hpp` (new)
- `native/src/intelligence/criticality_vector.cpp` (new)
- `native/include/cypha/intelligence/intelligence_profiler.hpp`
- `native/include/cypha/intelligence/soft_world_monitor.hpp`
- `native/include/cypha/cyphalm/lm_intelligence_monitor.hpp`
- `native/src/intelligence/intelligence_profiler.cpp`
- `native/src/intelligence/intelligence_profile_json.cpp`
- `native/src/intelligence/profile_from_model.cpp`
- `native/src/cyphalm/lm_intelligence_monitor.cpp`
- `native/apps/intelligence_rest_routes.cpp`
- `native/include/cypha/intelligence_rest.hpp`
- `native/tools/criticality_vector_p9_smoke.cpp` (new)
- `native/CMakeLists.txt`

## Hot-path overhead (2026-07-17, `build_perf_p9`)

**Claim check:** `CriticalityVector` is **not** invoked inside D17 `train_step`. Population is report/REST only (`intelligence_profile_to_json` / `intelligence_profile_report_json` / `GET /intelligence/criticality`). Mid estimators remain behind `enable_mid && step % 64 == 0`.

| Metric | Value |
|--------|-------|
| D17 hybrid train (8k steps, synthetic) | 73.26 s (~9157 µs/step) |
| Simulated per-step `criticality_vector()` ×8k | 0.0048 s (~0.60 µs/call) |
| **Simulated if called every train step** | **0.007%** of train wall |
| **Shipped hot-path overhead** | **0.000%** (not on train path) |
| BPC (profiler-train vs baseline) | **bit-identical** `7.994747130705` |

### Gate shipped

`CYPHA_CRITICALITY` — default **ON** (REST + report embed). Set `0` / `false` / `off` to strip `criticality_vector` from profile JSON and return 503 on `/intelligence/criticality`. Bench train path pays nothing either way.

Tool: `criticality_overhead_p9_bench` (`native/build_perf_p9`).

## Promotion path

1. Wire REST `/predict` session extras into `CriticalityHotInput` (routing entropy from MKE head when loaded).
2. Connect mid estimators: SSM spectral radius power iteration, stochastic `‖K−K̂‖_F` from `rff_features`, forgetting canary from frozen fixture LLR drift.
3. Optional `experiment_db` ring buffer for criticality history (cold cadence only after mid fields stabilize).
