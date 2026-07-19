# D17B expert / alpha reporting — diagnostic (2026-07-12)

**Status:** Shipped (reporting clarity + diagnostic tool). Root cause is **not** a warm-start count bug.  
**Priority:** [`RESEARCH_STATUS.md`](../RESEARCH_STATUS.md) Priority 3 Step 5.

## Question

D17B reports **mean_alpha ≈ 0.095** and **n_experts = 1** on the hybrid D17 profile. Is this a `compression_profile()` reporting bug (case a) or genuine CyphaDIF novelty-gated expert growth under the locked hybrid config (case b)?

## Tool

`d17b_expert_diagnose` — trains D17 hybrid (`apply_bench_profile("d17")` + `BenchMode::Hybrid`) on synthetic or real corpus, printing `compression_profile()` checkpoints and per-step `train_step.active_experts`.

```powershell
cmake --build native/build_ewc_d16 --target d17b_expert_diagnose
.\native\build_ewc_d16\d17b_expert_diagnose.exe [n_train] [chunk] [warm_start_n_experts] [real]
```

## Findings (synthetic, 8k train, default profile)

| Setting | `cfg_n_experts` | `n_experts` (live) | `mean_alpha` | `mean_expert_alpha` | `active_experts` max |
|---:|---:|---:|---:|---:|---:|
| Cold start (default) | 0 | **1** | 0.498 | **0.000** | 1 |
| Warm-start 8 experts | 8 | **8** | 0.485 | **0.000** | 8 |

### Case (b) — genuine single-expert dynamics on default profile

- D17 hybrid profiles set **`n_experts: 0`** (cold start). CyphaDIF grows experts via novelty gating (`cyphalm_dif.cpp` `route()`).
- On synthetic char-LM field inputs, novelty rarely fires beyond the first expert → **`n_experts = 1`** is expected, not under-reported.
- When `n_experts > 0` is set in config, **`compression_profile()["n_experts"]` matches `cfg.n_experts`** after warm-start — no warm-start count bug.

### Reporting clarity fix (shipped)

Previously, **`mean_alpha` blended GRIA projection α (fixed 0.5 in default hybrid)** with per-expert entropy-derived α (≈ 0), making D17B look like "low alpha with 1 expert" without separating components.

`compression_profile()` now also reports:

- `cfg_n_experts`, `max_experts` — profile warm-start vs cap
- `mean_expert_alpha` — CyphaDIF experts only (excludes GRIA)
- D17B bench JSON (`17B_alpha_spectrum`) passes through the new fields

**Interpretation:** use `mean_expert_alpha` + `n_experts` for CyphaDIF routing health; use `mean_alpha` only when GRIA+expert blend is intentional.

## Next steps (not in this commit)

- Re-run full D17 bench after overnight lock (do not touch `bench/BASELINE_REPORT.md` in flight).
- Consider raising `n_experts` in `cyphalm_d17_hybrid.json` if multi-expert DIF routing is desired for LM (separate ablation; d74 grid exists).
