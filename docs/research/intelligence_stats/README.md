# Intelligence Statistics — research papers

Seven universal intelligence statistics and their applications (P-space profiling, self-correction, soft-world simulation). **C++ implementation:** `native/include/cypha/intelligence/`.

| Paper | File | C++ status |
|-------|------|------------|
| I — Universal statistics | [universal_intelligence_statistics.md](universal_intelligence_statistics.md) | **Phase 1:** `IntelligenceProfiler`, `NIGStatisticState` |
| II — Applications | [intelligence_profile_applications.md](intelligence_profile_applications.md) | Planned (navigation loss, failure prediction) |
| III — Landscape / test bench | [intelligence_landscape_paper3.md](intelligence_landscape_paper3.md) | **Phase 1:** κ, health signal; bench TBD |
| IV — Self-correcting Cypha | [cypha_self_correcting_paper4.md](cypha_self_correcting_paper4.md) | **Phase 1:** `EpistemicThreshold`; context extension TBD |
| V — Soft world | [soft_world_paper5.md](soft_world_paper5.md) | **Phase 1:** `SoftWorldMonitor`; causal graph TBD |
| Upgrades index | [../upgrades/README.md](../upgrades/README.md) | RPSM, cell hypothesis, nonlinear boundary |

**Phase 2 (2026-06):** `profile_from_model`, `self_correcting_infer`, bench domain **d18**, `cypha_intelligence_bench`, REST `/intelligence/report`.

**Smoke tests:** `native_intelligence_profiler_smoke`, `native_intelligence_profiler_papers`, `native_intelligence_bench_smoke`.

**Implementation report:** [docs/reports/INTELLIGENCE_STATS_IMPLEMENTATION.md](../../reports/INTELLIGENCE_STATS_IMPLEMENTATION.md)
