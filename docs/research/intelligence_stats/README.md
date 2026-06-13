# Intelligence Statistics — research papers

Seven universal intelligence statistics and their applications (P-space profiling, self-correction, soft-world simulation). **C++ implementation:** `native/include/cypha/intelligence/`.

| Paper | File | C++ status |
|-------|------|------------|
| I — Universal statistics | [universal_intelligence_statistics.md](universal_intelligence_statistics.md) | **Phase 1:** `IntelligenceProfiler`, `NIGStatisticState` |
| II — Applications | [intelligence_profile_applications.md](intelligence_profile_applications.md) | Planned (navigation loss, failure prediction) |
| III — Landscape / test bench | [intelligence_landscape_paper3.md](intelligence_landscape_paper3.md) | **Phase 1:** κ, health signal; bench TBD |
| IV — Self-correcting Cypha | [cypha_self_correcting_paper4.md](cypha_self_correcting_paper4.md) | **Phase 1:** `EpistemicThreshold`; context extension TBD |
| V — Soft world | [soft_world_paper5.md](soft_world_paper5.md) | **Phase 1:** `SoftWorldMonitor` maturation signal; causal graph TBD |

**Smoke test:** `intelligence_profiler_smoke` — CTest `native_intelligence_profiler_smoke`.

**Papers II–V test:** `intelligence_profiler_papers` — CTest `native_intelligence_profiler_papers`.

**Implementation report:** [docs/reports/INTELLIGENCE_STATS_IMPLEMENTATION.md](../../reports/INTELLIGENCE_STATS_IMPLEMENTATION.md)
