#pragma once

namespace cypha::intelligence {

/// One row of the seven-statistic profile vector ``P = (α, D_eff, σ, τ, r_eu, L, C)``.
///
/// Factored out of ``intelligence_profiler.hpp`` (docs/reports/SOFT_WORLD_CAUSAL_GRAPH_PLAN.md
/// §9.7) so ``causal_graph.hpp`` can depend on this single-struct header instead of the full
/// ``IntelligenceProfiler`` declaration -- letting ``intelligence_profiler.hpp`` in turn embed a
/// ``CausalGraphMonitor`` member without an include cycle between the two headers.
struct ProfileObservation {
  double alpha = 0.5;
  double d_eff = 0.5;
  double sigma_branch = 0.5;
  double tau = 0.5;
  double r_eu = 0.5;
  double lipschitz = 0.5;
  double calibration = 0.5;
};

}  // namespace cypha::intelligence
