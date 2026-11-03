/// Smoke test: LmIntelligenceMonitor accumulates 32 synthetic LM steps and validates full profile.
#include <cmath>
#include <cstdio>
#include <vector>

#include "cypha/cyphalm/lm_intelligence_monitor.hpp"
#include "cypha/intelligence/intelligence_profiler.hpp"
#include "cypha/intelligence/profile_completeness.hpp"

namespace {

constexpr int kSteps = 32;
constexpr int kFieldDim = 8;
constexpr int kEmbedDim = 8;
constexpr int kVocab = 16;

std::vector<double> make_log_probs(int step, int target_id) {
  std::vector<double> log_probs(static_cast<std::size_t>(kVocab), -4.0);
  log_probs[static_cast<std::size_t>(target_id % kVocab)] = -0.1;
  if (step % 5 == 0) {
    log_probs[static_cast<std::size_t>((target_id + 1) % kVocab)] = -0.05;
  }
  return log_probs;
}

}  // namespace

int main() {
  cypha::intelligence::IntelligenceProfiler profiler;
  cypha::cyphalm::LmIntelligenceMonitor monitor;

  for (int step = 0; step < kSteps; ++step) {
    std::vector<double> embed(static_cast<std::size_t>(kEmbedDim));
    std::vector<double> field(static_cast<std::size_t>(kFieldDim));
    for (int d = 0; d < kEmbedDim; ++d) {
      embed[static_cast<std::size_t>(d)] =
          0.15 * static_cast<double>(step) + 0.03 * static_cast<double>(d);
    }
    for (int d = 0; d < kFieldDim; ++d) {
      field[static_cast<std::size_t>(d)] =
          std::sin(0.11 * static_cast<double>(step) + 0.07 * static_cast<double>(d));
    }

    const int target = step % kVocab;
    const std::vector<double> log_probs = make_log_probs(step, target);
    monitor.observe_token(embed, field, log_probs, 0.25 + 0.01 * static_cast<double>(step % 4),
                          0.15 + 0.005 * static_cast<double>(step % 3),
                          static_cast<std::int64_t>(target), kVocab);
  }

  monitor.flush_to_profiler(profiler);

  // docs/reports/SOFT_WORLD_CAUSAL_GRAPH_PLAN.md §9.7: `flush_to_profiler` now also feeds the
  // profiler's persistent `CausalGraphMonitor` with growing-prefix checkpoints reconstructed
  // from this monitor's own per-token history, so a single flush over enough steps (here,
  // kSteps=32, well above the 4-checkpoint x 4-min-samples floor) must leave both estimated
  // edges with real accumulated history -- not the single degenerate observation the old
  // fresh-per-report-call `CausalGraphMonitor` was stuck with.
  const auto& causal = profiler.causal_graph();
  if (causal.alpha_calibration_n() < 4 || causal.tau_r_eu_n() < 4) {
    std::printf("intelligence_lm_monitor_smoke: FAIL (causal graph under-accumulated: "
                "alpha_calibration_n=%d tau_r_eu_n=%d)\n",
                causal.alpha_calibration_n(), causal.tau_r_eu_n());
    return 1;
  }

  const auto snapshot = monitor.snapshot_observation();
  if (!cypha::intelligence::profile_observation_complete(snapshot)) {
    std::puts("intelligence_lm_monitor_smoke: FAIL (incomplete snapshot)");
    return 1;
  }

  const cypha::intelligence::ProfileCompletenessResult completeness =
      cypha::intelligence::validate_profile_completeness(profiler, 1);
  if (!completeness.all_complete) {
    std::puts("intelligence_lm_monitor_smoke: FAIL (profiler incomplete)");
    return 1;
  }
  if (!std::isfinite(completeness.kappa)) {
    std::puts("intelligence_lm_monitor_smoke: FAIL (kappa not finite)");
    return 1;
  }

  std::puts("intelligence_lm_monitor_smoke: PASS");
  return 0;
}
