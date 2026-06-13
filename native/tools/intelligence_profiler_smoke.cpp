/// Smoke test for intelligence profiler: NIG updates, measurers, κ and health signal.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "cypha/intelligence/intelligence_profiler.hpp"
#include "cypha/intelligence/measurers.hpp"
#include "cypha/intelligence/nig_statistic_state.hpp"

namespace {

constexpr double kTol = 1e-6;

bool near(double a, double b, double tol = kTol) {
  return std::abs(a - b) <= tol;
}

void test_nig_update() {
  cypha::intelligence::NigStatisticState state(0.0, 1.0, 2.0, 0.1);
  state.update(1.0);
  assert(near(state.mean(), 0.5));
  assert(state.lambda() == 2.0);
  assert(state.alpha() == 2.5);
  assert(state.epistemic_var() > 0.0);
  assert(state.aleatoric_var() > 0.0);
  assert(state.n_updates() == 1);
}

void test_measurers() {
  const std::vector<double> activations = {
      0.1, 0.2, 0.3,
      0.4, 0.5, 0.6,
      0.7, 0.8, 0.9,
  };
  const double d_eff =
      cypha::intelligence::compute_participation_ratio(activations.data(), 3, 3);
  assert(d_eff > 0.0 && d_eff <= 1.0);

  const std::vector<double> conf = {0.1, 0.3, 0.5, 0.7, 0.9, 0.9, 0.9, 0.9};
  const std::vector<int> correct = {0, 0, 1, 1, 1, 1, 0, 1};
  const double c = cypha::intelligence::compute_calibration(conf.data(), correct.data(),
                                                            static_cast<int>(conf.size()));
  assert(c >= 0.0 && c <= 1.0);

  const double r_eu = cypha::intelligence::compute_epistemic_ratio(0.7, 0.3);
  assert(near(r_eu, 0.7));

  const std::vector<double> input = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
  const std::vector<double> output = {0.2, 0.8, 0.1, 0.9, 0.3, 0.7};
  const double alpha =
      cypha::intelligence::compute_alpha_gria(input.data(), output.data(), 3, 2);
  assert(alpha >= 0.0 && alpha <= 1.0);
}

void test_profiler() {
  cypha::intelligence::IntelligenceProfiler profiler;
  const auto targets = cypha::intelligence::IntelligenceProfiler::critical_targets();
  assert(targets.size() == cypha::intelligence::kProfileStatisticCount);

  cypha::intelligence::ProfileObservation obs;
  obs.alpha = targets[0];
  obs.d_eff = targets[1];
  obs.sigma_branch = targets[2];
  obs.tau = targets[3];
  obs.r_eu = targets[4];
  obs.lipschitz = targets[5];
  obs.calibration = targets[6];

  for (int step = 0; step < 8; ++step) {
    profiler.update(obs);
  }

  const auto matrix = profiler.get_profile_matrix();
  assert(matrix.size() == cypha::intelligence::kProfileStatisticCount);
  for (const auto& row : matrix) {
    assert(row[0] >= 0.0);
    assert(row[1] > 0.0);
    assert(row[2] > 0.0);
  }

  const double kappa = profiler.criticality_score();
  assert(kappa > 0.99);

  const double health = profiler.health_signal();
  assert(health >= 0.0);

  const std::vector<double> input(12, 0.5);
  std::vector<double> output = input;
  for (std::size_t i = 0; i < output.size(); ++i) {
    output[i] += static_cast<double>(i % 3) * 0.05;
  }
  const std::vector<double> conf = {0.2, 0.8};
  const std::vector<int> labels = {0, 1};

  cypha::intelligence::ProfileBatch batch;
  batch.input = input.data();
  batch.output = output.data();
  batch.n_samples = 4;
  batch.n_dims = 3;
  batch.confidences = conf.data();
  batch.correct = labels.data();
  batch.n_labels = static_cast<int>(conf.size());
  batch.epistemic_var = 0.6;
  batch.aleatoric_var = 0.4;
  batch.sigma_branch = 0.5;
  batch.tau = 0.65;
  batch.lipschitz = 0.5;
  profiler.update_from_batch(batch);
}

}  // namespace

int main() {
  test_nig_update();
  test_measurers();
  test_profiler();
  std::puts("intelligence_profiler_smoke: PASS");
  return 0;
}
