/// Smoke test for hybrid EWC on SSM multiscale alpha + GRIA per-token alpha.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"

namespace {

void train_sequence(cypha::cyphalm::CyphaLMModel& model, const std::vector<int>& ids, int steps) {
  model.reset_context();
  const int n = std::min(steps, static_cast<int>(ids.size()) - 1);
  for (int i = 0; i < n; ++i) {
    (void)model.train_step(static_cast<std::uint32_t>(ids[static_cast<std::size_t>(i)]),
                           static_cast<std::uint32_t>(ids[static_cast<std::size_t>(i + 1)]));
  }
}

double l2_drift(const std::vector<double>& a, const std::vector<double>& b) {
  assert(a.size() == b.size());
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = a[i] - b[i];
    sum += d * d;
  }
  return std::sqrt(sum);
}

double probe_gria_alpha_drift(double ewc_lambda, cypha::cyphalm::ContextMode mode) {
  cypha::cyphalm::CyphaLMConfig cfg;
  cfg.vocab_size = 32;
  cfg.field_dim = 32;
  cfg.d_embed = 16;
  cfg.d_state = 16;
  cfg.ssm_layers = 1;
  cfg.gria_rank = 8;
  cfg.context_mode = mode;
  cfg.ewc_lambda = ewc_lambda;
  cfg.gria_lr = 0.05;
  cfg.ssm_lr = 0.001;
  cfg.seed = 19;

  const std::vector<int> task_a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const std::vector<int> task_b = {20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

  cypha::cyphalm::CyphaLMModel model(cfg);
  train_sequence(model, task_a, 30);
  model.ewc_snapshot();
  assert(model.hybrid_ewc_regularizer().covers_gria_alpha());
  if (model.active_ssm() != nullptr) {
    assert(model.hybrid_ewc_regularizer().covers_ssm_alpha());
  }
  train_sequence(model, task_b, 30);
  return model.ewc_penalty();
}

}  // namespace

int main() {
  const double baseline_ssm = probe_gria_alpha_drift(0.0, cypha::cyphalm::ContextMode::SsmGria);
  const double ewc_ssm = probe_gria_alpha_drift(0.5, cypha::cyphalm::ContextMode::SsmGria);
  assert(baseline_ssm > 1e-12);
  assert(ewc_ssm < baseline_ssm);

  const double baseline_hybrid = probe_gria_alpha_drift(0.0, cypha::cyphalm::ContextMode::Hybrid);
  const double ewc_hybrid = probe_gria_alpha_drift(0.5, cypha::cyphalm::ContextMode::Hybrid);
  assert(baseline_hybrid > 1e-12);
  assert(ewc_hybrid < baseline_hybrid);

  cypha::cyphalm::CyphaLMConfig cfg;
  cfg.vocab_size = 32;
  cfg.field_dim = 32;
  cfg.d_embed = 16;
  cfg.d_state = 16;
  cfg.ssm_layers = 1;
  cfg.context_mode = cypha::cyphalm::ContextMode::SsmGria;
  cfg.ewc_lambda = 0.5;
  cfg.seed = 23;
  cypha::cyphalm::CyphaLMModel model(cfg);
  train_sequence(model, {1, 2, 3, 4, 5, 6}, 4);
  model.ewc_snapshot();
  const auto m = model.train_step(7, 8);
  assert(m.ewc_penalty > 0.0);

  std::printf(
      "ewc_hybrid_smoke: ssm_penalty=%.6e->%.6e hybrid_penalty=%.6e->%.6e PASS\n", baseline_ssm,
      ewc_ssm, baseline_hybrid, ewc_hybrid);
  return 0;
}
