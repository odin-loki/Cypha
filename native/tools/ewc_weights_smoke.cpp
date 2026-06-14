/// Smoke test for hybrid EWC weight Fisher on GRIA U/V and SSM W_fast layer-0.
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

struct WeightDrift {
  double gria_u{0.0};
  double gria_v{0.0};
  double ssm_w_fast{0.0};
  double total{0.0};
};

WeightDrift measure_weight_drift(const cypha::cyphalm::CyphaLMModel& model,
                                 const std::vector<double>& anchor_u,
                                 const std::vector<double>& anchor_v,
                                 const std::vector<double>& anchor_w_fast) {
  WeightDrift out;
  const auto* gria = model.gria_routing();
  assert(gria != nullptr);
  out.gria_u = l2_drift(gria->U, anchor_u);
  out.gria_v = l2_drift(gria->V, anchor_v);
  const auto* ssm = model.active_ssm();
  assert(ssm != nullptr);
  out.ssm_w_fast = l2_drift(ssm->w_fast_layer0(), anchor_w_fast);
  out.total = out.gria_u + out.gria_v + out.ssm_w_fast;
  return out;
}

WeightDrift probe_weight_drift(double ewc_lambda) {
  cypha::cyphalm::CyphaLMConfig cfg;
  cfg.vocab_size = 32;
  cfg.field_dim = 32;
  cfg.d_embed = 16;
  cfg.d_state = 16;
  cfg.ssm_layers = 1;
  cfg.gria_rank = 8;
  cfg.bptt_steps = 1;
  cfg.context_mode = cypha::cyphalm::ContextMode::SsmGria;
  cfg.ewc_lambda = ewc_lambda;
  cfg.gria_lr = 0.06;
  cfg.ssm_lr = 0.25;
  cfg.train_ssm = true;
  cfg.seed = 29;

  const std::vector<int> task_a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const std::vector<int> task_b = {20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

  cypha::cyphalm::CyphaLMModel model(cfg);
  train_sequence(model, task_a, 80);
  const auto* gria = model.gria_routing();
  assert(gria != nullptr);
  const auto anchor_u = gria->U;
  const auto anchor_v = gria->V;
  const auto* ssm = model.active_ssm();
  assert(ssm != nullptr);
  const auto anchor_w_fast = ssm->w_fast_layer0();
  model.ewc_snapshot();
  assert(model.hybrid_ewc_regularizer().covers_gria_weights());
  assert(model.hybrid_ewc_regularizer().covers_gria_bias());
  assert(model.hybrid_ewc_regularizer().covers_ssm_w_fast());
  assert(model.hybrid_ewc_regularizer().covers_ssm_w_slow());
  train_sequence(model, task_b, 80);
  return measure_weight_drift(model, anchor_u, anchor_v, anchor_w_fast);
}

bool drift_reduced(const WeightDrift& baseline, const WeightDrift& ewc) {
  const double baseline_gria = baseline.gria_u + baseline.gria_v;
  const double ewc_gria = ewc.gria_u + ewc.gria_v;
  const double w_fast_eps = 1e-7;
  const bool w_fast_ok =
      baseline.ssm_w_fast < w_fast_eps || ewc.ssm_w_fast + w_fast_eps < baseline.ssm_w_fast;
  return ewc.total < baseline.total && ewc_gria + 1e-12 < baseline_gria && w_fast_ok;
}

}  // namespace

int main() {
  const auto baseline = probe_weight_drift(0.0);
  const auto ewc = probe_weight_drift(1.0);
  if (baseline.total <= 1e-12) {
    std::fprintf(stderr, "ewc_weights_smoke: baseline drift too small\n");
    return 1;
  }
  if (!drift_reduced(baseline, ewc)) {
    std::fprintf(stderr,
                 "ewc_weights_smoke: drift not reduced (total %.6e->%.6e gria_u %.6e->%.6e "
                 "gria_v %.6e->%.6e w_fast %.6e->%.6e)\n",
                 baseline.total, ewc.total, baseline.gria_u, ewc.gria_u, baseline.gria_v, ewc.gria_v,
                 baseline.ssm_w_fast, ewc.ssm_w_fast);
    return 1;
  }

  cypha::cyphalm::CyphaLMConfig cfg;
  cfg.vocab_size = 32;
  cfg.field_dim = 32;
  cfg.d_embed = 16;
  cfg.d_state = 16;
  cfg.ssm_layers = 1;
  cfg.gria_rank = 8;
  cfg.bptt_steps = 2;
  cfg.context_mode = cypha::cyphalm::ContextMode::SsmGria;
  cfg.ewc_lambda = 1.0;
  cfg.seed = 31;
  cypha::cyphalm::CyphaLMModel model(cfg);
  train_sequence(model, {1, 2, 3, 4, 5, 6}, 6);
  model.ewc_snapshot();
  assert(model.hybrid_ewc_regularizer().covers_gria_weights());
  assert(model.hybrid_ewc_regularizer().covers_gria_bias());
  assert(model.hybrid_ewc_regularizer().covers_ssm_w_fast());
  assert(model.hybrid_ewc_regularizer().covers_ssm_w_slow());
  const auto m = model.train_step(7, 8);
  assert(m.ewc_penalty > 0.0);

  std::printf(
      "ewc_weights_smoke: gria_u=%.6e->%.6e gria_v=%.6e->%.6e w_fast=%.6e->%.6e PASS\n",
      baseline.gria_u, ewc.gria_u, baseline.gria_v, ewc.gria_v, baseline.ssm_w_fast, ewc.ssm_w_fast);
  return 0;
}
