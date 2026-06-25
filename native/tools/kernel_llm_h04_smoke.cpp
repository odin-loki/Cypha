/// Smoke: H04 enables Phase 31 Nyström kernel LLR on CyphaDIF expert routing.
#include <cmath>
#include <cstdio>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cypha_cell_hypothesis.hpp"
#include "cypha/cyphalm/cyphalm_dif.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"

namespace {

bool routing_probs_differ(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) {
    return true;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::abs(a[i] - b[i]) > 1e-9) {
      return true;
    }
  }
  return false;
}

}  // namespace

int main() {
  cypha::cyphalm::CyphaLMConfig cfg;
  cfg.vocab_size = 64;
  cfg.field_dim = 32;
  cfg.d_embed = 32;
  cfg.n_experts = 4;
  cfg.seed = 11;
  cypha::cyphalm::apply_cell_variant("H04", cfg);

  if (!cfg.use_kernel_llr) {
    std::puts("kernel_llm_h04_smoke: FAIL (H04 use_kernel_llr not set)");
    return 1;
  }
  if (std::abs(cfg.kernel_blend - 0.25) > 1e-9) {
    std::puts("kernel_llm_h04_smoke: FAIL (H04 kernel_blend != 0.25)");
    return 1;
  }

  cypha::cyphalm::CyphaLMModel model(cfg);
  std::vector<int> ids;
  for (int i = 0; i < 96; ++i) {
    ids.push_back((i * 7 + 3) % static_cast<int>(cfg.vocab_size));
  }
  model.train_sequence(ids, 48, 1, nullptr);
  const double bpc = model.eval_bpc(ids, 32, nullptr);
  if (!std::isfinite(bpc)) {
    std::puts("kernel_llm_h04_smoke: FAIL (bpc not finite)");
    return 1;
  }

  cypha::cyphalm::CyphaLMConfig cfg_off = cfg;
  cfg_off.use_kernel_llr = false;
  cypha::cyphalm::CyphaDIF dif_on(cfg);
  cypha::cyphalm::CyphaDIF dif_off(cfg_off);
  std::vector<double> x(static_cast<std::size_t>(cfg.field_dim));
  for (int d = 0; d < cfg.field_dim; ++d) {
    x[static_cast<std::size_t>(d)] = 0.3 * std::sin(0.17 * static_cast<double>(d)) + 0.1;
  }
  for (int step = 0; step < 24; ++step) {
    const double* px = x.data();
    const int dim = cfg.field_dim;
    std::vector<double> y(static_cast<std::size_t>(cfg.field_dim));
    for (int d = 0; d < cfg.field_dim; ++d) {
      y[static_cast<std::size_t>(d)] = std::cos(0.11 * static_cast<double>(step + d));
    }
    dif_on.train_step(px, dim, y.data(), cfg.field_dim);
    dif_off.train_step(px, dim, y.data(), cfg.field_dim);
    x[static_cast<std::size_t>(step % cfg.field_dim)] += 0.05;
  }

  const auto out_on = dif_on.predict(x.data(), cfg.field_dim);
  const auto out_off = dif_off.predict(x.data(), cfg.field_dim);
  if (out_on.routing_probs.empty() || out_off.routing_probs.empty()) {
    std::puts("kernel_llm_h04_smoke: FAIL (empty routing probs)");
    return 1;
  }
  if (dif_on.expert_count() < 2) {
    std::puts("kernel_llm_h04_smoke: FAIL (expected >=2 experts for blend check)");
    return 1;
  }
  if (dif_on.kernel_n_basis() < 4) {
    std::puts("kernel_llm_h04_smoke: FAIL (Nyström n_basis < 4 after training)");
    return 1;
  }
  if (!routing_probs_differ(out_on.routing_probs, out_off.routing_probs)) {
    std::puts("kernel_llm_h04_smoke: FAIL (kernel LLR did not alter routing)");
    return 1;
  }

  const nlohmann::json state = dif_on.get_state();
  cypha::cyphalm::CyphaDIF dif_roundtrip(cfg);
  dif_roundtrip.set_state(state);
  if (dif_roundtrip.kernel_n_basis() != dif_on.kernel_n_basis()) {
    std::puts("kernel_llm_h04_smoke: FAIL (kernel snapshot roundtrip n_basis mismatch)");
    return 1;
  }
  const auto out_rt = dif_roundtrip.predict(x.data(), cfg.field_dim);
  if (out_rt.routing_probs.empty()) {
    std::puts("kernel_llm_h04_smoke: FAIL (roundtrip routing empty)");
    return 1;
  }

  std::puts("kernel_llm_h04_smoke: PASS");
  return 0;
}
