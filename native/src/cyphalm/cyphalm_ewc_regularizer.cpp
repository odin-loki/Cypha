#include "cypha/cyphalm/cyphalm_ewc_regularizer.hpp"

#include <cmath>

namespace cypha::cyphalm {

namespace {

constexpr double kFisherEps = 1e-8;

void build_diagonal_fisher_from_anchor(const std::vector<double>& anchor, std::vector<double>& fisher_out) {
  fisher_out.resize(anchor.size());
  for (std::size_t i = 0; i < anchor.size(); ++i) {
    fisher_out[i] = anchor[i] * anchor[i] + kFisherEps;
  }
}

double squared_penalty(const std::vector<double>& theta, const std::vector<double>& anchor,
                       const std::vector<double>& fisher) {
  if (theta.size() != anchor.size() || theta.size() != fisher.size()) {
    return 0.0;
  }
  double sum = 0.0;
  for (std::size_t i = 0; i < theta.size(); ++i) {
    const double d = theta[i] - anchor[i];
    sum += fisher[i] * d * d;
  }
  return 0.5 * sum;
}

void pull_toward_anchor(std::vector<double>& theta, const std::vector<double>& anchor,
                        const std::vector<double>& fisher, double strength) {
  if (theta.size() != anchor.size() || theta.size() != fisher.size() || strength <= 0.0) {
    return;
  }
  for (std::size_t i = 0; i < theta.size(); ++i) {
    const double d = theta[i] - anchor[i];
    theta[i] -= strength * fisher[i] * d;
  }
}

}  // namespace

void CyphaLMEwcRegularizer::snapshot(const CharLSTMHead& lstm) {
  anchor_Wx_ = lstm.Wx;
  anchor_Wh_ = lstm.Wh;
  build_diagonal_fisher_from_anchor(anchor_Wx_, fisher_Wx_);
  build_diagonal_fisher_from_anchor(anchor_Wh_, fisher_Wh_);
  grad_observations_ = 0;
}

void CyphaLMEwcRegularizer::observe_grads(const CharLSTMGrad& grads) {
  if (!has_snapshot()) {
    return;
  }
  const std::size_t next_count = grad_observations_ + 1;
  const double inv_n = 1.0 / static_cast<double>(next_count);

  auto update_block = [&](const std::vector<double>& grad, std::vector<double>& fisher) {
    if (grad.size() != fisher.size()) {
      return;
    }
    for (std::size_t i = 0; i < fisher.size(); ++i) {
      const double g2 = grad[i] * grad[i];
      fisher[i] = fisher[i] * static_cast<double>(grad_observations_) * inv_n + g2 * inv_n;
      if (fisher[i] < kFisherEps) {
        fisher[i] = kFisherEps;
      }
    }
  };

  update_block(grads.dWx, fisher_Wx_);
  update_block(grads.dWh, fisher_Wh_);
  grad_observations_ = next_count;
}

double CyphaLMEwcRegularizer::penalty(const CharLSTMHead& lstm) const {
  if (!has_snapshot()) {
    return 0.0;
  }
  return squared_penalty(lstm.Wx, anchor_Wx_, fisher_Wx_) +
         squared_penalty(lstm.Wh, anchor_Wh_, fisher_Wh_);
}

void CyphaLMEwcRegularizer::apply_pull(CharLSTMHead& lstm, double ewc_lambda, double lr) const {
  if (!has_snapshot() || ewc_lambda <= 0.0 || lr <= 0.0) {
    return;
  }
  const double strength = ewc_lambda * lr;
  pull_toward_anchor(lstm.Wx, anchor_Wx_, fisher_Wx_, strength);
  pull_toward_anchor(lstm.Wh, anchor_Wh_, fisher_Wh_, strength);
}

}  // namespace cypha::cyphalm
