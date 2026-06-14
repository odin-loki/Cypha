#include "cypha/ewc_regularizer.hpp"

#include <cmath>

namespace cypha {

namespace {

constexpr double kFisherEps = 1e-8;

void build_diagonal_fisher(const std::vector<double>& anchor, std::vector<double>& fisher_out) {
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

void EwcRegularizer::snapshot(const CyphaDifMemoryState& mem, const CyphaInferModel& infer) {
  anchor_D_ = mem.D;
  anchor_enc_w_ = infer.enc_w;
  build_diagonal_fisher(anchor_D_, fisher_D_);
  build_diagonal_fisher(anchor_enc_w_, fisher_enc_w_);
}

double EwcRegularizer::penalty(const CyphaDifMemoryState& mem, const CyphaInferModel& infer) const {
  if (!has_snapshot()) {
    return 0.0;
  }
  return squared_penalty(mem.D, anchor_D_, fisher_D_) + squared_penalty(infer.enc_w, anchor_enc_w_, fisher_enc_w_);
}

void EwcRegularizer::apply_pull(CyphaDifMemoryState& mem, CyphaInferModel& infer, double ewc_lambda,
                                double lr) const {
  if (!has_snapshot() || ewc_lambda <= 0.0 || lr <= 0.0) {
    return;
  }
  const double strength = ewc_lambda * lr;
  pull_toward_anchor(mem.D, anchor_D_, fisher_D_, strength);
  pull_toward_anchor(infer.enc_w, anchor_enc_w_, fisher_enc_w_, strength);
}

}  // namespace cypha
