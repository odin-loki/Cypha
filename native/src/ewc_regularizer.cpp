#include "cypha/ewc_regularizer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace cypha {

namespace {

constexpr double kFisherEps = 1e-8;
constexpr double kMinVar = 1e-4;

void build_diagonal_fisher(const std::vector<double>& anchor, std::vector<double>& fisher_out) {
  fisher_out.resize(anchor.size());
  for (std::size_t i = 0; i < anchor.size(); ++i) {
    fisher_out[i] = anchor[i] * anchor[i] + kFisherEps;
  }
}

// Real diagonal Fisher: F_i ~= E_x[(d loss / d theta_i)^2] over the calibration batch, using the
// *actual* closed-form loss/update math this codebase already computes elsewhere:
//
// 1. D (class deltas): `CyphaDifMemoryState::memory_train` (native/src/memory_train.cpp) returns
//    the scalar loss `-log_norm + 0.5*r.h_mu0 - cross_k + 0.5*d_sq_k`, where (for the true-label
//    row k) `cross_k = sum_j D[k,j]*r[j]` and `d_sq_k = sum_j D[k,j]^2*world_inv_v[j]`, with
//    `r[j] = (h[j]-mu0[j])*world_inv_v[j]`. Only the true-label row k appears in that formula, so
//    `d(loss)/d(D[k,j]) = D[k,j]*world_inv_v[j] - r[j]`, and 0 for every other row (matches the
//    codebase's own convention of only crediting the true label's row with this loss term).
//    `mu0 = world_mu` exactly here because `f_field` is all-zero for every fresh model this class
//    is used with (native/src/create_model.cpp) and is never mutated during training — the field
//    correction term `memory_train` would otherwise add is provably zero, not merely ignored.
//
// 2. enc_w (encoder projection): `contrastive_update_encoder_w` (native/src/encoder_contrastive.cpp)
//    documents itself as a "Fisher-Rao-style encoder update": `W[i,j] += lr*weight*diff[i]*f[j]`
//    where `diff = rj - rk` are Fisher-Rao residuals `(h-mu)/max(v,eps)` for the predicted (`j`)
//    and true (`k`) classes. The per-step contribution *before* scaling by `lr` — i.e.
//    `weight*diff[i]*f[j]` — is exactly this update's gradient direction, so it is reused directly
//    as `d(loss)/d(W[i,j])` for samples where that update actually fires (`pred != true`, the same
//    condition `dif_train_step_vector` uses); correctly-classified samples contribute 0, matching
//    that no encoder update happens for them.
//
// This does not introduce any new autodiff machinery — every derivative below is the closed form
// already implied by the existing hand-written update rules; this function just squares and
// accumulates it over a calibration batch instead of applying it as a single training step.
void accumulate_diagonal_fisher_from_calibration(const CyphaDifMemoryState& mem, const CyphaInferModel& infer,
                                                 const std::vector<std::vector<double>>& calib_x,
                                                 const std::vector<std::string>& calib_labels,
                                                 std::vector<double>& fisher_D_out,
                                                 std::vector<double>& fisher_enc_w_out) {
  const int d = mem.d_latent;
  const int K = static_cast<int>(mem.labels.size());
  fisher_D_out.assign(mem.D.size(), 0.0);
  fisher_enc_w_out.assign(infer.enc_w.size(), 0.0);
  if (d <= 0 || K == 0 || calib_x.empty()) {
    return;
  }

  std::vector<double> ctx_vec;
  context_prior_for_labels(infer, mem.labels, ctx_vec);

  int n_used = 0;
  std::vector<double> h(static_cast<std::size_t>(d));
  for (std::size_t s = 0; s < calib_x.size() && s < calib_labels.size(); ++s) {
    if (static_cast<int>(calib_x[s].size()) != d) {
      continue;
    }
    auto it_k = mem.label_index.find(calib_labels[s]);
    if (it_k == mem.label_index.end()) {
      continue;  // Fisher only estimated for already-known classes (matches memory_train's
                 // per-row loss attribution — a brand-new class has no D row yet to differentiate).
    }
    const int k_idx = it_k->second;
    batch_encode(infer, calib_x[s].data(), 1, h);

    // r[j] = (h - world_mu) * world_inv_v; mu0 == world_mu exactly (see derivation above).
    std::vector<double> r(static_cast<std::size_t>(d));
    for (int j = 0; j < d; ++j) {
      r[static_cast<std::size_t>(j)] =
          (h[static_cast<std::size_t>(j)] - mem.world_mu[static_cast<std::size_t>(j)]) *
          mem.world_inv_v[static_cast<std::size_t>(j)];
    }

    // d(loss)/d(D[k,j]) for the true-label row.
    for (int j = 0; j < d; ++j) {
      const double Dkj = mem.D[static_cast<std::size_t>(k_idx * d + j)];
      const double g = Dkj * mem.world_inv_v[static_cast<std::size_t>(j)] - r[static_cast<std::size_t>(j)];
      fisher_D_out[static_cast<std::size_t>(k_idx * d + j)] += g * g;
    }

    // Predicted class under the current (D, world) state, exactly as memory_train computes it,
    // to decide whether this sample would trigger an encoder contrastive update.
    int best_idx = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (int k = 0; k < K; ++k) {
      double cross = 0.0;
      double d_sq = 0.0;
      for (int j = 0; j < d; ++j) {
        const double Dkj = mem.D[static_cast<std::size_t>(k * d + j)];
        cross += Dkj * r[static_cast<std::size_t>(j)];
        d_sq += Dkj * Dkj * mem.world_inv_v[static_cast<std::size_t>(j)];
      }
      const double ctx = (k < static_cast<int>(ctx_vec.size())) ? ctx_vec[static_cast<std::size_t>(k)] : 0.0;
      const double score = cross - 0.5 * d_sq + ctx;
      if (score > best_score) {
        best_score = score;
        best_idx = k;
      }
    }

    if (best_idx != k_idx) {
      std::vector<double> mu_true;
      std::vector<double> v_true;
      std::vector<double> mu_pred;
      std::vector<double> v_pred;
      if (mem.class_mean_and_variance(calib_labels[s], mu_true, v_true) &&
          mem.class_mean_and_variance(mem.labels[static_cast<std::size_t>(best_idx)], mu_pred, v_pred) &&
          static_cast<int>(infer.enc_w.size()) == d * d) {
        std::vector<double> r_true(static_cast<std::size_t>(d));
        std::vector<double> r_pred(static_cast<std::size_t>(d));
        for (int j = 0; j < d; ++j) {
          r_true[static_cast<std::size_t>(j)] =
              (h[static_cast<std::size_t>(j)] - mu_true[static_cast<std::size_t>(j)]) /
              std::max(v_true[static_cast<std::size_t>(j)], kMinVar);
          r_pred[static_cast<std::size_t>(j)] =
              (h[static_cast<std::size_t>(j)] - mu_pred[static_cast<std::size_t>(j)]) /
              std::max(v_pred[static_cast<std::size_t>(j)], kMinVar);
        }
        for (int i = 0; i < d; ++i) {
          const double diff = r_pred[static_cast<std::size_t>(i)] - r_true[static_cast<std::size_t>(i)];
          for (int j = 0; j < d; ++j) {
            const double g = diff * h[static_cast<std::size_t>(j)];
            fisher_enc_w_out[static_cast<std::size_t>(i * d + j)] += g * g;
          }
        }
      }
    }
    ++n_used;
  }

  const double denom = static_cast<double>(std::max(n_used, 1));
  for (double& v : fisher_D_out) {
    v = v / denom + kFisherEps;
  }
  for (double& v : fisher_enc_w_out) {
    v = v / denom + kFisherEps;
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

void EwcRegularizer::snapshot_calibrated(const CyphaDifMemoryState& mem, const CyphaInferModel& infer,
                                         const std::vector<std::vector<double>>& calib_x,
                                         const std::vector<std::string>& calib_labels) {
  anchor_D_ = mem.D;
  anchor_enc_w_ = infer.enc_w;
  if (calib_x.empty()) {
    build_diagonal_fisher(anchor_D_, fisher_D_);
    build_diagonal_fisher(anchor_enc_w_, fisher_enc_w_);
    return;
  }
  accumulate_diagonal_fisher_from_calibration(mem, infer, calib_x, calib_labels, fisher_D_, fisher_enc_w_);
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
