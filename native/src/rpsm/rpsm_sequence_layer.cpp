#include "cypha/rpsm/rpsm_sequence_layer.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace cypha::rpsm {

namespace {

constexpr double kEps = 1e-12;

void init_psi_matrices(PsiMatrices& psi, const RpsmSequenceConfig& cfg, std::mt19937_64& rng) {
  const int d = cfg.feat_dim;
  const int k = cfg.n_classes;
  psi.feat_dim = d;
  psi.n_classes = k;
  psi.mu.assign(static_cast<std::size_t>((1 + k) * d), 0.0);
  psi.inv_var.assign(static_cast<std::size_t>(d), 1.0);
  psi.counts.assign(static_cast<std::size_t>(k), 1.0);
  psi.v_mean = 1.0;
  std::normal_distribution<double> nd(0.0, 0.05);
  for (int j = 0; j < d; ++j) {
    psi.mu[static_cast<std::size_t>(j)] = nd(rng);
  }
  for (int c = 0; c < k; ++c) {
    for (int j = 0; j < d; ++j) {
      psi.mu[static_cast<std::size_t>((1 + c) * d + j)] = nd(rng);
    }
  }
}

}  // namespace

RpsmSequenceLayer::RpsmSequenceLayer(RpsmSequenceConfig cfg) : cfg_(cfg) {
  std::mt19937_64 rng(cfg_.seed);
  init_psi_matrices(psi_, cfg_, rng);
  h_.assign(static_cast<std::size_t>(cfg_.state_dim), 0.0);
  psi_rows_.assign(static_cast<std::size_t>(cfg_.n_levels * cfg_.state_dim), 0.0);
  std::normal_distribution<double> nd(0.0, 0.02);
  w_enc_.assign(static_cast<std::size_t>(cfg_.feat_dim * cfg_.state_dim), 0.0);
  for (auto& v : w_enc_) v = nd(rng);
  w_carry_.assign(static_cast<std::size_t>(cfg_.feat_dim * cfg_.state_dim), 0.0);
  for (auto& v : w_carry_) v = nd(rng);
  feat_buf_.assign(static_cast<std::size_t>(cfg_.feat_dim), 0.0);
  llr_buf_.assign(static_cast<std::size_t>(cfg_.n_classes), 0.0);
}

void RpsmSequenceLayer::reset() {
  std::fill(h_.begin(), h_.end(), 0.0);
  std::fill(psi_rows_.begin(), psi_rows_.end(), 0.0);
}

void RpsmSequenceLayer::log_softmax_row(const double* logits, int k, double* log_out) {
  double max_z = logits[0];
  for (int i = 1; i < k; ++i) {
    max_z = std::max(max_z, logits[i]);
  }
  double sum = 0.0;
  for (int i = 0; i < k; ++i) {
    sum += std::exp(logits[i] - max_z);
  }
  const double log_z = max_z + std::log(sum + kEps);
  for (int i = 0; i < k; ++i) {
    log_out[i] = logits[i] - log_z;
  }
}

double RpsmSequenceLayer::step(const double* input, int input_dim, double* log_probs_out) {
  const int d = cfg_.feat_dim;
  const int sd = cfg_.state_dim;
  const int in_n = std::max(0, input_dim);

  std::fill(feat_buf_.begin(), feat_buf_.end(), 0.0);
  for (int j = 0; j < d; ++j) {
    double acc = 0.0;
    for (int i = 0; i < in_n && i < sd; ++i) {
      acc += w_enc_[static_cast<std::size_t>(j * sd + i)] * input[i];
    }
    for (int i = 0; i < sd; ++i) {
      acc += w_carry_[static_cast<std::size_t>(j * sd + i)] * h_[static_cast<std::size_t>(i)];
    }
    feat_buf_[static_cast<std::size_t>(j)] = std::tanh(acc);
  }

  batched_llr_gemm(feat_buf_.data(), 1, psi_, nullptr, llr_buf_.data());
  log_softmax_row(llr_buf_.data(), cfg_.n_classes, log_probs_out);

  const double a = cfg_.alpha_carry;
  for (int i = 0; i < sd; ++i) {
    double acc = (1.0 - a) * h_[static_cast<std::size_t>(i)];
    for (int j = 0; j < d; ++j) {
      acc += a * w_carry_[static_cast<std::size_t>(j * sd + i)] * feat_buf_[static_cast<std::size_t>(j)] /
             static_cast<double>(d);
    }
    h_[static_cast<std::size_t>(i)] = std::tanh(acc);
  }

  if (!psi_rows_.empty()) {
    const int copy = std::min(sd, cfg_.state_dim);
    for (int j = 0; j < copy; ++j) {
      psi_rows_[static_cast<std::size_t>(j)] = h_[static_cast<std::size_t>(j)];
    }
  }

  double norm_sq = 0.0;
  for (double v : h_) {
    norm_sq += v * v;
  }
  return std::sqrt(norm_sq + kEps);
}

}  // namespace cypha::rpsm
