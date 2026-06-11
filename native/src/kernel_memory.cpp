#include "cypha/kernel_memory.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cypha {

namespace {

constexpr double kEps = 1e-8;

double dot(const double* a, const double* b, int n) {
  double s = 0.0;
  for (int i = 0; i < n; ++i) {
    s += a[i] * b[i];
  }
  return s;
}

}  // namespace

KernelMemory::KernelMemory(int feat_dim, int M, std::uint64_t rng_seed)
    : feat_dim_(feat_dim),
      M_(M),
      gamma_(1.0 / static_cast<double>(std::max(feat_dim, 1))),
      basis_(static_cast<std::size_t>(M) * static_cast<std::size_t>(feat_dim), 0.0),
      rng_(static_cast<std::uint32_t>(rng_seed & 0xffffffffu)) {}

void KernelMemory::load_state(int n_basis, int n_seen, const double* basis_row_major, int basis_rows,
                              const std::map<std::string, std::vector<double>>& weights) {
  if (basis_rows != M_ || n_basis < 0 || n_basis > M_) {
    throw std::runtime_error("KernelMemory::load_state basis shape mismatch");
  }
  n_basis_ = n_basis;
  n_seen_ = n_seen;
  const std::size_t need = static_cast<std::size_t>(M_) * static_cast<std::size_t>(feat_dim_);
  basis_.assign(need, 0.0);
  for (std::size_t i = 0; i < need; ++i) {
    basis_[i] = basis_row_major[i];
  }
  weights_ = weights;
  for (const auto& pr : weights_) {
    if (static_cast<int>(pr.second.size()) != M_) {
      throw std::runtime_error("KernelMemory::load_state weight length mismatch");
    }
  }
}

void KernelMemory::phi(const double* h, std::vector<double>& out) const {
  out.assign(static_cast<std::size_t>(M_), 0.0);
  const int nb = n_basis_;
  if (nb <= 0) {
    return;
  }
  for (int i = 0; i < nb; ++i) {
    double sq = 0.0;
    const double* bi = basis_.data() + static_cast<std::size_t>(i * feat_dim_);
    for (int j = 0; j < feat_dim_; ++j) {
      const double diff = bi[j] - h[j];
      sq += diff * diff;
    }
    out[static_cast<std::size_t>(i)] = std::exp(-gamma_ * sq);
  }
}

void KernelMemory::score_all(const double* h, const std::vector<std::string>& labels,
                             std::vector<double>& scores) const {
  const int K = static_cast<int>(labels.size());
  scores.assign(static_cast<std::size_t>(K), 0.0);
  if (n_basis_ < 4) {
    return;
  }
  std::vector<double> phi_vec;
  phi(h, phi_vec);
  for (int i = 0; i < K; ++i) {
    auto it = weights_.find(labels[static_cast<std::size_t>(i)]);
    if (it == weights_.end()) {
      continue;
    }
    const std::vector<double>& w = it->second;
    const double wp = dot(w.data(), phi_vec.data(), M_);
    const double wn = dot(w.data(), w.data(), M_);
    scores[static_cast<std::size_t>(i)] = wp - 0.5 * wn;
  }
}

void KernelMemory::reservoir_update(const double* h, std::optional<int> fixed_j) {
  n_seen_ += 1;
  if (n_basis_ < M_) {
    double* slot = basis_.data() + static_cast<std::size_t>(n_basis_ * feat_dim_);
    for (int j = 0; j < feat_dim_; ++j) {
      slot[j] = h[j];
    }
    n_basis_ += 1;
    return;
  }
  int j = 0;
  if (fixed_j.has_value()) {
    j = *fixed_j;
  } else {
    std::uniform_int_distribution<int> dist(0, n_seen_ - 1);
    j = dist(rng_);
  }
  if (j < M_) {
    double* slot = basis_.data() + static_cast<std::size_t>(j * feat_dim_);
    for (int d = 0; d < feat_dim_; ++d) {
      slot[d] = h[d];
    }
  }
}

void KernelMemory::update(const double* h, const std::string& label, const std::vector<std::string>& all_labels,
                          double lr, std::optional<int> fixed_reservoir_j) {
  reservoir_update(h, fixed_reservoir_j);
  if (n_basis_ < 4) {
    return;
  }

  for (const auto& lbl : all_labels) {
    if (weights_.find(lbl) == weights_.end()) {
      weights_[lbl] = std::vector<double>(static_cast<std::size_t>(M_), 0.0);
    }
  }

  std::vector<double> phi_vec;
  phi(h, phi_vec);

  const int K = static_cast<int>(all_labels.size());
  std::vector<double> raw(static_cast<std::size_t>(K), 0.0);
  for (int i = 0; i < K; ++i) {
    const auto& w = weights_.at(all_labels[static_cast<std::size_t>(i)]);
    raw[static_cast<std::size_t>(i)] = dot(w.data(), phi_vec.data(), M_);
  }

  double mx = raw[0];
  for (int i = 1; i < K; ++i) {
    mx = std::max(mx, raw[i]);
  }
  double sum = 0.0;
  std::vector<double> probs(static_cast<std::size_t>(K), 0.0);
  for (int i = 0; i < K; ++i) {
    probs[static_cast<std::size_t>(i)] = std::exp(raw[static_cast<std::size_t>(i)] - mx);
    sum += probs[static_cast<std::size_t>(i)];
  }
  sum += kEps;
  for (int i = 0; i < K; ++i) {
    probs[static_cast<std::size_t>(i)] /= sum;
  }

  for (int i = 0; i < K; ++i) {
    const std::string& lbl = all_labels[static_cast<std::size_t>(i)];
    const double target = (lbl == label) ? 1.0 : 0.0;
    std::vector<double>& w = weights_.at(lbl);
    for (int m = 0; m < M_; ++m) {
      w[static_cast<std::size_t>(m)] += lr * (target - probs[static_cast<std::size_t>(i)]) * phi_vec[m];
    }
  }
}

}  // namespace cypha
