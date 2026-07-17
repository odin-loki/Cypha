#include "cypha/kernel_memory.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "cypha/load_cypha.hpp"
#include "cypha/rff_features.hpp"

namespace cypha {

namespace {

constexpr double kEps = 1e-8;
constexpr double kRidge = 1e-6;
constexpr int kCholeskyRetries = 6;

double dot(const double* a, const double* b, int n) {
  double s = 0.0;
  for (int i = 0; i < n; ++i) {
    s += a[i] * b[i];
  }
  return s;
}

double sq_dist(const double* a, const double* b, int n) {
  double s = 0.0;
  for (int i = 0; i < n; ++i) {
    const double d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

double rbf(const double* a, const double* b, int n, double gamma) {
  return std::exp(-gamma * sq_dist(a, b, n));
}

bool cholesky_lower(const double* sym_row_major, int n, double* L_row_major) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      L_row_major[i * n + j] = 0.0;
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      double sum = sym_row_major[i * n + j];
      for (int k = 0; k < j; ++k) {
        sum -= L_row_major[i * n + k] * L_row_major[j * n + k];
      }
      if (i == j) {
        if (sum <= kEps) {
          return false;
        }
        L_row_major[i * n + j] = std::sqrt(sum);
      } else {
        L_row_major[i * n + j] = sum / L_row_major[j * n + j];
      }
    }
  }
  return true;
}

void invert_lower_triangular(const double* L, int n, double* Linv) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      Linv[i * n + j] = 0.0;
    }
  }
  for (int i = 0; i < n; ++i) {
    Linv[i * n + i] = 1.0 / L[i * n + i];
    for (int j = 0; j < i; ++j) {
      double sum = 0.0;
      for (int k = j; k < i; ++k) {
        sum += L[i * n + k] * Linv[k * n + j];
      }
      Linv[i * n + j] = -sum / L[i * n + i];
    }
  }
}

void transpose(const double* A, int n, double* At) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      At[j * n + i] = A[i * n + j];
    }
  }
}

bool sym_eigh_whitening(const double* sym_row_major, int n, double ridge, double* whiten_row_major) {
  std::vector<double> a(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      a[static_cast<std::size_t>(i * n + j)] = sym_row_major[i * n + j];
    }
    a[static_cast<std::size_t>(i * n + i)] += ridge;
  }
  std::vector<double> evecs(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      evecs[static_cast<std::size_t>(i * n + j)] = (i == j) ? 1.0 : 0.0;
    }
  }
  for (int sweep = 0; sweep < 40; ++sweep) {
    double off = 0.0;
    for (int p = 0; p < n; ++p) {
      for (int q = p + 1; q < n; ++q) {
        off += std::abs(a[static_cast<std::size_t>(p * n + q)]);
      }
    }
    if (off <= 1e-12 * static_cast<double>(n * n)) {
      break;
    }
    for (int p = 0; p < n; ++p) {
      for (int q = p + 1; q < n; ++q) {
        const double apq = a[static_cast<std::size_t>(p * n + q)];
        if (std::abs(apq) <= 1e-15) {
          continue;
        }
        const double app = a[static_cast<std::size_t>(p * n + p)];
        const double aqq = a[static_cast<std::size_t>(q * n + q)];
        const double tau = (aqq - app) / (2.0 * apq);
        const double t = (tau >= 0.0) ? 1.0 / (tau + std::sqrt(1.0 + tau * tau))
                                       : -1.0 / (-tau + std::sqrt(1.0 + tau * tau));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = t * c;
        for (int k = 0; k < n; ++k) {
          const double akp = a[static_cast<std::size_t>(k * n + p)];
          const double akq = a[static_cast<std::size_t>(k * n + q)];
          a[static_cast<std::size_t>(k * n + p)] = c * akp - s * akq;
          a[static_cast<std::size_t>(p * n + k)] = a[static_cast<std::size_t>(k * n + p)];
          a[static_cast<std::size_t>(k * n + q)] = s * akp + c * akq;
          a[static_cast<std::size_t>(q * n + k)] = a[static_cast<std::size_t>(k * n + q)];
        }
        a[static_cast<std::size_t>(p * n + q)] = 0.0;
        a[static_cast<std::size_t>(q * n + p)] = 0.0;
        for (int k = 0; k < n; ++k) {
          const double vkp = evecs[static_cast<std::size_t>(k * n + p)];
          const double vkq = evecs[static_cast<std::size_t>(k * n + q)];
          evecs[static_cast<std::size_t>(k * n + p)] = c * vkp - s * vkq;
          evecs[static_cast<std::size_t>(k * n + q)] = s * vkp + c * vkq;
        }
      }
    }
  }
  const double floor = std::max(ridge, 1e-10);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      whiten_row_major[i * n + j] = 0.0;
    }
  }
  for (int k = 0; k < n; ++k) {
    const double lam = std::max(a[static_cast<std::size_t>(k * n + k)], floor);
    const double inv_sqrt = 1.0 / std::sqrt(lam);
    for (int i = 0; i < n; ++i) {
      const double vik = evecs[static_cast<std::size_t>(i * n + k)];
      for (int j = 0; j < n; ++j) {
        whiten_row_major[i * n + j] += vik * evecs[static_cast<std::size_t>(j * n + k)] * inv_sqrt;
      }
    }
  }
  return true;
}

}  // namespace

KernelMemory::KernelMemory(int feat_dim, int M, std::uint64_t rng_seed)
    : feat_dim_(feat_dim),
      M_(M),
      gamma_(1.0 / static_cast<double>(std::max(feat_dim, 1))),
      basis_(static_cast<std::size_t>(M) * static_cast<std::size_t>(feat_dim), 0.0),
      rng_(static_cast<std::uint32_t>(rng_seed & 0xffffffffu)) {}

KernelMemory KernelMemory::make_rff(int feat_dim, int M, double gamma, std::uint64_t rng_seed,
                                    RffProjectionKind projection) {
  KernelMemory km(feat_dim, M, rng_seed);
  km.rff_mode_ = true;
  km.rff_projection_ = projection;
  km.gamma_ = gamma;
  km.n_basis_ = M;
  km.n_seen_ = 0;
  std::mt19937 rng(static_cast<std::uint32_t>((rng_seed ^ 0x9e3779b97f4a7c15ull) & 0xffffffffu));
  init_rff_weights(projection, rng, gamma, M, feat_dim, km.rff_w_, km.rff_b_, true);
  return km;
}

double KernelMemory::auto_gamma_median_heuristic(const double* samples_row_major, int n, int feat_dim,
                                                 double gamma_scale, int max_samples, std::uint64_t rng_seed) {
  const double fallback = gamma_scale / static_cast<double>(std::max(feat_dim, 1));
  if (n < 2 || feat_dim <= 0) {
    return fallback;
  }
  std::vector<int> idx(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    idx[static_cast<std::size_t>(i)] = i;
  }
  if (n > max_samples) {
    std::mt19937 rng(static_cast<std::uint32_t>(rng_seed & 0xffffffffu));
    std::shuffle(idx.begin(), idx.end(), rng);
    idx.resize(static_cast<std::size_t>(max_samples));
  }
  const int ns = static_cast<int>(idx.size());
  std::vector<double> sq_dists;
  sq_dists.reserve(static_cast<std::size_t>(ns) * static_cast<std::size_t>(ns - 1) / 2);
  for (int i = 0; i < ns; ++i) {
    const double* a = samples_row_major + static_cast<std::size_t>(idx[static_cast<std::size_t>(i)]) *
                                              static_cast<std::size_t>(feat_dim);
    for (int j = i + 1; j < ns; ++j) {
      const double* b = samples_row_major + static_cast<std::size_t>(idx[static_cast<std::size_t>(j)]) *
                                                static_cast<std::size_t>(feat_dim);
      sq_dists.push_back(sq_dist(a, b, feat_dim));
    }
  }
  if (sq_dists.empty()) {
    return fallback;
  }
  const std::size_t mid = sq_dists.size() / 2;
  std::nth_element(sq_dists.begin(), sq_dists.begin() + static_cast<std::ptrdiff_t>(mid), sq_dists.end());
  const double med = sq_dists[mid];
  return gamma_scale / (2.0 * std::max(med, kEps));
}

double ridge_leverage_score(const double* h, const double* basis, int n_basis, int feat_dim, double gamma,
                            double ridge) {
  if (n_basis <= 0) {
    return 1.0;
  }
  std::vector<double> k_hb(static_cast<std::size_t>(n_basis), 0.0);
  for (int i = 0; i < n_basis; ++i) {
    const double* bi = basis + static_cast<std::size_t>(i * feat_dim);
    k_hb[static_cast<std::size_t>(i)] = rbf(bi, h, feat_dim, gamma);
  }
  std::vector<double> K_bb(static_cast<std::size_t>(n_basis) * static_cast<std::size_t>(n_basis), 0.0);
  for (int i = 0; i < n_basis; ++i) {
    const double* bi = basis + static_cast<std::size_t>(i * feat_dim);
    for (int j = 0; j <= i; ++j) {
      const double* bj = basis + static_cast<std::size_t>(j * feat_dim);
      const double kij = (i == j) ? 1.0 : rbf(bi, bj, feat_dim, gamma);
      K_bb[static_cast<std::size_t>(i * n_basis + j)] = kij;
      K_bb[static_cast<std::size_t>(j * n_basis + i)] = kij;
    }
    K_bb[static_cast<std::size_t>(i * n_basis + i)] += ridge;
  }
  std::vector<double> L(static_cast<std::size_t>(n_basis) * static_cast<std::size_t>(n_basis), 0.0);
  if (!cholesky_lower(K_bb.data(), n_basis, L.data())) {
    return 1.0;
  }
  std::vector<double> sol(static_cast<std::size_t>(n_basis), 0.0);
  for (int i = 0; i < n_basis; ++i) {
    double s = k_hb[static_cast<std::size_t>(i)];
    for (int k = 0; k < i; ++k) {
      s -= L[static_cast<std::size_t>(i * n_basis + k)] * sol[static_cast<std::size_t>(k)];
    }
    sol[static_cast<std::size_t>(i)] = s / L[static_cast<std::size_t>(i * n_basis + i)];
  }
  for (int i = n_basis - 1; i >= 0; --i) {
    double s = sol[static_cast<std::size_t>(i)];
    for (int k = i + 1; k < n_basis; ++k) {
      s -= L[static_cast<std::size_t>(k * n_basis + i)] * sol[static_cast<std::size_t>(k)];
    }
    sol[static_cast<std::size_t>(i)] = s / L[static_cast<std::size_t>(i * n_basis + i)];
  }
  double schur = 1.0;
  for (int i = 0; i < n_basis; ++i) {
    schur -= k_hb[static_cast<std::size_t>(i)] * sol[static_cast<std::size_t>(i)];
  }
  return std::max(schur, 1e-12);
}

void KernelMemory::init_leverage_landmarks_from_samples(const double* samples_row_major, int n, int feat_dim) {
  if (rff_mode_) {
    return;
  }
  if (feat_dim != feat_dim_) {
    throw std::runtime_error("KernelMemory::init_leverage_landmarks_from_samples feat_dim mismatch");
  }
  if (n <= 0) {
    return;
  }
  gamma_ = auto_gamma_median_heuristic(samples_row_major, n, feat_dim, gamma_scale_, 256,
                                     static_cast<std::uint64_t>(rng_()));
  std::vector<int> idx;
  select_leverage_landmark_indices(samples_row_major, n, feat_dim, M_, gamma_, kRidge,
                                   static_cast<std::uint64_t>(rng_()), idx);
  n_basis_ = static_cast<int>(idx.size());
  n_seen_ = n;
  basis_.assign(static_cast<std::size_t>(M_) * static_cast<std::size_t>(feat_dim_), 0.0);
  for (int i = 0; i < n_basis_; ++i) {
    const double* src = samples_row_major + static_cast<std::size_t>(idx[static_cast<std::size_t>(i)]) *
                                              static_cast<std::size_t>(feat_dim);
    double* dest = basis_.data() + static_cast<std::size_t>(i * feat_dim_);
    for (int j = 0; j < feat_dim_; ++j) {
      dest[j] = src[j];
    }
  }
  recompute_nystrom();
}

void KernelMemory::recompute_nystrom() {
  const int nb = n_basis_;
  whitening_.clear();
  if (nb <= 0) {
    return;
  }

  if (nb >= 2) {
    std::vector<double> sq_dists;
    sq_dists.reserve(static_cast<std::size_t>(nb) * static_cast<std::size_t>(nb - 1) / 2);
    for (int i = 0; i < nb; ++i) {
      const double* bi = basis_.data() + static_cast<std::size_t>(i * feat_dim_);
      for (int j = i + 1; j < nb; ++j) {
        const double* bj = basis_.data() + static_cast<std::size_t>(j * feat_dim_);
        sq_dists.push_back(sq_dist(bi, bj, feat_dim_));
      }
    }
    if (!sq_dists.empty()) {
      const std::size_t mid = sq_dists.size() / 2;
      std::nth_element(sq_dists.begin(), sq_dists.begin() + static_cast<std::ptrdiff_t>(mid),
                       sq_dists.end());
      const double med = sq_dists[mid];
      gamma_ = gamma_scale_ / (2.0 * std::max(med, kEps));
    }
  } else {
    gamma_ = gamma_scale_ / static_cast<double>(std::max(feat_dim_, 1));
  }

  std::vector<double> K_base(static_cast<std::size_t>(nb) * static_cast<std::size_t>(nb), 0.0);
  for (int i = 0; i < nb; ++i) {
    const double* bi = basis_.data() + static_cast<std::size_t>(i * feat_dim_);
    for (int j = 0; j <= i; ++j) {
      const double* bj = basis_.data() + static_cast<std::size_t>(j * feat_dim_);
      const double kij = (i == j) ? 1.0 : rbf(bi, bj, feat_dim_, gamma_);
      K_base[static_cast<std::size_t>(i * nb + j)] = kij;
      K_base[static_cast<std::size_t>(j * nb + i)] = kij;
    }
  }

  double ridge = kRidge;
  bool whitened = false;
  for (int attempt = 0; attempt < kCholeskyRetries; ++attempt) {
    std::vector<double> K = K_base;
    for (int i = 0; i < nb; ++i) {
      K[static_cast<std::size_t>(i * nb + i)] += ridge;
    }
    std::vector<double> L(static_cast<std::size_t>(nb) * static_cast<std::size_t>(nb), 0.0);
    if (!cholesky_lower(K.data(), nb, L.data())) {
      ridge *= 10.0;
      continue;
    }
    std::vector<double> Linv(static_cast<std::size_t>(nb) * static_cast<std::size_t>(nb), 0.0);
    invert_lower_triangular(L.data(), nb, Linv.data());
    whitening_.assign(static_cast<std::size_t>(nb) * static_cast<std::size_t>(nb), 0.0);
    transpose(Linv.data(), nb, whitening_.data());
    whitened = true;
    break;
  }
  if (whitened) {
    return;
  }

  whitening_.assign(static_cast<std::size_t>(nb) * static_cast<std::size_t>(nb), 0.0);
  if (sym_eigh_whitening(K_base.data(), nb, ridge, whitening_.data())) {
    return;
  }
  for (int i = 0; i < nb; ++i) {
    whitening_[static_cast<std::size_t>(i * nb + i)] = 1.0;
  }
}

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
  recompute_nystrom();
}

void KernelMemory::phi(const double* h, std::vector<double>& out) const {
  out.assign(static_cast<std::size_t>(M_), 0.0);
  if (rff_mode_) {
    const double scale = std::sqrt(2.0 / static_cast<double>(std::max(M_, 1)));
    for (int i = 0; i < M_; ++i) {
      double s = rff_b_[static_cast<std::size_t>(i)];
      const double* wi = rff_w_.data() + static_cast<std::size_t>(i * feat_dim_);
      s += dot(wi, h, feat_dim_);
      out[static_cast<std::size_t>(i)] = scale * std::cos(s);
    }
    return;
  }
  const int nb = n_basis_;
  if (nb <= 0) {
    return;
  }

  std::vector<double> k_hm(static_cast<std::size_t>(nb), 0.0);
  for (int i = 0; i < nb; ++i) {
    const double* bi = basis_.data() + static_cast<std::size_t>(i * feat_dim_);
    k_hm[static_cast<std::size_t>(i)] = rbf(bi, h, feat_dim_, gamma_);
  }

  if (static_cast<int>(whitening_.size()) == nb * nb) {
    for (int j = 0; j < nb; ++j) {
      double s = 0.0;
      for (int i = 0; i < nb; ++i) {
        s += k_hm[static_cast<std::size_t>(i)] * whitening_[static_cast<std::size_t>(i * nb + j)];
      }
      out[static_cast<std::size_t>(j)] = s;
    }
  } else {
    for (int i = 0; i < nb; ++i) {
      out[static_cast<std::size_t>(i)] = k_hm[static_cast<std::size_t>(i)];
    }
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
  if (rff_mode_) {
    // Fixed random projection — no landmark reservoir to maintain.
    return;
  }
  if (n_basis_ < M_) {
    double* slot = basis_.data() + static_cast<std::size_t>(n_basis_ * feat_dim_);
    for (int j = 0; j < feat_dim_; ++j) {
      slot[j] = h[j];
    }
    n_basis_ += 1;
    recompute_nystrom();
    return;
  }
  if (landmark_sampling_ == LandmarkSamplingKind::LeverageScore && n_basis_ >= M_) {
    const double lev_h = ridge_leverage_score(h, basis_.data(), n_basis_, feat_dim_, gamma_, kRidge);
    int min_slot = 0;
    double min_lev = ridge_leverage_score(basis_.data(), basis_.data(), n_basis_, feat_dim_, gamma_, kRidge);
    for (int i = 1; i < M_; ++i) {
      const double* bi = basis_.data() + static_cast<std::size_t>(i * feat_dim_);
      const double lev_i = ridge_leverage_score(bi, basis_.data(), n_basis_, feat_dim_, gamma_, kRidge);
      if (lev_i < min_lev) {
        min_lev = lev_i;
        min_slot = i;
      }
    }
    if (lev_h > min_lev) {
      double* dest = basis_.data() + static_cast<std::size_t>(min_slot * feat_dim_);
      for (int d = 0; d < feat_dim_; ++d) {
        dest[d] = h[d];
      }
      recompute_nystrom();
    }
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
    int slot = j;
    if (n_basis_ >= M_) {
      slot = 0;
      double best_d = sq_dist(basis_.data(), h, feat_dim_);
      for (int i = 1; i < M_; ++i) {
        const double* bi = basis_.data() + static_cast<std::size_t>(i * feat_dim_);
        const double d2 = sq_dist(bi, h, feat_dim_);
        if (d2 < best_d) {
          best_d = d2;
          slot = i;
        }
      }
    }
    double* dest = basis_.data() + static_cast<std::size_t>(slot * feat_dim_);
    for (int d = 0; d < feat_dim_; ++d) {
      dest[d] = h[d];
    }
    recompute_nystrom();
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

KernelMemory::Snapshot KernelMemory::export_snapshot() const {
  Snapshot snap;
  snap.feat_dim = feat_dim_;
  snap.M = M_;
  snap.gamma = gamma_;
  snap.n_basis = n_basis_;
  snap.n_seen = n_seen_;
  snap.basis_rowmajor = basis_;
  snap.weights = weights_;
  return snap;
}

void KernelMemory::import_snapshot(const Snapshot& snap) {
  if (snap.feat_dim != feat_dim_ || snap.M != M_) {
    throw std::runtime_error("KernelMemory::import_snapshot shape mismatch");
  }
  load_state(snap.n_basis, snap.n_seen, snap.basis_rowmajor.data(), M_, snap.weights);
}

namespace {

double cypha_node_as_double(const CNode& n) {
  if (n.kind == CNode::Float) {
    return n.f;
  }
  if (n.kind == CNode::Int) {
    return static_cast<double>(n.i);
  }
  throw std::runtime_error("kernel cypha: expected numeric node");
}

CNode cypha_node_f64(double v) {
  CNode n;
  n.kind = CNode::Float;
  n.f = v;
  return n;
}

CNode cypha_node_i64(std::int64_t v) {
  CNode n;
  n.kind = CNode::Int;
  n.i = v;
  return n;
}

CNode cypha_node_bool(bool v) {
  CNode n;
  n.kind = CNode::Bool;
  n.b = v;
  return n;
}

CNode cypha_tensor_1d(const std::vector<double>& data) {
  CNode t;
  t.kind = CNode::Tensor;
  t.shape = {static_cast<std::uint32_t>(data.size())};
  t.tensor = data;
  return t;
}

CNode cypha_tensor_2d_rowmajor(const std::vector<double>& data, int rows, int cols) {
  CNode t;
  t.kind = CNode::Tensor;
  t.shape = {static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(cols)};
  t.tensor = data;
  return t;
}

void root_map_assign(std::vector<std::pair<std::string, CNode>>& map, std::string key, CNode val) {
  for (auto& kv : map) {
    if (kv.first == key) {
      kv.second = std::move(val);
      return;
    }
  }
  map.emplace_back(std::move(key), std::move(val));
}

}  // namespace

void patch_kernel_into_root(CNode& root, const KernelMemory& km, bool use_kernel_llr, double kernel_blend) {
  if (root.kind != CNode::Map) {
    throw std::runtime_error("patch_kernel_into_root: root must be map");
  }
  root_map_assign(root.map, "use_kernel_llr", cypha_node_bool(use_kernel_llr));
  root_map_assign(root.map, "kernel_blend", cypha_node_f64(kernel_blend));
  if (!use_kernel_llr) {
    return;
  }
  const KernelMemory::Snapshot snap = km.export_snapshot();
  CNode km_node;
  km_node.kind = CNode::Map;
  km_node.map.emplace_back("feat_dim", cypha_node_i64(snap.feat_dim));
  km_node.map.emplace_back("M", cypha_node_i64(snap.M));
  km_node.map.emplace_back("gamma", cypha_node_f64(snap.gamma));
  km_node.map.emplace_back("n_basis", cypha_node_i64(snap.n_basis));
  km_node.map.emplace_back("n_seen", cypha_node_i64(snap.n_seen));
  km_node.map.emplace_back("basis_rowmajor",
                           cypha_tensor_2d_rowmajor(snap.basis_rowmajor, snap.M, snap.feat_dim));
  CNode weights_map;
  weights_map.kind = CNode::Map;
  for (const auto& pr : snap.weights) {
    weights_map.map.emplace_back(pr.first, cypha_tensor_1d(pr.second));
  }
  km_node.map.emplace_back("weights", std::move(weights_map));
  root_map_assign(root.map, "kernel_mem", std::move(km_node));
}

bool try_load_kernel_from_root(const CNode& root, KernelMemory& km, bool& use_kernel_llr_out,
                               double& kernel_blend_out) {
  use_kernel_llr_out = false;
  kernel_blend_out = 0.5;
  if (root.kind != CNode::Map) {
    return false;
  }
  if (const CNode* u = map_get(root, "use_kernel_llr"); u != nullptr && u->kind == CNode::Bool) {
    use_kernel_llr_out = u->b;
  }
  if (const CNode* b = map_get(root, "kernel_blend"); b != nullptr) {
    kernel_blend_out = cypha_node_as_double(*b);
  }
  if (!use_kernel_llr_out) {
    return false;
  }
  const CNode* kmn = map_get(root, "kernel_mem");
  if (kmn == nullptr || kmn->kind != CNode::Map) {
    return false;
  }
  const CNode& st = *kmn;
  const int feat_dim = static_cast<int>(cypha_node_as_double(map_get_required(st, "feat_dim")));
  const int M = static_cast<int>(cypha_node_as_double(map_get_required(st, "M")));
  const int n_basis = static_cast<int>(cypha_node_as_double(map_get_required(st, "n_basis")));
  const int n_seen = static_cast<int>(cypha_node_as_double(map_get_required(st, "n_seen")));
  const CNode& basis_node = map_get_required(st, "basis_rowmajor");
  if (basis_node.kind != CNode::Tensor || static_cast<int>(basis_node.tensor.size()) != M * feat_dim) {
    throw std::runtime_error("try_load_kernel_from_root: basis shape mismatch");
  }
  std::map<std::string, std::vector<double>> weights;
  const CNode& wmap = map_get_required(st, "weights");
  if (wmap.kind != CNode::Map) {
    throw std::runtime_error("try_load_kernel_from_root: weights must be map");
  }
  for (const auto& pr : wmap.map) {
    if (pr.second.kind != CNode::Tensor ||
        static_cast<int>(pr.second.tensor.size()) != M) {
      throw std::runtime_error("try_load_kernel_from_root: weight length mismatch");
    }
    weights[pr.first] = pr.second.tensor;
  }
  if (km.feat_dim() != feat_dim || km.M() != M) {
    km = KernelMemory(feat_dim, M, 0);
  }
  km.load_state(n_basis, n_seen, basis_node.tensor.data(), M, weights);
  return true;
}

std::vector<double> build_xor_pair_features(const double* x, int d) {
  std::vector<double> out(5, 0.0);
  if (d >= 1) {
    out[0] = x[0];
  }
  if (d >= 2) {
    out[1] = x[1];
    out[2] = x[0] * x[1];
    out[3] = x[0] * x[0];
    out[4] = x[1] * x[1];
  }
  return out;
}

}  // namespace cypha
