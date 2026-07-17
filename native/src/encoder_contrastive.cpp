#include "cypha/encoder_contrastive.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include "cypha/mt19937_rng.hpp"

namespace cypha {

namespace {

constexpr double kMinVar = 1e-4;
constexpr double kEncFroCap = 8.0;

void fisher_rao_residual(int d, const double* h, const double* mu, const double* v, std::vector<double>& out) {
  out.resize(static_cast<std::size_t>(d));
  for (int i = 0; i < d; ++i) {
    double den = std::max(v[static_cast<std::size_t>(i)], kMinVar);
    out[static_cast<std::size_t>(i)] =
        (h[static_cast<std::size_t>(i)] - mu[static_cast<std::size_t>(i)]) / den;
  }
}

void frobenius_cap(std::vector<double>& w) {
  double s = 0.0;
  for (double x : w) {
    s += x * x;
  }
  s = std::sqrt(s);
  if (!std::isfinite(s) || s <= 0.0) {
    std::fill(w.begin(), w.end(), 0.0);
    return;
  }
  if (s > kEncFroCap) {
    double t = kEncFroCap / s;
    for (double& x : w) {
      x *= t;
    }
  }
}

}  // namespace

void encoder_align_to_offsets(std::vector<double>& w_row_major, int d,
                              const std::vector<std::vector<double>>& delta_mus) {
  constexpr double kEpsAlign = 1e-8;
  if (d <= 0 || static_cast<int>(w_row_major.size()) != d * d) {
    return;
  }
  if (delta_mus.size() < 2) {
    return;
  }
  std::vector<std::vector<double>> D;
  D.reserve(delta_mus.size());
  for (const auto& row : delta_mus) {
    if (static_cast<int>(row.size()) != d) {
      continue;
    }
    double n2 = 0.0;
    for (int j = 0; j < d; ++j) {
      n2 += row[static_cast<std::size_t>(j)] * row[static_cast<std::size_t>(j)];
    }
    if (n2 > kEpsAlign * kEpsAlign) {
      D.push_back(row);
    }
  }
  if (D.size() < 2) {
    return;
  }
  const int cap = d / 4;
  if (cap <= 0) {
    return;
  }
  const int top_k = std::min(static_cast<int>(D.size()), cap);
  for (int ii = 0; ii < top_k; ++ii) {
    const std::vector<double>& Di = D[static_cast<std::size_t>(ii)];
    double n_sq = 0.0;
    for (int j = 0; j < d; ++j) {
      double v = Di[static_cast<std::size_t>(j)];
      n_sq += v * v;
    }
    n_sq = std::sqrt(n_sq) + kEpsAlign;
    std::vector<double> vdir(static_cast<std::size_t>(d));
    for (int j = 0; j < d; ++j) {
      vdir[static_cast<std::size_t>(j)] = Di[static_cast<std::size_t>(j)] / n_sq;
    }
    std::vector<double> proj(static_cast<std::size_t>(d), 0.0);
    for (int j = 0; j < d; ++j) {
      double s = 0.0;
      for (int i = 0; i < d; ++i) {
        s += w_row_major[static_cast<std::size_t>(i * d + j)] * vdir[static_cast<std::size_t>(i)];
      }
      proj[static_cast<std::size_t>(j)] = s;
    }
    double pn = 0.0;
    for (int j = 0; j < d; ++j) {
      double t = proj[static_cast<std::size_t>(j)];
      pn += t * t;
    }
    pn = std::sqrt(pn);
    if (pn < 0.1) {
      for (int i = 0; i < d; ++i) {
        double vi = vdir[static_cast<std::size_t>(i)];
        for (int j = 0; j < d; ++j) {
          w_row_major[static_cast<std::size_t>(i * d + j)] += 0.01 * vi * vdir[static_cast<std::size_t>(j)];
        }
      }
    }
  }
}

void apply_encoder_grad(std::vector<double>& w_row_major, int d, const double* f, const std::vector<double>& grad_h,
                        double weight, double lr, int& update_count_for_fro_cap) {
  for (int i = 0; i < d; ++i) {
    const double gi = grad_h[static_cast<std::size_t>(i)];
    if (!std::isfinite(gi)) {
      return;
    }
    for (int j = 0; j < d; ++j) {
      const double fv = f[static_cast<std::size_t>(j)];
      if (!std::isfinite(fv)) {
        return;
      }
      w_row_major[static_cast<std::size_t>(i * d + j)] += lr * weight * gi * fv;
    }
  }
  update_count_for_fro_cap += 1;
  if (update_count_for_fro_cap % 50 == 0) {
    frobenius_cap(w_row_major);
  }
}

void contrastive_update_encoder_w(std::vector<double>& w_row_major, int d, const double* f, const double* h,
                                  const double* mu_k, const double* v_k, const double* mu_j, const double* v_j,
                                  double weight, double lr, int& update_count_for_fro_cap) {
  if (d <= 0 || static_cast<int>(w_row_major.size()) != d * d) {
    return;
  }
  for (int i = 0; i < d; ++i) {
    if (!std::isfinite(h[static_cast<std::size_t>(i)]) || !std::isfinite(f[static_cast<std::size_t>(i)])) {
      return;
    }
  }
  std::vector<double> rk;
  std::vector<double> rj;
  fisher_rao_residual(d, h, mu_k, v_k, rk);
  fisher_rao_residual(d, h, mu_j, v_j, rj);
  std::vector<double> grad_h(static_cast<std::size_t>(d));
  for (int i = 0; i < d; ++i) {
    grad_h[static_cast<std::size_t>(i)] = rj[static_cast<std::size_t>(i)] - rk[static_cast<std::size_t>(i)];
  }
  apply_encoder_grad(w_row_major, d, f, grad_h, weight, lr, update_count_for_fro_cap);
}

void variational_ib_update_encoder_w(std::vector<double>& w_row_major, int d, const double* f, const double* h,
                                     const double* mu_k, const double* v_k, const double* mu_j, const double* v_j,
                                     double weight, double lr, double beta, int& update_count_for_fro_cap) {
  if (d <= 0 || static_cast<int>(w_row_major.size()) != d * d) {
    return;
  }
  for (int i = 0; i < d; ++i) {
    if (!std::isfinite(h[static_cast<std::size_t>(i)]) || !std::isfinite(f[static_cast<std::size_t>(i)])) {
      return;
    }
  }
  if (!std::isfinite(beta) || beta < 0.0) {
    return;
  }
  std::vector<double> rk;
  std::vector<double> rj;
  fisher_rao_residual(d, h, mu_k, v_k, rk);
  fisher_rao_residual(d, h, mu_j, v_j, rj);
  const double inv_sigma2 = 1.0 / (kVariationalIbPriorSigma * kVariationalIbPriorSigma);
  std::vector<double> grad_h(static_cast<std::size_t>(d));
  for (int i = 0; i < d; ++i) {
    const double compress = h[static_cast<std::size_t>(i)] * inv_sigma2;
    const double predict =
        beta * (rk[static_cast<std::size_t>(i)] - rj[static_cast<std::size_t>(i)]);
    grad_h[static_cast<std::size_t>(i)] = compress - predict;
  }
  apply_encoder_grad(w_row_major, d, f, grad_h, weight, lr, update_count_for_fro_cap);
}

double latent_class_mi_proxy(const std::vector<std::vector<double>>& h_samples,
                             const std::vector<std::string>& labels) {
  if (h_samples.empty() || h_samples.size() != labels.size()) {
    return 0.0;
  }
  const int d = static_cast<int>(h_samples[0].size());
  if (d <= 0) {
    return 0.0;
  }
  std::unordered_map<std::string, std::vector<std::size_t>> by_label;
  for (std::size_t i = 0; i < labels.size(); ++i) {
    if (static_cast<int>(h_samples[i].size()) != d) {
      return 0.0;
    }
    by_label[labels[i]].push_back(i);
  }
  if (by_label.size() < 2) {
    return 0.0;
  }
  std::vector<std::vector<double>> centroids;
  double within = 0.0;
  int within_n = 0;
  for (const auto& kv : by_label) {
    std::vector<double> c(static_cast<std::size_t>(d), 0.0);
    for (std::size_t ix : kv.second) {
      for (int j = 0; j < d; ++j) {
        c[static_cast<std::size_t>(j)] += h_samples[ix][static_cast<std::size_t>(j)];
      }
    }
    const double inv = 1.0 / static_cast<double>(kv.second.size());
    for (int j = 0; j < d; ++j) {
      c[static_cast<std::size_t>(j)] *= inv;
    }
    for (std::size_t ix : kv.second) {
      for (int j = 0; j < d; ++j) {
        const double diff = h_samples[ix][static_cast<std::size_t>(j)] - c[static_cast<std::size_t>(j)];
        within += diff * diff;
        within_n += 1;
      }
    }
    centroids.push_back(std::move(c));
  }
  if (within_n <= 0 || centroids.size() < 2) {
    return 0.0;
  }
  within /= static_cast<double>(within_n);
  double between = 0.0;
  int pairs = 0;
  for (std::size_t a = 0; a < centroids.size(); ++a) {
    for (std::size_t b = a + 1; b < centroids.size(); ++b) {
      double dist2 = 0.0;
      for (int j = 0; j < d; ++j) {
        const double diff = centroids[a][static_cast<std::size_t>(j)] - centroids[b][static_cast<std::size_t>(j)];
        dist2 += diff * diff;
      }
      between += dist2;
      pairs += 1;
    }
  }
  between /= static_cast<double>(std::max(pairs, 1));
  const double denom = within + between;
  if (denom <= 1e-18) {
    return 0.0;
  }
  return between / denom;
}

void init_encoder_projection_w(int d, std::uint64_t seed, std::vector<double>& w_row_major) {
  if (d <= 0) {
    w_row_major.clear();
    return;
  }
  NumpyDefaultRng rng(static_cast<int>(seed & 0xffffffffu));
  const std::size_t n = static_cast<std::size_t>(d) * static_cast<std::size_t>(d);
  std::vector<double> a(n);
  for (std::size_t i = 0; i < n; ++i) {
    a[i] = rng.normal(0.0, 1.0);
  }
  w_row_major.assign(n, 0.0);
  std::vector<double> col(static_cast<std::size_t>(d));
  for (int j = 0; j < d; ++j) {
    for (int i = 0; i < d; ++i) {
      col[static_cast<std::size_t>(i)] = a[static_cast<std::size_t>(i * d + j)];
    }
    for (int k = 0; k < j; ++k) {
      double dot = 0.0;
      for (int i = 0; i < d; ++i) {
        dot += col[static_cast<std::size_t>(i)] * w_row_major[static_cast<std::size_t>(i * d + k)];
      }
      for (int i = 0; i < d; ++i) {
        col[static_cast<std::size_t>(i)] -= dot * w_row_major[static_cast<std::size_t>(i * d + k)];
      }
    }
    double norm = 0.0;
    for (int i = 0; i < d; ++i) {
      norm += col[static_cast<std::size_t>(i)] * col[static_cast<std::size_t>(i)];
    }
    norm = std::sqrt(std::max(norm, 1e-18));
    for (int i = 0; i < d; ++i) {
      w_row_major[static_cast<std::size_t>(i * d + j)] = 0.5 * col[static_cast<std::size_t>(i)] / norm;
    }
  }
}

}  // namespace cypha
