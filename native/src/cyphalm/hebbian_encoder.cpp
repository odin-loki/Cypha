#include "cypha/cyphalm/hebbian_encoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace cypha::cyphalm {

namespace {

constexpr double kMinVar = 1e-4;
constexpr double kEncFroCap = 8.0;

void fisher_rao_residual(int d, const double* h, const double* mu, const double* v, std::vector<double>& out) {
  out.resize(static_cast<std::size_t>(d));
  for (int i = 0; i < d; ++i) {
    const double den = std::max(v[static_cast<std::size_t>(i)], kMinVar);
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
    const double t = kEncFroCap / s;
    for (double& x : w) {
      x *= t;
    }
  }
}

bool inputs_finite(int d, const double* f, const double* h) {
  for (int i = 0; i < d; ++i) {
    if (!std::isfinite(h[static_cast<std::size_t>(i)]) || !std::isfinite(f[static_cast<std::size_t>(i)])) {
      return false;
    }
  }
  return true;
}

}  // namespace

void biochemical_hebbian_update(std::vector<double>& w_row_major, int d, const double* f, const double* h,
                                  const double* mu_k, const double* v_k, const double* mu_j, const double* v_j,
                                  double weight, double lr, int& update_count) {
  if (d <= 0 || static_cast<int>(w_row_major.size()) != d * d) {
    return;
  }
  if (!inputs_finite(d, f, h)) {
    return;
  }
  std::vector<double> rk;
  std::vector<double> rj;
  fisher_rao_residual(d, h, mu_k, v_k, rk);
  fisher_rao_residual(d, h, mu_j, v_j, rj);
  const double signal = std::tanh(rk[0] - rj[0]);
  if (!std::isfinite(signal)) {
    return;
  }
  for (int i = 0; i < d; ++i) {
    const double hi = h[static_cast<std::size_t>(i)];
    for (int j = 0; j < d; ++j) {
      const double fv = f[static_cast<std::size_t>(j)];
      const double delta = signal * hi * fv;
      if (!std::isfinite(delta)) {
        return;
      }
      w_row_major[static_cast<std::size_t>(i * d + j)] += lr * weight * delta;
    }
  }
  update_count += 1;
  if (update_count % 50 == 0) {
    frobenius_cap(w_row_major);
  }
}

void HebbianEncoder::project(const double* f, double* h_out) const {
  if (d <= 0 || static_cast<int>(w.size()) != d * d || f == nullptr || h_out == nullptr) {
    return;
  }
  for (int i = 0; i < d; ++i) {
    double s = 0.0;
    for (int j = 0; j < d; ++j) {
      s += w[static_cast<std::size_t>(i * d + j)] * f[static_cast<std::size_t>(j)];
    }
    h_out[static_cast<std::size_t>(i)] = s;
  }
}

void HebbianEncoder::update_with_stats(const double* f, const double* h, const double* mu_k, const double* v_k,
                                       const double* mu_j, const double* v_j, double lr, double weight) {
  if (frozen || !update_rule) {
    return;
  }
  update_rule(w, d, f, h, mu_k, v_k, mu_j, v_j, weight, lr, update_count);
}

void HebbianEncoder::update(const double* f, const double* h, const std::string& label, double lr,
                            double weight) {
  if (frozen || !class_stats || !update_rule) {
    return;
  }
  std::vector<double> mu_k;
  std::vector<double> v_k;
  std::vector<double> mu_j;
  std::vector<double> v_j;
  if (!class_stats(label, mu_k, v_k)) {
    return;
  }
  const std::string& rival = competitor_label.empty() ? label : competitor_label;
  if (!class_stats(rival, mu_j, v_j)) {
    return;
  }
  update_with_stats(f, h, mu_k.data(), v_k.data(), mu_j.data(), v_j.data(), lr, weight);
}

}  // namespace cypha::cyphalm
