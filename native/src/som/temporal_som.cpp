#include "cypha/som/temporal_som.hpp"

#include "cypha/mt19937_rng.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace cypha::som {

namespace {

double l2_norm(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0.0;
  const int n = static_cast<int>(std::min(a.size(), b.size()));
  for (int i = 0; i < n; ++i) {
    const double d = a[static_cast<std::size_t>(i)] - b[static_cast<std::size_t>(i)];
    s += d * d;
  }
  for (std::size_t i = static_cast<std::size_t>(n); i < a.size(); ++i) {
    s += a[i] * a[i];
  }
  for (std::size_t i = static_cast<std::size_t>(n); i < b.size(); ++i) {
    s += b[i] * b[i];
  }
  return std::sqrt(s);
}

double median(std::vector<double> v) {
  if (v.empty()) {
    return 0.0;
  }
  const std::size_t mid = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid), v.end());
  if (v.size() % 2 == 1) {
    return v[mid];
  }
  const double lo = v[mid];
  std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid - 1), v.end());
  return 0.5 * (lo + v[mid - 1]);
}

}  // namespace

TemporalSOM::TemporalSOM(TemporalSOMConfig cfg)
    : M_(std::max(1, cfg.M)),
      L_max_(std::max(1, cfg.L_max)),
      eta_ts_(cfg.eta_ts) {
  cypha::NumpyDefaultRng rng(static_cast<int>(cfg.seed));

  centroids_.assign(static_cast<std::size_t>(M_),
                    std::vector<double>(static_cast<std::size_t>(L_max_), 0.0));
  for (int m = 0; m < M_; ++m) {
    for (int j = 0; j < L_max_; ++j) {
      centroids_[static_cast<std::size_t>(m)][static_cast<std::size_t>(j)] = rng.normal(0.0, 1.0) * 0.1;
    }
  }

  lambda_.assign(static_cast<std::size_t>(M_), std::vector<double>(2, 1.0));
  for (int m = 0; m < M_; ++m) {
    lambda_[static_cast<std::size_t>(m)][0] = rng.uniform(0.85, 1.15);
    lambda_[static_cast<std::size_t>(m)][1] = rng.uniform(0.85, 1.15);
  }
}

std::vector<double> TemporalSOM::autocorr_features(const std::vector<double>& x) {
  x_hist_.push_back(x);
  const std::size_t cap = static_cast<std::size_t>(L_max_ + 2);
  if (x_hist_.size() > cap) {
    x_hist_.erase(x_hist_.begin(), x_hist_.end() - static_cast<std::ptrdiff_t>(cap));
  }

  std::vector<double> feats(static_cast<std::size_t>(L_max_), 0.0);
  if (x_hist_.size() < 3) {
    return feats;
  }

  const int feat_dim = static_cast<int>(x_hist_.back().size());
  std::vector<double> mean(static_cast<std::size_t>(feat_dim), 0.0);
  double var_sum = 0.0;
  std::size_t count = 0;
  for (const auto& row : x_hist_) {
    for (int j = 0; j < feat_dim; ++j) {
      mean[static_cast<std::size_t>(j)] += row[static_cast<std::size_t>(j)];
    }
    ++count;
  }
  const double inv_n = 1.0 / static_cast<double>(count);
  for (double& m : mean) {
    m = m * inv_n + 1e-9;
  }
  for (const auto& row : x_hist_) {
    for (int j = 0; j < feat_dim; ++j) {
      const double d = row[static_cast<std::size_t>(j)];
      const double dm = d - mean[static_cast<std::size_t>(j)];
      var_sum += dm * dm;
    }
  }
  const double denom = var_sum * inv_n + 1e-9;

  const int max_lag = std::min(L_max_, static_cast<int>(x_hist_.size()));
  const auto& a = x_hist_.back();
  for (int lag = 1; lag < max_lag; ++lag) {
    const auto& b = x_hist_[x_hist_.size() - 1 - static_cast<std::size_t>(lag)];
    double acc = 0.0;
    for (int j = 0; j < feat_dim; ++j) {
      acc += (a[static_cast<std::size_t>(j)] - mean[static_cast<std::size_t>(j)]) *
             (b[static_cast<std::size_t>(j)] - mean[static_cast<std::size_t>(j)]);
    }
    feats[static_cast<std::size_t>(lag - 1)] = (acc / static_cast<double>(feat_dim)) / denom;
  }
  return feats;
}

std::tuple<int, double, double> TemporalSOM::step(const std::vector<double>& x, bool train) {
  if (x.empty()) {
    throw std::invalid_argument("TemporalSOM::step: empty input");
  }
  const std::vector<double> r = autocorr_features(x);

  std::vector<double> dists(static_cast<std::size_t>(M_), 0.0);
  int bmu = 0;
  double best = std::numeric_limits<double>::infinity();
  for (int m = 0; m < M_; ++m) {
    dists[static_cast<std::size_t>(m)] = l2_norm(centroids_[static_cast<std::size_t>(m)], r);
    if (dists[static_cast<std::size_t>(m)] < best) {
      best = dists[static_cast<std::size_t>(m)];
      bmu = m;
    }
  }

  if (train) {
    const double med = median(dists) + 1e-9;
    std::vector<double> h(static_cast<std::size_t>(M_), 0.0);
    double h_sum = 0.0;
    for (int m = 0; m < M_; ++m) {
      h[static_cast<std::size_t>(m)] = std::exp(-dists[static_cast<std::size_t>(m)] / med);
      h_sum += h[static_cast<std::size_t>(m)];
    }
    h_sum += 1e-12;
    for (int m = 0; m < M_; ++m) {
      const double w = h[static_cast<std::size_t>(m)] / h_sum;
      for (int j = 0; j < L_max_; ++j) {
        centroids_[static_cast<std::size_t>(m)][static_cast<std::size_t>(j)] +=
            eta_ts_ * w * (r[static_cast<std::size_t>(j)] -
                           centroids_[static_cast<std::size_t>(m)][static_cast<std::size_t>(j)]);
      }
    }
  }

  const double lam_fast = lambda_[static_cast<std::size_t>(bmu)][0];
  const double lam_slow = lambda_[static_cast<std::size_t>(bmu)][1];
  return {bmu, lam_fast, lam_slow};
}

}  // namespace cypha::som
