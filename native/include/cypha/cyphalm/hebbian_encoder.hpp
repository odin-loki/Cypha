#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace cypha::cyphalm {

/// Pluggable encoder update: `W += lr * weight * delta` (row-major `W`, `d×d`).
using HebbianEncoderUpdateFn = std::function<void(std::vector<double>& w_row_major, int d, const double* f,
                                                  const double* h, const double* mu_k, const double* v_k,
                                                  const double* mu_j, const double* v_j, double weight,
                                                  double lr, int& update_count)>;

/// Default biochemical rule (Cypha Tests 2B): `tanh(r_k[0]-r_j[0]) * outer(h, f)`.
void biochemical_hebbian_update(std::vector<double>& w_row_major, int d, const double* f, const double* h,
                                const double* mu_k, const double* v_k, const double* mu_j, const double* v_j,
                                double weight, double lr, int& update_count);

/// Resolve class mean/variance for train-step hooks.
using HebbianClassStatsFn =
    std::function<bool(const std::string& label, std::vector<double>& mu, std::vector<double>& v)>;

/// Encoder projection with swappable Hebbian/biochemical update rule.
struct HebbianEncoder {
  std::vector<double> w;
  int d{0};
  int update_count{0};
  bool frozen{false};
  HebbianEncoderUpdateFn update_rule{biochemical_hebbian_update};
  HebbianClassStatsFn class_stats;
  std::string competitor_label;

  void project(const double* f, double* h_out) const;
  /// Train hook: resolves `label` and `competitor_label` via `class_stats`, then applies `update_rule`.
  void update(const double* f, const double* h, const std::string& label, double lr, double weight = 1.0);
  /// Direct update with explicit class statistics (parity / tests).
  void update_with_stats(const double* f, const double* h, const double* mu_k, const double* v_k,
                         const double* mu_j, const double* v_j, double lr, double weight = 1.0);
};

}  // namespace cypha::cyphalm
