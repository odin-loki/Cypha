#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace cypha {

/// Nyström RBF kernel memory — port of ``Cypha.py`` ``KernelMemory``.
class KernelMemory {
 public:
  KernelMemory(int feat_dim, int M = 64, std::uint64_t rng_seed = 0);

  int feat_dim() const { return feat_dim_; }
  int M() const { return M_; }
  double gamma() const { return gamma_; }
  int n_basis() const { return n_basis_; }
  int n_seen() const { return n_seen_; }

  /// RBF features ``phi(h) ∈ R^M`` (zeros for unfilled slots).
  void phi(const double* h, std::vector<double>& out) const;

  /// Kernel LLR scores for ``labels`` — ``score_k = w_k·phi(h) - 0.5‖w_k‖²``.
  void score_all(const double* h, const std::vector<std::string>& labels,
                 std::vector<double>& scores) const;

  /// Reservoir update + one-step softmax gradient on weights.
  /// When ``fixed_reservoir_j`` is set, use that index instead of ``rng_`` (parity harness).
  void update(const double* h, const std::string& label, const std::vector<std::string>& all_labels,
              double lr = 0.05, std::optional<int> fixed_reservoir_j = std::nullopt);

  /// Restore state from a parity fixture or runtime checkpoint (not ``.cypha`` v3).
  void load_state(int n_basis, int n_seen, const double* basis_row_major, int basis_rows,
                  const std::map<std::string, std::vector<double>>& weights);

  const std::map<std::string, std::vector<double>>& weights() const { return weights_; }

 private:
  void reservoir_update(const double* h, std::optional<int> fixed_j);

  int feat_dim_{};
  int M_{};
  double gamma_{};
  std::vector<double> basis_;
  int n_basis_{0};
  int n_seen_{0};
  std::map<std::string, std::vector<double>> weights_;
  std::mt19937 rng_;
};

}  // namespace cypha
