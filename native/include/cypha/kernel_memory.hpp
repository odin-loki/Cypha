#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "cypha/load_cypha.hpp"

namespace cypha {
/// Nyström RBF kernel memory (reference parity fixture).
class KernelMemory {
 public:
  KernelMemory(int feat_dim, int M = 256, std::uint64_t rng_seed = 0);

  int feat_dim() const { return feat_dim_; }
  int M() const { return M_; }
  double gamma() const { return gamma_; }
  int n_basis() const { return n_basis_; }
  int n_seen() const { return n_seen_; }
  double gamma_scale() const { return gamma_scale_; }

  /// Multiplier on median-heuristic RBF bandwidth (1.0 = legacy default).
  void set_gamma_scale(double s) { gamma_scale_ = std::max(s, 1e-12); }

  /// Whitened Nyström features ``phi(h) ∈ R^M`` (zeros for unfilled slots).
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

  /// Portable snapshot for sidecar JSON / runtime checkpoints.
  struct Snapshot {
    int feat_dim{};
    int M{};
    double gamma{};
    int n_basis{};
    int n_seen{};
    std::vector<double> basis_rowmajor;
    std::map<std::string, std::vector<double>> weights;
  };

  Snapshot export_snapshot() const;
  void import_snapshot(const Snapshot& snap);

  const std::map<std::string, std::vector<double>>& weights() const { return weights_; }

 private:
  void reservoir_update(const double* h, std::optional<int> fixed_j);
  void recompute_nystrom();

  int feat_dim_{};
  int M_{};
  double gamma_{};
  std::vector<double> basis_;
  int n_basis_{0};
  int n_seen_{0};
  double gamma_scale_{1.0};
  /// Row-major ``n_basis × n_basis`` whitening ``K(landmarks, landmarks)^{-1/2}``.
  std::vector<double> whitening_;
  std::map<std::string, std::vector<double>> weights_;
  std::mt19937 rng_;
};

/// Embed kernel LLR state into a v3 ``.cypha`` root (Python ``save_state`` keys).
void patch_kernel_into_root(CNode& root, const KernelMemory& km, bool use_kernel_llr, double kernel_blend);

/// Load kernel state from root; returns false when ``use_kernel_llr`` is off or missing.
bool try_load_kernel_from_root(const CNode& root, KernelMemory& km, bool& use_kernel_llr_out,
                               double& kernel_blend_out);

/// XOR-aware kernel features: ``[x0, x1, x0·x1, x0², x1²]`` (5 dims when ``d >= 2``).
std::vector<double> build_xor_pair_features(const double* x, int d);

}  // namespace cypha
