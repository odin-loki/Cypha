#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "cypha/load_cypha.hpp"
#include "cypha/rff_features.hpp"

namespace cypha {
/// Nyström RBF kernel memory (reference parity fixture).
class KernelMemory {
 public:
  KernelMemory(int feat_dim, int M = 256, std::uint64_t rng_seed = 0);

  /// Landmark reservoir sampling mode (Phase 5 optimality).
enum class LandmarkSamplingKind {
  /// Legacy uniform reservoir + nearest-slot replacement.
  Uniform,
  /// Ridge-leverage-score weighted landmark selection (Alaoui–Mahoney).
  LeverageScore,
};

/// Random Fourier Features (RFF) basis: fixed random projection approximating the RBF kernel via
  /// Bochner's theorem (``phi(x) = sqrt(2/M) * cos(W x + b)``), in place of the online Nyström landmark
  /// sketch. Cheap (``O(M*feat_dim)`` per call, no Cholesky/eigh recompute) since the projection is frozen
  /// at construction — ``gamma`` must be supplied up front (see ``auto_gamma_median_heuristic``).
  static KernelMemory make_rff(int feat_dim, int M, double gamma, std::uint64_t rng_seed = 0,
                               RffProjectionKind projection = RffProjectionKind::IidGaussian);

  /// SORF / Fastfood structured orthogonal RFF basis (Phase 5 opt-in).
  static KernelMemory make_orthogonal_rff(int feat_dim, int M, double gamma, std::uint64_t rng_seed = 0) {
    return make_rff(feat_dim, M, gamma, rng_seed, RffProjectionKind::Sorf);
  }

  /// Median pairwise-distance ("auto-gamma") heuristic: ``gamma_scale / (2 * median(||a-b||^2))`` over up
  /// to ``max_samples`` rows sampled from ``samples_row_major`` (``n x feat_dim``). Same heuristic the
  /// Nyström path uses internally for its landmarks, exposed here for RFF gamma calibration from raw data.
  static double auto_gamma_median_heuristic(const double* samples_row_major, int n, int feat_dim,
                                            double gamma_scale = 1.0, int max_samples = 256,
                                            std::uint64_t rng_seed = 0);

  bool is_rff() const { return rff_mode_; }
  int feat_dim() const { return feat_dim_; }
  int M() const { return M_; }
  double gamma() const { return gamma_; }
  int n_basis() const { return n_basis_; }
  int n_seen() const { return n_seen_; }
  double gamma_scale() const { return gamma_scale_; }

  /// Multiplier on median-heuristic RBF bandwidth (1.0 = legacy default).
  void set_gamma_scale(double s) { gamma_scale_ = std::max(s, 1e-12); }

  /// Opt-in ridge-leverage-score landmark sampling for the Nyström reservoir (default ``Uniform``).
  void set_landmark_sampling(LandmarkSamplingKind kind) { landmark_sampling_ = kind; }
  LandmarkSamplingKind landmark_sampling() const { return landmark_sampling_; }

  /// Batch-initialize Nyström landmarks from a calibration set via leverage-score sampling.
  void init_leverage_landmarks_from_samples(const double* samples_row_major, int n, int feat_dim);

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

  bool rff_mode_{false};
  RffProjectionKind rff_projection_{RffProjectionKind::IidGaussian};
  LandmarkSamplingKind landmark_sampling_{LandmarkSamplingKind::Uniform};
  /// Row-major ``M × feat_dim`` random projection (RFF mode only).
  std::vector<double> rff_w_;
  std::vector<double> rff_b_;
};

/// Embed kernel LLR state into a v3 ``.cypha`` root (Python ``save_state`` keys).
void patch_kernel_into_root(CNode& root, const KernelMemory& km, bool use_kernel_llr, double kernel_blend);

/// Load kernel state from root; returns false when ``use_kernel_llr`` is off or missing.
bool try_load_kernel_from_root(const CNode& root, KernelMemory& km, bool& use_kernel_llr_out,
                               double& kernel_blend_out);

/// XOR-aware kernel features: ``[x0, x1, x0·x1, x0², x1²]`` (5 dims when ``d >= 2``).
std::vector<double> build_xor_pair_features(const double* x, int d);

}  // namespace cypha
