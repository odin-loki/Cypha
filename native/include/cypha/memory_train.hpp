#pragma once

#include "cypha/class_gmm.hpp"
#include "cypha/load_cypha.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cypha {

/// Filled by `CyphaDifMemoryState::memory_train` when `meta_out != nullptr`.
struct MemoryTrainMeta {
  std::string pred_label;
  bool correct{false};
  /// Post-update softmax confidence on LLRs (Python `train` → `post_conf`).
  double post_conf{0};
  /// Two highest post-update LLR classes (by LLR), when K ≥ 2 (for deliberate encoder).
  std::string llr_rank0;
  std::string llr_rank1;
  /// `max_k post_update_llr` (Python `train_step` → `llr_scale_ema` tracking).
  double post_llr_max{0.0};
};

/// Mutable CyphaDIF memory slice for `DIFMemory.train` parity (no encoder / field / replay).
struct CyphaDifMemoryState {
  int d_latent{};
  int field_dim{};
  std::vector<std::string> labels;
  std::vector<double> D{};
  /// Per-class mixture weights (``K×max_m`` row-major); meaningful when ``use_class_gmm``.
  std::vector<double> class_pi{};
  /// Active component count per class in ``[1, kClassGmmMaxM]``.
  std::vector<int> class_n_comp{};
  /// Opt-in multimodal classes (default **off** — legacy single Gaussian per class).
  bool use_class_gmm{false};
  int class_gmm_m{kClassGmmDefaultM};
  /// When non-empty, ``memory_train`` zeros update scales for labels whose prefix
  /// (before first ``_``) does not match — task-sticky CL (D16).
  std::string task_prefix_protect;
  int cypha_format{kCyphaFormatV3};
  std::vector<double> n_obs_buf{};
  std::vector<std::int64_t> n_correct{};
  std::unordered_map<std::string, int> label_index{};

  std::vector<double> f_field{};

  std::vector<double> world_mu{};
  std::vector<double> world_v{};
  std::vector<double> world_inv_v{};
  double world_v_mean{0};
  std::int64_t world_n{0};
  double world_drift_ema{0};
  std::vector<double> world_M2{};
  std::vector<double> world_buf{};
  double world_log_norm{0};
  double world_D_LOG2PI{0};
  std::vector<double> world_inv_d{};
  int world_log_n_ctr{0};

  static CyphaDifMemoryState from_cypha_root(const CNode& root, const double* f_field_row_major,
                                             int field_dim_in);

  /// Clone ``root`` and replace ``world`` / ``classes`` (and ``world.F_field`` when present) from training state.
  [[nodiscard]] static CNode merge_state_into_root_for_save(const CNode& root, const CyphaDifMemoryState& s);

  /// Recompute `world_log_norm` from `world_v` (Python `load_state` / disk load path). Resets `world_log_n_ctr`.
  void refresh_world_log_norm_from_v();

  /// Per-class diagonal Gaussian in latent space: μ = world_mu + delta_mu, v = world.v.
  bool class_mean_and_variance(const std::string& label, std::vector<double>& mu_out,
                               std::vector<double>& v_out) const;

  /// One `DIFMemory.train` step. Returns training loss (same formula as Python).
  double memory_train(const double* h, const std::string& label, const double* h_field,
                      const std::unordered_map<std::string, double>& context_prior, double temperature,
                      double ood_sigma, double world_lr, double delta_lr,
                      MemoryTrainMeta* meta_out = nullptr);

  /// Python `_dedup_check`: separate highly overlapping class deltas.
  void dedup_check(const std::string& label);

  [[nodiscard]] ClassGmmStorage gmm_view() const;
  int gmm_max_m() const { return use_class_gmm ? kClassGmmMaxM : 1; }
};

/// Max entry of Python `DIFMemory.classify` LLR dict (no softmax / gate).
double memory_max_classify_llr(const CyphaDifMemoryState& s, const double* h, const double* h_field,
                               const std::unordered_map<std::string, double>& context_prior);

/// `(h−μ_world)ᵀ diag(inv_v)(h−μ_world) / (d+ε)` for OOD EMA (Python `train_step` block).
double memory_mahal_world_scalar(const CyphaDifMemoryState& s, const double* h);

/// Fisher–Rao norm ``Σ Δμ²/v₀`` (Python ``ClassDifferential.fisher_rao_norm``).
double class_fisher_rao_norm(const double* delta_mu, int d, const double* v0);

/// Merge class differentials from ``other`` into ``self`` (Python ``CyphaDIF.merge_from``).
/// Returns labels newly copied from ``other``.
std::vector<std::string> memory_merge_from(CyphaDifMemoryState& self, const CyphaDifMemoryState& other,
                                           double weight_self = 0.5, double weight_other = 0.5);

}  // namespace cypha
