#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cypha/load_cypha.hpp"

namespace cypha {

/// M1 inference state for CyphaDIF + VectorEncoder (CPU float64).
struct CyphaInferModel {
  int d_latent{};
  int field_dim{};
  std::vector<std::string> labels;
  /// Per-class n_obs (same order as labels).
  std::vector<double> n_obs{};
  std::vector<double> D{};
  std::vector<double> enc_w{};
  std::vector<double> mu_world{};
  std::vector<double> inv_v{};
  double v_mean{};
  std::vector<double> f_field{};
  std::vector<double> field_h{};
  /// Optional NIG field transition (Python `field_W_T`); empty → skip native field dynamics.
  std::vector<double> field_w_t{};
  std::vector<float> field_a_eff{};
  std::vector<float> field_sr_vec{};
  /// Python `w_inject`: (field_dim × d_latent) row-major; empty → use identity only if d==field_dim.
  std::vector<double> w_inject{};
  double temperature{};
  double mahal_ema{};
  bool has_mahal_ema{false};
  double mahal_std_ema{0.5};
  double llr_ema{0.0};
  int saved_total_steps{0};
  /// Python `_total_correct` checkpoint from `.cypha` (native train increments in place when stepping).
  std::int64_t total_correct{0};
  double mid_n{};
  std::vector<std::pair<std::string, double>> mid_freq{};
  /// Sum of `mid_freq` values (Python `_mid_freq_total`).
  double mid_freq_total{0};
  /// Tier-1 / Tier-2 context (Python `TieredContextBuffer`).
  std::string ctx_last_label;
  std::unordered_map<std::string, double> t1_counts{};
  double t1_total{0};
  std::vector<std::pair<std::string, bool>> ctx_history{};
  std::unordered_map<std::string, std::unordered_map<std::string, double>> cooccur{};
  std::unordered_map<std::string, double> cooccur_tot{};
  std::unordered_map<std::string, std::unordered_map<std::string, double>> mid_trans{};
  std::unordered_map<std::string, double> mid_trans_tot{};
  double llr_scale_ema{0.0};
  int llr_scale_n{0};
  /// Python `llr_scale_baseline` / `base_temp` for `auto_recalibrate`.
  double llr_scale_baseline{0.0};
  double base_temp{0.0};
  /// Python `CausalField._step` — NIG field `evolve` count (native: incremented when `nig_field_evolve` runs).
  std::int64_t field_step{0};

  /// Load inference buffers from a `.cypha` root. If `world.F_field` is stored in the blob (same layout
  /// as Python `WorldPrior.F_field`), pass `f_field_row_major == nullptr`. Otherwise pass row-major floats.
  static CyphaInferModel from_root(const CNode& root, const double* f_field_row_major,
                                   int field_dim_in);
};

void batch_encode(const CyphaInferModel& m, const double* x_row_major, int n, std::vector<double>& h_out);

void score_matrix_use_field(const CyphaInferModel& m, const double* h_row_major, int n,
                            std::vector<double>& llr_out);

/// Convenience: ``batch_encode`` then ``score_matrix_use_field`` — ``llr_out`` is **n×K** row-major (``K = len(labels)``).
void batch_llr_from_x(const CyphaInferModel& m, const double* x_row_major, int n, std::vector<double>& llr_out);

void softmax_batch_like_python(const double* z_row_major, int n, int k, double eps,
                               std::vector<double>& probs_out);

void world_gate_vector_use_field(const CyphaInferModel& m, const double* h_row_major, int n,
                                 double gh_chi, double gh_psi, std::vector<double>& gates_out);

/// Tier-1+2 context prior logits (same order as `classes`). Used by `score_matrix` and native training.
void context_prior_for_labels(const CyphaInferModel& m, const std::vector<std::string>& classes,
                              std::vector<double>& ctx_out);

/// Python `TieredContextBuffer.record` (Tier-1 window + co-occurrence + mid EMAs).
void context_record_step(CyphaInferModel& m, const std::string& label, bool correct);

/// Learning-rate scale for GH-protected training (`R_base / max(R_eff, R_base)`), matching Python `gh_train_step`.
/// `mahal_sq` is Σⱼ (hⱼ−μⱼ)² inv_v_cleanⱼ / d (same as Python `mahal_sq`).
double gh_train_lr_scale(double mahal_sq, double r_base, double chi, double psi);

/// Python `_nig_R_eff` for GH training diagnostics (`mahal_sq` per-dim normalised innovation).
double nig_R_eff_gh(double mahal_sq, double r_base, double chi, double psi);

/// Session NIG χ adaptation (ψ unchanged). Matches Python `_nig_adapt` used after `gh_train_step`.
std::pair<double, double> nig_adapt_session_chi(double chi, double psi, double innovation_sq, double r_base,
                                                 double alpha = 0.98);

/// Python `CyphaDIF.auto_recalibrate` — EMA-adjust `temperature` from `llr_scale_ema` vs baseline (no-op if `llr_scale_n` < 50).
void auto_recalibrate_temperature(CyphaInferModel& m, double decay = 0.995);

/// Python `CyphaDIF.adapt_temperature`: grid-search T minimising ECE on labelled rows of `h` (same LLRs as `score_matrix_use_field`).
/// `true_class_idx[i]` ∈ [0, K). Updates `infer.temperature` and returns the chosen T.
double adapt_temperature_ece(CyphaInferModel& infer, const double* h_row_major, int n_cal, const int* true_class_idx,
                             int n_grid = 20, double T_min = 0.3, double T_max = 8.0, int n_bins = 10);

}  // namespace cypha
