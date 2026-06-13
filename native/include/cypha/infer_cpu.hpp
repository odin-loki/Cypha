#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cypha/kernel_memory.hpp"
#include "cypha/load_cypha.hpp"
#include "cypha/retrieval.hpp"

namespace cypha {

struct PreprocessorState;

/// Python ``CyphaDIF.deliberation_lo/hi`` defaults (disabled: lo >= hi).
constexpr double kDeliberationLoDefault = 1.0;
constexpr double kDeliberationHiDefault = 0.0;
constexpr const char* kUnknownLabel = "__unknown__";

/// Runtime inference modifiers (Python ``infer`` / ``gh_infer`` / REST ``POST /predict``).
struct CyphaInferOptions {
  double deliberation_lo{kDeliberationLoDefault};
  double deliberation_hi{kDeliberationHiDefault};
  bool use_field{true};
  double gh_chi{1.0};
  double gh_psi{1.0};
  double gh_alpha{0.98};
  /// Nyström kernel LLR (Python ``use_kernel_llr`` / ``_kernel_mem``).
  const KernelMemory* kernel_mem{nullptr};
  bool use_kernel_llr{false};
  double kernel_blend{0.5};
  /// Optional kernel feature vector (raw ``x`` or XOR pair features); defaults to ``h``.
  const double* kernel_x{nullptr};
};

struct ClassifyAtHResult {
  std::string label;
  double confidence{};
  double disc{};
  double world_gate{1.0};
  double r_eff{};
  double mahal_per_dim{};
  std::vector<double> llrs;
};

struct GhInferAtHResult {
  std::string label;
  double confidence{};
  double r_eff{};
  double chi_new{1.0};
  double psi_new{1.0};
  double t_adj{};
  std::vector<double> llrs;
};

struct InferAtHResult {
  std::string label;
  double confidence{};
  std::vector<double> llrs;
};

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
  /// Python ``deliberation_lo`` / ``deliberation_hi`` (defaults disable abstention).
  double deliberation_lo{kDeliberationLoDefault};
  double deliberation_hi{kDeliberationHiDefault};

  /// Load inference buffers from a `.cypha` root. If `world.F_field` is stored in the blob (same layout
  /// as Python `WorldPrior.F_field`), pass `f_field_row_major == nullptr`. Otherwise pass row-major floats.
  static CyphaInferModel from_root(const CNode& root, const double* f_field_row_major,
                                   int field_dim_in);
};

void batch_encode(const CyphaInferModel& m, const double* x_row_major, int n, std::vector<double>& h_out);

void score_matrix_use_field(const CyphaInferModel& m, const double* h_row_major, int n,
                            std::vector<double>& llr_out,
                            const KernelMemory* kernel_mem = nullptr, bool use_kernel_llr = false,
                            double kernel_blend = 0.5);

/// Convenience: ``batch_encode`` then ``score_matrix_use_field`` — ``llr_out`` is **n×K** row-major (``K = len(labels)``).
void batch_llr_from_x(const CyphaInferModel& m, const double* x_row_major, int n, std::vector<double>& llr_out);

void softmax_batch_reference(const double* z_row_major, int n, int k, double eps,
                             std::vector<double>& probs_out);

/// Shannon entropy of one softmax row (nats).
double row_entropy_from_probs(const double* p, int k, double eps = 1e-8);

/// Active-learning order: row indices sorted by descending predictive entropy (``batch_encode`` +
/// ``score_matrix_use_field`` + temperature-scaled softmax). ``n_rows`` rows of raw features in ``x_rowmajor``.
std::vector<int> uncertainty_rank_indices(const CyphaInferModel& m, const PreprocessorState* pre,
                                          const double* x_rowmajor, int n_rows, int n_features,
                                          double temperature = -1.0);

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

/// Python ``CyphaDIF._apply_deliberation``.
std::pair<std::string, double> apply_deliberation(const std::string& pred, double conf, double lo, double hi);

/// Python ``DIFMemory.classify`` for one latent row (optional ``h_field`` → μ₀ shift).
ClassifyAtHResult classify_at_h(const CyphaInferModel& m, const double* h, const double* h_field,
                                double temperature, const std::optional<double>& mahal_ema, double mahal_std_ema,
                                double gh_chi, double gh_psi, bool use_context_prior = true,
                                const KernelMemory* kernel_mem = nullptr, bool use_kernel_llr = false,
                                double kernel_blend = 0.5, const double* kernel_feat = nullptr);

/// Python ``CyphaDIF.infer`` (``use_field`` + deliberation + optional kernel LLR blend).
InferAtHResult infer_at_h(const CyphaInferModel& m, const double* h, const CyphaInferOptions& opt);

/// Python ``CyphaDIF.gh_infer`` (no field in μ₀ / T_adj; classify with GH gate).
/// When ``kernel_opt`` is non-null and ``use_kernel_llr``, blends Nyström kernel LLR in ``classify_at_h``.
GhInferAtHResult gh_infer_at_h(const CyphaInferModel& m, const double* h, double chi, double psi,
                               double alpha = 0.98, const CyphaInferOptions* kernel_opt = nullptr);

/// Python ``CyphaDIF.retrieve`` on raw ``x`` rows (encode then rank by class log-likelihood).
std::vector<RetrieveHit> retrieve_from_x(const CyphaInferModel& m, const double* query_x, const double* database_x,
                                         int n_db, int input_dim, int top_k, const CyphaInferOptions& opt,
                                         const std::optional<std::string>& label = std::nullopt);

/// FastAPI ``InferenceEngine`` anomaly from ``gh_infer`` ``R_eff`` and ``_mahal_ema``.
double gh_infer_anomaly_score(double r_eff, double mahal_ema_fallback);

}  // namespace cypha
