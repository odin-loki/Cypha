#pragma once

/// M4 — Regression stack (native port in progress).
///
/// Python reference: `DIFRegressor`, `RFFRegressor`, `MKERegressor`,
/// `TwoStageDIFRegressor` reference. Classification + `.cypha` v3 remain
/// the shipping path; this header is the native hook for mixture-of-experts
/// regression (`E[y|x] = Σ_k p(k|x) · μ_y,k`). See `docs/port/PORT_FULL_STACK.md` §M4.
///
/// **M4–M5:** RLS, two-stage combine, MKE routing softmax, native-LLR predict pipeline.
/// **Milestone 6:** two-stage ridge fit from LLR. **Milestone 7:** batched two-stage predict
/// (``two_stage_dif_predict_batch``). Use ``cypha::batch_llr_from_x`` for **N×K** LLR from raw **X**.
/// **E2E workflow (Python router):** fit quantile ``CyphaDIF`` in Python → export **LLR** + **X**, **y** →
/// ``two_stage_dif_ridge_fit_from_llr`` + ``two_stage_dif_predict_batch`` (see ``fixtures/two_stage_e2e_ridge/``).
/// **Full scalar MKE train step (router + experts):** ``mke_scalar_train_step.hpp`` — ``mke_scalar_train_step`` /
/// ``mke_scalar_train_step_from_phi`` compose RFF → LLR → expert RLS → ``dif_train_step_vector`` (same as
/// ``mke_train_step_parity``).
///
/// `predict_mixture_scalar` matches `DIFRegressor.predict` for **scalar** targets
/// (same Σ p·μ and √(Σ p·var) combination as reference parity fixture).

#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace cypha {

struct CyphaInferModel;

}  // namespace cypha

namespace cypha::regression {

/// Increment when adding regression surface (7 = batched two-stage predict).
constexpr int k_native_regression_milestone = 7;

/// Returns `k_native_regression_milestone` (stable symbol for the static library).
int native_regression_milestone();

/// Scalar MoE combine: `y = Σ probs[i]*mu[i]`, `unc = sqrt(max(Σ probs[i]*var[i], 0))`.
/// Arrays are length `k` (router order must match expert stats order, as in Python `enumerate(labels)`).
void predict_mixture_scalar(const double* probs, const double* mu, const double* var_ema,
                            std::size_t k, double& out_y_pred, double& out_uncertainty);

/// Batched MoE: ``probs`` (N×K row-major), ``mu_mat`` (K×d row-major), ``var_vec`` (K) —
/// matches ``DIFRegressor.predict_batch`` (`y = P @ mu_mat`, `unc = sqrt(max(P @ var,0))`).
void predict_mixture_batch(const double* probs, int n, int k, int d, const double* mu_mat,
                           const double* var_vec, double* out_y, double* out_unc);

/// One target EMA step per expert (`DIFRegressor` `_expert_mu` / `_expert_var`). If ``mu`` is
/// empty or length ≠ ``d``, initializes to ``y`` (first observation).
void expert_target_ema_step(std::vector<double>& mu, double& var_ema, int& n_updates, const double* y, int d,
                            double lr);

/// RFF features: φ_{i,d} = √(2/D)·cos((X Wᵀ + b)_{i,d}). ``W`` is **D×d_in** row-major (same layout as
/// Python ``RFFEncoder.W``), ``X`` is **n×d_in** row-major, ``out_phi`` is **n×D** row-major.
void rff_encode_batch_rowmajor(const double* X, int n, int d_in, const double* W, const double* b, int D,
                               double* out_phi);

/// Ridge with bias column: design row i is ``[phi_{i,0..D-1}, 1]``. ``lam`` is added to the diagonal of the
/// first ``d_feat`` normal equations only (bias unpenalized). Writes ``d_feat+1`` coefficients:
/// pred_i = Σ_j phi_{ij}·coef[j] + coef[d_feat]. Returns false if the normal matrix is not SPD.
bool ridge_fit_bias(const double* Phi, int n, int d_feat, double lam, const double* y_norm, double* out_coef);

/// ``out_pred[i] = Σ_j Phi[i,j]*coef[j] + bias`` where ``bias = coef[d_feat]``.
void linear_predict_with_bias(const double* Phi, int n, int d_feat, const double* coef, double* out_pred);

/// MKERegressor scalar expert heads: ``out[k] = Σ_d phi[d] * W[k,d]`` with ``W`` **K×d_feat** row-major.
void mke_expert_linear_dots(const double* phi, int d_feat, int K, const double* W_experts_rowmajor,
                            double* out_dots);

/// `RFFRegressor.train_step`: bias-augmented RLS on φ (length ``D``). ``P`` is **(D+1)×(D+1)** row-major
/// (precision / covariance per Python). Updates ``w`` (D), ``b``, ``P`` in place. Returns squared error
/// in **original** y scale: ``(err_norm * y_std)^2``.
double rff_rls_train_step(const double* phi, int D, double* w, double* b, double* P_rowmajor,
                          double y_raw, double y_mean, double y_std);

/// One `MKERegressor` scalar expert RLS step (Python `train_step` inner loop for one label).
/// ``P`` is **D×D** row-major. ``err`` is scalar ``y - ŷ`` (same for all experts in scalar mode).
/// If ``forgetting_factor < 1``, scales ``P ← P / forgetting_factor`` before the update (Python `ff`).
void mke_expert_rls_scalar_step(const double* phi, int D, double pi, double gh_scale, double err,
                                double forgetting_factor, double* w, double* P_rowmajor);

/// Two-stage prediction in **original** y scale: ``((LLR|X)·w1+b1 + φ2·w2+b2) * y_std + y_mean``.
/// ``w1`` is **K + d_in** (LLR block then X block, same order as ``np.c_[LLR, X] @ w1``).
double two_stage_dif_predict(const double* llr, int K, const double* x, int d_in, const double* w1, double b1,
                             const double* phi2, int D2, const double* w2, double b2, double y_mean,
                             double y_std);

/// Batched ``two_stage_dif_predict``: ``llr`` **n×K**, ``X`` **n×d_in**, ``phi2`` **n×D2** row-major; ``y_out`` length ``n`` (original scale).
void two_stage_dif_predict_batch(const double* llr, int n, int K, const double* X, int d_in, const double* w1,
                                 double b1, const double* phi2, int D2, const double* w2, double b2, double y_mean,
                                 double y_std, double* y_out);

/// Row softmax on ``llr / (temperature + eps)`` — matches Python ``_probs_from_llr_matrix`` / ``classify`` (K≤8 path).
void router_softmax_from_llr(const double* llr, int K, double temperature, double eps, double* probs_out);

/// Shannon entropy ``-Σ p log(p+eps)`` (nats), matches ``_shannon_entropy``.
double mke_routing_entropy(const double* probs, int K, double eps);

/// ``MKERegressor.predict`` scalar path: softmax routing then ``Σ_k p_k · expert_mu[k]``. If ``out_entropy != nullptr``,
/// writes routing entropy (same as Python tuple[1]).
double mke_scalar_predict_from_llr(const double* llr, int K, double temperature, double eps,
                                   const double* expert_mu, double* out_entropy);

/// End-to-end ``TwoStageDIFRegressor.predict``-style combine: native ``batch_encode`` + ``score_matrix_use_field`` on
/// ``clf``, then stage-2 RFF on ``x`` and ``two_stage_dif_predict``. ``d_in`` must equal ``clf.d_latent``; ``K = len(labels)``.
double two_stage_dif_predict_with_clf(const cypha::CyphaInferModel& clf, const double* x_row_major, int d_in,
                                      const double* enc2_W, const double* enc2_b, int D2, const double* w1, double b1,
                                      const double* w2, double b2, double y_mean, double y_std);

/// Two-stage **ridge fit** given precomputed ``LLR`` (**n×K** row-major), ``X`` (**n×d_in**), and raw targets
/// ``y_raw`` (length ``n``). Matches ``TwoStageDIFRegressor.fit`` after ``LLR`` is fixed: stage-1 design
/// ``[LLR|X|1]`` with ``(λ₁·n)·I`` on **all** coefficients (including bias); stage-2 RFF φ from ``enc2_*`` with
/// ``(λ₂·n)·I`` on all coefficients. Uses ``y_mean`` / ``y_std`` like Python (internally clamps std to **1e-8**).
/// ``out_w1`` length ``K+d_in``, ``out_w2`` length ``D2``.
bool two_stage_dif_ridge_fit_from_llr(const double* llr_rowmajor, int n, int K, const double* X_rowmajor, int d_in,
                                      const double* y_raw, double y_mean, double y_std, double lam1, double lam2,
                                      const double* enc2_W, const double* enc2_b, int D2, double* out_w1, double* out_b1,
                                      double* out_w2, double* out_b2);

/// One expert's scalar / vector target EMAs (mirrors `DIFRegressor` `_expert_*` dicts).
struct ExpertTargetStats {
  std::vector<double> mu;
  double var_ema{0.0};
  int n_updates{0};
};

}  // namespace cypha::regression
