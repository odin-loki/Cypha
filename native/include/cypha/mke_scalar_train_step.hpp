#pragma once

/// One scalar ``MKERegressor.train_step`` composition in native (RFF φ is already ``d_latent``-dim features).
///
/// Pipeline: ``refresh_world_log_norm_from_v`` → ``score_matrix_use_field(φ)`` → routing softmax →
/// weighted expert RLS (``mke_expert_rls_scalar_step``) → ``dif_train_step_vector`` on φ with the router label.
/// Matches ``mke_train_step_parity`` golden fixture (experts + router).
///
/// **Router label:** if ``router_train_label_override`` is non-null and non-empty, uses it (Python ``infer`` /
/// sidecar). Otherwise uses **argmax routing softmax** ``p`` (Python fallback when pred is ``__unknown__``).

#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace cypha {

class ReplayBuffer;
struct CyphaDifMemoryState;
struct CyphaInferModel;
struct TrainStepExtras;
struct TrainStepParams;

}  // namespace cypha

namespace cypha::regression {

struct MkeScalarTrainStepOutputs {
  double err_sq{0.0};
  double y_hat{0.0};
  double router_loss{0.0};
  std::string router_label;
};

/// RFF-encode ``x`` then delegate to ``mke_scalar_train_step_from_phi``.
double mke_scalar_train_step(
    CyphaInferModel& infer, CyphaDifMemoryState& mem, ReplayBuffer& replay, const double* x, int d_in, double y,
    const double* W_rff_row_major, const double* b_rff, int d_rff,
    std::unordered_map<std::string, std::vector<double>>& w_by_label,
    std::unordered_map<std::string, std::vector<double>>& p_by_label, const double* gh_scales,
    double temperature, double forgetting_factor, double pi_floor, const TrainStepParams& tsp, double world_lr,
    double delta_lr, double ood_sigma, std::mt19937& rng, int& enc_updates, TrainStepExtras* extras,
    const std::string* router_train_label_override, double softmax_eps, MkeScalarTrainStepOutputs* out);

/// Precomputed φ (length ``d_rff``, must equal ``infer.d_latent``). Returns squared error **before** updates
/// (same scalar as Python ``MKERegressor.train_step``). Fills ``out`` when non-null.
double mke_scalar_train_step_from_phi(
    CyphaInferModel& infer, CyphaDifMemoryState& mem, ReplayBuffer& replay, const double* phi, int d_rff, double y,
    std::unordered_map<std::string, std::vector<double>>& w_by_label,
    std::unordered_map<std::string, std::vector<double>>& p_by_label, const double* gh_scales,
    double temperature, double forgetting_factor, double pi_floor, const TrainStepParams& tsp, double world_lr,
    double delta_lr, double ood_sigma, std::mt19937& rng, int& enc_updates, TrainStepExtras* extras,
    const std::string* router_train_label_override, double softmax_eps, MkeScalarTrainStepOutputs* out);

}  // namespace cypha::regression
