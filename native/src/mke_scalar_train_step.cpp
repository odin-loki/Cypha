#include "cypha/mke_scalar_train_step.hpp"

#include <cmath>
#include <stdexcept>

#include "cypha/infer_cpu.hpp"
#include "cypha/memory_train.hpp"
#include "cypha/regression_stub.hpp"
#include "cypha/replay_buffer.hpp"
#include "cypha/train_step_vector.hpp"

namespace cypha::regression {

double mke_scalar_train_step_from_phi(
    CyphaInferModel& infer, CyphaDifMemoryState& mem, ReplayBuffer& replay, const double* phi, int d_rff, double y,
    std::unordered_map<std::string, std::vector<double>>& w_by_label,
    std::unordered_map<std::string, std::vector<double>>& p_by_label, const double* gh_scales,
    double temperature, double forgetting_factor, double pi_floor, const TrainStepParams& tsp, double world_lr,
    double delta_lr, double ood_sigma, std::mt19937& rng, int& enc_updates, TrainStepExtras* extras,
    const std::string* router_train_label_override, double softmax_eps, MkeScalarTrainStepOutputs* out) {
  if (d_rff != infer.d_latent) {
    throw std::runtime_error("mke_scalar_train_step_from_phi: d_rff != infer.d_latent");
  }

  mem.refresh_world_log_norm_from_v();

  std::vector<double> llr;
  score_matrix_use_field(infer, phi, 1, llr);
  const int K = static_cast<int>(infer.labels.size());
  if (static_cast<int>(llr.size()) != K) {
    throw std::runtime_error("mke_scalar_train_step_from_phi: llr K mismatch");
  }

  std::vector<double> p(static_cast<std::size_t>(K));
  router_softmax_from_llr(llr.data(), K, temperature, softmax_eps, p.data());

  double y_hat = 0.0;
  for (int i = 0; i < K; ++i) {
    const std::string& lbl = infer.labels[static_cast<std::size_t>(i)];
    auto wit = w_by_label.find(lbl);
    if (wit == w_by_label.end() || static_cast<int>(wit->second.size()) != d_rff) {
      throw std::runtime_error("mke_scalar_train_step_from_phi: missing w for label " + lbl);
    }
    double dp = 0.0;
    for (int t = 0; t < d_rff; ++t) {
      dp += wit->second[static_cast<std::size_t>(t)] * phi[t];
    }
    y_hat += p[static_cast<std::size_t>(i)] * dp;
  }
  const double err = y - y_hat;
  const double err_sq = err * err;

  for (int i = 0; i < K; ++i) {
    if (p[static_cast<std::size_t>(i)] < pi_floor) {
      continue;
    }
    const std::string& lbl = infer.labels[static_cast<std::size_t>(i)];
    auto wit = w_by_label.find(lbl);
    auto pit = p_by_label.find(lbl);
    if (wit == w_by_label.end() || pit == p_by_label.end()) {
      throw std::runtime_error("mke_scalar_train_step_from_phi: missing P for label " + lbl);
    }
    const double gh = (gh_scales != nullptr) ? gh_scales[static_cast<std::size_t>(i)] : 1.0;
    mke_expert_rls_scalar_step(phi, d_rff, p[static_cast<std::size_t>(i)], gh, err, forgetting_factor,
                               wit->second.data(), pit->second.data());
  }

  std::string router_label;
  if (router_train_label_override != nullptr && !router_train_label_override->empty()) {
    router_label = *router_train_label_override;
  } else {
    int best = 0;
    for (int i = 1; i < K; ++i) {
      if (p[static_cast<std::size_t>(i)] > p[static_cast<std::size_t>(best)]) {
        best = i;
      }
    }
    router_label = infer.labels[static_cast<std::size_t>(best)];
  }

  const double router_loss =
      dif_train_step_vector(infer, mem, replay, phi, d_rff, router_label, world_lr, delta_lr, world_lr, delta_lr,
                            ood_sigma, tsp, rng, enc_updates, nullptr, extras);

  if (out != nullptr) {
    out->err_sq = err_sq;
    out->y_hat = y_hat;
    out->router_loss = router_loss;
    out->router_label = std::move(router_label);
  }
  return err_sq;
}

double mke_scalar_train_step(
    CyphaInferModel& infer, CyphaDifMemoryState& mem, ReplayBuffer& replay, const double* x, int d_in, double y,
    const double* W_rff_row_major, const double* b_rff, int d_rff,
    std::unordered_map<std::string, std::vector<double>>& w_by_label,
    std::unordered_map<std::string, std::vector<double>>& p_by_label, const double* gh_scales,
    double temperature, double forgetting_factor, double pi_floor, const TrainStepParams& tsp, double world_lr,
    double delta_lr, double ood_sigma, std::mt19937& rng, int& enc_updates, TrainStepExtras* extras,
    const std::string* router_train_label_override, double softmax_eps, MkeScalarTrainStepOutputs* out) {
  std::vector<double> phi(static_cast<std::size_t>(d_rff));
  rff_encode_batch_rowmajor(x, 1, d_in, W_rff_row_major, b_rff, d_rff, phi.data());
  return mke_scalar_train_step_from_phi(infer, mem, replay, phi.data(), d_rff, y, w_by_label, p_by_label, gh_scales,
                                        temperature, forgetting_factor, pi_floor, tsp, world_lr, delta_lr, ood_sigma,
                                        rng, enc_updates, extras, router_train_label_override, softmax_eps, out);
}

}  // namespace cypha::regression

/// Referenced by ``cypha_qt_stub`` so the linker retains this translation unit when linking ``cypha_core`` statically.
extern "C" int cypha_core_mke_scalar_train_step_link_touch() { return 1; }
