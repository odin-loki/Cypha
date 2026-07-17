#include "cypha/mke_scalar_train_step.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

#include "cypha/em_step.hpp"
#include "cypha/infer_cpu.hpp"
#include "cypha/memory_train.hpp"
#include "cypha/regression_stub.hpp"
#include "cypha/replay_buffer.hpp"
#include "cypha/train_step_vector.hpp"

namespace cypha::regression {

namespace {

constexpr double kMkeEmSigma = 0.5;
constexpr double kMkeEmTemperature = 0.25;
constexpr int kMkeRouteWarmupSteps = 200;
constexpr double kMkeRouteWarmupTempScale = 4.0;
constexpr double kMkePInit = 1000.0;
constexpr double kMkeWInitScale = 0.01;

void mke_ensure_expert_state(const CyphaInferModel& infer, int d_rff,
                             std::unordered_map<std::string, std::vector<double>>& w_by_label,
                             std::unordered_map<std::string, std::vector<double>>& p_by_label,
                             std::mt19937& rng) {
  const int k = static_cast<int>(infer.labels.size());
  const std::size_t pd = static_cast<std::size_t>(d_rff) * static_cast<std::size_t>(d_rff);
  std::normal_distribution<double> w_dist(0.0, kMkeWInitScale);
  for (int i = 0; i < k; ++i) {
    const std::string& lbl = infer.labels[static_cast<std::size_t>(i)];
    if (w_by_label.find(lbl) == w_by_label.end()) {
      std::vector<double> w(static_cast<std::size_t>(d_rff));
      for (int t = 0; t < d_rff; ++t) {
        w[static_cast<std::size_t>(t)] = w_dist(rng);
      }
      w_by_label[lbl] = std::move(w);
    }
    if (p_by_label.find(lbl) == p_by_label.end()) {
      std::vector<double> p_mat(pd, 0.0);
      for (int t = 0; t < d_rff; ++t) {
        p_mat[static_cast<std::size_t>(t) * static_cast<std::size_t>(d_rff) + static_cast<std::size_t>(t)] =
            kMkePInit;
      }
      p_by_label[lbl] = std::move(p_mat);
    }
  }
}

double effective_routing_temperature(double temperature, const TrainStepExtras* extras) {
  double route_temp = temperature;
  if (extras != nullptr && extras->total_steps != nullptr) {
    const int step = *extras->total_steps;
    if (step >= 0 && step < kMkeRouteWarmupSteps) {
      const double t_hi = temperature * kMkeRouteWarmupTempScale;
      const double frac = static_cast<double>(step) / static_cast<double>(kMkeRouteWarmupSteps);
      route_temp = t_hi + (temperature - t_hi) * frac;
    }
  }
  return route_temp;
}

}  // namespace

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

  mke_ensure_expert_state(infer, d_rff, w_by_label, p_by_label, rng);

  mem.refresh_world_log_norm_from_v();

  std::vector<double> llr;
  score_matrix_use_field(infer, phi, 1, llr);
  const int K = static_cast<int>(infer.labels.size());
  if (static_cast<int>(llr.size()) != K) {
    throw std::runtime_error("mke_scalar_train_step_from_phi: llr K mismatch");
  }

  const double route_temp = effective_routing_temperature(temperature, extras);

  std::vector<double> p(static_cast<std::size_t>(K));
  router_softmax_from_llr(llr.data(), K, route_temp, softmax_eps, p.data());

  std::vector<double> dp(static_cast<std::size_t>(K));
  double y_hat = 0.0;
  for (int i = 0; i < K; ++i) {
    const std::string& lbl = infer.labels[static_cast<std::size_t>(i)];
    auto wit = w_by_label.find(lbl);
    if (wit == w_by_label.end() || static_cast<int>(wit->second.size()) != d_rff) {
      throw std::runtime_error("mke_scalar_train_step_from_phi: missing w for label " + lbl);
    }
    double dot = 0.0;
    for (int t = 0; t < d_rff; ++t) {
      dot += wit->second[static_cast<std::size_t>(t)] * phi[t];
    }
    dp[static_cast<std::size_t>(i)] = dot;
    y_hat += p[static_cast<std::size_t>(i)] * dot;
  }
  const double err = y - y_hat;
  const double err_sq = err * err;

  const double inv_two_sigma_sq = 0.5 / (kMkeEmSigma * kMkeEmSigma);
  std::vector<double> loglik(static_cast<std::size_t>(K));
  for (int i = 0; i < K; ++i) {
    const double resid = y - dp[static_cast<std::size_t>(i)];
    loglik[static_cast<std::size_t>(i)] = -inv_two_sigma_sq * resid * resid;
  }

  const double em_eps = std::max(pi_floor, kEmEps);
  const double uni_mix = std::min(std::max(pi_floor, kEmEps), 1.0);
  const double uni = 1.0 / static_cast<double>(K);
  std::vector<double> em_prior(static_cast<std::size_t>(K));
  for (int i = 0; i < K; ++i) {
    em_prior[static_cast<std::size_t>(i)] = (1.0 - uni_mix) * p[static_cast<std::size_t>(i)] + uni_mix * uni;
  }

  std::vector<double> r(static_cast<std::size_t>(K));
  responsibilities(loglik.data(), em_prior.data(), K, kMkeEmTemperature, em_eps, r.data());

  for (int i = 0; i < K; ++i) {
    const std::string& lbl = infer.labels[static_cast<std::size_t>(i)];
    auto wit = w_by_label.find(lbl);
    auto pit = p_by_label.find(lbl);
    if (wit == w_by_label.end() || pit == p_by_label.end()) {
      throw std::runtime_error("mke_scalar_train_step_from_phi: missing P for label " + lbl);
    }
    const double gh = (gh_scales != nullptr) ? gh_scales[static_cast<std::size_t>(i)] : 1.0;
    mke_expert_rls_scalar_step(phi, d_rff, r[static_cast<std::size_t>(i)], gh, err, forgetting_factor,
                               wit->second.data(), pit->second.data());
  }

  std::string router_label;
  if (router_train_label_override != nullptr && !router_train_label_override->empty()) {
    router_label = *router_train_label_override;
  } else {
    int best = 0;
    for (int i = 1; i < K; ++i) {
      if (r[static_cast<std::size_t>(i)] > r[static_cast<std::size_t>(best)]) {
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
    out->r = r;
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
