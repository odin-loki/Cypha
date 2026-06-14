#include "cypha/cyphalm/cyphalm_ewc_regularizer.hpp"

#include <cmath>

#include <nlohmann/json.hpp>

namespace cypha::cyphalm {

namespace {

constexpr double kFisherEps = 1e-8;

void build_diagonal_fisher_from_anchor(const std::vector<double>& anchor, std::vector<double>& fisher_out) {
  fisher_out.resize(anchor.size());
  for (std::size_t i = 0; i < anchor.size(); ++i) {
    fisher_out[i] = anchor[i] * anchor[i] + kFisherEps;
  }
}

double squared_penalty(const std::vector<double>& theta, const std::vector<double>& anchor,
                       const std::vector<double>& fisher) {
  if (theta.size() != anchor.size() || theta.size() != fisher.size()) {
    return 0.0;
  }
  double sum = 0.0;
  for (std::size_t i = 0; i < theta.size(); ++i) {
    const double d = theta[i] - anchor[i];
    sum += fisher[i] * d * d;
  }
  return 0.5 * sum;
}

void pull_toward_anchor(std::vector<double>& theta, const std::vector<double>& anchor,
                        const std::vector<double>& fisher, double strength) {
  if (theta.size() != anchor.size() || theta.size() != fisher.size() || strength <= 0.0) {
    return;
  }
  for (std::size_t i = 0; i < theta.size(); ++i) {
    const double d = theta[i] - anchor[i];
    theta[i] -= strength * fisher[i] * d;
  }
}

void observe_block_grads(const std::vector<double>& grad, std::vector<double>& fisher,
                         std::size_t& observations) {
  if (grad.size() != fisher.size()) {
    return;
  }
  const std::size_t next_count = observations + 1;
  const double inv_n = 1.0 / static_cast<double>(next_count);
  for (std::size_t i = 0; i < fisher.size(); ++i) {
    const double g2 = grad[i] * grad[i];
    fisher[i] = fisher[i] * static_cast<double>(observations) * inv_n + g2 * inv_n;
    if (fisher[i] < kFisherEps) {
      fisher[i] = kFisherEps;
    }
  }
  observations = next_count;
}

std::vector<double> json_to_vec(const nlohmann::json& j) {
  std::vector<double> out;
  if (!j.is_array()) {
    return out;
  }
  out.reserve(j.size());
  for (const auto& x : j) {
    out.push_back(x.get<double>());
  }
  return out;
}

}  // namespace

void CyphaLMEwcRegularizer::snapshot(const CharLSTMHead& lstm) {
  anchor_E_ = lstm.E;
  anchor_Wx_ = lstm.Wx;
  anchor_Wh_ = lstm.Wh;
  anchor_Wy_ = lstm.Wy;
  anchor_by_ = lstm.by;
  build_diagonal_fisher_from_anchor(anchor_E_, fisher_E_);
  build_diagonal_fisher_from_anchor(anchor_Wx_, fisher_Wx_);
  build_diagonal_fisher_from_anchor(anchor_Wh_, fisher_Wh_);
  build_diagonal_fisher_from_anchor(anchor_Wy_, fisher_Wy_);
  build_diagonal_fisher_from_anchor(anchor_by_, fisher_by_);
  grad_observations_ = 0;
}

void CyphaLMEwcRegularizer::observe_grads(const CharLSTMGrad& grads) {
  if (!has_snapshot()) {
    return;
  }
  const std::size_t next_count = grad_observations_ + 1;
  const double inv_n = 1.0 / static_cast<double>(next_count);

  auto update_block = [&](const std::vector<double>& grad, std::vector<double>& fisher) {
    if (grad.size() != fisher.size()) {
      return;
    }
    for (std::size_t i = 0; i < fisher.size(); ++i) {
      const double g2 = grad[i] * grad[i];
      fisher[i] = fisher[i] * static_cast<double>(grad_observations_) * inv_n + g2 * inv_n;
      if (fisher[i] < kFisherEps) {
        fisher[i] = kFisherEps;
      }
    }
  };

  update_block(grads.dE, fisher_E_);
  update_block(grads.dWx, fisher_Wx_);
  update_block(grads.dWh, fisher_Wh_);
  update_block(grads.dWy, fisher_Wy_);
  update_block(grads.dby, fisher_by_);
  grad_observations_ = next_count;
}

double CyphaLMEwcRegularizer::penalty(const CharLSTMHead& lstm) const {
  if (!has_snapshot()) {
    return 0.0;
  }
  return squared_penalty(lstm.E, anchor_E_, fisher_E_) +
         squared_penalty(lstm.Wx, anchor_Wx_, fisher_Wx_) +
         squared_penalty(lstm.Wh, anchor_Wh_, fisher_Wh_) +
         squared_penalty(lstm.Wy, anchor_Wy_, fisher_Wy_) +
         squared_penalty(lstm.by, anchor_by_, fisher_by_);
}

void CyphaLMEwcRegularizer::apply_pull(CharLSTMHead& lstm, double ewc_lambda, double lr) const {
  if (!has_snapshot() || ewc_lambda <= 0.0 || lr <= 0.0) {
    return;
  }
  const double strength = ewc_lambda * lr;
  pull_toward_anchor(lstm.E, anchor_E_, fisher_E_, strength);
  pull_toward_anchor(lstm.Wx, anchor_Wx_, fisher_Wx_, strength);
  pull_toward_anchor(lstm.Wh, anchor_Wh_, fisher_Wh_, strength);
  pull_toward_anchor(lstm.Wy, anchor_Wy_, fisher_Wy_, strength);
  pull_toward_anchor(lstm.by, anchor_by_, fisher_by_, strength);
}

nlohmann::json CyphaLMEwcRegularizer::get_state() const {
  return {
      {"anchor_E", anchor_E_},
      {"anchor_Wx", anchor_Wx_},
      {"anchor_Wh", anchor_Wh_},
      {"anchor_Wy", anchor_Wy_},
      {"anchor_by", anchor_by_},
      {"fisher_E", fisher_E_},
      {"fisher_Wx", fisher_Wx_},
      {"fisher_Wh", fisher_Wh_},
      {"fisher_Wy", fisher_Wy_},
      {"fisher_by", fisher_by_},
      {"grad_observations", grad_observations_},
  };
}

void CyphaLMEwcRegularizer::set_state(const nlohmann::json& state) {
  if (state.contains("anchor_E")) {
    anchor_E_ = json_to_vec(state.at("anchor_E"));
  }
  if (state.contains("anchor_Wx")) {
    anchor_Wx_ = json_to_vec(state.at("anchor_Wx"));
  }
  if (state.contains("anchor_Wh")) {
    anchor_Wh_ = json_to_vec(state.at("anchor_Wh"));
  }
  if (state.contains("anchor_Wy")) {
    anchor_Wy_ = json_to_vec(state.at("anchor_Wy"));
  }
  if (state.contains("anchor_by")) {
    anchor_by_ = json_to_vec(state.at("anchor_by"));
  }
  if (state.contains("fisher_E")) {
    fisher_E_ = json_to_vec(state.at("fisher_E"));
  }
  if (state.contains("fisher_Wx")) {
    fisher_Wx_ = json_to_vec(state.at("fisher_Wx"));
  }
  if (state.contains("fisher_Wh")) {
    fisher_Wh_ = json_to_vec(state.at("fisher_Wh"));
  }
  if (state.contains("fisher_Wy")) {
    fisher_Wy_ = json_to_vec(state.at("fisher_Wy"));
  }
  if (state.contains("fisher_by")) {
    fisher_by_ = json_to_vec(state.at("fisher_by"));
  }
  if (state.contains("grad_observations")) {
    grad_observations_ = state.at("grad_observations").get<std::size_t>();
  }
}

void HybridEwcRegularizer::snapshot(const CharLSTMHead* lstm, const CellAISSM* ssm,
                                    const GRIALowRank* gria) {
  if (lstm != nullptr) {
    lstm_.snapshot(*lstm);
  }
  anchor_ssm_alpha_.clear();
  fisher_ssm_alpha_.clear();
  if (ssm != nullptr) {
    anchor_ssm_alpha_ = ssm->multiscale_alpha();
    build_diagonal_fisher_from_anchor(anchor_ssm_alpha_, fisher_ssm_alpha_);
    ssm_grad_observations_ = 0;
  }
  anchor_gria_alpha_.clear();
  fisher_gria_alpha_.clear();
  anchor_gria_U_.clear();
  fisher_gria_U_.clear();
  anchor_gria_V_.clear();
  fisher_gria_V_.clear();
  if (gria != nullptr) {
    anchor_gria_alpha_ = gria->alpha;
    build_diagonal_fisher_from_anchor(anchor_gria_alpha_, fisher_gria_alpha_);
    gria_grad_observations_ = 0;
    anchor_gria_U_ = gria->U;
    build_diagonal_fisher_from_anchor(anchor_gria_U_, fisher_gria_U_);
    anchor_gria_V_ = gria->V;
    build_diagonal_fisher_from_anchor(anchor_gria_V_, fisher_gria_V_);
    gria_uv_grad_observations_ = 0;
  }
  anchor_ssm_w_fast_.clear();
  fisher_ssm_w_fast_.clear();
  if (ssm != nullptr && !ssm->w_fast_layer0().empty()) {
    anchor_ssm_w_fast_ = ssm->w_fast_layer0();
    build_diagonal_fisher_from_anchor(anchor_ssm_w_fast_, fisher_ssm_w_fast_);
    ssm_w_fast_grad_observations_ = 0;
  }
}

void HybridEwcRegularizer::observe_grads(const HybridEwcGradStub& grads) {
  if (grads.has_lstm) {
    lstm_.observe_grads(grads.lstm);
  }
  if (!anchor_ssm_alpha_.empty() && !grads.d_ssm_alpha.empty()) {
    observe_block_grads(grads.d_ssm_alpha, fisher_ssm_alpha_, ssm_grad_observations_);
  }
  if (!anchor_gria_alpha_.empty() && !grads.d_gria_alpha.empty()) {
    observe_block_grads(grads.d_gria_alpha, fisher_gria_alpha_, gria_grad_observations_);
  }
  if (!anchor_gria_U_.empty() && !anchor_gria_V_.empty() && !grads.d_gria_U.empty() &&
      !grads.d_gria_V.empty()) {
    const std::size_t next_count = gria_uv_grad_observations_ + 1;
    const double inv_n = 1.0 / static_cast<double>(next_count);
    auto update_block = [&](const std::vector<double>& grad, std::vector<double>& fisher) {
      if (grad.size() != fisher.size()) {
        return;
      }
      for (std::size_t i = 0; i < fisher.size(); ++i) {
        const double g2 = grad[i] * grad[i];
        fisher[i] = fisher[i] * static_cast<double>(gria_uv_grad_observations_) * inv_n + g2 * inv_n;
        if (fisher[i] < kFisherEps) {
          fisher[i] = kFisherEps;
        }
      }
    };
    update_block(grads.d_gria_U, fisher_gria_U_);
    update_block(grads.d_gria_V, fisher_gria_V_);
    gria_uv_grad_observations_ = next_count;
  }
  if (!anchor_ssm_w_fast_.empty() && !grads.d_ssm_w_fast.empty()) {
    observe_block_grads(grads.d_ssm_w_fast, fisher_ssm_w_fast_, ssm_w_fast_grad_observations_);
  }
}

double HybridEwcRegularizer::penalty(const CharLSTMHead* lstm, const CellAISSM* ssm,
                                     const GRIALowRank* gria) const {
  double sum = 0.0;
  if (lstm != nullptr && lstm_.has_snapshot()) {
    sum += lstm_.penalty(*lstm);
  }
  if (ssm != nullptr && !anchor_ssm_alpha_.empty()) {
    sum += squared_penalty(ssm->multiscale_alpha(), anchor_ssm_alpha_, fisher_ssm_alpha_);
  }
  if (gria != nullptr && !anchor_gria_alpha_.empty()) {
    sum += squared_penalty(gria->alpha, anchor_gria_alpha_, fisher_gria_alpha_);
  }
  if (gria != nullptr && !anchor_gria_U_.empty()) {
    sum += squared_penalty(gria->U, anchor_gria_U_, fisher_gria_U_);
  }
  if (gria != nullptr && !anchor_gria_V_.empty()) {
    sum += squared_penalty(gria->V, anchor_gria_V_, fisher_gria_V_);
  }
  if (ssm != nullptr && !anchor_ssm_w_fast_.empty()) {
    sum += squared_penalty(ssm->w_fast_layer0(), anchor_ssm_w_fast_, fisher_ssm_w_fast_);
  }
  return sum;
}

void HybridEwcRegularizer::apply_pull(CharLSTMHead* lstm, CellAISSM* ssm, GRIALowRank* gria,
                                      double ewc_lambda, double lstm_lr, double gria_lr,
                                      double ssm_lr) const {
  if (ewc_lambda <= 0.0) {
    return;
  }
  if (lstm != nullptr && lstm_.has_snapshot() && lstm_lr > 0.0) {
    lstm_.apply_pull(*lstm, ewc_lambda, lstm_lr);
  }
  if (ssm != nullptr && !anchor_ssm_alpha_.empty() && ssm_lr > 0.0) {
    pull_toward_anchor(ssm->multiscale_alpha_mut(), anchor_ssm_alpha_, fisher_ssm_alpha_,
                       ewc_lambda * ssm_lr);
  }
  if (gria != nullptr && !anchor_gria_alpha_.empty() && gria_lr > 0.0) {
    pull_toward_anchor(gria->alpha, anchor_gria_alpha_, fisher_gria_alpha_, ewc_lambda * gria_lr);
  }
  if (gria != nullptr && !anchor_gria_U_.empty() && gria_lr > 0.0) {
    pull_toward_anchor(gria->U, anchor_gria_U_, fisher_gria_U_, ewc_lambda * gria_lr);
  }
  if (gria != nullptr && !anchor_gria_V_.empty() && gria_lr > 0.0) {
    pull_toward_anchor(gria->V, anchor_gria_V_, fisher_gria_V_, ewc_lambda * gria_lr);
  }
  if (ssm != nullptr && !anchor_ssm_w_fast_.empty() && ssm_lr > 0.0) {
    pull_toward_anchor(ssm->w_fast_layer0_mut(), anchor_ssm_w_fast_, fisher_ssm_w_fast_,
                       ewc_lambda * ssm_lr);
  }
}

nlohmann::json HybridEwcRegularizer::get_state() const {
  return {
      {"lstm", lstm_.get_state()},
      {"anchor_ssm_alpha", anchor_ssm_alpha_},
      {"anchor_gria_alpha", anchor_gria_alpha_},
      {"anchor_gria_U", anchor_gria_U_},
      {"anchor_gria_V", anchor_gria_V_},
      {"anchor_ssm_w_fast", anchor_ssm_w_fast_},
      {"fisher_ssm_alpha", fisher_ssm_alpha_},
      {"fisher_gria_alpha", fisher_gria_alpha_},
      {"fisher_gria_U", fisher_gria_U_},
      {"fisher_gria_V", fisher_gria_V_},
      {"fisher_ssm_w_fast", fisher_ssm_w_fast_},
      {"ssm_grad_observations", ssm_grad_observations_},
      {"gria_grad_observations", gria_grad_observations_},
      {"gria_uv_grad_observations", gria_uv_grad_observations_},
      {"ssm_w_fast_grad_observations", ssm_w_fast_grad_observations_},
  };
}

void HybridEwcRegularizer::set_state(const nlohmann::json& state) {
  if (state.contains("lstm")) {
    lstm_.set_state(state.at("lstm"));
  }
  if (state.contains("anchor_ssm_alpha")) {
    anchor_ssm_alpha_ = json_to_vec(state.at("anchor_ssm_alpha"));
  }
  if (state.contains("anchor_gria_alpha")) {
    anchor_gria_alpha_ = json_to_vec(state.at("anchor_gria_alpha"));
  }
  if (state.contains("anchor_gria_U")) {
    anchor_gria_U_ = json_to_vec(state.at("anchor_gria_U"));
  }
  if (state.contains("anchor_gria_V")) {
    anchor_gria_V_ = json_to_vec(state.at("anchor_gria_V"));
  }
  if (state.contains("anchor_ssm_w_fast")) {
    anchor_ssm_w_fast_ = json_to_vec(state.at("anchor_ssm_w_fast"));
  }
  if (state.contains("fisher_ssm_alpha")) {
    fisher_ssm_alpha_ = json_to_vec(state.at("fisher_ssm_alpha"));
  }
  if (state.contains("fisher_gria_alpha")) {
    fisher_gria_alpha_ = json_to_vec(state.at("fisher_gria_alpha"));
  }
  if (state.contains("fisher_gria_U")) {
    fisher_gria_U_ = json_to_vec(state.at("fisher_gria_U"));
  }
  if (state.contains("fisher_gria_V")) {
    fisher_gria_V_ = json_to_vec(state.at("fisher_gria_V"));
  }
  if (state.contains("fisher_ssm_w_fast")) {
    fisher_ssm_w_fast_ = json_to_vec(state.at("fisher_ssm_w_fast"));
  }
  if (state.contains("ssm_grad_observations")) {
    ssm_grad_observations_ = state.at("ssm_grad_observations").get<std::size_t>();
  }
  if (state.contains("gria_grad_observations")) {
    gria_grad_observations_ = state.at("gria_grad_observations").get<std::size_t>();
  }
  if (state.contains("gria_uv_grad_observations")) {
    gria_uv_grad_observations_ = state.at("gria_uv_grad_observations").get<std::size_t>();
  }
  if (state.contains("ssm_w_fast_grad_observations")) {
    ssm_w_fast_grad_observations_ = state.at("ssm_w_fast_grad_observations").get<std::size_t>();
  }
}

}  // namespace cypha::cyphalm
