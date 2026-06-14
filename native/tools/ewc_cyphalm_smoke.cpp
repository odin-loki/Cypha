/// Smoke test for CyphaLM char-LSTM EWC on embed, Wx/Wh, and lm_head Wy/by.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"

namespace {

void train_sequence(cypha::cyphalm::CyphaLMModel& model, const std::vector<int>& ids, int steps) {
  model.reset_context();
  const int n = std::min(steps, static_cast<int>(ids.size()) - 1);
  for (int i = 0; i < n; ++i) {
    (void)model.train_step(static_cast<std::uint32_t>(ids[static_cast<std::size_t>(i)]),
                           static_cast<std::uint32_t>(ids[static_cast<std::size_t>(i + 1)]));
  }
}

double l2_drift(const std::vector<double>& a, const std::vector<double>& b) {
  assert(a.size() == b.size());
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = a[i] - b[i];
    sum += d * d;
  }
  return std::sqrt(sum);
}

struct EmbedHeadDrift {
  double embed{0.0};
  double lm_w{0.0};
  double lm_b{0.0};
  double total{0.0};
};

EmbedHeadDrift measure_embed_head_drift(const cypha::cyphalm::CharLSTMHead& lstm,
                                        const std::vector<double>& anchor_e,
                                        const std::vector<double>& anchor_wy,
                                        const std::vector<double>& anchor_by) {
  EmbedHeadDrift out;
  out.embed = l2_drift(lstm.E, anchor_e);
  out.lm_w = l2_drift(lstm.Wy, anchor_wy);
  out.lm_b = l2_drift(lstm.by, anchor_by);
  out.total = out.embed + out.lm_w + out.lm_b;
  return out;
}

double probe_forgetting(double ewc_lambda) {
  cypha::cyphalm::CyphaLMConfig cfg;
  cfg.vocab_size = 32;
  cfg.lstm_hidden = 16;
  cfg.context_mode = cypha::cyphalm::ContextMode::CharLstm;
  cfg.lstm_lr = 0.05;
  cfg.ewc_lambda = ewc_lambda;
  cfg.seed = 11;

  const std::vector<int> task_a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const std::vector<int> task_b = {20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

  cypha::cyphalm::CyphaLMModel model(cfg);
  train_sequence(model, task_a, 40);
  model.ewc_snapshot();
  train_sequence(model, task_b, 40);
  return model.ewc_penalty();
}

EmbedHeadDrift probe_embed_head_drift(double ewc_lambda) {
  cypha::cyphalm::CyphaLMConfig cfg;
  cfg.vocab_size = 32;
  cfg.lstm_hidden = 16;
  cfg.context_mode = cypha::cyphalm::ContextMode::CharLstm;
  cfg.lstm_lr = 0.05;
  cfg.ewc_lambda = ewc_lambda;
  cfg.seed = 13;

  const std::vector<int> task_a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const std::vector<int> task_b = {20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

  cypha::cyphalm::CyphaLMModel model(cfg);
  train_sequence(model, task_a, 40);
  const auto* lstm = model.char_lstm();
  assert(lstm != nullptr);
  const auto anchor_e = lstm->E;
  const auto anchor_wy = lstm->Wy;
  const auto anchor_by = lstm->by;
  model.ewc_snapshot();
  assert(model.ewc_regularizer().covers_embed_and_head());
  train_sequence(model, task_b, 40);
  return measure_embed_head_drift(*model.char_lstm(), anchor_e, anchor_wy, anchor_by);
}

}  // namespace

int main() {
  cypha::cyphalm::CyphaLMConfig cfg;
  cfg.vocab_size = 32;
  cfg.lstm_hidden = 16;
  cfg.context_mode = cypha::cyphalm::ContextMode::CharLstm;
  cfg.lstm_lr = 0.05;
  cfg.ewc_lambda = 0.5;
  cfg.seed = 7;

  cypha::cyphalm::CyphaLMModel model(cfg);
  const std::vector<int> warm = {1, 2, 3, 4, 5, 6};
  train_sequence(model, warm, 4);
  model.ewc_snapshot();
  assert(model.ewc_regularizer().covers_embed_and_head());
  assert(model.ewc_penalty() == 0.0);

  const auto m = model.train_step(7, 8);
  assert(m.ewc_penalty > 0.0);

  const double baseline_drift = probe_forgetting(0.0);
  const double ewc_drift = probe_forgetting(0.5);
  assert(baseline_drift > 1e-12);
  assert(ewc_drift < baseline_drift);

  const auto baseline_embed_head = probe_embed_head_drift(0.0);
  const auto ewc_embed_head = probe_embed_head_drift(0.5);
  assert(baseline_embed_head.total > 1e-12);
  assert(ewc_embed_head.total < baseline_embed_head.total);
  assert(ewc_embed_head.embed < baseline_embed_head.embed);
  assert(ewc_embed_head.lm_w < baseline_embed_head.lm_w);

  cypha::cyphalm::CyphaLMConfig hybrid_cfg;
  hybrid_cfg.vocab_size = 32;
  hybrid_cfg.lstm_hidden = 16;
  hybrid_cfg.field_dim = 32;
  hybrid_cfg.d_embed = 16;
  hybrid_cfg.d_state = 16;
  hybrid_cfg.ssm_layers = 1;
  hybrid_cfg.context_mode = cypha::cyphalm::ContextMode::Hybrid;
  hybrid_cfg.ewc_lambda = 0.5;
  hybrid_cfg.seed = 17;
  cypha::cyphalm::CyphaLMModel hybrid_model(hybrid_cfg);
  train_sequence(hybrid_model, warm, 4);
  hybrid_model.ewc_snapshot();
  assert(hybrid_model.hybrid_ewc_regularizer().covers_gria_alpha());
  assert(hybrid_model.hybrid_ewc_regularizer().covers_ssm_alpha());
  const auto hybrid_m = hybrid_model.train_step(7, 8);
  assert(hybrid_m.ewc_penalty > 0.0);

  std::printf(
      "ewc_cyphalm_smoke: baseline_penalty=%.6e ewc_penalty=%.6e "
      "embed_drift=%.6e->%.6e lm_w_drift=%.6e->%.6e PASS\n",
      baseline_drift, ewc_drift, baseline_embed_head.embed, ewc_embed_head.embed, baseline_embed_head.lm_w,
      ewc_embed_head.lm_w);
  return 0;
}
