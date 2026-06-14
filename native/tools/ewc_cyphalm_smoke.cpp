/// Smoke test for CyphaLM char-LSTM EWC on Wx/Wh (W_ih/W_hh).
#include <cassert>
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
  assert(model.ewc_penalty() == 0.0);

  const auto m = model.train_step(7, 8);
  assert(m.ewc_penalty > 0.0);

  const double baseline_drift = probe_forgetting(0.0);
  const double ewc_drift = probe_forgetting(0.5);
  assert(baseline_drift > 1e-12);
  assert(ewc_drift < baseline_drift);

  std::printf("ewc_cyphalm_smoke: baseline_penalty=%.6e ewc_penalty=%.6e PASS\n", baseline_drift, ewc_drift);
  return 0;
}
