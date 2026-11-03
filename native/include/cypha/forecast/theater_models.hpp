#pragma once

#include "cypha/cypha.hpp"
#include "cypha/online/ewc_regularizer.hpp"
#include "cypha/forecast/event_vocabulary.hpp"
#include "cypha/online/replay_buffer.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace cypha::forecast {

/// Per-theater isolated CyphaDIF models with EWC + priority replay (Phase 7 / D16F).
struct TheaterModelConfig {
  double ewc_lambda{2.0};
  int replay_cap{5000};
  double rare_event_boost{5.0};
};

class TheaterModelBank {
 public:
  explicit TheaterModelBank(TheaterModelConfig cfg = {});

  [[nodiscard]] Cypha* model_for(Theater t);
  [[nodiscard]] const Cypha* model_for(Theater t) const;

  void train_step(Theater t, const double* x, int n, const std::string& label, bool rare_event = false);
  void ewc_snapshot_all();

 private:
  TheaterModelConfig cfg_;
  std::unordered_map<int, std::unique_ptr<Cypha>> models_;
  std::unordered_map<int, std::unique_ptr<EwcRegularizer>> ewc_;
  std::unordered_map<int, std::vector<std::vector<double>>> rare_replay_;
};

}  // namespace cypha::forecast
