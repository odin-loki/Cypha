#pragma once

#include "cypha/lm/cyphalm_model.hpp"
#include "cypha/forecast/dispute_data.hpp"
#include "cypha/forecast/event_vocabulary.hpp"
#include "cypha/forecast/gdelt_monitor.hpp"
#include "cypha/forecast/interpretability.hpp"
#include "cypha/forecast/node_estimator.hpp"
#include "cypha/forecast/rollout_tree.hpp"
#include "cypha/forecast/theater_models.hpp"
#include "cypha/forecast/views_scoring.hpp"

#include <filesystem>
#include <string>

namespace cypha::forecast {

struct ForecastPipelineConfig {
  NodeEstimatorConfig node_cfg;
  RolloutTreeConfig tree_cfg;
  TheaterModelConfig theater_cfg;
  int sequence_epochs{2};
  int vocab_size{4096};
};

struct ForecastPipelineResult {
  NodeEstimatorResult node_result;
  double sequence_eval_bpc{0.0};
  RolloutTree scenario_tree;
  std::vector<AnnotatedPath> paths;
  ViewsValidationResult views_validation;
  int drift_alarms{0};
  std::string detail;
};

/// End-to-end forecasting pipeline (Phases 2–9 orchestrator).
class ForecastPipeline {
 public:
  explicit ForecastPipeline(ForecastPipelineConfig cfg = {});

  ForecastPipelineResult run(const std::filesystem::path& data_dir);

  [[nodiscard]] const EventVocabulary& vocab() const { return vocab_; }
  [[nodiscard]] cyphalm::CyphaLMModel* sequence_model() { return seq_.get(); }

 private:
  ForecastPipelineConfig cfg_;
  EventVocabulary vocab_;
  std::unique_ptr<cyphalm::CyphaLMModel> seq_;
  TheaterModelBank theaters_;
  GdeltMonitor monitor_;
};

}  // namespace cypha::forecast
