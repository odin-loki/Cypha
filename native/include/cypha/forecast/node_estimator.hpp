#pragma once

#include "cypha/cypha.hpp"
#include "cypha/forecast/dispute_data.hpp"
#include "cypha/classify/preprocessor.hpp"

#include <string>
#include <vector>

namespace cypha::forecast {

struct NodeEstimatorConfig {
  bool use_orf{true};
  int rff_dim{256};
  double ewc_lambda{2.0};
  int epochs{3};
};

struct NodeEstimatorResult {
  double train_accuracy{0.0};
  double eval_accuracy{0.0};
  int n_train{0};
  int n_eval{0};
  std::string detail;
};

/// Train CyphaDIF classifiers on MID dispute→escalation features (Phase 3).
class NodeEstimator {
 public:
  explicit NodeEstimator(NodeEstimatorConfig cfg = {});

  NodeEstimatorResult train_on_mid(const std::vector<DisputeRecord>& records, double eval_frac = 0.2);

  [[nodiscard]] const Cypha& model() const { return cypha_; }
  [[nodiscard]] Cypha& model() { return cypha_; }

 private:
  NodeEstimatorConfig cfg_;
  Cypha cypha_;
  PreprocessorState pre_;
};

}  // namespace cypha::forecast
