#pragma once

namespace cypha::cyphalm {

/// Native CyphaLM training / hybrid configuration (Tier 1 A1).
struct CyphaLMNativeConfig {
  int bptt_steps = 256;
  bool train_ssm = true;
  /// Maximum LSTM blend weight; SSM-first cap keeps alpha <= 1 - max_lstm_weight.
  double max_lstm_weight = 0.5;
};

/// Cap hybrid blend alpha so LSTM weight never exceeds max_lstm_weight.
/// raw_alpha is GRIA/SSM weight in [0, 1]; returns capped value.
inline double capped_hybrid_alpha(double raw_alpha, const CyphaLMNativeConfig& cfg) {
  const double cap = 1.0 - cfg.max_lstm_weight;
  if (raw_alpha > cap) {
    return cap;
  }
  if (raw_alpha < 0.0) {
    return 0.0;
  }
  return raw_alpha;
}

}  // namespace cypha::cyphalm
