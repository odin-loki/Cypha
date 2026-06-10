#pragma once

#include "cypha/cyphalm/char_lstm.hpp"
#include "cypha/cyphalm/gria_lowrank.hpp"

#include <cstdint>
#include <vector>

namespace cypha {
namespace cyphalm {

struct CyphaLMBatchConfig {
  int vocab_size{256};
  int lstm_hidden{128};
  int field_dim{160};
  int gria_rank{32};
  double lstm_lr{0.05};
  double gria_lr{0.05};
  double blend_logit{0.0};
  std::uint64_t seed{42};
};

/// Batched CyphaLM Tier-0 hot path: char LSTM + low-rank GRIA over parallel batch rows.
class CyphaLMBatch {
 public:
  CyphaLMBatchConfig cfg;
  CharLSTMHead lstm;
  GRIALowRank gria;

  explicit CyphaLMBatch(CyphaLMBatchConfig config = {});

  /// Train ``bptt_len`` steps on ``batch_size`` sequences (row-major token layout).
  /// ``token_ids`` / ``next_ids`` length = batch_size * bptt_len.
  /// ``field_vecs`` length = batch_size * bptt_len * field_dim (GRIA input per step).
  /// Returns mean cross-entropy loss over batch * steps.
  double train_sequence_batch(const int* token_ids, const int* next_ids, const double* field_vecs, int batch_size,
                              int bptt_len);

 private:
  std::vector<std::vector<double>> h_states_;
  std::vector<std::vector<double>> c_states_;
};

}  // namespace cyphalm
}  // namespace cypha
