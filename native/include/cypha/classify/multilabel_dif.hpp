#pragma once

#include "cypha/infer_cpu.hpp"
#include "cypha/memory_train.hpp"
#include "cypha/replay_buffer.hpp"
#include "cypha/train_step_vector.hpp"

#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace cypha {

/// Parameters for ``MultiLabelDIF`` (Python ``MultiLabelDIF`` defaults).
struct MultiLabelDifParams {
  int input_dim{8};
  int field_dim{24};
  int master_seed{42};
  double world_lr{0.008};
  double delta_lr{0.05};
  double ood_sigma{15.0};
  TrainStepParams train{};
  /// Optional per-label RNG seeds (Python ``hash(label) % 2**32``); empty → ``std::hash`` fallback.
  std::unordered_map<std::string, std::uint32_t> label_rng_seeds{};
  /// Optional fixed ``enc_W`` per label (row-major d×d) for parity harnesses.
  std::unordered_map<std::string, std::vector<double>> initial_enc_w{};
  /// Optional ``w_inject`` per label (row-major field_dim×d) matching Python ``CyphaDIF._W_inject``.
  std::unordered_map<std::string, std::vector<double>> initial_w_inject{};
  /// Optional ``field_W_T`` / ``field_sr_vec`` per label (parity harness).
  std::unordered_map<std::string, std::vector<double>> initial_field_w_t{};
  std::unordered_map<std::string, std::vector<double>> initial_field_sr_vec{};
};

/// One binary ``CyphaDIF`` (pos/neg) per semantic label, sharing ``VectorEncoder`` input dim.
class MultiLabelDif {
 public:
  explicit MultiLabelDif(MultiLabelDifParams params);

  [[nodiscard]] std::unordered_map<std::string, double> train_step(
      const double* x, int d, const std::unordered_map<std::string, bool>& labels);

  [[nodiscard]] std::unordered_map<std::string, double> predict(const double* x, int d) const;

  /// label → (n,) P(label=True|x)
  [[nodiscard]] std::unordered_map<std::string, std::vector<double>> predict_batch(const double* x_row_major,
                                                                                  int n, int d) const;

  [[nodiscard]] std::vector<std::string> labels() const;

 private:
  struct BinaryClf {
    CyphaInferModel infer;
    CyphaDifMemoryState mem;
    ReplayBuffer replay;
    std::mt19937 rng;
    int enc_update_count{0};
    int total_steps{0};

    explicit BinaryClf(int replay_cap) : replay(replay_cap) {}
  };

  MultiLabelDifParams params_;
  std::unordered_map<std::string, BinaryClf> classifiers_;

  BinaryClf& get_or_create(const std::string& label);
  void init_binary_clf(const std::string& label, BinaryClf& out);
  static std::uint32_t label_seed(const MultiLabelDifParams& p, const std::string& label);
};

}  // namespace cypha
