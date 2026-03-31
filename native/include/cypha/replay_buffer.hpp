#pragma once

#include <random>
#include <string>
#include <vector>

namespace cypha {

/// Priority replay matching `PriorityReplayBuffer` decay + weighted sampling (subset).
class ReplayBuffer {
 public:
  explicit ReplayBuffer(int capacity);

  void push(const double* h, const double* f, int d, const std::string& label, double loss);
  int size() const { return len_; }
  /// Weighted sample without replacement (Efraimidis–Spirakis keys). Returns latent `h` + label for `memory_train`.
  /// If ``fixed_u01`` is set, consumes ``len_`` values from it (one per buffer slot) instead of ``rng``.
  void sample(int n, std::mt19937& rng, std::vector<std::vector<double>>& h_out,
              std::vector<std::string>& labels_out, const double* fixed_u01 = nullptr,
              std::size_t* fixed_u01_pos = nullptr, std::size_t fixed_u01_len = 0) const;

 private:
  int cap_{};
  int len_{};
  int insert_n_{};
  std::vector<double> h_store_;
  std::vector<double> f_store_;
  std::vector<std::string> labels_;
  std::vector<double> w_cache_;
  static constexpr double kDecay = 0.9999000049998333;  // exp(-0.0001)
  static constexpr double kPriorityEps = 1e-12;
};

}  // namespace cypha
