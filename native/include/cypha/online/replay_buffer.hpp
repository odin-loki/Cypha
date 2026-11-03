#pragma once

#include <cstddef>
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

  /// Buffer contents for ``Cypha::save`` / ``Cypha::load``: rows ``0..len-1`` of ``h``, their priority weights
  /// and labels. The ``f`` rows are not kept: nothing reads them back.
  struct State {
    int cap{1};
    int len{0};
    int insert_n{0};
    int d{0};
    std::vector<double> h;  ///< ``len × d`` row-major
    std::vector<double> w;  ///< ``len`` priority weights
    std::vector<std::string> labels;  ///< ``len`` labels
  };

  State export_state() const {
    State s;
    s.cap = cap_;
    s.len = len_;
    s.insert_n = insert_n_;
    s.d = h_store_.empty() ? 0 : static_cast<int>(h_store_.size() / static_cast<std::size_t>(cap_));
    const std::size_t n = static_cast<std::size_t>(len_);
    const std::size_t d = static_cast<std::size_t>(s.d);
    s.h.assign(h_store_.begin(), h_store_.begin() + static_cast<std::ptrdiff_t>(n * d));
    s.w.assign(w_cache_.begin(), w_cache_.begin() + static_cast<std::ptrdiff_t>(n));
    s.labels.assign(labels_.begin(), labels_.begin() + static_cast<std::ptrdiff_t>(n));
    return s;
  }

  /// Inverse of ``export_state``. Returns false (buffer unchanged) when the sizes do not agree.
  bool import_state(const State& s) {
    const int cap = s.cap < 1 ? 1 : s.cap;
    const std::size_t n = static_cast<std::size_t>(s.len);
    const std::size_t d = static_cast<std::size_t>(s.d);
    if (s.len < 0 || s.len > cap || s.d < 0 || (s.len > 0 && s.d < 1) || s.h.size() != n * d ||
        s.w.size() != n || s.labels.size() != n) {
      return false;
    }
    cap_ = cap;
    len_ = s.len;
    insert_n_ = s.insert_n;
    h_store_.clear();
    f_store_.clear();
    if (d > 0) {
      h_store_.assign(static_cast<std::size_t>(cap_) * d, 0.0);
      f_store_.assign(static_cast<std::size_t>(cap_) * d, 0.0);
      for (std::size_t i = 0; i < n * d; ++i) {
        h_store_[i] = s.h[i];
      }
    }
    labels_.assign(static_cast<std::size_t>(cap_), std::string());
    w_cache_.assign(static_cast<std::size_t>(cap_), 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      labels_[i] = s.labels[i];
      w_cache_[i] = s.w[i];
    }
    return true;
  }

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
