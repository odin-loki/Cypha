#include "cypha/replay_buffer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <limits>
#include <numeric>

namespace cypha {

namespace {

constexpr double kPriorityLossEps = 1e-3;
constexpr double kLogUeps = 1e-3;

}  // namespace

ReplayBuffer::ReplayBuffer(int capacity) : cap_(std::max(1, capacity)) {
  labels_.resize(static_cast<std::size_t>(cap_));
  w_cache_.assign(static_cast<std::size_t>(cap_), 0.0);
}

void ReplayBuffer::push(const double* h, const double* f, int d, const std::string& label, double loss) {
  if (d < 1) {
    return;
  }
  const std::size_t stride = static_cast<std::size_t>(d);
  if (static_cast<int>(h_store_.size()) < cap_ * d) {
    h_store_.assign(static_cast<std::size_t>(cap_ * d), 0.0);
    f_store_.assign(static_cast<std::size_t>(cap_ * d), 0.0);
  }
  insert_n_ += 1;
  const double loss_v = std::abs(loss) + kPriorityLossEps;
  if (len_ > 0) {
    for (int i = 0; i < len_; ++i) {
      w_cache_[static_cast<std::size_t>(i)] *= kDecay;
    }
  }
  int idx = len_;
  if (len_ < cap_) {
    len_ += 1;
  } else {
    int argmin = 0;
    double wm = w_cache_[0];
    for (int i = 1; i < len_; ++i) {
      if (w_cache_[static_cast<std::size_t>(i)] < wm) {
        wm = w_cache_[static_cast<std::size_t>(i)];
        argmin = i;
      }
    }
    idx = argmin;
  }
  for (int j = 0; j < d; ++j) {
    h_store_[static_cast<std::size_t>(idx) * stride + static_cast<std::size_t>(j)] =
        h[static_cast<std::size_t>(j)];
    f_store_[static_cast<std::size_t>(idx) * stride + static_cast<std::size_t>(j)] =
        f[static_cast<std::size_t>(j)];
  }
  labels_[static_cast<std::size_t>(idx)] = label;
  w_cache_[static_cast<std::size_t>(idx)] = loss_v;
}

void ReplayBuffer::sample(int n, std::mt19937& rng, std::vector<std::vector<double>>& h_out,
                          std::vector<std::string>& labels_out, const double* fixed_u01,
                          std::size_t* fixed_u01_pos, std::size_t fixed_u01_len) const {
  h_out.clear();
  labels_out.clear();
  if (len_ <= 0) {
    return;
  }
  const int d = static_cast<int>(h_store_.size() / static_cast<std::size_t>(cap_));
  if (d < 1) {
    return;
  }
  n = std::min(n, len_);
  const std::size_t stride = static_cast<std::size_t>(d);
  std::vector<double> keys(static_cast<std::size_t>(len_));
  std::uniform_real_distribution<double> u(0.0, 1.0);
  auto next_u01 = [&]() -> double {
    if (fixed_u01 != nullptr && fixed_u01_pos != nullptr) {
      if (*fixed_u01_pos >= fixed_u01_len) {
        throw std::runtime_error("ReplayBuffer::sample: fixed_u01 exhausted");
      }
      return fixed_u01[(*fixed_u01_pos)++];
    }
    return u(rng);
  };
  for (int i = 0; i < len_; ++i) {
    double z = next_u01() + kLogUeps;
    keys[static_cast<std::size_t>(i)] = std::log(z) / w_cache_[static_cast<std::size_t>(i)];
  }
  std::vector<int> ord(static_cast<std::size_t>(len_));
  std::iota(ord.begin(), ord.end(), 0);
  if (n >= len_) {
    for (int i = 0; i < len_; ++i) {
      int r = ord[static_cast<std::size_t>(i)];
      std::vector<double> row(static_cast<std::size_t>(d));
      for (int j = 0; j < d; ++j) {
        row[static_cast<std::size_t>(j)] = h_store_[static_cast<std::size_t>(r) * stride + static_cast<std::size_t>(j)];
      }
      h_out.push_back(std::move(row));
      labels_out.push_back(labels_[static_cast<std::size_t>(r)]);
    }
    return;
  }
  std::partial_sort(ord.begin(), ord.begin() + n, ord.end(), [&](int a, int b) {
    const double ka = keys[static_cast<std::size_t>(a)];
    const double kb = keys[static_cast<std::size_t>(b)];
    if (ka != kb) {
      return ka > kb;
    }
    return a < b;
  });
  for (int k = 0; k < n; ++k) {
    int r = ord[static_cast<std::size_t>(k)];
    std::vector<double> row(static_cast<std::size_t>(d));
    for (int j = 0; j < d; ++j) {
      row[static_cast<std::size_t>(j)] = h_store_[static_cast<std::size_t>(r) * stride + static_cast<std::size_t>(j)];
    }
    h_out.push_back(std::move(row));
    labels_out.push_back(labels_[static_cast<std::size_t>(r)]);
  }
}

}  // namespace cypha
