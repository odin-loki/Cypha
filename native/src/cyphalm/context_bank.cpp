#include "cypha/cyphalm/context_bank.hpp"

#include <cmath>
#include <stdexcept>

namespace cypha::cyphalm {

ContextBank::ContextBank(int embed_dim, int capacity)
    : embed_dim_(embed_dim), capacity_(capacity) {
  if (embed_dim_ < 1 || capacity_ < 1) {
    throw std::invalid_argument("ContextBank: embed_dim and capacity must be >= 1");
  }
  storage_.assign(static_cast<std::size_t>(capacity_ * embed_dim_), 0.0);
}

void ContextBank::reset() {
  head_ = 0;
  count_ = 0;
  std::fill(storage_.begin(), storage_.end(), 0.0);
}

void ContextBank::push(const std::vector<double>& embed) {
  push(embed.data(), static_cast<int>(embed.size()));
}

void ContextBank::push(const double* embed, int dim) {
  if (dim != embed_dim_) {
    throw std::invalid_argument("ContextBank::push: embed dimension mismatch");
  }
  const std::size_t off = static_cast<std::size_t>(head_ * embed_dim_);
  for (int i = 0; i < embed_dim_; ++i) {
    storage_[off + static_cast<std::size_t>(i)] = embed[i];
  }
  head_ = (head_ + 1) % capacity_;
  if (count_ < capacity_) {
    ++count_;
  }
}

std::vector<double> ContextBank::linear_attention(const std::vector<double>& query) const {
  if (static_cast<int>(query.size()) != embed_dim_) {
    throw std::invalid_argument("ContextBank::linear_attention: query dimension mismatch");
  }
  if (count_ == 0) {
    return std::vector<double>(static_cast<std::size_t>(embed_dim_), 0.0);
  }

  std::vector<double> out(static_cast<std::size_t>(embed_dim_), 0.0);
  double weight_sum = 0.0;

  for (int i = 0; i < count_; ++i) {
    const int slot = (head_ - count_ + i + capacity_) % capacity_;
    const std::size_t off = static_cast<std::size_t>(slot * embed_dim_);
    double w = 0.0;
    for (int d = 0; d < embed_dim_; ++d) {
      w += query[static_cast<std::size_t>(d)] * storage_[off + static_cast<std::size_t>(d)];
    }
    weight_sum += w;
    for (int d = 0; d < embed_dim_; ++d) {
      out[static_cast<std::size_t>(d)] += w * storage_[off + static_cast<std::size_t>(d)];
    }
  }

  if (std::abs(weight_sum) > 1e-12) {
    for (double& v : out) {
      v /= weight_sum;
    }
  }
  return out;
}

}  // namespace cypha::cyphalm
