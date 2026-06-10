#pragma once

#include <cstddef>
#include <vector>

namespace cypha::cyphalm {

/// Ring buffer of recent embeddings with linear (dot-product) attention — O(K*d).
class ContextBank {
 public:
  explicit ContextBank(int embed_dim = 64, int capacity = 512);

  void reset();
  void push(const std::vector<double>& embed);
  void push(const double* embed, int dim);

  /// Linear attention: weighted sum of bank entries; weights = dot(query, entry).
  std::vector<double> linear_attention(const std::vector<double>& query) const;

  int size() const { return count_; }
  int capacity() const { return capacity_; }
  int embed_dim() const { return embed_dim_; }

 private:
  int embed_dim_;
  int capacity_;
  int head_{0};
  int count_{0};
  std::vector<double> storage_;
};

}  // namespace cypha::cyphalm
