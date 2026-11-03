#pragma once

#include "cypha/infer_cpu.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace cypha {

/// Mahalanobis similarity index over encoded latents (Python ``SimilarityIndex``).
class SimilarityIndex {
 public:
  explicit SimilarityIndex(const CyphaInferModel& clf);

  /// Encode and store one example; returns storage index.
  int add(const double* x, int d, nlohmann::json metadata = nullptr);

  std::vector<int> add_batch(const double* x_row_major, int n, int d,
                             const std::vector<nlohmann::json>* metadatas = nullptr);

  double similarity(const double* x1, int d, const double* x2) const;

  struct QueryHit {
    int index{};
    nlohmann::json metadata;
    double similarity{};
  };

  std::vector<QueryHit> query(const double* x, int d, int k = 5, bool return_similarities = true) const;

  std::vector<std::vector<QueryHit>> query_batch(const double* x_row_major, int n, int d, int k = 5) const;

  [[nodiscard]] std::size_t size() const { return meta_.size(); }

 private:
  CyphaInferModel clf_;
  std::vector<double> h_store_;
  std::vector<nlohmann::json> meta_;
  int d_latent_{0};

  void encode_one(const double* x, int d, std::vector<double>& h_out) const;
  double h_similarity(const double* h1, const double* h2) const;
  std::vector<double> similarities_to_query(const double* h_q) const;
};

}  // namespace cypha
