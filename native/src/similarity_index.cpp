#include "cypha/similarity_index.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cypha {

namespace {

constexpr double kEps = 1e-8;

}  // namespace

SimilarityIndex::SimilarityIndex(const CyphaInferModel& clf) : clf_(clf), d_latent_(clf.d_latent) {
  if (d_latent_ <= 0) {
    throw std::invalid_argument("SimilarityIndex: invalid latent dim");
  }
}

void SimilarityIndex::encode_one(const double* x, int d, std::vector<double>& h_out) const {
  if (d != clf_.d_latent) {
    throw std::invalid_argument("encode: d mismatch");
  }
  batch_encode(clf_, x, 1, h_out);
}

double SimilarityIndex::h_similarity(const double* h1, const double* h2) const {
  double sum = 0.0;
  for (int j = 0; j < d_latent_; ++j) {
    const double diff = h1[j] - h2[j];
    sum += diff * diff * clf_.inv_v[static_cast<std::size_t>(j)];
  }
  const double mahal = sum / (static_cast<double>(d_latent_) + kEps);
  return std::exp(-0.5 * mahal);
}

std::vector<double> SimilarityIndex::similarities_to_query(const double* h_q) const {
  const std::size_t n = meta_.size();
  std::vector<double> sims(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    sims[i] = h_similarity(h_q, h_store_.data() + i * static_cast<std::size_t>(d_latent_));
  }
  return sims;
}

int SimilarityIndex::add(const double* x, int d, nlohmann::json metadata) {
  std::vector<double> h;
  encode_one(x, d, h);
  const std::size_t idx = meta_.size();
  h_store_.insert(h_store_.end(), h.begin(), h.end());
  meta_.push_back(std::move(metadata));
  return static_cast<int>(idx);
}

std::vector<int> SimilarityIndex::add_batch(const double* x_row_major, int n, int d,
                                            const std::vector<nlohmann::json>* metadatas) {
  if (n <= 0) {
    return {};
  }
  std::vector<double> h;
  batch_encode(clf_, x_row_major, n, h);
  const int start = static_cast<int>(meta_.size());
  h_store_.insert(h_store_.end(), h.begin(), h.end());
  for (int i = 0; i < n; ++i) {
    nlohmann::json md = nullptr;
    if (metadatas != nullptr && static_cast<int>(metadatas->size()) > i) {
      md = (*metadatas)[static_cast<std::size_t>(i)];
    }
    meta_.push_back(std::move(md));
  }
  std::vector<int> out(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    out[static_cast<std::size_t>(i)] = start + i;
  }
  return out;
}

double SimilarityIndex::similarity(const double* x1, int d, const double* x2) const {
  std::vector<double> h1;
  std::vector<double> h2;
  encode_one(x1, d, h1);
  encode_one(x2, d, h2);
  return h_similarity(h1.data(), h2.data());
}

std::vector<SimilarityIndex::QueryHit> SimilarityIndex::query(const double* x, int d, int k,
                                                              bool return_similarities) const {
  if (meta_.empty()) {
    return {};
  }
  std::vector<double> h_q;
  encode_one(x, d, h_q);
  std::vector<double> sims = similarities_to_query(h_q.data());
  const int n = static_cast<int>(sims.size());
  const int k_actual = std::min(k, n);
  std::vector<int> order(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    order[static_cast<std::size_t>(i)] = i;
  }
  std::partial_sort(order.begin(), order.begin() + k_actual, order.end(),
                    [&](int a, int b) { return sims[static_cast<std::size_t>(a)] > sims[static_cast<std::size_t>(b)]; });

  std::vector<QueryHit> hits(static_cast<std::size_t>(k_actual));
  for (int r = 0; r < k_actual; ++r) {
    const int i = order[static_cast<std::size_t>(r)];
    hits[static_cast<std::size_t>(r)].index = i;
    hits[static_cast<std::size_t>(r)].metadata = meta_[static_cast<std::size_t>(i)];
    if (return_similarities) {
      hits[static_cast<std::size_t>(r)].similarity = sims[static_cast<std::size_t>(i)];
    }
  }
  return hits;
}

std::vector<std::vector<SimilarityIndex::QueryHit>> SimilarityIndex::query_batch(const double* x_row_major, int n,
                                                                                 int d, int k) const {
  std::vector<std::vector<QueryHit>> out(static_cast<std::size_t>(n));
  if (meta_.empty() || n <= 0) {
    return out;
  }
  std::vector<double> h_q;
  batch_encode(clf_, x_row_major, n, h_q);
  const int stored = static_cast<int>(meta_.size());
  const int k_actual = std::min(k, stored);
  for (int m = 0; m < n; ++m) {
    const double* hq = h_q.data() + static_cast<std::size_t>(m * d_latent_);
    std::vector<double> sims = similarities_to_query(hq);
    std::vector<int> order(static_cast<std::size_t>(stored));
    for (int i = 0; i < stored; ++i) {
      order[static_cast<std::size_t>(i)] = i;
    }
    std::partial_sort(order.begin(), order.begin() + k_actual, order.end(), [&](int a, int b) {
      return sims[static_cast<std::size_t>(a)] > sims[static_cast<std::size_t>(b)];
    });
    std::vector<QueryHit> hits(static_cast<std::size_t>(k_actual));
    for (int r = 0; r < k_actual; ++r) {
      const int i = order[static_cast<std::size_t>(r)];
      hits[static_cast<std::size_t>(r)] = {i, meta_[static_cast<std::size_t>(i)], sims[static_cast<std::size_t>(i)]};
    }
    out[static_cast<std::size_t>(m)] = std::move(hits);
  }
  return out;
}

}  // namespace cypha
