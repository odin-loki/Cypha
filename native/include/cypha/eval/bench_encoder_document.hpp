#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "cypha/bench/bench_encoder_text.hpp"

namespace cypha::bench {

/// Mirrors ``DocumentEncoder`` (TF-IDF hash + book segmentation).
class DocumentEncoder {
  public:
    explicit DocumentEncoder(int max_features = 2000, int min_n = 1, int max_n = 2);

    void fit(const std::vector<std::string>& documents);
    std::vector<std::vector<float>> encode_batch(const std::vector<std::string>& texts) const;
    std::vector<std::string> segment_book(const std::filesystem::path& path, int segment_chars = 500) const;

    int dim() const { return enc_.dim(); }

  private:
    TextHashEncoder enc_;
};

struct NewsDocumentDataset {
    std::string source;
    std::vector<std::vector<double>> x;
    std::vector<int> y;
    std::vector<std::string> texts;
};

/// Synthetic newsgroup-style corpus (sklearn 20news not bundled in native).
NewsDocumentDataset load_news_documents(int max_samples, std::uint64_t seed);

struct GutenbergSegments {
    std::vector<std::string> segments;
    std::vector<std::string> labels;
};

GutenbergSegments load_gutenberg_segments(int max_per_book, std::uint64_t seed);

/// Reduce TF-IDF rows to ``out_dim`` via top-variance feature selection (SVD substitute).
std::vector<std::vector<double>> reduce_features(const std::vector<std::vector<double>>& raw, int out_dim);

}  // namespace cypha::bench
