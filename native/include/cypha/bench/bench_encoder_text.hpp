#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace cypha::bench {

/// Hash / bag-of-words text encoder (mirrors ``TextEncoder`` / ``CharNgramEncoder`` for bench).
class TextHashEncoder {
  public:
    explicit TextHashEncoder(int max_features = 1000, int min_n = 1, int max_n = 2);

    void fit(const std::vector<std::string>& documents);
    std::vector<float> encode(const std::string& text) const;
    std::vector<std::vector<float>> encode_batch(const std::vector<std::string>& texts) const;

    int dim() const { return max_features_; }
    bool fitted() const { return fitted_; }

  private:
    int max_features_;
    int min_n_;
    int max_n_;
    bool fitted_{false};
    std::vector<float> idf_;

    static std::vector<std::string> tokenize(const std::string& text);
    static std::vector<std::string> word_ngrams(const std::vector<std::string>& tokens, int min_n, int max_n);
    std::size_t feature_index(const std::string& gram) const;
};

/// Character n-gram bag encoder (mirrors ``CharNgramEncoder``).
class CharNgramEncoder {
  public:
    CharNgramEncoder(int n = 5, int vocab_size = 200);

    void build_vocab(const std::string& text);
    std::vector<float> encode(const std::string& window) const;

    int dim() const { return vocab_size_; }

  private:
    int n_;
    int vocab_size_;
    std::unordered_map<std::string, int> vocab_;
};

}  // namespace cypha::bench
