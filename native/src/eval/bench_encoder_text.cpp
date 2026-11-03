#include "cypha/bench/bench_encoder_text.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace cypha::bench {

TextHashEncoder::TextHashEncoder(int max_features, int min_n, int max_n)
    : max_features_(max_features), min_n_(min_n), max_n_(max_n), idf_(static_cast<std::size_t>(max_features), 1.0f) {}

std::vector<std::string> TextHashEncoder::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string cur;
    for (unsigned char ch : text) {
        if (std::isalnum(ch)) {
            cur.push_back(static_cast<char>(std::tolower(ch)));
        } else if (!cur.empty()) {
            tokens.push_back(cur);
            cur.clear();
        }
    }
    if (!cur.empty()) tokens.push_back(cur);
    return tokens;
}

std::vector<std::string> TextHashEncoder::word_ngrams(const std::vector<std::string>& tokens, int min_n,
                                                      int max_n) {
    std::vector<std::string> grams;
    if (tokens.empty()) return grams;
    for (int n = min_n; n <= max_n; ++n) {
        if (static_cast<int>(tokens.size()) < n) continue;
        for (std::size_t i = 0; i + static_cast<std::size_t>(n) <= tokens.size(); ++i) {
            std::ostringstream oss;
            for (int j = 0; j < n; ++j) {
                if (j > 0) oss << ' ';
                oss << tokens[i + static_cast<std::size_t>(j)];
            }
            grams.push_back(oss.str());
        }
    }
    return grams;
}

std::size_t TextHashEncoder::feature_index(const std::string& gram) const {
    const std::size_t h = std::hash<std::string>{}(gram);
    return h % static_cast<std::size_t>(max_features_);
}

void TextHashEncoder::fit(const std::vector<std::string>& documents) {
    idf_.assign(static_cast<std::size_t>(max_features_), 0.0f);
    const int n_docs = static_cast<int>(documents.size());
    for (const auto& doc : documents) {
        const auto tokens = tokenize(doc);
        const auto grams = word_ngrams(tokens, min_n_, max_n_);
        std::vector<std::size_t> seen;
        seen.reserve(grams.size());
        for (const auto& g : grams) {
            const std::size_t idx = feature_index(g);
            if (std::find(seen.begin(), seen.end(), idx) == seen.end()) {
                seen.push_back(idx);
                idf_[idx] += 1.0f;
            }
        }
    }
    for (float& df : idf_) {
        if (df > 0.0f) {
            df = static_cast<float>(std::log((1.0 + n_docs) / (1.0 + df)) + 1.0);
        } else {
            df = 1.0f;
        }
    }
    fitted_ = true;
}

std::vector<float> TextHashEncoder::encode(const std::string& text) const {
    if (!fitted_) throw std::runtime_error("TextHashEncoder: call fit() before encode()");
    std::vector<float> vec(static_cast<std::size_t>(max_features_), 0.0f);
    const auto tokens = tokenize(text);
    const auto grams = word_ngrams(tokens, min_n_, max_n_);
    std::unordered_map<std::size_t, float> tf;
    for (const auto& g : grams) {
        const std::size_t idx = feature_index(g);
        tf[idx] += 1.0f;
    }
    for (const auto& [idx, count] : tf) {
        const float sublinear = 1.0f + static_cast<float>(std::log(static_cast<double>(count)));
        vec[idx] = sublinear * idf_[idx];
    }
    double norm = 0.0;
    for (float v : vec) norm += static_cast<double>(v) * static_cast<double>(v);
    norm = std::sqrt(norm);
    if (norm > 1e-12) {
        for (float& v : vec) v = static_cast<float>(static_cast<double>(v) / norm);
    }
    return vec;
}

std::vector<std::vector<float>> TextHashEncoder::encode_batch(const std::vector<std::string>& texts) const {
    std::vector<std::vector<float>> out;
    out.reserve(texts.size());
    for (const auto& t : texts) out.push_back(encode(t));
    return out;
}

CharNgramEncoder::CharNgramEncoder(int n, int vocab_size) : n_(n), vocab_size_(vocab_size) {}

void CharNgramEncoder::build_vocab(const std::string& text) {
    std::unordered_map<std::string, int> counts;
    if (static_cast<int>(text.size()) >= n_) {
        for (std::size_t i = 0; i + static_cast<std::size_t>(n_) <= text.size(); ++i) {
            const std::string ng = text.substr(i, static_cast<std::size_t>(n_));
            counts[ng] += 1;
        }
    }
    std::vector<std::pair<std::string, int>> ranked;
    ranked.reserve(counts.size());
    for (const auto& kv : counts) ranked.emplace_back(kv.first, kv.second);
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.second > b.second || (a.second == b.second && a.first < b.first); });
    vocab_.clear();
    for (int i = 0; i < vocab_size_ && i < static_cast<int>(ranked.size()); ++i) {
        vocab_[ranked[static_cast<std::size_t>(i)].first] = i;
    }
}

std::vector<float> CharNgramEncoder::encode(const std::string& window) const {
    std::vector<float> vec(static_cast<std::size_t>(vocab_size_), 0.0f);
    if (static_cast<int>(window.size()) < n_) return vec;
    for (std::size_t i = 0; i + static_cast<std::size_t>(n_) <= window.size(); ++i) {
        const std::string ng = window.substr(i, static_cast<std::size_t>(n_));
        const auto it = vocab_.find(ng);
        if (it != vocab_.end()) vec[static_cast<std::size_t>(it->second)] += 1.0f;
    }
    float total = 0.0f;
    for (float v : vec) total += v;
    if (total > 0.0f) {
        for (float& v : vec) v /= total;
    }
    return vec;
}

}  // namespace cypha::bench
