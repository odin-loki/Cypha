#include "cypha/bench/bench_encoder_document.hpp"

#include "cypha/bench/bench_paths.hpp"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace cypha::bench {

namespace fs = std::filesystem;

DocumentEncoder::DocumentEncoder(int max_features, int min_n, int max_n) : enc_(max_features, min_n, max_n) {}

void DocumentEncoder::fit(const std::vector<std::string>& documents) { enc_.fit(documents); }

std::vector<std::vector<float>> DocumentEncoder::encode_batch(const std::vector<std::string>& texts) const {
    return enc_.encode_batch(texts);
}

std::vector<std::string> DocumentEncoder::segment_book(const fs::path& path, int segment_chars) const {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open book: " + path.string());
    std::ostringstream oss;
    oss << in.rdbuf();
    const std::string text = oss.str();
    std::vector<std::string> segments;
    if (text.empty()) return segments;
    const int stride = std::max(segment_chars, 1);
    for (std::size_t i = 0; i + static_cast<std::size_t>(segment_chars) <= text.size();
         i += static_cast<std::size_t>(stride)) {
        segments.push_back(text.substr(i, static_cast<std::size_t>(segment_chars)));
    }
    if (segments.empty() && !text.empty()) segments.push_back(text.substr(0, static_cast<std::size_t>(segment_chars)));
    return segments;
}

namespace {

std::string synthetic_passage(int category, int variant) {
    static const char* kTopics[] = {"science", "sports", "politics", "technology", "health",
                                    "finance", "travel", "education", "entertainment", "automotive"};
    const int topic_idx = category % 10;
    std::ostringstream oss;
    oss << "Article about " << kTopics[topic_idx] << " topic variant " << variant
        << ". Researchers and enthusiasts discuss recent developments in " << kTopics[topic_idx]
        << ". Community members share opinions, data, and practical advice. "
        << "This passage is synthetic for native bench document classification.";
    return oss.str();
}

}  // namespace

NewsDocumentDataset load_news_documents(int max_samples, std::uint64_t seed) {
    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    NewsDocumentDataset ds;
    ds.source = "synthetic_newsgroups";
    ds.texts.reserve(static_cast<std::size_t>(max_samples));
    ds.y.reserve(static_cast<std::size_t>(max_samples));
    for (int i = 0; i < max_samples; ++i) {
        const int cls = static_cast<int>(i % 20);
        ds.texts.push_back(synthetic_passage(cls, i));
        ds.y.push_back(cls);
    }
    DocumentEncoder enc(2000, 1, 2);
    enc.fit(ds.texts);
    const auto raw = enc.encode_batch(ds.texts);
    ds.x = reduce_features([&]() {
        std::vector<std::vector<double>> rows;
        rows.reserve(raw.size());
        for (const auto& row : raw) {
            std::vector<double> d(row.size());
            for (std::size_t j = 0; j < row.size(); ++j) d[j] = static_cast<double>(row[j]);
            rows.push_back(std::move(d));
        }
        return rows;
    }(), 100);
    return ds;
}

GutenbergSegments load_gutenberg_segments(int max_per_book, std::uint64_t seed) {
    (void)seed;
    DocumentEncoder enc;
    const std::unordered_map<std::string, fs::path> books = {
        {"alice", data_dir() / "gutenberg" / "alice.txt"},
        {"sherlock", data_dir() / "gutenberg" / "sherlock_holmes.txt"},
        {"moby", data_dir() / "gutenberg" / "moby_dick.txt"},
    };
    GutenbergSegments out;
    for (const auto& [name, path] : books) {
        std::vector<std::string> segs;
        if (fs::is_regular_file(path)) {
            segs = enc.segment_book(path, 500);
        } else {
            const std::string text = "Synthetic passage for " + name + ". ";
            for (int i = 0; i < 4; ++i) segs.push_back(text + std::to_string(i));
        }
        const int take = std::min(static_cast<int>(segs.size()), max_per_book);
        for (int i = 0; i < take; ++i) {
            out.segments.push_back(segs[static_cast<std::size_t>(i)]);
            out.labels.push_back(name);
        }
    }
    if (out.segments.empty()) {
        out.segments.assign(30, "Synthetic document segment.");
        out.labels.assign(30, "synthetic");
    }
    return out;
}

std::vector<std::vector<double>> reduce_features(const std::vector<std::vector<double>>& raw, int out_dim) {
    if (raw.empty()) return {};
    const int d = static_cast<int>(raw.front().size());
    const int k = std::min(out_dim, d);
    std::vector<double> var(static_cast<std::size_t>(d), 0.0);
    std::vector<double> mean(static_cast<std::size_t>(d), 0.0);
    for (const auto& row : raw) {
        for (int j = 0; j < d; ++j) mean[static_cast<std::size_t>(j)] += row[static_cast<std::size_t>(j)];
    }
    for (double& m : mean) m /= static_cast<double>(raw.size());
    for (const auto& row : raw) {
        for (int j = 0; j < d; ++j) {
            const double diff = row[static_cast<std::size_t>(j)] - mean[static_cast<std::size_t>(j)];
            var[static_cast<std::size_t>(j)] += diff * diff;
        }
    }
    std::vector<int> idx(static_cast<std::size_t>(d));
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                    [&](int a, int b) { return var[static_cast<std::size_t>(a)] > var[static_cast<std::size_t>(b)]; });
    std::vector<std::vector<double>> out;
    out.reserve(raw.size());
    for (const auto& row : raw) {
        std::vector<double> reduced(static_cast<std::size_t>(k));
        for (int j = 0; j < k; ++j) reduced[static_cast<std::size_t>(j)] = row[static_cast<std::size_t>(idx[static_cast<std::size_t>(j)])];
        out.push_back(std::move(reduced));
    }
    return out;
}

}  // namespace cypha::bench
