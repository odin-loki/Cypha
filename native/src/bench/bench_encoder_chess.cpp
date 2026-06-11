#include "cypha/bench/bench_encoder_chess.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace cypha::bench {

std::vector<float> ChessEncoder::encode_synthetic(const std::vector<int>& material) const {
    std::vector<float> vec(kDim, 0.0f);
    const int n = std::min(static_cast<int>(material.size()), kDim);
    for (int i = 0; i < n; ++i) {
        vec[static_cast<std::size_t>(i)] = static_cast<float>(material[static_cast<std::size_t>(i)]) / 10.0f;
    }
    return vec;
}

ChessDataset load_chess_dataset(int n_samples, std::uint64_t seed) {
    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    std::uniform_int_distribution<int> piece_count(0, 11);
    ChessEncoder enc;
    ChessDataset ds;
    ds.source = "synthetic";
    ds.x.reserve(static_cast<std::size_t>(n_samples));
    ds.y.reserve(static_cast<std::size_t>(n_samples));
    for (int i = 0; i < n_samples; ++i) {
        std::vector<int> material(12);
        int white_sum = 0;
        int black_sum = 0;
        for (int j = 0; j < 12; ++j) {
            material[static_cast<std::size_t>(j)] = piece_count(rng);
            if (j < 6) white_sum += material[static_cast<std::size_t>(j)];
            else black_sum += material[static_cast<std::size_t>(j)];
        }
        const auto feat = enc.encode_synthetic(material);
        std::vector<double> row(feat.size());
        for (std::size_t k = 0; k < feat.size(); ++k) row[k] = static_cast<double>(feat[k]);
        ds.x.push_back(std::move(row));
        ds.y.push_back(std::tanh(static_cast<double>(white_sum - black_sum)));
    }
    return ds;
}

}  // namespace cypha::bench
