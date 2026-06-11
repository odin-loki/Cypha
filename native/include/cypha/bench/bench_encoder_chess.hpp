#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cypha::bench {

/// Mirrors ``ChessEncoder`` (synthetic path when python-chess unavailable).
class ChessEncoder {
  public:
    static constexpr int kDim = 50;

    std::vector<float> encode_synthetic(const std::vector<int>& material) const;
};

struct ChessDataset {
    std::string source;
    std::vector<std::vector<double>> x;
    std::vector<double> y;
};

ChessDataset load_chess_dataset(int n_samples, std::uint64_t seed);

}  // namespace cypha::bench
