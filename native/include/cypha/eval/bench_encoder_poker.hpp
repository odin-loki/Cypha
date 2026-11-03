#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace cypha::bench {

/// Mirrors ``PokerEncoder`` synthetic path (no treys in native).
class PokerEncoder {
  public:
    static constexpr int kDim = 18;

    std::pair<std::vector<float>, std::string> generate_random_situation(std::mt19937& rng) const;

  private:
    std::vector<float> synthetic_vector(double pot, double stack, int position) const;
};

struct PokerDataset {
    std::vector<std::vector<double>> x;
    std::vector<std::string> y;
};

PokerDataset generate_poker_dataset(int n_hands, std::uint64_t seed);

}  // namespace cypha::bench
