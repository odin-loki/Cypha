#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cypha::bench {

/// Mirrors ``GoEncoder`` for 9×9 boards.
class GoEncoder {
  public:
    static constexpr int kDim = 168;

    std::vector<float> encode(const std::vector<float>& board, int turn = 1) const;

  private:
    std::vector<float> compute_liberties(const std::vector<float>& board) const;
};

struct GoDataset {
    std::vector<std::vector<double>> x;
    std::vector<double> y_reg;
    std::vector<std::string> y_cls;
};

GoDataset generate_go_dataset(int n_samples, std::uint64_t seed);

}  // namespace cypha::bench
