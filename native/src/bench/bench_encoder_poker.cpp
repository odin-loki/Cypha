#include "cypha/bench/bench_encoder_poker.hpp"

#include <algorithm>
#include <cmath>

namespace cypha::bench {

std::vector<float> PokerEncoder::synthetic_vector(double pot, double stack, int position) const {
    std::mt19937 rng(static_cast<std::mt19937::result_type>(
        static_cast<std::uint32_t>(pot * 1000.0 + stack * 100.0 + position)));
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const double hs = uni(rng);
    std::vector<float> vec(kDim, 0.0f);
    vec[0] = static_cast<float>(hs);
    const int bucket = std::min(8, static_cast<int>(hs * 8.0));
    vec[8 + bucket] = 1.0f;
    vec[11] = static_cast<float>(uni(rng));
    vec[12] = static_cast<float>(uni(rng));
    vec[13] = static_cast<float>(uni(rng) > 0.5 ? 1.0 : 0.0);
    vec[14] = static_cast<float>(pot);
    vec[15] = static_cast<float>(stack);
    vec[16] = static_cast<float>(position) / 2.0f;
    vec[17] = static_cast<float>(uni(rng) * 0.2);
    return vec;
}

std::pair<std::vector<float>, std::string> PokerEncoder::generate_random_situation(std::mt19937& rng) const {
    std::uniform_int_distribution<int> pot_dist(1, 49);
    std::uniform_int_distribution<int> stack_dist(10, 199);
    std::uniform_int_distribution<int> pos_dist(0, 2);
    const double pot = static_cast<double>(pot_dist(rng)) / 100.0;
    const double stack = static_cast<double>(stack_dist(rng)) / 200.0;
    const int pos = pos_dist(rng);
    const auto vec = synthetic_vector(pot, stack, pos);
    const double hs = static_cast<double>(vec[0]);
    std::string label = "fold";
    if (hs > 0.65) label = "raise";
    else if (hs > 0.35) label = "call";
    return {vec, label};
}

PokerDataset generate_poker_dataset(int n_hands, std::uint64_t seed) {
    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    PokerEncoder enc;
    PokerDataset ds;
    ds.x.reserve(static_cast<std::size_t>(n_hands));
    ds.y.reserve(static_cast<std::size_t>(n_hands));
    for (int i = 0; i < n_hands; ++i) {
        const auto [vec, label] = enc.generate_random_situation(rng);
        std::vector<double> row(vec.size());
        for (std::size_t k = 0; k < vec.size(); ++k) row[k] = static_cast<double>(vec[k]);
        ds.x.push_back(std::move(row));
        ds.y.push_back(label);
    }
    return ds;
}

}  // namespace cypha::bench
