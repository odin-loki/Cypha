#include "cypha/bench/bench_encoder_go.hpp"

#include <algorithm>
#include <random>
#include <stdexcept>

namespace cypha::bench {

namespace {

constexpr int kBoardSize = 9;
constexpr int kCells = kBoardSize * kBoardSize;

std::vector<float> generate_synthetic_board(std::mt19937& rng, int n_stones, float& territory) {
    std::vector<float> board(static_cast<std::size_t>(kCells), 0.0f);
    std::vector<int> positions(static_cast<std::size_t>(kCells));
    for (int i = 0; i < kCells; ++i) positions[static_cast<std::size_t>(i)] = i;
    std::shuffle(positions.begin(), positions.end(), rng);
    const int n = std::min(n_stones, kCells);
    for (int i = 0; i < n; ++i) {
        board[static_cast<std::size_t>(positions[static_cast<std::size_t>(i)])] =
            (i < n / 2) ? 1.0f : -1.0f;
    }
    territory = 0.0f;
    for (float v : board) territory += v;
    return board;
}

}  // namespace

std::vector<float> GoEncoder::compute_liberties(const std::vector<float>& board) const {
    std::vector<float> liberties(static_cast<std::size_t>(kCells), 0.0f);
    for (int r = 0; r < kBoardSize; ++r) {
        for (int c = 0; c < kBoardSize; ++c) {
            const int idx = r * kBoardSize + c;
            if (board[static_cast<std::size_t>(idx)] == 0.0f) continue;
            int count = 0;
            if (r > 0 && board[static_cast<std::size_t>((r - 1) * kBoardSize + c)] == 0.0f) ++count;
            if (r + 1 < kBoardSize && board[static_cast<std::size_t>((r + 1) * kBoardSize + c)] == 0.0f) ++count;
            if (c > 0 && board[static_cast<std::size_t>(r * kBoardSize + c - 1)] == 0.0f) ++count;
            if (c + 1 < kBoardSize && board[static_cast<std::size_t>(r * kBoardSize + c + 1)] == 0.0f) ++count;
            liberties[static_cast<std::size_t>(idx)] = static_cast<float>(count);
        }
    }
    return liberties;
}

std::vector<float> GoEncoder::encode(const std::vector<float>& board, int turn) const {
    if (static_cast<int>(board.size()) != kCells) {
        throw std::runtime_error("GoEncoder: expected 9x9 board");
    }
    const auto liberties = compute_liberties(board);
    std::vector<float> features;
    features.reserve(static_cast<std::size_t>(kDim));
    features.insert(features.end(), board.begin(), board.end());
    for (float l : liberties) features.push_back(l / 4.0f);

    float black_stones = 0.0f;
    float white_stones = 0.0f;
    float empty_squares = 0.0f;
    for (float v : board) {
        if (v > 0.0f) ++black_stones;
        else if (v < 0.0f) ++white_stones;
        else ++empty_squares;
    }
    features.push_back(black_stones / 81.0f);
    features.push_back(white_stones / 81.0f);
    features.push_back(empty_squares / 81.0f);
    features.push_back((black_stones - white_stones) / 81.0f);
    features.push_back(static_cast<float>(turn));
    features.push_back((black_stones + white_stones) / 81.0f);
    return features;
}

GoDataset generate_go_dataset(int n_samples, std::uint64_t seed) {
    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    std::uniform_int_distribution<int> stone_dist(5, 39);
    GoEncoder enc;
    GoDataset ds;
    ds.x.reserve(static_cast<std::size_t>(n_samples));
    ds.y_reg.reserve(static_cast<std::size_t>(n_samples));
    ds.y_cls.reserve(static_cast<std::size_t>(n_samples));
    for (int i = 0; i < n_samples; ++i) {
        float territory = 0.0f;
        const auto board = generate_synthetic_board(rng, stone_dist(rng), territory);
        const auto feat = enc.encode(board);
        std::vector<double> row(feat.size());
        for (std::size_t k = 0; k < feat.size(); ++k) row[k] = static_cast<double>(feat[k]);
        ds.x.push_back(std::move(row));
        ds.y_reg.push_back(static_cast<double>(territory));
        ds.y_cls.push_back(territory >= 0.0f ? "black" : "white");
    }
    return ds;
}

}  // namespace cypha::bench
