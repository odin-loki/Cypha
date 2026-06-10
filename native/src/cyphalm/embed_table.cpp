#include "cypha/cyphalm/embed_table.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace cypha::cyphalm {

namespace {

// Park-Miller LCG matching common test seeding.
struct Rng {
    std::uint32_t state;
    explicit Rng(std::uint32_t seed) : state(seed ? seed : 1u) {}
    double normal() {
        double u1 = static_cast<double>(next()) / static_cast<double>(0x7FFFFFFFu);
        double u2 = static_cast<double>(next()) / static_cast<double>(0x7FFFFFFFu);
        if (u1 < 1e-12) u1 = 1e-12;
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.14159265358979323846 * u2);
    }
    std::uint32_t next() {
        state = static_cast<std::uint32_t>((static_cast<std::uint64_t>(state) * 48271u) % 2147483647u);
        return state;
    }
};

}  // namespace

EmbedTable::EmbedTable(std::uint32_t vocab_size, std::uint32_t d_embed, std::uint32_t seed)
    : vocab_size_(vocab_size), d_embed_(d_embed), table_(vocab_size * d_embed) {
    Rng rng(seed);
    for (auto& v : table_) v = rng.normal() * 0.02;
}

const double* EmbedTable::embed(std::uint32_t token_id) const {
    if (token_id >= vocab_size_) throw std::out_of_range("token_id out of range");
    return table_.data() + static_cast<std::size_t>(token_id) * d_embed_;
}

std::vector<double> EmbedTable::embed_vec(std::uint32_t token_id) const {
    const double* p = embed(token_id);
    return std::vector<double>(p, p + d_embed_);
}

}  // namespace cypha::cyphalm
