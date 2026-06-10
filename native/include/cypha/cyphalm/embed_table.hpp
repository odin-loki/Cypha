#pragma once

#include <cstdint>
#include <vector>

namespace cypha::cyphalm {

/// Fixed lookup-table token embedding (IzaacEmbedding stub).
class EmbedTable {
public:
    EmbedTable(std::uint32_t vocab_size, std::uint32_t d_embed, std::uint32_t seed);

    std::uint32_t vocab_size() const { return vocab_size_; }
    std::uint32_t dim() const { return d_embed_; }

    const std::vector<double>& table() const { return table_; }
    std::vector<double>& table() { return table_; }

    const double* embed(std::uint32_t token_id) const;
    std::vector<double> embed_vec(std::uint32_t token_id) const;

private:
    std::uint32_t vocab_size_;
    std::uint32_t d_embed_;
    std::vector<double> table_;  // row-major [vocab_size, d_embed]
};

}  // namespace cypha::cyphalm
