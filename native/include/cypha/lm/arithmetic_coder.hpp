#pragma once

/// 32-bit range / arithmetic coder for predictive neural compression (LLMZip-style).
/// Consumes integer CDF mass over a finite alphabet; pairs with ``predictive_codec``.

#include <cstdint>
#include <vector>

namespace cypha::cyphalm {

/// Cumulative distribution: ``cum[0]=0``, ``cum[k+1]=cum[k]+count[k]``, ``cum[n]=total``.
/// All counts must be ≥ 1 so every symbol is encodable.
struct SymbolCdf {
    std::vector<std::uint32_t> cum;  // size = alphabet + 1
    std::uint32_t total() const { return cum.empty() ? 0u : cum.back(); }
    std::size_t alphabet() const { return cum.size() < 2 ? 0 : cum.size() - 1; }
};

/// Build a CDF from natural-log probabilities (softmax → scaled integer masses).
/// ``scale`` is the total mass target (default 1<<20). Each symbol gets ≥1 count.
SymbolCdf cdf_from_log_probs(const std::vector<double>& log_probs, std::uint32_t scale = (1u << 20));

class ArithmeticEncoder {
 public:
    ArithmeticEncoder();
    void encode(std::uint32_t symbol, const SymbolCdf& cdf);
    /// Flush final bits; returns the compressed bitstream.
    std::vector<std::uint8_t> finish();

 private:
    void renorm();
    void emit_bit(int bit);
    void emit_pending(int bit);

    std::uint32_t low_{0};
    std::uint32_t high_{0xffffffffu};
    int pending_{0};
    std::vector<std::uint8_t> out_;
    int bit_buf_{0};
    int bit_count_{0};
};

class ArithmeticDecoder {
 public:
    explicit ArithmeticDecoder(const std::vector<std::uint8_t>& bytes);
    std::uint32_t decode(const SymbolCdf& cdf);

 private:
    int get_bit();
    void renorm();

    std::uint32_t low_{0};
    std::uint32_t high_{0xffffffffu};
    std::uint32_t code_{0};
    const std::vector<std::uint8_t>* bytes_{nullptr};
    std::size_t byte_pos_{0};
    int bit_buf_{0};
    int bit_count_{0};
};

}  // namespace cypha::cyphalm
