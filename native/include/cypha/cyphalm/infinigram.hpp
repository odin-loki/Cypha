#pragma once

/// Byte-level ∞-gram model over a training corpus (after Liu et al. 2024,
/// "Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion
/// Tokens"): a suffix array finds the longest suffix of the current context
/// that occurs in the corpus, and the next-byte distribution is the count of
/// each byte following *every* occurrence of it. hp's match models follow only
/// the most recent occurrence of a context; this counts all of them.
///
/// Index file (cyphalm_infinigram_build): "IGR1", uint64 n, n text bytes
/// (padded to 8), n uint32 suffix-array entries. Mapped read-only, so every
/// process serving it shares one copy.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cypha::cyphalm {

class InfiniGram {
 public:
    /// Build the index for ``text`` (SA-IS, O(n)) and write it to ``path``.
    static void build(const std::uint8_t* text, std::size_t n, const std::string& path);

    explicit InfiniGram(const std::string& path);
    ~InfiniGram();
    InfiniGram(const InfiniGram&) = delete;
    InfiniGram& operator=(const InfiniGram&) = delete;

    struct Result {
        int n = 0;                              ///< length of the longest matching suffix
        std::uint64_t total = 0;                ///< occurrences followed by a byte
        std::array<std::uint32_t, 256> count{}; ///< next-byte counts over those occurrences
    };

    /// Next-byte counts after the longest suffix of ``ctx[0..len)`` (oldest
    /// byte first) that occurs in the corpus, trying at most ``max_n`` bytes.
    /// ``hint`` (the previous call's n + 1, or -1) bounds the search.
    Result query(const std::uint8_t* ctx, std::size_t len, int max_n, int hint = -1) const;

    std::size_t size() const { return n_; }

 private:
    // [lo, hi) of suffixes starting with pat[0..m).
    void range(const std::uint8_t* pat, std::size_t m, std::size_t& lo, std::size_t& hi) const;

    void* map_ = nullptr;
    std::size_t map_len_ = 0;
    std::size_t n_ = 0;
    const std::uint8_t* text_ = nullptr;
    const std::uint32_t* sa_ = nullptr;
};

}  // namespace cypha::cyphalm
