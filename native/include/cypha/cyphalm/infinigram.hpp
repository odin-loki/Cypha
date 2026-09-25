#pragma once

/// Byte-level ∞-gram model over a training corpus (after Liu et al. 2024,
/// "Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion
/// Tokens"): a suffix array finds the longest suffix of the current context
/// that occurs in the corpus, and the next-byte distribution is the count of
/// each byte following *every* occurrence of it. hp's match models follow only
/// the most recent occurrence of a context; this counts all of them.
///
/// Index file (cyphalm_infinigram_build): "IGR2", uint64 n, uint64 bits, n
/// text bytes (padded to 8), then the n suffix-array entries bit-packed at
/// ``bits`` = ceil(log2 n) each (27 for 95 MB: 4.4 bytes a text byte instead
/// of 5), 8 bytes of padding. "IGR1" (32-bit entries) still loads. Mapped
/// read-only, so every process serving it shares one copy. Or skip the file:
/// ``open`` on the plain corpus builds the same index in memory at load time.
/// It maps the corpus too (shared page cache, not private memory), sorts into
/// one 4-bytes-a-byte buffer and packs that in place, so the peak is 4 bytes
/// a text byte and the index then keeps ceil(log2 n) / 8 of private memory.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace cypha::cyphalm {

class InfiniGram {
 public:
    /// Build the index for ``text`` (libsais induced sorting, O(n)) and write it to ``path``.
    static void build(const std::uint8_t* text, std::size_t n, const std::string& path);

    /// Map a stored index (IGR1 / IGR2).
    explicit InfiniGram(const std::string& path);
    /// Index ``text`` in memory, just in time (~6 s for 95 MB on one core;
    /// every core with cmake -DCYPHA_INFINIGRAM_OPENMP=ON).
    InfiniGram(const std::uint8_t* text, std::size_t n);
    /// The same, keeping ``text`` itself (no copy).
    explicit InfiniGram(std::vector<std::uint8_t>&& text);
    /// A stored index, or a plain-text corpus indexed on the spot (its first
    /// ``max_bytes`` bytes; 0 = all), chosen by the file's magic. The corpus
    /// stays mapped read-only while the index lives: do not rewrite the file
    /// under a running server.
    static std::shared_ptr<const InfiniGram> open(const std::string& path, std::size_t max_bytes = 0);
    /// True when ``path`` is a stored index (IGR1 / IGR2 magic), false for a
    /// plain-text corpus (``open`` then indexes its bytes). Throws if unreadable.
    static bool is_index_file(const std::string& path);
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
    /// ``hint`` (-1: none) must bound that length: the previous byte's n + 1
    /// (a match grows by at most one byte per byte read), or a length known
    /// to occur (a backoff). The bound itself is tried first, then bisected
    /// below; the result is the same as without it. ``min_total`` > 1 asks
    /// instead for the longest suffix followed by a byte at least that many
    /// times (``hint`` must then bound that length; n = 0 when even the
    /// empty context has fewer, a corpus that short).
    Result query(const std::uint8_t* ctx, std::size_t len, int max_n, int hint = -1,
                 std::uint64_t min_total = 1) const;

    std::size_t size() const { return n_; }

    /// Longest prefix of ``s[0..len)`` (at most ``cap`` bytes) that occurs in
    /// the corpus, at least ``at_least`` (a known lower bound); ``pos`` is one
    /// corpus position where it occurs and ``count`` how often.
    std::size_t match_prefix(const std::uint8_t* s, std::size_t len, std::size_t cap, std::size_t at_least,
                             std::size_t& pos, std::size_t& count) const;

 private:
    InfiniGram() = default;
    // Sort text_[0..n_) and pack the suffix array into own_packed_.
    void index_text_();

    // [lo, hi) of suffixes starting with pat[0..m).
    void range(const std::uint8_t* pat, std::size_t m, std::size_t& lo, std::size_t& hi) const;

    // Suffix-array entry i (bit-packed, or plain uint32 for IGR1).
    std::size_t sa(std::size_t i) const {
        if (bits_ == 32) return sa32_[i];
        const std::size_t bit = i * static_cast<std::size_t>(bits_);
        std::uint64_t w;
        std::memcpy(&w, packed_ + bit / 8, 8);
        return static_cast<std::size_t>((w >> (bit % 8)) & mask_);
    }

    void* map_ = nullptr;
    std::size_t map_len_ = 0;
    std::size_t n_ = 0;
    const std::uint8_t* text_ = nullptr;
    const std::uint32_t* sa32_ = nullptr;
    const std::uint8_t* packed_ = nullptr;
    int bits_ = 32;
    std::uint64_t mask_ = 0;
    struct FreeBytes {
        void operator()(std::uint8_t* p) const noexcept { std::free(p); }
    };
    std::vector<std::uint8_t> own_text_;  // in-memory (just-in-time) index: corpus when not mapped
    std::unique_ptr<std::uint8_t, FreeBytes> own_packed_;  // and its packed suffix array
};

}  // namespace cypha::cyphalm
