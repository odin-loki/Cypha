#pragma once

/// InfiniGram's query over a compressed index (RAM_PROFILE M5): a bidirectional
/// FM-index. It keeps the BWT of the training text and the BWT of the reversed
/// text, each in a wavelet tree, and nothing else: no suffix array and no copy
/// of the text. On the 8 MiB protocol's training text (19.6 MB) it takes 26 MB
/// where InfiniGram takes 78 MB (text 19.6 + suffix array 58.3).
///
/// Answers are InfiniGram's exactly: the same n, total and 256 counts for every
/// query (checked by CTest native_fm_infinigram_test and on the whole holdout).
///
/// How a query runs. The context's suffix grows to the left one byte a step
/// (backward search on the text's BWT). The reversed text's row range is kept
/// in step with it (the bidirectional trick, Lam et al. 2009). The reversed
/// text's BWT holds, for each occurrence, the byte that follows it in the text,
/// so the next-byte counts are symbol counts over that range.
///
/// The wavelet trees are alphabetic and shaped by weight: each node splits its
/// byte range where the two sides' counts are closest. A common byte sits near
/// the root (5.3 bits a symbol on average instead of 8), and byte order is kept,
/// which the bidirectional step needs.
///
/// A query costs one step a context byte. A caller that reads text one byte at
/// a time passes a Cursor: when the last match plus the new byte still occurs,
/// that is one step, so a long copy costs the same per byte as a short one.

#include "cypha/cyphalm/infinigram.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace cypha::cyphalm {

namespace fm_detail {

/// Bit vector with rank: one 32-bit count per 512 bits (6.25% over the bits).
struct RankBits {
    std::vector<std::uint64_t> w;
    std::vector<std::uint32_t> cum;
    void reset(std::size_t nbits);
    void finish();
    std::size_t rank1(std::size_t i) const {
        const std::size_t word = i >> 6, blk = word >> 3;
        std::size_t r = cum[blk];
        for (std::size_t k = blk << 3; k < word; ++k) r += static_cast<std::size_t>(std::popcount(w[k]));
        if (i & 63) r += static_cast<std::size_t>(std::popcount(w[word] & ((std::uint64_t{1} << (i & 63)) - 1)));
        return r;
    }
    std::size_t bytes() const { return w.size() * 8 + cum.size() * 4; }
};

/// Wavelet tree over a byte sequence, alphabetic and shaped by weight.
class AlphaTree {
 public:
    void build(std::vector<std::uint8_t> seq);
    /// Ranks of c in [0, lo) and [0, hi), and the count of symbols below c in [lo, hi).
    void step(int c, std::size_t lo, std::size_t hi, std::size_t& rlo, std::size_t& rhi, std::size_t& less) const {
        less = 0;
        if (!freq_[c]) {
            rlo = rhi = 0;
            return;
        }
        int v = root_;
        while (v >= 0) {
            const Node& nd = nodes_[static_cast<std::size_t>(v)];
            const std::size_t o1 = nd.bits.rank1(lo), h1 = nd.bits.rank1(hi);
            if (c > nd.mid) {
                less += (hi - h1) - (lo - o1);
                lo = o1, hi = h1, v = nd.right;
            } else {
                lo -= o1, hi -= h1, v = nd.left;
            }
        }
        rlo = lo, rhi = hi;
    }
    /// Adds the count of each symbol in [lo, hi) to out[symbol].
    void counts(std::size_t lo, std::size_t hi, std::uint32_t* out) const { counts_(root_, lo, hi, out); }
    std::size_t bytes() const;
    void write(std::ostream& out) const;
    void read(std::istream& in);

 private:
    struct Node {
        RankBits bits;              // 1: the symbol goes right
        int mid = 0;                // symbols <= mid go left
        int left = -1, right = -1;  // a node, or -(symbol + 1) for a leaf
    };
    int make_(const std::vector<int>& syms, std::size_t a, std::size_t b, std::vector<std::uint8_t>& seq);
    void counts_(int v, std::size_t lo, std::size_t hi, std::uint32_t* out) const;
    std::vector<Node> nodes_;
    std::uint64_t freq_[256] = {};
    int root_ = -1;
};

}  // namespace fm_detail

class FmInfiniGram {
 public:
    /// Index a copy of text[0..n) (at most 2^31 - 1 bytes). Peak while building:
    /// about 7 bytes a text byte (6 when the text is moved in).
    FmInfiniGram(const std::uint8_t* text, std::size_t n);
    /// Index ``text`` and consume it (it is reversed in place for the second BWT).
    explicit FmInfiniGram(std::vector<std::uint8_t>&& text);

    /// InfiniGram::query, same arguments and same result.
    InfiniGram::Result query(const std::uint8_t* ctx, std::size_t len, int max_n, int hint = -1,
                             std::uint64_t min_total = 1) const;

    /// The last match, for a caller that queries the same text one byte further
    /// on with the hint chain (hint = last n + 1). The query checks that the
    /// bytes before the new one are the last context's (a memcmp of at most cap
    /// bytes, usually the match) before it extends the match by the new byte,
    /// so a stale or reused cursor costs time, never a wrong answer. One cursor
    /// a thread.
    struct Cursor {
        int n = -1;                       // the last match's length (-1: none yet)
        std::size_t loR = 0, sz = 0;      // its rows in the reversed text's BWT
        std::size_t need = 1, cap = 0;    // the last query's min_total and length bound
        std::vector<std::uint8_t> tail;   // the last context's last cap bytes
    };
    InfiniGram::Result query(const std::uint8_t* ctx, std::size_t len, int max_n, int hint, std::uint64_t min_total,
                             Cursor& cur) const;

    /// InfiniGram::match_prefix without the position: the longest prefix of s[0..len)
    /// (at most cap bytes) that occurs in the corpus, and how often (the corpus size
    /// for the empty prefix). A prefix grows to the right: one step on the reversed BWT.
    std::size_t match_prefix(const std::uint8_t* s, std::size_t len, std::size_t cap, std::size_t& count) const;

    /// A stored index (RAM_PROFILE M8): "FMIG", version 1, little-endian, with a checksum.
    /// Loading reads the arrays as they are, so serving needs no text and no build.
    void save(const std::string& path) const;
    static FmInfiniGram load(const std::string& path);

    std::size_t size() const { return n_; }
    /// Bytes the index holds.
    std::size_t bytes() const { return fwd_.bytes() + rev_.bytes(); }

 private:
    FmInfiniGram() = default;
    struct Found {
        int n = 0;
        std::size_t loR = 0, sz = 0;
    };
    Found search_(const std::uint8_t* ctx, std::size_t len, std::size_t cap, std::size_t need) const;
    InfiniGram::Result counts_(int n, std::size_t loR, std::size_t sz) const;
    static std::size_t cap_(std::size_t len, int max_n, int hint);

    std::size_t n_ = 0;
    std::size_t C_[256] = {};          // first row of each byte (row 0 is the empty suffix)
    std::size_t dT_ = 0, dR_ = 0;      // the $ row of each BWT (it stores 0)
    fm_detail::AlphaTree fwd_, rev_;   // BWT of text$ and of reversed(text)$
};

}  // namespace cypha::cyphalm
