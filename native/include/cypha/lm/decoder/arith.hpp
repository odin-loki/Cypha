#pragma once

/// The decoder's arithmetic coder (docs/DECODER.md, docs/reports/K5_MIXDEC.md). Odin Loch, 2026-11-12.
///
/// Generation as the reverse of compression: a range decoder reads a stream of random bytes and splits its interval in
/// proportion to the model's probabilities, so each symbol is drawn with the model's probability (exact up to the
/// 24-bit quantisation of the probabilities below). The encoder is the compressor: encoding the decoded symbols with
/// the same distributions gives back the same bytes (the selftest checks the round trip).
///
/// A 64-bit range coder: range is kept at or above 2^56 by shifting out one byte at a time (LZMA's scheme widened from
/// 32 to 64 bits, with its cache byte for the carry). Each distribution is quantised to integer frequencies that sum to
/// 2^24, every symbol of weight above 0 at least 1, so range / 2^24 is at least 2^32 and the rounding error of an interval is below 2^-32.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace cypha::cyphalm::decoder {

inline constexpr int kFreqBits = 24;
inline constexpr std::uint64_t kFreqTotal = std::uint64_t{1} << kFreqBits;
inline constexpr std::uint64_t kTop = std::uint64_t{1} << 56;

/// p[0, k) (any non-negative weights, not all 0) as integer frequencies summing to kFreqTotal, each at least 1, into
/// cum (k + 1 entries, cum[0] = 0); a weight above 0 gets at least 1, a weight of 0 gets 0. The largest weight takes the
/// rounding remainder. k must be at most kFreqTotal / 2.
inline void quantise(const double* p, std::size_t k, std::vector<std::uint64_t>& cum) {
    cum.assign(k + 1, 0);
    double z = 0;
    std::size_t top = 0;
    for (std::size_t i = 0; i < k; ++i) {
        z += std::max(p[i], 0.0);
        if (p[i] > p[top]) top = i;
    }
    std::vector<std::uint64_t> f(k, 0);
    std::uint64_t used = 0;
    for (std::size_t i = 0; i < k; ++i)
        if (p[i] > 0) f[i] = 1, ++used;  // a weight of 0 (a vetoed word) is never decoded
    const std::uint64_t spare = kFreqTotal - used;  // each symbol of weight > 0 has 1; the rest is shared in proportion
    for (std::size_t i = 0; i < k; ++i) {
        const std::uint64_t x = z > 0 ? static_cast<std::uint64_t>(std::floor(std::max(p[i], 0.0) / z * static_cast<double>(spare))) : 0;
        f[i] += x, used += x;
    }
    if (used > kFreqTotal) used = kFreqTotal;  // cannot happen (floors), kept for safety
    f[top] += kFreqTotal - used;
    for (std::size_t i = 0; i < k; ++i) cum[i + 1] = cum[i] + f[i];
}

/// The decoder. Its bytes come from `next` (random bytes when writing; an encoder's output in the selftest).
class ArithDecoder {
 public:
    explicit ArithDecoder(std::function<std::uint8_t()> next) : next_(std::move(next)) {
        for (int i = 0; i < 9; ++i) code_ = (code_ << 8) | next_();  // the first byte is the encoder's cache byte (0)
    }
    /// One symbol from the weights p[0, k).
    std::size_t decode(const double* p, std::size_t k) {
        if (k <= 1) return 0;
        quantise(p, k, cum_);
        const std::uint64_t r = range_ >> kFreqBits;
        const std::uint64_t v = std::min<std::uint64_t>(code_ / r, kFreqTotal - 1);
        const std::size_t s = static_cast<std::size_t>(std::upper_bound(cum_.begin(), cum_.end(), v) - cum_.begin()) - 1;
        code_ -= r * cum_[s];
        range_ = s + 1 == k ? range_ - r * cum_[s] : r * (cum_[s + 1] - cum_[s]);
        while (range_ < kTop) {
            code_ = (code_ << 8) | next_();
            range_ <<= 8;
        }
        ++symbols_;
        return s;
    }
    std::uint64_t symbols() const { return symbols_; }

 private:
    std::function<std::uint8_t()> next_;
    std::uint64_t code_ = 0, range_ = ~std::uint64_t{0}, symbols_ = 0;
    std::vector<std::uint64_t> cum_;
};

/// The encoder (the compressor's side), for the round-trip check.
class ArithEncoder {
 public:
    void encode(const double* p, std::size_t k, std::size_t s) {
        if (k <= 1) return;
        quantise(p, k, cum_);
        const std::uint64_t r = range_ >> kFreqBits;
        low_ += static_cast<unsigned __int128>(r) * cum_[s];
        range_ = s + 1 == k ? range_ - r * cum_[s] : r * (cum_[s + 1] - cum_[s]);
        while (range_ < kTop) {
            shift_low();
            range_ <<= 8;
        }
    }
    std::vector<std::uint8_t> finish() {
        for (int i = 0; i < 9; ++i) shift_low();
        return out_;
    }

 private:
    void shift_low() {
        const unsigned __int128 one = 1;
        if (low_ < ((one << 64) - (one << 56)) || low_ >= (one << 64)) {
            const auto carry = static_cast<std::uint8_t>(low_ >> 64);
            std::uint8_t c = cache_;
            do {
                out_.push_back(static_cast<std::uint8_t>(c + carry));
                c = 0xFF;
            } while (--pending_ != 0);
            cache_ = static_cast<std::uint8_t>(low_ >> 56);
        }
        ++pending_;
        low_ = (low_ & ((one << 56) - 1)) << 8;
    }
    unsigned __int128 low_ = 0;
    std::uint64_t range_ = ~std::uint64_t{0};
    std::uint8_t cache_ = 0;
    std::uint64_t pending_ = 1;
    std::vector<std::uint8_t> out_;
    std::vector<std::uint64_t> cum_;
};

/// Selftest: random distributions, decode random bytes, encode the decoded symbols, decode the encoder's bytes: the same
/// symbols. And the decoded frequencies of a fixed distribution match it.
inline bool arith_selftest(std::string& err) {
    std::mt19937_64 rng(20261112);
    for (int trial = 0; trial < 20; ++trial) {
        const std::size_t n = 2000;
        std::vector<std::vector<double>> dists(n);
        for (auto& d : dists) {
            d.resize(1 + rng() % 300);
            for (auto& x : d) x = std::pow(static_cast<double>(rng() >> 11) * 0x1.0p-53, 4.0);
        }
        std::mt19937_64 src(static_cast<std::uint64_t>(trial));
        ArithDecoder dec([&] { return static_cast<std::uint8_t>(src() >> 56); });
        std::vector<std::size_t> sym(n);
        for (std::size_t i = 0; i < n; ++i) sym[i] = dec.decode(dists[i].data(), dists[i].size());
        ArithEncoder enc;
        for (std::size_t i = 0; i < n; ++i) enc.encode(dists[i].data(), dists[i].size(), sym[i]);
        const auto bytes = enc.finish();
        std::size_t at = 0;
        ArithDecoder back([&] { return at < bytes.size() ? bytes[at++] : std::uint8_t{0}; });
        for (std::size_t i = 0; i < n; ++i)
            if (back.decode(dists[i].data(), dists[i].size()) != sym[i]) return err = "arith: round trip trial " + std::to_string(trial), false;
    }
    const double p[4] = {0.5, 0.25, 0.125, 0.125};
    std::mt19937_64 src(7);
    ArithDecoder dec([&] { return static_cast<std::uint8_t>(src() >> 56); });
    double c[4] = {};
    const int N = 200000;
    for (int i = 0; i < N; ++i) c[dec.decode(p, 4)] += 1;
    for (int s = 0; s < 4; ++s)
        if (std::abs(c[s] / N - p[s]) > 0.005) return err = "arith: frequencies", false;
    return true;
}

}  // namespace cypha::cyphalm::decoder
