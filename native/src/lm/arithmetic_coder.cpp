#include "cypha/cyphalm/arithmetic_coder.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cypha::cyphalm {

SymbolCdf cdf_from_log_probs(const std::vector<double>& log_probs, std::uint32_t scale) {
    if (log_probs.empty()) {
        throw std::runtime_error("cdf_from_log_probs: empty log_probs");
    }
    if (scale < static_cast<std::uint32_t>(log_probs.size())) {
        scale = static_cast<std::uint32_t>(log_probs.size());
    }
    const std::size_t n = log_probs.size();
    double max_lp = log_probs[0];
    for (double v : log_probs) max_lp = std::max(max_lp, v);

    std::vector<double> p(n);
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        p[i] = std::exp(log_probs[i] - max_lp);
        sum += p[i];
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) {
        throw std::runtime_error("cdf_from_log_probs: invalid probability mass");
    }

    SymbolCdf cdf;
    cdf.cum.assign(n + 1, 0);
    std::uint32_t remaining = scale;
    // First pass: floor masses, guarantee ≥1.
    std::vector<std::uint32_t> counts(n, 1u);
    remaining -= static_cast<std::uint32_t>(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double share = p[i] / sum;
        const auto add = static_cast<std::uint32_t>(share * static_cast<double>(remaining));
        counts[i] += add;
    }
    std::uint32_t used = 0;
    for (std::uint32_t c : counts) used += c;
    // Assign leftover to the mode (largest p).
    if (used < scale) {
        std::size_t best = 0;
        for (std::size_t i = 1; i < n; ++i) {
            if (p[i] > p[best]) best = i;
        }
        counts[best] += (scale - used);
    } else if (used > scale) {
        // Extremely rare with floor+share; trim from mode.
        std::size_t best = 0;
        for (std::size_t i = 1; i < n; ++i) {
            if (counts[i] > counts[best]) best = i;
        }
        const std::uint32_t over = used - scale;
        if (counts[best] > 1u + over) counts[best] -= over;
    }
    for (std::size_t i = 0; i < n; ++i) {
        cdf.cum[i + 1] = cdf.cum[i] + counts[i];
    }
    return cdf;
}

ArithmeticEncoder::ArithmeticEncoder() = default;

void ArithmeticEncoder::emit_bit(int bit) {
    bit_buf_ = (bit_buf_ << 1) | (bit & 1);
    ++bit_count_;
    if (bit_count_ == 8) {
        out_.push_back(static_cast<std::uint8_t>(bit_buf_ & 0xff));
        bit_buf_ = 0;
        bit_count_ = 0;
    }
}

void ArithmeticEncoder::emit_pending(int bit) {
    emit_bit(bit);
    while (pending_ > 0) {
        emit_bit(1 - bit);
        --pending_;
    }
}

void ArithmeticEncoder::renorm() {
    constexpr std::uint32_t kHalf = 0x80000000u;
    constexpr std::uint32_t kQuarter = 0x40000000u;
    for (;;) {
        if (high_ < kHalf) {
            emit_pending(0);
        } else if (low_ >= kHalf) {
            emit_pending(1);
            low_ -= kHalf;
            high_ -= kHalf;
        } else if (low_ >= kQuarter && high_ < 3u * kQuarter) {
            ++pending_;
            low_ -= kQuarter;
            high_ -= kQuarter;
        } else {
            break;
        }
        low_ <<= 1;
        high_ = (high_ << 1) | 1u;
    }
}

void ArithmeticEncoder::encode(std::uint32_t symbol, const SymbolCdf& cdf) {
    if (symbol >= cdf.alphabet()) {
        throw std::runtime_error("ArithmeticEncoder::encode: symbol out of range");
    }
    const std::uint32_t total = cdf.total();
    if (total == 0) {
        throw std::runtime_error("ArithmeticEncoder::encode: empty CDF");
    }
    const std::uint64_t range = static_cast<std::uint64_t>(high_ - low_) + 1ull;
    const std::uint32_t c_lo = cdf.cum[symbol];
    const std::uint32_t c_hi = cdf.cum[symbol + 1];
    high_ = low_ + static_cast<std::uint32_t>((range * c_hi) / total - 1ull);
    low_ = low_ + static_cast<std::uint32_t>((range * c_lo) / total);
    renorm();
}

std::vector<std::uint8_t> ArithmeticEncoder::finish() {
    ++pending_;
    if (low_ < 0x40000000u) {
        emit_pending(0);
    } else {
        emit_pending(1);
    }
    if (bit_count_ > 0) {
        out_.push_back(static_cast<std::uint8_t>((bit_buf_ << (8 - bit_count_)) & 0xff));
        bit_buf_ = 0;
        bit_count_ = 0;
    }
    return out_;
}

ArithmeticDecoder::ArithmeticDecoder(const std::vector<std::uint8_t>& bytes) : bytes_(&bytes) {
    for (int i = 0; i < 32; ++i) {
        code_ = (code_ << 1) | static_cast<std::uint32_t>(get_bit());
    }
}

int ArithmeticDecoder::get_bit() {
    if (bit_count_ == 0) {
        if (byte_pos_ < bytes_->size()) {
            bit_buf_ = (*bytes_)[byte_pos_++];
        } else {
            bit_buf_ = 0;
        }
        bit_count_ = 8;
    }
    const int bit = (bit_buf_ >> 7) & 1;
    bit_buf_ = (bit_buf_ << 1) & 0xff;
    --bit_count_;
    return bit;
}

void ArithmeticDecoder::renorm() {
    constexpr std::uint32_t kHalf = 0x80000000u;
    constexpr std::uint32_t kQuarter = 0x40000000u;
    for (;;) {
        if (high_ < kHalf) {
            // nothing
        } else if (low_ >= kHalf) {
            low_ -= kHalf;
            high_ -= kHalf;
            code_ -= kHalf;
        } else if (low_ >= kQuarter && high_ < 3u * kQuarter) {
            low_ -= kQuarter;
            high_ -= kQuarter;
            code_ -= kQuarter;
        } else {
            break;
        }
        low_ <<= 1;
        high_ = (high_ << 1) | 1u;
        code_ = (code_ << 1) | static_cast<std::uint32_t>(get_bit());
    }
}

std::uint32_t ArithmeticDecoder::decode(const SymbolCdf& cdf) {
    const std::uint32_t total = cdf.total();
    if (total == 0 || cdf.alphabet() == 0) {
        throw std::runtime_error("ArithmeticDecoder::decode: empty CDF");
    }
    const std::uint64_t range = static_cast<std::uint64_t>(high_ - low_) + 1ull;
    const std::uint64_t offset =
        ((static_cast<std::uint64_t>(code_ - low_) + 1ull) * total - 1ull) / range;
    // Find symbol with cum[s] <= offset < cum[s+1]
    std::uint32_t lo = 0;
    std::uint32_t hi = static_cast<std::uint32_t>(cdf.alphabet());
    while (lo + 1 < hi) {
        const std::uint32_t mid = (lo + hi) >> 1;
        if (cdf.cum[mid] <= offset) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const std::uint32_t symbol = lo;
    const std::uint32_t c_lo = cdf.cum[symbol];
    const std::uint32_t c_hi = cdf.cum[symbol + 1];
    high_ = low_ + static_cast<std::uint32_t>((range * c_hi) / total - 1ull);
    low_ = low_ + static_cast<std::uint32_t>((range * c_lo) / total);
    renorm();
    return symbol;
}

}  // namespace cypha::cyphalm
