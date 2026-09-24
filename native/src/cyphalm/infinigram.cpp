#include "cypha/cyphalm/infinigram.hpp"

#include "libsais.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace cypha::cyphalm {

namespace {

// Suffix array of text[0..n) (libsais, induced sorting), packed little-endian
// at ``bits`` = ceil(log2 n) per entry with 8 bytes of padding.
std::vector<std::uint8_t> sort_and_pack(const std::uint8_t* text, std::size_t n, int& bits) {
    if (n == 0 || n >= (std::size_t{1} << 31) - 2) throw std::runtime_error("InfiniGram: bad corpus size");
    std::vector<std::int32_t> SA(n);
    if (libsais(text, SA.data(), static_cast<std::int32_t>(n), 0, nullptr) != 0)
        throw std::runtime_error("InfiniGram: suffix sort failed");
    bits = 1;
    while ((std::size_t{1} << bits) < n) ++bits;
    std::vector<std::uint8_t> packed((n * static_cast<std::size_t>(bits) + 7) / 8 + 8, 0);
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t v = static_cast<std::uint64_t>(SA[i]);
        const std::size_t bit = i * static_cast<std::size_t>(bits);
        std::uint64_t w;
        std::memcpy(&w, packed.data() + bit / 8, 8);
        w |= v << (bit % 8);
        std::memcpy(packed.data() + bit / 8, &w, 8);
    }
    return packed;
}

}  // namespace

void InfiniGram::build(const std::uint8_t* text, std::size_t n, const std::string& path) {
    int bits = 0;
    const std::vector<std::uint8_t> packed = sort_and_pack(text, n, bits);
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("InfiniGram: cannot write " + path);
    const char magic[4] = {'I', 'G', 'R', '2'};
    out.write(magic, 4);
    const std::uint64_t n64 = n, bits64 = static_cast<std::uint64_t>(bits);
    out.write(reinterpret_cast<const char*>(&n64), 8);
    out.write(reinterpret_cast<const char*>(&bits64), 8);
    out.write(reinterpret_cast<const char*>(text), static_cast<std::streamsize>(n));
    const std::size_t pad = (8 - (20 + n) % 8) % 8;
    const char zeros[8] = {};
    out.write(zeros, static_cast<std::streamsize>(pad));
    out.write(reinterpret_cast<const char*>(packed.data()), static_cast<std::streamsize>(packed.size()));
    if (!out) throw std::runtime_error("InfiniGram: write failed " + path);
}

InfiniGram::InfiniGram(const std::uint8_t* text, std::size_t n)
    : own_text_(text, text + n) {
    own_packed_ = sort_and_pack(own_text_.data(), n, bits_);
    n_ = n;
    mask_ = (std::uint64_t{1} << bits_) - 1;
    text_ = own_text_.data();
    packed_ = own_packed_.data();
}

std::shared_ptr<const InfiniGram> InfiniGram::open(const std::string& path, std::size_t max_bytes) {
    char magic[4] = {};
    {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("InfiniGram: cannot open " + path);
        f.read(magic, 4);
    }
    if (std::memcmp(magic, "IGR1", 4) == 0 || std::memcmp(magic, "IGR2", 4) == 0)
        return std::make_shared<const InfiniGram>(path);
    // Plain text: index it now (just in time) instead of storing an index.
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    std::size_t n = static_cast<std::size_t>(f.tellg());
    if (max_bytes > 0 && max_bytes < n) n = max_bytes;
    std::vector<std::uint8_t> text(n);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(text.data()), static_cast<std::streamsize>(n));
    if (!f) throw std::runtime_error("InfiniGram: cannot read " + path);
    return std::make_shared<const InfiniGram>(text.data(), n);
}

InfiniGram::InfiniGram(const std::string& path) {
#if defined(_WIN32)
    throw std::runtime_error("InfiniGram: not supported on Windows");
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("InfiniGram: cannot open " + path);
    struct stat st {};
    ::fstat(fd, &st);
    map_len_ = static_cast<std::size_t>(st.st_size);
    map_ = ::mmap(nullptr, map_len_, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (map_ == MAP_FAILED) {
        map_ = nullptr;
        throw std::runtime_error("InfiniGram: mmap failed " + path);
    }
    const auto* p = static_cast<const std::uint8_t*>(map_);
    const bool v2 = map_len_ >= 20 && std::memcmp(p, "IGR2", 4) == 0;
    if (!v2 && (map_len_ < 12 || std::memcmp(p, "IGR1", 4) != 0)) throw std::runtime_error("InfiniGram: bad index " + path);
    std::uint64_t n = 0;
    std::memcpy(&n, p + 4, 8);
    n_ = static_cast<std::size_t>(n);
    if (v2) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, p + 12, 8);
        bits_ = static_cast<int>(bits);
        mask_ = (std::uint64_t{1} << bits_) - 1;
        text_ = p + 20;
        const std::size_t pad = (8 - (20 + n_) % 8) % 8;
        packed_ = p + 20 + n_ + pad;
        if (bits_ < 1 || bits_ > 32 || 20 + n_ + pad + (n_ * static_cast<std::size_t>(bits_) + 7) / 8 + 8 > map_len_)
            throw std::runtime_error("InfiniGram: truncated index " + path);
    } else {
        text_ = p + 12;
        const std::size_t pad = (8 - (12 + n_) % 8) % 8;
        sa32_ = reinterpret_cast<const std::uint32_t*>(p + 12 + n_ + pad);
        if (12 + n_ + pad + 4 * n_ > map_len_) throw std::runtime_error("InfiniGram: truncated index " + path);
    }
#endif
}

InfiniGram::~InfiniGram() {
#if !defined(_WIN32)
    if (map_ != nullptr) ::munmap(map_, map_len_);
#endif
}

void InfiniGram::range(const std::uint8_t* pat, std::size_t m, std::size_t& lo, std::size_t& hi) const {
    // Compare the suffix at SA[i] with pat over m bytes (a shorter suffix sorts first).
    auto cmp = [&](std::size_t i) {
        const std::size_t pos = sa(i);
        const std::size_t avail = n_ - pos;
        const std::size_t k = std::min(avail, m);
        const int c = std::memcmp(text_ + pos, pat, k);
        if (c != 0) return c;
        return avail < m ? -1 : 0;
    };
    std::size_t a = 0, b = n_;
    while (a < b) {
        const std::size_t mid = a + (b - a) / 2;
        if (cmp(mid) < 0) a = mid + 1;
        else b = mid;
    }
    lo = a;
    b = n_;
    while (a < b) {
        const std::size_t mid = a + (b - a) / 2;
        if (cmp(mid) <= 0) a = mid + 1;
        else b = mid;
    }
    hi = a;
}

InfiniGram::Result InfiniGram::query(const std::uint8_t* ctx, std::size_t len, int max_n, int hint) const {
    Result r;
    const int cap = static_cast<int>(std::min<std::size_t>(len, static_cast<std::size_t>(std::max(0, max_n))));
    // Longest n in [0, cap] whose suffix occurs followed by some byte. Occurrence
    // is monotone in n, so bisect; ``hint`` caps it (the longest match grows by
    // at most one byte per step).
    auto usable = [&](int n, std::size_t& lo, std::size_t& hi) {
        if (n == 0) {
            lo = 0;
            hi = n_;
        } else {
            range(ctx + len - static_cast<std::size_t>(n), static_cast<std::size_t>(n), lo, hi);
        }
        // Occurrences at the very end of the corpus have no next byte; they sort
        // first in the range (shortest suffix).
        while (lo < hi && sa(lo) + static_cast<std::size_t>(n) >= n_) ++lo;
        return lo < hi;
    };
    int good = 0, bad = (hint >= 0 ? std::min(cap, hint) : cap) + 1;
    std::size_t glo = 0, ghi = 0;
    usable(0, glo, ghi);
    while (bad - good > 1) {
        const int mid = good + (bad - good) / 2;
        std::size_t lo = 0, hi = 0;
        if (usable(mid, lo, hi)) {
            good = mid;
            glo = lo;
            ghi = hi;
        } else {
            bad = mid;
        }
    }
    r.n = good;
    r.total = ghi - glo;
    // Suffixes in [glo, ghi) are sorted by their byte at offset n: walk the
    // runs with one binary search per distinct next byte.
    std::size_t i = glo;
    const std::size_t off = static_cast<std::size_t>(good);
    while (i < ghi) {
        const std::uint8_t b = text_[sa(i) + off];
        std::size_t a = i + 1, e = ghi;
        while (a < e) {
            const std::size_t mid = a + (e - a) / 2;
            if (text_[sa(mid) + off] <= b) a = mid + 1;
            else e = mid;
        }
        r.count[b] += static_cast<std::uint32_t>(a - i);
        i = a;
    }
    return r;
}

}  // namespace cypha::cyphalm
