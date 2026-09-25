#include "cypha/cyphalm/infinigram.hpp"

#include "libsais.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <fstream>
#include <stdexcept>
#include <utility>
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
// at ``bits`` = ceil(log2 n) per entry with 8 bytes of padding. The sort
// writes 32-bit entries into the result's own buffer, which is then packed in
// place (entry i is read before any packed byte reaches it: i entries write
// i * bits / 8 < 4 i bytes) and shrunk, so the peak is 4 n bytes, not 4 n plus
// the packed copy.
struct Packed {
    std::uint8_t* data;  // std::malloc; the caller frees it
    std::size_t size;
};

Packed sort_and_pack(const std::uint8_t* text, std::size_t n, int& bits) {
    if (n == 0 || n >= (std::size_t{1} << 31) - 2) throw std::runtime_error("InfiniGram: bad corpus size");
    auto* buf = static_cast<std::uint8_t*>(std::malloc(4 * n + 8));
    if (buf == nullptr) throw std::bad_alloc();
    auto* sa = reinterpret_cast<std::int32_t*>(buf);
#if defined(LIBSAIS_OPENMP)
    const std::int32_t rc = libsais_omp(text, sa, static_cast<std::int32_t>(n), 0, nullptr, 0);  // 0: all threads
#else
    const std::int32_t rc = libsais(text, sa, static_cast<std::int32_t>(n), 0, nullptr);
#endif
    if (rc != 0) {
        std::free(buf);
        throw std::runtime_error("InfiniGram: suffix sort failed");
    }
    bits = 1;
    while ((std::size_t{1} << bits) < n) ++bits;
    std::uint64_t acc = 0;
    int have = 0;
    std::size_t out = 0;
    for (std::size_t i = 0; i < n; ++i) {
        std::uint32_t v;
        std::memcpy(&v, buf + 4 * i, 4);
        acc |= static_cast<std::uint64_t>(v) << have;
        for (have += bits; have >= 8; have -= 8, acc >>= 8) buf[out++] = static_cast<std::uint8_t>(acc);
    }
    if (have > 0) buf[out++] = static_cast<std::uint8_t>(acc);
    const std::size_t size = out + 8;
    std::memset(buf + out, 0, 8);
    if (void* smaller = std::realloc(buf, size)) buf = static_cast<std::uint8_t*>(smaller);
    return {buf, size};
}

}  // namespace

void InfiniGram::build(const std::uint8_t* text, std::size_t n, const std::string& path) {
    int bits = 0;
    const Packed packed = sort_and_pack(text, n, bits);
    const std::unique_ptr<std::uint8_t, FreeBytes> hold(packed.data);
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
    out.write(reinterpret_cast<const char*>(packed.data), static_cast<std::streamsize>(packed.size));
    if (!out) throw std::runtime_error("InfiniGram: write failed " + path);
}

InfiniGram::InfiniGram(const std::uint8_t* text, std::size_t n)
    : InfiniGram(std::vector<std::uint8_t>(text, text + n)) {}

InfiniGram::InfiniGram(std::vector<std::uint8_t>&& text) : own_text_(std::move(text)) {
    text_ = own_text_.data();
    n_ = own_text_.size();
    index_text_();
}

void InfiniGram::index_text_() {
    const Packed packed = sort_and_pack(text_, n_, bits_);
    own_packed_.reset(packed.data);
    mask_ = (std::uint64_t{1} << bits_) - 1;
    packed_ = own_packed_.get();
}

bool InfiniGram::is_index_file(const std::string& path) {
    char magic[4] = {};
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("InfiniGram: cannot open " + path);
    f.read(magic, 4);
    return std::memcmp(magic, "IGR1", 4) == 0 || std::memcmp(magic, "IGR2", 4) == 0;
}

std::shared_ptr<const InfiniGram> InfiniGram::open(const std::string& path, std::size_t max_bytes) {
    if (is_index_file(path)) return std::make_shared<const InfiniGram>(path);
    // Plain text: index it now (just in time) instead of storing an index.
#if !defined(_WIN32)
    // Map the corpus rather than copy it: page cache, shared, not private memory.
    std::shared_ptr<InfiniGram> g(new InfiniGram());
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("InfiniGram: cannot open " + path);
    struct stat st {};
    ::fstat(fd, &st);
    std::size_t n = static_cast<std::size_t>(st.st_size);
    if (max_bytes > 0 && max_bytes < n) n = max_bytes;
    if (n == 0) {
        ::close(fd);
        throw std::runtime_error("InfiniGram: empty corpus " + path);
    }
    void* map = ::mmap(nullptr, n, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (map == MAP_FAILED) throw std::runtime_error("InfiniGram: mmap failed " + path);
    ::madvise(map, n, MADV_WILLNEED);
    g->map_ = map;  // unmapped by the destructor, also if indexing throws
    g->map_len_ = n;
    g->text_ = static_cast<const std::uint8_t*>(map);
    g->n_ = n;
    g->index_text_();
    return g;
#else
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    std::size_t n = static_cast<std::size_t>(f.tellg());
    if (max_bytes > 0 && max_bytes < n) n = max_bytes;
    std::vector<std::uint8_t> text(n);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(text.data()), static_cast<std::streamsize>(n));
    if (!f) throw std::runtime_error("InfiniGram: cannot read " + path);
    return std::make_shared<const InfiniGram>(std::move(text));  // no second copy of the corpus
#endif
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
    // Lower bound; every suffix seen above pat also bounds the upper search.
    std::size_t a = 0, b = n_, above = n_;
    while (a < b) {
        const std::size_t mid = a + (b - a) / 2;
        const int c = cmp(mid);
        if (c < 0) a = mid + 1;
        else b = mid;
        if (c > 0) above = mid;
    }
    lo = a;
    b = above;
    while (a < b) {
        const std::size_t mid = a + (b - a) / 2;
        if (cmp(mid) <= 0) a = mid + 1;
        else b = mid;
    }
    hi = a;
}

std::size_t InfiniGram::match_prefix(const std::uint8_t* s, std::size_t len, std::size_t cap,
                                     std::size_t at_least, std::size_t& pos, std::size_t& count) const {
    // Occurrence is monotone in the prefix length: bisect.
    std::size_t good = std::min(at_least, std::min(len, cap)), bad = std::min(len, cap) + 1;
    std::size_t lo = 0, hi = n_;
    if (good > 0) range(s, good, lo, hi);
    if (lo >= hi) {  // the lower bound was wrong: start from nothing
        good = 0;
        lo = 0;
        hi = n_;
    }
    while (bad - good > 1) {
        const std::size_t mid = good + (bad - good) / 2;
        std::size_t a = 0, b = 0;
        range(s, mid, a, b);
        if (a < b) {
            good = mid;
            lo = a;
            hi = b;
        } else {
            bad = mid;
        }
    }
    pos = lo < hi ? sa(lo) : 0;
    count = hi - lo;
    return good;
}

InfiniGram::Result InfiniGram::query(const std::uint8_t* ctx, std::size_t len, int max_n, int hint,
                                     std::uint64_t min_total) const {
    Result r;
    const int cap = static_cast<int>(std::min<std::size_t>(len, static_cast<std::size_t>(std::max(0, max_n))));
    const std::size_t need = static_cast<std::size_t>(std::max<std::uint64_t>(1, min_total));
    // Longest n in [0, cap] whose suffix occurs followed by some byte at least
    // ``need`` times. That count never grows with n (every occurrence of a
    // longer suffix is one of the shorter), so bisect; ``hint`` caps it (the
    // longest match grows by at most one byte per step).
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
        return hi - lo >= need;
    };
    int good = 0, bad = (hint >= 0 ? std::min(cap, hint) : cap) + 1;
    std::size_t glo = 0, ghi = 0;
    usable(0, glo, ghi);
    if (hint >= 0 && bad > 1) {
        // The bound is usually the answer (the match grew by a byte, or a
        // backoff asks for a length known to occur): one range() instead of
        // a bisection. Same result: the longest usable n in [0, bound].
        std::size_t lo = 0, hi = 0;
        if (usable(bad - 1, lo, hi)) {
            good = bad - 1;
            glo = lo;
            ghi = hi;
        } else {
            bad = bad - 1;
        }
    }
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
