#pragma once
//
// hp/hash_table.hpp — demand-zero hash tables for context/match slots.
//
// Storage comes straight from the OS (mmap; calloc on Windows) and is never
// written at construction, so pages cost RSS only once a slot is touched and
// a 1.5 GB predictor constructs in milliseconds. Untouched pages read as
// zero, which is bit-identical to std::vector<T>(n, 0). Linux also gets
// MADV_HUGEPAGE to cut TLB misses on the random slot probes.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <istream>
#include <ostream>
#include <type_traits>
#include <utility>
#include <vector>
#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "hp/blob_io.hpp"

namespace hp {

/// Source file for mapping tables on checkpoint load (see MapScope). While a
/// scope is active on this thread, ZeroBuf::read maps large tables straight
/// from the file (MAP_PRIVATE: copy-on-write) instead of copying them into
/// anonymous memory. Pages the model never writes stay clean and file-backed:
/// shared between processes and reclaimable by the kernel.
struct MapSource {
    int fd = -1;
    std::size_t min_bytes = 1 << 16;  // smaller tables are read normally
};

class MapScope {
 public:
    explicit MapScope(const MapSource* src) : prev_(active_) { active_ = src; }
    ~MapScope() { active_ = prev_; }
    MapScope(const MapScope&) = delete;
    MapScope& operator=(const MapScope&) = delete;
    static const MapSource* active() { return active_; }

 private:
    const MapSource* prev_;
    static inline thread_local const MapSource* active_ = nullptr;
};

// FNV-1a over raw bytes; Predictor::learned_digest() folds learned state with it.
inline std::uint64_t fnv_bytes(std::uint64_t h, const void* p, std::size_t n) {
    const auto* b = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 0x100000001B3ull;
    return h;
}

// Fixed-size, zero-initialised array backed by demand-zero pages.
// Copyable (deep copy) so Predictor::copy_state_from keeps working.
template <typename T>
class ZeroBuf {
    static_assert(std::is_trivially_copyable<T>::value, "ZeroBuf T must be trivially copyable");

 public:
    ZeroBuf() = default;
    explicit ZeroBuf(std::size_t n) : n_(n), p_(alloc_(n)) {}

    ~ZeroBuf();
    ZeroBuf(const ZeroBuf& o) : n_(o.n_), p_(alloc_(o.n_)) { copy_(o); }
    ZeroBuf& operator=(const ZeroBuf& o) {
        if (this != &o) {
            if (n_ != o.n_ || map_base_ != nullptr) {
                release_();
                n_ = o.n_;
                p_ = alloc_(n_);
            }
            copy_(o);
        }
        return *this;
    }
    ZeroBuf(ZeroBuf&& o) noexcept
        : n_(std::exchange(o.n_, 0)), p_(std::exchange(o.p_, nullptr)),
          map_base_(std::exchange(o.map_base_, nullptr)), map_len_(std::exchange(o.map_len_, 0)) {}
    ZeroBuf& operator=(ZeroBuf&& o) noexcept {
        std::swap(n_, o.n_);
        std::swap(p_, o.p_);
        std::swap(map_base_, o.map_base_);
        std::swap(map_len_, o.map_len_);
        return *this;
    }
    /// True when the storage is a copy-on-write mapping of a checkpoint file.
    bool file_mapped() const { return map_base_ != nullptr; }

    std::size_t size() const { return n_; }
    T* data() { return p_; }
    const T* data() const { return p_; }
    T& operator[](std::size_t i) { return p_[i]; }
    const T& operator[](std::size_t i) const { return p_[i]; }

    // Same byte format as blob::write_vec / read_vec on std::vector<T>.
    void write(std::ostream& os) const {
        const std::uint64_t n = n_;
        blob::write_pod(os, n);
        if (n_ > 0) os.write(reinterpret_cast<const char*>(p_), static_cast<std::streamsize>(n_ * sizeof(T)));
    }
    void read(std::istream& is) {
        std::uint64_t n = 0;
        blob::read_pod(is, n);
        const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(T);
#if !defined(_WIN32)
        const MapSource* src = MapScope::active();
        if (src != nullptr && src->fd >= 0 && bytes >= src->min_bytes) {
            const std::streamoff off = is.tellg();
            if (off >= 0) {
                const long page = sysconf(_SC_PAGESIZE);
                const std::size_t start = static_cast<std::size_t>(off);
                const std::size_t aligned = start - start % static_cast<std::size_t>(page);
                const std::size_t len = bytes + (start - aligned);
                void* base = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE, src->fd,
                                  static_cast<off_t>(aligned));
                if (base != MAP_FAILED) {
                    release_();
                    n_ = static_cast<std::size_t>(n);
                    map_base_ = base;
                    map_len_ = len;
                    p_ = reinterpret_cast<T*>(static_cast<char*>(base) + (start - aligned));
                    is.seekg(static_cast<std::streamoff>(start + bytes));
                    return;
                }
            }
        }
#endif
        if (static_cast<std::size_t>(n) != n_ || map_base_ != nullptr) *this = ZeroBuf(static_cast<std::size_t>(n));
        if (n_ > 0) is.read(reinterpret_cast<char*>(p_), static_cast<std::streamsize>(bytes));
    }

 private:
    void release_() {
#if !defined(_WIN32)
        if (map_base_ != nullptr) {
            munmap(map_base_, map_len_);
            map_base_ = nullptr;
            map_len_ = 0;
            p_ = nullptr;
            return;
        }
#endif
        free_(p_, n_);
        p_ = nullptr;
    }
    void copy_(const ZeroBuf& o) {
        if (n_ > 0) std::memcpy(p_, o.p_, n_ * sizeof(T));
    }
    static T* alloc_(std::size_t n) {
        if (n == 0) return nullptr;
        const std::size_t bytes = n * sizeof(T);
#if defined(_WIN32)
        // calloc keeps <windows.h> out of a header that Cypha includes widely.
        void* p = std::calloc(n, sizeof(T));
        if (p == nullptr) std::abort();
#else
        void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) std::abort();
#if defined(MADV_HUGEPAGE)
        if (bytes >= (std::size_t{2} << 20)) madvise(p, bytes, MADV_HUGEPAGE);
#endif
#endif
        return static_cast<T*>(p);
    }
    static void free_(T* p, std::size_t n) {
        if (p == nullptr) return;
#if defined(_WIN32)
        (void)n;
        std::free(p);
#else
        munmap(p, n * sizeof(T));
#endif
    }

    std::size_t n_ = 0;
    T* p_ = nullptr;
    void* map_base_ = nullptr;  // file mapping (MapScope), else anonymous
    std::size_t map_len_ = 0;
};

template <typename T>
ZeroBuf<T>::~ZeroBuf() {
    release_();
}

template <typename T>
class HashTable {
 public:
    explicit HashTable(int table_bits)
        : mask_((1u << table_bits) - 1),
          tab_(static_cast<std::size_t>(1) << table_bits) {}

    T get(std::uint32_t idx) const { return tab_[idx & mask_]; }
    T& ref(std::uint32_t idx) { return tab_[idx & mask_]; }
    T& at(std::size_t i) { return tab_[i]; }
    const T& at(std::size_t i) const { return tab_[i]; }
    std::size_t size() const { return tab_.size(); }
    const T* data() const { return tab_.data(); }
    T* data() { return tab_.data(); }

    /// Reallocate at ``bits`` (zeroed); callers refill (table folding).
    void resize_bits(int bits) {
        mask_ = (1u << bits) - 1;
        tab_ = ZeroBuf<T>(static_cast<std::size_t>(1) << bits);
    }

    void checkpoint_write(std::ostream& os) const {
        blob::write_pod(os, mask_);
        tab_.write(os);
    }

    void checkpoint_read(std::istream& is) {
        blob::read_pod(is, mask_);
        tab_.read(is);
    }

 private:
    std::uint32_t mask_;
    ZeroBuf<T> tab_;
};

}  // namespace hp
