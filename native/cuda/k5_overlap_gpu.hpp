// The overlap check's heavy loop on an NVIDIA GPU (docs/reports/OPT_GPU.md, GENERATION_PLAN §13.41 O4).
//
// scripts/k5_overlap.cpp hashes every 32-byte window of a text (64-bit FNV-1a) and looks each hash up in a sorted set
// of forbidden spans. Every window is independent, so one GPU thread takes one window. The maths is the same as on the
// CPU: the same hash, an exact look-up in the same sorted set. Results are integers (positions, counts), so they are
// identical, not close.
//
//   Overlap g;                         // device 0; throws std::runtime_error if there is none
//   auto set = g.span_set(texts);      // sorted unique hashes of every window of every text (= k5_overlap's span_set)
//   g.hits(docs, set, out);            // out[i] = start positions of doc i's windows whose hash is in set (= hit_pos)
//
// Texts and docs must not hold '\n' (they are lines): the device packs them with '\n' between them, and a window that
// holds a '\n' is not a window of one text.
//
// Author: Odin Loch
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace cypha::cuda_o4 {

inline constexpr int kK = 32;  // window bytes (k5_overlap's K)

struct Stats {
    std::string device;            // name of the GPU
    std::size_t free_at_start = 0;  // bytes free on the device when Overlap was made
    std::size_t peak_bytes = 0;     // the most device memory this object held at once
    double kernel_ms = 0;           // time inside kernels and device sorts
    double copy_ms = 0;             // host <-> device copies
    std::size_t windows = 0;        // windows hashed
    std::size_t set_uploads = 0;    // how often a set was sent to the device
    std::size_t host_sorts = 0;     // span sets sorted on the host (too little device memory, or host_sort)
};

class Overlap {
 public:
    // chunk_mb: bytes of text sent to the device at a time (less on a crowded device). host_sort: sort span sets on the
    // host even when the device has room (the fallback's path; for the selftest).
    explicit Overlap(std::size_t chunk_mb = 64, bool host_sort = false);
    ~Overlap();
    Overlap(const Overlap&) = delete;
    Overlap& operator=(const Overlap&) = delete;

    // Sorted, unique hashes of every 32-byte window of every text.
    std::vector<std::uint64_t> span_set(const std::vector<std::string_view>& texts);
    // For each doc, the start positions (ascending) of its windows whose hash is in set (sorted, unique).
    void hits(const std::vector<std::string_view>& docs, const std::vector<std::uint64_t>& set, std::vector<std::vector<std::size_t>>& out);

    const Stats& stats() const;

 private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace cypha::cuda_o4
