// The overlap check's heavy loop on an NVIDIA GPU (k5_overlap_gpu.hpp; docs/reports/OPT_GPU.md).
//
// Kernel design:
//   - One thread per window start. A block of 256 threads loads its 256 + 31 bytes once into shared memory; each
//     thread hashes its 32 bytes from there (64-bit FNV-1a, byte by byte, as the CPU does) and notes any '\n'.
//   - Look-up: the sorted set gets a directory on the hash's top B bits (B <= 22, about 2-4 hashes a bucket); the
//     thread reads its bucket's two bounds, then binary-searches the few hashes inside. Exact, like std::binary_search.
//   - Each warp writes its 32 answers as one 32-bit word (__ballot_sync). The host reads the bit map and lists the hit
//     positions of each document in order.
//   - Text goes to the device in chunks (two pinned buffers on two streams): the host packs one chunk while the device
//     works on the other.
//   - span_set: the same hash kernel writes every window's hash and a flag; CUB compacts the flagged hashes, sorts them
//     (radix sort) and removes duplicates. If the device has too little memory for the sort, the host sorts the same
//     hashes (std::sort, std::unique): the same set either way.
//
// Author: Odin Loch
#include "k5_overlap_gpu.hpp"

#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>

namespace cypha::cuda_o4 {
namespace {

constexpr int kTB = 256;  // threads per block, one window each
constexpr std::uint64_t kBasis = 1469598103934665603ull, kPrime = 1099511628211ull;
constexpr int kDirBitsMax = 22;

struct OutOfMemory : std::runtime_error {
    using std::runtime_error::runtime_error;
};

void ck(cudaError_t e, const char* what) {
    if (e == cudaErrorMemoryAllocation) {
        cudaGetLastError();
        throw OutOfMemory(std::string("cuda: ") + what + ": out of device memory");
    }
    if (e != cudaSuccess) throw std::runtime_error(std::string("cuda: ") + what + ": " + cudaGetErrorString(e));
}

// The block's windows' bytes in shared memory; bytes past n read as '\n' (no window there).
__device__ __forceinline__ void load_tile(const unsigned char* __restrict__ txt, std::uint64_t n, std::uint64_t base, unsigned char* s) {
    for (int j = static_cast<int>(threadIdx.x); j < kTB + kK - 1; j += kTB) {
        const std::uint64_t q = base + static_cast<std::uint64_t>(j);
        s[j] = q < n ? txt[q] : static_cast<unsigned char>('\n');
    }
}

// FNV-1a of the 32 bytes at s (k5_overlap's fnv); false if one of them is '\n' (the window crosses a line's end).
__device__ __forceinline__ bool window_hash(const unsigned char* s, std::uint64_t& h) {
    std::uint64_t x = kBasis;
    unsigned nl = 0;
#pragma unroll
    for (int k = 0; k < kK; ++k) {
        const unsigned c = s[k];
        nl |= c == '\n';
        x = (x ^ c) * kPrime;
    }
    h = x;
    return nl == 0;
}

// bits[w] bit t: the window starting at 32w + t is a window of one line and its hash is in the set.
__global__ void hits_kernel(const unsigned char* __restrict__ txt, std::uint64_t n, const std::uint64_t* __restrict__ set,
                            const std::uint32_t* __restrict__ dir, int shift, std::uint32_t* __restrict__ bits) {
    __shared__ unsigned char s[kTB + kK];
    const std::uint64_t base = static_cast<std::uint64_t>(blockIdx.x) * kTB;
    load_tile(txt, n, base, s);
    __syncthreads();
    const std::uint64_t i = base + threadIdx.x;
    bool found = false;
    std::uint64_t h = 0;
    if (i + kK <= n && window_hash(s + threadIdx.x, h)) {
        const std::uint32_t b = static_cast<std::uint32_t>(h >> shift);
        std::uint32_t lo = dir[b];
        const std::uint32_t end = dir[b + 1];
        std::uint32_t hi = end;
        while (lo < hi) {
            const std::uint32_t mid = (lo + hi) >> 1;
            if (set[mid] < h) lo = mid + 1;
            else hi = mid;
        }
        found = lo < end && set[lo] == h;
    }
    const unsigned m = __ballot_sync(0xffffffffu, found);
    if ((threadIdx.x & 31u) == 0) bits[i >> 5] = m;
}

// Every window's hash and whether it is a window of one line.
__global__ void span_kernel(const unsigned char* __restrict__ txt, std::uint64_t n, std::uint64_t* __restrict__ hash,
                            unsigned char* __restrict__ flag) {
    __shared__ unsigned char s[kTB + kK];
    const std::uint64_t base = static_cast<std::uint64_t>(blockIdx.x) * kTB;
    load_tile(txt, n, base, s);
    __syncthreads();
    const std::uint64_t i = base + threadIdx.x;
    if (i >= n) return;
    std::uint64_t h = 0;
    const bool ok = i + kK <= n && window_hash(s + threadIdx.x, h);
    hash[i] = h;
    flag[i] = ok ? 1 : 0;
}

double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

}  // namespace

struct Overlap::Impl {
    Stats st;
    std::size_t cap = 0;  // bytes of text per chunk
    std::size_t held = 0;
    bool host_sort = false;

    // Device memory, counted.
    void* dalloc(std::size_t n, const char* what) {
        void* p = nullptr;
        ck(cudaMalloc(&p, std::max<std::size_t>(n, 16)), what);
        held += std::max<std::size_t>(n, 16);
        st.peak_bytes = std::max(st.peak_bytes, held);
        return p;
    }
    void dfree(void*& p, std::size_t n) {
        if (!p) return;
        cudaFree(p);
        held -= std::max<std::size_t>(n, 16);
        p = nullptr;
    }

    // Two chunk slots.
    struct Slot {
        unsigned char* h_txt = nullptr;
        std::uint32_t* h_bits = nullptr;
        unsigned char* d_txt = nullptr;
        std::uint32_t* d_bits = nullptr;
        cudaStream_t s = nullptr;
        cudaEvent_t e[4] = {};
        std::size_t cap = 0, words = 0;
        // pending work: docs [first, last) at offsets
        bool busy = false;
        std::size_t first = 0, last = 0;
        std::vector<std::size_t> off;
    } slot[2];

    // The loaded set.
    std::uint64_t* d_set = nullptr;
    std::uint32_t* d_dir = nullptr;
    std::size_t set_n = 0, dir_n = 0;
    int shift = 63;
    std::uint64_t fp_n = ~0ull, fp_x = 0, fp_y = 0;
    const void* fp_p = nullptr;

    static std::size_t words_for(std::size_t cap) { return (cap / kTB + 2) * (kTB / 32); }

    void ensure_slots(std::size_t need) {
        if (slot[0].cap >= need) return;
        for (auto& sl : slot) {
            if (sl.busy) throw std::logic_error("o4 gpu: slot busy while resizing");
            if (sl.h_txt) cudaFreeHost(sl.h_txt), sl.h_txt = nullptr;
            if (sl.h_bits) cudaFreeHost(sl.h_bits), sl.h_bits = nullptr;
            void* p = sl.d_txt;
            dfree(p, sl.cap);
            p = sl.d_bits;
            dfree(p, sl.words * 4);
            sl.d_txt = nullptr, sl.d_bits = nullptr;
            sl.cap = need, sl.words = words_for(need);
            ck(cudaMallocHost(reinterpret_cast<void**>(&sl.h_txt), sl.cap), "cudaMallocHost text");
            ck(cudaMallocHost(reinterpret_cast<void**>(&sl.h_bits), sl.words * 4), "cudaMallocHost bits");
            sl.d_txt = static_cast<unsigned char*>(dalloc(sl.cap, "chunk text"));
            sl.d_bits = static_cast<std::uint32_t*>(dalloc(sl.words * 4, "chunk bits"));
        }
    }

    void load_set(const std::vector<std::uint64_t>& set) {
        std::uint64_t x = 0, y = 0;  // two checksums: the xor, and a sum of mixed values (order-sensitive)
        for (const auto v : set) x ^= v, y = y * 0x9E3779B97F4A7C15ull + (v ^ (v >> 29));
        if (fp_n == set.size() && fp_x == x && fp_y == y && fp_p == set.data()) return;
        if (set.size() >= 0xFFFFFFFFull) throw std::runtime_error("o4 gpu: set too large for 32-bit indices");
        void* p = d_set;
        dfree(p, set_n * 8);
        p = d_dir;
        dfree(p, dir_n * 4);
        d_set = nullptr, d_dir = nullptr;
        set_n = set.size();
        int B = 1;
        while (B < kDirBitsMax && (std::size_t{1} << (B + 2)) <= set_n) ++B;  // about 2-4 hashes a bucket
        shift = 64 - B;
        dir_n = (std::size_t{1} << B) + 1;
        std::vector<std::uint32_t> dir(dir_n);
        std::size_t j = 0;
        for (std::size_t b = 0; b + 1 < dir_n; ++b) {
            while (j < set_n && (set[j] >> shift) < b) ++j;
            dir[b] = static_cast<std::uint32_t>(j);
        }
        dir[dir_n - 1] = static_cast<std::uint32_t>(set_n);
        d_set = static_cast<std::uint64_t*>(dalloc(set_n * 8, "set"));
        d_dir = static_cast<std::uint32_t*>(dalloc(dir_n * 4, "directory"));
        const double t0 = now_ms();
        if (set_n) ck(cudaMemcpy(d_set, set.data(), set_n * 8, cudaMemcpyHostToDevice), "copy set");
        ck(cudaMemcpy(d_dir, dir.data(), dir_n * 4, cudaMemcpyHostToDevice), "copy directory");
        st.copy_ms += now_ms() - t0;
        fp_n = set.size(), fp_x = x, fp_y = y, fp_p = set.data();
        ++st.set_uploads;
    }

    void launch(Slot& sl, std::size_t n) {
        ck(cudaEventRecord(sl.e[0], sl.s), "event");
        ck(cudaMemcpyAsync(sl.d_txt, sl.h_txt, n, cudaMemcpyHostToDevice, sl.s), "copy text");
        ck(cudaEventRecord(sl.e[1], sl.s), "event");
        const unsigned blocks = static_cast<unsigned>((n + kTB - 1) / kTB);
        hits_kernel<<<blocks, kTB, 0, sl.s>>>(sl.d_txt, n, d_set, d_dir, shift, sl.d_bits);
        ck(cudaGetLastError(), "hits kernel");
        ck(cudaEventRecord(sl.e[2], sl.s), "event");
        ck(cudaMemcpyAsync(sl.h_bits, sl.d_bits, static_cast<std::size_t>(blocks) * (kTB / 32) * 4, cudaMemcpyDeviceToHost, sl.s), "copy bits");
        ck(cudaEventRecord(sl.e[3], sl.s), "event");
        st.windows += n;
        sl.busy = true;
    }

    void finish(Slot& sl, const std::vector<std::string_view>& docs, std::vector<std::vector<std::size_t>>& out) {
        if (!sl.busy) return;
        ck(cudaStreamSynchronize(sl.s), "sync");
        float a = 0, b = 0, c = 0;
        cudaEventElapsedTime(&a, sl.e[0], sl.e[1]);
        cudaEventElapsedTime(&b, sl.e[1], sl.e[2]);
        cudaEventElapsedTime(&c, sl.e[2], sl.e[3]);
        st.copy_ms += a + c, st.kernel_ms += b;
        const std::uint32_t* bits = sl.h_bits;
        for (std::size_t d = sl.first; d < sl.last; ++d) {
            auto& v = out[d];
            v.clear();
            const std::size_t sz = docs[d].size();
            if (sz < static_cast<std::size_t>(kK)) continue;
            const std::size_t o = sl.off[d - sl.first], lo = o, hi = o + sz - kK + 1;  // window starts [lo, hi)
            for (std::size_t w = lo >> 5; w <= (hi - 1) >> 5; ++w) {
                std::uint32_t m = bits[w];
                while (m) {
                    const std::size_t pos = (w << 5) + static_cast<std::size_t>(__builtin_ctz(m));
                    m &= m - 1;
                    if (pos >= lo && pos < hi) v.push_back(pos - o);
                }
            }
        }
        sl.busy = false;
    }
};

Overlap::Overlap(std::size_t chunk_mb, bool host_sort) : p_(std::make_unique<Impl>()) {
    p_->host_sort = host_sort;
    int n = 0;
    ck(cudaGetDeviceCount(&n), "cudaGetDeviceCount");
    if (n < 1) throw std::runtime_error("cuda: no device");
    ck(cudaSetDevice(0), "cudaSetDevice");
    cudaDeviceProp pr{};
    ck(cudaGetDeviceProperties(&pr, 0), "cudaGetDeviceProperties");
    p_->st.device = pr.name;
    std::size_t fr = 0, tot = 0;
    ck(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo");
    p_->st.free_at_start = fr;
    // Two chunks of text and their bit maps; on a crowded device (other jobs share the GPU), smaller chunks.
    p_->cap = std::min(std::max<std::size_t>(chunk_mb, 1) << 20, std::max<std::size_t>(fr / 16, std::size_t{4} << 20));
    for (auto& sl : p_->slot) {
        ck(cudaStreamCreateWithFlags(&sl.s, cudaStreamNonBlocking), "stream");
        for (auto& e : sl.e) ck(cudaEventCreate(&e), "event");
    }
}

Overlap::~Overlap() {
    if (!p_) return;
    for (auto& sl : p_->slot) {
        if (sl.s) cudaStreamSynchronize(sl.s);
        if (sl.h_txt) cudaFreeHost(sl.h_txt);
        if (sl.h_bits) cudaFreeHost(sl.h_bits);
        if (sl.d_txt) cudaFree(sl.d_txt);
        if (sl.d_bits) cudaFree(sl.d_bits);
        for (auto& e : sl.e)
            if (e) cudaEventDestroy(e);
        if (sl.s) cudaStreamDestroy(sl.s);
    }
    if (p_->d_set) cudaFree(p_->d_set);
    if (p_->d_dir) cudaFree(p_->d_dir);
}

const Stats& Overlap::stats() const { return p_->st; }

void Overlap::hits(const std::vector<std::string_view>& docs, const std::vector<std::uint64_t>& set, std::vector<std::vector<std::size_t>>& out) {
    Impl& m = *p_;
    out.assign(docs.size(), {});
    if (docs.empty()) return;
    m.load_set(set);
    std::size_t big = 0;
    for (const auto& d : docs) big = std::max(big, d.size());
    m.ensure_slots(std::max(m.cap, big + 1));
    std::size_t i = 0;
    int k = 0;
    while (i < docs.size()) {
        Impl::Slot& sl = m.slot[k];
        m.finish(sl, docs, out);  // the slot's previous chunk, while the other slot's runs
        sl.first = i;
        sl.off.clear();
        std::size_t o = 0;
        while (i < docs.size() && o + docs[i].size() + 1 <= sl.cap) {
            sl.off.push_back(o);
            if (!docs[i].empty()) std::memcpy(sl.h_txt + o, docs[i].data(), docs[i].size());
            o += docs[i].size();
            sl.h_txt[o++] = '\n';
            ++i;
        }
        sl.last = i;
        m.launch(sl, o);
        k ^= 1;
    }
    m.finish(m.slot[k], docs, out);
    m.finish(m.slot[k ^ 1], docs, out);
}

std::vector<std::uint64_t> Overlap::span_set(const std::vector<std::string_view>& texts) {
    Impl& m = *p_;
    // Windows (with repeats) and the packed text, chunk by chunk.
    std::size_t W = 0, big = 0;
    for (const auto& t : texts) {
        if (t.size() >= static_cast<std::size_t>(kK)) W += t.size() - kK + 1;
        big = std::max(big, t.size());
    }
    std::vector<std::uint64_t> res;
    if (W == 0) return res;
    if (W >= static_cast<std::size_t>(INT_MAX)) throw std::runtime_error("o4 gpu: too many windows for one set");
    // Text hashed at a time: at most 8 MB (64 MB of hashes), or one whole text if longer.
    const std::size_t scap = std::max<std::size_t>(std::min<std::size_t>(m.cap, std::size_t{8} << 20), big + 1);
    m.ensure_slots(std::max(m.cap, scap));
    Impl::Slot& sl = m.slot[0];
    std::uint64_t* d_keys = static_cast<std::uint64_t*>(m.dalloc(W * 8, "span keys"));
    std::uint64_t* d_hash = static_cast<std::uint64_t*>(m.dalloc(scap * 8, "span hashes"));
    unsigned char* d_flag = static_cast<unsigned char*>(m.dalloc(scap, "span flags"));
    int* d_num = static_cast<int*>(m.dalloc(sizeof(int), "span count"));
    void* d_tmp = nullptr;
    std::size_t tmp_n = 0;
    std::size_t got = 0;
    cudaEvent_t e0 = sl.e[0], e1 = sl.e[1];
    float ms = 0;
    for (std::size_t i = 0; i < texts.size();) {
        std::size_t o = 0;
        while (i < texts.size() && o + texts[i].size() + 1 <= scap) {
            if (!texts[i].empty()) std::memcpy(sl.h_txt + o, texts[i].data(), texts[i].size());
            o += texts[i].size();
            sl.h_txt[o++] = '\n';
            ++i;
        }
        const double t0 = now_ms();
        ck(cudaMemcpyAsync(sl.d_txt, sl.h_txt, o, cudaMemcpyHostToDevice, sl.s), "copy text");
        ck(cudaStreamSynchronize(sl.s), "sync");
        m.st.copy_ms += now_ms() - t0;
        ck(cudaEventRecord(e0, sl.s), "event");
        span_kernel<<<static_cast<unsigned>((o + kTB - 1) / kTB), kTB, 0, sl.s>>>(sl.d_txt, o, d_hash, d_flag);
        ck(cudaGetLastError(), "span kernel");
        std::size_t need = 0;
        ck(cub::DeviceSelect::Flagged(nullptr, need, d_hash, d_flag, d_keys + got, d_num, static_cast<int>(o), sl.s), "select size");
        if (need > tmp_n) {
            m.dfree(d_tmp, tmp_n);
            d_tmp = m.dalloc(need, "select temp");
            tmp_n = need;
        }
        ck(cub::DeviceSelect::Flagged(d_tmp, need, d_hash, d_flag, d_keys + got, d_num, static_cast<int>(o), sl.s), "select");
        ck(cudaEventRecord(e1, sl.s), "event");
        int nsel = 0;
        ck(cudaMemcpyAsync(&nsel, d_num, sizeof(int), cudaMemcpyDeviceToHost, sl.s), "copy count");
        ck(cudaStreamSynchronize(sl.s), "sync");
        cudaEventElapsedTime(&ms, e0, e1);
        m.st.kernel_ms += ms;
        m.st.windows += o;
        got += static_cast<std::size_t>(nsel);
    }
    if (got != W) throw std::logic_error("o4 gpu: span window count differs from the host's");
    void* p = d_hash;
    m.dfree(p, scap * 8);
    p = d_flag;
    m.dfree(p, scap);
    // Sort and unique on the device; with too little memory, on the host.
    bool on_device = !m.host_sort;
    std::uint64_t* d_alt = nullptr;
    if (on_device) try {
        d_alt = static_cast<std::uint64_t*>(m.dalloc(W * 8, "span sort buffer"));
        cub::DoubleBuffer<std::uint64_t> kb(d_keys, d_alt);
        std::size_t need = 0;
        ck(cub::DeviceRadixSort::SortKeys(nullptr, need, kb, static_cast<int>(W), 0, 64, sl.s), "sort size");
        std::size_t need2 = 0;
        ck(cub::DeviceSelect::Unique(nullptr, need2, kb.Current(), kb.Alternate(), d_num, static_cast<int>(W), sl.s), "unique size");
        need = std::max(need, need2);
        if (need > tmp_n) {
            m.dfree(d_tmp, tmp_n);
            d_tmp = m.dalloc(need, "sort temp");
            tmp_n = need;
        }
        ck(cudaEventRecord(e0, sl.s), "event");
        ck(cub::DeviceRadixSort::SortKeys(d_tmp, need, kb, static_cast<int>(W), 0, 64, sl.s), "sort");
        ck(cub::DeviceSelect::Unique(d_tmp, need, kb.Current(), kb.Alternate(), d_num, static_cast<int>(W), sl.s), "unique");
        ck(cudaEventRecord(e1, sl.s), "event");
        int nu = 0;
        ck(cudaMemcpyAsync(&nu, d_num, sizeof(int), cudaMemcpyDeviceToHost, sl.s), "copy count");
        ck(cudaStreamSynchronize(sl.s), "sync");
        cudaEventElapsedTime(&ms, e0, e1);
        m.st.kernel_ms += ms;
        res.resize(static_cast<std::size_t>(nu));
        const double t0 = now_ms();
        ck(cudaMemcpy(res.data(), kb.Alternate(), res.size() * 8, cudaMemcpyDeviceToHost), "copy set");
        m.st.copy_ms += now_ms() - t0;
    } catch (const OutOfMemory&) {
        on_device = false;
    }
    if (!on_device) {
        res.resize(W);
        const double t0 = now_ms();
        ck(cudaMemcpy(res.data(), d_keys, W * 8, cudaMemcpyDeviceToHost), "copy keys");
        m.st.copy_ms += now_ms() - t0;
        std::sort(res.begin(), res.end());
        res.erase(std::unique(res.begin(), res.end()), res.end());
        ++m.st.host_sorts;
    }
    p = d_alt;
    m.dfree(p, W * 8);
    p = d_keys;
    m.dfree(p, W * 8);
    m.dfree(d_tmp, tmp_n);
    p = d_num;
    m.dfree(p, sizeof(int));
    return res;
}

}  // namespace cypha::cuda_o4
