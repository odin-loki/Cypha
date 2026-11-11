// A byte LSTM learning online (Odin Loch, 2026-11-04; docs/GENERATION_PLAN.md §13.18). Embedding 16, hidden 64, a
// softmax over the 256 next bytes; truncated BPTT over 20 bytes and Adam (step 0.003). It predicts each byte from the
// state that has read every byte before it, then learns it. Used inside the bitwise GLN (bitwise_gln.hpp) and on its
// own as a content-mixer expert (cyphalm_trie --lstm), as PAQ8 and cmix use an LSTM.
// Rule 10 (PLAN): an expert inside CyphaLM. It answers nothing.
#pragma once

#include <algorithm>
#include <thread>
#include <cstdlib>
#include <chrono>
#include <atomic>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <new>
#include <random>
#include <string>
#include <vector>

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define CYPHA_LSTM_SSE2 1
#else
#define CYPHA_LSTM_SSE2 0
#endif
// OPT_SIMD (GENERATION_PLAN §13.41 O5): AVX-512 kernels, compiled for that target only (GCC and Clang on x86-64) and
// chosen at run time when the CPU has AVX-512F. Elsewhere the SSE2 and scalar code below runs, with the same numbers.
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define CYPHA_LSTM_AVX512 1
#define CYPHA_LSTM_T512 __attribute__((target("avx512f")))
#else
#define CYPHA_LSTM_AVX512 0
#endif
#if defined(_MSC_VER)
#define CYPHA_RESTRICT __restrict
#else
#define CYPHA_RESTRICT __restrict__
#endif

namespace cypha::online {

// Binary save and load of vectors and plain values (the GLN's file, "GLN1": a learned model can be trained once, at
// scale, and read by every run, §13.17).
namespace gln_io {
template <class T>
void put(std::ostream& o, const T& v) { o.write(reinterpret_cast<const char*>(&v), sizeof v); }
template <class T>
void get(std::istream& i, T& v) { i.read(reinterpret_cast<char*>(&v), sizeof v); }
template <class T, class A>
void put_vec(std::ostream& o, const std::vector<T, A>& v) {
    put(o, static_cast<std::uint64_t>(v.size()));
    o.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
}
template <class T, class A>
void get_vec(std::istream& i, std::vector<T, A>& v) {
    std::uint64_t n = 0;
    get(i, n);
    v.resize(static_cast<std::size_t>(n));
    i.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
}
}  // namespace gln_io

inline double gln_squash(double x) { return 1.0 / (1.0 + std::exp(-std::clamp(x, -30.0, 30.0))); }
inline double gln_stretch(double p) {
    p = std::clamp(p, 1e-4, 1.0 - 1e-4);
    return std::log(p / (1.0 - p));
}

// A small pool of spinning worker threads for ByteLSTMT (ONLINE_LEARNING_UPGRADES.md, "Optimisation", 2026-11-05).
// run(f) calls f(t, n) once for each part t < n, part 0 on the calling thread, and returns when all are done. Workers spin,
// then yield, then sleep briefly when idle for long, so an idle pool costs little. barrier() (OPT_SIMD) holds every part
// of the running job until all have reached it.
class LstmPool {
public:
    explicit LstmPool(int n) : n_(n) {
        for (int t = 1; t < n; ++t) th_.emplace_back([this, t] { work(t); });
    }
    ~LstmPool() {
        stop_.store(true, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);
        for (auto& t : th_) t.join();
    }
    LstmPool(const LstmPool&) = delete;
    LstmPool& operator=(const LstmPool&) = delete;
    int size() const { return n_; }
    template <class F>
    void run(F& f) {
        job_ = &f;
        call_ = [](void* c, int t, int n) { (*static_cast<F*>(c))(t, n); };
        done_.store(0, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);
        f(0, n_);
        int spins = 0;
        while (done_.load(std::memory_order_acquire) != n_ - 1)
            if (++spins > 4096) std::this_thread::yield();
    }
    // Inside a job: every part waits here until all n have arrived (a sense-counting spin barrier).
    void barrier() {
        const unsigned g = bgen_.load(std::memory_order_acquire);
        if (bcnt_.fetch_add(1, std::memory_order_acq_rel) == n_ - 1) {
            bcnt_.store(0, std::memory_order_relaxed);
            bgen_.store(g + 1, std::memory_order_release);
        } else {
            int spins = 0;
            while (bgen_.load(std::memory_order_acquire) == g)
                if (++spins > 4096) std::this_thread::yield();
        }
    }

private:
    void work(int t) {
        unsigned seen = 0;
        for (;;) {
            unsigned g;
            long spins = 0;
            while ((g = gen_.load(std::memory_order_acquire)) == seen) {
                if (++spins > 200000) std::this_thread::sleep_for(std::chrono::microseconds(50));
                else if (spins > 4096) std::this_thread::yield();
            }
            seen = g;
            if (stop_.load(std::memory_order_relaxed)) return;
            call_(job_, t, n_);
            done_.fetch_add(1, std::memory_order_release);
        }
    }
    int n_;
    std::vector<std::thread> th_;
    std::atomic<unsigned> gen_{0};
    std::atomic<int> done_{0};
    std::atomic<bool> stop_{false};
    alignas(64) std::atomic<unsigned> bgen_{0};
    alignas(64) std::atomic<int> bcnt_{0};
    void* job_ = nullptr;
    void (*call_)(void*, int, int) = nullptr;
};

// out[r] = bias[r] + sum_j W[r][j] x[j] for rows r in [r0, r1) of a row-major W (len columns), each row summed in order
// j = 0, 1, ...: with SSE2, two rows share a register (one lane each), eight rows at a time, so every row's sum is the
// scalar loop's, bit for bit (no FMA, no reassociation).
inline void lstm_rows(const double* W, int len, const double* x, const double* bias, double* out, int r0, int r1) {
    int r = r0;
#if CYPHA_LSTM_SSE2
    if ((len & 1) == 0) {
        for (; r + 8 <= r1; r += 8) {
            const double* w = W + static_cast<std::size_t>(r) * len;
            __m128d a0 = _mm_set_pd(bias[r + 1], bias[r]), a1 = _mm_set_pd(bias[r + 3], bias[r + 2]),
                    a2 = _mm_set_pd(bias[r + 5], bias[r + 4]), a3 = _mm_set_pd(bias[r + 7], bias[r + 6]);
            for (int j = 0; j < len; j += 2) {
                const __m128d x0 = _mm_set1_pd(x[j]), x1 = _mm_set1_pd(x[j + 1]);
#define CYPHA_LSTM_PAIR(A, R)                                                                    \
    {                                                                                            \
        const __m128d u = _mm_loadu_pd(w + static_cast<std::size_t>(R) * len + j);               \
        const __m128d v = _mm_loadu_pd(w + static_cast<std::size_t>(R + 1) * len + j);           \
        A = _mm_add_pd(A, _mm_mul_pd(_mm_unpacklo_pd(u, v), x0));                                \
        A = _mm_add_pd(A, _mm_mul_pd(_mm_unpackhi_pd(u, v), x1));                                \
    }
                CYPHA_LSTM_PAIR(a0, 0)
                CYPHA_LSTM_PAIR(a1, 2)
                CYPHA_LSTM_PAIR(a2, 4)
                CYPHA_LSTM_PAIR(a3, 6)
#undef CYPHA_LSTM_PAIR
            }
            _mm_storeu_pd(out + r, a0);
            _mm_storeu_pd(out + r + 2, a1);
            _mm_storeu_pd(out + r + 4, a2);
            _mm_storeu_pd(out + r + 6, a3);
        }
    }
#endif
    for (; r < r1; ++r) {
        double a = bias[r];
        const double* w = W + static_cast<std::size_t>(r) * len;
        for (int j = 0; j < len; ++j) a += w[j] * x[j];
        out[r] = a;
    }
}

// Adam over len elements from the scalar code (the old loop): each element's arithmetic as std::clamp and the formula
// below. Dividing by c1 or c2 when it is exactly 1 gives the dividend itself, so that division is skipped (bit-identical).
inline void lstm_adam_scalar(double* CYPHA_RESTRICT w, double* CYPHA_RESTRICT m, double* CYPHA_RESTRICT v,
                             const double* CYPHA_RESTRICT gs, std::size_t len, double n, double c1, double c2, double lr) {
    for (std::size_t j = 0; j < len; ++j) {
        const double x = gs[j] / n;
        const double g = (x < -5.0) ? -5.0 : (5.0 < x) ? 5.0 : x;  // std::clamp(x, -5, 5)
        m[j] = 0.9 * m[j] + 0.1 * g;
        v[j] = 0.999 * v[j] + 0.001 * g * g;
        const double mh = c1 != 1.0 ? m[j] / c1 : m[j], vh = c2 != 1.0 ? v[j] / c2 : v[j];
        w[j] -= lr * mh / (std::sqrt(vh) + 1e-8);
    }
}

#if CYPHA_LSTM_AVX512
// The AVX-512 kernels (OPT_SIMD). Every lane holds one output's running sum and adds its terms in the scalar loop's
// order. A product goes through an empty asm, so the compiler cannot fuse it with the add (no FMA, whatever the flags):
// each product and each sum is rounded on its own, as in the scalar code. Division and square root are IEEE in every
// width. So each output is the scalar loop's, bit for bit. Only the scalar callers do the math outside these kernels.
CYPHA_LSTM_T512 inline __m512d lstm_mul8(__m512d a, __m512d b) {
    __m512d p = _mm512_mul_pd(a, b);
    __asm__("" : "+v"(p));
    return p;
}
CYPHA_LSTM_T512 inline __m512d lstm_madd8(__m512d acc, __m512d a, __m512d b) { return _mm512_add_pd(acc, lstm_mul8(a, b)); }

// Lanes across rows, in tiles of 4 x 8 rows: tile t's weights are Wb[(t * len + j) * 32 + q * 8 + l] (column j, block q,
// lane l), and out[q * qs + t * ts + l] = bias[same] + sum_j W(row) x[j], j = 0, 1, ... Two tiles at a time.
CYPHA_LSTM_T512 inline void lstm_tiles512(const double* Wb, int len, const double* x, const double* bias, double* out, int qs,
                                          int ts, int t0, int t1) {
    int t = t0;
    for (; t + 2 <= t1; t += 2) {
        const double* w0 = Wb + static_cast<std::size_t>(t) * len * 32;
        const double* w1 = w0 + static_cast<std::size_t>(len) * 32;
        const int o0 = t * ts, o1 = (t + 1) * ts;
        __m512d a0 = _mm512_loadu_pd(bias + o0), a1 = _mm512_loadu_pd(bias + qs + o0), a2 = _mm512_loadu_pd(bias + 2 * qs + o0),
                a3 = _mm512_loadu_pd(bias + 3 * qs + o0);
        __m512d b0 = _mm512_loadu_pd(bias + o1), b1 = _mm512_loadu_pd(bias + qs + o1), b2 = _mm512_loadu_pd(bias + 2 * qs + o1),
                b3 = _mm512_loadu_pd(bias + 3 * qs + o1);
        for (int j = 0; j < len; ++j) {
            const __m512d xj = _mm512_set1_pd(x[j]);
            const double* u = w0 + static_cast<std::size_t>(j) * 32;
            const double* v = w1 + static_cast<std::size_t>(j) * 32;
            a0 = lstm_madd8(a0, _mm512_load_pd(u), xj);
            a1 = lstm_madd8(a1, _mm512_load_pd(u + 8), xj);
            a2 = lstm_madd8(a2, _mm512_load_pd(u + 16), xj);
            a3 = lstm_madd8(a3, _mm512_load_pd(u + 24), xj);
            b0 = lstm_madd8(b0, _mm512_load_pd(v), xj);
            b1 = lstm_madd8(b1, _mm512_load_pd(v + 8), xj);
            b2 = lstm_madd8(b2, _mm512_load_pd(v + 16), xj);
            b3 = lstm_madd8(b3, _mm512_load_pd(v + 24), xj);
        }
        _mm512_storeu_pd(out + o0, a0);
        _mm512_storeu_pd(out + qs + o0, a1);
        _mm512_storeu_pd(out + 2 * qs + o0, a2);
        _mm512_storeu_pd(out + 3 * qs + o0, a3);
        _mm512_storeu_pd(out + o1, b0);
        _mm512_storeu_pd(out + qs + o1, b1);
        _mm512_storeu_pd(out + 2 * qs + o1, b2);
        _mm512_storeu_pd(out + 3 * qs + o1, b3);
    }
    if (t < t1) {
        const double* w0 = Wb + static_cast<std::size_t>(t) * len * 32;
        const int o0 = t * ts;
        __m512d a0 = _mm512_loadu_pd(bias + o0), a1 = _mm512_loadu_pd(bias + qs + o0), a2 = _mm512_loadu_pd(bias + 2 * qs + o0),
                a3 = _mm512_loadu_pd(bias + 3 * qs + o0);
        for (int j = 0; j < len; ++j) {
            const __m512d xj = _mm512_set1_pd(x[j]);
            const double* u = w0 + static_cast<std::size_t>(j) * 32;
            a0 = lstm_madd8(a0, _mm512_load_pd(u), xj);
            a1 = lstm_madd8(a1, _mm512_load_pd(u + 8), xj);
            a2 = lstm_madd8(a2, _mm512_load_pd(u + 16), xj);
            a3 = lstm_madd8(a3, _mm512_load_pd(u + 24), xj);
        }
        _mm512_storeu_pd(out + o0, a0);
        _mm512_storeu_pd(out + qs + o0, a1);
        _mm512_storeu_pd(out + 2 * qs + o0, a2);
        _mm512_storeu_pd(out + 3 * qs + o0, a3);
    }
}

// Lanes across columns: out[c] = init[c] (or +0) + sum over rows r = 0, 1, ..., R - 1 of W[r][c] coef[r], for NV blocks
// of 8 columns. Element (r, block v, lane l) is at W[r * rs + v * vs + l]: rs = row stride and vs = 8 for a row-major W;
// rs = 8 and vs = 8 R for a column-blocked copy (each block of 8 columns stored row after row, so it streams).
template <int NV>
CYPHA_LSTM_T512 inline void lstm_colsum512_n(const double* W, std::size_t rs, std::size_t vs, int R, const double* coef,
                                             const double* init, double* out) {
    __m512d acc[NV];
    for (int v = 0; v < NV; ++v) acc[v] = init ? _mm512_loadu_pd(init + 8 * v) : _mm512_setzero_pd();
    for (int r = 0; r < R; ++r) {
        const __m512d c = _mm512_set1_pd(coef[r]);
        const double* w = W + static_cast<std::size_t>(r) * rs;
        for (int v = 0; v < NV; ++v) acc[v] = lstm_madd8(acc[v], _mm512_loadu_pd(w + v * vs), c);
    }
    for (int v = 0; v < NV; ++v) _mm512_storeu_pd(out + 8 * v, acc[v]);
}
// The same for the column blocks [b0, b1) (init and out indexed by column).
CYPHA_LSTM_T512 inline void lstm_colsum512(const double* W, std::size_t rs, std::size_t vs, int R, const double* coef,
                                           const double* init, double* out, int b0, int b1) {
    int b = b0;
    auto w = [&](int k) { return W + static_cast<std::size_t>(k) * vs; };
    auto in = [&](int k) { return init ? init + static_cast<std::size_t>(k) * 8 : nullptr; };
    auto o = [&](int k) { return out + static_cast<std::size_t>(k) * 8; };
    for (; b1 - b >= 8; b += 8) lstm_colsum512_n<8>(w(b), rs, vs, R, coef, in(b), o(b));
    if (b1 - b >= 4) lstm_colsum512_n<4>(w(b), rs, vs, R, coef, in(b), o(b)), b += 4;
    if (b1 - b >= 2) lstm_colsum512_n<2>(w(b), rs, vs, R, coef, in(b), o(b)), b += 2;
    if (b1 - b >= 1) lstm_colsum512_n<1>(w(b), rs, vs, R, coef, in(b), o(b));
}
// Column blocks of a row-major block: dst[(b * nr + r) * 8 + l] = src[r * ld + 8 b + l] for rows r < nrows of the
// block (r counted from row0 in dst), blocks b < nb. Exact copies.
CYPHA_LSTM_T512 inline void lstm_colblocks512(const double* src, std::size_t ld, int row0, int nrows, int nb, double* dst,
                                              std::size_t nr) {
    for (int r = 0; r < nrows; ++r)
        for (int b = 0; b < nb; ++b)
            _mm512_store_pd(dst + (static_cast<std::size_t>(b) * nr + static_cast<std::size_t>(row0 + r)) * 8,
                            _mm512_load_pd(src + static_cast<std::size_t>(r) * ld + 8 * static_cast<std::size_t>(b)));
}

// Four weight rows' gradients at once, lanes across columns: row i's g[i * ldg + c] = +0 + sum over the steps
// it = T - 1, ..., 0 (last step first) of coef[it * cs + i] vec[it][c], for NV blocks of 8 columns from column col. The
// four rows share each column load.
template <int NV>
CYPHA_LSTM_T512 inline void lstm_grad4_512_n(const double* const* vec, const double* coef, std::size_t cs, int T, double* g,
                                             std::size_t ldg, int col) {
    __m512d acc[4][NV];
    for (int i = 0; i < 4; ++i)
        for (int v = 0; v < NV; ++v) acc[i][v] = _mm512_setzero_pd();
    for (int it = T - 1; it >= 0; --it) {
        const double* c = coef + static_cast<std::size_t>(it) * cs;
        const double* x = vec[it] + col;
        __m512d xv[NV];
        for (int v = 0; v < NV; ++v) xv[v] = _mm512_load_pd(x + 8 * v);
        for (int i = 0; i < 4; ++i) {
            const __m512d ci = _mm512_set1_pd(c[i]);
            for (int v = 0; v < NV; ++v) acc[i][v] = lstm_madd8(acc[i][v], ci, xv[v]);
        }
    }
    for (int i = 0; i < 4; ++i)
        for (int v = 0; v < NV; ++v) _mm512_store_pd(g + static_cast<std::size_t>(i) * ldg + col + 8 * v, acc[i][v]);
}
CYPHA_LSTM_T512 inline void lstm_grad4_512(const double* const* vec, const double* coef, std::size_t cs, int T, int len,
                                           double* g, std::size_t ldg) {
    const int nb = len / 8;
    int b = 0;
    for (; nb - b >= 4; b += 4) lstm_grad4_512_n<4>(vec, coef, cs, T, g, ldg, 8 * b);
    if (nb - b >= 2) lstm_grad4_512_n<2>(vec, coef, cs, T, g, ldg, 8 * b), b += 2;
    if (nb - b >= 1) lstm_grad4_512_n<1>(vec, coef, cs, T, g, ldg, 8 * b);
}

// Eight rows of a row-major block (row stride ld) into 8-lane columns: dst[c * ds + l] = src[l * ld + c], c < len.
// Exact copies (gathers).
CYPHA_LSTM_T512 inline void lstm_cols8_512(const double* src, std::size_t ld, int len, double* dst, std::size_t ds) {
    const long long s = static_cast<long long>(ld);
    const __m512i idx = _mm512_set_epi64(7 * s, 6 * s, 5 * s, 4 * s, 3 * s, 2 * s, s, 0);
    for (int c = 0; c < len; ++c)
        _mm512_store_pd(dst + static_cast<std::size_t>(c) * ds, _mm512_i64gather_pd(idx, src + c, 8));
}

// Adam, eight elements at a time, each element's arithmetic as lstm_adam_scalar's. Returns how many it did (a multiple
// of 8); the caller does the rest with the scalar loop.
CYPHA_LSTM_T512 inline std::size_t lstm_adam512(double* CYPHA_RESTRICT w, double* CYPHA_RESTRICT m, double* CYPHA_RESTRICT v,
                                                const double* CYPHA_RESTRICT gs, std::size_t len, double n, double c1,
                                                double c2, double lr) {
    const __m512d vn = _mm512_set1_pd(n), lo = _mm512_set1_pd(-5.0), hi = _mm512_set1_pd(5.0), b1 = _mm512_set1_pd(0.9),
                  a1 = _mm512_set1_pd(0.1), b2 = _mm512_set1_pd(0.999), a2 = _mm512_set1_pd(0.001), vc1 = _mm512_set1_pd(c1),
                  vc2 = _mm512_set1_pd(c2), eps = _mm512_set1_pd(1e-8), vlr = _mm512_set1_pd(lr);
    const bool d1 = c1 != 1.0, d2 = c2 != 1.0;
    std::size_t j = 0;
    for (; j + 8 <= len; j += 8) {
        const __m512d x = _mm512_div_pd(_mm512_loadu_pd(gs + j), vn);
        // (x < -5) ? -5 : (5 < x) ? 5 : x, as std::clamp
        const __mmask8 mlo = _mm512_cmp_pd_mask(x, lo, _CMP_LT_OS), mhi = _mm512_cmp_pd_mask(hi, x, _CMP_LT_OS);
        __m512d g = _mm512_mask_blend_pd(mhi, x, hi);
        g = _mm512_mask_blend_pd(mlo, g, lo);
        const __m512d mm = _mm512_add_pd(lstm_mul8(b1, _mm512_loadu_pd(m + j)), lstm_mul8(a1, g));
        const __m512d vv = _mm512_add_pd(lstm_mul8(b2, _mm512_loadu_pd(v + j)), lstm_mul8(lstm_mul8(a2, g), g));
        _mm512_storeu_pd(m + j, mm);
        _mm512_storeu_pd(v + j, vv);
        const __m512d mh = d1 ? _mm512_div_pd(mm, vc1) : mm, vh = d2 ? _mm512_div_pd(vv, vc2) : vv;
        const __m512d step = _mm512_div_pd(lstm_mul8(vlr, mh), _mm512_add_pd(_mm512_sqrt_pd(vh), eps));
        _mm512_storeu_pd(w + j, _mm512_sub_pd(_mm512_loadu_pd(w + j), step));
    }
    return j;
}
#endif  // CYPHA_LSTM_AVX512

// Whether this run uses the AVX-512 kernels: the CPU (and OS) has AVX-512F, and CYPHA_LSTM_AVX512 is not 0.
inline bool lstm_use_avx512() {
#if CYPHA_LSTM_AVX512
    static const bool on = [] {
        if (const char* e = std::getenv("CYPHA_LSTM_AVX512"))
            if (std::atoi(e) == 0) return false;
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx512f") != 0;
    }();
    return on;
#else
    return false;
#endif
}

// 64-byte aligned storage for the LSTM's arrays (whole cache lines, aligned vector loads).
template <class T>
struct LstmAlloc {
    using value_type = T;
    LstmAlloc() = default;
    template <class U>
    LstmAlloc(const LstmAlloc<U>&) {}
    T* allocate(std::size_t n) { return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{64})); }
    void deallocate(T* p, std::size_t) { ::operator delete(p, std::align_val_t{64}); }
    template <class U>
    bool operator==(const LstmAlloc<U>&) const { return true; }
    template <class U>
    bool operator!=(const LstmAlloc<U>&) const { return false; }
};
using lstm_vec = std::vector<double, LstmAlloc<double>>;

// A byte LSTM: embedding 16, hidden HH (64 as ByteLSTM), softmax over 256; learns online by truncated BPTT (window 20
// unless given) and Adam. ONLINE_LEARNING_UPGRADES (2026-11-04): the hidden size is a template parameter and the window a
// constructor argument, so context8k can try a bigger LSTM (--lstm-size, --lstm-window); ByteLSTM (64, 20) is unchanged,
// and the GLN and every other user read it as before.
// Optimisation (2026-11-05), every number unchanged: each dot product runs across rows (lstm_rows: one SIMD lane per row,
// each row summed in its own order, no reassociation); the weight gradients are summed once per window, fused with a
// vectorised Adam (the same arithmetic per element); the softmax that dist() computes is reused by BPTT when it is for
// the same state, and tanh(c) is kept from the forward pass; rows are split across a few threads (set_threads, or
// CYPHA_LSTM_THREADS; 4 by default from hidden 128, else 1). Every row is computed by one thread in its original order,
// so the output is the same for any thread count.
// OPT_SIMD (2026-11-05, GENERATION_PLAN §13.41 O5), every number unchanged again: with AVX-512 every matrix product runs
// across its outputs, eight lanes, each lane one output's sum in the old order. Each product reads a copy of W or Wo laid
// out for it (tiles of 32 rows for the gates and the logits, blocks of 8 columns for BPTT's sums down the columns), and
// the copies are rewritten after each Adam step. A whole BPTT window is one job: each thread owns some hidden units
// (their dh and gate errors) and the matching columns of the input's error, so a step needs one barrier; the
// embedding's columns are summed after the window. Each weight block's gradient over the window is summed four rows at
// a time. Adam skips its divisions by c1 and c2 once they are exactly 1. The window's steps live in place (no copies).
// Without AVX-512 (or with CYPHA_LSTM_AVX512=0) the SSE2 and scalar code runs as before.
template <int HH>
class ByteLSTMT {
public:
    static constexpr int H = HH, E = 16, X = E + H, kDefaultWindow = 20;
    static_assert(H % 8 == 0 && E % 8 == 0, "the AVX-512 tiles need multiples of 8");

    explicit ByteLSTMT(std::uint64_t seed = 0, double lr = 0.003, int window = kDefaultWindow)
        : lr_(lr), win_(window), v512_(lstm_use_avx512()) {
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> nd(0.0, 1.0);
        auto fill = [&](lstm_vec& v, std::size_t n, double s) {
            v.resize(n);
            for (auto& x : v) x = nd(rng) * s;
        };
        fill(Em_, 256 * E, 0.1);
        fill(W_, 4 * H * X, 0.1);
        b_.assign(4 * H, 0.0);
        for (int k = H; k < 2 * H; ++k) b_[k] = 1.0;  // forget gate bias
        fill(Wo_, 256 * H, 0.1);
        bo_.assign(256, 0.0);
        for (auto* p : {&Em_, &W_, &b_, &Wo_, &bo_}) {
            m_.emplace_back(p->size(), 0.0);
            v_.emplace_back(p->size(), 0.0);
        }
        h_.fill(0.0);
        c_.fill(0.0);
        slots_.resize(static_cast<std::size_t>(std::max(win_, 1)) + 1);
        relayout();
        forward(32);  // the state after reading a space predicts the first byte
    }

    const std::array<double, H>& h() const { return h_; }

    // Worker threads for this LSTM (1: none). The output is the same for any count.
    void set_threads(int n) { pool_.set(n); }
    int threads() const { return pool_.want(); }
    // Whether the AVX-512 kernels run (OPT_SIMD).
    bool avx512() const { return v512_; }

    // The distribution of the next byte from the state that has read every byte so far.
    void dist(double* p) const {
        double* z = zs_;
        auto job = [&](int t, int n) {
#if CYPHA_LSTM_AVX512
            if (v512_) {  // 8 tiles of 32 outputs
                lstm_tiles512(Wot_.data(), H, h_.data(), bo_.data(), z, 8, 32, 8 * t / n, 8 * (t + 1) / n);
                return;
            }
#endif
            const int y0 = 256 * t / n, y1 = 256 * (t + 1) / n;
            logits(h_.data(), z, y0, y1);
        };
        pool_.run(job);
        softmax(z, p);
        if (!moved_) {  // the softmax of the current step's h: BPTT reuses it
            std::copy(p, p + 256, pc_.begin());
            pv_ = true;
        }
    }

    // y followed the bytes read so far: learn that (when learn), then read y.
    void step(std::uint8_t y, bool learn) {
        if (learn) {
            Step& s = slots_[static_cast<std::size_t>(nbuf_)];
            s.y = y;
            s.has_p = pv_;
            if (pv_) s.p = pc_;
            if (++nbuf_ >= win_) {
                if (v512_) bptt512();
                else bptt();
                nbuf_ = 0;
            }
        }
        forward(y);
    }

    struct State {
        std::array<double, H> h, c;
    };
    void write(std::ostream& o) const {
        using namespace gln_io;
        for (const auto* v : {&Em_, &W_, &b_, &Wo_, &bo_}) put_vec(o, *v);
        for (const auto& v : m_) put_vec(o, v);
        for (const auto& v : v_) put_vec(o, v);
        put(o, t_);
        put(o, h_);
        put(o, c_);
    }
    void read(std::istream& i) {
        using namespace gln_io;
        for (auto* v : {&Em_, &W_, &b_, &Wo_, &bo_}) get_vec(i, *v);
        for (auto& v : m_) get_vec(i, v);
        for (auto& v : v_) get_vec(i, v);
        get(i, t_);
        get(i, h_);
        get(i, c_);
        nbuf_ = 0;
        relayout();
        forward(32);  // a fresh read of a space after the stored state
    }
    // §13.24: the learned LSTM to a file and back ("LST1"), so it can learn a text once (cyphalm_trie lstm-train) and be read
    // by any run.
    bool save_file(const std::string& path) const {
        std::ofstream o(path, std::ios::binary);
        o.write("LST1", 4);
        write(o);
        return static_cast<bool>(o);
    }
    static std::unique_ptr<ByteLSTMT> load_file(const std::string& path) {
        std::ifstream i(path, std::ios::binary);
        char magic[4] = {0, 0, 0, 0};
        i.read(magic, 4);
        if (!i || std::memcmp(magic, "LST1", 4) != 0) return nullptr;
        auto L = std::make_unique<ByteLSTMT>();
        L->read(i);
        return i ? std::move(L) : nullptr;
    }
    State save() const { return {h_, c_}; }
    // Restores the recurrent state (for writing: no learning happens between save and restore).
    void restore(const State& s) {
        h_ = s.h;
        c_ = s.c;
        moved_ = true;  // h_ may now differ from the current step's h: dist() no longer feeds BPTT until the next forward
        pv_ = false;
    }

private:
    struct Step {
        std::uint8_t x = 0, y = 0;
        bool has_p = false;
        alignas(64) std::array<double, X> xh;
        alignas(64) std::array<double, H> i, f, o, g, c_prev, c, tc, h;
        alignas(64) std::array<double, 256> p;  // the softmax of h, when dist() computed it
    };

    // The pool, made on first use; a copy of the LSTM gets its own (threads are not copied).
    struct PoolSlot {
        PoolSlot() : n_(default_threads()) {}
        PoolSlot(const PoolSlot& o) : n_(o.n_) {}
        PoolSlot& operator=(const PoolSlot& o) {
            if (this != &o) {
                n_ = o.n_;
                p_.reset();
            }
            return *this;
        }
        void set(int n) {
            n_ = std::clamp(n, 1, 64);
            p_.reset();
        }
        int want() const { return n_; }
        template <class F>
        void run(F& f) const {
            if (n_ <= 1) {
                f(0, 1);
                return;
            }
            if (!p_) p_ = std::make_unique<LstmPool>(n_);
            p_->run(f);
        }
        void sync() const {  // inside a job: all parts meet here
            if (n_ > 1) p_->barrier();
        }
        static int default_threads() {
            if (const char* e = std::getenv("CYPHA_LSTM_THREADS")) return std::clamp(std::atoi(e), 1, 64);
            return H >= 128 ? 4 : 1;
        }
        int n_;
        mutable std::unique_ptr<LstmPool> p_;
    };

    // z[y] = bo[y] + sum_k Wo[y][k] h[k] for y in [y0, y1), each in order k = 0, 1, ...
    void logits(const double* h, double* z, int y0, int y1) const { lstm_rows(Wo_.data(), H, h, bo_.data(), z, y0, y1); }

    static void softmax(const double* z, double* p) {
        double mx = -1e300;
        for (int y = 0; y < 256; ++y) {
            p[y] = z[y];
            mx = std::max(mx, z[y]);
        }
        double s = 0;
        for (int y = 0; y < 256; ++y) s += (p[y] = std::exp(p[y] - mx));
        for (int y = 0; y < 256; ++y) p[y] /= s;
    }

    // The copies the AVX-512 kernels read, exact copies of the values. Wf_ holds W's rows g * H + k in tiles of 8 units (4
    // gates x 8 lanes a column) for the gates; Wot_ holds Wo's rows in tiles of 32 outputs for the logits. Wc_ and Woc_
    // hold W and Wo by blocks of 8 columns, each block row after row, for BPTT's sums down the columns (dxh, dh).
    std::size_t wf_at(int r, int j) const {
        const int g = r / H, k = r % H;
        return (static_cast<std::size_t>(k / 8) * X + static_cast<std::size_t>(j)) * 32 + static_cast<std::size_t>(g) * 8 +
               static_cast<std::size_t>(k % 8);
    }
    static std::size_t wo_at(int y, int k) {
        return (static_cast<std::size_t>(y / 32) * H + static_cast<std::size_t>(k)) * 32 +
               static_cast<std::size_t>((y % 32) / 8) * 8 + static_cast<std::size_t>(y % 8);
    }
    static std::size_t cb_at(int rows, int r, int j) {
        return (static_cast<std::size_t>(j / 8) * rows + static_cast<std::size_t>(r)) * 8 + static_cast<std::size_t>(j % 8);
    }
    void relayout() {
        if (!v512_) return;
        Wf_.assign(W_.size(), 0.0);
        Wot_.assign(Wo_.size(), 0.0);
        Wc_.assign(W_.size(), 0.0);
        Woc_.assign(Wo_.size(), 0.0);
        for (int r = 0; r < 4 * H; ++r)
            for (int j = 0; j < X; ++j) {
                const double w = W_[static_cast<std::size_t>(r) * X + static_cast<std::size_t>(j)];
                Wf_[wf_at(r, j)] = w;
                Wc_[cb_at(4 * H, r, j)] = w;
            }
        for (int y = 0; y < 256; ++y)
            for (int k = 0; k < H; ++k) {
                const double w = Wo_[static_cast<std::size_t>(y) * H + static_cast<std::size_t>(k)];
                Wot_[wo_at(y, k)] = w;
                Woc_[cb_at(256, y, k)] = w;
            }
    }

    // The gates of units [k0, k1) from their pre-activations a (the old loop, unchanged).
    void gate_units(Step& s, const double* a, int k0, int k1) const {
        for (int k = k0; k < k1; ++k) {
            s.i[k] = 1 / (1 + std::exp(-a[k]));
            s.f[k] = 1 / (1 + std::exp(-a[H + k]));
            s.o[k] = 1 / (1 + std::exp(-a[2 * H + k]));
            s.g[k] = std::tanh(a[3 * H + k]);
            s.c_prev[k] = c_[k];
            s.c[k] = s.f[k] * c_[k] + s.i[k] * s.g[k];
            s.tc[k] = std::tanh(s.c[k]);
            s.h[k] = s.o[k] * s.tc[k];
        }
    }

    // Reads xb into the current step (slots_[nbuf_]).
    void forward(std::uint8_t xb) {
        Step& s = slots_[static_cast<std::size_t>(nbuf_)];
        s.x = xb;
        s.has_p = false;
        for (int k = 0; k < E; ++k) s.xh[k] = Em_[static_cast<std::size_t>(xb) * E + k];
        for (int k = 0; k < H; ++k) s.xh[E + k] = h_[k];
        double* a = as_;
#if CYPHA_LSTM_AVX512
        if (v512_) {  // one job: each thread's units, their 4 gate rows, then their gates
            auto job = [&](int t, int n) {
                const int kb0 = (H / 8) * t / n, kb1 = (H / 8) * (t + 1) / n;
                lstm_tiles512(Wf_.data(), X, s.xh.data(), b_.data(), a, H, 8, kb0, kb1);
                gate_units(s, a, 8 * kb0, 8 * kb1);
            };
            pool_.run(job);
        } else
#endif
        {
            auto rows = [&](int t, int n) {  // the 4H gate rows, a contiguous range per thread
                lstm_rows(W_.data(), X, s.xh.data(), b_.data(), a, 4 * H * t / n, 4 * H * (t + 1) / n);
            };
            pool_.run(rows);
            auto gates = [&](int t, int n) { gate_units(s, a, H * t / n, H * (t + 1) / n); };
            pool_.run(gates);
        }
        h_ = s.h;
        c_ = s.c;
        moved_ = false;
        pv_ = false;
    }

    // The errors of units [k0, k1) at step s from dh (the old loop, unchanged): the gates' errors da, and dc_next.
    void gate_errors(const Step& s, const double* dh, double* da, double* dc_next, int k0, int k1) const {
        for (int k = k0; k < k1; ++k) {
            const double tc = s.tc[k];
            const double d_o = dh[k] * tc * s.o[k] * (1 - s.o[k]);
            const double dc = dh[k] * s.o[k] * (1 - tc * tc) + dc_next[k];
            da[k] = dc * s.g[k] * s.i[k] * (1 - s.i[k]);
            da[H + k] = dc * s.c_prev[k] * s.f[k] * (1 - s.f[k]);
            da[2 * H + k] = d_o;
            da[3 * H + k] = dc * s.i[k] * (1 - s.g[k] * s.g[k]);
            dc_next[k] = dc * s.f[k];
        }
    }

    // Adam over a run of parameter block q's elements (j0, len) with gradient sums gs.
    void adam(int q, std::size_t j0, std::size_t len, double* w, const double* gs, double n, double c1, double c2) {
        double* m = m_[static_cast<std::size_t>(q)].data() + j0;
        double* v = v_[static_cast<std::size_t>(q)].data() + j0;
        std::size_t j = 0;
#if CYPHA_LSTM_AVX512
        if (v512_) j = lstm_adam512(w, m, v, gs, len, n, c1, c2, lr_);
#endif
        lstm_adam_scalar(w + j, m + j, v + j, gs + j, len - j, n, c1, c2, lr_);
    }

#if CYPHA_LSTM_AVX512
    // OPT_SIMD: truncated BPTT over the window slots_[0, nbuf_) and one Adam step, as one job. Every value is the one
    // bptt() computes, in the same order; only which thread computes it changes.
    void bptt512() {
        const int T = nbuf_;
        das_.resize(static_cast<std::size_t>(T) * 4 * H);
        ps_.resize(static_cast<std::size_t>(T) * 256);
        dxe_.resize(static_cast<std::size_t>(T) * E);
        gE_.assign(Em_.size(), 0.0);
        xs_.resize(static_cast<std::size_t>(T));
        hs_.resize(static_cast<std::size_t>(T));
        for (int it = 0; it < T; ++it) {
            xs_[static_cast<std::size_t>(it)] = slots_[static_cast<std::size_t>(it)].xh.data();
            hs_[static_cast<std::size_t>(it)] = slots_[static_cast<std::size_t>(it)].h.data();
        }
        std::fill(dhn_, dhn_ + H, 0.0);
        std::fill(dcn_, dcn_ + H, 0.0);
        ++t_;
        const double n = static_cast<double>(T);
        const double c1 = 1 - std::pow(0.9, t_), c2 = 1 - std::pow(0.999, t_);
        auto job = [&](int t, int nt) {
            // Each step's output error p - onehot(y): dist()'s softmax, else computed here; a step per thread in turn.
            alignas(64) double z[256];
            for (int it = t; it < T; it += nt) {
                const Step& s = slots_[static_cast<std::size_t>(it)];
                double* p = &ps_[static_cast<std::size_t>(it) * 256];
                if (s.has_p) {
                    std::copy(s.p.begin(), s.p.end(), p);
                } else {
                    lstm_tiles512(Wot_.data(), H, s.h.data(), bo_.data(), z, 8, 32, 0, 8);
                    softmax(z, p);
                }
                p[s.y] -= 1;
            }
            pool_.sync();
            // This thread's units [8 kb0, 8 kb1) and the input-error columns that feed them back, E + 8 kb0 on.
            const int kb0 = (H / 8) * t / nt, kb1 = (H / 8) * (t + 1) / nt;
            for (int it = T - 1; it >= 0; --it) {
                const Step& s = slots_[static_cast<std::size_t>(it)];
                const double* p = &ps_[static_cast<std::size_t>(it) * 256];
                double* da = &das_[static_cast<std::size_t>(it) * 4 * H];
                if (kb1 > kb0) {  // dh[k] = dh_next[k] + sum over y = 0, 1, ... of Wo[y][k] p[y]; then the gates' errors
                    lstm_colsum512(Woc_.data(), 8, 256 * 8, 256, p, dhn_, dh_, kb0, kb1);
                    gate_errors(s, dh_, da, dcn_, 8 * kb0, 8 * kb1);
                }
                pool_.sync();
                // The input's error: dxh[j] = sum over the rows r = 0, 1, ... of W[r][j] da[r], in that order; here its
                // hidden part, which feeds the step before back.
                if (kb1 > kb0) lstm_colsum512(Wc_.data(), 8, 4 * H * 8, 4 * H, da, nullptr, dxh_, E / 8 + kb0, E / 8 + kb1);
                for (int k = 8 * kb0; k < 8 * kb1; ++k) dhn_[k] = dxh_[E + k];
            }
            pool_.sync();
            // The embedding part of every step's dxh (the same sums), a step per thread in turn; then its gradient,
            // added in the old order (last step first).
            for (int it = t; it < T; it += nt)
                lstm_colsum512(Wc_.data(), 8, 4 * H * 8, 4 * H, &das_[static_cast<std::size_t>(it) * 4 * H], nullptr,
                               &dxe_[static_cast<std::size_t>(it) * E], 0, E / 8);
            pool_.sync();
            if (t == 0)
                for (int it = T - 1; it >= 0; --it)
                    for (int k = 0; k < E; ++k)
                        gE_[static_cast<std::size_t>(slots_[static_cast<std::size_t>(it)].x) * E + k] +=
                            dxe_[static_cast<std::size_t>(it) * E + k];
            pool_.sync();
            // The update, in blocks of 8 rows per thread: the rows' gradients over the steps (last first), Adam, and the
            // block's place in the tiled copy.
            alignas(64) double g[8 * (X > 256 ? X : 256)];
            const int rb0 = (4 * H / 8) * t / nt, rb1 = (4 * H / 8) * (t + 1) / nt;
            for (int r0 = 8 * rb0; r0 < 8 * rb1; r0 += 8) {
                lstm_grad4_512(xs_.data(), &das_[static_cast<std::size_t>(r0)], 4 * H, T, X, g, X);
                lstm_grad4_512(xs_.data(), &das_[static_cast<std::size_t>(r0) + 4], 4 * H, T, X, g + 4 * X, X);
                for (int i = 0; i < 8; ++i) {
                    const int r = r0 + i;
                    double gb = 0;
                    for (int it = T - 1; it >= 0; --it) gb += das_[static_cast<std::size_t>(it) * 4 * H + r];
                    adam(1, static_cast<std::size_t>(r) * X, X, &W_[static_cast<std::size_t>(r) * X], g + i * X, n, c1, c2);
                    adam(2, static_cast<std::size_t>(r), 1, &b_[static_cast<std::size_t>(r)], &gb, n, c1, c2);
                }
                lstm_cols8_512(&W_[static_cast<std::size_t>(r0) * X], X, X, &Wf_[wf_at(r0, 0)], 32);
                lstm_colblocks512(&W_[static_cast<std::size_t>(r0) * X], X, r0, 8, X / 8, Wc_.data(), 4 * H);
            }
            const int yb0 = 32 * t / nt, yb1 = 32 * (t + 1) / nt;
            for (int y0 = 8 * yb0; y0 < 8 * yb1; y0 += 8) {
                lstm_grad4_512(hs_.data(), &ps_[static_cast<std::size_t>(y0)], 256, T, H, g, H);
                lstm_grad4_512(hs_.data(), &ps_[static_cast<std::size_t>(y0) + 4], 256, T, H, g + 4 * H, H);
                for (int i = 0; i < 8; ++i) {
                    const int y = y0 + i;
                    double gb = 0;
                    for (int it = T - 1; it >= 0; --it) gb += ps_[static_cast<std::size_t>(it) * 256 + y];
                    adam(3, static_cast<std::size_t>(y) * H, H, &Wo_[static_cast<std::size_t>(y) * H], g + i * H, n, c1, c2);
                    adam(4, static_cast<std::size_t>(y), 1, &bo_[static_cast<std::size_t>(y)], &gb, n, c1, c2);
                }
                lstm_cols8_512(&Wo_[static_cast<std::size_t>(y0) * H], H, H, &Wot_[wo_at(y0, 0)], 32);
                lstm_colblocks512(&Wo_[static_cast<std::size_t>(y0) * H], H, y0, 8, H / 8, Woc_.data(), 256);
            }
            const std::size_t ne = Em_.size(), e0 = ne * t / nt, e1 = ne * (t + 1) / nt;
            adam(0, e0, e1 - e0, Em_.data() + e0, gE_.data() + e0, n, c1, c2);
        };
        pool_.run(job);
    }
#else
    void bptt512() { bptt(); }
#endif

    // Truncated BPTT over the window slots_[0, nbuf_), then one Adam step. The backward pass keeps each step's gate and
    // output errors (da, p); the weight gradients are then summed once per window, each element over the steps in the
    // same order as before (last step first), fused with Adam. Only dh and dxh, which feed the next step back, are
    // computed step by step.
    void bptt() {
        const int T = nbuf_;
        das_.resize(static_cast<std::size_t>(T) * 4 * H);
        ps_.resize(static_cast<std::size_t>(T) * 256);
        gE_.assign(Em_.size(), 0.0);
        std::array<double, H> dh_next{}, dc_next{};
        std::array<double, X> dxh;
        for (int it = T - 1; it >= 0; --it) {
            const Step& s = slots_[static_cast<std::size_t>(it)];
            double* p = &ps_[static_cast<std::size_t>(it) * 256];
            double* da = &das_[static_cast<std::size_t>(it) * 4 * H];
            if (s.has_p) {
                std::copy(s.p.begin(), s.p.end(), p);
            } else {
                double* z = zs_;
                auto lj = [&](int t, int n) { logits(s.h.data(), z, 256 * t / n, 256 * (t + 1) / n); };
                pool_.run(lj);
                softmax(z, p);
            }
            p[s.y] -= 1;
            // dh and the gates' errors, a range of hidden units per thread (each unit's sum over y in order).
            auto ja = [&](int t, int n) {
                const int k0 = H * t / n, k1 = H * (t + 1) / n;
                double dh[H];
                for (int k = k0; k < k1; ++k) dh[k] = dh_next[k];
                for (int y = 0; y < 256; ++y) {
                    const double py = p[y];
                    const double* wo = &Wo_[static_cast<std::size_t>(y) * H];
                    for (int k = k0; k < k1; ++k) dh[k] += wo[k] * py;
                }
                gate_errors(s, dh, da, dc_next.data(), k0, k1);
            };
            pool_.run(ja);
            // The input's error: dxh[j] = sum over the rows r = 0, 1, ... of W[r][j] da[r], in that order. On the calling
            // thread: it streams all of W once a step, and every split across threads (by columns, or by rows as a
            // wavefront that keeps the order) measured slower on Odin's PC.
            for (int j = 0; j < X; ++j) dxh[j] = 0;
            for (int r = 0; r < 4 * H; ++r) {
                const double d = da[r];
                const double* w = &W_[static_cast<std::size_t>(r) * X];
                for (int j = 0; j < X; ++j) dxh[j] += w[j] * d;
            }
            for (int k = 0; k < E; ++k) gE_[static_cast<std::size_t>(s.x) * E + k] += dxh[k];
            for (int k = 0; k < H; ++k) dh_next[k] = dxh[E + k];
        }
        ++t_;
        const double n = static_cast<double>(T);
        const double c1 = 1 - std::pow(0.9, t_), c2 = 1 - std::pow(0.999, t_);
        // Adam over a run of elements. With SSE2, two elements a time: IEEE division and square root give the same
        // result in SIMD as in scalar code, and each element's arithmetic is unchanged (no FMA in the default build).
        auto adam2 = [&](int q, std::size_t j0, std::size_t len, double* CYPHA_RESTRICT w, const double* CYPHA_RESTRICT gs) {
            double* CYPHA_RESTRICT m = m_[static_cast<std::size_t>(q)].data() + j0;
            double* CYPHA_RESTRICT v = v_[static_cast<std::size_t>(q)].data() + j0;
            const double lr = lr_;
            std::size_t j = 0;
#if CYPHA_LSTM_SSE2
            const __m128d vn = _mm_set1_pd(n), lo = _mm_set1_pd(-5.0), hi = _mm_set1_pd(5.0), b1 = _mm_set1_pd(0.9),
                          a1 = _mm_set1_pd(0.1), b2 = _mm_set1_pd(0.999), a2 = _mm_set1_pd(0.001), vc1 = _mm_set1_pd(c1),
                          vc2 = _mm_set1_pd(c2), eps = _mm_set1_pd(1e-8), vlr = _mm_set1_pd(lr);
            for (; j + 2 <= len; j += 2) {
                const __m128d x = _mm_div_pd(_mm_loadu_pd(gs + j), vn);
                // (x < -5) ? -5 : (5 < x) ? 5 : x, as std::clamp
                const __m128d mlo = _mm_cmplt_pd(x, lo), mhi = _mm_cmplt_pd(hi, x);
                __m128d g = _mm_or_pd(_mm_and_pd(mhi, hi), _mm_andnot_pd(mhi, x));
                g = _mm_or_pd(_mm_and_pd(mlo, lo), _mm_andnot_pd(mlo, g));
                const __m128d mm = _mm_add_pd(_mm_mul_pd(b1, _mm_loadu_pd(m + j)), _mm_mul_pd(a1, g));
                const __m128d vv = _mm_add_pd(_mm_mul_pd(b2, _mm_loadu_pd(v + j)), _mm_mul_pd(_mm_mul_pd(a2, g), g));
                _mm_storeu_pd(m + j, mm);
                _mm_storeu_pd(v + j, vv);
                const __m128d step = _mm_div_pd(_mm_mul_pd(vlr, _mm_div_pd(mm, vc1)), _mm_add_pd(_mm_sqrt_pd(_mm_div_pd(vv, vc2)), eps));
                _mm_storeu_pd(w + j, _mm_sub_pd(_mm_loadu_pd(w + j), step));
            }
#endif
            for (; j < len; ++j) {
                const double x = gs[j] / n;
                const double g = (x < -5.0) ? -5.0 : (5.0 < x) ? 5.0 : x;  // std::clamp(x, -5, 5)
                m[j] = 0.9 * m[j] + 0.1 * g;
                v[j] = 0.999 * v[j] + 0.001 * g * g;
                w[j] -= lr * (m[j] / c1) / (std::sqrt(v[j] / c2) + 1e-8);
            }
        };
        auto upd = [&](int t, int nt) {
            double g[X > 256 ? X : 256];
            // Gate rows: W, b.
            const int r0 = 4 * H * t / nt, r1 = 4 * H * (t + 1) / nt;
            for (int r = r0; r < r1; ++r) {
                for (int j = 0; j < X; ++j) g[j] = 0;
                double gb = 0;
                for (int it = T - 1; it >= 0; --it) {
                    const double d = das_[static_cast<std::size_t>(it) * 4 * H + r];
                    const double* xh = slots_[static_cast<std::size_t>(it)].xh.data();
                    gb += d;
                    for (int j = 0; j < X; ++j) g[j] += d * xh[j];
                }
                adam2(1, static_cast<std::size_t>(r) * X, X, &W_[static_cast<std::size_t>(r) * X], g);
                adam2(2, static_cast<std::size_t>(r), 1, &b_[static_cast<std::size_t>(r)], &gb);
            }
            // Output rows: Wo, bo.
            const int y0 = 256 * t / nt, y1 = 256 * (t + 1) / nt;
            for (int y = y0; y < y1; ++y) {
                for (int k = 0; k < H; ++k) g[k] = 0;
                double gb = 0;
                for (int it = T - 1; it >= 0; --it) {
                    const double py = ps_[static_cast<std::size_t>(it) * 256 + y];
                    const double* h = slots_[static_cast<std::size_t>(it)].h.data();
                    gb += py;
                    for (int k = 0; k < H; ++k) g[k] += py * h[k];
                }
                adam2(3, static_cast<std::size_t>(y) * H, H, &Wo_[static_cast<std::size_t>(y) * H], g);
                adam2(4, static_cast<std::size_t>(y), 1, &bo_[static_cast<std::size_t>(y)], &gb);
            }
            // The embedding.
            const std::size_t ne = Em_.size(), e0 = ne * t / nt, e1 = ne * (t + 1) / nt;
            adam2(0, e0, e1 - e0, Em_.data() + e0, gE_.data() + e0);
        };
        pool_.run(upd);
    }

    double lr_;
    int win_;
    int t_ = 0;
    bool v512_ = false;                     // OPT_SIMD: the AVX-512 kernels and the tiled copies
    lstm_vec Em_, W_, b_, Wo_, bo_;
    std::vector<lstm_vec> m_, v_;
    lstm_vec Wf_, Wot_, Wc_, Woc_;          // the copies of W and Wo the AVX-512 kernels read (see relayout)
    lstm_vec das_, ps_, gE_;                // BPTT's per-step gate and output errors; the embedding's gradient
    lstm_vec dxe_;                          // bptt512: each step's dxh over the embedding's columns
    std::vector<const double*> xs_, hs_;    // BPTT: each step's xh and h
    std::array<double, H> h_, c_;
    std::vector<Step> slots_;               // the window's steps [0, nbuf_), then the current step
    int nbuf_ = 0;
    mutable double zs_[256];                // logits scratch
    alignas(64) double as_[4 * H];          // gate pre-activations scratch
    alignas(64) double dh_[H], dhn_[H], dcn_[H], dxh_[X];  // bptt512's shared rows (each thread its own entries)
    mutable std::array<double, 256> pc_;    // dist()'s softmax of the current step's h
    mutable bool pv_ = false;
    bool moved_ = false;                    // restore() moved h_ away from the current step's h
    PoolSlot pool_;
};

using ByteLSTM = ByteLSTMT<64>;  // the LSTM of §13.18: hidden 64, window 20

}  // namespace cypha::online
