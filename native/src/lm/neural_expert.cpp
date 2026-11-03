#include "cypha/cyphalm/neural_expert.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>

// MinGW's x64 GCC refuses a 32-byte stack boundary, then still emits vmovaps
// for YMM spills, which faults. The AVX kernels stay available everywhere else.
#if defined(__x86_64__) && defined(__GNUC__) && !defined(__MINGW32__)
#include <immintrin.h>
#define CYPHA_NN_X86 1
// MinGW callers keep the 16-byte SysV/x64 stack. AVX spills use aligned moves,
// so a target("avx*") function must realign the stack on entry or it faults.
#define CYPHA_NN_AVX2 __attribute__((target("avx2,fma"), force_align_arg_pointer))
#define CYPHA_NN_AVX512 __attribute__((target("avx512f,avx2,fma"), force_align_arg_pointer))
#else
#define CYPHA_NN_X86 0
#endif

namespace cypha::cyphalm {

namespace {

// ---------------------------------------------------------------- kernels
//
// Three kernel sets, one chosen per process (neural_kernel()): "portable" is
// the original code (no FMA; bit for bit the results of earlier builds);
// "avx2" and "avx512" use FMA, so they differ from it in float rounding only.
// The two FMA sets are bit-identical to each other: every output lane is the
// same chain of fused operations in the same order, whatever the vector width.

#if defined(__GNUC__) && defined(__x86_64__) && defined(__linux__) && !defined(__clang__)
#define CYPHA_NN_CLONES __attribute__((target_clones("avx2", "default")))
#else
#define CYPHA_NN_CLONES
#endif

// out[r] = bias[r] + sum_k w[r][k] * x[k] with bf16 weights (the upper 16
// bits of a float32): half the memory traffic, which is what bounds a single
// stream. Eight partial sums so the compiler vectorises without reassociating.
CYPHA_NN_CLONES
void matvec_bf16_portable(const std::uint16_t* w, const float* x, const float* bias, float* out, int rows,
                          int cols) {
    const int c8 = cols & ~7;
    for (int r = 0; r < rows; ++r) {
        const std::uint16_t* wr = w + static_cast<std::size_t>(r) * cols;
        float a[8] = {};
        for (int k = 0; k < c8; k += 8)
            for (int j = 0; j < 8; ++j) {
                const std::uint32_t u = static_cast<std::uint32_t>(wr[k + j]) << 16;
                float f;
                std::memcpy(&f, &u, sizeof(f));
                a[j] += f * x[k + j];
            }
        float s = 0.0f;
        for (int j = 0; j < 8; ++j) s += a[j];
        for (int k = c8; k < cols; ++k) {
            const std::uint32_t u = static_cast<std::uint32_t>(wr[k]) << 16;
            float f;
            std::memcpy(&f, &u, sizeof(f));
            s += f * x[k];
        }
        out[r] = s + (bias ? bias[r] : 0.0f);
    }
}

CYPHA_NN_CLONES
float dot(const float* a, const float* b, int n) {
    float acc[8] = {};
    const int n8 = n & ~7;
    for (int k = 0; k < n8; k += 8)
        for (int j = 0; j < 8; ++j) acc[j] += a[k + j] * b[k + j];
    float s = 0.0f;
    for (int j = 0; j < 8; ++j) s += acc[j];
    for (int k = n8; k < n; ++k) s += a[k] * b[k];
    return s;
}

// dot(a, b') with b'[k] = b[k * stride]: the same arithmetic in the same order.
float dot_strided(const float* a, const float* b, std::size_t stride, int n) {
    float acc[8] = {};
    const int n8 = n & ~7;
    for (int k = 0; k < n8; k += 8)
        for (int j = 0; j < 8; ++j) acc[j] += a[k + j] * b[static_cast<std::size_t>(k + j) * stride];
    float s = 0.0f;
    for (int j = 0; j < 8; ++j) s += acc[j];
    for (int k = n8; k < n; ++k) s += a[k] * b[static_cast<std::size_t>(k) * stride];
    return s;
}

void matvec_f32_portable(const float* w, const float* x, const float* bias, float* out, int rows, int cols) {
    for (int r = 0; r < rows; ++r) out[r] = dot(w + static_cast<std::size_t>(r) * cols, x, cols) + bias[r];
}

// y[k] -= a * x[k]
void sub_scaled_portable(float* y, float a, const float* x, int n) {
    for (int k = 0; k < n; ++k) y[k] -= a * x[k];
}

// Softmax of each head's scores sc[h][0, n) in place (the attention weights).
void softmax_heads(float* sc, int heads, int ctx, int n) {
    for (int hh = 0; hh < heads; ++hh) {
        float* s = sc + static_cast<std::size_t>(hh) * ctx;
        float mx = -1e30f;
        for (int t = 0; t < n; ++t) mx = std::max(mx, s[t]);
        float z = 0.0f;
        for (int t = 0; t < n; ++t) z += (s[t] = std::exp(s[t] - mx));
        for (int t = 0; t < n; ++t) s[t] /= z;
    }
}

// Causal attention of the query at position n - 1: keys Kt [d][ctx]
// (transposed, so the fast kernels vectorise over positions), values
// V [ctx][d], scores sc [heads][ctx] (scratch); att [d] out.
void attend_portable(const float* q, const float* Kt, const float* V, int n, int d, int heads, int ctx, float scale,
                     float* sc, float* att) {
    const int hd = d / heads;
    for (int hh = 0; hh < heads; ++hh)
        for (int t = 0; t < n; ++t)
            sc[static_cast<std::size_t>(hh) * ctx + t] =
                dot_strided(q + hh * hd, Kt + static_cast<std::size_t>(hh) * hd * ctx + t, ctx, hd) * scale;
    softmax_heads(sc, heads, ctx, n);
    for (int hh = 0; hh < heads; ++hh) {
        const float* w = sc + static_cast<std::size_t>(hh) * ctx;
        float* o = att + hh * hd;
        std::fill(o, o + hd, 0.0f);
        for (int t = 0; t < n; ++t) {
            const float* v = V + static_cast<std::size_t>(t) * d + hh * hd;
            for (int k = 0; k < hd; ++k) o[k] += w[t] * v[k];
        }
    }
}

#if CYPHA_NN_X86

// The FMA bf16 kernels split each 32-column chunk of a row into its even and
// odd columns (a shift and an AND of the loaded 32-bit words: no shuffle), so
// x is laid out the same way, per chunk x[0], x[2], ..., x[30], x[1], ...,
// x[31], zero-padded to whole chunks. Lane j < 16 of a row's accumulators sums
// column 2j of every chunk, lane 16 + j column 2j + 1 (fused, chunk order).
const float* split_x(const float* x, int cols) {
    thread_local std::vector<float> buf;
    const int chunks = (cols + 31) / 32;
    if (buf.size() < static_cast<std::size_t>(chunks) * 32) buf.resize(static_cast<std::size_t>(chunks) * 32);
    float* o = buf.data();
    for (int m = 0; m < chunks; ++m, o += 32)
        for (int i = 0; i < 16; ++i) {
            const int e = 32 * m + 2 * i;
            o[i] = e < cols ? x[e] : 0.0f;
            o[16 + i] = e + 1 < cols ? x[e + 1] : 0.0f;
        }
    return buf.data();
}

// Both FMA sets reduce a row's 32 lanes the same way: (lane j + lane j + 16)
// for j < 16, then j + (j + 8), then this, lanes j + (j + 4), + (j + 2), + 1.
CYPHA_NN_AVX2 inline float hsum8(__m256 v) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_movehdup_ps(a));
    return _mm_cvtss_f32(a);
}

// ---- AVX2 + FMA: 16 ymm registers, so two rows of four accumulators.

// One 32-column chunk of R rows (x split as above) into four accumulators each.
template <int R>
CYPHA_NN_AVX2 inline void bf16_chunk_avx2(const std::uint16_t* const* wr, const float* x, __m256* e0, __m256* o0,
                                          __m256* e1, __m256* o1) {
    const __m256i hi = _mm256_set1_epi32(static_cast<int>(0xFFFF0000u));
    const __m256 xe0 = _mm256_loadu_ps(x), xe1 = _mm256_loadu_ps(x + 8);
    const __m256 xo0 = _mm256_loadu_ps(x + 16), xo1 = _mm256_loadu_ps(x + 24);
    for (int i = 0; i < R; ++i) {
        const __m256i u0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wr[i]));
        const __m256i u1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wr[i] + 16));
        e0[i] = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(u0, 16)), xe0, e0[i]);
        o0[i] = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_and_si256(u0, hi)), xo0, o0[i]);
        e1[i] = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(u1, 16)), xe1, e1[i]);
        o1[i] = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_and_si256(u1, hi)), xo1, o1[i]);
    }
}

template <int R>
CYPHA_NN_AVX2 inline void mv_bf16_rows_avx2(const std::uint16_t* w, std::size_t cols, const float* xs, int full,
                                            int tail, const float* bias, float* out) {
    __m256 e0[R], o0[R], e1[R], o1[R];
    for (int i = 0; i < R; ++i) e0[i] = o0[i] = e1[i] = o1[i] = _mm256_setzero_ps();
    const std::uint16_t* wr[R];
    for (int i = 0; i < R; ++i) wr[i] = w + i * cols;
    for (int m = 0; m < full; ++m) {
        bf16_chunk_avx2<R>(wr, xs + 32 * static_cast<std::size_t>(m), e0, o0, e1, o1);
        for (int i = 0; i < R; ++i) wr[i] += 32;
    }
    if (tail != 0) {  // last partial chunk, zero-padded
        std::uint16_t tw[R][32] = {};
        const std::uint16_t* tr[R];
        for (int i = 0; i < R; ++i) {
            std::copy_n(wr[i], tail, tw[i]);
            tr[i] = tw[i];
        }
        bf16_chunk_avx2<R>(tr, xs + 32 * static_cast<std::size_t>(full), e0, o0, e1, o1);
    }
    for (int i = 0; i < R; ++i)
        out[i] = hsum8(_mm256_add_ps(_mm256_add_ps(e0[i], o0[i]), _mm256_add_ps(e1[i], o1[i]))) +
                 (bias ? bias[i] : 0.0f);
}

CYPHA_NN_AVX2 void matvec_bf16_avx2(const std::uint16_t* w, const float* x, const float* bias, float* out, int rows,
                                    int cols) {
    const float* xs = split_x(x, cols);
    const std::size_t c = static_cast<std::size_t>(cols);
    int r = 0;
    for (; r + 2 <= rows; r += 2)
        mv_bf16_rows_avx2<2>(w + r * c, c, xs, cols / 32, cols % 32, bias ? bias + r : nullptr, out + r);
    for (; r < rows; ++r)
        mv_bf16_rows_avx2<1>(w + r * c, c, xs, cols / 32, cols % 32, bias ? bias + r : nullptr, out + r);
}

// Float weights (the adapted output layer): lane j sums column j of every
// 32-column chunk; the same reduction.
template <int R>
CYPHA_NN_AVX2 inline void mv_f32_rows_avx2(const float* w, std::size_t cols, const float* x, int full, int tail,
                                           const float* bias, float* out) {
    __m256 a0[R], a1[R], a2[R], a3[R];
    for (int i = 0; i < R; ++i) a0[i] = a1[i] = a2[i] = a3[i] = _mm256_setzero_ps();
    float tw[R][32], tx[32];
    for (int m = 0; m <= full; ++m) {
        const float* wr[R];
        const float* xm = x + 32 * static_cast<std::size_t>(m);
        if (m == full) {
            if (tail == 0) break;
            std::fill(tx, tx + 32, 0.0f);
            std::copy_n(xm, tail, tx);
            xm = tx;
            for (int i = 0; i < R; ++i) {
                std::fill(tw[i], tw[i] + 32, 0.0f);
                std::copy_n(w + i * cols + 32 * static_cast<std::size_t>(m), tail, tw[i]);
                wr[i] = tw[i];
            }
        } else {
            for (int i = 0; i < R; ++i) wr[i] = w + i * cols + 32 * static_cast<std::size_t>(m);
        }
        const __m256 x0 = _mm256_loadu_ps(xm), x1 = _mm256_loadu_ps(xm + 8);
        const __m256 x2 = _mm256_loadu_ps(xm + 16), x3 = _mm256_loadu_ps(xm + 24);
        for (int i = 0; i < R; ++i) {
            a0[i] = _mm256_fmadd_ps(_mm256_loadu_ps(wr[i]), x0, a0[i]);
            a1[i] = _mm256_fmadd_ps(_mm256_loadu_ps(wr[i] + 8), x1, a1[i]);
            a2[i] = _mm256_fmadd_ps(_mm256_loadu_ps(wr[i] + 16), x2, a2[i]);
            a3[i] = _mm256_fmadd_ps(_mm256_loadu_ps(wr[i] + 24), x3, a3[i]);
        }
    }
    for (int i = 0; i < R; ++i)
        out[i] = hsum8(_mm256_add_ps(_mm256_add_ps(a0[i], a2[i]), _mm256_add_ps(a1[i], a3[i]))) + bias[i];
}

CYPHA_NN_AVX2 void matvec_f32_avx2(const float* w, const float* x, const float* bias, float* out, int rows,
                                   int cols) {
    const std::size_t c = static_cast<std::size_t>(cols);
    int r = 0;
    for (; r + 2 <= rows; r += 2) mv_f32_rows_avx2<2>(w + r * c, c, x, cols / 32, cols % 32, bias + r, out + r);
    for (; r < rows; ++r) mv_f32_rows_avx2<1>(w + r * c, c, x, cols / 32, cols % 32, bias + r, out + r);
}

CYPHA_NN_AVX2 void sub_scaled_avx2(float* y, float a, const float* x, int n) {
    const __m256 av = _mm256_set1_ps(a);
    int k = 0;
    for (; k + 8 <= n; k += 8)
        _mm256_storeu_ps(y + k, _mm256_fnmadd_ps(av, _mm256_loadu_ps(x + k), _mm256_loadu_ps(y + k)));
    for (; k < n; ++k) y[k] = std::fma(-a, x[k], y[k]);
}

// Attention: score t of head h is a fused chain over the head's dimensions
// (in order, from 0), times scale; output lane k a fused chain over positions.
template <int B>
CYPHA_NN_AVX2 inline void scores_avx2(const float* q, const float* Kh, std::size_t ctx, int hd, int t, float scale,
                                      float* s) {
    __m256 a[B];
    for (int b = 0; b < B; ++b) a[b] = _mm256_setzero_ps();
    for (int k = 0; k < hd; ++k) {
        const __m256 qk = _mm256_set1_ps(q[k]);
        const float* row = Kh + k * ctx + t;
        for (int b = 0; b < B; ++b) a[b] = _mm256_fmadd_ps(qk, _mm256_loadu_ps(row + 8 * b), a[b]);
    }
    for (int b = 0; b < B; ++b) _mm256_storeu_ps(s + t + 8 * b, _mm256_mul_ps(a[b], _mm256_set1_ps(scale)));
}

template <int B>
CYPHA_NN_AVX2 inline void values_avx2(const float* sc, const float* V, int n, int d, int hd, std::size_t ctx, int k0,
                                      float* att) {
    __m256 a[B];
    const float* w[B];
    for (int b = 0; b < B; ++b) {
        a[b] = _mm256_setzero_ps();
        w[b] = sc + static_cast<std::size_t>((k0 + 8 * b) / hd) * ctx;
    }
    for (int t = 0; t < n; ++t) {
        const float* v = V + static_cast<std::size_t>(t) * d + k0;
        for (int b = 0; b < B; ++b) a[b] = _mm256_fmadd_ps(_mm256_set1_ps(w[b][t]), _mm256_loadu_ps(v + 8 * b), a[b]);
    }
    for (int b = 0; b < B; ++b) _mm256_storeu_ps(att + k0 + 8 * b, a[b]);
}

// Lanes the vector loops leave over (score positions, or all value lanes
// when a head is not whole vectors): the same fused chains one at a time.
CYPHA_NN_AVX2 inline float score_one(const float* q, const float* Kh, std::size_t ctx, int hd, int t, float scale) {
    float a = 0.0f;
    for (int k = 0; k < hd; ++k) a = std::fma(q[k], Kh[k * ctx + t], a);
    return a * scale;
}

CYPHA_NN_AVX2 inline void values_one(const float* sc, const float* V, int n, int d, int hd, std::size_t ctx,
                                     float* att) {
    for (int k = 0; k < d; ++k) {
        const float* w = sc + static_cast<std::size_t>(k / hd) * ctx;
        float a = 0.0f;
        for (int t = 0; t < n; ++t) a = std::fma(w[t], V[static_cast<std::size_t>(t) * d + k], a);
        att[k] = a;
    }
}

CYPHA_NN_AVX2 void attend_avx2(const float* q, const float* Kt, const float* V, int n, int d, int heads, int ctx,
                               float scale, float* sc, float* att) {
    const int hd = d / heads;
    const std::size_t c = static_cast<std::size_t>(ctx);
    for (int hh = 0; hh < heads; ++hh) {
        const float* qh = q + hh * hd;
        const float* Kh = Kt + static_cast<std::size_t>(hh) * hd * c;
        float* s = sc + static_cast<std::size_t>(hh) * c;
        int t = 0;
        for (; t + 64 <= n; t += 64) scores_avx2<8>(qh, Kh, c, hd, t, scale, s);
        for (; t + 8 <= n; t += 8) scores_avx2<1>(qh, Kh, c, hd, t, scale, s);
        for (; t < n; ++t) s[t] = score_one(qh, Kh, c, hd, t, scale);
    }
    softmax_heads(sc, heads, ctx, n);
    if (hd % 8 != 0) return values_one(sc, V, n, d, hd, c, att);
    int k0 = 0;
    for (; k0 + 64 <= d; k0 += 64) values_avx2<8>(sc, V, n, d, hd, c, k0, att);
    for (; k0 < d; k0 += 8) values_avx2<1>(sc, V, n, d, hd, c, k0, att);
}

// ---- AVX-512: four rows of two accumulators (lanes 0-15, 16-31).

template <int R>
CYPHA_NN_AVX512 inline void bf16_chunk_avx512(const std::uint16_t* const* wr, const float* x, __m512* e, __m512* o) {
    const __m512i hi = _mm512_set1_epi32(static_cast<int>(0xFFFF0000u));
    const __m512 xe = _mm512_loadu_ps(x), xo = _mm512_loadu_ps(x + 16);
    for (int i = 0; i < R; ++i) {
        const __m512i u = _mm512_loadu_si512(wr[i]);
        e[i] = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(u, 16)), xe, e[i]);
        o[i] = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_and_si512(u, hi)), xo, o[i]);
    }
}

template <int R>
CYPHA_NN_AVX512 inline void mv_bf16_rows_avx512(const std::uint16_t* w, std::size_t cols, const float* xs, int full,
                                                int tail, const float* bias, float* out) {
    __m512 e[R], o[R];
    for (int i = 0; i < R; ++i) e[i] = o[i] = _mm512_setzero_ps();
    const std::uint16_t* wr[R];
    for (int i = 0; i < R; ++i) wr[i] = w + i * cols;
    for (int m = 0; m < full; ++m) {
        bf16_chunk_avx512<R>(wr, xs + 32 * static_cast<std::size_t>(m), e, o);
        for (int i = 0; i < R; ++i) wr[i] += 32;
    }
    if (tail != 0) {
        std::uint16_t tw[R][32] = {};
        const std::uint16_t* tr[R];
        for (int i = 0; i < R; ++i) {
            std::copy_n(wr[i], tail, tw[i]);
            tr[i] = tw[i];
        }
        bf16_chunk_avx512<R>(tr, xs + 32 * static_cast<std::size_t>(full), e, o);
    }
    for (int i = 0; i < R; ++i) {
        const __m512 v = _mm512_add_ps(e[i], o[i]);
        const __m256 lo = _mm512_castps512_ps256(v);
        const __m256 up = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1));
        out[i] = hsum8(_mm256_add_ps(lo, up)) + (bias ? bias[i] : 0.0f);
    }
}

CYPHA_NN_AVX512 void matvec_bf16_avx512(const std::uint16_t* w, const float* x, const float* bias, float* out,
                                        int rows, int cols) {
    const float* xs = split_x(x, cols);
    const std::size_t c = static_cast<std::size_t>(cols);
    int r = 0;
    for (; r + 4 <= rows; r += 4)
        mv_bf16_rows_avx512<4>(w + r * c, c, xs, cols / 32, cols % 32, bias ? bias + r : nullptr, out + r);
    for (; r < rows; ++r)
        mv_bf16_rows_avx512<1>(w + r * c, c, xs, cols / 32, cols % 32, bias ? bias + r : nullptr, out + r);
}

template <int R>
CYPHA_NN_AVX512 inline void mv_f32_rows_avx512(const float* w, std::size_t cols, const float* x, int full, int tail,
                                               const float* bias, float* out) {
    __m512 a0[R], a1[R];
    for (int i = 0; i < R; ++i) a0[i] = a1[i] = _mm512_setzero_ps();
    float tw[R][32], tx[32];
    for (int m = 0; m <= full; ++m) {
        const float* wr[R];
        const float* xm = x + 32 * static_cast<std::size_t>(m);
        if (m == full) {
            if (tail == 0) break;
            std::fill(tx, tx + 32, 0.0f);
            std::copy_n(xm, tail, tx);
            xm = tx;
            for (int i = 0; i < R; ++i) {
                std::fill(tw[i], tw[i] + 32, 0.0f);
                std::copy_n(w + i * cols + 32 * static_cast<std::size_t>(m), tail, tw[i]);
                wr[i] = tw[i];
            }
        } else {
            for (int i = 0; i < R; ++i) wr[i] = w + i * cols + 32 * static_cast<std::size_t>(m);
        }
        const __m512 x0 = _mm512_loadu_ps(xm), x1 = _mm512_loadu_ps(xm + 16);
        for (int i = 0; i < R; ++i) {
            a0[i] = _mm512_fmadd_ps(_mm512_loadu_ps(wr[i]), x0, a0[i]);
            a1[i] = _mm512_fmadd_ps(_mm512_loadu_ps(wr[i] + 16), x1, a1[i]);
        }
    }
    for (int i = 0; i < R; ++i) {
        const __m512 v = _mm512_add_ps(a0[i], a1[i]);
        const __m256 lo = _mm512_castps512_ps256(v);
        const __m256 up = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1));
        out[i] = hsum8(_mm256_add_ps(lo, up)) + bias[i];
    }
}

CYPHA_NN_AVX512 void matvec_f32_avx512(const float* w, const float* x, const float* bias, float* out, int rows,
                                       int cols) {
    const std::size_t c = static_cast<std::size_t>(cols);
    int r = 0;
    for (; r + 4 <= rows; r += 4) mv_f32_rows_avx512<4>(w + r * c, c, x, cols / 32, cols % 32, bias + r, out + r);
    for (; r < rows; ++r) mv_f32_rows_avx512<1>(w + r * c, c, x, cols / 32, cols % 32, bias + r, out + r);
}

CYPHA_NN_AVX512 void sub_scaled_avx512(float* y, float a, const float* x, int n) {
    const __m512 av = _mm512_set1_ps(a);
    int k = 0;
    for (; k + 16 <= n; k += 16)
        _mm512_storeu_ps(y + k, _mm512_fnmadd_ps(av, _mm512_loadu_ps(x + k), _mm512_loadu_ps(y + k)));
    for (; k < n; ++k) y[k] = std::fma(-a, x[k], y[k]);
}

template <int B>
CYPHA_NN_AVX512 inline void scores_avx512(const float* q, const float* Kh, std::size_t ctx, int hd, int t,
                                          float scale, float* s) {
    __m512 a[B];
    for (int b = 0; b < B; ++b) a[b] = _mm512_setzero_ps();
    for (int k = 0; k < hd; ++k) {
        const __m512 qk = _mm512_set1_ps(q[k]);
        const float* row = Kh + k * ctx + t;
        for (int b = 0; b < B; ++b) a[b] = _mm512_fmadd_ps(qk, _mm512_loadu_ps(row + 16 * b), a[b]);
    }
    for (int b = 0; b < B; ++b) _mm512_storeu_ps(s + t + 16 * b, _mm512_mul_ps(a[b], _mm512_set1_ps(scale)));
}

template <int B>
CYPHA_NN_AVX512 inline void values_avx512(const float* sc, const float* V, int n, int d, int hd, std::size_t ctx,
                                          int k0, float* att) {
    __m512 a[B];
    const float* w[B];
    for (int b = 0; b < B; ++b) {
        a[b] = _mm512_setzero_ps();
        w[b] = sc + static_cast<std::size_t>((k0 + 16 * b) / hd) * ctx;
    }
    for (int t = 0; t < n; ++t) {
        const float* v = V + static_cast<std::size_t>(t) * d + k0;
        for (int b = 0; b < B; ++b)
            a[b] = _mm512_fmadd_ps(_mm512_set1_ps(w[b][t]), _mm512_loadu_ps(v + 16 * b), a[b]);
    }
    for (int b = 0; b < B; ++b) _mm512_storeu_ps(att + k0 + 16 * b, a[b]);
}

CYPHA_NN_AVX512 void attend_avx512(const float* q, const float* Kt, const float* V, int n, int d, int heads, int ctx,
                                   float scale, float* sc, float* att) {
    const int hd = d / heads;
    const std::size_t c = static_cast<std::size_t>(ctx);
    for (int hh = 0; hh < heads; ++hh) {
        const float* qh = q + hh * hd;
        const float* Kh = Kt + static_cast<std::size_t>(hh) * hd * c;
        float* s = sc + static_cast<std::size_t>(hh) * c;
        int t = 0;
        for (; t + 128 <= n; t += 128) scores_avx512<8>(qh, Kh, c, hd, t, scale, s);
        for (; t + 16 <= n; t += 16) scores_avx512<1>(qh, Kh, c, hd, t, scale, s);
        for (; t < n; ++t) s[t] = score_one(qh, Kh, c, hd, t, scale);
    }
    softmax_heads(sc, heads, ctx, n);
    if (hd % 16 != 0) return values_one(sc, V, n, d, hd, c, att);
    int k0 = 0;
    for (; k0 + 128 <= d; k0 += 128) values_avx512<8>(sc, V, n, d, hd, c, k0, att);
    for (; k0 < d; k0 += 16) values_avx512<1>(sc, V, n, d, hd, c, k0, att);
}

#endif  // CYPHA_NN_X86

struct Kernels {
    const char* name;
    void (*matvec_bf16)(const std::uint16_t* w, const float* x, const float* bias, float* out, int rows, int cols);
    void (*matvec_f32)(const float* w, const float* x, const float* bias, float* out, int rows, int cols);
    void (*sub_scaled)(float* y, float a, const float* x, int n);
    void (*attend)(const float* q, const float* Kt, const float* V, int n, int d, int heads, int ctx, float scale,
                   float* sc, float* att);
};

constexpr Kernels kKernels[] = {
    {"portable", matvec_bf16_portable, matvec_f32_portable, sub_scaled_portable, attend_portable},
#if CYPHA_NN_X86
    {"avx2", matvec_bf16_avx2, matvec_f32_avx2, sub_scaled_avx2, attend_avx2},
    {"avx512", matvec_bf16_avx512, matvec_f32_avx512, sub_scaled_avx512, attend_avx512},
#endif
};
constexpr int kKernelCount = static_cast<int>(sizeof(kKernels) / sizeof(kKernels[0]));

bool kernel_supported(int k) {
#if CYPHA_NN_X86
    __builtin_cpu_init();
    if (k == 1) return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    if (k == 2) return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
    return k == 0;
}

int kernel_index(const std::string& name) {
    for (int k = 0; k < kKernelCount; ++k)
        if (name == kKernels[k].name) return k;
    return -1;
}

// The best kernel the CPU runs, capped by CYPHA_NN_KERNEL (auto | avx512 |
// avx2 | portable; an ISA the CPU lacks falls back to the next one down).
int initial_kernel() {
    int cap = kKernelCount - 1;
    if (const char* env = std::getenv("CYPHA_NN_KERNEL"); env != nullptr && std::strcmp(env, "auto") != 0) {
        if (std::strcmp(env, "avx512") == 0 || std::strcmp(env, "avx2") == 0 || std::strcmp(env, "portable") == 0) {
            cap = std::min(cap, std::strcmp(env, "portable") == 0 ? 0 : std::strcmp(env, "avx2") == 0 ? 1 : 2);
        } else {
            throw std::invalid_argument(std::string("CYPHA_NN_KERNEL: expected auto, avx512, avx2 or portable, got ") +
                                        env);
        }
    }
    while (cap > 0 && !kernel_supported(cap)) --cap;
    return cap;
}

std::atomic<int>& kernel_slot() {
    static std::atomic<int> k{initial_kernel()};
    return k;
}

const Kernels& kernels() { return kKernels[kernel_slot().load(std::memory_order_relaxed)]; }

std::uint16_t to_bf16(float f) {  // round to nearest even
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    u += 0x7FFFu + ((u >> 16) & 1u);
    return static_cast<std::uint16_t>(u >> 16);
}

std::vector<std::uint16_t> to_bf16(const std::vector<float>& v) {
    std::vector<std::uint16_t> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = to_bf16(v[i]);
    return out;
}

float sigmoid(float v) { return 1.0f / (1.0f + std::exp(-v)); }

template <class T>
void read_array(std::istream& is, std::vector<T>& v, std::size_t n) {
    v.resize(n);
    is.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(T)));
}

void log_softmax(const float* logits, std::array<double, 256>& out) {
    const float mx = *std::max_element(logits, logits + 256);
    double z = 0.0;
    for (int b = 0; b < 256; ++b) z += std::exp(static_cast<double>(logits[b] - mx));
    const double lz = std::log(z) + mx;
    for (int b = 0; b < 256; ++b) out[static_cast<std::size_t>(b)] = logits[b] - lz;
}

void layer_norm(const float* x, const float* w, const float* b, float* out, int d) {
    double mean = 0.0, var = 0.0;
    for (int k = 0; k < d; ++k) mean += x[k];
    mean /= d;
    for (int k = 0; k < d; ++k) var += (x[k] - mean) * (x[k] - mean);
    var /= d;
    const double inv = 1.0 / std::sqrt(var + 1e-5);
    for (int k = 0; k < d; ++k) out[k] = static_cast<float>((x[k] - mean) * inv) * w[k] + b[k];
}

}  // namespace

std::string neural_kernel() { return kernels().name; }

bool set_neural_kernel(const std::string& name) {
    const int k = kernel_index(name);
    if (k < 0 || !kernel_supported(k)) return false;
    kernel_slot().store(k, std::memory_order_relaxed);
    return true;
}

std::shared_ptr<const ByteNeuralExpert> load_neural_expert(const std::string& path) {
    char magic[4] = {};
    {
        std::ifstream is(path, std::ios::binary);
        if (!is) throw std::runtime_error("neural expert: cannot open " + path);
        is.read(magic, 4);
    }
    if (std::memcmp(magic, "BLM1", 4) == 0) return ByteLstmExpert::load(path);
    if (std::memcmp(magic, "BGT1", 4) == 0) return ByteGptExpert::load(path);
    throw std::runtime_error("neural expert: unknown file " + path);
}

void ByteNeuralExpert::adapt_output(State& s, std::uint8_t byte, int d) {
    if (s.adapt_lr <= 0.0f || s.ow.empty() || s.hlast.size() != static_cast<std::size_t>(d)) return;
    // d(-log p_y)/d logit_b = p_b - [b == y]; logits = ow h + ob.
    const auto sub_scaled = kernels().sub_scaled;
    for (int b = 0; b < 256; ++b) {
        const float g = static_cast<float>(std::exp(s.log_p[static_cast<std::size_t>(b)])) - (b == byte ? 1.0f : 0.0f);
        const float step = s.adapt_lr * g;
        sub_scaled(&s.ow[static_cast<std::size_t>(b) * d], step, s.hlast.data(), d);
        s.ob[static_cast<std::size_t>(b)] -= step;
    }
}

void ByteNeuralExpert::output_logits(State& s, const float* h, int d, const std::uint16_t* w_bf16,
                                     const float* bias) {
    float logits[256];
    if (!s.ow.empty()) {
        kernels().matvec_f32(s.ow.data(), h, s.ob.data(), logits, 256, d);
        s.hlast.assign(h, h + d);
    } else {
        kernels().matvec_bf16(w_bf16, h, bias, logits, 256, d);
    }
    log_softmax(logits, s.log_p);
}

// ---------------------------------------------------------------- LSTM

std::shared_ptr<const ByteLstmExpert> ByteLstmExpert::load(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) throw std::runtime_error("ByteLstmExpert: cannot open " + path);
    char magic[4];
    std::uint32_t hdr[3];
    is.read(magic, 4);
    is.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    if (!is || std::memcmp(magic, "BLM1", 4) != 0) throw std::runtime_error("ByteLstmExpert: bad file " + path);
    auto m = std::make_shared<ByteLstmExpert>();
    m->layers_ = static_cast<int>(hdr[0]);
    m->d_ = static_cast<int>(hdr[1]);
    m->emb_ = static_cast<int>(hdr[2]);
    if (m->layers_ < 1 || m->layers_ > 8 || m->d_ < 1 || m->d_ > 8192 || m->emb_ < 1 || m->emb_ > 8192) {
        throw std::runtime_error("ByteLstmExpert: bad shape in " + path);
    }
    const std::size_t d = static_cast<std::size_t>(m->d_);
    read_array(is, m->emb_w_, 256 * static_cast<std::size_t>(m->emb_));
    for (int l = 0; l < m->layers_; ++l) {
        const std::size_t in = l == 0 ? static_cast<std::size_t>(m->emb_) : d;
        std::vector<float> wih, whh, bih, bhh;
        read_array(is, wih, 4 * d * in);
        read_array(is, whh, 4 * d * d);
        read_array(is, bih, 4 * d);
        read_array(is, bhh, 4 * d);
        std::vector<float> w(4 * d * (in + d));
        for (std::size_t r = 0; r < 4 * d; ++r) {
            std::copy_n(&wih[r * in], in, &w[r * (in + d)]);
            std::copy_n(&whh[r * d], d, &w[r * (in + d) + in]);
        }
        for (std::size_t r = 0; r < 4 * d; ++r) bih[r] += bhh[r];
        m->w_.push_back(to_bf16(w));
        m->b_.push_back(std::move(bih));
    }
    std::vector<float> ow;
    read_array(is, ow, 256 * d);
    m->out_w_ = to_bf16(ow);
    read_array(is, m->out_b_, 256);
    if (!is) throw std::runtime_error("ByteLstmExpert: truncated " + path);
    return m;
}

std::size_t ByteLstmExpert::parameter_count() const {
    std::size_t n = emb_w_.size() + out_w_.size() + out_b_.size();
    for (std::size_t l = 0; l < w_.size(); ++l) n += w_[l].size() + b_[l].size();
    return n;
}

ByteNeuralExpert::State ByteLstmExpert::initial_state() const {
    State s;
    s.a.assign(static_cast<std::size_t>(layers_) * d_, 0.0f);
    s.b.assign(static_cast<std::size_t>(layers_) * d_, 0.0f);
    s.work.assign(work_size_(), 0.0f);
    step(s, 0);
    return s;
}

void ByteLstmExpert::init_adaptation(State& s) const {
    s.ow.resize(out_w_.size());
    for (std::size_t i = 0; i < out_w_.size(); ++i) {
        const std::uint32_t u = static_cast<std::uint32_t>(out_w_[i]) << 16;
        std::memcpy(&s.ow[i], &u, sizeof(float));
    }
    s.ob = out_b_;
    s.hlast.assign(s.a.end() - d_, s.a.end());  // top layer h (log_p came from it)
}

std::size_t ByteLstmExpert::work_size_() const {
    return static_cast<std::size_t>(std::max(emb_, d_)) + 5 * static_cast<std::size_t>(d_);  // [x | h], gates
}

void ByteLstmExpert::step(State& s, std::uint8_t byte) const {
    adapt_output(s, byte, d_);
    const int d = d_;
    if (s.work.size() < work_size_()) s.work.resize(work_size_());
    float* xin = s.work.data();  // [layer input | this layer's h]
    float* gates = xin + std::max(emb_, d) + d;
    const auto matvec = kernels().matvec_bf16;
    int in = emb_;
    std::copy_n(&emb_w_[static_cast<std::size_t>(byte) * emb_], emb_, xin);
    for (int l = 0; l < layers_; ++l) {
        float* h = &s.a[static_cast<std::size_t>(l) * d];
        float* c = &s.b[static_cast<std::size_t>(l) * d];
        std::copy_n(h, d, xin + in);
        matvec(w_[l].data(), xin, b_[l].data(), gates, 4 * d, in + d);
        for (int k = 0; k < d; ++k) {
            const float i = sigmoid(gates[k]);
            const float f = sigmoid(gates[d + k]);
            const float g = std::tanh(gates[2 * d + k]);
            const float o = sigmoid(gates[3 * d + k]);
            c[k] = f * c[k] + i * g;
            h[k] = o * std::tanh(c[k]);
        }
        std::copy_n(h, d, xin);  // the next layer's input
        in = d;
    }
    output_logits(s, xin, d, out_w_.data(), out_b_.data());
}

// ---------------------------------------------------------------- Transformer

std::shared_ptr<const ByteGptExpert> ByteGptExpert::load(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) throw std::runtime_error("ByteGptExpert: cannot open " + path);
    char magic[4];
    std::uint32_t hdr[4];
    is.read(magic, 4);
    is.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    if (!is || std::memcmp(magic, "BGT1", 4) != 0) throw std::runtime_error("ByteGptExpert: bad file " + path);
    auto m = std::make_shared<ByteGptExpert>();
    m->layers_ = static_cast<int>(hdr[0]);
    m->d_ = static_cast<int>(hdr[1]);
    m->heads_ = static_cast<int>(hdr[2]);
    m->ctx_ = static_cast<int>(hdr[3]);
    if (m->layers_ < 1 || m->layers_ > 64 || m->d_ < 8 || m->d_ > 8192 || m->heads_ < 1 || m->d_ % m->heads_ != 0 ||
        m->ctx_ < 4 || m->ctx_ > 65536) {
        throw std::runtime_error("ByteGptExpert: bad shape in " + path);
    }
    const std::size_t d = static_cast<std::size_t>(m->d_);
    read_array(is, m->emb_, 256 * d);
    m->emb_bf_ = to_bf16(m->emb_);
    read_array(is, m->pos_, static_cast<std::size_t>(m->ctx_) * d);
    for (int l = 0; l < m->layers_; ++l) {
        Layer L;
        std::vector<float> w;
        read_array(is, L.ln1_w, d);
        read_array(is, L.ln1_b, d);
        read_array(is, w, 3 * d * d);
        L.qkv = to_bf16(w);
        read_array(is, w, d * d);
        L.proj = to_bf16(w);
        read_array(is, L.ln2_w, d);
        read_array(is, L.ln2_b, d);
        read_array(is, w, 4 * d * d);
        L.fc = to_bf16(w);
        read_array(is, w, 4 * d * d);
        L.fc2 = to_bf16(w);
        m->L_.push_back(std::move(L));
    }
    read_array(is, m->lnf_w_, d);
    read_array(is, m->lnf_b_, d);
    if (!is) throw std::runtime_error("ByteGptExpert: truncated " + path);
    return m;
}

std::size_t ByteGptExpert::parameter_count() const {
    std::size_t n = emb_.size() + pos_.size() + lnf_w_.size() + lnf_b_.size();
    for (const auto& L : L_)
        n += L.ln1_w.size() * 4 + L.qkv.size() + L.proj.size() + L.fc.size() + L.fc2.size();
    return n;
}

ByteNeuralExpert::State ByteGptExpert::initial_state() const {
    State s;
    const std::size_t cache = static_cast<std::size_t>(layers_) * ctx_ * d_;
    s.a.assign(cache, 0.0f);
    s.b.assign(cache, 0.0f);
    s.hist.reserve(static_cast<std::size_t>(ctx_));
    s.work.assign(work_size_(), 0.0f);
    s.pos = 0;
    step(s, 0);
    return s;
}

void ByteGptExpert::init_adaptation(State& s) const {
    s.ow = emb_;  // tied output = token embedding
    s.ob.assign(256, 0.0f);
    s.hlast.clear();  // filled by the next step's logits
}

std::size_t ByteGptExpert::work_size_() const {
    // x, h, qkv, att, y, f; attention scores [heads][ctx]
    return 11 * static_cast<std::size_t>(d_) + static_cast<std::size_t>(heads_) * ctx_;
}

void ByteGptExpert::step(State& s, std::uint8_t byte) const {
    adapt_output(s, byte, d_);
    if (s.pos >= ctx_) {
        // Window full: re-prime on the last ctx/2 bytes read (learned absolute
        // positions, so the cache cannot slide).
        const std::size_t keep = static_cast<std::size_t>(ctx_ / 2) - 1;
        s.hist.erase(s.hist.begin(), s.hist.end() - static_cast<std::ptrdiff_t>(keep));
        s.pos = 0;
        for (std::size_t i = 0; i < keep; ++i) forward_(s, s.hist[i], false);
    }
    s.hist.push_back(byte);
    forward_(s, byte, true);
}

void ByteGptExpert::forward_(State& s, std::uint8_t byte, bool want_logits) const {
    const int d = d_, hd = d_ / heads_, p = s.pos;
    const std::size_t du = static_cast<std::size_t>(d), cu = static_cast<std::size_t>(ctx_);
    if (s.work.size() < work_size_()) s.work.resize(work_size_());
    float* x = s.work.data();
    float* h = x + du;
    float* qkv = h + du;
    float* att = qkv + 3 * du;
    float* y = att + du;
    float* f = y + du;
    float* sc = f + 4 * du;
    const Kernels& kn = kernels();
    for (int k = 0; k < d; ++k) x[k] = emb_[static_cast<std::size_t>(byte) * d + k] + pos_[static_cast<std::size_t>(p) * d + k];
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    for (int l = 0; l < layers_; ++l) {
        const Layer& L = L_[static_cast<std::size_t>(l)];
        layer_norm(x, L.ln1_w.data(), L.ln1_b.data(), h, d);
        kn.matvec_bf16(L.qkv.data(), h, nullptr, qkv, 3 * d, d);
        float* Kt = &s.a[static_cast<std::size_t>(l) * cu * du];  // [d][ctx]
        float* V = &s.b[static_cast<std::size_t>(l) * cu * du];   // [ctx][d]
        for (int k = 0; k < d; ++k) Kt[static_cast<std::size_t>(k) * cu + p] = qkv[du + k];
        std::copy_n(qkv + 2 * du, d, V + static_cast<std::size_t>(p) * du);
        kn.attend(qkv, Kt, V, p + 1, d, heads_, ctx_, scale, sc, att);
        kn.matvec_bf16(L.proj.data(), att, nullptr, y, d, d);
        for (int k = 0; k < d; ++k) x[k] += y[k];
        layer_norm(x, L.ln2_w.data(), L.ln2_b.data(), h, d);
        kn.matvec_bf16(L.fc.data(), h, nullptr, f, 4 * d, d);
        for (int k = 0; k < 4 * d; ++k) f[k] = 0.5f * f[k] * (1.0f + std::erf(f[k] * 0.70710678f));  // exact GELU
        kn.matvec_bf16(L.fc2.data(), f, nullptr, y, d, 4 * d);
        for (int k = 0; k < d; ++k) x[k] += y[k];
    }
    s.pos = p + 1;
    if (!want_logits) return;
    layer_norm(x, lnf_w_.data(), lnf_b_.data(), h, d);
    output_logits(s, h, d, emb_bf_.data(), nullptr);
}

}  // namespace cypha::cyphalm
