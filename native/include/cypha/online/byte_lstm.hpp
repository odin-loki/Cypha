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
#include <random>
#include <string>
#include <vector>

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define CYPHA_LSTM_SSE2 1
#else
#define CYPHA_LSTM_SSE2 0
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
template <class T>
void put_vec(std::ostream& o, const std::vector<T>& v) {
    put(o, static_cast<std::uint64_t>(v.size()));
    o.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
}
template <class T>
void get_vec(std::istream& i, std::vector<T>& v) {
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
// then yield, then sleep briefly when idle for long, so an idle pool costs little.
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
template <int HH>
class ByteLSTMT {
public:
    static constexpr int H = HH, E = 16, X = E + H, kDefaultWindow = 20;

    explicit ByteLSTMT(std::uint64_t seed = 0, double lr = 0.003, int window = kDefaultWindow) : lr_(lr), win_(window) {
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> nd(0.0, 1.0);
        auto fill = [&](std::vector<double>& v, std::size_t n, double s) {
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
        cache_ = forward(32);  // the state after reading a space predicts the first byte
    }

    const std::array<double, H>& h() const { return h_; }

    // Worker threads for this LSTM (1: none). The output is the same for any count.
    void set_threads(int n) { pool_.set(n); }
    int threads() const { return pool_.want(); }

    // The distribution of the next byte from the state that has read every byte so far.
    void dist(double* p) const {
        double* z = zs_;
        auto job = [&](int t, int n) {
            const int y0 = 256 * t / n, y1 = 256 * (t + 1) / n;
            logits(h_.data(), z, y0, y1);
        };
        pool_.run(job);
        double mx = -1e300;
        for (int y = 0; y < 256; ++y) {
            p[y] = z[y];
            mx = std::max(mx, z[y]);
        }
        double s = 0;
        for (int y = 0; y < 256; ++y) s += (p[y] = std::exp(p[y] - mx));
        for (int y = 0; y < 256; ++y) p[y] /= s;
        if (!moved_) {  // the softmax of cache_.h: BPTT reuses it
            std::copy(p, p + 256, pc_.begin());
            pv_ = true;
        }
    }

    // y followed the bytes read so far: learn that (when learn), then read y.
    void step(std::uint8_t y, bool learn) {
        if (learn) {
            cache_.y = y;
            cache_.has_p = pv_;
            if (pv_) cache_.p = pc_;
            buf_.push_back(cache_);
            if (static_cast<int>(buf_.size()) >= win_) bptt();
        }
        cache_ = forward(y);
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
        buf_.clear();
        cache_ = forward(32);  // a fresh read of a space after the stored state
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
        moved_ = true;  // h_ may now differ from cache_.h: dist() no longer feeds BPTT until the next forward
        pv_ = false;
    }

private:
    struct Step {
        std::uint8_t x = 0, y = 0;
        bool has_p = false;
        std::array<double, X> xh;
        std::array<double, H> i, f, o, g, c_prev, c, tc, h;
        std::array<double, 256> p;  // the softmax of h, when dist() computed it
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
        static int default_threads() {
            if (const char* e = std::getenv("CYPHA_LSTM_THREADS")) return std::clamp(std::atoi(e), 1, 64);
            return H >= 128 ? 4 : 1;
        }
        int n_;
        mutable std::unique_ptr<LstmPool> p_;
    };

    // z[y] = bo[y] + sum_k Wo[y][k] h[k] for y in [y0, y1), each in order k = 0, 1, ...
    void logits(const double* h, double* z, int y0, int y1) const { lstm_rows(Wo_.data(), H, h, bo_.data(), z, y0, y1); }

    Step forward(std::uint8_t xb) {
        Step s;
        s.x = xb;
        for (int k = 0; k < E; ++k) s.xh[k] = Em_[static_cast<std::size_t>(xb) * E + k];
        for (int k = 0; k < H; ++k) s.xh[E + k] = h_[k];
        double* a = as_;
        auto rows = [&](int t, int n) {  // the 4H gate rows, a contiguous range per thread
            lstm_rows(W_.data(), X, s.xh.data(), b_.data(), a, 4 * H * t / n, 4 * H * (t + 1) / n);
        };
        pool_.run(rows);
        auto gates = [&](int t, int n) {
            for (int k = H * t / n; k < H * (t + 1) / n; ++k) {
                s.i[k] = 1 / (1 + std::exp(-a[k]));
                s.f[k] = 1 / (1 + std::exp(-a[H + k]));
                s.o[k] = 1 / (1 + std::exp(-a[2 * H + k]));
                s.g[k] = std::tanh(a[3 * H + k]);
                s.c_prev[k] = c_[k];
                s.c[k] = s.f[k] * c_[k] + s.i[k] * s.g[k];
                s.tc[k] = std::tanh(s.c[k]);
                s.h[k] = s.o[k] * s.tc[k];
            }
        };
        pool_.run(gates);
        h_ = s.h;
        c_ = s.c;
        moved_ = false;
        pv_ = false;
        return s;
    }

    // Truncated BPTT over buf_, then one Adam step. The backward pass keeps each step's gate and output errors (da, p);
    // the weight gradients are then summed once per window, each element over the steps in the same order as before
    // (last step first), fused with Adam. Only dh and dxh, which feed the next step back, are
    // computed step by step.
    void bptt() {
        const int T = static_cast<int>(buf_.size());
        das_.resize(static_cast<std::size_t>(T) * 4 * H);
        ps_.resize(static_cast<std::size_t>(T) * 256);
        gE_.assign(Em_.size(), 0.0);
        std::array<double, H> dh_next{}, dc_next{};
        std::array<double, X> dxh;
        for (int it = T - 1; it >= 0; --it) {
            const Step& s = buf_[static_cast<std::size_t>(it)];
            double* p = &ps_[static_cast<std::size_t>(it) * 256];
            double* da = &das_[static_cast<std::size_t>(it) * 4 * H];
            if (s.has_p) {
                std::copy(s.p.begin(), s.p.end(), p);
            } else {
                double* z = zs_;
                auto lj = [&](int t, int n) { logits(s.h.data(), z, 256 * t / n, 256 * (t + 1) / n); };
                pool_.run(lj);
                double mx = -1e300;
                for (int y = 0; y < 256; ++y) {
                    p[y] = z[y];
                    mx = std::max(mx, z[y]);
                }
                double sum = 0;
                for (int y = 0; y < 256; ++y) sum += (p[y] = std::exp(p[y] - mx));
                for (int y = 0; y < 256; ++y) p[y] /= sum;
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
        auto adam = [&](int q, std::size_t j0, std::size_t len, double* CYPHA_RESTRICT w, const double* CYPHA_RESTRICT gs) {
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
                    const double* xh = buf_[static_cast<std::size_t>(it)].xh.data();
                    gb += d;
                    for (int j = 0; j < X; ++j) g[j] += d * xh[j];
                }
                adam(1, static_cast<std::size_t>(r) * X, X, &W_[static_cast<std::size_t>(r) * X], g);
                adam(2, static_cast<std::size_t>(r), 1, &b_[static_cast<std::size_t>(r)], &gb);
            }
            // Output rows: Wo, bo.
            const int y0 = 256 * t / nt, y1 = 256 * (t + 1) / nt;
            for (int y = y0; y < y1; ++y) {
                for (int k = 0; k < H; ++k) g[k] = 0;
                double gb = 0;
                for (int it = T - 1; it >= 0; --it) {
                    const double py = ps_[static_cast<std::size_t>(it) * 256 + y];
                    const double* h = buf_[static_cast<std::size_t>(it)].h.data();
                    gb += py;
                    for (int k = 0; k < H; ++k) g[k] += py * h[k];
                }
                adam(3, static_cast<std::size_t>(y) * H, H, &Wo_[static_cast<std::size_t>(y) * H], g);
                adam(4, static_cast<std::size_t>(y), 1, &bo_[static_cast<std::size_t>(y)], &gb);
            }
            // The embedding.
            const std::size_t ne = Em_.size(), e0 = ne * t / nt, e1 = ne * (t + 1) / nt;
            adam(0, e0, e1 - e0, Em_.data() + e0, gE_.data() + e0);
        };
        pool_.run(upd);
        buf_.clear();
    }

    double lr_;
    int win_;
    int t_ = 0;
    std::vector<double> Em_, W_, b_, Wo_, bo_;
    std::vector<std::vector<double>> m_, v_;
    std::vector<double> das_, ps_, gE_;     // BPTT's per-step gate and output errors; the embedding's gradient
    std::array<double, H> h_, c_;
    Step cache_;
    std::vector<Step> buf_;
    mutable double zs_[256];                // logits scratch
    double as_[4 * H];                      // gate pre-activations scratch
    mutable std::array<double, 256> pc_;    // dist()'s softmax of cache_.h
    mutable bool pv_ = false;
    bool moved_ = false;                    // restore() moved h_ away from cache_.h
    PoolSlot pool_;
};

using ByteLSTM = ByteLSTMT<64>;  // the LSTM of §13.18: hidden 64, window 20

}  // namespace cypha::online
