// A bitwise gated linear network with a byte LSTM inside it (docs/GENERATION_PLAN.md §13.13 and §13.15; Odin Loch,
// 2026-11-03). The C++ form of bench/fuzzy/glnb.py, learning online on every part.
//
// Every byte is 8 binary predictions, most significant bit first, with the partial byte c0 as PAQ keeps it.
//   - Inputs: hashed bit counts of orders 0..5 (the bytes before, the order, the partial byte), pretrained on CyphaLM's
//     training text and updated online; the LSTM's probability of the bit (from its byte distribution); a bias.
//   - The LSTM: a byte LSTM (embedding 16, hidden 64, softmax over 256) learning online by truncated BPTT (window 20)
//     and Adam. Its state is part of every neuron's gate, and it predicts each byte from the state that has read every
//     byte before it.
//   - Layer 1: N neurons, each a gated geometric mixer whose weights are chosen per bit position by hs half-spaces of
//     the side vector (LSTM state, inputs, and the extra values: the mixer's feedback). Layer 2: 4 neurons (2
//     half-spaces). A final neuron per bit position, then an APM (33 points per partial byte), averaged with it.
//   - Every neuron learns online from its own log loss (step 0.002); the APM steps 0.02.
// extra > 0 adds side values the caller sets before each byte (set_extra): with the content mixer's recent loss and
// the GLN's loss against it, the GLN reads the mixer and the mixer reads the GLN, one recurrent loop (§13.15).
// Counts are fixed point (x128, 16 bits); glnb.py keeps floats. Half-spaces come from mt19937_64, not numpy, so the
// two agree in kind, not bit for bit.
// Rule 10 (PLAN): an expert inside CyphaLM. It answers nothing.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

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

// A byte LSTM: embedding 16, hidden 64, softmax over 256; learns online by truncated BPTT (window 20) and Adam.
class ByteLSTM {
public:
    static constexpr int H = 64, E = 16, X = E + H, T = 20;

    explicit ByteLSTM(std::uint64_t seed = 0, double lr = 0.003) : lr_(lr) {
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

    // The distribution of the next byte from the state that has read every byte so far.
    void dist(double* p) const {
        double mx = -1e300;
        for (int y = 0; y < 256; ++y) {
            double z = bo_[y];
            for (int k = 0; k < H; ++k) z += Wo_[static_cast<std::size_t>(y) * H + k] * h_[k];
            p[y] = z;
            mx = std::max(mx, z);
        }
        double s = 0;
        for (int y = 0; y < 256; ++y) s += (p[y] = std::exp(p[y] - mx));
        for (int y = 0; y < 256; ++y) p[y] /= s;
    }

    // y followed the bytes read so far: learn that (when learn), then read y.
    void step(std::uint8_t y, bool learn) {
        if (learn) {
            cache_.y = y;
            buf_.push_back(cache_);
            if (static_cast<int>(buf_.size()) >= T) bptt();
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
    State save() const { return {h_, c_}; }
    // Restores the recurrent state (for writing: no learning happens between save and restore).
    void restore(const State& s) {
        h_ = s.h;
        c_ = s.c;
    }

private:
    struct Step {
        std::uint8_t x = 0, y = 0;
        std::array<double, X> xh;
        std::array<double, H> i, f, o, g, c_prev, c, h;
    };

    Step forward(std::uint8_t xb) {
        Step s;
        s.x = xb;
        for (int k = 0; k < E; ++k) s.xh[k] = Em_[static_cast<std::size_t>(xb) * E + k];
        for (int k = 0; k < H; ++k) s.xh[E + k] = h_[k];
        for (int r = 0; r < 4 * H; ++r) {
            double a = b_[r];
            const double* w = &W_[static_cast<std::size_t>(r) * X];
            for (int k = 0; k < X; ++k) a += w[k] * s.xh[k];
            const int q = r / H, k = r % H;
            if (q == 0) s.i[k] = 1 / (1 + std::exp(-a));
            else if (q == 1) s.f[k] = 1 / (1 + std::exp(-a));
            else if (q == 2) s.o[k] = 1 / (1 + std::exp(-a));
            else s.g[k] = std::tanh(a);
        }
        for (int k = 0; k < H; ++k) {
            s.c_prev[k] = c_[k];
            s.c[k] = s.f[k] * c_[k] + s.i[k] * s.g[k];
            s.h[k] = s.o[k] * std::tanh(s.c[k]);
        }
        h_ = s.h;
        c_ = s.c;
        return s;
    }

    void bptt() {
        std::vector<std::vector<double>> G;
        for (auto* p : {&Em_, &W_, &b_, &Wo_, &bo_}) G.emplace_back(p->size(), 0.0);
        auto &gE = G[0], &gW = G[1], &gb = G[2], &gWo = G[3], &gbo = G[4];
        std::array<double, H> dh_next{}, dc_next{};
        double p[256];
        for (auto it = buf_.rbegin(); it != buf_.rend(); ++it) {
            const Step& s = *it;
            double mx = -1e300;
            for (int y = 0; y < 256; ++y) {
                double z = bo_[y];
                for (int k = 0; k < H; ++k) z += Wo_[static_cast<std::size_t>(y) * H + k] * s.h[k];
                p[y] = z;
                mx = std::max(mx, z);
            }
            double sum = 0;
            for (int y = 0; y < 256; ++y) sum += (p[y] = std::exp(p[y] - mx));
            for (int y = 0; y < 256; ++y) p[y] /= sum;
            p[s.y] -= 1;
            std::array<double, H> dh = dh_next;
            for (int y = 0; y < 256; ++y) {
                gbo[y] += p[y];
                for (int k = 0; k < H; ++k) {
                    gWo[static_cast<std::size_t>(y) * H + k] += p[y] * s.h[k];
                    dh[k] += Wo_[static_cast<std::size_t>(y) * H + k] * p[y];
                }
            }
            std::array<double, 4 * H> da;
            for (int k = 0; k < H; ++k) {
                const double tc = std::tanh(s.c[k]);
                const double d_o = dh[k] * tc * s.o[k] * (1 - s.o[k]);
                const double dc = dh[k] * s.o[k] * (1 - tc * tc) + dc_next[k];
                da[k] = dc * s.g[k] * s.i[k] * (1 - s.i[k]);
                da[H + k] = dc * s.c_prev[k] * s.f[k] * (1 - s.f[k]);
                da[2 * H + k] = d_o;
                da[3 * H + k] = dc * s.i[k] * (1 - s.g[k] * s.g[k]);
                dc_next[k] = dc * s.f[k];
            }
            std::array<double, X> dxh{};
            for (int r = 0; r < 4 * H; ++r) {
                gb[r] += da[r];
                const double* w = &W_[static_cast<std::size_t>(r) * X];
                double* gw = &gW[static_cast<std::size_t>(r) * X];
                for (int k = 0; k < X; ++k) {
                    gw[k] += da[r] * s.xh[k];
                    dxh[k] += w[k] * da[r];
                }
            }
            for (int k = 0; k < E; ++k) gE[static_cast<std::size_t>(s.x) * E + k] += dxh[k];
            for (int k = 0; k < H; ++k) dh_next[k] = dxh[E + k];
        }
        ++t_;
        const double n = static_cast<double>(buf_.size());
        const double c1 = 1 - std::pow(0.9, t_), c2 = 1 - std::pow(0.999, t_);
        std::vector<double>* P[5] = {&Em_, &W_, &b_, &Wo_, &bo_};
        for (int q = 0; q < 5; ++q)
            for (std::size_t j = 0; j < P[q]->size(); ++j) {
                const double g = std::clamp(G[q][j] / n, -5.0, 5.0);
                m_[q][j] = 0.9 * m_[q][j] + 0.1 * g;
                v_[q][j] = 0.999 * v_[q][j] + 0.001 * g * g;
                (*P[q])[j] -= lr_ * (m_[q][j] / c1) / (std::sqrt(v_[q][j] / c2) + 1e-8);
            }
        buf_.clear();
    }

    double lr_;
    int t_ = 0;
    std::vector<double> Em_, W_, b_, Wo_, bo_;
    std::vector<std::vector<double>> m_, v_;
    std::array<double, H> h_, c_;
    Step cache_;
    std::vector<Step> buf_;
};

class BitwiseGLN {
public:
    struct Params {
        int neurons = 32;  // layer 1
        int hs = 4;        // half-spaces per layer-1 neuron: 2^hs weight sets per bit position
        int tb = 22;       // hash table bits per order
        bool lstm = true;
        int extra = 0;     // side values the caller sets (the mixer's feedback)
        std::uint64_t seed = 0;
    };
    static constexpr int kOrders = 6;

    explicit BitwiseGLN(const Params& p) : p_(p), lstm_(p.seed) {
        K_ = kOrders + (p.lstm ? 1 : 0) + 1;
        D_ = (p.lstm ? ByteLSTM::H : 0) + K_ + p.extra;
        mask_ = (std::uint64_t{1} << p.tb) - 1;
        for (auto& t : n1_) t.assign(std::size_t{1} << p.tb, 0);
        for (auto& t : n_) t.assign(std::size_t{1} << p.tb, 0);
        std::mt19937_64 rng(p.seed + 1);
        std::normal_distribution<double> nd(0.0, 1.0);
        H1_.resize(static_cast<std::size_t>(p.neurons) * p.hs * D_);
        for (auto& x : H1_) x = nd(rng);
        H2_.resize(static_cast<std::size_t>(4) * 2 * D_);
        for (auto& x : H2_) x = nd(rng);
        W1_.assign(static_cast<std::size_t>(p.neurons) * (1u << p.hs) * 8 * K_, 0.3);
        for (std::size_t q = K_ - 1; q < W1_.size(); q += K_) W1_[q] = 0.0;  // the bias starts at 0
        W2_.assign(static_cast<std::size_t>(4) * 4 * 8 * (p.neurons + 1), 1.0 / p.neurons);
        W3_.assign(8 * 5, 0.25);
        apm_.resize(256 * 33);
        for (int c = 0; c < 256; ++c)
            for (int i = 0; i < 33; ++i) apm_[static_cast<std::size_t>(c) * 33 + i] = gln_squash((i - 16) / 2.0);
        hist_.fill(32);
        extra_.assign(static_cast<std::size_t>(p.extra), 0.0);
    }

    // Bit counts of orders 0..5 over text (CyphaLM's training text), capped at 255 per context as glnb.py's pretrain.
    void pretrain(const std::uint8_t* text, std::size_t n) {
        std::vector<float> c1(std::size_t{1} << p_.tb), c(std::size_t{1} << p_.tb);
        for (int k = 0; k < kOrders; ++k) {
            std::fill(c1.begin(), c1.end(), 0.0f);
            std::fill(c.begin(), c.end(), 0.0f);
            for (std::size_t t = 0; t < n; ++t) {
                std::uint64_t h = 0;
                for (int i = 1; i <= k; ++i) h = h * 1099511628211ull + (t >= static_cast<std::size_t>(i) ? text[t - i] + 1u : 0u);
                unsigned c0 = 1;
                for (int j = 0; j < 8; ++j) {
                    const unsigned bit = (text[t] >> (7 - j)) & 1u;
                    const std::size_t key = hash(h, k, c0);
                    c1[key] += static_cast<float>(bit);
                    c[key] += 1.0f;
                    c0 = (c0 << 1) | bit;
                }
            }
            for (std::size_t q = 0; q < c.size(); ++q) {
                if (c[q] > 255.0f) {
                    c1[q] *= 255.0f / c[q];
                    c[q] = 255.0f;
                }
                n1_[k][q] = static_cast<std::uint16_t>(std::lround(c1[q] * 128.0f));
                n_[k][q] = static_cast<std::uint16_t>(std::lround(c[q] * 128.0f));
            }
        }
    }

    void set_extra(const double* v) { std::copy(v, v + p_.extra, extra_.begin()); }

    // Bits of byte y under the model; then learns it (learn) or only reads it.
    double code(std::uint8_t y, bool learn = true) {
        Ctx cx = context();
        double bits = 0;
        unsigned c0 = 1;
        for (int j = 0; j < 8; ++j) {
            const unsigned bit = (y >> (7 - j)) & 1u;
            Fwd f;
            const double p = bit_p(cx, c0, j, f);
            bits += -std::log2(bit ? p : 1 - p);
            if (learn) update(f, c0, j, bit);
            c0 = (c0 << 1) | bit;
        }
        if (p_.lstm) lstm_.step(y, learn);
        std::rotate(hist_.begin(), hist_.begin() + 1, hist_.end());
        hist_.back() = y;
        return bits;
    }

    // The probability of each of the 256 next bytes (255 bit predictions, depth by depth). Learns nothing.
    void byte_dist(double* P) {
        Ctx cx = context();
        double cur[256], nxt[256];
        cur[0] = 1.0;
        for (int j = 0; j < 8; ++j) {
            const int m = 1 << j;
            for (int r = 0; r < m; ++r) {
                Fwd f;
                const double p1 = bit_p(cx, static_cast<unsigned>(m | r), j, f);
                nxt[2 * r] = cur[r] * (1 - p1);
                nxt[2 * r + 1] = cur[r] * p1;
            }
            std::copy(nxt, nxt + 2 * m, cur);
        }
        std::copy(cur, cur + 256, P);
    }

    // The recurrent state, for writing a continuation and coming back (nothing learns in between).
    struct State {
        std::array<std::uint8_t, 8> hist;
        ByteLSTM::State lstm;
    };
    State save() const { return {hist_, lstm_.save()}; }
    void restore(const State& s) {
        hist_ = s.hist;
        lstm_.restore(s.lstm);
    }

    // The whole learned model, to a file and back (§13.17). load() returns false on a bad file.
    bool save_file(const std::string& path) const {
        using namespace gln_io;
        std::ofstream o(path, std::ios::binary);
        o.write("GLN1", 4);
        put(o, p_);
        for (int k = 0; k < kOrders; ++k) put_vec(o, n1_[k]), put_vec(o, n_[k]);
        for (const auto* v : {&H1_, &H2_, &W1_, &W2_, &W3_, &apm_}) put_vec(o, *v);
        put(o, hist_);
        lstm_.write(o);
        return static_cast<bool>(o);
    }
    static std::unique_ptr<BitwiseGLN> load_file(const std::string& path, int extra = 0) {
        using namespace gln_io;
        std::ifstream i(path, std::ios::binary);
        char magic[4] = {0, 0, 0, 0};
        i.read(magic, 4);
        if (!i || std::memcmp(magic, "GLN1", 4) != 0) return nullptr;
        Params p;
        get(i, p);
        if (p.extra != extra) return nullptr;  // the side vector's size is fixed by the stored half-spaces
        auto g = std::make_unique<BitwiseGLN>(p);
        for (int k = 0; k < kOrders; ++k) get_vec(i, g->n1_[k]), get_vec(i, g->n_[k]);
        for (auto* v : {&g->H1_, &g->H2_, &g->W1_, &g->W2_, &g->W3_, &g->apm_}) get_vec(i, *v);
        get(i, g->hist_);
        g->lstm_.read(i);
        return i ? std::move(g) : nullptr;
    }
    const Params& params() const { return p_; }

    std::size_t bytes() const {
        return 2 * kOrders * (std::size_t{1} << p_.tb) * sizeof(std::uint16_t) +
               (H1_.size() + H2_.size() + W1_.size() + W2_.size() + W3_.size() + apm_.size()) * sizeof(double);
    }

private:
    struct Ctx {
        std::uint64_t hh[kOrders];
        double cum[257];
        std::vector<double> z;   // side vector: LSTM state, then inputs (filled per bit), then extra
        std::vector<double> g1;  // each layer-1 half-space's dot with the parts of z fixed for the byte (state, extra)
        std::vector<double> g2;  // the same for layer 2
    };
    struct Fwd {
        std::size_t keys[kOrders];
        double x[kOrders + 2];
        int g1[256];
        double p1[256], s1[257], p2[4], s2[5], p3;
        int g2[4], lo;
        double fr;
    };

    std::size_t hash(std::uint64_t h, int k, unsigned c0) const {
        return static_cast<std::size_t>(((h * 0x9E3779B97F4A7C15ull) ^ (static_cast<std::uint64_t>(c0) * 0x100000001B3ull) ^
                                         (static_cast<std::uint64_t>(k) * 0xC2B2AE3D27D4EB4Full)) >> 7 & mask_);
    }

    Ctx context() const {
        Ctx cx;
        for (int k = 0; k < kOrders; ++k) {
            std::uint64_t v = 0;
            for (int i = 1; i <= k; ++i) v = v * 1099511628211ull + hist_[8 - i] + 1u;
            cx.hh[k] = v;
        }
        cx.z.assign(static_cast<std::size_t>(D_), 0.0);
        if (p_.lstm) {
            double pl[256];
            lstm_.dist(pl);
            cx.cum[0] = 0;
            for (int y = 0; y < 256; ++y) cx.cum[y + 1] = cx.cum[y] + pl[y];
            std::copy(lstm_.h().begin(), lstm_.h().end(), cx.z.begin());
        }
        std::copy(extra_.begin(), extra_.end(), cx.z.end() - p_.extra);
        // The LSTM state and the extra values are fixed for the whole byte; only the K inputs change bit by bit. So each
        // half-space's dot with the fixed parts is taken once here, and bit_p adds the inputs' part (about 9x less work).
        const int off = p_.lstm ? ByteLSTM::H : 0;
        auto fixed_dot = [&](const double* hp) {
            double s = 0;
            for (int d = 0; d < off; ++d) s += hp[d] * cx.z[static_cast<std::size_t>(d)];
            for (int d = off + K_; d < D_; ++d) s += hp[d] * cx.z[static_cast<std::size_t>(d)];
            return s;
        };
        cx.g1.resize(static_cast<std::size_t>(p_.neurons) * p_.hs);
        for (std::size_t q = 0; q < cx.g1.size(); ++q) cx.g1[q] = fixed_dot(&H1_[q * D_]);
        cx.g2.resize(8);
        for (std::size_t q = 0; q < 8; ++q) cx.g2[q] = fixed_dot(&H2_[q * D_]);
        return cx;
    }

    double bit_p(Ctx& cx, unsigned c0, int j, Fwd& f) const {
        const int K = K_, N = p_.neurons, off = p_.lstm ? ByteLSTM::H : 0;
        for (int k = 0; k < kOrders; ++k) {
            f.keys[k] = hash(cx.hh[k], k, c0);
            const double p = (n1_[k][f.keys[k]] / 128.0 + 0.4) / (n_[k][f.keys[k]] / 128.0 + 0.8);
            f.x[k] = gln_stretch(p);
        }
        int q = kOrders;
        if (p_.lstm) {
            const unsigned r = c0 - (1u << j), lo = r << (8 - j), hi = (r + 1) << (8 - j), mid = (lo + hi) / 2;
            const double tot = cx.cum[hi] - cx.cum[lo];
            f.x[q++] = gln_stretch((cx.cum[hi] - cx.cum[mid]) / std::max(tot, 1e-12));
        }
        f.x[q] = 1.0;
        for (int n = 0; n < N; ++n) {
            int g = 0;
            for (int m = 0; m < p_.hs; ++m) {
                const std::size_t q = static_cast<std::size_t>(n) * p_.hs + m;
                const double* hp = &H1_[q * D_ + off];
                double s = cx.g1[q];
                for (int k = 0; k < K; ++k) s += hp[k] * f.x[k];
                if (s > 0) g |= 1 << m;
            }
            f.g1[n] = g;
            const double* w = &W1_[((static_cast<std::size_t>(n) * (1u << p_.hs) + g) * 8 + j) * K];
            double s = 0;
            for (int k = 0; k < K; ++k) s += w[k] * f.x[k];
            f.s1[n] = std::clamp(s, -12.0, 12.0);
            f.p1[n] = 1 / (1 + std::exp(-f.s1[n]));
        }
        f.s1[N] = 1.0;
        for (int i = 0; i < 4; ++i) {
            int g = 0;
            for (int m = 0; m < 2; ++m) {
                const std::size_t q = static_cast<std::size_t>(i) * 2 + m;
                const double* hp = &H2_[q * D_ + off];
                double s = cx.g2[q];
                for (int k = 0; k < K; ++k) s += hp[k] * f.x[k];
                if (s > 0) g |= 1 << m;
            }
            f.g2[i] = g;
            const double* w = &W2_[((static_cast<std::size_t>(i) * 4 + g) * 8 + j) * (N + 1)];
            double s = 0;
            for (int k = 0; k <= N; ++k) s += w[k] * f.s1[k];
            f.s2[i] = std::clamp(s, -12.0, 12.0);
            f.p2[i] = 1 / (1 + std::exp(-f.s2[i]));
        }
        f.s2[4] = 1.0;
        double s3 = 0;
        for (int k = 0; k < 5; ++k) s3 += W3_[static_cast<std::size_t>(j) * 5 + k] * f.s2[k];
        s3 = std::clamp(s3, -12.0, 12.0);
        f.p3 = gln_squash(s3);
        const double pos = (s3 + 8) * 2;
        f.lo = static_cast<int>(std::clamp(std::floor(pos), 0.0, 31.0));
        f.fr = std::clamp(pos - f.lo, 0.0, 1.0);
        const double* a = &apm_[static_cast<std::size_t>(c0) * 33];
        const double pa = a[f.lo] * (1 - f.fr) + a[f.lo + 1] * f.fr;
        return std::clamp(0.5 * f.p3 + 0.5 * pa, 1e-6, 1 - 1e-6);
    }

    void update(const Fwd& f, unsigned c0, int j, unsigned bit) {
        const int K = K_, N = p_.neurons;
        const double b = bit;
        for (int n = 0; n < N; ++n) {
            double* w = &W1_[((static_cast<std::size_t>(n) * (1u << p_.hs) + f.g1[n]) * 8 + j) * K];
            const double e = b - f.p1[n];
            for (int k = 0; k < K; ++k) w[k] = std::clamp(w[k] + 0.002 * e * f.x[k], -8.0, 8.0);
        }
        for (int i = 0; i < 4; ++i) {
            double* w = &W2_[((static_cast<std::size_t>(i) * 4 + f.g2[i]) * 8 + j) * (N + 1)];
            const double e = b - f.p2[i];
            for (int k = 0; k <= N; ++k) w[k] = std::clamp(w[k] + 0.002 * e * f.s1[k], -8.0, 8.0);
        }
        for (int k = 0; k < 5; ++k) {
            double& w = W3_[static_cast<std::size_t>(j) * 5 + k];
            w = std::clamp(w + 0.002 * (b - f.p3) * f.s2[k], -8.0, 8.0);
        }
        double* a = &apm_[static_cast<std::size_t>(c0) * 33];
        a[f.lo] += 0.02 * (1 - f.fr) * (b - a[f.lo]);
        a[f.lo + 1] += 0.02 * f.fr * (b - a[f.lo + 1]);
        for (int k = 0; k < kOrders; ++k) {
            std::uint16_t& c1 = n1_[k][f.keys[k]];
            std::uint16_t& c = n_[k][f.keys[k]];
            c1 = static_cast<std::uint16_t>(c1 + 128u * bit);
            c = static_cast<std::uint16_t>(c + 128u);
            if (c > 255u * 128u) {
                c1 = static_cast<std::uint16_t>(c1 / 2);
                c = static_cast<std::uint16_t>(c / 2);
            }
        }
    }

    Params p_;
    int K_ = 0, D_ = 0;
    std::uint64_t mask_ = 0;
    std::vector<std::uint16_t> n1_[kOrders], n_[kOrders];
    std::vector<double> H1_, H2_, W1_, W2_, W3_, apm_;
    std::array<std::uint8_t, 8> hist_;
    std::vector<double> extra_;
    ByteLSTM lstm_;
};

}  // namespace cypha::online
